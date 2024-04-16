import pytorch_lightning as pl
import torch
from .metrics import classification_report, confusion_matrix, regression_report, regression_plot
from tqdm import tqdm
from torchmetrics import ConfusionMatrix
from data_utils import VGGish, ConvHead, AST_head, DEVICE

from typing import Literal

from data_utils.dataset import IGNORE_INDEX

OUTPUT = 'output'
LOSS = 'loss'

class ASPEDModel(pl.LightningModule): 
    def __init__(self, segment_length = 10, h_dim = 128, lr = 0.005, nhead = 4, token_dim = 128, dropout = 0.2, nEncoders = 1,
                  encoder: Literal['conv', 'vggish', 'vggish-finetune', 'conv-lite', 'ast'] = 'conv', n_classes=2):
        '''
        args:
            segment_length = length of input sequence
            h_dim = hidden dimension of encoder
            lr = learning rate 
            nhead = number of attention heads for transformer encoder
            token_dim = transformer input dim
            dropout = dropout probability of transformer encoder
            nEncoders = number of transformer blocks 
            encoder = type of encoder to use. only accepts below args
                conv: trainable vggish network, no pretraining (80M trainable params)
                vggish: vggish pretrained network, frozen weights (0 trainable params)
                vggish-finetune: trainable vggish pretrained network (80M trainable params)
                conv-lite: 5-layer convBlock (2M trainable params)
                ast: trainable imagenet/audioset pretrained Audio Spectrogram Transformer (80M trainable params)
        '''
        super(ASPEDModel, self).__init__()
        self.save_hyperparameters()
          
        # Define model architecture

        self.segment_length = segment_length
        self.h_dim = h_dim
        self.token_dim = token_dim
        self.encoder_type = encoder
        self.n_classes = n_classes

        self.pe = torch.nn.Embedding(segment_length, token_dim)
        self.pe_input = torch.Tensor(range(segment_length)).long()

        transformer_layers = torch.nn.TransformerEncoderLayer(token_dim, nhead, h_dim, dropout, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(transformer_layers, nEncoders)

        self._proj = torch.nn.Sequential(torch.nn.Linear(h_dim, h_dim), torch.nn.ReLU(), torch.nn.Linear(h_dim, n_classes))

        if self.n_classes > 1:
            self._activation = torch.nn.Sigmoid() #torch.nn.Softmax(dim=1)
        else:
            self._activation = torch.nn.Sigmoid()
        
        if self.encoder_type == 'conv':
            self.encoder = VGGish(pproc=False, trainable=True, pre_trained=False)
        elif self.encoder_type == 'vggish':
            self.encoder = VGGish(pproc=False, trainable=False, pre_trained=True)
        elif self.encoder_type == 'vggish-finetune':
            self.encoder = VGGish(pproc=False, trainable=True, pre_trained=True)
        elif self.encoder_type == 'conv-lite':
            self.encoder = ConvHead(latent_dim=h_dim)
        elif self.encoder_type == 'ast':
            self.encoder = AST_head(n_classes=n_classes)

            #AST does not use temporal transformer layer, set all to dummy functions
            self.pe = torch.nn.Identity()
            self.pe_input = torch.Tensor([0]).long()
            self.transformer = torch.nn.Identity()
            self._proj = torch.nn.Identity()
        else:
            self.encoder = None
        
        if self.segment_length == 1:
            #if segment length is 1, set temporal transformer layer to dummy functions
            self.pe = torch.nn.Identity()
            self.pe_input = torch.Tensor([0]).long()
            self.transformer = torch.nn.Identity()

        if self.n_classes > 2:
            #shared weights for each class, unique bias term for rank monotonicity
            self._proj = torch.nn.Sequential(torch.nn.Linear(h_dim, h_dim), torch.nn.ReLU(), torch.nn.Linear(h_dim, 1, bias=False))
            self.bias = torch.nn.Parameter(data=torch.rand(1, self.n_classes))
            torch.nn.init.uniform_(self.bias, a=-1/(2**0.5), b=1/(2**0.5))
        


        
        # Defining learning rate
        self.lr = lr
          
        # Define loss 
        #self.loss_fn = torch.nn.NLLLoss()
        if self.n_classes == 1:
            self.loss_fn = torch.nn.MSELoss()
        elif self.n_classes == 2:
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index = IGNORE_INDEX)
        else:
            self.loss_fn = torch.nn.BCELoss() #torch.nn.CrossEntropyLoss(ignore_index = IGNORE_INDEX, reduction='none')
            

        #Cache Validation outputs 
        self.outs = list()

        #Define metrics
        if n_classes == 2:
            self.report = classification_report
        else:
            self.report = regression_report

        self.cls_threshold = None

    def proj(self, X):
        if self.n_classes > 2:
            return self._proj(X) + self.bias
        return self._proj(X)
    
    def activation(self, X):
        X = self._activation(X)
        if self.n_classes == 1:
            return X #torch.round(X * 6) / 6
        if self.cls_threshold is None and len(X.shape) <= 2:
            return torch.round(X)
        elif len(X.shape) > 2:
            return torch.sum(torch.round(X), dim=1) #torch.argmax(X, dim=1)
        mask = X >= self.cls_threshold

        X[mask] = 1
        X[~mask] = 0

        return X
    
    def ordered_cls_loss(self, X, y, eq_weight=False):
        X = self._activation(X)
        target = y.view(-1).unsqueeze(dim=-1)
        with torch.no_grad():
            a = torch.arange(self.n_classes, device=self.device).unsqueeze(dim=0)
            z = torch.tile(a, (target.shape[0],1))
            y_ = torch.zeros_like(z)
            y_[z < target] = 1
        y_ = y_.view(*y.shape, -1).transpose(1,2)

        if eq_weight:
            _ , counts = torch.unique(y_, return_counts = True)
            weights = 1. / counts.float()
            weights = weights[y_.long()]
            return (weights * torch.nn.functional.binary_cross_entropy(X, y_.float(), reduction='none')).sum()
            

        return self.loss_fn(X, y_.float())

    def weighted_loss(self, X, y):
        '''
        Binary class-weighted loss term. Refer to ICASSP 2024 paper for details
        '''

        idx_pos = torch.argwhere(y > 0)
        idx_neg = torch.argwhere(y == 0)

        try:
            pos_w = min(1, (1/(idx_pos.flatten().shape[0])) / ((1/(idx_pos.flatten().shape[0]) + 1/(idx_neg.flatten().shape[0])) + 0))
        except ZeroDivisionError:
            pos_w = 0
        
        pos_loss = pos_w * self.loss_fn(X[idx_pos], y[idx_pos])
        pos_loss = torch.Tensor([0]).to(DEVICE) if torch.isnan(pos_loss) else pos_loss

        neg_loss = (1 - pos_w) * self.loss_fn(X[idx_neg], y[idx_neg])
        neg_loss = torch.Tensor([0]).to(DEVICE) if torch.isnan(neg_loss) else neg_loss

        return  pos_loss + neg_loss

    def loss(self, X, y):
        if self.n_classes == 1:
            return self.loss_fn(self._activation(X.flatten()), y.flatten())
        elif self.n_classes > 2:
            return self.ordered_cls_loss(X, y)
        else:
            y = y.long()
            return self.weighted_loss(X, y)


    def forward(self, x):
        """
        :param x: [batch_size x segment_length x 1 x 96 x 64] (encoder type: vggish, conv, vggish-finetune, conv-lite)
                  [batch_size x segment_length x 100 x 128] (encoder type: ast)

        :return output: [batch_size, segment_length, n_classes]
        """
        output = self.encoder(x)
        
        output = self.pe(self.pe_input.to(self.device)) + output

        output = self.transformer(output)

        output = self.proj(output)

        return output
    
    def common_step(self, batch):
        X, y = batch

        pred = self.forward(X).permute(0, 2, 1)

        return {LOSS: self.loss(pred, y), OUTPUT: pred}

    def training_step(self, batch, batch_idx):
        
        model_output = self.common_step(batch)
        
        self.log('train_loss', model_output[LOSS].detach().cpu().item(), prog_bar=True, sync_dist=True)

        return model_output[LOSS]

    def validation_step(self, batch, batch_idx):
        model_output = self.common_step(batch)

        self.outs.append((self.activation(model_output[OUTPUT]), batch[1]))

        self.log('val_loss', model_output[LOSS].detach().cpu().item(), prog_bar=True, sync_dist=True)

        return model_output[LOSS]
    
    
    def test_step(self, batch, batch_idx):
        model_output = self.common_step(batch) 
        self.outs.append((self.activation(model_output[OUTPUT]), batch[1]))

        self.log('test_loss', 0, prog_bar=True, sync_dist=True)

        return model_output[LOSS]
    
    def common_eval(self, stage='val'):
        pred, targ = self.collect_test_batches()
        if len(pred.shape) > 1:
            pred_onset = pred.flatten()
            targ_onset = targ.flatten()

        pred_onset = pred_onset[targ_onset != IGNORE_INDEX].tolist()
        targ_onset = targ_onset[targ_onset != IGNORE_INDEX].tolist()
        

        rep = self.report(targ_onset, pred_onset, output_dict=True)
        cls_rep = classification_report([int(x) for x in targ_onset], pred_onset, output_dict=True)
        rep.update(cls_rep)
        self.log_nested_dict(rep, stage=stage)
        self.outs.clear()

        return pred_onset, targ_onset, rep
        
    def log_nested_dict(self, X: dict, stage='val'):
        for k, v in X.items():
            if type(v) == dict:
                d = {'/'.join([k, k2]):v2 for k2, v2 in v.items()}
                self.log_nested_dict(d, stage=stage)
            else:
                self.log(f'{stage} {k}', v, prog_bar=True, sync_dist=True)
            
    def on_validation_epoch_end(self):
        pred, targ, rep = self.common_eval(stage='val')
        #loss = self.loss(pred, targ)
        return rep

    def collect_test_batches(self):
        with torch.no_grad():
            X_total = None
            y_total = None
            for batch in self.outs:
                X, y = batch
                if X_total is None:
                    X_total = X.detach().cpu()
                    y_total = y.detach().cpu()
                X_total = torch.cat((X_total, X.detach().cpu()), axis=0)
                y_total = torch.cat((y_total, y.detach().cpu()), axis=0)
        
        return X_total, y_total
        
    def on_test_epoch_end(self):
        pred_onset, targ_onset, rep = self.common_eval(stage='test')

        if self.n_classes > 1:
            print(confusion_matrix(targ_onset, pred_onset), sep='\n')
        else:
            regression_plot(targ_onset, pred_onset)
        return rep
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-9)