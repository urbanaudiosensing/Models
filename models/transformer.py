import pytorch_lightning as pl
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from torchmetrics import ConfusionMatrix
from .models import ConvHead, AST_head

class VanillaTransformer(pl.LightningModule): 
    def __init__(self, window_size = 10, h_dim = 128, lr = 0.005, nhead = 4, token_dim = 128, dropout = 0.2, nEncoders = 1,
                 reduced=True, head = 'conv'):
        super(VanillaTransformer, self).__init__()
        self.save_hyperparameters()
          
        # Define model architecture

        self.window_size = window_size
        self.h_dim = h_dim
        self.token_dim = token_dim
        self.reduced = reduced
        self.head = head

        self.pe = torch.nn.Embedding(window_size, token_dim)
        self.pe_input = torch.Tensor(range(window_size)).long()

        self.cls_token = torch.nn.Embedding(1, token_dim)

        encoder_layers = torch.nn.TransformerEncoderLayer(token_dim, nhead, h_dim, dropout, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layers, nEncoders)

        if self.reduced:
            self.decoder = torch.nn.Sequential(torch.nn.Linear(h_dim * window_size, 1), torch.nn.Sigmoid())
        else:
            self.decoder = torch.nn.Sequential(torch.nn.Linear(h_dim, h_dim), torch.nn.Linear(h_dim, 2), torch.nn.LogSoftmax(dim=-1))
        
        if self.head == 'conv':
            self.conv = ConvHead() 
        elif self.head == 'ast':
            self.conv = AST_head()
        else:
            self.conv = None

        
        # Defining learning rate
        self.lr = lr
          
        # Define loss 
        self.loss_fn = torch.nn.NLLLoss()
        #self.loss_fn = torch.nn.BCELoss()

        #Cache Validation outputs 
        self.outs = list()

        #Define metrics
        self.cls_report = classification_report

        self.cls_threshold = None

        self.cm = ConfusionMatrix(task='binary', num_classes=2).to(torch.device("cpu"))
    
    def activation(self, X):
        if self.cls_threshold is None and len(X.shape) <= 2:
            return torch.round(X)
        elif len(X.shape) > 2:
            return torch.argmax(X, dim=1)
        mask = X >= self.cls_threshold

        X[mask] = 1
        X[~mask] = 0

        return X

    def loss(self, X, y):
        y = y.long()
        #X = X.contiguous()

        idx_pos = torch.argwhere(y == 1)
        idx_neg = torch.argwhere(y == 0)

        try:
            pos_w = min(1, (1/(idx_pos.flatten().shape[0])) / ((1/(idx_pos.flatten().shape[0]) + 1/(idx_neg.flatten().shape[0])) + 0))
        except ZeroDivisionError:
            pos_w = 0
            
        pos_loss = pos_w * self.loss_fn(X[idx_pos].contiguous(), y[idx_pos].contiguous())
        pos_loss = 0 if torch.isnan(pos_loss) else pos_loss

        return  pos_loss + \
            (1 - pos_w) * self.loss_fn(X[idx_neg].contiguous(), y[idx_neg].contiguous())


        
    def init_weights(self):
        initrange = 0.02
        nn.init.trunc_normal_(self.encoder.weight, a=-initrange, b=initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.zeros_(self.decoder_bias)
        nn.init.trunc_normal_(self.decoder.weight, a=-initrange, b=initrange)
        nn.init.trunc_normal_(self.vocab.weight, a=-initrange, b=initrange)
        nn.init.trunc_normal_(self.pe.weight, a=-initrange, b=initrange)


    def forward(self, x):
        """
        :param x: [batch_size, window_size, 128] or [batch_size x window_size x 1 x H x W]
        :return: [batch_size, 1]
        """
        if type(self.conv) == ConvHead:
            x = x.view(-1, *x.shape[2:])
            x = self.conv(x).view(-1, self.window_size, self.h_dim)
        elif type(self.conv) == AST_head:
            x = x.view(-1, *x.shape[2:])
            return self.conv(x).view(-1, self.window_size)
        else:
            pass
        
        x = self.pe(self.pe_input.to(self.device)) + x

        output = self.encoder(x)

        if self.reduced:
            output = output.view(self.window_size * self.h_dim, -1).transpose(0, 1)
        output = self.decoder(output)

        return output.squeeze() if (self.reduced) else output.squeeze().transpose(1, 2)

    
    def training_step(self, batch, batch_idx):
        X, y = batch

        pred = self.forward(X)

        loss = self.loss(pred, y)

        self.log('train_loss', loss.detach().item(), prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        
        pred = self.forward(X)
        
        loss = self.loss(pred, y)

        self.outs.append((self.activation(pred), y))

        self.log('val_loss', loss.detach().item(), prog_bar=True, sync_dist=True)

        return loss
    
    
    def test_step(self, batch, batch_idx):

        X, y = batch
        
        pred = self.forward(X)

        loss = self.loss(pred, y)

        self.outs.append((self.activation(pred), y))

        self.log('test_loss', loss.detach().item(), prog_bar=True, sync_dist=True)

        return loss
    
    def on_validation_epoch_end(self):
        pred, targ = self.collect_test_batches()

        if len(pred.shape) > 1:
            pred_onset = pred.flatten().tolist()
            targ_onset = targ.flatten().tolist()
        
        rep = self.cls_report(targ_onset, pred_onset, output_dict=True)
        self.log_dict(rep,prog_bar=True, sync_dist=True)
        self.outs.clear()

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
        pred, targ = self.collect_test_batches()

        if len(pred.shape) > 1:
            pred_onset = pred.flatten().tolist()
            targ_onset = targ.flatten().tolist()
        with torch.no_grad():
            rep = self.cls_report(targ_onset, pred_onset, output_dict=True)
            self.log_dict(rep,prog_bar=True, sync_dist=True)
        self.outs.clear()

        print('onset', confusion_matrix(targ_onset, pred_onset), sep='\n')

        return rep
    
    def configure_optimizers(self):
        return  torch.optim.Adam(self.parameters(), lr=self.lr)