import numpy as np
from typing import Literal

import torch.utils.data as Data
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from .dataset import ASPEDv2Dataset, SR
from .transforms import VGGish, VGGish_PreProc

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class AspedDataModule(pl.LightningDataModule):
    #Class Variables
    _transform = torch.nn.Identity()

    def __init__(self, dataset, batch_size=32, num_workers=6, 
                 transform :Literal['vggish', 'ast', None] = None) -> None:
        super().__init__()

        self.dataset = dataset


        self.batch_size = batch_size
        self.num_workers = num_workers

        if transform == 'vggish':
            AspedDataModule._transform = VGGish()
            self.num_workers = 0
        else:
            AspedDataModule._transform = torch.nn.Identity()
        
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_splits()
        self.sampler = self.init_weighted_sampler(self.dataset, self.train_dataset.indices)

    def get_splits(self):
        #Ensure portions of train segments are not leaked into test segments

        max_idx = len(self.dataset) - ASPEDv2Dataset.segment_length
        segment_idx = torch.arange(0, max_idx - 1, ASPEDv2Dataset.segment_length).int()
        idx = segment_idx[torch.randperm(segment_idx.shape[0])]

        #idx = torch.randperm(len(self.dataset))
        num = idx.shape[0] // 10
        if ASPEDv2Dataset.n_classes == 1 or ASPEDv2Dataset.n_classes > 2:
            train_dataset = AugmentationDataset(self.dataset, idx[:8*num])
            train_dataset = Data.Subset(train_dataset, torch.arange(len(train_dataset)))
        else:
            train_dataset = Data.Subset(self.dataset, idx[:8*num])

        return train_dataset, Data.Subset(self.dataset, idx[8*num:9*num]), Data.Subset(self.dataset, idx[9*num:])

    @staticmethod
    def _transform_collate(batch):
        X = torch.cat([b[0].unsqueeze(dim=0) for b in batch], axis=0)
        y = torch.cat([b[1].unsqueeze(dim=0) for b in batch], axis=0)
        
        if len(X.shape) < 5:
            X = AspedDataModule._transform(X, SR)

        return X, y

    def train_dataloader(self):
        if type(AspedDataModule._transform) == VGGish or type(AspedDataModule._transform) == VGGish_PreProc:
            return Data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                                   num_workers=self.num_workers, collate_fn=AspedDataModule._transform_collate)
        return Data.DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        if type(AspedDataModule._transform) == VGGish or type(AspedDataModule._transform) == VGGish_PreProc:
            return Data.DataLoader(self.val_dataset, batch_size=self.batch_size,
                                   num_workers=self.num_workers, collate_fn=AspedDataModule._transform_collate)
        
        return Data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if type(AspedDataModule._transform) == VGGish or type(AspedDataModule._transform) == VGGish_PreProc:
            return Data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                   num_workers=self.num_workers, collate_fn=AspedDataModule._transform_collate)
        
        return Data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def init_weighted_sampler(self, concat_dataset, indices):
        labels = []
        for dataset in concat_dataset.datasets:
            labels.append(dataset.labels)
        labels = torch.ceil(ASPEDv2Dataset.threshold(torch.cat(labels, axis=0)[indices]))

        data = []
        for x in tqdm(range(labels.shape[0] - self.dataset.datasets[0].segment_length)):
            #data.append(torch.max(labels[x:x+self.dataset.datasets[0].segment_length].unsqueeze(dim=1), axis = 1)[0])
            data.append(torch.max(labels[x:x+self.dataset.datasets[0].segment_length].unsqueeze(dim=-1), axis=0)[0])
        data = torch.cat(data, dim=0)

        if len(data.shape) == 1:
            y , counts = torch.unique(data, return_counts = True)
        else:
            _ , counts = torch.unique(torch.max(data, axis=1)[0], return_counts = True)

        print(y, counts, data.shape)
        #weights = [1./x for x in counts]
        weights = 1. / counts.float()
        #weights[0] = 0 #ignore -1 idx

        if len(data.shape) > 1:
            data, _ = torch.max(data, axis=1)

        return Data.WeightedRandomSampler(weights[data.long()], data.shape[0], replacement=True)

class AugmentationDataset(Data.Dataset):
    class AugmentedSampleDataset(Data.Dataset):
        def __init__(self, core_dataset, indices, labels, new_samples):
            self.core_dataset = core_dataset
            self.labels = labels
            self.new_samples = new_samples
            self.indices = indices
            y = torch.unique(self.labels[self.indices])
            self.map = {k.int().item():torch.argwhere(self.labels[self.indices] == k.int().item()) for k in y} 
            self.sample_map = self.construct_sample_map()

        def __len__(self):
            return self.new_samples.shape[0] - ASPEDv2Dataset.segment_length
        
        def construct_sample_map(self):
            return self.new_samples
            '''
            sample_list = list()
            for v in tqdm(self.new_samples.tolist()):
                v_1 = torch.randint(int(v), (1,)).item()
                v_2 = v - v_1
                v_1_idx = torch.randint(self.map[v_1].shape[0], (1,)).item()
                v_2_idx = torch.randint(self.map[v_2].shape[0], (1,)).item()
                sample_list.append([self.indices[self.map[v_1][v_1_idx]],
                                     self.indices[self.map[v_2][v_2_idx]]])
            return torch.Tensor(sample_list)
            '''
        
        def __getitem__(self, idx):
            val = self.sample_map[idx:idx + ASPEDv2Dataset.segment_length]
            '''
            Pre-computed samples

            sample = list()
            labels = list()
            ASPEDv2Dataset.do_transform = False
            for v in val:
                X_1, y_1 = self.core_dataset[v[0]]
                X_2, y_2 = self.core_dataset[v[1]]
                sample.append((X_1[0,:] + X_2[0,:]).unsqueeze(dim=0))
                labels.append((y_1[0] + y_2[0]).unsqueeze(dim=-1))
            ASPEDv2Dataset.do_transform = True
            max_val = ASPEDv2Dataset.n_classes - 1 if ASPEDv2Dataset.n_classes > 1 else 1.0
            return ASPEDv2Dataset.transform(torch.cat(sample, axis=0), SR), torch.clamp(torch.cat(labels, axis=0), max=max_val)
            '''
            
            '''
            Dynamically-computed samples
            '''
            sample = list()
            labels = list()
            ASPEDv2Dataset.do_transform = False
            for v in val:
                v = int(v.item())
                v_1 = torch.randint(int(v), (1,)).item()
                v_2 = v - v_1
                v_1_idx = torch.randint(self.map[v_1].shape[0], (1,)).item()
                v_2_idx = torch.randint(self.map[v_2].shape[0], (1,)).item()
                v_1_idx = self.indices[self.map[v_1][v_1_idx]].item()
                v_2_idx = self.indices[self.map[v_2][v_2_idx]].item()
                #debug(v_1_idx, v_2_idx)
                X_1, y_1 = self.core_dataset[v_1_idx]
                X_2, y_2 = self.core_dataset[v_2_idx]
                sample.append((X_1[0,:] + X_2[0,:]).unsqueeze(dim=0))
                labels.append((y_1[0] + y_2[0]).unsqueeze(dim=-1))
            ASPEDv2Dataset.do_transform = True
            max_val = ASPEDv2Dataset.n_classes - 1 if ASPEDv2Dataset.n_classes > 1 else 1.0
            return ASPEDv2Dataset.transform(torch.cat(sample, axis=0), SR), torch.clamp(torch.cat(labels, axis=0), max=max_val)

    def __init__(self, concat_dataset, indices=None, poisson_rate='uniform'):
        self.core_dataset = concat_dataset
        self.labels = [dataset.labels for dataset in concat_dataset.datasets]
        self.labels = torch.cat(self.labels, axis=0)
        self.transform = ASPEDv2Dataset._transform
        #ASPEDv2Dataset._transform = torch.nn.Identity()
        #AspedDataModule._transform = VGGish_PreProc()
        if poisson_rate == 'mean':
            self.poisson_rate = torch.mean(self.labels[self.labels > 0])
        elif isinstance(poisson_rate, int):
            self.poisson_rate = poisson_rate
        elif poisson_rate == 'uniform':
            self.poisson_rate = 'uniform'
        else:
            raise Exception('Poisson rate must be one of [int, \'mean\', \'uniform\']')
        
        self.indices = indices
        
        if not indices is None: #flatten for intermediate indices
            x = torch.arange(ASPEDv2Dataset.segment_length).tile(indices.shape[0],).view(-1, ASPEDv2Dataset.segment_length)
            z = indices.unsqueeze(dim=1).tile(1, ASPEDv2Dataset.segment_length)
            self.flat_indices = (x + z).flatten()
        else:
            self.flat_indices = torch.arange(self.labels.shape[0])

        self.dataset = self.setup()
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        X, y = self.dataset[self.indices[idx]]
        return X, y #self.transform(X, SR), y

    def poisson(self, x):
        return ((self.poisson_rate**x)/np.math.factorial(x))*np.exp(-self.poisson_rate)
    
    def setup(self):
        #sample amt such that dataset contains half 0's and half 1+'s
        train_labels = self.labels[self.flat_indices]
        num_to_sample = train_labels[train_labels == 0].shape[0] - train_labels[train_labels > 0].shape[0]

        if self.poisson_rate != 'uniform':
            #oversample taking into account prob of sampling 0 such that sampled 1+ elements ~= num_to_sample
            num_to_sample = int(num_to_sample * (1 + self.poisson(0)) // 1)

            new_samples = torch.ones(num_to_sample) * self.poisson_rate
            max_val = ASPEDv2Dataset.max_val if ASPEDv2Dataset.n_classes == 1 else ASPEDv2Dataset.n_classes - 1
            new_samples = torch.clamp(torch.poisson(new_samples), max=max_val)
            new_samples = new_samples[new_samples > 0]
        else:
            new_samples = torch.randint(1,7,(num_to_sample,))

        augmented_dataset = AugmentationDataset.AugmentedSampleDataset(self.core_dataset, self.flat_indices, 
                                                   self.labels, new_samples)
        
        self.indices = torch.cat((self.indices, torch.arange(len(augmented_dataset)) 
                                  + len(self.core_dataset)), axis=0)
        
        return Data.ConcatDataset((self.core_dataset, augmented_dataset))

if __name__ == '__main__':
    '''
    Testing
    '''
    from transforms import VGGish
    from dataset import SR
    import time

    TEST_DIR = ["/media/fast_drive/audio_data/Test_7262023", "/media/fast_drive/audio_data/Test_08092023", 
                "/media/fast_drive/audio_data/Test_10242023", "/media/fast_drive/audio_data/Test_11072023"]
    X = ASPEDv2Dataset.from_dirs(TEST_DIR, segment_length=10, transform='vggish-mel')
    vggish = VGGish(pproc=False).to(ASPEDv2Dataset.device)
    print(ASPEDv2Dataset.device)

    datamod = AspedDataModule(X, batch_size=512)
    dataloader = datamod.train_dataloader()

    lim = len(dataloader)
    start = time.time()
    print("testing....")
    labels = []
    for item, label in tqdm(dataloader, total=lim):
        #y = vggish(item.view(-1, *item.shape[2:]))
        #y = y.view(*item.shape[:2], y.shape[-1])
        #raise Exception(item.shape)
        #item = item.to(DEVICE)
        #y = vggish(item)
        labels.append(label)
        pass
    end = time.time()

    labels = torch.cat(labels, axis=0).flatten()
    num_p = labels[labels == 0].sum() / labels.sum()
    num_n = labels[labels == 1].sum() / labels.sum()
    num_o = labels[labels == -1].sum() / labels.sum()
    print(num_n, num_o, num_p)

    print(f'time per execution: {((len(dataloader)*((end - start)/lim)))/60:.2e}m')
