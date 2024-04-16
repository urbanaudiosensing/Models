import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import numpy as np
import json
import os
import pandas as pd

from .transforms import VGGish, VGGish_PreProc, ASTFeatureExtractor, AST_PreProc

from typing import Literal
from tqdm import tqdm


METADATA_PATH = 'Labels/cam2rec.json'
AUDIO_EXT = '.npy' #Audio Files are pre-processed into 1-channel 16Khz numpy arrays
AUDIO_PREFIX = 'DR-05X-{0}'
LABELS_SUFFIX = '{}.csv'
LABEL_HEADER = 'recorder{0}_{1}m'
VIEW_PREFIX = 'view_'
SR = 16000
IGNORE_INDEX = -1 #label for ground truth obstructed by view, ignored in gradient computation

class ASPEDv2Dataset(Dataset):
    
    _transform = torch.nn.Identity() #Pre-process the raw waveform as input to n_classesl
    n_classes = 'cls'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    min_val = 1
    max_val = 1
    do_transform = True

    @classmethod
    def transform(cls, *args) -> torch.Tensor:
        if type(ASPEDv2Dataset._transform) == torch.nn.Identity:
            return ASPEDv2Dataset._transform(args[0])
        else:
            return ASPEDv2Dataset._transform(*args)
    
    @classmethod
    def threshold(cls, X: torch.Tensor) -> torch.Tensor:
        if ASPEDv2Dataset.n_classes == 1:
            return torch.clamp(X / 6, max=1.0) #normalize values for regression
        else: #Constrain class labels to [0, n_classes - 1], where all values < min_val are set to 0
            return torch.clamp(X // ASPEDv2Dataset.min_val, max=ASPEDv2Dataset.n_classes - 1) 
        
    def __init__(self, rec_path, labels, segment_length = 1, n_classes = 2):
        #Instance vars
        self.rec_path = rec_path
        self.labels = labels

        #Class Vars
        ASPEDv2Dataset.segment_length = segment_length
        ASPEDv2Dataset.n_classes = n_classes #1 sets model to regression mode
         
        # Load data into a NumPy memory-mapped array
        self.data, self.indices = self._load_data()

        #for locations with uneven audio-labels
        self.labels = self.labels[:self.indices[-1] // SR]

    def __len__(self):
        return len(self.labels) - (self.segment_length + 1)
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.long()
        # Load the data sample at the specified index
        c = np.searchsorted(self.indices, idx * SR, side='right') - 1
        #c = min(c, len(self.data) - 1)

        file_offset = idx * SR - self.indices[c]
        try:
            #if (c < len(self.data) - 1) and ((idx * SR) + (SR * self.segment_length) > self.indices[c+1]):
            if (c < len(self.data) - 1) and (file_offset + (SR * self.segment_length) >= len(self.data[c])):
                item_1 = torch.Tensor(self.data[c][file_offset:].copy())
                item_2 = torch.Tensor(self.data[c+1][:(SR * self.segment_length) - item_1.shape[0]].copy())
                item = torch.cat((item_1, item_2), axis = 0)
            else:
                item = torch.Tensor(self.data[c][file_offset:file_offset + (SR * self.segment_length)].copy())
        except Exception as e:
            print(e)
            raise Exception(idx, self.indices, len(self),file_offset,len(self.indices), c, idx * SR, self.indices[c], self.indices[-1], idx * SR + (SR * self.segment_length), len(self.data), (c < len(self.data) - 1))

        #if item.shape[0] % SR != 0:
            #item = self._zeropad(item, (item.shape[0]//SR + 1) * SR)
        
        #item = item.to(ASPEDv2Dataset.device)
        labels = self.labels[idx:idx+self.segment_length] #.to(ASPEDv2Dataset.device)

        if not ASPEDv2Dataset.do_transform:
            return item.view(self.segment_length, SR), ASPEDv2Dataset.threshold(labels)

        return ASPEDv2Dataset.transform(item.view(self.segment_length, SR), SR), ASPEDv2Dataset.threshold(labels)
    
    def _zeropad(self, X, num):
        if len(X) >= num:
            return X
        return torch.cat([X, torch.zeros((num - X.shape[0],))])


    def _load_data(self):
        # Assuming each file contains data that can be loaded into a NumPy array
        # Modify this function based on your data loading logic
        file_list = sorted([os.path.join(self.rec_path, x) for x in os.listdir(self.rec_path) if x.endswith(AUDIO_EXT)], key=lambda x: x.split('/')[-1])
        #file_list = [file_list[-1]] + file_list[1:-1] #to account for split_{}.npy
        data_list = [np.load(file_path, mmap_mode='r') for file_path in file_list]

        indices = np.cumsum([0] + [x.shape[0] for x in data_list])

        return (data_list, indices)
     
    @staticmethod
    def from_dirs(dirs: list[str], radius: Literal[1,3,6,9] = 6, segment_length = 1,
                  transform :Literal['vggish', 'vggish-mel', 'ast', None] = None, n_classes= 2) -> ConcatDataset:
        
        if transform == 'vggish':
            ASPEDv2Dataset._transform = VGGish()#.to(ASPEDv2Dataset.device)
        elif transform == 'vggish-mel':
            ASPEDv2Dataset._transform = VGGish_PreProc()#.to(ASPEDv2Dataset.device)
        elif transform == 'ast':
            ASPEDv2Dataset._transform = torch.nn.Identity() #not implemented yet
        
        ASPEDv2Dataset.n_classes = n_classes

        dataset = []
        
        for d in tqdm(dirs):
            with open(os.path.join(d, METADATA_PATH), 'r') as f:
                metadata = json.load(f)
            for cam, recs in metadata.items():
                try:
                    labels = pd.read_csv(*[os.path.join(d, 'Labels', x) for x in 
                                        os.listdir(os.path.join(d, 'Labels')) if x.endswith(LABELS_SUFFIX.format(cam))])
                except:
                    continue
                
                for i in range(1, len(recs) + 1):
                    working_dir = os.path.join(d, AUDIO_PREFIX.format(recs[i - 1]))

                    label = torch.Tensor(labels[LABEL_HEADER.format(i, radius)].values)
                    views = torch.Tensor(labels[VIEW_PREFIX+LABEL_HEADER.format(i, radius)].values)
                    label[views == 1] = IGNORE_INDEX
                    ASPEDv2Dataset.max_val = max(label.max().item(), ASPEDv2Dataset.max_val)
                    dataset.append(ASPEDv2Dataset(working_dir, label, segment_length=segment_length, n_classes=n_classes))
        
        return ConcatDataset(dataset)
    
    @staticmethod
    def from_dirs_v1(dirs: list[str], radius: Literal[1,3,6,9] = 6, segment_length = 1,
                  transform :Literal['vggish', 'vggish-mel', 'ast', None] = None, n_classes= 2) -> ConcatDataset:
        
        ASPEDv2Dataset.n_classes = n_classes
        dataset = []
        
        for d in tqdm(dirs):
            for s in os.listdir(d):
                path = os.path.join(d,s)
                try:
                    label_files = list(filter(lambda x: x.endswith(LABELS_SUFFIX[-3:]) and not
                                        'processed' in x, [os.path.join(path,'Labels', x) for x in os.listdir(os.path.join(path, 'Labels'))]))
                    labels = [pd.read_csv(x) for x in sorted(label_files)]
                    labels = pd.concat(labels, axis=0)
                except Exception as e:
                    print(e)
                    continue
                audio_path = os.path.join(path, 'Audio')
                
                for i, rec in enumerate(sorted(os.listdir(audio_path))):
                    label = torch.Tensor(labels[LABEL_HEADER.format(i + 1, radius)].values)
                    ASPEDv2Dataset.max_val = max(label.max().item(), ASPEDv2Dataset.max_val)
                    working_dir = os.path.join(audio_path, rec)
                    dataset.append(ASPEDv2Dataset(working_dir, label, segment_length=segment_length, n_classes=n_classes))
                    
        c_dataset = ConcatDataset(dataset)

        if transform == 'vggish':
            ASPEDv2Dataset._transform = VGGish()#.to(ASPEDv2Dataset.device)
        elif transform == 'vggish-mel':
            ASPEDv2Dataset._transform = VGGish_PreProc()#.to(ASPEDv2Dataset.device)
        elif transform == 'ast':
            ASPEDv2Dataset._transform = AST_PreProc()
        else:
            ASPEDv2Dataset._transform = torch.nn.Identity()
        return c_dataset

if __name__ == '__main__':
    '''
    Testing
    '''
    import time
    from random import randint

    #TEST_DIR = ["/media/fast_drive/audio_data/Test_7262023", "/media/fast_drive/audio_data/Test_08092023", 
                #"/media/fast_drive/audio_data/Test_10242023", "/media/fast_drive/audio_data/Test_11072023"]
    TEST_DIR = ["/media/fast_drive/audio_data_aligned/Session_5242023", "/media/fast_drive/audio_data_aligned/Session_6012023", 
                "/media/fast_drive/audio_data_aligned/Session_6072023", "/media/fast_drive/audio_data_aligned/Session_6212023",
                "/media/fast_drive/audio_data_aligned/Session_6282023"]
    X = ASPEDv2Dataset.from_dirs_v1(TEST_DIR, segment_length=9, transform='vggish-mel')
    print(type(X), len(X))

    lim = 100 #len(X)

    start = time.time()

    num_zero = 0
    err = list()
    print("testing....")
    for x in tqdm(range(lim)):
        idx = randint(0, len(X))
        batch = X[idx]
        if batch[0].sum() == 0:
            num_zero += 1
            err.append(idx)
    end = time.time()
    print([x.shape for x in batch])
    print(err)

    print(f'time per execution: {((len(X)*((end - start)/lim)))/60:.2e}m', len(err))


