import json
import uuid 
import os
from torch import Tensor

import matplotlib.pyplot as plt
import numpy as np

FILE_EXT = '.json'

#manifest fields
CONFIG = 'config'
EVAL = 'eval'
METADATA = 'metadata'
VIZ_DIR = 'viz'
MODEL_PATH = 'model_path'

class manifestHandler():
    def __init__(self, config = dict(), eval = dict(), metadata = dict(), model_path='', name = str(uuid.uuid4()), save_path=None):
        self.save_path = save_path
        self.name = name 
        self.config = config 
        self.eval = eval 
        self.metadata = metadata
        self.model_path = model_path
        self.manifest = {CONFIG: config, EVAL: eval, METADATA: metadata, MODEL_PATH:model_path}
        self.viz_filepath = None
    
    def add(self, X: dict[str, dict]):
        for k in X.keys():
            if k not in self.manifest.keys():
                self.manifest.update(X)
            else:
                self.manifest[k].update(X[k])
    
    def save(self):
        '''
        saves json file of training/evaluation manifest with following entries:

            CONFIG: specified data/model configuration for given run
            EVAL: summary of evaluation metrics for given run
            METADATA: [partially deprecated], any relevant metadata for given run
            MODEL_PATH: filepath to best performing checkpoint for given run

        File name: [encoder_type]-r-[training radius]-tt-[training threshold]-[count].json

        '''
        assert not self.save_path is None

        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        
        dir_contents = os.listdir(self.save_path)

        radius = self.manifest[CONFIG]['data_params']['radius']
        train_thresh = self.manifest[CONFIG]['model_params']['n_classes']
        encoder = self.manifest[CONFIG]['model_params']['encoder']
        self.name = f'{encoder}-r-{radius}-tt-{train_thresh}'

        count = len([x for x in dir_contents if x.startswith(self.name)])
        self.name = f'{self.name}-{count}'

        #os.mkdir(os.path.join(self.save_path, self.name))

        self.manifest = self.proc_dict(self.manifest)
        print(self.manifest)

        with open(os.path.join(self.save_path, f'{self.name}{FILE_EXT}'), 'w') as f:
            json.dump(self.manifest, f, indent=4)
    
    '''
    if any key in manifest dict is a torch tensor, convert to string
    '''
    def proc_dict(self, d):
        for x in d.keys():
            if type(d[x]) == dict:
                if type(x) == Tensor:
                    d[str(x.item())] = self.proc_dict(d[x])
                else:
                    d[str(x)] = self.proc_dict(d[x])

            elif type(d[x]) == Tensor:
                if type(x) == Tensor:
                    d[str(x.item())] = str(d[x].item())
                else:
                    d[str(x)] = str(d[x].item())
            else:
                if type(x) == Tensor:
                    d[str(x.item())] = str(d[x])
                else:
                    d[str(x)] = str(d[x])
        return d

        