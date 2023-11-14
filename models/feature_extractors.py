import torch
import librosa
import numpy as np


class VGGish():
    def __init__(self):
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.eval()
    
    def forward(self, frame, sr):
        frame = self.pow2pad(frame)
        if len(frame.shape) > 1:
            frame = np.mean(frame, axis=1)
        return self.model.forward(frame, sr)

    def pow2pad(self, X):
        total_length = int(2**(np.ceil(np.log2(X.shape[0]))))
        return np.concatenate((X, np.zeros((total_length - X.shape[0], X.shape[1]))), axis=0)
    

class mfcc():
    def __init__(self, n_mfcc = 20):
        self.n_mfcc = n_mfcc
    
    def forward(self, frame, samplerate):
        frame = self.pow2pad(frame)
        if len(frame.shape) > 1:
            frame = np.mean(frame, axis=1)
        return torch.Tensor(librosa.feature.mfcc(y=frame, sr=samplerate, n_mfcc=self.n_mfcc))

    def pow2pad(self, X):
        total_length = int(2**(np.ceil(np.log2(X.shape[0]))))
        return np.concatenate((X, np.zeros((total_length - X.shape[0], X.shape[1]))), axis=0)

class m_spec():
    def __init__(self, n_mels = 128, n_fft = 512):
        self.n_mels = n_mels
        self.n_fft = 512
    
    def forward(self, frame, samplerate):
        frame = self.pow2pad(frame)
        if len(frame.shape) > 1:
            frame = np.mean(frame, axis=1)
        return torch.Tensor(librosa.feature.melspectrogram(y=frame, sr=samplerate, n_mels=self.n_mels, n_fft = self.n_fft))

    def pow2pad(self, X):
        total_length = int(2**(np.ceil(np.log2(X.shape[0]))))
        return np.concatenate((X, np.zeros((total_length - X.shape[0], X.shape[1]))), axis=0)