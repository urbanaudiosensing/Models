import torch
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
from transformers import ASTFeatureExtractor

from .backend import _VGGish

NUM_FRAMES = 96
SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = 64
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.

'''
from torchaudio.prototype.pipelines import VGGISH

VGGISH_SR = VGGISH.sample_rate

class VGGish_PreProc(nn.Module):
    def __init__(self):
        super().__init__()
        self.F = VGGISH.get_input_processor()
    def forward(self, X, sr):
        X = F.resample(X, sr, VGGISH_SR)
        return self.F(X.flatten())

class VGGish(nn.Module):
    def __init__(self, pproc = True):
        super().__init__()
        self.preprocess = VGGish_PreProc() if pproc else None
        self.model = VGGISH.get_model()

    def forward(self, *args):
        #Expect one of [Tensor, SampleRate] or [Tensor]
        
        if self.preprocess is None:
            return self.model(args[0])
        return self.model(self.preprocess(*args))
        

class AST_PreProc(nn.Module):
    pass

'''

class VGGish(torch.nn.Module):
    def __init__(self, pproc=True, sr=SAMPLE_RATE, trainable = False, pre_trained=True):
        super().__init__()
        self.model = _VGGish(preprocess=False, pretrained=pre_trained) #torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.preprocess = False
        self.preprocess = pproc
        self.sr = sr
        self.trainable = trainable

        if trainable:
            self.model.train()
            self.model.postprocess = False
        else:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.preprocessor = VGGish_PreProc()
    
    def forward(self, frame, sr=None):
        if sr is None:
            sr = self.sr
        if len(frame.shape) < 4:
            frame = self.preprocessor(frame, sr)
        
        if not self.trainable:
            with torch.no_grad():
                return self._forward(frame, sr)
        return self._forward(frame, sr)
            
    
    def _forward(self, frame, sr):
        if (len(frame.shape) > 4): #Shape [N, segment_len, 1, T, F]
                output = frame.view(-1, *frame.shape[2:])
                output = self.model(output, sr)
                return output.view(*frame.shape[:2], output.shape[-1])
        return self.model(frame, sr)
    
class VGGish_PreProc(torch.nn.Module):
    def __init__(
        self,
        #input_freq=16000, #samples
        resample_freq=SAMPLE_RATE, #samples
        #n_fft=512, #bands
        n_mel=NUM_MEL_BINS, #bands
        hop_length=STFT_HOP_LENGTH_SECONDS, #Seconds
        win_length = STFT_WINDOW_LENGTH_SECONDS, #Seconds
        do_normalize = True,
        mean=-5.8703, #measured for ASPEDv1
        std = 0.0043 #measured for ASPEDv1
    ):
        super().__init__()
        hop_length = np.round(hop_length * resample_freq).astype(int)
        win_length = np.round(win_length * resample_freq).astype(int)

        n_fft = 2 ** int(np.ceil(np.log(win_length) / np.log(2.0)))

        self.resample_freq = SAMPLE_RATE
        self.do_normalize = do_normalize
        self.mean = mean
        self.std = std
        '''
        if resample_freq == input_freq:
            self.resample = torch.nn.Identity()
        else:
            self.resample = T.Resample(orig_freq=input_freq, new_freq=resample_freq)
        '''
        self.spec = T.Spectrogram(n_fft=n_fft, power=2, hop_length=hop_length, win_length = win_length)

        self.mel_scale = T.MelScale(
            n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1,f_min = MEL_MIN_HZ, f_max=MEL_MAX_HZ)

    def forward(self, waveform: torch.Tensor, sr:int) -> torch.Tensor:
        # Resample the input
        if sr != self.resample_freq:
            resampled = F.resample(waveform, sr, self.resample_freq)
        else:
            resampled = waveform

        # Convert to power spectrogram
        spec = self.spec(resampled)

        # Convert to mel-scale
        mel = self.mel_scale(spec) + 1e-10 #log offset

        logmel = torch.log(mel.permute(0, 2, 1)[:, :NUM_FRAMES, :]).unsqueeze(dim=1)

        if self.do_normalize:
            return (logmel - self.mean) / self.std
        return logmel
    
class AST_PreProc(torch.nn.Module):
    def __init__(
        self,
        mu=-7.8449515625, #Measured for ASPEDv1 
        sigma=0.8363084375, #Measured for ASPEDv1 
        max_length = 100,
        mel_bins=128,
        sr = 16000
    ):
        super().__init__()
        self.ast_fn = ASTFeatureExtractor(do_normalize=True, max_length=max_length, 
                                          sampling_rate = sr, mean=mu, std=sigma, mel_bins=mel_bins)
        self.sr = sr

    def forward(self, waveform: torch.Tensor, sr:int) -> torch.Tensor:
        # Resample the input
        if sr != self.sr:
            resampled = F.resample(waveform, sr, self.sr)
        else:
            resampled = waveform
        if len(resampled.shape) > 2:
            resampled = resampled.view(-1, *resampled.shape[2:])
        
        X = list(resampled.squeeze().numpy())

        output = self.ast_fn(X, sampling_rate=16000, return_tensors ='pt')['input_values']

        return torch.Tensor(output)