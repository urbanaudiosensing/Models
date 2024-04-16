import numpy as np
import os
from tqdm import tqdm
import torchaudio
import torchaudio.functional as F
import torch
import sys

'''
Script to convert the audio from wav to 16Khz downsampled, 1-channel (mono) numpy arrays
for easy dataloading (memmap, etc.)

structure of source directory will be replicated in the destination directory

Use: python convert_to_npy.py [source directory of audio data for session] [destination directory of numpy arrays]

    OR update SOURCE_PATH, DESTINATION_PATH below 

    update AUDIO_EXT depending on audio file extension

Dependencies: torch, torchaudio, tqdm
'''

SOURCE_PATH = '../Path_to_dataset/'

DESTINATION_PATH = '../Path_to_output/'

AUDIO_EXT = '.flac'


if __name__ == '__main__':
    if len(sys.argv) == 3:
        SOURCE_PATH, DESTINATION_PATH = tuple(sys.argv[1:])

    try:
        os.mkdir(DESTINATION_PATH)
    except FileExistsError as e:
        pass

    pbar = tqdm(os.walk(SOURCE_PATH), total=len(list(os.walk(SOURCE_PATH))))
    for root, dirs, files in pbar:
        prefix = root[len(SOURCE_PATH):]

        try:
            for x in dirs:
                os.mkdir(os.path.join(DESTINATION_PATH,prefix, x))

        except FileExistsError as e:
            pass
        
        audio_files = sorted([x for x in files if x.endswith(AUDIO_EXT) and x != 'bandpassed.wav'])

        #to exclude indices after end file
        if len(audio_files) > 0:
            audio_files = [audio_files[-1]] + audio_files[1:-1]
        
        for x in range(len(audio_files)):
            if 'end' in audio_files[x]:
                audio_files = audio_files[:x+1]
                break
        if len(audio_files) > 0:
            if not('split' in audio_files[0] and 'end' in audio_files[-1]):
                continue
        
        for idx, x in enumerate(audio_files):
            pbar.set_description(f'{round(idx / len(audio_files) * 100, 2)}%')
            waveform, sr = torchaudio.load(os.path.join(root, x))

            waveform = F.resample(waveform, orig_freq=sr, new_freq=16000)

            waveform = torch.mean(waveform, axis=0).numpy()

            waveform = waveform.astype(np.float32)

            with open(f'{os.path.join(DESTINATION_PATH, prefix, x[:-len(AUDIO_EXT)])}.npy', 'wb') as f:
                np.save(f, waveform)
        