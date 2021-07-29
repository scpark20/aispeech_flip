import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import librosa
from os import listdir
from os.path import isdir, isfile, join
from jamo import text_to_tokens, tokens_to_text, n_symbols, refine_ksponspeech

class KSponSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.file_list = self._get_file_list(root_dir)
        print(len(self.file_list))
        
    def _get_file_list(self, root_dir):
        file_list = []
        files = [join(root_dir, f) for f in listdir(root_dir) if join(root_dir, f)]
        for file in files:
            if isdir(file):
                ret_list = self._get_file_list(file)
                file_list.extend(ret_list)
            elif isfile(file):
                if file.endswith('txt'):
                    pcm_file = file[:-3] + 'pcm'
                    file_list.append({'txt': file, 'pcm': pcm_file})

        return file_list
    
    def _get_audio(self, file):
        with open(file, 'rb') as f:
            wav = np.fromfile(f, dtype=np.int16)
            wav = wav / 32768.
            S = librosa.feature.melspectrogram(wav, sr=16000, n_fft=1024, n_mels=80, hop_length=256)
            S = (np.log10(S + 1e-5) - np.log10(1e-5)) / -np.log10(1e-5)
            
        return S.T
            
    def _get_text(self, file):
        with open(file, 'r', encoding='cp949') as f:
            l = f.read()
            l = refine_ksponspeech(l)
            array = text_to_tokens(l)
        #array = np.pad(array, (1, 1), 'constant', constant_values=(0, 0))
        return array
        
    def __getitem__(self, index):
        while True:
            text = self._get_text(self.file_list[index]['txt'])
            if len(text) > 180:
                index = (index + 1) % self.__len__()
                continue

            audio = self._get_audio(self.file_list[index]['pcm'])    
            if len(audio) > 450:
                index = (index + 1) % self.__len__()
                continue
                
            break
        
        return torch.FloatTensor(audio), torch.LongTensor(text)
        
    def __len__(self):
        return len(self.file_list)
    
class DataCollate():
    def __call__(self, batch):
        audio_lengths = []
        text_lengths = []
        for audio, text in batch:
            audio_lengths.append(len(audio))
            text_lengths.append(len(text))
            
        audio_max_length = max(audio_lengths)
        text_max_length = max(text_lengths)
        
        audio_padded = torch.FloatTensor(len(batch), audio_max_length, 80)
        audio_padded.zero_()
        audio_lengths = torch.from_numpy(np.array(audio_lengths)).long()
        
        text_padded = torch.LongTensor(len(batch), text_max_length)
        text_padded.zero_()
        text_lengths = torch.from_numpy(np.array(text_lengths)).long()
        
        for i, (audio, text) in enumerate(batch):
            audio_padded[i, :len(audio)] = audio
            text_padded[i, :len(text)] = text
            
        outputs = {'audio': audio_padded,
                   'audio_lengths': audio_lengths,
                   'text': text_padded,
                   'text_lengths': text_lengths
                  }
        
        return outputs