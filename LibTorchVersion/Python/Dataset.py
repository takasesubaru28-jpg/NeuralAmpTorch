import torch
from torch.utils.data import Dataset
import os
import numpy as np
from scipy.io import wavfile
from scipy import signal

class GuitarAmpDataset(Dataset):
    def __init__(self, input_wav_path, output_wav_paths, input_size, step_size, sample_rate):
        self.input_size = input_size
        self.step_size = step_size
        self.output_wav_paths = output_wav_paths
        self.sample_rate = sample_rate
        
        _, x_all = wavfile.read(input_wav_path)
        x_all = x_all.astype(np.float32)
        self.x_all = x_all.reshape(-1, 1)
        
        self.chunks_per_file = (len(self.x_all) - self.input_size) // self.step_size + 1
        
        self.params = []
        for out_path in output_wav_paths:
            filename = os.path.basename(out_path)
            p_str = os.path.splitext(filename)[0].split("_")[-1]
            p = np.array([float(p) for p in p_str.split("&")], dtype=np.float32) / 10
            self.params.append(p)

        self.current_file_idx = -1
        self.current_y_data = None

    def shuffle_files(self):
        combined = list(zip(self.output_wav_paths, self.params))
        import random
        random.shuffle(combined)
        self.output_wav_paths, self.params = zip(*combined)
        self.current_file_idx = -1

    def __len__(self):
        return len(self.output_wav_paths) * self.chunks_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.chunks_per_file
        start_pos = (idx % self.chunks_per_file) * self.step_size
        
        if file_idx != self.current_file_idx:
            _, y_all = wavfile.read(self.output_wav_paths[file_idx])
            y_all = y_all.astype(np.float32)
            self.current_y_data = y_all.reshape(-1, 1)
            self.current_file_idx = file_idx

        x = self.x_all[start_pos : start_pos + self.input_size]
        y = self.current_y_data[start_pos : start_pos + self.input_size]
        p = self.params[file_idx]

        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(p)
    

def apply_hpf(data, fs, cutoff=20):
    sos = signal.butter(4, cutoff, 'hp', fs=fs, output='sos')
    return signal.sosfilt(sos, data)