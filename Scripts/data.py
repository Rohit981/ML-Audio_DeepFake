import pandas as pd
import librosa
import os
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np
import torch

class ASVpoofDataset(Dataset):
    def __init__(self, base_dir, part="train", sr=16000, n_mels=128):
        super().__init__()

        self.base_dir = base_dir
        self.part = part
        self.audio_dir = os.path.join(self.base_dir, f"ASVspoof2019_LA_{part}", "flac")
        protocol_file = os.path.join(self.base_dir, "ASVspoof2019_LA_cm_protocols", 
                                     f"ASVspoof2019.LA.cm.{part}.trl.txt")

        self.sr = sr
        self.n_mels = n_mels
        self.label_map = {"bonafide":0, "spoof":1}

        #Parse protocol to get file names and labels
        self.file_list, self.file_labels = self._parse_protocol(protocol_file)

    def _parse_protocol(self, protocol_file):
        files,labels = [], []
        with open(protocol_file, "r") as f:
            for line in f:
                parts = line.strip().split()

                files.append(parts[1])
                labels.append(self.label_map[parts[4].strip()])
            
        return files,labels
    
    def __len__(self):
        return len(self.file_list)
    
    def data_balancing(self, train_dataset):
        #Get list of labels directly from dataset object
        train_labels = train_dataset.file_labels

        #Define the counts for bonafide and spoof
        count_0 = train_labels.count(0)
        count_1 = train_labels.count(1)
        class_count = [count_0, count_1]
        class_weights = [1.0 / c for c in class_count] # Calculates inverse frequency

        #Assign a specific sampling weight to every file in the dataset
        sample_weights = [class_weights[label] for label in train_labels]
        sample_weights = torch.tensor(sample_weights, dtype=torch.float)

        #Create weighted random sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return sampler
    
    def __getitem__(self, index):
        #Construct path and load data with librosa
        audio_path = os.path.join(self.audio_dir, f"{self.file_list[index]}.flac")
        y,_ = librosa.load(audio_path,sr=self.sr)

        #Extract mel spectrogram using Librosa
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)

        #Convert power to decibels
        mel_decb = librosa.power_to_db(mel_spec,ref=np.max)
        
        #Adding padding for time frames to set it to 128
        target_shape = 128
        if mel_decb.shape[1] < target_shape:
            pad_width = target_shape - mel_decb.shape[1]
            mel_decb = np.pad(
                mel_decb,
                pad_width=((0,0), (0,pad_width)),
                mode="constant"
            )
        else:
            mel_decb = mel_decb[:, :target_shape]

        #Add a channel dimensions for CNN
        mel_decb = np.expand_dims(mel_decb,axis=0)

        #Convert to pytorch labels
        X_tensor = torch.tensor(mel_decb,dtype=torch.float32)

        #Add Gausian Noise for data augmentation to train dataset only
        if self.part == "train":
            noise = torch.rand_like(X_tensor)*0.005
            X_tensor = X_tensor + noise

        Y_tensor = torch.tensor(self.file_labels[index], dtype=torch.float32)

        return X_tensor, Y_tensor



