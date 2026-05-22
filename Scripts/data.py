import pandas as pd
import librosa
import os
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np
import torch
from tqdm import tqdm

class ASVpoofDataset(Dataset):
    def __init__(self, base_dir, part="train", sr=16000, n_mels=128,precompute=False):
        super().__init__()

        self.base_dir = base_dir
        self.part = part
        self.audio_dir = os.path.join(self.base_dir, f"ASVspoof2019_LA_{part}", "flac")
        protocol_file = os.path.join(self.base_dir, "ASVspoof2019_LA_cm_protocols", 
                                     f"ASVspoof2019.LA.cm.{part}.trl.txt")
        
        #Save Audio files directory in .npy
        save_base_dir = r"D:\Deep Neural Network\ML-Audio_DeepFake\Datasets\Precompute"
        self.save_dir = os.path.join(save_base_dir, part)
        

        self.sr = sr
        self.n_mels = n_mels
        self.label_map = {"bonafide":0, "spoof":1}

        #Parse protocol to get file names and labels
        self.file_list, self.file_labels = self._parse_protocol(protocol_file)

        #Only Precompute the data if need to
        if precompute:
            self.pre_compute_mel_spec()

        #Cache all arrays straight into RAM to avoid disk bottlenecks
        if part == "train" or part=="dev":
            print(f"Loading {part} dataset into RAM...")
            self.cached_features=[]
            for file_name in tqdm(self.file_list, desc="Ram-caching"):
                npy_path = os.path.join(self.save_dir, f"{file_name}.npy")
                # Read from disk exactly once here
                self.cached_features.append(np.load(npy_path))

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
    
    #Precompute the dataset and store them in .npy
    def pre_compute_mel_spec(self):
        os.makedirs(self.save_dir,exist_ok=True)

        for file_name in tqdm(self.file_list, desc=f"Precomputing {self.part}"):
            #Construct save and audio path and load data with librosa
            save_path = os.path.join(self.save_dir, f"{file_name}.npy")
            if os.path.exists(save_path):
                continue

            audio_path = os.path.join(self.audio_dir, f"{file_name}.flac")

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
            mel_decb = np.expand_dims(mel_decb,axis=0).astype(np.float32)

            #Save the computed file
            np.save(save_path,mel_decb)

    def __getitem__(self, index):

        # npy_path = os.path.join(self.save_dir, f"{self.file_list[index]}.npy")

        mel_decb = self.cached_features[index]
       
        #Convert to pytorch labels
        X_tensor = torch.from_numpy(mel_decb).float()

        # #Add Gausian Noise for data augmentation to train dataset only
        # if self.part == "train":
        #     noise = torch.randn_like(X_tensor)*0.005
        #     X_tensor = X_tensor + noise

        Y_tensor = torch.tensor(self.file_labels[index], dtype=torch.float32)

        return X_tensor, Y_tensor



