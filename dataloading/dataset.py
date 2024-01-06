# this file will be custom depending on the project

from torch.utils.data import Dataset
import torch
import os

class CustomDataset(Dataset):
    
    def __init__(self, X, y, length, transform = None):
        self.transform = transform
        self.length = length
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = { 'L': self.X[idx], 'AB': self.y[idx] }
        
        if self.transform:
            sample = self.transform(sample)

        return sample
        