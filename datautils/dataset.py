# this file will be custom depending on the project

from torch.utils.data import Dataset
import torch
import os
from torchvision.io import read_image

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
       
       
class CustomImagenetDataset(Dataset):
    
    def __init__(self, data_files):
        self.data_files = data_files
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        rgb_tensor = read_image(self.data_files[idx]) 
        sample = { 'RGB': rgb_tensor }
        
        return sample