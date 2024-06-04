# this file is to read data folder
# and get information such as length, x and y

from common.utils import get_config

import numpy as np
import os

class DataReader:
    
    def __init__(self):
        self.data_dir = get_config()['data_dir']
    
    def get_full_data(self):
        data = np.load(os.path.join(self.data_dir, 'ae_data.npz'))
        length = data['RGBtr'].shape[0] + data['RGBte'].shape[0]
        return { 'data': data, 'len': length }
    
    def get_split_data(self):
        data = np.load(os.path.join(self.data_dir, 'ae_data.npz'))
        return {
            'RGBtr': data['RGBtr'],
            'Ltr': np.expand_dims(data['Ltr'], 1),
            'ABtr': np.transpose(data['ABtr'], (0, 3, 1, 2)),
            'RGBte': data['RGBte'],
            'Lte': np.expand_dims(data['Lte'], 1),
            'ABte': np.transpose(data['ABte'], (0, 3, 1, 2)),
            'ftr_len': data['RGBtr'].shape[0],
            'te_len': data['RGBte'].shape[0]
        }


class ImagenetteReader:
    
    def __init__(self):
        self.data_dir = get_config()['data_dir']
    
    def get_data_filepaths(self):
        train_path = os.path.join(self.data_dir, 'imagenette2-320/train')
        test_path = os.path.join(self.data_dir, 'imagenette2-320/val')
        train_data = os.listdir(train_path)
        train_data_paths = [os.path.join(train_path, f) for f in train_data]
        test_data = os.listdir(test_path)
        test_data_paths = [os.path.join(test_path, f) for f in test_data]
        
        return train_data_paths, test_data_paths
    
    