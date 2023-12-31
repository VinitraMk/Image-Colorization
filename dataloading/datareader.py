# this file is to read data folder
# and get information such as length, x and y

from common.config import data_dir

import numpy as np
import os

class DataReader:
    
    def __init__(self):
        pass
    
    def get_full_data(self):
        data = np.load(os.path.join(data_dir, 'ae_data.npz'))
        length = data['RGBtr'].shape[0] + data['RGBTe'].shape[0]
        return { 'data': data, 'len': length }
    
    def get_split_data(self):
        data = np.load(os.path.join(data_dir, 'ae_data.npz'))
        return {
            'RGBtr': data['RGBtr'],
            'Ltr': data['Ltr'],
            'ABtr': data['ABtr'],
            'RGBte': data['RGBte'],
            'Lte': data['Lte'],
            'ABte': data['ABte'],
            'ftr_len': data['RGBtr'].shape[0],
            'te_len': data['RGBte'].shape[0]
        }


    