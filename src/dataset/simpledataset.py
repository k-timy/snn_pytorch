"""
Author: Kourosh T. Baghaei
April 2021
"""

from torch.utils.data import Dataset
import torch

class SimpleDataset(Dataset):
    def __init__(self, x_values, y_values, out_shape):
        self.X = x_values
        self.y = y_values
        self.out_shape = out_shape

    def __len__(self):
        return (len(self.X))
        
    def __getitem__(self, index):
        return (torch.as_tensor(self.X[index].reshape(self.out_shape),dtype=torch.float32),
                torch.as_tensor(self.y[index],dtype=torch.long))
    
