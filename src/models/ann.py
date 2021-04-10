"""
Author: Kourosh T. Baghaei
April 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .databasednormalizer import DataBasedNormalizer

class ANN(nn.Module):
    """ A fully connected network with ReLU activations and no bias."""
    def __init__(self):
        super(ANN, self).__init__()
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(784, 1200,bias=False)        
        self.fc2 = nn.Linear(1200, 1200,bias=False)
        self.fc3 = nn.Linear(1200, 10,bias=False)

        self.fc1.weight = nn.Parameter(self.calc_weights(self.fc1.weight.shape))
        self.fc2.weight = nn.Parameter(self.calc_weights(self.fc2.weight.shape))
        self.fc3.weight = nn.Parameter(self.calc_weights(self.fc3.weight.shape))

        self.dbnorm = DataBasedNormalizer(self.get_linears())
    def calc_weights(self,shape):
        w = (torch.rand(shape,dtype=torch.float32) - 0.5) * 0.01 * 2
        return w

    def forward(self, x):        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        self.dbnorm.update(0,x)
        x = self.fc2(x)                
        x = F.relu(x)
        x = self.dropout2(x)
        self.dbnorm.update(1,x)
        x = self.fc3(x)
        output = F.relu(x)
        self.dbnorm.update(2,output)
        return output

    def get_linears(self):
        return [self.fc1,self.fc2,self.fc3]
