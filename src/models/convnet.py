"""
Author: Kourosh T. Baghaei
April 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1,bias=False)
        self.conv2 = nn.Conv2d(16, 32, 3, 2,bias=False)
        
        # for FCN
        self.conv3 = nn.Conv2d(32, 16, 2 , 1,bias=False)
        self.conv4 = nn.Conv2d(16, 10, 1,bias=False)

        # for ANN + CNN
        self.fc1 = nn.Linear(288, 64,bias=False)
        self.fc2 = nn.Linear(64, 10,bias=False)

    def forward_fcn(self, x): # FCN
        x = self.conv1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x,2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x,2)        
        x = self.conv3(x)
        x = F.relu(x)
        x = F.avg_pool2d(x,2)    
        x = self.conv4(x)
        x = F.relu(x)    
        output = torch.squeeze(x)
        
        return output
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x,2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x,2)
        x = torch.flatten(x, start_dim=1)                      
        x = self.fc1(x)
        x = self.fc2(x)



        output = F.relu(x)
        output = torch.squeeze(output)
        return output
