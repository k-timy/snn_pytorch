"""
Author: Kourosh T. Baghaei
April 2021
"""


import torch
import torch.nn as nn
import numpy as np
from src.models.spikingconv import SpikingConv2D

# a batch of one single image with one single channel
img = torch.as_tensor(
    [
        [
            [
                [ 0, 1, 1, 1],
                [ 0, 1, 0 ,0],
                [ 0, 1, 0, 0],
                [ 0, 1, 0, 0]
            ]
        ]
    ]
).float()

# Kernels: 
kernels = torch.as_tensor(
    [
        [
            [
                [0.0, 0.2],
                [0.0, 0.2]
            ]
        ],
        [
            [
                [0.1, 0.1],
                [0.1, 0.0]
            ]
        ],
        [
            [
                [0.15, 0.15],
                [0.0, 0.0]
            ]
        ],

    ]
).float()

conv_lay = torch.nn.Conv2d(1,3,2,bias=False)
conv_lay.weight = torch.nn.Parameter(kernels)
print(conv_lay.weight)
print(kernels.shape)
print(img.shape)

print('regular conv layer:')
out_conv = conv_lay(img)
print('out conv:')
print(out_conv)
print('sum spikes:')    
print('-------------------')

conv_spike = SpikingConv2D(conv_lay)
for i in range(10):
    print('((( {} )))'.format(i + 1))
    out_conv = conv_spike(img)
    print('out conv:')
    print(out_conv)
    print('sum spikes:')
    print(conv_spike.sum_spikes)    
    print('-------------------')
