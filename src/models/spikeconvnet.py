"""
Author: Kourosh T. Baghaei
April 2021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convnet import ConvNet
from .spikingconv2D import SpikingConv2D
from .spikingpool2D import SpikingAveragePool2D
from .spikinglinear import SpikingLinear

class SpikeConv(nn.Module):
    def __init__(self, conv_net : ConvNet = None, device = None, dt=0.001, ref_period=0.0, threshold=1, all_sim_steps=False):
        super(SpikeConv, self).__init__()
        self.conv1 = SpikingConv2D(conv2d_nn=conv_net.conv1,device=device)
        self.conv2 = SpikingConv2D(conv2d_nn=conv_net.conv2,device=device)
        self.device = device
        self.avg_pool1 = SpikingAveragePool2D(device=device)
        self.avg_pool2 = SpikingAveragePool2D(device=device)
        
        self.lin1 = SpikingLinear(conv_net.fc1, device=device)
        self.lin2 = SpikingLinear(conv_net.fc2, device=device)
        
        # if set true, the output shape would be: (simulation_steps, batch_size)
        # otherwise: (batch_size,)
        self.all_sim_steps = all_sim_steps        

    def reset_layers(self):
        self.conv1.reset_layer()
        self.conv2.reset_layer()
        self.avg_pool1.reset_layer()
        self.avg_pool2.reset_layer()
        self.lin1.reset_layer()
        self.lin2.reset_layer()

    def forward(self, x): # FCN
        """
            Shape of X : (batch_size, d1,d2,...,dn, simulation_epochs)    
            Shape of Output: (batch_size,last_layer_out,simulation_epochs)
        """
        output_all_time = None
        if self.all_sim_steps:
            output_all_time = torch.zeros((x.shape[0],x.shape[-1]),requires_grad=False).to(self.device)             

        sim_steps = x.shape[-1]
        for si in range(sim_steps):
            tmp = x[...,si]            
            xi = self.conv1(tmp)
            xi = self.avg_pool1(xi)
            xi = self.conv2(xi)
            xi = self.avg_pool2(xi)
            xi = torch.flatten(xi, start_dim=1)
            xi = self.lin1(xi)
            xi = self.lin2(xi)

            if self.all_sim_steps:                
                sqz = torch.squeeze(self.lin2.sum_spikes)
                output_all_time[...,si] = torch.argmax(sqz, dim=1)
        
        output = None
        if self.all_sim_steps:
            # Change the shape to: (sim_step , batch)
            # so that it could be compared against targets with shape: (batch,)            
            output = output_all_time.transpose(0,1)
        else:
            sqz = torch.squeeze(self.lin2.sum_spikes)
            output = torch.argmax(sqz,dim=1)

        # reset layers to start simulation from zero state on the next batch
        self.reset_layers()
        return output 