"""
Author: Kourosh T. Baghaei
April 2021

SpikeNet: a neural network that is created by converting a regular neural network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .spikinglinear import SpikingLinear

class SpikeNet(nn.Module):
    def __init__(self,ann_model,device,dt=0.1,ref_period=0.0,batch_size=100,threshold=1,all_sim_steps=False):
        super(SpikeNet, self).__init__()
        linear_layers = ann_model.get_linears()
        self.sl1 = SpikingLinear(linear_nn=linear_layers[0],device=device,dt=dt,ref_period=ref_period,batch_size=batch_size,threshold=threshold)
        self.sl2 = SpikingLinear(linear_nn=linear_layers[1],device=device,dt=dt,ref_period=ref_period,batch_size=batch_size,threshold=threshold)
        self.sl3 = SpikingLinear(linear_nn=linear_layers[2],device=device,dt=dt,ref_period=ref_period,batch_size=batch_size,threshold=threshold)
        self.device = device
        # if set true, the output shape would be: (simulation_steps, batch_size)
        # otherwise: (batch_size,)
        self.all_sim_steps = all_sim_steps

    def forward(self, x):
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
            xi = self.sl1(tmp)
            xi = self.sl2(xi)
            xi = self.sl3(xi)
            if self.all_sim_steps:
                output_all_time[...,si] = torch.argmax(self.sl3.sum_spikes,dim=1)
        
        output = None
        if self.all_sim_steps:
            # Change the shape to: (sim_step , batch)
            # so that it could be compared against targets with shape: (batch,)            
            output = output_all_time.transpose(0,1)
        else:
            output = torch.argmax(self.sl3.sum_spikes,dim=1)
        
        self.reset_layers()
        return output

    def reset_layers(self):
        self.sl1.reset_layer()
        self.sl2.reset_layer()
        self.sl3.reset_layer()

    def set_threshold(self,new_threshold):
        for name, child in self.named_children():
            child.set_threshold(new_threshold)