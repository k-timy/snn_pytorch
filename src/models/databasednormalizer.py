"""
Author: Kourosh T. Baghaei
April 2021
"""

# DataBasedNormalizer

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter

# Currently only works on Linear layers
class DataBasedNormalizer(object):
    """
    Pytorch implementation of weight normalization introduced in paper:
    'Fast-Classifying, High-Accuracy Spiking Deep Networks Through Weight and Threshold Balancing'
    available at: https://doi.org/10.1109/IJCNN.2015.7280696
    """
    def __init__(self, layers):
        self.__layers = layers
        self.layers_count = len(layers)
        self.__max_activations = torch.zeros(self.layers_count)
        self.__max_weights = torch.zeros(self.layers_count)
        for li in range(self.layers_count):
            self.__max_weights[li] = layers[li].weight.max()
        self.recording = False
        

    # This function should be called in order to make this class work
    # otherwise, it wont work to save performance
    def start_log(self,device):
        self.recording = True
        self.__max_activations = self.__max_activations.to(device)
        self.__max_weights = self.__max_weights.to(device)
    # This function ends logging
    def end_log(self):
        self.recording = False

    def __geq(self,t1,t2):
        return torch.where(t1 > t2, t1, t2)

    def update(self, layer_index, activations):
        if not self.recording:
            return
        prev_act_max = self.__max_activations[layer_index]
        act_max = activations.max()        
        self.__max_activations[layer_index] = self.__geq(act_max,prev_act_max)
    
    def __backup_weights(self):
        self.backups = []
        for li in self.__layers:
            self.backups.append(li.weight.detach().clone())

    def revert(self):
        for li in range(self.layers_count):
            self.__layers[li].weight = Parameter(self.backups[li])               

    def normalize(self):
        self.__backup_weights()
        previous_factor = 1
        factors = torch.zeros(self.layers_count)
        with torch.no_grad():
            for i in range(self.layers_count):
                scale_factor = self.__geq(self.__max_weights[i],self.__max_activations[i])
                applied_inv_factor = scale_factor / previous_factor
                self.__layers[i].weight = Parameter(self.__layers[i].weight / applied_inv_factor)
                factors[i] = 1 / applied_inv_factor
                previous_factor = applied_inv_factor        
        return factors
