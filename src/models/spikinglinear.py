"""
Author: Kourosh T. Baghaei
April 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikingLinear(nn.Module):
    """
    Applies a linear transformation on the input spikes. And holds the states of 
    the neurons. It's weights are initialized from a linear model with no bias.

    batch_size : can be given in constructor. Or once the layer is applied on
    some data.
    inputs to this layer are one single time step of spikes. And the outputs are
    also one single time step.
    """
    def __init__(self,linear_nn : nn.Linear = None,device=None, **kwargs ):        
        super(SpikingLinear,self).__init__()
        if not isinstance(linear_nn,nn.Linear):
            raise TypeError("The input arguemnt `linear_nn` MUST be of type {}".
                            format(nn.Linear))
        self.in_features = linear_nn.in_features
        self.out_features = linear_nn.out_features        
        self.sim_params = {
            "batch_size" : None,
            "dt" : torch.as_tensor(0.1,dtype=torch.float32),
            "threshold" : torch.as_tensor(1,dtype=torch.float32),
            "ref_period" : torch.as_tensor(0,dtype=torch.float32),
            "dtype" : torch.float32, # TODO get this from input layer          
        }
        self.device = device
        # set whatever parameters are available in the constructor
        self._init_sim_params(**kwargs)
        self.zero = torch.as_tensor(0,dtype=self.sim_params['dtype'])
        with torch.no_grad():
            src_weights = linear_nn.weight.cpu().detach()
            new_weights = torch.Tensor(src_weights.detach().clone())
            if device is not None:
                new_weights = new_weights.to(device)
                self.zero = self.zero.to(device)
            self.weight = nn.Parameter(new_weights,requires_grad=False)        
            if self.sim_params["batch_size"] is not None:
                self.init_spiking_params()
        
    def init_spiking_params(self):        
        self.spike_params_shape = (self.sim_params["batch_size"],self.out_features)
        self.neuron_ref_cooldown = torch.zeros(self.spike_params_shape,requires_grad=False)
        self.sum_spikes = torch.zeros(self.spike_params_shape,requires_grad=False)
        self.membrane_potential = torch.zeros(self.spike_params_shape,requires_grad=False)
        self.zeros = torch.zeros(self.spike_params_shape,requires_grad=False)
        if self.device is not None:            
            self.neuron_ref_cooldown = self.neuron_ref_cooldown.to(self.device)
            self.sum_spikes = self.sum_spikes.to(self.device)
            self.membrane_potential = self.membrane_potential.to(self.device)
            self.zeros = self.zeros.to(self.device)
    
    def set_threshold(self,new_threshold):
        self.sim_params['threshold'] =  torch.as_tensor(new_threshold,dtype=torch.float32,device=self.device)

    def reset_layer(self):
        self.init_spiking_params()

    def _init_sim_params(self,**kwargs):
        for k,v in self.sim_params.items():
            varg = kwargs.get(k)
            if varg is not None:
                if k in ["batch_size","dtype"]:
                    self.sim_params[k] = varg
                else:
                    self.sim_params[k] = torch.as_tensor(varg,dtype=self.sim_params["dtype"])
                    if self.device is not None:
                        self.sim_params[k] = self.sim_params[k].to(self.device)

    def forward(self, spikes):
        if self.sim_params["batch_size"] is None:
            self.sim_params["batch_size"] = spikes.shape[0]
            self.init_spiking_params()            
        
        device = self.device
        with torch.no_grad():
            # Get input impulse from incoming spikes
            impulse = F.linear(spikes,self.weight)
            
            # Do not add impulse if neuron is in refactory period
            in_ref_per = self.neuron_ref_cooldown > self.zero
            if device is not None:
                in_ref_per = in_ref_per.to(device)
            impulse = torch.where(in_ref_per,self.zero, impulse)
            # Add input impulse to membrane potentials
            self.membrane_potential = self.membrane_potential + impulse
            # Check for spiking neurons
            
            spikings_bool = self.membrane_potential >= self.sim_params["threshold"]
            if device is not None:    
                spikings_bool = spikings_bool.to(device)
            
            # Reset the potential of the membrane if it has exceeded the threshold
            self.membrane_potential = torch.where(spikings_bool, self.zero, self.membrane_potential)
            # Excited neurons should go to cooldown state
            self.neuron_ref_cooldown = torch.where(spikings_bool, self.sim_params["ref_period"], self.neuron_ref_cooldown)
            # Cooldown timer count-down
            self.neuron_ref_cooldown = self.neuron_ref_cooldown - self.sim_params["dt"]
            # Prevent from getting negative values
            self.neuron_ref_cooldown = torch.max(self.neuron_ref_cooldown,self.zero)
            # calculate the output of this layer
            out_spikes = spikings_bool.type(self.sim_params["dtype"]) 
            self.sum_spikes = self.sum_spikes + out_spikes

        return out_spikes