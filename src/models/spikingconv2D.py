"""
Author: Kourosh T. Baghaei
April 2021

Spiking Convolutional 2D Layer : 2D Convolutional SNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import types

class SpikingConv2D(nn.Module):
    """
    Applies a 2D Convolution transformation on the input spikes. And holds the states of 
    the neurons. It's weights are initialized from a regular 2D convolution layer with no bias.

    batch_size : can be given in constructor. Or once the layer is applied on
    some data.
    inputs to this layer are one single time step of spikes. And the outputs are
    also one single time step.
    """
    def __init__(self,conv2d_nn : nn.Conv2d = None,device=None, **kwargs ):        
        super(SpikingConv2D,self).__init__()
        if not isinstance(conv2d_nn,nn.Conv2d):
            raise TypeError("The input arguemnt `conv2d_nn` MUST be of type {}".
                            format(nn.Conv2d))
        self.in_channels = conv2d_nn.in_channels
        self.out_channels = conv2d_nn.out_channels
        self.kernel_size = conv2d_nn.kernel_size
        self.stride = conv2d_nn.stride
        self.padding = conv2d_nn.padding       
        self._sim_params_dict = {
            "batch_size" : None,
            "dt" : torch.as_tensor(0.001,dtype=torch.float32),
            "threshold" : torch.as_tensor(1,dtype=torch.float32),
            "ref_period" : torch.as_tensor(0,dtype=torch.float32),
            "dtype" : torch.float32, # TODO get this from input layer          
        }
        self.sim_params = types.SimpleNamespace()
        self.device = device
        self.spike_params_shape = None
        # set whatever parameters are available in the constructor
        self._init_sim_params(**kwargs)
        self.zero = torch.as_tensor(0,dtype=self._sim_params_dict['dtype'])
        with torch.no_grad():
            new_weights = conv2d_nn.weight.cpu().detach().clone()            
            if device is not None:
                new_weights = new_weights.to(device)
                self.zero = self.zero.to(device)
            self.weight = nn.Parameter(new_weights,requires_grad=False)        
        
    def init_spiking_params(self):                
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
        self.sim_params.threshold = torch.as_tensor(new_threshold,dtype=self.sim_params.dtype,device=self.device)

    def reset_layer(self):
        self.init_spiking_params()

    def _init_sim_params(self,**kwargs):
        for k,v in self._sim_params_dict.items():            
            varg = kwargs.get(k) 
            # if varg is not set
            if varg is not None:             
                if k in ["batch_size","dtype"]:
                    setattr(self.sim_params, k, varg)
                else:                    
                    if self.device is not None:
                        setattr(self.sim_params, k, torch.as_tensor(varg,dtype=self._sim_params_dict["dtype"]).to(self.device))
                    else:
                        setattr(self.sim_params, k, torch.as_tensor(varg,dtype=self._sim_params_dict["dtype"]))
            # if varg == None
            else:                        
                if k in ["batch_size","dtype"]:
                    setattr(self.sim_params, k, v)
                else:
                    v_ = torch.as_tensor(v,dtype=self._sim_params_dict["dtype"])
                    if self.device is not None:
                        setattr(self.sim_params, k, v_.to(self.device))
                    else:
                        setattr(self.sim_params, k, v_)
                    

    def clone(self):
        raise Exception("TODO: Not implemented yet!")

    def _get_shape_of_output(self, spikes) -> torch.Tensor.shape:
        """
        Regardless of the Batchsize, Channels, Height and Width of the input data,
        The pooling layer works. However, the spike neurons should be able to keep
        track of the membraine poteintials, refactory cooldown, etc. So this
        function gets the shape of the output based on a batch of input spikes and
        lazily initializes the SNN parameters.
        """
        sample_out = None
        with torch.no_grad():
            sample_ones = torch.ones_like(spikes)
            sample_out = self._conv2d(sample_ones)
        return sample_out.shape                       

    def _conv2d(self, spikes):
        return F.conv2d(spikes, self.weight, stride=self.stride, padding=self.padding)

    def forward(self, spikes):
        if self.spike_params_shape is None:
            self.sim_params.batch_size = spikes.shape[0]
            self.spike_params_shape = self._get_shape_of_output(spikes)
            self.init_spiking_params()
        
        device = self.device
        with torch.no_grad():
            # Get input impulse from incoming spikes
            impulse = self._conv2d(spikes)

            # Do not add impulse if neuron is in refactory period
            in_ref_per = self.neuron_ref_cooldown > self.zero
            if device is not None:
                in_ref_per = in_ref_per.to(device)
            impulse = torch.where(in_ref_per, self.zero, impulse)
            # Add input impulse to membrane potentials
            self.membrane_potential = self.membrane_potential + impulse
            # Check for spiking neurons
            
            spikings_bool = self.membrane_potential >= self.sim_params.threshold
            if device is not None:    
                spikings_bool = spikings_bool.to(device)
            
            # Reset the potential of the membrane if it has exceeded the threshold
            self.membrane_potential = torch.where(spikings_bool, self.zero, self.membrane_potential)
            # Excited neurons should go to cooldown state
            self.neuron_ref_cooldown = torch.where(spikings_bool, self.sim_params.ref_period, self.neuron_ref_cooldown)
            # Cooldown timer count-down
            self.neuron_ref_cooldown = self.neuron_ref_cooldown - self.sim_params.dt
            # Prevent from getting negative values
            self.neuron_ref_cooldown = torch.max(self.neuron_ref_cooldown,self.zero)
            # calculate the output of this layer
            out_spikes = spikings_bool.type(self.sim_params.dtype)
            self.sum_spikes = self.sum_spikes + out_spikes

        return out_spikes