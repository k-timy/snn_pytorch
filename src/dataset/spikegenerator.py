"""
Author: Kourosh T. Baghaei
April 2021
"""

from torch.utils.data import DataLoader
import torch
import numpy as np

class SpikeGenerator(object):
    """
    SpikeGenerator: an object that creates stream of random poisson activations from input vectors.
    Takes in a torch dataset and provide spikes given the configuration.
    """
    def __init__(self, dataset, sim_batch_size, sim_dt = 0.001, sim_duration = 0.0105, sim_maxrate = 1000):
        super(SpikeGenerator,self).__init__()                

        # The actual number of spikes representing a single pixel of data during the whole time of simulation.
        # It is also the number of steps of a simulation.
        self.simulation_steps = int(np.ceil(sim_duration / sim_dt))
        
        # limiting the number of spikes in the whole session of simulation,
        # given the max_rate and delta_time. By providing a probability scale factor.
        self.probability_scale_factor = 1.0 / (sim_dt * sim_maxrate)

        # init the dataloader
        self.dataloader = DataLoader(dataset,batch_size=sim_batch_size,shuffle=False,drop_last=True,collate_fn=self._spykize)

    def _spykize(self,batch_of_samples):
        with torch.no_grad():
            inputs = []
            labels = []
            for k,v in batch_of_samples:
                inputs.append(k)
                labels.append(v)                              
            batch = torch.stack(inputs,dim=0).type(torch.float32)
            batch_prob = batch[...,np.newaxis]
            shape_of_spykes = tuple([*batch_prob.shape[:-1],self.simulation_steps])
            rnd = torch.rand(shape_of_spykes) * torch.as_tensor(self.probability_scale_factor,dtype=torch.float32)
            new_batch = rnd <= batch_prob            
        return batch, new_batch.type(torch.float32),torch.stack(labels,dim=0)
    def __iter__(self):
        for batch in self.dataloader:
            yield batch

    def __len__(self):
        return len(self.dataloader.dataset)