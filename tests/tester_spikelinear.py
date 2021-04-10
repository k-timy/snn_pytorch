"""
Author: Kourosh T. Baghaei

TestSpikeLinear: a test class for SpikingLinear layer.

April 2021
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.spikinglinear import SpikingLinear

class TestSpikeLinear(unittest.TestCase):
    
    def setUp(self):
        self.expected_results = {
            'cool_down' : torch.Tensor([
                # ts = 0
                [
                    [0., 0.],
                    [0., 0.],
                    [0., 0.]
                ],
                # ts = 1
                [
                    [0.0000, 0.0000],
                    [0.0000, 0.0000],
                    [0.0000, 0.2500]
                ],
                # ts = 2
                [
                    [0.0000, 0.2500],
                    [0.0000, 0.2500],
                    [0.2500, 0.1500]
                ],
                # ts = 3
                [
                    [0.2500, 0.1500],
                    [0.2500, 0.1500],
                    [0.1500, 0.0500]
                ],
                # ts = 4
                [
                    [0.1500, 0.0500],
                    [0.1500, 0.0500],
                    [0.0500, 0.0000]
                ],
                # ts = 5
                [
                    [0.0500, 0.0000],
                    [0.0500, 0.0000],
                    [0.0000, 0.2500]
                ],
                # ts = 6
                [
                    [0.0000, 0.0000],
                    [0.0000, 0.0000],
                    [0.0000, 0.1500]
                ],
                # ts = 7
                [
                    [0.0000, 0.2500],
                    [0.0000, 0.2500],
                    [0.2500, 0.0500]
                ],
                # ts = 8
                [
                    [0.0000, 0.1500],
                    [0.0000, 0.1500],
                    [0.1500, 0.0000]
                ],
                # ts = 9
                [
                    [0.2500, 0.0500],
                    [0.2500, 0.0500],
                    [0.0500, 0.2500]
                ],
            ]),
            'membrane_potential' : torch.Tensor([
                # ts = 0
                [
                    [0.4000, 0.6000],
                    [0.4000, 0.8000],
                    [0.8000, 0.0000]
                ],
                # ts = 1
                [
                    [0.8000, 0.0000],
                    [0.8000, 0.0000],
                    [0.0000, 0.0000]
                ],
                # ts = 2
                [
                    [0., 0.],
                    [0., 0.],
                    [0., 0.]
                ],
                # ts = 3
                [
                    [0., 0.],
                    [0., 0.],
                    [0., 0.]
                ],
                # ts = 4
                [
                    [0., 0.],
                    [0., 0.],
                    [0., 0.]
                ],
                # ts = 5
                [
                    [0.0000, 0.6000],
                    [0.0000, 0.8000],
                    [0.8000, 0.0000]
                ],
                # ts = 6
                [
                    [0.4000, 0.0000],
                    [0.4000, 0.0000],
                    [0.0000, 0.0000]
                ],
                # ts = 7
                [
                    [0.8000, 0.0000],
                    [0.8000, 0.0000],
                    [0.0000, 0.0000]
                ],
                # ts = 8
                [
                    [0., 0.],
                    [0., 0.],
                    [0., 0.]
                ],
                # ts = 9
                [
                    [0., 0.],
                    [0., 0.],
                    [0., 0.]
                ]
            ]),
            'output_spikes' : torch.Tensor([
                # ts = 0
                [
                    [0., 0.],
                    [0., 0.],
                    [0., 1.]
                ],
                # ts = 1
                [
                    [0., 1.],
                    [0., 1.],
                    [1., 0.]
                ],
                # ts = 2
                [
                    [1., 0.],
                    [1., 0.],
                    [0., 0.]
                ],
                # ts = 3
                [
                    [0., 0.],
                    [0., 0.],
                    [0., 0.]
                ],
                # ts = 4
                [
                    [0., 0.],
                    [0., 0.],
                    [0., 1.]
                ],
                # ts = 5
                [
                    [0., 0.],
                    [0., 0.],
                    [0., 0.]
                ],
                # ts = 6
                [
                    [0., 1.],
                    [0., 1.],
                    [1., 0.]
                ],
                # ts = 7
                [
                    [0., 0.],
                    [0., 0.],
                    [0., 0.]
                ],
                # ts = 8
                [
                    [1., 0.],
                    [1., 0.],
                    [0., 1.]
                ],
                # ts = 9
                [
                    [0., 0.],
                    [0., 0.],
                    [0., 0.]
                ]
            ])
        }   

    def test_spiking_linear(self):
        # create a Linear layer
        fc1 = nn.Linear(3, 2,bias=False)
        # set weights
        fc1.weight = nn.Parameter(torch.as_tensor(np.linspace(0,1,6).reshape(3,2).transpose(),dtype=torch.float32))
        # convert Linear layer to a Spiking layer
        sl1 = SpikingLinear(fc1,dt=0.1,ref_period=0.35,batch_size=3)
        # an input
        a = torch.as_tensor(np.asarray([[0,1,0],[1,1,0],[0,0,1]],dtype=np.float32))

        for timestep in range(10):
            eq_as_num = torch.isclose(sl1.neuron_ref_cooldown, self.expected_results['cool_down'][timestep]).long()          
            self.assertTrue(torch.prod(eq_as_num).item() == 1)
            b = sl1(a)
            eq_as_num = torch.isclose(sl1.membrane_potential,self.expected_results['membrane_potential'][timestep]).long()
            self.assertTrue(torch.prod(eq_as_num).item() == 1)
            eq_as_num = torch.isclose(b, self.expected_results['output_spikes'][timestep]).long()
            self.assertTrue(torch.prod(eq_as_num).item() == 1)