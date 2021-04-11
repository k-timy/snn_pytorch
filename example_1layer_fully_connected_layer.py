import torch
import torch.nn as nn
import numpy as np
from src.models.spikinglinear import SpikingLinear

def main():
    # create a Linear layer
    fc1 = nn.Linear(3, 2,bias=False)
    # set weights
    fc1.weight = nn.Parameter(torch.as_tensor(np.linspace(0,1,6).reshape(3,2).transpose(),dtype=torch.float32))
    # convert Linear layer to a Spiking layer
    sl1 = SpikingLinear(fc1,dt=0.1,ref_period=0.35,batch_size=3)
    # an input
    a = torch.as_tensor(np.asarray([[0,1,0],[1,1,0],[0,0,1]],dtype=np.float32))

    print(type(sl1))
    a = torch.as_tensor(np.asarray([[0,1,0],[1,1,0],[0,0,1]],dtype=np.float32))
    print('here is a:')
    print(a)
    print('get weights',sl1.weight)
    for i in range(10):
        print()
        print()
        print('####   {}   ####'.format(i))
        print('cool down:')
        print(sl1.neuron_ref_cooldown)
        print('mem pot:')
        print(sl1.membrane_potential)
        print('after feed:')
        b = sl1(a)
        print()   
        print('mem pot:')
        print(sl1.membrane_potential)
        print('out spykes')
        print(b)
        print(type(b))
    print('get weights again',sl1.weight)
    
if __name__ == '__main__':
    main()    
