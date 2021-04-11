"""
Author: Kourosh T. Baghaei
April 2021

"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.spikinglinear import SpikingLinear
from src.dataset.simpledataset import SimpleDataset
from src.models.helpers import prep_dataset
from src.models.ann import ANN
from src.models.helpers import test_ann, test_snn, train
from src.dataset.spikegenerator import SpikeGenerator
from src.models.spikenet import SpikeNet

def main():
    # prep torch and device
    seed = 0
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # prep datasets
    train_set_x,train_set_y, test_set_x,test_set_y = prep_dataset()

    train_ds = SimpleDataset(train_set_x,train_set_y,(784,))
    test_ds = SimpleDataset(test_set_x,test_set_y,(784,))

    batch_size = 100
    epochs = 5
    learning_rate = 0.1
    momentum = 0.5

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,drop_last=True)

    # create and train an ANN
    print('Training a regular fully connected network called: "ANN"')

    model = ANN().to(device)
    global ann_model
    ann_model = model
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test_ann(model, device, test_loader)

    # create and train an SNN, derived from the ANN above
    print('Create a SNN version of ANN')
    test_spyke_gen = SpikeGenerator(test_ds,sim_batch_size=batch_size,sim_dt=0.001,sim_duration=0.050,sim_maxrate=500)
    snn_model = SpikeNet(ann_model, device,dt=0.001,ref_period=0.0,batch_size=batch_size,threshold=1,all_sim_steps=True)

    print('Test the SNN:')
    for epoch in range(1, 2):
        test_snn(snn_model, device, test_spyke_gen)
    

if __name__ == '__main__':
    main()
