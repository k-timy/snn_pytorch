"""
Author: Kourosh T. Baghaei
April 2021
"""

import torch
from src.dataset.simpledataset import SimpleDataset
from src.dataset.spikegenerator import SpikeGenerator
from src.models.spikeconvnet import SpikeConv
from src.models.convnet import ConvNet
from src.models.helpers import prep_dataset,train_cnn,test_cnn,test_cnn_spiking
import torch.optim as optim

def main():
    
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(0)

    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 256
    epochs = 10

    # prep datasets
    train_set_x,train_set_y, test_set_x,test_set_y = prep_dataset()

    train_ds = SimpleDataset(train_set_x, train_set_y, (1, 28, 28))
    test_ds = SimpleDataset(test_set_x, test_set_y, (1, 28, 28))

    train_loader = torch.utils.data.DataLoader(train_ds,batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds,batch_size=batch_size, shuffle=False)

    cnn_net = ConvNet().to(device)
    optimizer = optim.Adam(cnn_net.parameters())

    for epoch in range(1, epochs + 1):
        train_cnn(cnn_net, device, train_loader, optimizer, epoch)
        test_cnn(cnn_net, device, test_loader)
    
    snn_conv = SpikeConv(conv_net = cnn_net, device=device,all_sim_steps=True).to(device)
    test_loader = SpikeGenerator(test_ds,sim_batch_size=batch_size,sim_dt=0.001, sim_duration=0.100)
    
    for epoch in range(1, 1 + 1):
        test_cnn_spiking(snn_conv, device, test_loader,batch_size)    

if __name__ == '__main__':
    main()