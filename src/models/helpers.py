"""
Author: Kourosh T. Baghaei
April 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import numpy as np

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            tmp = torch.zeros((target.shape[0],10),dtype=torch.float32)
            for i in range(target.shape[0]):
                tmp[i,target[i]] = torch.as_tensor(1,dtype=torch.float32)
            target = tmp.to(device)
        
        optimizer.zero_grad()
        output = model(data)
  
        loss = F.mse_loss(output, target,reduction='sum') * 0.5 * (1 / output.shape[0]) # The Torch's mse_loss() does not have division by 2. Though the matlab implementation has it.
        loss.backward()
        optimizer.step()
                
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))

def test_ann(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target_onehot = torch.zeros((target.shape[0],10),dtype=torch.float32)
            for i in range(target.shape[0]):
                target_onehot[i,target[i]] = torch.as_tensor(1,dtype=torch.float32)
                #target = tmp
            target_onehot = target_onehot.to(device)                
            output = model(data)

            test_loss += F.mse_loss(output, target_onehot).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def test_snn(model, device, test_spyke_gen):
    model.eval()
    test_loss = 0
    correct = np.zeros(test_spyke_gen.simulation_steps)
    with torch.no_grad():
        for dat_orig, data, target in test_spyke_gen:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output            
            # Sum up the correct predictions in each of the simulation steps
            correct += pred.eq(target).sum(1).cpu().numpy()
    test_loss /= len(test_spyke_gen)

    for i in range(test_spyke_gen.simulation_steps):
        print('\nSim step: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            i, test_loss, correct[i], len(test_spyke_gen),
            100. * correct[i] / len(test_spyke_gen)))


def train_cnn(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        target_onehot = torch.zeros((target.shape[0],10),dtype=torch.float32)
        for i in range(target.shape[0]):
            target_onehot[i,target[i]] = torch.as_tensor(1,dtype=torch.float32)            
        target_onehot = target_onehot.to(device)      

        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target_onehot,reduction='sum') * 0.5 * (1 / output.shape[0])
        loss.backward()
        optimizer.step()    
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))        

def test_cnn(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            target_onehot = torch.zeros((target.shape[0],10),dtype=torch.float32)
            for i in range(target.shape[0]):
                target_onehot[i,target[i]] = torch.as_tensor(1,dtype=torch.float32)            
            target_onehot = target_onehot.to(device)

            output = model(data)
            test_loss += F.mse_loss(output, target_onehot, reduction='sum').item() * 0.5 * (1 / output.shape[0])  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def test_cnn_spiking(model, device, test_spyke_gen, batch_size):
    model.eval()
    test_loss = 0
    batch_index = 0
    correct = np.zeros(test_spyke_gen.simulation_steps)
    with torch.no_grad():
        for dat_orig, data, target in test_spyke_gen:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output                                 
            # Sum up the correct predictions in each of the simulation steps
            correct += pred.eq(target).sum(1).cpu().numpy()
            print('Progress: {}/{}'.format(batch_index * batch_size,len(test_spyke_gen)))
            print('worst corrects so far: {}/{}'.format(correct[0],batch_size * (batch_index + 1)))
            print('best corrects so far: {}/{}'.format(correct[-1],batch_size* (batch_index + 1)))
            
            batch_index += 1
    test_loss /= len(test_spyke_gen)
    for i in range(test_spyke_gen.simulation_steps):
        print('\nSim step: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            i, test_loss, correct[i], len(test_spyke_gen),
            100. * correct[i] / len(test_spyke_gen)))

def prep_dataset():
    BATCH_SIZE = 64

    train_set = datasets.MNIST('./data', train=True, download=True)
    test_set = datasets.MNIST('./data', train=False, download=True)

    train_set_x = train_set.data.numpy() / 255.0
    test_set_x = test_set.data.numpy() / 255.0

    train_set_y = train_set.targets.numpy()
    test_set_y = test_set.targets.numpy()

    return train_set_x, train_set_y, test_set_x, test_set_y