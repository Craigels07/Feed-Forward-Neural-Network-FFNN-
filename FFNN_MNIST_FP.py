# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 08:21:41 2022

@author: craig
"""

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform))

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transform))

class Net(nn.Module):
    def __init__(self, epochs=1):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)

        self.epochs = epochs

    def forward_pass(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = torch.softmax(x, dim=0)
        return x
    
    def one_hot_encode(self, y):
        encoded = torch.zeros([10], dtype=torch.float64)
        encoded[y[0]] = 1.
        return encoded

    def train(self, train_loader, optimizer, criterion):
        start_time = time.time()
        loss = None

        for iteration in range(self.epochs):
            for x,y in train_loader:
                y = self.one_hot_encode(y)
                optimizer.zero_grad()
                output = self.forward_pass(torch.flatten(x))
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

            print('Epoch: {0}, Time Spent: {1:.2f}s, Loss: {2}'.format(
                iteration+1, time.time() - start_time, loss
            ))


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

model.train(train_loader, optimizer, criterion)   
        