import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#gets the device we will be training our model on
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")



#defining relevant variables 
batch_size = 64
learning_rate = 0.001
num_epochs = 20
num_classes = 1
train_loader = 100 #arbitrarily defined at the moment
# train_loader will be replaced with the data loader
hidden_size = 3

#defining the neural network
# Convolutional Neural Network

class NeuralNetwork(nn.Module):


   #module
   # convolutional layers - CL
   # CL's - building blocks of the neural network
   # consists of: filters, strides, padding, feature map
    def __init__(self, in_channels = 12, num_classes = 1 ):
        #will output a score of 1 
        #number of channels of the input will be 12 (12 chess pieces)
        super().__init__()
        #First convolutional layer: 
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 12, 
                               kernel_size= 3, stride =1, padding = 0) 
        #padding is zero because we don't care about information in the edges of images
        #2nd convolutional layer:
        self.conv2 = nn.Conv2d(in_channels = 12, out_channels = 24, kernel_size = 3, stride = 1, padding = 0)
        #3rd convolutional layer:
        self.conv3 = nn.Conv2d(in_channels = 12, out_channels= 36, kernel_size = 3, padding = 0)
    
        #fully connected layers 
        self.fc1 = nn.Linear(36, 288 )

        #activation functions 
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

        #the batches
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
    
    def forward(self, x): 
        # x: torch.Tensor - input tensor


        x_input = torch.clone(x)
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.activation2(x)


        return x
    #returns output tensor
    #defines the computation 
    #takes input data -> processes through the netwrok's layers => returns output
    # outputs the activations / predictions of the model

model = NeuralNetwork(num_classes)
#Define Loss & Optimizer 
#Set Loss function with criterion 
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)
total_step = len(train_loader)
#training the network
#epochs determines how many iterations to train the network on 
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    #for loop iterates through the number of epochs
    #the number of epochs determines the number of iterations
    #the inner for loop goes through the images and labels within the data loader
    #we make forward passes and then calculate the loss 
    #we do a backwatd pass 
    #last set gradients to zero every update 

