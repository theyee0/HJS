import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
from torch import nn

labels = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}

data = pd.read_csv("positions.csv", usecols=['fen', 'score'])

def fen_to_tensor(fen):
    tensor = torch.zeros(13, 8, 8)
    tokens = fen.split()
    rows = tokens[0].split('/')

    if tokens[1] == "w":
        tensor[12] = 1
    for i, row in enumerate(rows):
        col = 0
        for j in range(len(row)):
            if row[j] >= '1' and row[j] <= '8':
                col += int(row[j])
            else:
                n = labels[row[j]]
                tensor[n][i][col] = 1
                col += 1
    return tensor

pos_list = []
scores_list = []

for i in range(len(data['fen'])): 
    tensor = fen_to_tensor(data['fen'][i])
    pos_list.append(tensor)

    if math.isnan(data['score'][i]):
        if tensor[12][0][0] == 1:
            scores_list.append(10000)
        else:
            scores_list.append(-10000)
    else:
        scores_list.append(data['score'][i])


class ChessPosData(Dataset):
    def __init__(self, pos_list, scores_list):
        self.pos_list = pos_list
        self.scores_list = scores_list

    def __len__(self):
        return len(self.pos_list)
    
    def __getitem__(self, index):
        return self.pos_list[index], torch.tensor(self.scores_list[index]).float()

training_data = ChessPosData(pos_list, scores_list)
train_loader = DataLoader(training_data, batch_size=64, shuffle=True)

#gets the device we will be training our model on
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

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

