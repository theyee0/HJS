import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
from torch import nn
import os

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def fen_to_tensor(fen):
    labels = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    }

    tensor = torch.zeros(13, 8, 8)
    tokens = fen.split()
    rows = tokens[0].split('/')

    if tokens[1] == "w":
        tensor[12] = torch.ones(8,8)
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


class ChessPosData(Dataset):
    def __init__(self, pos_list, scores_list):
        self.pos_list = pos_list
        self.scores_list = scores_list

    def __len__(self):
        return len(self.pos_list)
    
    def __getitem__(self, index):
        return self.pos_list[index], self.scores_list[index]


#gets the device we will be training our model on

#defining relevant variables 
learning_rate = 0.001

hidden_size = 3

#defining the neural network
# Convolutional Neural Network

class NeuralNetwork(nn.Module):

   #module
   # convolutional layers - CL
   # CL's - building blocks of the neural network
   # consists of: filters, strides, padding, feature map
    def __init__(self):
        #will output a score of 1 
        #number of channels of the input will be 12 (12 chess pieces)
        super().__init__()

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        #First convolutional layer: 
        self.conv1 = nn.Conv2d(in_channels = 13, out_channels = 26, 
                               kernel_size= 3, stride =1, padding = 1)
        #padding is zero because we don't care about information in the edges of images
        #2nd convolutional layer:
        self.conv2 = nn.Conv2d(in_channels = 26, out_channels = 26, kernel_size = 3, stride = 1, padding = 1)
        #3rd convolutional layer:
        self.conv3 = nn.Conv2d(in_channels = 26, out_channels= 26, kernel_size = 3, padding = 1)
    
        #flatten
        self.flatten = nn.Flatten(start_dim = 1)

        #fully connected layers 
        self.fc1 = nn.Linear(64 * 26, 64 * 26)

        self.unflatten = nn.Unflatten(1, (26, 8, 8))

        self.fc2 = nn.Linear(64*26, 1)

        #activation functions 
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

        #the batches
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
    
    def forward(self, x):
        # x: torch.Tensor - input tensor
        x = self.dropout1(x)

        x = self.conv1(x)
        x = self.flatten(x)
        x = self.dropout2(x)
        x = self.fc1(x)
        x = self.activation1(x)

        x = self.unflatten(x)

        x = self.conv2(x)
        x = self.dropout3(x)
        x = self.flatten(x)
        x = self.fc2(x)
        x = self.activation2(x)

        x = self.flatten(x)
        return x
    #returns output tensor
    #defines the computation 
    #takes input data -> processes through the netwrok's layers => returns output
    # outputs the activations / predictions of the model


def train():
    print(device)

    #data = pd.read_csv("positions.csv", usecols=['fen', 'score'])

    script_directory = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_directory, 'positions.csv')

    # Use the new path variable
    data = pd.read_csv(csv_path, usecols=['fen', 'score'])

    pos_list = []
    scores_list = []

    num_fen = len(data['fen'])

    for i in range(num_fen): 
        print(f"\rLoading fen string {i + 1} of {num_fen} ({(i + 1.0) / num_fen})", end='')
        tensor = fen_to_tensor(data['fen'][i])
        pos_list.append(tensor)

        if math.isnan(data['score'][i]):
            if tensor[12][0][0] == 1:
                scores_list.append(10000)
            else:
                scores_list.append(-10000)
        else:
            scores_list.append(data['score'][i])

    print()
    scores_list = torch.Tensor(scores_list)

    training_data = ChessPosData(pos_list, scores_list)
    train_loader = DataLoader(training_data, batch_size=64, shuffle=True)

    model = NeuralNetwork()
    #Define Loss & Optimizer 
    #Set Loss function with criterion 
    criterion = nn.MSELoss()

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)

    num_epochs = 20

    total_step = len(train_loader)
    #training the network
    #epochs determines how many iterations to train the network on 
    for epoch in range(num_epochs):
        print(f"Training epoch {epoch + 1}")

        for i, (pos_tensor, score_tensor) in enumerate(train_loader):
            print(f"\rCompleted {i + 1} of {total_step} in current epoch ({epoch + 1})", end='')
            pos_tensor = pos_tensor.to(device)
            score_tensor = score_tensor.to(device)

            outputs = model(pos_tensor)

            loss = criterion(outputs, score_tensor.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print()
        print()
        #for loop iterates through the number of epochs
        #the number of epochs determines the number of iterations
        #the inner for loop goes through the images and labels within the data loader
        #we make forward passes and then calculate the loss 
        #we do a backwatd pass 
        #last set gradients to zero every update 

    torch.save(model.state_dict(), "model.pt")

def predict(board):
    tensor = fen_to_tensor(board).unsqueeze(dim = 0)

    print(tensor)

predict("4r2r/1R1bkpp1/8/p1R5/P2N4/2P1B1Pp/5P1P/6K1 b - - 0 31")
