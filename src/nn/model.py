import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
import chess
from torch import nn
import os


class ChessPosData(Dataset):
    def __init__(self, pos_list, scores_list):
        self.pos_list = pos_list
        self.scores_list = scores_list

    def __len__(self):
        return len(self.pos_list)
    
    def __getitem__(self, index):
        return self.pos_list[index], self.scores_list[index]


# Convolutional Neural Network
class NeuralNetwork(nn.Module):
   # Module
   # convolutional layers - CL
   # CL's - building blocks of the neural network
   # consists of: filters, strides, padding, feature map
    def __init__(self):
        #will output a score of 1 
        #number of channels of the input will be 12 (12 chess pieces)
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Flatten(start_dim = 1),
            nn.Linear(64 * 13, 64 * 16),
            nn.SELU(),
            nn.Linear(64 * 16, 64 * 8),
            nn.SELU(),
            nn.Linear(64 * 8, 64),
            nn.SELU(),
            nn.Linear(64, 16),
            nn.SELU(),
            nn.Linear(16, 1))

    def forward(self, x):
        # x: torch.Tensor - input tensor
        x = self.layer_stack(x)

        return x


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


def load_tensors_from_fen(data):
    # Store the list of Tensors for position/score
    position_list = []
    scores_list = []

    num_fen = len(data['fen'])

    for i in range(num_fen): 
        print(f"\rLoading fen string {i + 1} of {num_fen} ({(i + 1.0) / num_fen})", end='')
        tensor = fen_to_tensor(data['fen'][i])
        position_list.append(tensor)

        if math.isnan(data['score'][i]):
            if tensor[12][0][0] == 1:
                scores_list.append(10000)
            else:
                scores_list.append(-10000)
        else:
            scores_list.append(data['score'][i])
    print()

    return (
        torch.stack(position_list),
        torch.Tensor(scores_list))


def train():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    model = NeuralNetwork()
    model.to(device)

    # Read fen/score data from CSV file
    csv_path = "positions.csv"
    data = pd.read_csv(csv_path, usecols=['fen', 'score'])

    position_list, scores_list = load_tensors_from_fen(data)

    training_data = ChessPosData(position_list, scores_list)
    train_loader = DataLoader(training_data, batch_size=128, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    num_epochs = 20
    total_steps = len(train_loader)

    #training the network
    #epochs determines how many iterations to train the network on 
    for epoch in range(num_epochs):
        print(f"Training epoch {epoch + 1}")

        for i, (pos_tensor, score_tensor) in enumerate(train_loader):
            print(f"\rCompleted {i + 1} of {total_steps} in current epoch ({epoch + 1})", end='')
            pos_tensor = pos_tensor.to(device)
            score_tensor = score_tensor.to(device)

            outputs = model(pos_tensor)

            loss = criterion(outputs, score_tensor.unsqueeze(1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print()
        print()
        #for loop iterates through the number of epochs
        #the number of epochs determines the number of iterations
        #the inner for loop goes through the images and labels within the data loader
        #we make forward passes and then calculate the loss 
        #we do a backwatd pass 
        #last set gradients to zero every update 

    torch.save(model.state_dict(), "model.pt")


def load_model(filename):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    model = NeuralNetwork()
    model.to(device)

    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    return model


def predict(board, model):
    with torch.no_grad():
        tensor = torch.zeros(13, 8, 8)
        if (board.turn == chess.WHITE):
            tensor[12] = torch.ones(8, 8)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            code = piece.piece_type - 1
            if piece.color == chess.BLACK:
                code += 6

            tensor[code][square // 8][square % 8] = 1

        tensor = tensor.unsqueeze(dim = 0)
        return model(tensor)[0].item()
