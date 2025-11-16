import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
import chess
from torch import nn
import os

torch.set_default_dtype(torch.bfloat16)

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

        self.layer_stack_1 = nn.Sequential(
            nn.Conv2d(in_channels = 12, out_channels = 24, kernel_size = 3, stride = 1, padding = 1),
            nn.Conv2d(in_channels = 24, out_channels = 48, kernel_size = 5, stride = 1, padding = 2))

        self.layer_stack_2 = nn.Sequential(
            nn.Flatten(start_dim = 1),
            nn.Linear(64 * 48, 64 * 16),
            nn.SELU(),
            nn.Linear(64 * 16, 64 * 4),
            nn.SELU(),
            nn.Linear(64 * 4, 1))

    def forward(self, x):
        # x: torch.Tensor - input tensor
        x = self.layer_stack_1(x)
        x = self.layer_stack_2(x)

        return x


def board_to_tensor(board):
    tensor = torch.zeros(12, 8, 8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        code = piece.piece_type - 1
        if piece.color == chess.BLACK:
            code += 6

        tensor[code][square // 8][square % 8] = 1

    return tensor


def load_tensors_from_fen(data):
    # Store the list of Tensors for position/score
    position_list = []
    scores_list = []

    num_fen = 16 #len(data['fen'])
    board = chess.Board()

    for i in range(num_fen): 
        print(f"\rLoading fen string {i + 1} of {num_fen} ({(i + 1.0) / num_fen})", end='')

        if math.isnan(data['score'][i]):
            continue

        board.set_fen(data['fen'][i])
                      
        if board.turn == chess.BLACK:
            board.apply_mirror()
            position_list.append(board_to_tensor(board))
            scores_list.append(-data['score'][i])
        else:
            position_list.append(board_to_tensor(board))
            scores_list.append(data['score'][i])

    return (
        torch.stack(position_list),
        torch.Tensor(scores_list))


def train():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    model = NeuralNetwork()
    model.to(device)
    model.tox(torch.bfloat16)

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
    model.to(torch.bfloat16)

    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    return model


def load_model_and_predict(fen):
    model = load_model("model.pt")

    board = chess.Board().set_fen(fen)

    with torch.no_grad():
        tensor = model(board_to_tensor(board)[0].unsqueeze(dim = 0))

        return tensor[0].item()


def predict(board, model):
    with torch.no_grad():
        tensor = board_to_tensor(board)

        tensor = tensor.unsqueeze(dim = 0)
        return model(tensor)[0].item()
