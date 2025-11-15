from .utils import chess_manager, GameContext
from chess import Move
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

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

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.1)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    move_weights = [random.random() for _ in legal_moves]
    total_weight = sum(move_weights)
    # Normalize so probabilities sum to 1
    move_probs = {
        move: weight / total_weight
        for move, weight in zip(legal_moves, move_weights)
    }
    ctx.logProbabilities(move_probs)

    return random.choices(legal_moves, weights=move_weights, k=1)[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
