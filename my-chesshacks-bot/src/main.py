from .utils import chess_manager, GameContext
from chess import Move, Board
import random
import time

from .engine.search import EngineState

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

engine = EngineState(Board())

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    global engine

    engine.load_board(ctx.board)

    return engine.get_move_time(engine.judge_time(ctx.timeLeft))


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
