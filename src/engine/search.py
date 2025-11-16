import chess
import chess.polyglot
import time
import math

class EngineState:
    board: chess.Board

    nn_model = None

    endtime = 0
    rem_time = 0

    prev_eval = 0


    def __init__(self, board, model, predict):
        self.board = board
        self.endtime = 0
        self.nn_model = model
        self.predict = predict


    def value(self, piece):
        values = [0, 100, 300, 300, 500, 900, 0, float("inf")]

        if piece is None:
            return 0
        else:
            return values[piece.piece_type]


    def load_board(self, board):
        self.board = board


    def move_key(self, m):
        """Returns a tuple (hash_move, killer, history)"""
        zobrist = chess.polyglot.zobrist_hash(self.board) % self.hash_size

        ht = self.wht if self.board.turn == chess.WHITE else self.bht

        hash_move = ht[zobrist] and ht[zobrist]["board"] == self.board.board_fen() and ht[zobrist]["move"] == m
        killer = 0
        history = 0

        return (hash_move, killer, history)


    def judge_time(self, rem_time):
        return 0 * (rem_time + 0.1)

    def get_move_time(self, max_time):
        self.endtime = time.time_ns() + int(max_time * 1e9)

        best_score = -float("inf")
        best_move = chess.Move.null()

        for move in self.board.legal_moves:
            self.board.push(move)

            score = -self.evaluate()

            self.board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    
    def evaluate(self):
        prediction = self.predict(self.board, self.nn_model)
        return prediction
