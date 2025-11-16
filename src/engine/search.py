import chess
import chess.polyglot
import time
import math

class EngineState:
    board: chess.Board
    hash_size = 65536
    wht = [None] * hash_size
    bht = [None] * hash_size

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
        print(f"Info: Total time: {rem_time}")

        s = self.prev_eval if self.board.turn == chess.WHITE else -self.prev_eval
        k = 0.009
        shift = -100
        max_frac = 0.1

        estimate = max_frac * (rem_time / 1000) / (1 + math.exp(-k * (shift - s)))

        print(f"Info: Time target: {estimate}")
        return estimate


    def hash_add(self, m, score, depth, node_type):
        zobrist = chess.polyglot.zobrist_hash(self.board) % self.hash_size

        ht = self.wht if self.board.turn == chess.WHITE else self.bht

        ht[zobrist] = {"board": self.board.board_fen(), "move": m, "score": score, "depth": depth, "type": node_type}


    def get_move_time(self, max_time):
        moves = list(self.board.legal_moves)

        if len(moves) == 0:
            return chess.Move.null()

        initial_depth = 2

        ret_score = None
        ret_move = moves[0]

        depth = initial_depth

        self.endtime = time.time_ns() + int(max_time * 1e9)

        ht = self.wht if chess.WHITE else self.bht

        try:
            while depth <= initial_depth or time.time_ns() < self.endtime:
                best_score = -float("inf")
                best_move = None

                alpha = -float("inf")

                moves.sort(key = lambda x: self.move_key(x), reverse = True)

                for move in moves:
                    self.board.push(move)

                    score = -self.pvs(-(alpha + 1), -alpha, depth - 1)

                    if alpha < score:
                        score = -self.pvs(-float("inf"), -alpha, depth - 1)

                    self.board.pop()

                    if score > best_score:
                        best_score = score
                        best_move = move

                    alpha = max(score, alpha)


                    if depth > initial_depth and time.time_ns() >= self.endtime:
                        break

                if depth > initial_depth and time.time_ns() >= self.endtime:
                    break

                print(f"Depth: {depth} - Score: {best_score}")

                ret_score = best_score
                ret_move = best_move
                print(best_move)
                depth += 1

            self.hash_add(ret_move, ret_score, depth, "pv")

            self.prev_eval = ret_score if self.board.turn == chess.WHITE else -ret_score

            print(ret_move)
        except Exception as e:
            print(f"Exception: {e}")

        return ret_move


    def pvs(self, alpha, beta, depth):
        try:

            if depth == 0:
                return self.quiesce(alpha, beta, 10)

            best_score = -float("inf")
            best_move = None

            ht = self.wht if self.board.turn == chess.WHITE else self.bht

            moves = sorted(list(self.board.legal_moves), key = lambda x: self.move_key(x), reverse = True)

            if len(moves) == 0:
                return -float("inf") if self.board.is_check() else 0

            for move in moves:
                self.board.push(move)

                score = -self.pvs(-(alpha + 1), -alpha, depth - 1)

                if alpha < score and score < beta:
                    score = -self.pvs(-beta, -alpha, depth - 1)

                self.board.pop()

                if score > best_score:
                    best_score = score
                    best_move = move

                if score >= beta:
                    self.hash_add(move, score, depth, "cut")
                    return score

                alpha = max(score, alpha)

            self.hash_add(best_move, best_score, depth, "pv")
        except Exception as e:
            print(f"Exception: {e}")

        return best_score


    def quiesce(self, alpha, beta, depth):
        try:
            MAX_DELTA = 300

            stand_pat = self.evaluate()

            if depth == 0:
                return stand_pat

            if stand_pat + MAX_DELTA < alpha or stand_pat > beta:
                return stand_pat

            alpha = max(stand_pat, alpha)

            best_score = stand_pat
            best_move = None

            ht = self.wht if self.board.turn == chess.WHITE else self.bht

            moves = sorted(filter(lambda x: self.board.is_capture(x), self.board.legal_moves), key = lambda x: self.move_key(x), reverse = True)

            for move in moves:
                self.board.push(move)

                score = -self.quiesce(-(alpha + 1), -alpha, depth -1)

                if alpha < score and score < beta:
                    score = -self.quiesce(-beta, -alpha, depth - 1)

                self.board.pop()

                if (score > best_score):
                    best_score = score
                    best_move = move

                if score >= beta:
                    self.hash_add(move, score, depth, "cut")
                    return score

                alpha = max(score, alpha)

            self.hash_add(best_move, best_score, depth, "pv")
        except Exception as e:
            print(f"Exception: {e}")
            
        return best_score


    def evaluate(self):
        prediction = self.predict(self.board, self.nn_model)

        return prediction if board.turn == chess.WHITE else -prediction
