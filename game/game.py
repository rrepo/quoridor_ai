#game.py
from game.board import Board
from game.movegen import legal_moves
from ai.search import best_move, clear_tt

class QuoridorGame:
    def __init__(self, zobrist_seed=None):
        clear_tt()
        self.board = Board()
        # allow deterministic zobrist if desired
        if zobrist_seed is not None:
            self.board.init_zobrist(seed=zobrist_seed)

    def __str__(self):
        lines = []
        for row in range(9):
            row_str = ""
            for col in range(9):
                node = row * 9 + col
                if node == self.board.players[0].pos:
                    row_str += " 0 "
                elif node == self.board.players[1].pos:
                    row_str += " 1 "
                else:
                    row_str += " . "
                if col < 8:
                    row_str += "|" if (col, row) in self.board.v_walls or (col, row - 1) in self.board.v_walls else " "
            lines.append(row_str)
            if row < 8:
                sep = ""
                for col in range(9):
                    sep += "---" if (col, row) in self.board.h_walls or (col - 1, row) in self.board.h_walls else "   "
                    if col < 8:
                        sep += " "
                lines.append(sep)
        p0 = self.board.players[0]
        p1 = self.board.players[1]
        lines.append(f"Turn: Player {self.board.turn}  |  P0 walls: {p0.walls}  P1 walls: {p1.walls}")
        return "\n".join(lines)

    def ai_move(self, depth=3):
        move = best_move(self.board, depth)
        if move:
            self.board.make_move(move)
        return move