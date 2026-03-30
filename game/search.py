# search.py
from game.board import Board

def is_terminal(board):
    if board.players[0].pos // 9 == 8:
        return True, 0
    if board.players[1].pos // 9 == 0:
        return True, 1
    return False, -1

