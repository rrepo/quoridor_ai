import numpy as np
from game.board import Board, Player
from game.movegen import legal_moves
from game.pathfinding import shortest_path

def test_initial_setup():
    print("--- Testing Initial Setup ---")
    board = Board()
    assert board.players[0].pos == 4   # P0: (4, 0)
    assert board.players[1].pos == 76  # P1: (4, 8)
    assert board.players[0].walls == 10
    assert len(legal_moves(board)) > 0
    print("Success: Initial setup is correct.")

def test_pawn_movement():
    print("\n--- Testing Pawn Movement ---")
    board = Board()
    # P0 moves down: 4 -> 13
    board.make_move(("move", 13))
    assert board.players[0].pos == 13
    assert board.turn == 1
    
    # Undo move
    board.undo_move()
    assert board.players[0].pos == 4
    assert board.turn == 0
    print("Success: Basic move and undo work.")

def test_jump_move():
    print("\n--- Testing Jump Move ---")
    board = Board()
    # プレイヤーを隣接させる (P0をP1のすぐ上に配置)
    board.players[0].pos = 67 # (4, 7)
    board.players[1].pos = 76 # (4, 8)
    
    moves = [m[1] for m in legal_moves(board) if m[0] == "move"]
    # P1(76)を飛び越えて、盤外でない場合はジャンプが発生するはず（この場合、左右に避けるか手前に戻るなど）
    # ※コリドールのルール：後ろが壁なら横に飛ぶ。後ろが空なら後ろへ。
    print(f"P0 at 67, P1 at 76. Possible moves for P0: {moves}")
    assert len(moves) > 0
    print("Success: Jump moves calculated.")

def test_wall_blocking():
    print("\n--- Testing Wall Blocking ---")
    board = Board()
    # P0 (pos: 4) の目の前に水平壁を置いて移動を制限する
    # apply_hwall(x, y) は (x, y) と (x+1, y) の間の上下を塞ぐ
    # 4(4,0) と 13(4,1) の間を塞ぐには、y=0 の位置に壁を置く
    board.make_move(("hwall", 4, 0)) # (4,0)-(5,0)の間に水平壁
    
    moves_after_wall = [m[1] for m in legal_moves(board) if m[0] == "move"]
    # 4から下(13)へ行けなくなっているはず
    assert 13 not in moves_after_wall
    print(f"Success: Wall correctly blocks movement. Moves from 4: {moves_after_wall}")

def test_pathfinding_and_illegal_wall():
    print("\n--- Testing Pathfinding & Illegal Wall (Enclosure) ---")
    board = Board()
    # P1のゴールを完全に塞ぐような壁の配置を試みる (movegen内のロジック確認)
    # 実際には wall_moves が shortest_path を使ってチェックしている
    
    # 非常に狭い道を作る
    board.h_walls.add((0, 0))
    board.apply_hwall(0, 0)
    
    path_len = shortest_path(board, 0)
    print(f"Shortest path length for P0: {path_len}")
    assert path_len is not None
    print("Success: Pathfinding works.")

def test_zobrist_consistency():
    print("\n--- Testing Zobrist Consistency ---")
    board = Board()
    initial_hash = board.zobrist
    
    # 適当な手順
    move1 = ("move", 13)
    move2 = ("hwall", 2, 2)
    
    board.make_move(move1)
    board.make_move(move2)
    assert board.zobrist != initial_hash
    
    board.undo_move()
    board.undo_move()
    
    assert board.zobrist == initial_hash
    assert len(board.zobrist_counter) <= 1 # 初期状態のみ
    print("Success: Zobrist hash is consistent after undo.")

def test_tensor_shape():
    print("\n--- Testing NN Tensor Conversion ---")
    board = Board()
    tensor = board.to_tensor()
    assert tensor.shape == (4, 9, 9)
    assert tensor[0, 0, 4] == 1.0 # P0 position
    assert tensor[1, 8, 4] == 1.0 # P1 position
    print("Success: Tensor shape and mapping are correct.")

if __name__ == "__main__":
    try:
        test_initial_setup()
        test_pawn_movement()
        test_jump_move()
        test_wall_blocking()
        test_pathfinding_and_illegal_wall()
        test_zobrist_consistency()
        test_tensor_shape()
        print("\nALL TESTS PASSED!")
    except AssertionError as e:
        print(f"\nTEST FAILED!")
        raise e