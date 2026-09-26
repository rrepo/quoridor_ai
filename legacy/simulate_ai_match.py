# simulate_ai_match.py（マルチプロセス対応版）

from game.board import Board
from game.search import is_terminal
from ai.search import best_move, clear_tt
from ai.cache import clear_dist_cache
from ai.evaluation import evaluate as simple_eval
from ai.eval_func import evaluate as improved_eval

import multiprocessing
import os

MAX_TURNS = 200


class AIPlayer:
    def __init__(self, name: str = "AI", depth: int = 3, eval_func=None):
        self.name = name
        self.depth = depth
        self.eval_func = eval_func

    def select_move(self, board):
        return best_move(board, self.depth)


def play_match(
    player0: AIPlayer,
    player1: AIPlayer,
    verbose: bool = True,
) -> int:
    board = Board()
    clear_tt()

    for turn_count in range(1, MAX_TURNS + 1):
        clear_dist_cache()
        current_player = player0 if board.turn == 0 else player1
        move = current_player.select_move(board)

        if move is None:
            if verbose:
                print(f"Turn {turn_count}: {current_player.name} (P{board.turn}) に合法手なし → 負け")
            return 1 - board.turn

        if verbose:
            print(f"Turn {turn_count:>3}: {current_player.name} (P{board.turn}) → {move}")

        board.make_move(move)
        terminal, winner = is_terminal(board)
        if terminal:
            if verbose:
                winner_name = player0.name if winner == 0 else player1.name
                print(f"\n=== ゲーム終了 勝者: {winner_name} (Player {winner}) ===")
            return winner

    if verbose:
        print(f"最大ターン数 ({MAX_TURNS}) 到達 → 引き分け")
    return -1


# ワーカー関数：プロセスに渡せるようトップレベルで定義する必要がある
def _play_one_game(args):
    """multiprocessing.Pool のワーカー関数（picklable にするためトップレベル定義）"""
    game_index, p0_name, p0_depth, p0_eval_name, p1_name, p1_depth, p1_eval_name, alternate = args

    eval_map = {"improved": improved_eval, "simple": simple_eval}
    p0 = AIPlayer(name=p0_name, depth=p0_depth, eval_func=eval_map[p0_eval_name])
    p1 = AIPlayer(name=p1_name, depth=p1_depth, eval_func=eval_map[p1_eval_name])

    if alternate and game_index % 2 == 1:
        result = play_match(p1, p0, verbose=False)
        # 先後を入れ替えた分を元に戻す
        if result == 0:
            result = 1
        elif result == 1:
            result = 0
    else:
        result = play_match(p0, p1, verbose=False)

    return game_index, result


def run_series(
    player0: AIPlayer,
    player1: AIPlayer,
    n_games: int = 10,
    alternate: bool = True,
    n_workers: int = None,
) -> dict:
    """
    マルチプロセスで n_games 局を並列実行する。

    Args:
        n_workers: 並列プロセス数。None の場合は CPU コア数を使用。
    """
    if n_workers is None:
        n_workers = os.cpu_count()

    print(f"並列プロセス数: {n_workers} / CPU コア数: {os.cpu_count()}")

    # 評価関数名を文字列で渡す（lambda や関数オブジェクトは pickle 不可）
    def get_eval_name(player):
        return "improved" if player.eval_func is improved_eval else "simple"

    task_args = [
        (
            i,
            player0.name, player0.depth, get_eval_name(player0),
            player1.name, player1.depth, get_eval_name(player1),
            alternate,
        )
        for i in range(n_games)
    ]

    wins = {0: 0, 1: 0, -1: 0}
    results = [None] * n_games

    # imap_unordered で完了したゲームから順次表示
    with multiprocessing.Pool(processes=n_workers) as pool:
        for game_index, result in pool.imap_unordered(_play_one_game, task_args):
            wins[result] += 1
            results[game_index] = result
            done = sum(v for v in wins.values())
            label = "draw" if result == -1 else f"P{result} win"
            print(f"Game {game_index + 1:>3}/{n_games} 完了: {label}  (累計: P0={wins[0]}, P1={wins[1]}, draw={wins[-1]})")

    print(f"\n--- {n_games} games 結果 ---")
    print(f"  {player0.name} (P0) 勝利: {wins[0]} ({wins[0]/n_games*100:.1f}%)")
    print(f"  {player1.name} (P1) 勝利: {wins[1]} ({wins[1]/n_games*100:.1f}%)")
    print(f"  引き分け:              {wins[-1]} ({wins[-1]/n_games*100:.1f}%)")

    return {"wins_p0": wins[0], "wins_p1": wins[1], "draws": wins[-1]}


if __name__ == "__main__":
    import argparse

    # Windows では必須、Linux/Mac でも安全のため
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Quoridor AI match simulator")
    parser.add_argument("--depth",   type=int, default=3,  help="探索深さ (default: 3)")
    parser.add_argument("--games",   type=int, default=1,  help="対局数 (default: 1)")
    parser.add_argument("--workers", type=int, default=None, help="並列プロセス数 (default: CPUコア数)")
    parser.add_argument("--quiet",   action="store_true",  help="手順出力を抑制する")
    parser.add_argument("--p0eval",  choices=["improved", "simple"], default="improved")
    parser.add_argument("--p1eval",  choices=["improved", "simple"], default="simple")
    args = parser.parse_args()

    eval_map = {"improved": improved_eval, "simple": simple_eval}

    ai0 = AIPlayer(name=f"P0({args.p0eval})", depth=args.depth, eval_func=eval_map[args.p0eval])
    ai1 = AIPlayer(name=f"P1({args.p1eval})", depth=args.depth, eval_func=eval_map[args.p1eval])

    if args.games == 1:
        play_match(ai0, ai1, verbose=not args.quiet)
    else:
        run_series(ai0, ai1, n_games=args.games, alternate=False, n_workers=args.workers)