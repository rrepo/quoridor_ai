# board.py

from dataclasses import dataclass, field
import random
from typing import List, Set, Tuple
import numpy as np

BOARD_SIZE = 9
NODE_COUNT = BOARD_SIZE * BOARD_SIZE
MAX_WALLS  = 10

_GOAL_MASK = [
    sum(1 << (72 + c) for c in range(9)),
    sum(1 << c for c in range(9)),
]


@dataclass
class Player:
    pos: int
    walls: int = MAX_WALLS


@dataclass
class Board:
    players: List[Player] = field(default_factory=lambda: [Player(4), Player(76)])
    h_walls: Set[Tuple[int, int]] = field(default_factory=set)
    v_walls: Set[Tuple[int, int]] = field(default_factory=set)

    # ------------------------------------------------------------------
    # edges リストを廃止し adj ビットマスクに統一
    #
    # 旧実装の問題点:
    #   - edges[a].remove(b) が O(n)（リストの線形探索）
    #   - b not in edges[a]  が O(n)（同上）
    #   - adj ビットマスクと edges リストの二重管理で冗長
    #
    # 修正:
    #   - adj のみを正とし、edges フィールドを廃止
    #   - remove_edge / add_edge は adj のビット操作のみ（O(1)）
    #   - BFS・合法手生成など edges を参照していた箇所は
    #     adj のビットをイテレートする _neighbors(node) で代替
    # ------------------------------------------------------------------
    adj: List[int] = field(default_factory=lambda: [0] * NODE_COUNT)

    # dirs は movegen で方向ベース判定に使用。そのまま残す。
    dirs: List[dict] = field(default_factory=lambda: [{} for _ in range(NODE_COUNT)])

    turn: int = 0
    zobrist: int = 0

    z_pos:   List[List[int]] = field(init=False, default_factory=list)
    z_hwall: List[List[int]] = field(init=False, default_factory=list)
    z_vwall: List[List[int]] = field(init=False, default_factory=list)
    z_turn:  int = field(init=False, default=0)

    move_stack: List = field(default_factory=list)

    # ------------------------------------------------------------------
    # zobrist_history を list から dict カウンタに変更
    #
    # 旧実装の問題点:
    #   zobrist_history.count(zobrist) が O(n)
    #   深い探索ほど履歴が長くなり、全ノードで繰り返し呼ばれるため
    #   探索木全体で O(depth × n) のコストになる。
    #
    # 修正:
    #   dict でハッシュ値 → 出現回数を管理。参照・更新ともに O(1)。
    #   count() の代わりに zobrist_counter.get(zobrist, 0) を使う。
    # ------------------------------------------------------------------
    zobrist_counter: dict = field(default_factory=dict)

    def __post_init__(self):
        self.init_edges()
        self.init_zobrist(seed=None)

    # ------------------------------------------------------------------
    # 隣接ノードのイテレータ（adj ビットマスクから生成）
    # edges リストの代替。BFS・合法手生成で使用する。
    # ------------------------------------------------------------------
    @staticmethod
    def _iter_bits(mask: int):
        """ビットマスクからセットされているビット位置を順に yield する。"""
        while mask:
            lsb = mask & (-mask)       # 最下位ビットを取り出す
            yield lsb.bit_length() - 1
            mask ^= lsb

    def neighbors(self, node: int):
        """node の現在の隣接ノード一覧を返すジェネレータ。"""
        yield from self._iter_bits(self.adj[node])

    # ------------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------------
    def init_edges(self):
        BS = BOARD_SIZE
        for node in range(NODE_COUNT):
            x = node % BS
            y = node // BS
            if y > 0:      self._link(node, node - BS)
            if y < BS - 1: self._link(node, node + BS)
            if x > 0:      self._link(node, node - 1)
            if x < BS - 1: self._link(node, node + 1)

        for node in range(NODE_COUNT):
            x = node % BS
            y = node // BS
            self.dirs[node] = {
                'up':    node - BS if y > 0      else -1,
                'down':  node + BS if y < BS - 1 else -1,
                'left':  node - 1  if x > 0      else -1,
                'right': node + 1  if x < BS - 1 else -1,
            }

    def _link(self, a: int, b: int):
        """adj ビットマスクのみで辺を管理する（edges リストは廃止）。"""
        self.adj[a] |= (1 << b)
        self.adj[b] |= (1 << a)

    def init_zobrist(self, seed=None):
        rnd = random.Random(seed)
        self.z_pos   = [[rnd.getrandbits(64) for _ in range(NODE_COUNT)]     for _ in range(2)]
        self.z_hwall = [[rnd.getrandbits(64) for _ in range(BOARD_SIZE - 1)] for _ in range(BOARD_SIZE - 1)]
        self.z_vwall = [[rnd.getrandbits(64) for _ in range(BOARD_SIZE - 1)] for _ in range(BOARD_SIZE - 1)]
        self.z_turn  = rnd.getrandbits(64)
        self.zobrist = 0
        for i in range(2):
            self.zobrist ^= self.z_pos[i][self.players[i].pos]

    # ------------------------------------------------------------------
    # 辺の削除・追加（O(1)）
    # ------------------------------------------------------------------
    def remove_edge(self, a: int, b: int):
        """adj ビットマスクから辺を削除する。O(1)。"""
        self.adj[a] &= ~(1 << b)
        self.adj[b] &= ~(1 << a)

    def add_edge(self, a: int, b: int):
        """adj ビットマスクに辺を追加する。O(1)。"""
        self.adj[a] |= (1 << b)
        self.adj[b] |= (1 << a)

    # ------------------------------------------------------------------
    # 壁の適用・取消
    # ------------------------------------------------------------------
    def apply_hwall(self, x: int, y: int):
        tl = y * BOARD_SIZE + x
        tr = tl + 1
        bl = tl + BOARD_SIZE
        br = bl + 1
        self.remove_edge(tl, bl)
        self.remove_edge(tr, br)

    def undo_hwall(self, x: int, y: int):
        tl = y * BOARD_SIZE + x
        tr = tl + 1
        bl = tl + BOARD_SIZE
        br = bl + 1
        self.add_edge(tl, bl)
        self.add_edge(tr, br)

    def apply_vwall(self, x: int, y: int):
        tl = y * BOARD_SIZE + x
        tr = tl + 1
        bl = tl + BOARD_SIZE
        br = bl + 1
        self.remove_edge(tl, tr)
        self.remove_edge(bl, br)

    def undo_vwall(self, x: int, y: int):
        tl = y * BOARD_SIZE + x
        tr = tl + 1
        bl = tl + BOARD_SIZE
        br = bl + 1
        self.add_edge(tl, tr)
        self.add_edge(bl, br)

    # ------------------------------------------------------------------
    # 指し手の実行・取消
    # ------------------------------------------------------------------
    def make_move(self, move):
        player = self.players[self.turn]

        if move[0] == "move":
            old, new = player.pos, move[1]
            self.move_stack.append(("move", old, new))
            self.zobrist ^= self.z_pos[self.turn][old]
            player.pos = new
            self.zobrist ^= self.z_pos[self.turn][new]

        elif move[0] == "hwall":
            x, y = move[1], move[2]
            self.move_stack.append(("hwall", x, y))
            self.h_walls.add((x, y))
            player.walls -= 1
            self.zobrist ^= self.z_hwall[y][x]
            self.apply_hwall(x, y)

        elif move[0] == "vwall":
            x, y = move[1], move[2]
            self.move_stack.append(("vwall", x, y))
            self.v_walls.add((x, y))
            player.walls -= 1
            self.zobrist ^= self.z_vwall[y][x]
            self.apply_vwall(x, y)

        self.turn ^= 1
        self.zobrist ^= self.z_turn

        # zobrist_counter を更新（O(1)）
        z = self.zobrist
        self.zobrist_counter[z] = self.zobrist_counter.get(z, 0) + 1

    def undo_move(self):
        # zobrist_counter をデクリメント（O(1)）
        z = self.zobrist
        cnt = self.zobrist_counter.get(z, 1) - 1
        if cnt <= 0:
            del self.zobrist_counter[z]
        else:
            self.zobrist_counter[z] = cnt

        self.turn ^= 1
        self.zobrist ^= self.z_turn

        move = self.move_stack.pop()
        player = self.players[self.turn]

        if move[0] == "move":
            old, new = move[1], move[2]
            self.zobrist ^= self.z_pos[self.turn][new]
            player.pos = old
            self.zobrist ^= self.z_pos[self.turn][old]

        elif move[0] == "hwall":
            x, y = move[1], move[2]
            self.h_walls.remove((x, y))
            player.walls += 1
            self.zobrist ^= self.z_hwall[y][x]
            self.undo_hwall(x, y)

        elif move[0] == "vwall":
            x, y = move[1], move[2]
            self.v_walls.remove((x, y))
            player.walls += 1
            self.zobrist ^= self.z_vwall[y][x]
            self.undo_vwall(x, y)

    def make_pass(self):
        self.move_stack.append(("pass",))
        self.turn ^= 1
        self.zobrist ^= self.z_turn
        z = self.zobrist
        self.zobrist_counter[z] = self.zobrist_counter.get(z, 0) + 1

    def undo_pass(self):
        z = self.zobrist
        cnt = self.zobrist_counter.get(z, 1) - 1
        if cnt <= 0:
            del self.zobrist_counter[z]
        else:
            self.zobrist_counter[z] = cnt

        self.move_stack.pop()
        self.turn ^= 1
        self.zobrist ^= self.z_turn
        
    def to_tensor(self):
        """
        盤面状態を NN 用のテンソル (C, H, W) 形式で返す。
        C: チャンネル (自分、相手、横壁、縦壁)
        H, W: 9x9
        """
        # 4枚のプレーンを用意 (9x9)
        # 0: 自分の駒, 1: 相手の駒, 2: 横壁, 3: 縦壁
        tensor = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        # 駒の位置
        my_p = self.players[self.turn]
        opp_p = self.players[1 - self.turn]
        tensor[0, my_p.pos // 9, my_p.pos % 9] = 1.0
        tensor[1, opp_p.pos // 9, opp_p.pos % 9] = 1.0

        # 壁の位置 (x, y) はその隙間の左上を示す
        for (x, y) in self.h_walls:
            tensor[2, y, x] = 1.0
        for (x, y) in self.v_walls:
            tensor[3, y, x] = 1.0

        return tensor
    
    def get_action_mask(self):
        """
        全アクション (209種) のうち、合法なものに 1.0、違法なものに 0.0 を入れた配列を返す。
        Index 0-80: コマの移動 (移動先 node index)
        Index 81-144: 横壁 (x, y) -> 81 + y*8 + x
        Index 145-208: 縦壁 (x, y) -> 145 + y*8 + x
        """
        mask = np.zeros(81 + 64 + 64, dtype=np.float32)
        moves = legal_moves(self)

        for m in moves:
            if m[0] == "move":
                mask[m[1]] = 1.0
            elif m[0] == "hwall":
                x, y = m[1], m[2]
                mask[81 + y * 8 + x] = 1.0
            elif m[0] == "vwall":
                x, y = m[1], m[2]
                mask[145 + y * 8 + x] = 1.0

        return mask
    
    def clone(self):
        """現在の盤面を高速に複製する"""
        new_board = Board()
        new_board.players = [Player(p.pos, p.walls) for p in self.players]
        new_board.h_walls = self.h_walls.copy()
        new_board.v_walls = self.v_walls.copy()
        new_board.adj = self.adj.copy()
        new_board.turn = self.turn
        new_board.zobrist = self.zobrist
        new_board.zobrist_counter = self.zobrist_counter.copy()
        return new_board
    
    def encode_move(move) -> int:
        """タプル形式の指し手を 0-208 の整数に変換"""
        if move[0] == "move":
            return move[1]  # 0-80
        elif move[0] == "hwall":
            return 81 + move[2] * 8 + move[1]  # 81-144
        elif move[0] == "vwall":
            return 145 + move[2] * 8 + move[1]  # 145-208
        raise ValueError(f"Unknown move type: {move}")

    def decode_move(action_index: int):
        """0-208 の整数をタプル形式の指し手に変換"""
        if action_index < 81:
            return ("move", action_index)
        elif action_index < 145:
            idx = action_index - 81
            return ("hwall", idx % 8, idx // 8)
        else:
            idx = action_index - 145
            return ("vwall", idx % 8, idx // 8)