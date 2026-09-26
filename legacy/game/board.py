# board.py
#
# 盤面表現（すべて整数ビットボード）
#
#   マス      : node = row * 9 + col（0〜80）
#   通行可能辺: open_d（bit i = i と i+9 の間が通れる）/ open_r（bit i = i と i+1）
#               これだけを状態として持つ。隣接ノードは neighbor_mask() で都度求める
#               （adj は互換用の読み取り専用プロパティ）
#   壁        : 壁座標 (x, y)（0〜7）を w = y * 8 + x の 64bit マスク hmask / vmask で保持
#               h_walls / v_walls はそのマスクを読み書きする set 互換のビュー
#
# 壁 (x, y) は左上マス tl = y*9 + x を基準に 2x2 マスの境界を塞ぐ:
#   水平壁: tl↔tl+9, tl+1↔tl+10 を遮断（open_d の bit tl, tl+1）
#   垂直壁: tl↔tl+1, tl+9↔tl+10 を遮断（open_r の bit tl, tl+9）

from collections.abc import MutableSet
from dataclasses import dataclass, field
import random
from typing import List, Set, Tuple
import numpy as np

BOARD_SIZE = 9
NODE_COUNT = BOARD_SIZE * BOARD_SIZE
MAX_WALLS  = 10
WALL_GRID  = BOARD_SIZE - 1          # 壁座標は 0〜7
WALL_COUNT = WALL_GRID * WALL_GRID   # 向きごとに 64 通り

_GOAL_MASK = [
    sum(1 << (72 + c) for c in range(9)),
    sum(1 << c for c in range(9)),
]

ALL_NODES = (1 << NODE_COUNT) - 1

# 方向別の通行可能マスク（壁なし状態）
#   FULL_D: bit i が立っていれば i と i+9（下）が通行可能（row 0〜7）
#   FULL_R: bit i が立っていれば i と i+1（右）が通行可能（col 0〜7）
FULL_D = sum(1 << (r * 9 + c) for r in range(8) for c in range(9))
FULL_R = sum(1 << (r * 9 + c) for r in range(9) for c in range(8))

# 壁座標（64bit）用の定数マスク
ALL_WALLS = (1 << WALL_COUNT) - 1
W_COL0 = sum(1 << (y * 8) for y in range(8))       # x == 0
W_COL7 = W_COL0 << 7                               # x == 7
W_ROW0 = 0xFF                                      # y == 0
W_ROW7 = 0xFF << 56                                # y == 7

# 共有の指し手タプル（生成のたびにタプルを作らないため）
PAWN_MOVE  = tuple(("move", i) for i in range(NODE_COUNT))
HWALL_MOVE = tuple(("hwall", w % 8, w // 8) for w in range(WALL_COUNT))
VWALL_MOVE = tuple(("vwall", w % 8, w // 8) for w in range(WALL_COUNT))


def neighbor_mask(bits, D, R):
    """bits（ノード集合）に隣接し、辺が通れるノードの集合。"""
    return ((bits & D) << 9) | ((bits >> 9) & D) | ((bits & R) << 1) | ((bits >> 1) & R)


def _initial_dirs():
    BS = BOARD_SIZE
    dirs = []
    for node in range(NODE_COUNT):
        x, y = node % BS, node // BS
        dirs.append({
            'up':    node - BS if y > 0      else -1,
            'down':  node + BS if y < BS - 1 else -1,
            'left':  node - 1  if x > 0      else -1,
            'right': node + 1  if x < BS - 1 else -1,
        })
    return dirs


_DIRS = _initial_dirs()   # 読み取り専用。全盤面で共有する


class WallView(MutableSet):
    """
    盤面の壁マスク（hmask / vmask）を (x, y) の集合として見せるビュー。
    set と同様に in / 反復 / len / add / discard / remove が使える。
    add / discard はマスクのみを変更する（辺の遮断やハッシュは変えない）。
    """
    __slots__ = ('_board', '_h')

    def __init__(self, board, horizontal):
        self._board = board
        self._h = horizontal

    def _get(self):
        return self._board.hmask if self._h else self._board.vmask

    def _set(self, m):
        if self._h:
            self._board.hmask = m
        else:
            self._board.vmask = m

    def __contains__(self, xy):
        try:
            x, y = xy
        except (TypeError, ValueError):
            return False
        if not (0 <= x < WALL_GRID and 0 <= y < WALL_GRID):
            return False
        return bool((self._get() >> (y * 8 + x)) & 1)

    def __iter__(self):
        m = self._get()
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            yield (w % 8, w // 8)
            m ^= lsb

    def __len__(self):
        return self._get().bit_count()

    def add(self, xy):
        x, y = xy
        self._set(self._get() | (1 << (y * 8 + x)))

    def discard(self, xy):
        if xy in self:
            x, y = xy
            self._set(self._get() & ~(1 << (y * 8 + x)))

    def copy(self):
        return set(self)

    def __repr__(self):
        return f"WallView({set(self)!r})"


# ----------------------------------------------------------------------
# Zobrist テーブル（盤面ごとに生成。clone() した盤面同士は表を共有する）
# ----------------------------------------------------------------------
class _ZobristTables:
    __slots__ = ('pos', 'hwall', 'vwall', 'turn', 'walls', 'hw', 'vw')

    def __init__(self, seed):
        rnd = random.Random(seed)
        self.pos   = [[rnd.getrandbits(64) for _ in range(NODE_COUNT)] for _ in range(2)]
        self.hwall = [[rnd.getrandbits(64) for _ in range(WALL_GRID)] for _ in range(WALL_GRID)]
        self.vwall = [[rnd.getrandbits(64) for _ in range(WALL_GRID)] for _ in range(WALL_GRID)]
        self.turn  = rnd.getrandbits(64)
        self.walls = [[rnd.getrandbits(64) for _ in range(MAX_WALLS + 1)] for _ in range(2)]
        # make_move 用のフラットな表（w = y*8 + x）
        self.hw = [self.hwall[w // 8][w % 8] for w in range(WALL_COUNT)]
        self.vw = [self.vwall[w // 8][w % 8] for w in range(WALL_COUNT)]



@dataclass(slots=True)
class Player:
    pos: int
    walls: int = MAX_WALLS


@dataclass(slots=True)
class Board:
    players: List[Player] = field(default_factory=lambda: [Player(4), Player(76)])

    # 壁（w = y*8 + x のビット集合）。h_walls / v_walls はこれのビュー
    hmask: int = field(init=False, default=0)
    vmask: int = field(init=False, default=0)
    _hview: WallView = field(init=False, default=None, repr=False)
    _vview: WallView = field(init=False, default=None, repr=False)

    # 方向ベースの隣接表（読み取り専用・全盤面で共有）
    dirs: List[dict] = field(default_factory=lambda: _DIRS)

    # 方向別ビットボード。BFS を 1 層あたりシフト演算 4 回で展開するためのもの。
    open_d: int = field(init=False, default=FULL_D)
    open_r: int = field(init=False, default=FULL_R)

    turn: int = 0
    zobrist: int = 0

    z_pos:   List[List[int]] = field(init=False, default_factory=list)
    z_hwall: List[List[int]] = field(init=False, default_factory=list)
    z_vwall: List[List[int]] = field(init=False, default_factory=list)
    z_turn:  int = field(init=False, default=0)
    # 壁の残数もハッシュに含める（残数が違えば評価値・合法手が変わるため）
    z_walls: List[List[int]] = field(init=False, default_factory=list)
    _z_hw:   List[int] = field(init=False, default_factory=list)
    _z_vw:   List[int] = field(init=False, default_factory=list)

    move_stack: List = field(default_factory=list)

    # ハッシュ値 → 出現回数（千日手判定用。参照・更新とも O(1)）
    zobrist_counter: dict = field(default_factory=dict)

    def __post_init__(self):
        self._hview = WallView(self, True)
        self._vview = WallView(self, False)
        self._set_zobrist_tables(_ZobristTables(None))

    @property
    def h_walls(self) -> Set[Tuple[int, int]]:
        return self._hview

    @property
    def v_walls(self) -> Set[Tuple[int, int]]:
        return self._vview

    # ------------------------------------------------------------------
    # 隣接ノード
    # ------------------------------------------------------------------
    @staticmethod
    def _iter_bits(mask: int):
        """ビットマスクからセットされているビット位置を順に yield する。"""
        while mask:
            lsb = mask & (-mask)
            yield lsb.bit_length() - 1
            mask ^= lsb

    def neighbors(self, node: int):
        """node の現在の隣接ノード一覧を返すジェネレータ。"""
        yield from self._iter_bits(neighbor_mask(1 << node, self.open_d, self.open_r))

    @property
    def adj(self) -> List[int]:
        """互換用: 各ノードの隣接ノードのビットマスク（都度計算・読み取り専用）。"""
        D, R = self.open_d, self.open_r
        return [neighbor_mask(1 << n, D, R) for n in range(NODE_COUNT)]

    @property
    def edges(self) -> List[Set[int]]:
        """互換用: 各ノードの隣接集合（低速・テスト用途）。"""
        return [set(self._iter_bits(m)) for m in self.adj]

    # ------------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------------
    def init_edges(self):
        """壁のない状態の辺に戻す。"""
        self.open_d = FULL_D
        self.open_r = FULL_R

    def _link(self, a: int, b: int):
        self.add_edge(a, b)

    def init_zobrist(self, seed=None):
        """Zobrist 表を作り直す。seed を指定すると決定的な表になる。"""
        self._set_zobrist_tables(_ZobristTables(seed))

    def _set_zobrist_tables(self, zt):
        self.z_pos   = zt.pos
        self.z_hwall = zt.hwall
        self.z_vwall = zt.vwall
        self.z_turn  = zt.turn
        self.z_walls = zt.walls
        self._z_hw   = zt.hw
        self._z_vw   = zt.vw
        self.zobrist = self.compute_zobrist()

    def compute_zobrist(self) -> int:
        """現在の盤面からハッシュを計算し直す（差分更新の検証用にも使える）。"""
        z = 0
        for i in range(2):
            z ^= self.z_pos[i][self.players[i].pos]
            z ^= self.z_walls[i][self.players[i].walls]
        for x, y in self.h_walls:
            z ^= self.z_hwall[y][x]
        for x, y in self.v_walls:
            z ^= self.z_vwall[y][x]
        if self.turn:
            z ^= self.z_turn
        return z

    # ------------------------------------------------------------------
    # 辺の削除・追加（低レベル API）
    # 注意: 壁の集合（h_walls / v_walls）は更新しない。壁の合法性判定は
    #       壁の集合を基準にするため、通常は壁の API を使うこと。
    # ------------------------------------------------------------------
    def remove_edge(self, a: int, b: int):
        lo = a if a < b else b
        if abs(a - b) == BOARD_SIZE:
            self.open_d &= ~(1 << lo)
        else:
            self.open_r &= ~(1 << lo)

    def add_edge(self, a: int, b: int):
        lo = a if a < b else b
        if abs(a - b) == BOARD_SIZE:
            self.open_d |= (1 << lo)
        else:
            self.open_r |= (1 << lo)

    # ------------------------------------------------------------------
    # 壁による辺の遮断・復元（壁の集合・ハッシュは変更しない）
    # ------------------------------------------------------------------
    def apply_hwall(self, x: int, y: int):
        self.open_d &= ~(3 << (y * 9 + x))

    def undo_hwall(self, x: int, y: int):
        self.open_d |= 3 << (y * 9 + x)

    def apply_vwall(self, x: int, y: int):
        self.open_r &= ~(0x201 << (y * 9 + x))     # bit tl と tl+9

    def undo_vwall(self, x: int, y: int):
        self.open_r |= 0x201 << (y * 9 + x)

    # ------------------------------------------------------------------
    # 指し手の実行・取消
    # move_stack の形式: ("move", 旧位置, 新位置) / ("hwall", x, y) / ("vwall", x, y)
    # ------------------------------------------------------------------
    def make_move(self, move):
        t = self.turn
        player = self.players[t]
        kind = move[0]

        if kind == "move":
            old = player.pos
            new = move[1]
            self.move_stack.append(("move", old, new))
            zp = self.z_pos[t]
            z = self.zobrist ^ zp[old] ^ zp[new]
            player.pos = new

        else:
            x = move[1]
            y = move[2]
            w = y * 8 + x
            tl = y * 9 + x
            walls = player.walls
            zw = self.z_walls[t]
            z = self.zobrist ^ zw[walls] ^ zw[walls - 1]
            player.walls = walls - 1
            if kind == "hwall":
                self.move_stack.append(move)
                self.hmask |= 1 << w
                z ^= self._z_hw[w]
                self.open_d &= ~(3 << tl)
            elif kind == "vwall":
                self.move_stack.append(move)
                self.vmask |= 1 << w
                z ^= self._z_vw[w]
                self.open_r &= ~(0x201 << tl)
            else:
                player.walls = walls
                raise ValueError(f"Unknown move type: {move}")

        z ^= self.z_turn
        self.zobrist = z
        self.turn = t ^ 1
        zc = self.zobrist_counter
        zc[z] = zc.get(z, 0) + 1

    def undo_move(self):
        z = self.zobrist
        zc = self.zobrist_counter
        cnt = zc.get(z, 1) - 1
        if cnt <= 0:
            zc.pop(z, None)
        else:
            zc[z] = cnt

        t = self.turn ^ 1
        self.turn = t
        z ^= self.z_turn

        move = self.move_stack.pop()
        player = self.players[t]
        kind = move[0]

        if kind == "move":
            zp = self.z_pos[t]
            z ^= zp[move[2]] ^ zp[move[1]]
            player.pos = move[1]

        elif kind != "pass":
            x = move[1]
            y = move[2]
            w = y * 8 + x
            tl = y * 9 + x
            walls = player.walls
            zw = self.z_walls[t]
            z ^= zw[walls] ^ zw[walls + 1]
            player.walls = walls + 1
            if kind == "hwall":
                self.hmask &= ~(1 << w)
                z ^= self._z_hw[w]
                self.open_d |= 3 << tl
            else:
                self.vmask &= ~(1 << w)
                z ^= self._z_vw[w]
                self.open_r |= 0x201 << tl

        self.zobrist = z

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
            self.zobrist_counter.pop(z, None)
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
        tensor = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

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
        from game.movegen import pawn_dest_mask, legal_wall_masks  # 循環 import 回避
        mask = np.zeros(81 + 64 + 64, dtype=np.float32)
        pm = pawn_dest_mask(self)
        if self.players[self.turn].walls > 0:
            hm, vm = legal_wall_masks(self)
        else:
            hm = vm = 0
        for i in self._iter_bits(pm):
            mask[i] = 1.0
        for w in self._iter_bits(hm):
            mask[81 + w] = 1.0
        for w in self._iter_bits(vm):
            mask[145 + w] = 1.0
        return mask

    def clone(self):
        """現在の盤面を高速に複製する（__init__ を経由しない）。"""
        nb = object.__new__(Board)
        nb.players = [Player(p.pos, p.walls) for p in self.players]
        nb.hmask = self.hmask
        nb.vmask = self.vmask
        nb._hview = WallView(nb, True)
        nb._vview = WallView(nb, False)
        nb.dirs = self.dirs
        nb.open_d = self.open_d
        nb.open_r = self.open_r
        nb.turn = self.turn
        nb.zobrist = self.zobrist
        # Zobrist テーブルは共有（複製後の make_move でハッシュが整合するように）
        nb.z_pos   = self.z_pos
        nb.z_hwall = self.z_hwall
        nb.z_vwall = self.z_vwall
        nb.z_turn  = self.z_turn
        nb.z_walls = self.z_walls
        nb._z_hw   = self._z_hw
        nb._z_vw   = self._z_vw
        nb.move_stack = self.move_stack.copy()
        nb.zobrist_counter = self.zobrist_counter.copy()
        return nb

    @staticmethod
    def encode_move(move) -> int:
        """タプル形式の指し手を 0-208 の整数に変換"""
        if move[0] == "move":
            return move[1]  # 0-80
        elif move[0] == "hwall":
            return 81 + move[2] * 8 + move[1]  # 81-144
        elif move[0] == "vwall":
            return 145 + move[2] * 8 + move[1]  # 145-208
        raise ValueError(f"Unknown move type: {move}")

    @staticmethod
    def decode_move(action_index: int):
        """0-208 の整数をタプル形式の指し手に変換"""
        if action_index < 81:
            return PAWN_MOVE[action_index]
        elif action_index < 145:
            return HWALL_MOVE[action_index - 81]
        else:
            return VWALL_MOVE[action_index - 145]
