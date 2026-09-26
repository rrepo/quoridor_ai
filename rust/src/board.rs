//! 盤面・指し手の実行と取消・合法手生成

use std::cell::Cell;
use std::fmt;
use std::sync::Arc;

use crate::consts::*;
use crate::path::{distance, expand, reachable, trace_path};
use crate::zobrist::Zobrist;

/// パス（ヌルムーブ）を表す番号。探索用で、合法手には含まれない。
pub const PASS: u8 = 255;

/// 壁の合法性判定に使う「ゴールへ通じる経路」（最短とは限らない）。
///
/// 壁が経路を切らなければ、その経路の持ち主は必ずゴールへ行ける。
/// make でこの経路を差分更新し、探索中の再計算を避ける。
/// 有効かどうかは Board::path_valid のビットで持つ（壁で無効化しても中身は残し、
/// undo ではビットを戻すだけで済むようにする）。
#[derive(Clone, Copy, Default)]
struct PathCache {
    /// 経路の辺を切る壁（水平 / 垂直）。経路の辺をすべて含んでいればよい（上位集合で可）
    hcut: u64,
    vcut: u64,
    /// 経路上のマス（u128 を 2 語に分けて 8 バイト境界に収める）
    nodes: [u64; 2],
}

impl PathCache {
    #[inline(always)]
    fn has_node(&self, n: u8) -> bool {
        (self.nodes[(n >> 6) as usize] >> (n & 63)) & 1 != 0
    }
    #[inline(always)]
    fn add_node(&mut self, n: u8) {
        self.nodes[(n >> 6) as usize] |= 1u64 << (n & 63);
    }
}

#[derive(Clone, Copy)]
struct Undo {
    action: u8,
    /// コマ移動のときの移動前の位置
    prev_pos: u8,
    /// 指す前の経路キャッシュの有効ビット
    path_valid: u8,
    /// コマ移動のとき、動いた側の指す前の経路キャッシュ
    saved: PathCache,
}

/// 合法手のリスト（最大 5 + 128 手なので固定長の配列で持つ）
#[derive(Clone)]
pub struct MoveList {
    /// 末尾に 8 バイトの書き込み余裕を持たせている（push_wall_bits の一括書き込み用）
    moves: [u8; 256],
    len: usize,
}

/// バイト値 b → 立っているビット位置（0〜7）を下位バイトから詰めた u64
const BIT_OFFSETS: [u64; 256] = {
    let mut t = [0u64; 256];
    let mut b = 0;
    while b < 256 {
        let mut v = 0u64;
        let mut k = 0;
        let mut i = 0;
        while i < 8 {
            if (b >> i) & 1 != 0 {
                v |= (i as u64) << (8 * k);
                k += 1;
            }
            i += 1;
        }
        t[b] = v;
        b += 1;
    }
    t
};

impl MoveList {
    #[inline(always)]
    pub fn new() -> Self {
        MoveList { moves: [0; 256], len: 0 }
    }
    #[inline(always)]
    fn push(&mut self, a: u8) {
        self.moves[self.len] = a;
        self.len += 1;
    }
    /// 壁のマスク m（bit w）を手番号 base + w として追加する。8 手ずつまとめて書き込む。
    #[inline(always)]
    fn push_wall_bits(&mut self, mut m: u64, base: u8) {
        let mut chunk_base = base as u64;
        while m != 0 {
            let b = (m & 0xFF) as usize;
            if b != 0 {
                // 各バイトに (base + 8k) を足す（値は 208 以下なので桁あふれしない）
                let packed = BIT_OFFSETS[b] + chunk_base * 0x0101_0101_0101_0101;
                self.moves[self.len..self.len + 8].copy_from_slice(&packed.to_le_bytes());
                self.len += b.count_ones() as usize;
            }
            m >>= 8;
            chunk_base += 8;
        }
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        &self.moves[..self.len]
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct Board {
    pub open_d: u128,
    pub open_r: u128,
    pub hmask: u64,
    pub vmask: u64,
    pub pos: [u8; 2],
    pub walls: [u8; 2],
    pub turn: u8,
    pub hash: u64,
    stack: Vec<Undo>,
    /// 各手の後のハッシュ（千日手判定用）
    history: Vec<u64>,
    z: Arc<Zobrist>,
    paths: Cell<[PathCache; 2]>,
    /// paths の有効ビット（bit p = プレイヤー p）
    path_valid: Cell<u8>,
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    pub fn new() -> Self {
        Self::with_zobrist(Zobrist::default_shared())
    }

    /// seed から Zobrist テーブルを作る（決定的なハッシュが欲しいとき）
    pub fn with_seed(seed: u64) -> Self {
        Self::with_zobrist(Arc::new(Zobrist::new(seed)))
    }

    fn with_zobrist(z: Arc<Zobrist>) -> Self {
        let mut b = Board {
            open_d: FULL_D,
            open_r: FULL_R,
            hmask: 0,
            vmask: 0,
            pos: [4, 76],
            walls: [MAX_WALLS; 2],
            turn: 0,
            hash: 0,
            stack: Vec::with_capacity(256),
            history: Vec::with_capacity(256),
            z,
            paths: Cell::new([PathCache::default(); 2]),
            path_valid: Cell::new(0),
        };
        b.hash = b.compute_hash();
        b
    }

    /// 任意の局面を作る（壁は合法に置かれている前提）。
    pub fn from_parts(pos: [u8; 2], walls: [u8; 2], turn: u8, hmask: u64, vmask: u64) -> Self {
        let mut b = Board::new();
        b.pos = pos;
        b.walls = walls;
        b.turn = turn & 1;
        b.hmask = hmask;
        b.vmask = vmask;
        let mut m = hmask;
        while m != 0 {
            let w = m.trailing_zeros() as usize;
            b.open_d &= !hwall_d_bits(w);
            m &= m - 1;
        }
        let mut m = vmask;
        while m != 0 {
            let w = m.trailing_zeros() as usize;
            b.open_r &= !vwall_r_bits(w);
            m &= m - 1;
        }
        b.hash = b.compute_hash();
        b
    }

    /// 現在の盤面からハッシュを計算し直す
    pub fn compute_hash(&self) -> u64 {
        let z = &*self.z;
        let mut h = 0;
        for p in 0..2 {
            h ^= z.pos[p][self.pos[p] as usize];
            h ^= z.walls[p][self.walls[p] as usize];
        }
        let mut m = self.hmask;
        while m != 0 {
            h ^= z.hwall[m.trailing_zeros() as usize];
            m &= m - 1;
        }
        let mut m = self.vmask;
        while m != 0 {
            h ^= z.vwall[m.trailing_zeros() as usize];
            m &= m - 1;
        }
        if self.turn == 1 {
            h ^= z.turn;
        }
        h
    }

    // ------------------------------------------------------------------
    // 指し手の実行・取消（合法性は検査しない。is_legal で事前に確認すること）
    // ------------------------------------------------------------------
    #[inline]
    pub fn make(&mut self, a: u8) {
        let t = self.turn as usize;
        let z = &*self.z;
        let mut h = self.hash;
        let valid = self.path_valid.get();
        if a < HWALL_BASE {
            let old = self.pos[t];
            let mut pc = self.paths.get();
            self.stack.push(Undo { action: a, prev_pos: old, path_valid: valid, saved: pc[t] });
            h ^= z.pos[t][old as usize] ^ z.pos[t][a as usize];
            self.pos[t] = a;
            // 経路キャッシュ: 移動先が経路上ならそのまま、1 マスの移動なら経路を延ばす
            let p = &mut pc[t];
            if valid >> t & 1 != 0 && !p.has_node(a) {
                let (o, n) = (old as usize, a as usize);
                if o + 9 == n {
                    p.hcut |= HCUT_D[o];
                } else if n + 9 == o {
                    p.hcut |= HCUT_D[n];
                } else if o + 1 == n && n % 9 != 0 {
                    p.vcut |= VCUT_R[o];
                } else if n + 1 == o && o % 9 != 0 {
                    p.vcut |= VCUT_R[n];
                } else {
                    self.path_valid.set(valid & !(1 << t)); // ジャンプ・斜め移動
                }
                p.add_node(a);
                self.paths.set(pc);
            }
        } else {
            let wl = self.walls[t] as usize;
            h ^= z.walls[t][wl] ^ z.walls[t][wl - 1];
            self.walls[t] -= 1;
            self.stack.push(Undo { action: a, prev_pos: 0, path_valid: valid, saved: PathCache::default() });
            // 経路キャッシュ: 経路を切る壁なら無効化（中身は残す）
            let pc = self.paths.get();
            let mut v = valid;
            if a < VWALL_BASE {
                let w = (a - HWALL_BASE) as usize;
                self.hmask |= 1u64 << w;
                self.open_d &= !hwall_d_bits(w);
                h ^= z.hwall[w];
                for (p, c) in pc.iter().enumerate() {
                    if c.hcut >> w & 1 != 0 {
                        v &= !(1 << p);
                    }
                }
            } else {
                let w = (a - VWALL_BASE) as usize;
                self.vmask |= 1u64 << w;
                self.open_r &= !vwall_r_bits(w);
                h ^= z.vwall[w];
                for (p, c) in pc.iter().enumerate() {
                    if c.vcut >> w & 1 != 0 {
                        v &= !(1 << p);
                    }
                }
            }
            self.path_valid.set(v);
        }
        h ^= z.turn;
        self.hash = h;
        self.turn ^= 1;
        self.history.push(h);
    }

    /// 直前の手を取り消す。取り消す手がなければ false。
    #[inline]
    pub fn undo(&mut self) -> bool {
        let u = match self.stack.pop() {
            Some(u) => u,
            None => return false,
        };
        self.history.pop();
        self.path_valid.set(u.path_valid);
        self.turn ^= 1;
        let t = self.turn as usize;
        let z = &*self.z;
        let mut h = self.hash ^ z.turn;
        let a = u.action;
        if a == PASS {
            // 手番とハッシュの手番ビットのみ
        } else if a < HWALL_BASE {
            h ^= z.pos[t][a as usize] ^ z.pos[t][u.prev_pos as usize];
            self.pos[t] = u.prev_pos;
            let mut pc = self.paths.get();
            pc[t] = u.saved;
            self.paths.set(pc);
        } else {
            let wl = self.walls[t] as usize;
            h ^= z.walls[t][wl] ^ z.walls[t][wl + 1];
            self.walls[t] += 1;
            if a < VWALL_BASE {
                let w = (a - HWALL_BASE) as usize;
                self.hmask &= !(1u64 << w);
                self.open_d |= hwall_d_bits(w);
                h ^= z.hwall[w];
            } else {
                let w = (a - VWALL_BASE) as usize;
                self.vmask &= !(1u64 << w);
                self.open_r |= vwall_r_bits(w);
                h ^= z.vwall[w];
            }
        }
        self.hash = h;
        true
    }

    /// パス（ヌルムーブ）。undo で取り消せる。
    #[inline]
    pub fn make_pass(&mut self) {
        self.stack.push(Undo {
            action: PASS,
            prev_pos: 0,
            path_valid: self.path_valid.get(),
            saved: PathCache::default(),
        });
        self.turn ^= 1;
        self.hash ^= self.z.turn;
        self.history.push(self.hash);
    }

    /// 現在の局面がこれまでの手で何回現れたか（初期局面は数えない）
    pub fn repetitions(&self) -> u32 {
        let h = self.hash;
        self.history.iter().filter(|&&x| x == h).count() as u32
    }

    pub fn ply(&self) -> usize {
        self.stack.len()
    }

    /// 終局していれば勝者（0 / 1）
    #[inline]
    pub fn winner(&self) -> Option<u8> {
        if self.pos[0] >= 72 {
            Some(0)
        } else if self.pos[1] < 9 {
            Some(1)
        } else {
            None
        }
    }

    // ------------------------------------------------------------------
    // 経路
    // ------------------------------------------------------------------
    /// player のゴール行までの最短距離（コマは障害物とみなさない）
    #[inline]
    pub fn shortest_path(&self, player: usize) -> Option<u32> {
        distance(1u128 << self.pos[player], GOAL[player], self.open_d, self.open_r)
    }

    /// 最短経路のマス列（始点〜ゴール行）
    pub fn shortest_path_nodes(&self, player: usize) -> Option<Vec<u8>> {
        let mut rev = Vec::with_capacity(32);
        let mut goal_node = None;
        trace_path(self.pos[player] as usize, GOAL[player], self.open_d, self.open_r, |prev, cur| {
            if goal_node.is_none() {
                goal_node = Some(cur);
                rev.push(cur as u8);
            }
            rev.push(prev as u8);
        })?;
        if rev.is_empty() {
            rev.push(self.pos[player]);
        }
        rev.reverse();
        Some(rev)
    }

    /// player の最短経路（shortest_path_nodes と同じ経路）を切る壁の集合 (水平, 垂直)
    #[inline]
    pub fn path_cut_masks(&self, player: usize) -> Option<(u64, u64)> {
        let mut hcut = 0u64;
        let mut vcut = 0u64;
        trace_path(self.pos[player] as usize, GOAL[player], self.open_d, self.open_r, |prev, cur| {
            if prev + 9 == cur {
                hcut |= HCUT_D[prev];
            } else if cur + 9 == prev {
                hcut |= HCUT_D[cur];
            } else if prev + 1 == cur {
                vcut |= VCUT_R[prev];
            } else {
                vcut |= VCUT_R[cur];
            }
        })?;
        Some((hcut, vcut))
    }

    /// 両プレイヤーの経路キャッシュ（無効なら計算し直す）。到達不能なら None。
    #[inline]
    fn cached_paths(&self) -> Option<[PathCache; 2]> {
        let mut pc = self.paths.get();
        let valid = self.path_valid.get();
        if valid != 3 {
            for (p, c) in pc.iter_mut().enumerate() {
                if valid >> p & 1 != 0 {
                    continue;
                }
                let (mut hcut, mut vcut) = (0u64, 0u64);
                let mut nodes = PathCache::default();
                nodes.add_node(self.pos[p]);
                trace_path(self.pos[p] as usize, GOAL[p], self.open_d, self.open_r, |prev, cur| {
                    nodes.add_node(cur as u8);
                    if prev + 9 == cur {
                        hcut |= HCUT_D[prev];
                    } else if cur + 9 == prev {
                        hcut |= HCUT_D[cur];
                    } else if prev + 1 == cur {
                        vcut |= VCUT_R[prev];
                    } else {
                        vcut |= VCUT_R[cur];
                    }
                })?;
                *c = PathCache { hcut, vcut, nodes: nodes.nodes };
            }
            self.paths.set(pc);
            self.path_valid.set(3);
        }
        Some(pc)
    }

    // ------------------------------------------------------------------
    // 合法手
    // ------------------------------------------------------------------
    /// 手番側のコマの移動先（ジャンプ・斜め移動を含む）
    #[inline]
    pub fn pawn_dest_mask(&self) -> u128 {
        let t = self.turn as usize;
        let pos = self.pos[t] as i32;
        let opp = self.pos[t ^ 1] as i32;
        let (d, r) = (self.open_d, self.open_r);
        let c = 1u128 << pos;
        let mut m = expand(c, d, r);
        let ob = 1u128 << opp;
        if m & ob != 0 {
            // 相手と隣接 → 直進ジャンプ、塞がれていれば相手の左右（斜め）
            m ^= ob;
            let ao = expand(ob, d, r);
            let jump = 2 * opp - pos;
            if (0..81).contains(&jump) && (ao >> jump) & 1 != 0 {
                m |= 1u128 << jump;
            } else {
                m |= ao & !c;
            }
        }
        m
    }

    /// 重なり・交差のない壁の集合 (水平, 垂直)
    #[inline]
    pub fn valid_wall_masks(&self) -> (u64, u64) {
        let (hw, vw) = (self.hmask, self.vmask);
        let vh = !(hw | ((hw << 1) & W_NOT_COL0) | ((hw >> 1) & W_NOT_COL7) | vw);
        let vv = !(vw | (vw << 8) | (vw >> 8) | hw);
        (vh, vv)
    }

    /// 合法な壁の集合 (水平, 垂直)。restrict で調べる壁を限定できる。壁の残数は見ない。
    ///
    /// 1. 重なり・交差をビット演算で除外
    /// 2. 閉路を作らない壁（既存の壁のかたまり・盤端のうち同じ成分に 2 点以上で接しない壁）は
    ///    経路を断たない（cycle_risky_masks。64 通りまとめて計算）
    /// 3. 残りのうち、各プレイヤーの経路（キャッシュ）の辺を切る壁だけ BFS で確認する
    ///
    /// 前提: 両プレイヤーともゴールに到達可能な（合法な）局面であること。
    pub fn legal_wall_masks_restricted(&self, restrict_h: u64, restrict_v: u64) -> (u64, u64) {
        let (vh, vv) = self.valid_wall_masks();
        let mut ch = vh & restrict_h;
        let mut cv = vv & restrict_v;
        if ch | cv == 0 {
            return (0, 0);
        }

        let (risky_h, risky_v) = cycle_risky_masks(self.hmask, self.vmask);

        let mut need_h = ch & risky_h;
        let mut need_v = cv & risky_v;
        if need_h | need_v == 0 {
            return (ch, cv);
        }

        let (d, r) = (self.open_d, self.open_r);
        let s0 = 1u128 << self.pos[0];
        let s1 = 1u128 << self.pos[1];

        let Some([p0, p1]) = self.cached_paths() else {
            return (0, 0);
        };
        let (h0, v0, h1, v1) = (p0.hcut, p0.vcut, p1.hcut, p1.vcut);
        need_h &= h0 | h1;
        need_v &= v0 | v1;

        while need_h != 0 {
            let w = need_h.trailing_zeros() as usize;
            let bit = 1u64 << w;
            need_h &= need_h - 1;
            let d2 = d & !hwall_d_bits(w);
            if (h0 & bit != 0 && !reachable(s0, GOAL[0], d2, r))
                || (h1 & bit != 0 && !reachable(s1, GOAL[1], d2, r))
            {
                ch ^= bit;
            }
        }
        while need_v != 0 {
            let w = need_v.trailing_zeros() as usize;
            let bit = 1u64 << w;
            need_v &= need_v - 1;
            let r2 = r & !vwall_r_bits(w);
            if (v0 & bit != 0 && !reachable(s0, GOAL[0], d, r2))
                || (v1 & bit != 0 && !reachable(s1, GOAL[1], d, r2))
            {
                cv ^= bit;
            }
        }
        (ch, cv)
    }

    #[inline]
    pub fn legal_wall_masks(&self) -> (u64, u64) {
        self.legal_wall_masks_restricted(!0, !0)
    }

    /// 全合法手
    #[inline]
    pub fn legal_actions(&self) -> MoveList {
        let mut ml = MoveList::new();
        let mut m = self.pawn_dest_mask();
        while m != 0 {
            ml.push(m.trailing_zeros() as u8);
            m &= m - 1;
        }
        if self.walls[self.turn as usize] > 0 {
            let (h, v) = self.legal_wall_masks();
            ml.push_wall_bits(h, HWALL_BASE);
            ml.push_wall_bits(v, VWALL_BASE);
        }
        ml
    }

    /// 全合法手の数（リストを作らない）
    #[inline]
    pub fn count_legal_actions(&self) -> u32 {
        let mut n = self.pawn_dest_mask().count_ones();
        if self.walls[self.turn as usize] > 0 {
            let (h, v) = self.legal_wall_masks();
            n += h.count_ones() + v.count_ones();
        }
        n
    }

    /// 手 a が合法か
    pub fn is_legal(&self, a: u8) -> bool {
        if a < HWALL_BASE {
            return (self.pawn_dest_mask() >> a) & 1 != 0;
        }
        if a as usize >= ACTION_COUNT || self.walls[self.turn as usize] == 0 {
            return false;
        }
        if a < VWALL_BASE {
            let bit = 1u64 << (a - HWALL_BASE);
            self.legal_wall_masks_restricted(bit, 0).0 != 0
        } else {
            let bit = 1u64 << (a - VWALL_BASE);
            self.legal_wall_masks_restricted(0, bit).1 != 0
        }
    }

    // ------------------------------------------------------------------
    // perft（合法手生成の正しさと速さの検証用）
    // ------------------------------------------------------------------
    /// 深さ depth の葉の数。末端でも make/undo を行う（終局判定はしない）
    pub fn perft(&mut self, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let ml = self.legal_actions();
        let mut n = 0;
        for &a in ml.as_slice() {
            self.make(a);
            n += if depth == 1 { 1 } else { self.perft(depth - 1) };
            self.undo();
        }
        n
    }

    /// perft と同じ値を、末端の 1 手前で手数だけ数えて求める（高速版）
    pub fn perft_bulk(&mut self, depth: u32) -> u64 {
        match depth {
            0 => 1,
            1 => self.count_legal_actions() as u64,
            _ => {
                let ml = self.legal_actions();
                let mut n = 0;
                for &a in ml.as_slice() {
                    self.make(a);
                    n += self.perft_bulk(depth - 1);
                    self.undo();
                }
                n
            }
        }
    }
}

/// 閉路を作る壁（同じ連結成分に 2 点以上で接する壁）の集合 (水平, 垂直)
///
/// 格子点（内部 8x8 点 + 盤端）を壁でつないだグラフの連結成分を塗りつぶしで求め、
/// 新しい壁の接点のうち 2 つが同じ成分に属するときだけ閉路ができる。
#[inline]
pub fn cycle_risky_masks(hw: u64, vw: u64) -> (u64, u64) {
    let occ = hw | ((hw << 1) & W_NOT_COL0) | ((hw >> 1) & W_NOT_COL7) | vw | (vw << 8) | (vw >> 8);
    // 点 p と p+1 をつなぐ水平の線分 / 点 p と p+8 をつなぐ垂直の線分
    let hlink = (hw | (hw >> 1)) & W_NOT_COL7;
    let vlink = (vw & !W_ROW7) | (vw >> 8);
    // 盤端に直接つながる点
    let border_pts = (hw & (W_COL0 | W_COL7)) | (vw & (W_ROW0 | W_ROW7));
    let fill = |seed: u64| -> u64 {
        let mut c = seed;
        loop {
            let n = c | ((c & hlink) << 1) | ((c >> 1) & hlink) | ((c & vlink) << 8) | ((c >> 8) & vlink);
            if n == c {
                return c;
            }
            c = n;
        }
    };
    let mut rh = 0u64;
    let mut rv = 0u64;
    let mut add = |c: u64, border: bool| {
        let (bl, br, bt, bb) = if border { (W_COL0, W_COL7, W_ROW0, W_ROW7) } else { (0, 0, 0, 0) };
        let tl = ((c << 1) & W_NOT_COL0) | bl;
        let tr = ((c >> 1) & W_NOT_COL7) | br;
        rh |= (tl & c) | (tl & tr) | (c & tr);
        let tt = (c << 8) | bt;
        let tb = (c >> 8) | bb;
        rv |= (tt & c) | (tt & tb) | (c & tb);
    };
    let bc = fill(border_pts);
    add(bc, true);
    let mut rest = occ & !bc;
    while rest != 0 {
        let c = fill(rest & rest.wrapping_neg());
        add(c, false);
        rest &= !c;
    }
    (rh, rv)
}

// ----------------------------------------------------------------------
// 指し手の変換
// ----------------------------------------------------------------------
/// 指し手の種類と座標: ("move", node) / ("hwall", x, y) / ("vwall", x, y)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    Pawn(u8),
    HWall(u8, u8),
    VWall(u8, u8),
}

pub fn decode_action(a: u8) -> Option<Move> {
    match a {
        0..=80 => Some(Move::Pawn(a)),
        81..=144 => {
            let w = a - HWALL_BASE;
            Some(Move::HWall(w % 8, w / 8))
        }
        145..=208 => {
            let w = a - VWALL_BASE;
            Some(Move::VWall(w % 8, w / 8))
        }
        _ => None,
    }
}

pub fn encode_move(m: Move) -> Option<u8> {
    match m {
        Move::Pawn(n) if n < 81 => Some(n),
        Move::HWall(x, y) if x < 8 && y < 8 => Some(HWALL_BASE + y * 8 + x),
        Move::VWall(x, y) if x < 8 && y < 8 => Some(VWALL_BASE + y * 8 + x),
        _ => None,
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hw = |x: i32, y: i32| x >= 0 && x < 8 && y >= 0 && y < 8 && (self.hmask >> (y * 8 + x)) & 1 != 0;
        let vw = |x: i32, y: i32| x >= 0 && x < 8 && y >= 0 && y < 8 && (self.vmask >> (y * 8 + x)) & 1 != 0;
        for row in 0..9i32 {
            let mut line = String::new();
            for col in 0..9i32 {
                let n = (row * 9 + col) as u8;
                line.push_str(if n == self.pos[0] {
                    " 0 "
                } else if n == self.pos[1] {
                    " 1 "
                } else {
                    " . "
                });
                if col < 8 {
                    line.push(if vw(col, row) || vw(col, row - 1) { '|' } else { ' ' });
                }
            }
            writeln!(f, "{}", line.trim_end())?;
            if row < 8 {
                let mut sep = String::new();
                for col in 0..9i32 {
                    sep.push_str(if hw(col, row) || hw(col - 1, row) { "---" } else { "   " });
                    if col < 8 {
                        sep.push(' ');
                    }
                }
                writeln!(f, "{}", sep.trim_end())?;
            }
        }
        write!(
            f,
            "Turn: Player {}  |  P0 walls: {}  P1 walls: {}",
            self.turn, self.walls[0], self.walls[1]
        )
    }
}
