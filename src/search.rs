//! 探索（legacy/ai/search.py の移植 + Lazy SMP による並列化 + 時間制限）
//!
//! 反復深化 + aspiration window、PVS、移動手の LMR、ヌルムーブ枝刈り、置換表、
//! キラー手・history による手順付けは Python 版と同じ構成。
//! Python 版からの変更点:
//!   - 評価値は 100 倍の整数（探索窓の幅 1 が正確な意味を持つ）
//!   - 壁の手順付けで向きを取り違えていたバグ（"hwall" を 'h' と比較していた）を修正
//!   - 複数スレッド（置換表を共有し、各スレッドが同じ局面を探索する Lazy SMP）
//!   - 時間制限（反復深化の途中で打ち切り、最後に完了した深さの手を返す）
//!   - 壁の候補の範囲: 相手に並ばれた・追い越された後は相手のゴール側も含める（wall_rows）
//!   - history に上限を設ける（HISTORY_MAX。長い探索でキラー手や前回の最善手を追い越さないように）

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::board::Board;
use crate::consts::*;
use crate::eval::{evaluate_with, wall_score, WallScoreCtx, INF, MATE};
use crate::tt::{Tt, EXACT, GEN_WINDOW, LOWER, NO_MOVE, UPPER};

const MAX_DEPTH: usize = 64;
/// これ以上の深さでは壁手を wall_score で並べる
const WALL_ORDER_DEPTH: i32 = 2;
const NMP_DEPTH_MIN: i32 = 3;
const NMP_REDUCTION: i32 = 2;
const NMP_WALL_MIN: u8 = 2;
/// Python 版の 30.0 に相当
const ASPIRATION_WINDOW: i32 = 3000;
const MAX_RETRIES: u32 = 5;
/// history がこれを超えたら全体を半分にする。並べ替えキーでキラー手（800_000〜）や
/// ルートの前回の最善手（1_000_000）を追い越さないよう、50_000 + history を十分小さく保つ。
const HISTORY_MAX: i32 = 1 << 16;

/// 前方の壁（行マスク）: P0 は y >= row - 1、P1 は y <= row
const FORWARD: [[u64; 9]; 2] = {
    let mut t = [[0u64; 9]; 2];
    let mut row = 0;
    while row < 9 {
        let mut y = 0;
        while y < 8 {
            if y + 1 >= row {
                t[0][row] |= 0xFF << (8 * y);
            }
            if y <= row {
                t[1][row] |= 0xFF << (8 * y);
            }
            y += 1;
        }
        row += 1;
    }
    t
};

/// マス (x, y) の周囲 1 マス以内の壁座標（ルートの手順付けで使う）
const NEAR: [u64; 81] = {
    let mut t = [0u64; 81];
    let mut n = 0;
    while n < 81 {
        let (x, y) = ((n % 9) as i32, (n / 9) as i32);
        let mut m = 0u64;
        let mut dy = -1;
        while dy <= 1 {
            let mut dx = -1;
            while dx <= 1 {
                let (wx, wy) = (x + dx, y + dy);
                if wx >= 0 && wx < 8 && wy >= 0 && wy < 8 {
                    m |= 1u64 << (wy * 8 + wx);
                }
                dx += 1;
            }
            dy += 1;
        }
        t[n] = m;
        n += 1;
    }
    t
};

/// 壁の候補を探す範囲（行マスク）: 手番側から見て前方。
///
/// 相手に並ばれた・追い越された後は、前方だけでは相手のゴール前の壁が入らないので、
/// 相手から見て前方（相手の経路の残り）も加える。
fn wall_rows(b: &Board, me: usize) -> u64 {
    let (my_row, en_row) = ((b.pos[me] / 9) as usize, (b.pos[me ^ 1] / 9) as usize);
    let passed = if me == 0 { en_row <= my_row } else { en_row >= my_row };
    let fwd = FORWARD[me][my_row];
    if passed {
        fwd | FORWARD[me ^ 1][en_row]
    } else {
        fwd
    }
}

/// 探索用に絞り込んだ壁（両者の最短経路を切る、wall_rows の範囲の合法な壁）と、その計算に使った切断マスク
fn candidate_walls(b: &Board) -> Option<((u64, u64), [(u64, u64); 2])> {
    crate::stat!(CANDIDATES);
    let me = b.turn as usize;
    if b.walls[me] == 0 {
        return None;
    }
    let c0 = b.path_cut_masks(0)?;
    let c1 = b.path_cut_masks(1)?;
    let fwd = wall_rows(b, me);
    let walls = b.legal_wall_masks_restricted((c0.0 | c1.0) & fwd, (c0.1 | c1.1) & fwd);
    Some((walls, [c0, c1]))
}

#[inline(always)]
fn wall_action(horizontal: bool, w: usize) -> u8 {
    if horizontal { HWALL_BASE + w as u8 } else { VWALL_BASE + w as u8 }
}

/// (並べ替えキー, 手) の固定長リスト（容量 N）。キーの降順・同点は追加順に並べる。
/// ノードごとに作るので、ゼロ初期化はしない。
struct Scored<const N: usize> {
    items: [std::mem::MaybeUninit<(i64, u8)>; N],
    len: usize,
}

impl<const N: usize> Scored<N> {
    #[inline(always)]
    fn new() -> Self {
        Scored { items: [std::mem::MaybeUninit::uninit(); N], len: 0 }
    }
    #[inline(always)]
    fn push(&mut self, key: i64, mv: u8) {
        self.items[self.len].write((key, mv));
        self.len += 1;
    }
    #[inline(always)]
    fn slice_mut(&mut self) -> &mut [(i64, u8)] {
        // SAFETY: items[..len] は push で初期化済み
        unsafe { std::slice::from_raw_parts_mut(self.items.as_mut_ptr() as *mut (i64, u8), self.len) }
    }
    #[inline(always)]
    fn sort(&mut self) {
        let s = self.slice_mut();
        if s.len() <= 16 {
            // 少数なら挿入ソート（安定: 同点は追加順）
            for i in 1..s.len() {
                let x = s[i];
                let mut j = i;
                while j > 0 && s[j - 1].0 < x.0 {
                    s[j] = s[j - 1];
                    j -= 1;
                }
                s[j] = x;
            }
            return;
        }
        // 追加順を保つ（安定ソート相当）ため、同点は元の位置で比べる
        for (i, it) in s.iter_mut().enumerate() {
            it.0 = it.0.saturating_mul(512) - i as i64;
        }
        s.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    }
    #[inline(always)]
    fn moves(&self) -> impl Iterator<Item = u8> + '_ {
        // SAFETY: 同上
        unsafe { std::slice::from_raw_parts(self.items.as_ptr() as *const (i64, u8), self.len) }.iter().map(|x| x.1)
    }
}

/// コマの手は最大 5 手、壁は最大 128 手、ルートは両方
type PawnList = Scored<8>;
type WallList = Scored<128>;
type RootList = Scored<136>;

/// 最短距離のキャッシュのエントリ数（2 の冪）
const DIST_CACHE: usize = 1 << 12;
/// 最短経路の切断マスクのキャッシュのエントリ数（2 の冪）
const CUT_CACHE: usize = 1 << 12;
const NO_DIST: u8 = 255;

/// スレッドごとに持ち越す状態（Python 版ではモジュール全体の killer_moves / history / 距離キャッシュ）
#[derive(Clone)]
struct ThreadState {
    killers: [[u8; 2]; MAX_DEPTH],
    history: [i32; ACTION_COUNT],
    /// Board::dist_key → 最短距離（NO_DIST = 到達不能）。距離は壁とコマの位置だけで決まるので対局を通して有効
    dist: Vec<(u64, u8)>,
    /// Board::dist_key → 最短経路を切る壁 (水平, 垂直)
    cuts: Vec<(u64, u64, u64)>,
}

impl ThreadState {
    fn new() -> Self {
        ThreadState {
            killers: [[NO_MOVE; 2]; MAX_DEPTH],
            history: [0; ACTION_COUNT],
            dist: vec![(0, 0); DIST_CACHE],
            cuts: vec![(0, 0, 0); CUT_CACHE],
        }
    }
}

/// スレッド間で共有する停止フラグと、いちばん深く探索し終えた結果
struct Shared {
    stop: AtomicBool,
    /// (完了した深さ, 最善手, 評価値)
    best: Mutex<(u32, u8, i32)>,
}

struct Worker<'a> {
    tt: &'a Tt,
    gen: u8,
    shared: &'a Shared,
    deadline: Option<Instant>,
    is_main: bool,
    nodes: u64,
    stopped: bool,
    st: &'a mut ThreadState,
}

impl Worker<'_> {
    /// player の最短距離（キャッシュ付き）
    #[inline(always)]
    fn dist(&mut self, b: &Board, player: usize) -> Option<u32> {
        let k = b.dist_key(player);
        let e = &mut self.st.dist[k as usize & (DIST_CACHE - 1)];
        if e.0 == k {
            return if e.1 == NO_DIST { None } else { Some(e.1 as u32) };
        }
        let d = b.shortest_path(player);
        *e = (k, d.map_or(NO_DIST, |v| v as u8));
        d
    }

    /// player の最短経路を切る壁（キャッシュ付き）
    #[inline(always)]
    fn cuts(&mut self, b: &Board, player: usize) -> Option<(u64, u64)> {
        let k = b.dist_key(player);
        let e = &mut self.st.cuts[k as usize & (CUT_CACHE - 1)];
        if e.0 == k {
            return Some((e.1, e.2));
        }
        let c = b.path_cut_masks(player)?;
        *e = (k, c.0, c.1);
        Some(c)
    }

    #[inline(always)]
    fn evaluate(&mut self, b: &Board) -> i32 {
        let me = b.turn as usize;
        let (md, ed) = (self.dist(b, me), self.dist(b, me ^ 1));
        evaluate_with(b, md, ed)
    }

    /// candidate_walls のキャッシュ付き版
    fn candidate_walls(&mut self, b: &Board) -> Option<((u64, u64), [(u64, u64); 2])> {
        crate::stat!(CANDIDATES);
        let me = b.turn as usize;
        if b.walls[me] == 0 {
            return None;
        }
        let c0 = self.cuts(b, 0)?;
        let c1 = self.cuts(b, 1)?;
        let fwd = wall_rows(b, me);
        let walls = b.legal_wall_masks_restricted((c0.0 | c1.0) & fwd, (c0.1 | c1.1) & fwd);
        Some((walls, [c0, c1]))
    }

    #[inline(always)]
    fn poll(&mut self) {
        if self.is_main {
            if let Some(d) = self.deadline {
                if Instant::now() >= d {
                    self.shared.stop.store(true, Ordering::Relaxed);
                }
            }
        }
        if self.shared.stop.load(Ordering::Relaxed) {
            self.stopped = true;
        }
    }

    fn alphabeta(&mut self, b: &mut Board, depth: i32, mut alpha: i32, mut beta: i32, null_ok: bool) -> i32 {
        self.nodes += 1;
        if self.nodes & 1023 == 0 {
            self.poll();
        }
        if self.stopped {
            return 0;
        }
        let orig_alpha = alpha;

        // 千日手
        if crate::timed!(CYC_REP, b.repetitions_upto(2)) >= 2 {
            return 0;
        }
        // 終局
        if let Some(w) = b.winner() {
            return if w == b.turn { MATE + depth } else { -(MATE + depth) };
        }
        if depth <= 0 {
            return crate::timed!(CYC_EVAL, self.evaluate(b));
        }

        // 置換表
        let key = b.hash;
        let mut tt_move = NO_MOVE;
        if let Some(e) = crate::timed!(CYC_TT, self.tt.probe(key)) {
            if self.gen.wrapping_sub(e.gen) <= GEN_WINDOW {
                if e.mv != NO_MOVE && b.is_legal(e.mv) {
                    tt_move = e.mv;
                }
                if e.depth as i32 >= depth {
                    match e.flag {
                        EXACT => return e.score,
                        LOWER => alpha = alpha.max(e.score),
                        _ => beta = beta.min(e.score),
                    }
                    if alpha >= beta {
                        return e.score;
                    }
                }
            }
        }

        let me = b.turn as usize;
        let (my_walls, enemy_walls) = (b.walls[me], b.walls[me ^ 1]);
        let di = (depth as usize).min(MAX_DEPTH - 1);
        let d8 = depth.min(255) as u8;

        // ヌルムーブ枝刈り
        if null_ok && depth >= NMP_DEPTH_MIN && my_walls >= NMP_WALL_MIN && enemy_walls >= NMP_WALL_MIN && beta < MATE {
            b.make_pass();
            let ns = -self.alphabeta(b, depth - NMP_REDUCTION - 1, -beta, -beta + 1, false);
            b.undo();
            if self.stopped {
                return 0;
            }
            if ns >= beta {
                return beta;
            }
        }

        let mut best = -INF;
        let mut best_mv = NO_MOVE;

        macro_rules! cutoff {
            ($mv:expr, $killer:expr) => {{
                if $killer {
                    let k = &mut self.st.killers[di];
                    k[1] = k[0];
                    k[0] = $mv;
                }
                let h = &mut self.st.history[$mv as usize];
                *h += depth * depth;
                if *h > HISTORY_MAX {
                    for h in self.st.history.iter_mut() {
                        *h >>= 1;
                    }
                }
                self.tt.store(key, self.gen, d8, LOWER, best, best_mv);
                return best;
            }};
        }

        // フェーズ 1: 置換表の手（フルウィンドウ）
        if tt_move != NO_MOVE {
            b.make(tt_move);
            let s = -self.alphabeta(b, depth - 1, -beta, -alpha, true);
            b.undo();
            if self.stopped {
                return 0;
            }
            if s > best {
                best = s;
                best_mv = tt_move;
            }
            if s > alpha {
                alpha = s;
            }
            if alpha >= beta {
                cutoff!(tt_move, false);
            }
        }

        // フェーズ 2: 移動手（PVS + LMR）
        #[cfg(feature = "stats")]
        let __t = unsafe { core::arch::x86_64::_rdtsc() };
        let pawn_mask = b.pawn_dest_mask();
        let mut pawns = PawnList::new();
        {
            let k = self.st.killers[di];
            let mut m = pawn_mask;
            while m != 0 {
                let mv = m.trailing_zeros() as u8;
                m &= m - 1;
                if mv == tt_move {
                    continue;
                }
                let key = if mv == k[0] {
                    900_000
                } else if mv == k[1] {
                    800_000
                } else {
                    50_000 + self.st.history[mv as usize] as i64
                };
                pawns.push(key, mv);
            }
        }
        pawns.sort();
        #[cfg(feature = "stats")]
        crate::stats::counters::CYC_PAWNGEN.fetch_add(unsafe { core::arch::x86_64::_rdtsc() } - __t, Ordering::Relaxed);

        let mut is_first = true;
        for (i, mv) in pawns.moves().enumerate() {
            b.make(mv);
            let s = if is_first {
                is_first = false;
                -self.alphabeta(b, depth - 1, -beta, -alpha, true)
            } else if i >= 3 && depth >= 3 {
                let mut s = -self.alphabeta(b, depth - 2, -alpha - 1, -alpha, true);
                if s > alpha {
                    s = -self.alphabeta(b, depth - 1, -alpha - 1, -alpha, true);
                    if alpha < s && s < beta {
                        s = -self.alphabeta(b, depth - 1, -beta, -alpha, true);
                    }
                }
                s
            } else {
                let mut s = -self.alphabeta(b, depth - 1, -alpha - 1, -alpha, true);
                if alpha < s && s < beta {
                    s = -self.alphabeta(b, depth - 1, -beta, -alpha, true);
                }
                s
            };
            b.undo();
            if self.stopped {
                return 0;
            }
            if s > best {
                best = s;
                best_mv = mv;
            }
            if s > alpha {
                alpha = s;
            }
            if alpha >= beta {
                cutoff!(mv, true);
            }
        }

        // フェーズ 3: 壁手（PVS のみ）
        let cand = if my_walls == 0 { None } else { crate::timed!(CYC_CAND, self.candidate_walls(b)) };
        let ((mut hm, mut vm), cuts) = match cand {
            Some(c) => c,
            None => ((0, 0), [(0, 0); 2]),
        };
        if pawn_mask == 0 && hm | vm == 0 {
            return self.evaluate(b);
        }
        if tt_move >= HWALL_BASE {
            if tt_move < VWALL_BASE {
                hm &= !(1u64 << (tt_move - HWALL_BASE));
            } else {
                vm &= !(1u64 << (tt_move - VWALL_BASE));
            }
        }
        if hm | vm != 0 {
            #[cfg(feature = "stats")]
            let __t = unsafe { core::arch::x86_64::_rdtsc() };
            let mut walls = WallList::new();
            let k = self.st.killers[di];
            let ctx = if depth >= WALL_ORDER_DEPTH {
                let dist = [self.dist(b, 0), self.dist(b, 1)];
                Some(WallScoreCtx::with_dist(b, cuts, dist))
            } else {
                None
            };
            for (horizontal, mut m) in [(true, hm), (false, vm)] {
                while m != 0 {
                    let w = m.trailing_zeros() as usize;
                    m &= m - 1;
                    let mv = wall_action(horizontal, w);
                    let key = if mv == k[0] {
                        900_000
                    } else if mv == k[1] {
                        800_000
                    } else if let Some(ctx) = &ctx {
                        (wall_score(b, horizontal, w, ctx) * 1000.0) as i64 + self.st.history[mv as usize] as i64
                    } else {
                        self.st.history[mv as usize] as i64
                    };
                    walls.push(key, mv);
                }
            }
            walls.sort();
            #[cfg(feature = "stats")]
            crate::stats::counters::CYC_ORDER.fetch_add(unsafe { core::arch::x86_64::_rdtsc() } - __t, Ordering::Relaxed);

            for mv in walls.moves() {
                b.make(mv);
                let s = if is_first {
                    is_first = false;
                    -self.alphabeta(b, depth - 1, -beta, -alpha, true)
                } else {
                    let mut s = -self.alphabeta(b, depth - 1, -alpha - 1, -alpha, true);
                    if alpha < s && s < beta {
                        s = -self.alphabeta(b, depth - 1, -beta, -alpha, true);
                    }
                    s
                };
                b.undo();
                if self.stopped {
                    return 0;
                }
                if s > best {
                    best = s;
                    best_mv = mv;
                }
                if s > alpha {
                    alpha = s;
                }
                if alpha >= beta {
                    cutoff!(mv, true);
                }
            }
        }

        let flag = if best <= orig_alpha {
            UPPER
        } else if best >= beta {
            LOWER
        } else {
            EXACT
        };
        self.tt.store(key, self.gen, d8, flag, best, best_mv);
        best
    }

    /// ルートの手順付け（Python 版 _order_moves。壁の向きのバグは修正済み）
    fn order_root(&self, b: &Board, moves: &[u8], d: i32, tt_move: u8, cuts: [(u64, u64); 2]) -> RootList {
        let me = b.turn as usize;
        let near = if d >= WALL_ORDER_DEPTH {
            let mut m = 0u64;
            if let Some(path) = b.shortest_path_nodes(me ^ 1) {
                for &n in &path {
                    m |= NEAR[n as usize];
                }
            }
            Some(m)
        } else {
            None
        };
        let ctx = WallScoreCtx::new(b, cuts);
        let k = self.st.killers[(d as usize).min(MAX_DEPTH - 1)];
        let mut out = RootList::new();
        for &mv in moves {
            let key = if mv == tt_move {
                1_000_000
            } else if mv == k[0] {
                900_000
            } else if mv == k[1] {
                800_000
            } else if mv < HWALL_BASE {
                50_000 + self.st.history[mv as usize] as i64
            } else {
                let (horizontal, w) = if mv < VWALL_BASE {
                    (true, (mv - HWALL_BASE) as usize)
                } else {
                    (false, (mv - VWALL_BASE) as usize)
                };
                let ws = match near {
                    Some(n) if n >> w & 1 == 0 => None,
                    _ => Some(wall_score(b, horizontal, w, &ctx)),
                };
                let base = match ws {
                    Some(ws) => (ws * 1000.0) as i64,
                    None => -10_000,
                };
                base + self.st.history[mv as usize] as i64
            };
            out.push(key, mv);
        }
        out.sort();
        out
    }

    /// 反復深化（Python 版 best_move）。戻り値: (最善手, 評価値, 完了した深さ)
    fn iterate(&mut self, b: &mut Board, moves: &[u8], cuts: [(u64, u64); 2], max_depth: u32, lead: u32) -> (u8, i32, u32) {
        let mut best_mv = NO_MOVE;
        let mut prev_score = 0;
        let mut completed = 0;
        for iter_d in 1..=max_depth {
            // 補助スレッドは 1 手先の深さを探索して置換表を先に埋める
            let d = (iter_d + lead).min(max_depth.max(iter_d)) as i32;
            let orig_alpha = (prev_score - ASPIRATION_WINDOW).max(-INF);
            let orig_beta = (prev_score + ASPIRATION_WINDOW).min(INF);
            let (mut alpha, mut beta) = (orig_alpha, orig_beta);
            let mut window = ASPIRATION_WINDOW;
            let mut retries = 0;
            let (best_score, current_best) = loop {
                let tt_move = if best_mv != NO_MOVE {
                    best_mv
                } else {
                    self.tt.probe(b.hash).map_or(NO_MOVE, |e| e.mv)
                };
                let ordered = self.order_root(b, moves, d, tt_move, cuts);
                let mut best_score = -INF;
                let mut current_best = NO_MOVE;
                let loop_alpha = alpha;
                let mut is_first = true;
                for mv in ordered.moves() {
                    b.make(mv);
                    let s = if is_first {
                        is_first = false;
                        -self.alphabeta(b, d - 1, -beta, -alpha, true)
                    } else {
                        let mut s = -self.alphabeta(b, d - 1, -alpha - 1, -alpha, true);
                        if alpha < s && s < beta {
                            s = -self.alphabeta(b, d - 1, -beta, -alpha, true);
                        }
                        s
                    };
                    b.undo();
                    if self.stopped {
                        break;
                    }
                    if s > best_score {
                        best_score = s;
                        current_best = mv;
                    }
                    if s > alpha {
                        alpha = s;
                    }
                }
                if self.stopped {
                    return (best_mv, prev_score, completed);
                }
                let fail_low = best_score <= loop_alpha;
                let fail_high = best_score >= beta;
                if !(fail_low || fail_high) {
                    break (best_score, current_best);
                }
                retries += 1;
                if retries > MAX_RETRIES {
                    if alpha == -INF && beta == INF {
                        break (best_score, current_best);
                    }
                    alpha = -INF;
                    beta = INF;
                    continue;
                }
                window = (window * 2).min(INF);
                if fail_low {
                    alpha = (prev_score - window).max(-INF);
                    beta = orig_beta;
                } else {
                    alpha = orig_alpha;
                    beta = (prev_score + window).min(INF);
                }
            };
            prev_score = best_score;
            if current_best != NO_MOVE {
                best_mv = current_best;
            }
            completed = d as u32;
            // どのスレッドでも、より深く探索し終えた結果を採用する
            let finished = best_score >= MATE || d as u32 >= max_depth;
            {
                let mut sb = self.shared.best.lock().unwrap();
                if d as u32 > sb.0 && best_mv != NO_MOVE {
                    *sb = (d as u32, best_mv, best_score);
                }
            }
            if finished {
                self.shared.stop.store(true, Ordering::Relaxed);
                break;
            }
        }
        (best_mv, prev_score, completed)
    }
}

/// 探索の条件
#[derive(Clone, Debug)]
pub struct Limits {
    /// 最大の深さ（時間制限だけにしたいときは大きな値）
    pub max_depth: u32,
    /// 時間制限（None なら深さのみ）
    pub time: Option<Duration>,
    /// スレッド数（1 なら単一スレッド）
    pub threads: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { max_depth: 4, time: None, threads: 1 }
    }
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    /// 最善手（合法手がなければ None）
    pub best: Option<u8>,
    /// 手番側から見た評価値（100 倍の整数）
    pub score: i32,
    /// 完了した深さ
    pub depth: u32,
    /// 全スレッドの探索ノード数
    pub nodes: u64,
    pub elapsed: Duration,
}

/// 置換表・キラー手・history を対局を通して持ち越す探索器
pub struct Searcher {
    tt: Arc<Tt>,
    gen: u8,
    states: Vec<ThreadState>,
}

impl Searcher {
    pub fn new(tt_mb: usize) -> Self {
        Searcher { tt: Arc::new(Tt::new(tt_mb)), gen: 0, states: vec![ThreadState::new()] }
    }

    /// 置換表・キラー手・history を消す（Python 版 clear_tt）
    pub fn clear(&mut self) {
        self.tt.clear();
        self.gen = 0;
        for s in &mut self.states {
            *s = ThreadState::new();
        }
    }

    pub fn search(&mut self, board: &Board, limits: &Limits) -> SearchResult {
        let start = Instant::now();
        let mut root = board.clone();

        // ルートの候補手（移動手 + 絞り込んだ壁）
        let mut moves: Vec<u8> = Vec::with_capacity(160);
        let mut m = root.pawn_dest_mask();
        while m != 0 {
            moves.push(m.trailing_zeros() as u8);
            m &= m - 1;
        }
        let mut cuts = [(0, 0); 2];
        if let Some(((mut h, mut v), c)) = candidate_walls(&root) {
            cuts = c;
            while h != 0 {
                moves.push(HWALL_BASE + h.trailing_zeros() as u8);
                h &= h - 1;
            }
            while v != 0 {
                moves.push(VWALL_BASE + v.trailing_zeros() as u8);
                v &= v - 1;
            }
        } else {
            if let (Some(c0), Some(c1)) = (root.path_cut_masks(0), root.path_cut_masks(1)) {
                cuts = [c0, c1];
            }
        }
        let done = |best: Option<u8>| SearchResult { best, score: 0, depth: 0, nodes: 0, elapsed: start.elapsed() };
        if moves.is_empty() {
            return done(None);
        }
        if moves.len() == 1 {
            return done(Some(moves[0]));
        }

        // 新しい世代（Python 版 _new_generation: history を半減）
        self.gen = self.gen.wrapping_add(1);
        let threads = limits.threads.max(1);
        while self.states.len() < threads {
            self.states.push(ThreadState::new());
        }
        for s in &mut self.states {
            for h in s.history.iter_mut() {
                *h >>= 1;
            }
        }

        let shared = Shared { stop: AtomicBool::new(false), best: Mutex::new((0, NO_MOVE, 0)) };
        // 桁あふれするほど長い時間は制限なしとみなす
        let deadline = limits.time.and_then(|t| start.checked_add(t));
        let max_depth = limits.max_depth.clamp(1, MAX_DEPTH as u32 - 2);
        let (tt, gen) = (&*self.tt, self.gen);
        let (main_state, helper_states) = self.states[..threads].split_first_mut().unwrap();

        let nodes = std::thread::scope(|s| {
            let handles: Vec<_> = helper_states
                .iter_mut()
                .enumerate()
                .map(|(i, st)| {
                    let mut b = root.clone();
                    let (shared, moves) = (&shared, &moves);
                    s.spawn(move || {
                        let mut w = Worker { tt, gen, shared, deadline: None, is_main: false, nodes: 0, stopped: false, st };
                        // 半数の補助スレッドは 1 つ深い深さを先に探索する
                        w.iterate(&mut b, moves, cuts, max_depth, (i as u32 + 1) & 1);
                        w.nodes
                    })
                })
                .collect();
            let mut w = Worker { tt, gen, shared: &shared, deadline, is_main: true, nodes: 0, stopped: false, st: main_state };
            crate::timed!(CYC_TOTAL, w.iterate(&mut root, &moves, cuts, max_depth, 0));
            shared.stop.store(true, Ordering::Relaxed);
            w.nodes + handles.into_iter().map(|h| h.join().unwrap()).sum::<u64>()
        });

        let (depth, mv, score) = *shared.best.lock().unwrap();
        // 深さ 1 も終わらないうちに時間切れになった場合は候補の先頭を返す
        let best = if mv == NO_MOVE { Some(moves[0]) } else { Some(mv) };
        SearchResult { best, score, depth, nodes, elapsed: start.elapsed() }
    }
}

/// 複数スレッドで perft を数える（ルートの手をスレッドに分配する）
pub fn perft_parallel(board: &Board, depth: u32, threads: usize, bulk: bool) -> u64 {
    if depth <= 1 || threads <= 1 {
        let mut b = board.clone();
        return if bulk { b.perft_bulk(depth) } else { b.perft(depth) };
    }
    let moves: Vec<u8> = board.legal_actions().as_slice().to_vec();
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let (next, moves) = (&next, &moves);
                let mut b = board.clone();
                s.spawn(move || {
                    let mut n = 0;
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= moves.len() {
                            return n;
                        }
                        b.make(moves[i]);
                        n += if bulk { b.perft_bulk(depth - 1) } else { b.perft(depth - 1) };
                        b.undo();
                    }
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_walls_after_passing() {
        // P0 は 6 行目、P1 は 2 行目（追い越された後）。P1 のゴール前（y <= 1）の壁も候補に入る
        let b = Board::from_parts([58, 22], [10, 10], 0, 0, 0);
        let ((h, _), _) = candidate_walls(&b).unwrap();
        assert_ne!(h & 0xFFFF, 0, "P1 のゴール前の水平壁が候補にない: {h:#x}");
        // 追い越す前は従来どおり前方だけ
        let b = Board::from_parts([22, 58], [10, 10], 0, 0, 0);
        let ((h, v), _) = candidate_walls(&b).unwrap();
        assert_eq!((h | v) & 0xFF, 0);
    }

    #[test]
    fn history_stays_bounded() {
        let mut b = Board::new();
        for a in [13, 67, 22, 58] {
            b.make(a);
        }
        let mut s = Searcher::new(16);
        let lim = Limits { max_depth: 60, time: Some(Duration::from_millis(300)), threads: 1 };
        for _ in 0..3 {
            s.search(&b, &lim);
        }
        let max = s.states[0].history.iter().copied().max().unwrap();
        assert!(max <= HISTORY_MAX, "{max}");
    }

    #[test]
    fn huge_time_limit_does_not_panic() {
        let b = Board::new();
        let r = Searcher::new(1).search(&b, &Limits { max_depth: 2, time: Some(Duration::MAX), threads: 1 });
        assert!(b.is_legal(r.best.unwrap()));
    }
}
