//! 探索（legacy/ai/search.py の移植 + Lazy SMP による並列化 + 時間制限）
//!
//! 反復深化 + aspiration window、PVS、移動手の LMR、ヌルムーブ枝刈り、置換表、
//! キラー手・history による手順付けは Python 版と同じ構成。
//! Python 版からの変更点:
//!   - 評価値は 100 倍の整数（探索窓の幅 1 が正確な意味を持つ）
//!   - 壁の手順付けで向きを取り違えていたバグ（"hwall" を 'h' と比較していた）を修正
//!   - 複数スレッド（置換表を共有し、各スレッドが同じ局面を探索する Lazy SMP）。
//!     補助スレッドは反復の深さをずらし（SKIP_SIZE / SKIP_PHASE）、他のスレッドが探索中の
//!     子局面は後回しにして（ABDADA）、スレッド間で同じ部分木を重複して探索しにくくしている。
//!     補助スレッドは Searcher が常駐させて使い回す（探索のたびにスレッドを作らない）
//!   - 時間制限（反復深化の途中で打ち切り、最後に完了した深さの手を返す）
//!   - 壁の候補の範囲: 相手に並ばれた・追い越された後は相手のゴール側も含める（wall_rows）
//!   - history に上限を設ける（HISTORY_MAX。長い探索でキラー手や前回の最善手を追い越さないように）

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::board::Board;
use crate::consts::*;
use crate::eval::{evaluate_with, wall_score, wall_score_with, WallScoreCtx, INF, MATE};
use crate::path::distance;
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

/// 探索用に絞り込んだ壁（両者の最短経路を切る、wall_rows の範囲の合法な壁）と、その計算に使った切断マスク。
/// cuts(p) はプレイヤー p の最短経路の切断マスク（到達不能なら None）
#[inline]
fn candidate_walls_with(
    b: &Board,
    mut cuts: impl FnMut(usize) -> Option<(u64, u64)>,
) -> Option<((u64, u64), [(u64, u64); 2])> {
    crate::stat!(CANDIDATES);
    let me = b.turn as usize;
    if b.walls[me] == 0 {
        return None;
    }
    let c0 = cuts(0)?;
    let c1 = cuts(1)?;
    let fwd = wall_rows(b, me);
    let walls = b.legal_wall_masks_restricted((c0.0 | c1.0) & fwd, (c0.1 | c1.1) & fwd);
    Some((walls, [c0, c1]))
}

/// candidate_walls_with（切断マスクをその場で計算する。ルートで使う）
fn candidate_walls(b: &Board) -> Option<((u64, u64), [(u64, u64); 2])> {
    candidate_walls_with(b, |p| b.path_cut_masks(p))
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
///
/// 主スレッドの分は Searcher が、補助スレッドの分はそれぞれのスレッドが持つ（別々の場所に置かれるので、
/// スレッド間でキャッシュラインを共有しない）。
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
    /// いずれかのスレッドが探索中の局面のハッシュ（ABDADA。添字は hash & (BUSY_SIZE - 1)）。
    /// 単一スレッドでは使わないので空（busy_slot / is_busy は smp のときだけ呼ぶ）
    busy: Vec<AtomicU64>,
}

impl Shared {
    /// smp が偽（単一スレッド）なら探索中の局面の表（128KB）を作らない
    fn new(smp: bool) -> Self {
        Shared {
            stop: AtomicBool::new(false),
            best: Mutex::new((0, NO_MOVE, 0)),
            busy: if smp { (0..BUSY_SIZE).map(|_| AtomicU64::new(0)).collect() } else { Vec::new() },
        }
    }

    #[inline(always)]
    fn busy_slot(&self, key: u64) -> &AtomicU64 {
        &self.busy[key as usize & (BUSY_SIZE - 1)]
    }

    /// 他のスレッドが局面 key を探索中か
    #[inline(always)]
    fn is_busy(&self, key: u64) -> bool {
        self.busy_slot(key).load(Ordering::Relaxed) == key
    }
}

/// 探索中の局面の表のエントリ数（2 の冪）
const BUSY_SIZE: usize = 1 << 14;
/// 残り深さがこれ以上の局面だけ探索中として登録する（浅い局面は登録・確認の手間の方が大きい）
const BUSY_MIN_DEPTH: i32 = 3;

/// 補助スレッドが飛ばす反復の深さ（Stockfish の Lazy SMP と同じ規則）。
/// 補助スレッド i は ((深さ + PHASE) / SIZE) が奇数の深さを飛ばし、スレッドごとに別の深さを先に探索する。
const SKIP_SIZE: [u32; 20] = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4];
const SKIP_PHASE: [u32; 20] = [0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7];

/// 局面を探索中として登録し、Drop で登録を外す（ABDADA）
struct BusyGuard<'a> {
    slot: Option<(&'a AtomicU64, u64)>,
}

impl<'a> BusyGuard<'a> {
    #[inline(always)]
    fn new(shared: &'a Shared, key: u64, enable: bool) -> Self {
        if !enable {
            return BusyGuard { slot: None };
        }
        let slot = shared.busy_slot(key);
        // 他のスレッドが同じ局面を登録済みなら、そのスレッドの登録を消さないよう何もしない
        if slot.load(Ordering::Relaxed) == key {
            return BusyGuard { slot: None };
        }
        slot.store(key, Ordering::Relaxed);
        BusyGuard { slot: Some((slot, key)) }
    }
}

impl Drop for BusyGuard<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        if let Some((slot, key)) = self.slot {
            // 別の局面に上書きされていたら消さない
            let _ = slot.compare_exchange(key, 0, Ordering::Relaxed, Ordering::Relaxed);
        }
    }
}

/// 探索し終えずに後回しにした手（ABDADA）
struct Deferred {
    moves: [u8; 136],
    len: usize,
}

impl Deferred {
    #[inline(always)]
    fn new() -> Self {
        Deferred { moves: [0; 136], len: 0 }
    }
    #[inline(always)]
    fn push(&mut self, mv: u8) {
        self.moves[self.len] = mv;
        self.len += 1;
    }
    #[inline(always)]
    fn as_slice(&self) -> &[u8] {
        &self.moves[..self.len]
    }
}

struct Worker<'a> {
    tt: &'a Tt,
    gen: u8,
    shared: &'a Shared,
    deadline: Option<Instant>,
    is_main: bool,
    /// 複数スレッドで探索しているか（単一スレッドなら探索中の局面の登録を省く）
    smp: bool,
    nodes: u64,
    stopped: bool,
    st: &'a mut ThreadState,
}

impl Worker<'_> {
    /// player の最短距離（キャッシュ付き）
    #[inline(always)]
    fn dist(&mut self, b: &Board, player: usize) -> Option<u32> {
        self.dist_keyed(b.dist_key(player), b.pos[player], player, b.open_d, b.open_r)
    }

    /// dist_key が key の局面での player（位置 start）の最短距離（キャッシュ付き）。d / r はその局面の辺。
    /// 壁の手順付けで「壁を置いた後」の距離を求めるときにも使い、その壁を実際に置いた子局面の評価でも当たるようにする
    #[inline(always)]
    fn dist_keyed(&mut self, key: u64, start: u8, player: usize, d: u128, r: u128) -> Option<u32> {
        let e = &mut self.st.dist[key as usize & (DIST_CACHE - 1)];
        if e.0 == key {
            return if e.1 == NO_DIST { None } else { Some(e.1 as u32) };
        }
        let v = distance(1u128 << start, GOAL[player], d, r);
        *e = (key, v.map_or(NO_DIST, |v| v as u8));
        v
    }

    /// 壁を置いた子局面の距離を、距離のキャッシュに先に入れておく。
    /// 壁が player の最短経路（cuts）を切らなければ、その経路が残り、壁が増えて距離が縮むことはないので、距離は dist のまま。
    /// 子局面（多くは探索の末端）の評価で BFS をせずに済む
    #[inline(always)]
    fn seed_wall_child(&mut self, b: &Board, horizontal: bool, w: usize, cuts: &[(u64, u64); 2], dist: &[Option<u32>; 2]) {
        let bit = 1u64 << w;
        let wz = b.wall_zobrist(horizontal, w);
        for p in 0..2 {
            let cut = (if horizontal { cuts[p].0 } else { cuts[p].1 }) & bit != 0;
            if let (false, Some(d)) = (cut, dist[p]) {
                let key = b.dist_key(p) ^ wz;
                self.st.dist[key as usize & (DIST_CACHE - 1)] = (key, d as u8);
            }
        }
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
        candidate_walls_with(b, |p| self.cuts(b, p))
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

    /// 手 mv の子局面を PVS で探索する。first なら全幅、そうでなければヌルウィンドウで探索し、
    /// 窓に入ったら全幅で探索し直す。lmr なら最初のヌルウィンドウ探索を 1 手浅くする。
    /// defer が真で、他のスレッドがその子局面を探索中なら、探索せずに None を返す（呼び出し側で後回しにする）。
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn search_child(&mut self, b: &mut Board, mv: u8, depth: i32, alpha: i32, beta: i32, first: bool, lmr: bool, defer: bool) -> Option<i32> {
        b.make(mv);
        // 子局面が末端でなければ置換表を引くので、千日手・終局の判定をしている間に先読みしておく
        if depth > 1 {
            self.tt.prefetch(b.hash);
        }
        if defer && self.shared.is_busy(b.hash) {
            b.undo();
            return None;
        }
        let s = if first {
            -self.alphabeta(b, depth - 1, -beta, -alpha, true)
        } else {
            let mut s = if lmr {
                let s = -self.alphabeta(b, depth - 2, -alpha - 1, -alpha, true);
                if s > alpha {
                    -self.alphabeta(b, depth - 1, -alpha - 1, -alpha, true)
                } else {
                    s
                }
            } else {
                -self.alphabeta(b, depth - 1, -alpha - 1, -alpha, true)
            };
            if alpha < s && s < beta {
                s = -self.alphabeta(b, depth - 1, -beta, -alpha, true);
            }
            s
        };
        b.undo();
        Some(s)
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

        // 置換表・ヌルムーブで戻らず、実際に手を読む局面だけを探索中として登録する（ABDADA）。
        // 共有の表への書き込みは他のコアのキャッシュを無効にするので、すぐ戻る局面では行わない。
        // 登録はこの関数から戻るときに BusyGuard の Drop で外れる。
        let shared = self.shared;
        let _busy = BusyGuard::new(shared, key, self.smp && depth >= BUSY_MIN_DEPTH);

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
        // 子局面の値 s で最善手・窓を更新し、β カットなら戻る
        macro_rules! update {
            ($s:expr, $mv:expr, $killer:expr) => {{
                let s = $s;
                if self.stopped {
                    return 0;
                }
                if s > best {
                    best = s;
                    best_mv = $mv;
                }
                if s > alpha {
                    alpha = s;
                }
                if alpha >= beta {
                    cutoff!($mv, $killer);
                }
            }};
        }

        // フェーズ 1: 置換表の手（フルウィンドウ）
        if tt_move != NO_MOVE {
            let s = self.search_child(b, tt_move, depth, alpha, beta, true, false, false).unwrap();
            update!(s, tt_move, false);
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

        // 複数スレッドのとき、他のスレッドが探索中の子局面は後回しにする（最初の手は後回しにしない）
        let can_defer = self.smp && depth - 1 >= BUSY_MIN_DEPTH;
        let mut deferred = Deferred::new();
        // 置換表の手を全幅で読んだ後は、残りの手はすべてヌルウィンドウで読む（PVS）
        let mut is_first = tt_move == NO_MOVE;
        for (i, mv) in pawns.moves().enumerate() {
            let lmr = i >= 3 && depth >= 3;
            match self.search_child(b, mv, depth, alpha, beta, is_first, lmr, can_defer && !is_first) {
                Some(s) => {
                    is_first = false;
                    update!(s, mv, true);
                }
                None => deferred.push(mv),
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
        // 置換表の手（フェーズ 1 で探索済み）が壁なら候補から外す。NO_MOVE（255）は壁の番号の範囲外なので除く
        // （除かないと 1 << 110 になり、リリースビルドではシフト量が 46 に丸められて垂直壁 (6,5) が消えていた）
        if tt_move != NO_MOVE && tt_move >= HWALL_BASE {
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
            let dist = [self.dist(b, 0), self.dist(b, 1)];
            let ctx = if depth >= WALL_ORDER_DEPTH { Some(WallScoreCtx::with_dist(b, cuts, dist)) } else { None };
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
                        let wz = b.wall_zobrist(horizontal, w);
                        let ws = wall_score_with(b, horizontal, w, ctx, |p, d, r| {
                            self.dist_keyed(b.dist_key(p) ^ wz, b.pos[p], p, d, r)
                        });
                        (ws * 1000.0) as i64 + self.st.history[mv as usize] as i64
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
                let (horizontal, w) =
                    if mv < VWALL_BASE { (true, (mv - HWALL_BASE) as usize) } else { (false, (mv - VWALL_BASE) as usize) };
                self.seed_wall_child(b, horizontal, w, &cuts, &dist);
                match self.search_child(b, mv, depth, alpha, beta, is_first, false, can_defer && !is_first) {
                    Some(s) => {
                        is_first = false;
                        update!(s, mv, true);
                    }
                    None => deferred.push(mv),
                }
            }
        }

        // 後回しにした手（その間に他のスレッドが置換表を埋めているので、多くは置換表で即座に返る）
        for &mv in deferred.as_slice() {
            let s = self.search_child(b, mv, depth, alpha, beta, false, false, false).unwrap();
            update!(s, mv, true);
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

    /// 反復深化（Python 版 best_move）。idx はスレッド番号（0 = 主スレッド）。
    /// 戻り値: (最善手, 評価値, 完了した深さ)
    fn iterate(&mut self, b: &mut Board, moves: &[u8], cuts: [(u64, u64); 2], max_depth: u32, idx: usize) -> (u8, i32, u32) {
        let mut best_mv = NO_MOVE;
        let mut prev_score = 0;
        let mut completed = 0;
        for d in 1..=max_depth {
            // 補助スレッドは深さを飛ばしながら進み、主スレッドより先の深さで置換表を埋める（最大深さは飛ばさない）
            if idx > 0 && d < max_depth {
                let i = (idx - 1) % SKIP_SIZE.len();
                if ((d + SKIP_PHASE[i]) / SKIP_SIZE[i]) % 2 != 0 {
                    continue;
                }
            }
            let d = d as i32;
            let can_defer = self.smp && d - 1 >= BUSY_MIN_DEPTH;
            let orig_alpha = (prev_score - ASPIRATION_WINDOW).max(-INF);
            let orig_beta = (prev_score + ASPIRATION_WINDOW).min(INF);
            let (mut alpha, mut beta) = (orig_alpha, orig_beta);
            let mut window = ASPIRATION_WINDOW;
            let mut retries = 0;
            // 窓を超えた手（探索し直すときに先頭に置く）
            let mut fail_high_mv = NO_MOVE;
            let (best_score, current_best) = loop {
                let tt_move = if fail_high_mv != NO_MOVE {
                    fail_high_mv
                } else if best_mv != NO_MOVE {
                    best_mv
                } else {
                    self.tt.probe(b.hash).map_or(NO_MOVE, |e| e.mv)
                };
                let ordered = self.order_root(b, moves, d, tt_move, cuts);
                let mut best_score = -INF;
                let mut current_best = NO_MOVE;
                let loop_alpha = alpha;
                let mut is_first = true;
                let mut deferred = Deferred::new();
                'moves: {
                    // 子局面の値 s で最善手・窓を更新する。停止したとき・窓を超えたときは残りを探索しない
                    macro_rules! update {
                        ($s:expr, $mv:expr) => {{
                            let s = $s;
                            if self.stopped {
                                break 'moves;
                            }
                            if s > best_score {
                                best_score = s;
                                current_best = $mv;
                            }
                            if s > alpha {
                                alpha = s;
                            }
                            if alpha >= beta {
                                break 'moves;
                            }
                        }};
                    }
                    for mv in ordered.moves() {
                        match self.search_child(b, mv, d, alpha, beta, is_first, false, can_defer && !is_first) {
                            Some(s) => {
                                is_first = false;
                                update!(s, mv);
                            }
                            None => deferred.push(mv),
                        }
                    }
                    for &mv in deferred.as_slice() {
                        update!(self.search_child(b, mv, d, alpha, beta, false, false, false).unwrap(), mv);
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
                if fail_high {
                    fail_high_mv = current_best;
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
        Limits { max_depth: 4, time: None, threads: default_threads() }
    }
}

/// 既定のスレッド数（CPU の論理スレッド数）
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
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

/// 補助スレッドに渡す探索の依頼
struct Job {
    /// 探索の通し番号。完了の知らせに付けて返させ、前の探索の知らせと区別する
    id: u64,
    root: Board,
    moves: Arc<[u8]>,
    cuts: [(u64, u64); 2],
    max_depth: u32,
    tt: Arc<Tt>,
    gen: u8,
    shared: Arc<Shared>,
}

/// 常駐する補助スレッド。探索のたびにスレッドを作ると、短い探索ではその時間の方が長くなるため、
/// 作ったスレッドを Searcher が持ち続け、依頼を送って起こす（待っている間は CPU を使わない）。
struct Helper {
    /// Drop で先に閉じてスレッドのループを終わらせるため Option にしている
    jobs: Option<mpsc::Sender<Job>>,
    /// 探索が終わると (探索の通し番号, ノード数) が届く
    done: mpsc::Receiver<(u64, u64)>,
    handle: Option<JoinHandle<()>>,
}

impl Helper {
    /// idx はスレッド番号（1 から。反復の深さの飛ばし方を決める）
    fn spawn(idx: usize) -> Helper {
        let (jobs, rx) = mpsc::channel::<Job>();
        let (done_tx, done) = mpsc::channel::<(u64, u64)>();
        let handle = std::thread::Builder::new()
            .name(format!("search-helper-{idx}"))
            .spawn(move || helper_loop(idx, rx, done_tx))
            .expect("failed to spawn a search helper thread");
        Helper { jobs: Some(jobs), done, handle: Some(handle) }
    }

    /// スレッドが終了しているか（探索中のパニックでしか終了しない）
    fn is_dead(&self) -> bool {
        self.handle.as_ref().map_or(true, |h| h.is_finished())
    }

    fn send(&self, job: Job) {
        self.jobs.as_ref().unwrap().send(job).expect("search helper thread panicked");
    }

    /// 通し番号 id の探索が終わるのを待ち、ノード数を返す。
    /// 前の探索の知らせ（主スレッドがパニックして待たなかった分）が残っていたら読み捨てる。
    fn wait(&self, id: u64) -> u64 {
        loop {
            let (done_id, nodes) = self.done.recv().expect("search helper thread panicked");
            if done_id == id {
                return nodes;
            }
        }
    }
}

impl Drop for Helper {
    fn drop(&mut self) {
        // 送信側を閉じるとスレッドのループが終わるので、それを待つ
        self.jobs.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// 戻るとき（パニックで巻き戻るときも）に停止フラグを立てる。
/// 補助スレッドは時間制限を見ないので、立てずに Helper の Drop（join）に進むと待ち続けてしまう。
struct StopOnDrop<'a>(&'a AtomicBool);

impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// 補助スレッドの本体: 依頼が来るたびに探索し、ノード数を返す
fn helper_loop(idx: usize, jobs: mpsc::Receiver<Job>, done: mpsc::Sender<(u64, u64)>) {
    let mut st = ThreadState::new();
    let mut last_gen = 0u8;
    while let Ok(job) = jobs.recv() {
        // 世代が進んだ回数だけ history を半減する（使われなかった探索の分も含める）
        let shift = u32::from(job.gen.wrapping_sub(last_gen)).min(31);
        last_gen = job.gen;
        for h in st.history.iter_mut() {
            *h >>= shift;
        }
        let mut b = job.root;
        let mut w = Worker {
            tt: &*job.tt,
            gen: job.gen,
            shared: &*job.shared,
            deadline: None,
            is_main: false,
            smp: true,
            nodes: 0,
            stopped: false,
            st: &mut st,
        };
        w.iterate(&mut b, &*job.moves, job.cuts, job.max_depth, idx);
        let nodes = w.nodes;
        if done.send((job.id, nodes)).is_err() {
            break;
        }
    }
}

/// 置換表・キラー手・history を対局を通して持ち越す探索器
pub struct Searcher {
    tt: Arc<Tt>,
    gen: u8,
    /// 主スレッド（search を呼んだスレッド）の状態
    main: ThreadState,
    /// 常駐する補助スレッド（必要な数まで増やす）
    helpers: Vec<Helper>,
    /// 探索の通し番号（補助スレッドの完了の知らせを区別する）
    search_id: u64,
}

impl Searcher {
    pub fn new(tt_mb: usize) -> Self {
        Searcher { tt: Arc::new(Tt::new(tt_mb)), gen: 0, main: ThreadState::new(), helpers: Vec::new(), search_id: 0 }
    }

    /// 置換表・キラー手・history を消す（Python 版 clear_tt）。補助スレッドは終了させ、次の探索で作り直す。
    pub fn clear(&mut self) {
        self.tt.clear();
        self.gen = 0;
        self.main = ThreadState::new();
        self.helpers.clear();
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

        // 新しい世代（Python 版 _new_generation: history を半減。補助スレッドは依頼を受けたときに半減する）
        self.gen = self.gen.wrapping_add(1);
        for h in self.main.history.iter_mut() {
            *h >>= 1;
        }
        let threads = limits.threads.max(1);
        let n_helpers = threads - 1;
        // パニックで終了した補助スレッドは作り直す（catch_unwind で Searcher を使い回した場合）
        let alive = n_helpers.min(self.helpers.len());
        for (i, h) in self.helpers[..alive].iter_mut().enumerate() {
            if h.is_dead() {
                *h = Helper::spawn(i + 1);
            }
        }
        while self.helpers.len() < n_helpers {
            self.helpers.push(Helper::spawn(self.helpers.len() + 1));
        }
        self.search_id += 1;
        let id = self.search_id;

        let smp = threads > 1;
        let shared = Arc::new(Shared::new(smp));
        let _stop = StopOnDrop(&shared.stop);
        // 桁あふれするほど長い時間は制限なしとみなす
        let deadline = limits.time.and_then(|t| start.checked_add(t));
        let max_depth = limits.max_depth.clamp(1, MAX_DEPTH as u32 - 2);
        let (tt, gen) = (Arc::clone(&self.tt), self.gen);
        let moves: Arc<[u8]> = moves.into();

        for h in &self.helpers[..n_helpers] {
            h.send(Job {
                id,
                root: root.clone(),
                moves: Arc::clone(&moves),
                cuts,
                max_depth,
                tt: Arc::clone(&tt),
                gen,
                shared: Arc::clone(&shared),
            });
        }
        let mut w =
            Worker { tt: &*tt, gen, shared: &*shared, deadline, is_main: true, smp, nodes: 0, stopped: false, st: &mut self.main };
        crate::timed!(CYC_TOTAL, w.iterate(&mut root, &*moves, cuts, max_depth, 0));
        let main_nodes = w.nodes;
        // 補助スレッドを止めて、全員が探索を終えるのを待つ（次の探索までに置換表などへの書き込みを終わらせる）
        shared.stop.store(true, Ordering::Relaxed);
        let nodes = main_nodes + self.helpers[..n_helpers].iter().map(|h| h.wait(id)).sum::<u64>();

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
        let max = s.main.history.iter().copied().max().unwrap();
        assert!(max <= HISTORY_MAX, "{max}");
    }

    #[test]
    fn many_threads_search() {
        // 論理スレッド数より多くても、深さを飛ばす補助スレッド・後回し（ABDADA）を通って最大深さまで探索する
        let mut b = Board::new();
        for a in [13, 67, 22, 58] {
            b.make(a);
        }
        let mut s = Searcher::new(16);
        for threads in [2, 16, 24, 4] {
            let r = s.search(&b, &Limits { max_depth: 8, time: None, threads });
            assert!(b.is_legal(r.best.unwrap()) && r.depth == 8, "{threads}: {r:?}");
        }
        // 補助スレッドは常駐して使い回す（減らしても終了させない）
        assert_eq!(s.helpers.len(), 23);
        // clear で終了させ、次の探索で作り直す
        s.clear();
        assert!(s.helpers.is_empty());
        let r = s.search(&b, &Limits { max_depth: 6, time: Some(Duration::from_millis(500)), threads: 3 });
        assert!(b.is_legal(r.best.unwrap()) && s.helpers.len() == 2);
    }

    #[test]
    fn recovers_after_abandoned_search() {
        let b = Board::new();
        let mut s = Searcher::new(4);
        let lim = Limits { max_depth: 5, time: None, threads: 3 };
        assert!(b.is_legal(s.search(&b, &lim).best.unwrap()));
        // 主スレッドがパニックして完了を待たなかった状態: 依頼を送ったまま、完了の知らせを読まずに残す
        let shared = Arc::new(Shared::new(true));
        shared.stop.store(true, Ordering::Relaxed);
        s.helpers[0].send(Job {
            id: s.search_id,
            root: b.clone(),
            moves: Arc::from(b.legal_actions().as_slice()),
            cuts: [(0, 0); 2],
            max_depth: 5,
            tt: Arc::clone(&s.tt),
            gen: s.gen,
            shared,
        });
        // 補助スレッドがパニックで終了した状態
        let h = &mut s.helpers[1];
        h.jobs.take();
        h.handle.take().unwrap().join().unwrap();
        assert!(s.helpers[1].is_dead());
        // 次の探索は、残った知らせを読み捨て、終了したスレッドを作り直して最後まで探索する
        let r = s.search(&b, &lim);
        assert!(b.is_legal(r.best.unwrap()) && r.depth == 5, "{r:?}");
        assert!(!s.helpers[1].is_dead(), "終了した補助スレッドが作り直されていない");
    }

    #[test]
    fn huge_time_limit_does_not_panic() {
        let b = Board::new();
        let r = Searcher::new(1).search(&b, &Limits { max_depth: 2, time: Some(Duration::MAX), threads: 1 });
        assert!(b.is_legal(r.best.unwrap()));
    }
}
