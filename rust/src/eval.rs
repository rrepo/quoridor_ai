//! 評価関数と壁の手順付け用スコア（ai/eval_func.py・ai/wall_evaluation.py の移植）
//!
//! 評価値は Python 版の 100 倍の整数（小数の重みを保ったまま、探索窓を整数で扱うため）。

use crate::board::Board;
use crate::consts::GOAL;
use crate::path::distance;

/// 勝ち（Python 版の 9000.0 に相当）。終局スコアは MATE + 残り深さ。
pub const MATE: i32 = 900_000;
/// 探索の無限大（Python 版の 99999 に相当）
pub const INF: i32 = 9_999_900;

const SCALE: f32 = 100.0;

const W_DIST_DIFF: f32 = 10.0;
const W_PROGRESS: f32 = 5.0;
const W_ENEMY_PROGRESS: f32 = 1.5;
const W_WALL_DIFF: f32 = 0.3;
const W_RACE_BOOST: f32 = 15.0;
const W_FUNNEL: f32 = 2.0;
const W_CENTER: f32 = 0.4;
const W_NO_WALL_BONUS: f32 = 5.0;
const W_WALL_POS: f32 = 0.3;

const RACE_WALL_THRESH: u8 = 3;
const RACE_DIST_THRESH: i32 = 5;
const FUNNEL_THRESH: i32 = 4;

/// 手番側から見た評価値
pub fn evaluate(b: &Board) -> i32 {
    let me = b.turn as usize;
    evaluate_with(b, b.shortest_path(me), b.shortest_path(me ^ 1))
}

/// 両者の最短距離（手番側, 相手）を与えて評価する
pub fn evaluate_with(b: &Board, my_dist: Option<u32>, enemy_dist: Option<u32>) -> i32 {
    crate::stat!(EVAL);
    let me = b.turn as usize;
    let en = me ^ 1;
    let my_dist = match my_dist {
        Some(d) => d as i32,
        None => return -MATE,
    };
    let enemy_dist = match enemy_dist {
        Some(d) => d as i32,
        None => return MATE,
    };
    let (my_walls, enemy_walls) = (b.walls[me], b.walls[en]);
    let (my_row, enemy_row) = ((b.pos[me] / 9) as f32, (b.pos[en] / 9) as f32);
    let (my_col, enemy_col) = ((b.pos[me] % 9) as f32, (b.pos[en] % 9) as f32);

    // 1. 距離差
    let dist_weight = W_DIST_DIFF + if enemy_walls == 0 { W_NO_WALL_BONUS } else { 0.0 };
    let dist_diff = enemy_dist - my_dist;
    let mut score = dist_diff as f32 * dist_weight;

    // 2. 進行度
    let (my_progress, enemy_progress) = if me == 0 {
        (my_row / 8.0, (8.0 - enemy_row) / 8.0)
    } else {
        ((8.0 - my_row) / 8.0, enemy_row / 8.0)
    };
    score += my_progress * W_PROGRESS;
    score -= enemy_progress * W_ENEMY_PROGRESS;

    // 3. 壁残数差
    score += (my_walls as f32 - enemy_walls as f32) * W_WALL_DIFF;

    // 4. 終盤の競走
    if my_walls <= RACE_WALL_THRESH && enemy_walls <= RACE_WALL_THRESH && dist_diff.abs() <= RACE_DIST_THRESH {
        score += dist_diff as f32 * W_RACE_BOOST;
    }

    // 5. 閉じ込め
    if dist_diff > FUNNEL_THRESH {
        score += (dist_diff - FUNNEL_THRESH) as f32 * W_FUNNEL;
    }

    // 6. センター
    score += (1.0 - (my_col - 4.0).abs() / 4.0) * W_CENTER;
    score -= (1.0 - (enemy_col - 4.0).abs() / 4.0) * W_CENTER * 0.5;

    // 7. 壁の位置（P0 から見て Σ(2y/7 - 1) = 2/7·Σy − 枚数。P1 なら符号反転）
    if b.wall_count != 0 {
        let total = 2.0 * b.wall_ysum as f32 / 7.0 - b.wall_count as f32;
        score += if me == 0 { total } else { -total } * W_WALL_POS;
    }

    // 四捨五入（f32::round と同じく 0 から遠い方へ）。round() は命令セットによっては関数呼び出しになるため
    let x = score * SCALE;
    (x + if x >= 0.0 { 0.5 } else { -0.5 }) as i32
}

/// 壁の手順付け用の文脈（ノードごとに 1 回計算する）
pub struct WallScoreCtx {
    pub me: usize,
    /// 両プレイヤーの最短経路を切る壁 (水平, 垂直)
    pub cuts: [(u64, u64); 2],
    /// 両プレイヤーの現在の最短距離
    pub dist: [Option<u32>; 2],
}

impl WallScoreCtx {
    pub fn new(b: &Board, cuts: [(u64, u64); 2]) -> Self {
        Self::with_dist(b, cuts, [b.shortest_path(0), b.shortest_path(1)])
    }
    pub fn with_dist(b: &Board, cuts: [(u64, u64); 2], dist: [Option<u32>; 2]) -> Self {
        WallScoreCtx { me: b.turn as usize, cuts, dist }
    }
}

/// 壁 w（水平なら horizontal）を手番側が置いたときのスコア（Python 版 wall_score、向きのバグは修正済み）
///
/// 相手の距離の伸び × 1.5 − 自分の距離の伸び。最短経路を切らない壁は距離が変わらないので BFS しない。
pub fn wall_score(b: &Board, horizontal: bool, w: usize, ctx: &WallScoreCtx) -> f32 {
    crate::stat!(WALL_SCORE);
    let (me, en) = (ctx.me, ctx.me ^ 1);
    let bit = 1u64 << w;
    let cuts = |p: usize| if horizontal { ctx.cuts[p].0 & bit != 0 } else { ctx.cuts[p].1 & bit != 0 };
    let (d, r) = if horizontal {
        (b.open_d & !crate::consts::hwall_d_bits(w), b.open_r)
    } else {
        (b.open_d, b.open_r & !crate::consts::vwall_r_bits(w))
    };
    let after = |p: usize| distance(1u128 << b.pos[p], GOAL[p], d, r);
    let delta = |p: usize| match (ctx.dist[p], after(p)) {
        (Some(before), Some(a)) => a as f32 - before as f32,
        _ => 0.0,
    };
    let mut score = 0.0;
    if cuts(en) {
        score += delta(en) * 1.5;
    }
    if cuts(me) {
        score -= delta(me);
    }
    score
}
