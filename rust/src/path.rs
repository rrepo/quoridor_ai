//! ビットボード BFS
//!
//! 1 層の展開はシフト演算 4 回:
//!   下へ (f & D) << 9 / 上へ (f >> 9) & D / 右へ (f & R) << 1 / 左へ (f >> 1) & R

use crate::consts::{ALL_NODES, STATIC_NB};

/// bits（ノード集合）に隣接し、辺が通れるノードの集合
#[inline(always)]
pub fn expand(bits: u128, d: u128, r: u128) -> u128 {
    ((bits & d) << 9) | ((bits >> 9) & d) | ((bits & r) << 1) | ((bits >> 1) & r)
}

/// start（集合）から goal（集合）までの最短距離。到達不能なら None。
#[inline]
pub fn distance(start: u128, goal: u128, d: u128, r: u128) -> Option<u32> {
    if start & goal != 0 {
        return Some(0);
    }
    let mut f = start;
    let mut unvisited = ALL_NODES ^ start;
    let mut dist = 0;
    loop {
        dist += 1;
        f = expand(f, d, r) & unvisited;
        if f == 0 {
            return None;
        }
        if f & goal != 0 {
            return Some(dist);
        }
        unvisited ^= f;
    }
}

/// start（集合）から goal（集合）へ到達できるか
#[inline]
pub fn reachable(start: u128, goal: u128, d: u128, r: u128) -> bool {
    if start & goal != 0 {
        return true;
    }
    let mut f = start;
    let mut unvisited = ALL_NODES ^ start;
    loop {
        f = expand(f, d, r) & unvisited;
        if f == 0 {
            return false;
        }
        if f & goal != 0 {
            return true;
        }
        unvisited ^= f;
    }
}

/// BFS の層を記録しながら最短経路を 1 本復元する。
///
/// ゴールは最初に届いた層の最小番号のマス、各層では辺が通れる隣接マスのうち
/// 最小番号を選ぶ（Python 版 `shortest_path_nodes` と同じ経路）。
/// `f(prev, cur)` を経路の各辺（始点側 → ゴール側の逆順）について呼ぶ。
/// 戻り値は経路の長さ（辺の数）。到達不能なら None。
#[inline]
pub fn trace_path<F: FnMut(usize, usize)>(start: usize, goal: u128, d: u128, r: u128, mut f: F) -> Option<u32> {
    let s = 1u128 << start;
    if s & goal != 0 {
        return Some(0);
    }
    // 層の記録用（書いた範囲しか読まないのでゼロ初期化しない）
    let mut layers = [std::mem::MaybeUninit::<u128>::uninit(); 81];
    layers[0].write(s);
    let mut n = 1;
    let mut frontier = s;
    let mut unvisited = ALL_NODES ^ s;
    let hit = loop {
        frontier = expand(frontier, d, r) & unvisited;
        if frontier == 0 {
            return None;
        }
        let hit = frontier & goal;
        if hit != 0 {
            break hit;
        }
        unvisited ^= frontier;
        layers[n].write(frontier);
        n += 1;
    };

    let mut cur = hit.trailing_zeros() as usize;
    for k in (0..n).rev() {
        // SAFETY: layers[0..n] は上のループで書き込み済み
        let mut cand = unsafe { layers[k].assume_init() } & STATIC_NB[cur];
        let prev = loop {
            let p = cand.trailing_zeros() as usize;
            let open = if p + 9 == cur {
                (d >> p) & 1
            } else if cur + 9 == p {
                (d >> cur) & 1
            } else if p + 1 == cur {
                (r >> p) & 1
            } else {
                (r >> cur) & 1
            };
            if open != 0 {
                break p;
            }
            cand &= cand - 1;
        };
        f(prev, cur);
        cur = prev;
    }
    Some(n as u32)
}
