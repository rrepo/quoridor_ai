//! Zobrist ハッシュのテーブル

use std::sync::{Arc, OnceLock};

use crate::consts::{MAX_WALLS, NODE_COUNT};

pub struct Zobrist {
    pub pos: [[u64; NODE_COUNT]; 2],
    pub hwall: [u64; 64],
    pub vwall: [u64; 64],
    pub turn: u64,
    /// 壁の残数（残数が違えば合法手・評価が変わるためハッシュに含める）
    pub walls: [[u64; MAX_WALLS as usize + 1]; 2],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Zobrist {
    pub fn new(seed: u64) -> Self {
        let mut s = seed;
        let mut z = Zobrist {
            pos: [[0; NODE_COUNT]; 2],
            hwall: [0; 64],
            vwall: [0; 64],
            turn: 0,
            walls: [[0; MAX_WALLS as usize + 1]; 2],
        };
        for p in 0..2 {
            for n in 0..NODE_COUNT {
                z.pos[p][n] = splitmix64(&mut s);
            }
        }
        for w in 0..64 {
            z.hwall[w] = splitmix64(&mut s);
            z.vwall[w] = splitmix64(&mut s);
        }
        z.turn = splitmix64(&mut s);
        for p in 0..2 {
            for c in 0..=MAX_WALLS as usize {
                z.walls[p][c] = splitmix64(&mut s);
            }
        }
        z
    }

    /// 既定のテーブル（プロセス内で共有）
    pub fn default_shared() -> Arc<Zobrist> {
        static DEFAULT: OnceLock<Arc<Zobrist>> = OnceLock::new();
        DEFAULT.get_or_init(|| Arc::new(Zobrist::new(0x5155_4F52_4944_4F52))).clone()
    }
}
