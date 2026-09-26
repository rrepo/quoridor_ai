//! 盤面の定数と事前計算テーブル
//!
//! マス      : node = row * 9 + col（0〜80）。u128 の下位 81 ビットを使う。
//! 通行可能辺: open_d の bit i = i と i+9（下）の間が通れる
//!             open_r の bit i = i と i+1（右）の間が通れる
//! 壁        : 壁座標 (x, y)（0〜7）を w = y * 8 + x として u64 で持つ。
//!             壁 (x, y) の左上マスは tl = y * 9 + x。
//!             水平壁は tl↔tl+9 と tl+1↔tl+10 を、垂直壁は tl↔tl+1 と tl+9↔tl+10 を塞ぐ。
//! 指し手    : u8 の番号。0〜80 = コマの移動先、81〜144 = 水平壁 81+w、145〜208 = 垂直壁 145+w

pub const BOARD_SIZE: usize = 9;
pub const NODE_COUNT: usize = 81;
pub const MAX_WALLS: u8 = 10;

pub const ACTION_COUNT: usize = 209;
pub const HWALL_BASE: u8 = 81;
pub const VWALL_BASE: u8 = 145;

pub const ALL_NODES: u128 = (1u128 << 81) - 1;
/// 壁のない状態の下方向の辺（row 0〜7 の全マス）
pub const FULL_D: u128 = (1u128 << 72) - 1;
/// 壁のない状態の右方向の辺（col 0〜7 の全マス）
pub const FULL_R: u128 = {
    let mut m = 0u128;
    let mut r = 0;
    while r < 9 {
        m |= 0xFFu128 << (r * 9);
        r += 1;
    }
    m
};

/// ゴール行: P0 は row 8、P1 は row 0
pub const GOAL: [u128; 2] = [0x1FFu128 << 72, 0x1FF];

pub const W_COL0: u64 = 0x0101_0101_0101_0101;
pub const W_COL7: u64 = W_COL0 << 7;
pub const W_ROW0: u64 = 0xFF;
pub const W_ROW7: u64 = 0xFF << 56;
pub const W_NOT_COL0: u64 = !W_COL0;
pub const W_NOT_COL7: u64 = !W_COL7;

/// 壁 w の左上マス
pub const WALL_TL: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut w = 0;
    while w < 64 {
        t[w] = ((w / 8) * 9 + w % 8) as u8;
        w += 1;
    }
    t
};

/// 壁を無視した盤上の隣接マス
pub const STATIC_NB: [u128; 81] = {
    let mut t = [0u128; 81];
    let mut n = 0;
    while n < 81 {
        let c = n % 9;
        let r = n / 9;
        let mut m = 0u128;
        if r > 0 { m |= 1u128 << (n - 9); }
        if r < 8 { m |= 1u128 << (n + 9); }
        if c > 0 { m |= 1u128 << (n - 1); }
        if c < 8 { m |= 1u128 << (n + 1); }
        t[n] = m;
        n += 1;
    }
    t
};

/// 下方向の辺（マス a と a+9）を切る水平壁の集合: (c, r) と (c-1, r)
pub const HCUT_D: [u64; 81] = {
    let mut t = [0u64; 81];
    let mut a = 0;
    while a < 81 {
        let c = a % 9;
        let r = a / 9;
        if r < 8 {
            let mut m = 0u64;
            if c < 8 { m |= 1u64 << (r * 8 + c); }
            if c > 0 { m |= 1u64 << (r * 8 + c - 1); }
            t[a] = m;
        }
        a += 1;
    }
    t
};

/// 右方向の辺（マス a と a+1）を切る垂直壁の集合: (c, r) と (c, r-1)
pub const VCUT_R: [u64; 81] = {
    let mut t = [0u64; 81];
    let mut a = 0;
    while a < 81 {
        let c = a % 9;
        let r = a / 9;
        if c < 8 {
            let mut m = 0u64;
            if r < 8 { m |= 1u64 << (r * 8 + c); }
            if r > 0 { m |= 1u64 << ((r - 1) * 8 + c); }
            t[a] = m;
        }
        a += 1;
    }
    t
};

/// 水平壁 w が塞ぐ下方向の辺
#[inline(always)]
pub const fn hwall_d_bits(w: usize) -> u128 {
    3u128 << WALL_TL[w]
}

/// 垂直壁 w が塞ぐ右方向の辺
#[inline(always)]
pub const fn vwall_r_bits(w: usize) -> u128 {
    0x201u128 << WALL_TL[w]
}
