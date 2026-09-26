//! 置換表（複数スレッドで共有・ロックなし）
//!
//! 1 エントリ = 2 つの AtomicU64（key ^ data, data）。読み出し時に key ^ data が一致しなければ
//! 他スレッドの書き込み途中とみなして捨てる（Lazy SMP でよく使われる方式）。
//!
//! 置き換え規則は Python 版（legacy/ai/search.py の _tt_store）と同じ:
//!   - 同じ局面・同じ世代で、既存の深さ >= 新しい深さ なら上書きしない
//!   - 別の局面で、既存が新しい世代（GEN_WINDOW 以内）かつ既存の深さ > 新しい深さ なら上書きしない

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub const EXACT: u8 = 0;
pub const LOWER: u8 = 1;
pub const UPPER: u8 = 2;

/// この世代数以内のエントリを有効とみなす
pub const GEN_WINDOW: u8 = 4;

pub const NO_MOVE: u8 = 255;

#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub depth: u8,
    pub flag: u8,
    pub gen: u8,
    pub mv: u8,
    pub score: i32,
}

impl Entry {
    #[inline(always)]
    fn pack(&self) -> u64 {
        (self.mv as u64)
            | (self.depth as u64) << 8
            | (self.flag as u64) << 16
            | (self.gen as u64) << 24
            | (self.score as u32 as u64) << 32
    }
    #[inline(always)]
    fn unpack(d: u64) -> Self {
        Entry {
            mv: d as u8,
            depth: (d >> 8) as u8,
            flag: (d >> 16) as u8 & 3,
            gen: (d >> 24) as u8,
            score: (d >> 32) as u32 as i32,
        }
    }
}

pub struct Tt {
    slots: Vec<[AtomicU64; 2]>,
    mask: usize,
}

impl Tt {
    /// size_mb メガバイト（2 の冪のエントリ数に切り下げ）
    pub fn new(size_mb: usize) -> Self {
        let bytes = size_mb.max(1) << 20;
        let n = 1usize << (usize::BITS - 1 - (bytes / 16).leading_zeros());
        let mut slots = Vec::with_capacity(n);
        slots.resize_with(n, || [AtomicU64::new(0), AtomicU64::new(0)]);
        Tt { slots, mask: n - 1 }
    }

    pub fn entries(&self) -> usize {
        self.slots.len()
    }

    pub fn clear(&self) {
        for s in &self.slots {
            s[0].store(0, Relaxed);
            s[1].store(0, Relaxed);
        }
    }

    #[inline(always)]
    fn slot(&self, key: u64) -> &[AtomicU64; 2] {
        &self.slots[key as usize & self.mask]
    }

    #[inline(always)]
    fn read(&self, key: u64) -> Option<(u64, Entry)> {
        let s = self.slot(key);
        let k = s[0].load(Relaxed);
        let d = s[1].load(Relaxed);
        if k == 0 && d == 0 {
            return None;
        }
        Some((k ^ d, Entry::unpack(d)))
    }

    /// 同じ局面のエントリ（世代は見ない）
    #[inline]
    pub fn probe(&self, key: u64) -> Option<Entry> {
        match self.read(key) {
            Some((k, e)) if k == key => Some(e),
            _ => None,
        }
    }

    #[inline]
    pub fn store(&self, key: u64, gen: u8, depth: u8, flag: u8, score: i32, mv: u8) {
        if let Some((k, e)) = self.read(key) {
            let recent = gen.wrapping_sub(e.gen) <= GEN_WINDOW;
            if k == key && e.gen == gen && e.depth >= depth {
                return;
            }
            if k != key && recent && e.depth > depth {
                return;
            }
        }
        let d = Entry { depth, flag, gen, mv, score }.pack();
        let s = self.slot(key);
        s[0].store(key ^ d, Relaxed);
        s[1].store(d, Relaxed);
    }
}
