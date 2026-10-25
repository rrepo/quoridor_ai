//! 計測用カウンタ（`stats` feature を有効にしたときだけ数える。無効なら何もしない）

#[cfg(feature = "stats")]
pub mod counters {
    use std::sync::atomic::AtomicU64;
    pub static DISTANCE: AtomicU64 = AtomicU64::new(0);
    pub static REACHABLE: AtomicU64 = AtomicU64::new(0);
    pub static TRACE: AtomicU64 = AtomicU64::new(0);
    pub static EVAL: AtomicU64 = AtomicU64::new(0);
    pub static WALL_SCORE: AtomicU64 = AtomicU64::new(0);
    pub static CANDIDATES: AtomicU64 = AtomicU64::new(0);
    pub static LEGAL_MASKS: AtomicU64 = AtomicU64::new(0);
    pub static CYC_EVAL: AtomicU64 = AtomicU64::new(0);
    pub static CYC_TT: AtomicU64 = AtomicU64::new(0);
    pub static CYC_CAND: AtomicU64 = AtomicU64::new(0);
    pub static CYC_ORDER: AtomicU64 = AtomicU64::new(0);
    pub static CYC_PAWNGEN: AtomicU64 = AtomicU64::new(0);
    pub static CYC_MAKE: AtomicU64 = AtomicU64::new(0);
    pub static CYC_REP: AtomicU64 = AtomicU64::new(0);
    pub static CYC_TOTAL: AtomicU64 = AtomicU64::new(0);
}

/// 式の実行にかかった CPU サイクル（rdtsc）を name に加算する（stats 無効時は式をそのまま評価）
#[macro_export]
macro_rules! timed {
    ($name:ident, $e:expr) => {{
        #[cfg(feature = "stats")]
        let __t = unsafe { core::arch::x86_64::_rdtsc() };
        let __r = $e;
        #[cfg(feature = "stats")]
        $crate::stats::counters::$name
            .fetch_add(unsafe { core::arch::x86_64::_rdtsc() } - __t, std::sync::atomic::Ordering::Relaxed);
        __r
    }};
}

#[macro_export]
macro_rules! stat {
    ($name:ident) => {
        #[cfg(feature = "stats")]
        $crate::stats::counters::$name.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    };
}

/// カウンタの一覧（stats feature が無効なら空）
pub fn snapshot() -> Vec<(&'static str, u64)> {
    #[cfg(feature = "stats")]
    {
        use counters::*;
        use std::sync::atomic::Ordering::Relaxed;
        return vec![
            ("distance (BFS)", DISTANCE.load(Relaxed)),
            ("reachable (BFS)", REACHABLE.load(Relaxed)),
            ("trace_path (BFS+経路)", TRACE.load(Relaxed)),
            ("evaluate", EVAL.load(Relaxed)),
            ("wall_score", WALL_SCORE.load(Relaxed)),
            ("candidate_walls", CANDIDATES.load(Relaxed)),
            ("legal_wall_masks", LEGAL_MASKS.load(Relaxed)),
            ("cycles: total", CYC_TOTAL.load(Relaxed)),
            ("cycles: evaluate", CYC_EVAL.load(Relaxed)),
            ("cycles: tt probe", CYC_TT.load(Relaxed)),
            ("cycles: candidate walls", CYC_CAND.load(Relaxed)),
            ("cycles: wall ordering", CYC_ORDER.load(Relaxed)),
            ("cycles: pawn gen+sort", CYC_PAWNGEN.load(Relaxed)),
            ("cycles: make+undo", CYC_MAKE.load(Relaxed)),
            ("cycles: repetition", CYC_REP.load(Relaxed)),
        ];
    }
    #[cfg(not(feature = "stats"))]
    vec![]
}
