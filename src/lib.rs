//! コリドールのルールエンジン（ビットボード実装）
//!
//! - `Board`: 盤面・指し手の実行と取消・合法手生成・最短経路
//! - 指し手は u8 の番号（0〜80 = コマの移動先、81〜144 = 水平壁、145〜208 = 垂直壁）
//! - `Searcher`: αβ 探索（legacy/ai/search.py の移植、Lazy SMP による並列化・時間制限付き）
//! - Windows では cdylib（quoridor_rs.dll）が CPython 拡張モジュール `quoridor_rs` を兼ねる
//!   （PyO3 などの依存クレートを使わず、CPython の安定 ABI を実行時に解決して呼ぶ）

pub mod board;
pub mod consts;
pub mod eval;
pub mod path;
pub mod search;
#[macro_use]
pub mod stats;
pub mod tt;
pub mod zobrist;

// Python の関数は実行時に python3XX.dll から取得するので、常にビルドしても Python は不要
#[cfg(windows)]
mod capi;

pub use board::{decode_action, encode_move, Board, Move, MoveList, PASS};
pub use search::{perft_parallel, Limits, SearchResult, Searcher};
pub use consts::{ACTION_COUNT, HWALL_BASE, VWALL_BASE};
