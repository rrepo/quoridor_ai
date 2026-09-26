//! コリドールのルールエンジン（ビットボード実装）
//!
//! - `Board`: 盤面・指し手の実行と取消・合法手生成・最短経路
//! - 指し手は u8 の番号（0〜80 = コマの移動先、81〜144 = 水平壁、145〜208 = 垂直壁）
//! - `capi` feature を有効にすると CPython 拡張モジュール `quoridor_rs` としてビルドできる
//!   （PyO3 などの依存クレートを使わず、CPython の安定 ABI を直接呼ぶ）

pub mod board;
pub mod consts;
pub mod path;
pub mod zobrist;

#[cfg(feature = "capi")]
mod capi;

pub use board::{decode_action, encode_move, Board, Move, MoveList, PASS};
pub use consts::{ACTION_COUNT, HWALL_BASE, VWALL_BASE};
