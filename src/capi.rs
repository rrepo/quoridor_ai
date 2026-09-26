//! CPython 拡張モジュール `quoridor_rs`（PyO3 などを使わない直接実装）
//!
//! - CPython の安定 ABI（Limited API, 3.10 以降）の関数だけを使う
//! - 関数はインポート時に、読み込み済みの python3XX.dll から GetProcAddress で取得する
//!   （import ライブラリやビルドスクリプトが不要。Smart App Control 環境でもビルドできる）
//! - メソッドは METH_FASTCALL で受け取り、引数タプルの生成を避ける
//!
//! 指し手は整数の手番号（0〜80 = コマの移動先、81〜144 = 水平壁、145〜208 = 垂直壁）でも、
//! 既存 Python 版と同じタプル ("move", n) / ("hwall", x, y) / ("vwall", x, y) でも渡せる。

#![allow(non_snake_case, non_upper_case_globals, clippy::missing_safety_doc)]

use std::ffi::{c_char, c_int, c_long, c_ulong, c_void, CStr};
use std::ptr::{self, null, null_mut};

use crate::board::{decode_action, encode_move as encode, Board, Move};
use crate::consts::{ACTION_COUNT, HWALL_BASE, MAX_WALLS, VWALL_BASE};
use crate::search::{default_threads, perft_parallel, Limits, Searcher};
use std::sync::Mutex;
use std::time::Duration;

/// Board.search が使う探索器（置換表などを呼び出し間で持ち越す。Python 版のモジュール変数と同じ扱い）
static SEARCHER: Mutex<Option<Searcher>> = Mutex::new(None);

type Obj = *mut c_void;
type Ssize = isize;

// ----------------------------------------------------------------------
// CPython の定数・構造体（Include/*.h より）
// ----------------------------------------------------------------------
const METH_O: c_int = 0x0008;
const METH_NOARGS: c_int = 0x0004;
const METH_KEYWORDS: c_int = 0x0002;
const METH_STATIC: c_int = 0x0020;
const METH_FASTCALL: c_int = 0x0080;

const Py_tp_alloc: c_int = 47;
const Py_tp_dealloc: c_int = 52;
const Py_tp_doc: c_int = 56;
const Py_tp_methods: c_int = 64;
const Py_tp_new: c_int = 65;
const Py_tp_repr: c_int = 66;
const Py_tp_str: c_int = 70;
const Py_tp_getset: c_int = 73;
const Py_tp_free: c_int = 74;

const Py_TPFLAGS_IMMUTABLETYPE: c_ulong = 1 << 8;
const Py_TPFLAGS_LONG_SUBCLASS: c_ulong = 1 << 24;
const Py_TPFLAGS_TUPLE_SUBCLASS: c_ulong = 1 << 26;

const PYTHON_ABI_VERSION: c_int = 3;
const PY_VECTORCALL_ARGUMENTS_OFFSET: Ssize = 1 << (Ssize::BITS - 1);

/// PyObject のヘッダ（GIL ありビルドの 64bit: refcnt 8 バイト + ob_type）
#[repr(C)]
struct PyObjectHead {
    ob_refcnt: i64,
    ob_type: Obj,
}

#[repr(C)]
struct PyMethodDef {
    ml_name: *const c_char,
    ml_meth: *const c_void,
    ml_flags: c_int,
    ml_doc: *const c_char,
}
unsafe impl Sync for PyMethodDef {}

#[repr(C)]
struct PyGetSetDef {
    name: *const c_char,
    get: *const c_void,
    set: *const c_void,
    doc: *const c_char,
    closure: *mut c_void,
}
unsafe impl Sync for PyGetSetDef {}

#[repr(C)]
struct PyTypeSlot {
    slot: c_int,
    pfunc: *const c_void,
}

#[repr(C)]
struct PyTypeSpec {
    name: *const c_char,
    basicsize: c_int,
    itemsize: c_int,
    flags: c_ulong,
    slots: *mut PyTypeSlot,
}

#[repr(C)]
struct PyModuleDefBase {
    ob_base: PyObjectHead,
    m_init: *const c_void,
    m_index: Ssize,
    m_copy: Obj,
}

#[repr(C)]
struct PyModuleDef {
    m_base: PyModuleDefBase,
    m_name: *const c_char,
    m_doc: *const c_char,
    m_size: Ssize,
    m_methods: *const PyMethodDef,
    m_slots: *const c_void,
    m_traverse: *const c_void,
    m_clear: *const c_void,
    m_free: *const c_void,
}

// ----------------------------------------------------------------------
// CPython API の関数テーブル（インポート時に解決）
// ----------------------------------------------------------------------
#[link(name = "kernel32")]
extern "system" {
    fn GetModuleHandleA(name: *const c_char) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
}

macro_rules! py_api {
    ($( fn $name:ident($($arg:ty),*) -> $ret:ty; )*) => {
        struct Api {
            $( $name: unsafe extern "C" fn($($arg),*) -> $ret, )*
            none: Obj,
            exc_value_error: Obj,
            exc_type_error: Obj,
            exc_index_error: Obj,
        }
        unsafe fn load_api(h: *mut c_void) -> Option<Api> {
            unsafe fn sym(h: *mut c_void, name: &CStr) -> Option<*mut c_void> {
                let p = GetProcAddress(h, name.as_ptr());
                if p.is_null() { None } else { Some(p) }
            }
            unsafe fn data(h: *mut c_void, name: &CStr) -> Option<Obj> {
                Some(*(sym(h, name)? as *const Obj))
            }
            Some(Api {
                $( $name: std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) -> $ret>(
                    sym(h, &CStr::from_bytes_with_nul(concat!(stringify!($name), "\0").as_bytes()).unwrap())?
                ), )*
                none: sym(h, c"_Py_NoneStruct")?,
                exc_value_error: data(h, c"PyExc_ValueError")?,
                exc_type_error: data(h, c"PyExc_TypeError")?,
                exc_index_error: data(h, c"PyExc_IndexError")?,
            })
        }
    };
}

py_api! {
    fn PyModule_Create2(*mut PyModuleDef, c_int) -> Obj;
    fn PyModule_AddObjectRef(Obj, *const c_char, Obj) -> c_int;
    fn PyModule_AddIntConstant(Obj, *const c_char, c_long) -> c_int;
    fn PyType_FromSpec(*mut PyTypeSpec) -> Obj;
    fn PyType_GetSlot(Obj, c_int) -> *mut c_void;
    fn PyType_GetFlags(Obj) -> c_ulong;
    fn Py_IncRef(Obj) -> ();
    fn Py_DecRef(Obj) -> ();
    fn PyLong_FromLong(c_long) -> Obj;
    fn PyLong_FromUnsignedLongLong(u64) -> Obj;
    fn PyLong_AsLongLong(Obj) -> i64;
    fn PyLong_AsUnsignedLongLong(Obj) -> u64;
    fn PyNumber_Lshift(Obj, Obj) -> Obj;
    fn PyNumber_Or(Obj, Obj) -> Obj;
    fn PyBool_FromLong(c_long) -> Obj;
    fn PyFloat_FromDouble(f64) -> Obj;
    fn PyFloat_AsDouble(Obj) -> f64;
    fn PyObject_IsTrue(Obj) -> c_int;
    fn PyTuple_New(Ssize) -> Obj;
    fn PyTuple_SetItem(Obj, Ssize, Obj) -> c_int;
    fn PyTuple_GetItem(Obj, Ssize) -> Obj;
    fn PyTuple_Size(Obj) -> Ssize;
    fn PyList_New(Ssize) -> Obj;
    fn PyList_SetItem(Obj, Ssize, Obj) -> c_int;
    fn PyDict_GetItemString(Obj, *const c_char) -> Obj;
    fn PyUnicode_FromStringAndSize(*const c_char, Ssize) -> Obj;
    fn PyUnicode_AsUTF8AndSize(Obj, *mut Ssize) -> *const c_char;
    fn PyBytes_FromStringAndSize(*const c_char, Ssize) -> Obj;
    fn PyErr_SetString(Obj, *const c_char) -> ();
    fn PyErr_Occurred() -> Obj;
    fn PyEval_SaveThread() -> *mut c_void;
    fn PyEval_RestoreThread(*mut c_void) -> ();
}

// インポート時に 1 度だけ設定し、以後は読み取りのみ（GIL 下）
static mut API: *const Api = null();
static mut BOARD_TYPE: Obj = null_mut();
/// 手番号 → タプル形式の指し手（キャッシュ）
static mut MOVE_TUPLES: [Obj; ACTION_COUNT] = [null_mut(); ACTION_COUNT];

#[inline(always)]
unsafe fn api() -> &'static Api {
    &*API
}

unsafe fn none() -> Obj {
    let n = api().none;
    (api().Py_IncRef)(n);
    n
}

unsafe fn raise(exc: Obj, msg: &str) -> Obj {
    let mut s = msg.as_bytes().to_vec();
    s.retain(|&b| b != 0);
    s.push(0);
    (api().PyErr_SetString)(exc, s.as_ptr() as *const c_char);
    null_mut()
}

#[inline(always)]
unsafe fn type_flags(o: Obj) -> c_ulong {
    (api().PyType_GetFlags)((*(o as *const PyObjectHead)).ob_type)
}

#[inline(always)]
unsafe fn is_int(o: Obj) -> bool {
    type_flags(o) & Py_TPFLAGS_LONG_SUBCLASS != 0
}

#[inline(always)]
unsafe fn is_tuple(o: Obj) -> bool {
    type_flags(o) & Py_TPFLAGS_TUPLE_SUBCLASS != 0
}

#[inline(always)]
unsafe fn py_int(v: i64) -> Obj {
    (api().PyLong_FromLong)(v as c_long)
}

unsafe fn py_u128(v: u128) -> Obj {
    let a = api();
    let lo = (a.PyLong_FromUnsignedLongLong)(v as u64);
    if v >> 64 == 0 {
        return lo;
    }
    let hi = (a.PyLong_FromUnsignedLongLong)((v >> 64) as u64);
    let sh = (a.PyLong_FromLong)(64);
    let hi_sh = (a.PyNumber_Lshift)(hi, sh);
    let r = (a.PyNumber_Or)(hi_sh, lo);
    (a.Py_DecRef)(lo);
    (a.Py_DecRef)(hi);
    (a.Py_DecRef)(sh);
    (a.Py_DecRef)(hi_sh);
    r
}

/// 要素の参照を「盗む」タプル生成
unsafe fn py_tuple(items: &[Obj]) -> Obj {
    let t = (api().PyTuple_New)(items.len() as Ssize);
    for (i, &o) in items.iter().enumerate() {
        (api().PyTuple_SetItem)(t, i as Ssize, o);
    }
    t
}

unsafe fn py_str(s: &str) -> Obj {
    (api().PyUnicode_FromStringAndSize)(s.as_ptr() as *const c_char, s.len() as Ssize)
}

unsafe fn str_of(o: Obj) -> Option<&'static str> {
    let mut n: Ssize = 0;
    let p = (api().PyUnicode_AsUTF8AndSize)(o, &mut n);
    if p.is_null() {
        return None;
    }
    std::str::from_utf8(std::slice::from_raw_parts(p as *const u8, n as usize)).ok()
}

unsafe fn int_of(o: Obj) -> Option<i64> {
    if !is_int(o) {
        return None;
    }
    let v = (api().PyLong_AsLongLong)(o);
    if v == -1 && !(api().PyErr_Occurred)().is_null() {
        return None;
    }
    Some(v)
}

/// 指し手オブジェクト → 手番号。失敗時は例外を設定して None。
unsafe fn to_action(mv: Obj) -> Option<u8> {
    if is_int(mv) {
        return match int_of(mv) {
            Some(a) if (0..ACTION_COUNT as i64).contains(&a) => Some(a as u8),
            Some(_) => {
                raise(api().exc_value_error, "action out of range (0..208)");
                None
            }
            None => None,
        };
    }
    if is_tuple(mv) {
        let n = (api().PyTuple_Size)(mv);
        if n == 2 || n == 3 {
            let kind = str_of((api().PyTuple_GetItem)(mv, 0));
            let a = int_of((api().PyTuple_GetItem)(mv, 1));
            let b = if n == 3 { int_of((api().PyTuple_GetItem)(mv, 2)) } else { Some(0) };
            // 負の値や 256 以上は u8 に変換できない → 不正な手として扱う
            let ab = a.zip(b).and_then(|(a, b)| Some((u8::try_from(a).ok()?, u8::try_from(b).ok()?)));
            if let (Some(kind), Some((a, b))) = (kind, ab) {
                let m = match (kind, n) {
                    ("move", 2) => Some(Move::Pawn(a)),
                    ("hwall", 3) => Some(Move::HWall(a, b)),
                    ("vwall", 3) => Some(Move::VWall(a, b)),
                    _ => None,
                };
                if let Some(action) = m.and_then(encode) {
                    return Some(action);
                }
            }
        }
    }
    if (api().PyErr_Occurred)().is_null() {
        raise(
            api().exc_value_error,
            "move must be an action index (0..208) or ('move', n) / ('hwall', x, y) / ('vwall', x, y)",
        );
    }
    None
}

unsafe fn move_tuple(a: u8) -> Obj {
    let t = MOVE_TUPLES[a as usize];
    (api().Py_IncRef)(t);
    t
}

/// FASTCALL|KEYWORDS の引数を位置 or キーワードで取り出す
unsafe fn arg(args: *const Obj, nargs: Ssize, kwnames: Obj, pos: usize, name: &str) -> Option<Obj> {
    let nargs = (nargs & !PY_VECTORCALL_ARGUMENTS_OFFSET) as usize;
    if pos < nargs {
        return Some(*args.add(pos));
    }
    if !kwnames.is_null() {
        let nk = (api().PyTuple_Size)(kwnames) as usize;
        for i in 0..nk {
            if str_of((api().PyTuple_GetItem)(kwnames, i as Ssize)) == Some(name) {
                return Some(*args.add(nargs + i));
            }
        }
    }
    None
}

unsafe fn bool_arg(o: Option<Obj>, default: bool) -> Option<bool> {
    match o {
        None => Some(default),
        Some(o) => match (api().PyObject_IsTrue)(o) {
            -1 => None,
            v => Some(v != 0),
        },
    }
}

unsafe fn player_arg(o: Obj) -> Option<usize> {
    match int_of(o) {
        Some(p @ 0..=1) => Some(p as usize),
        Some(_) => {
            raise(api().exc_index_error, "player must be 0 or 1");
            None
        }
        None => {
            if (api().PyErr_Occurred)().is_null() {
                raise(api().exc_type_error, "player must be an int");
            }
            None
        }
    }
}

// ----------------------------------------------------------------------
// Board オブジェクト
// ----------------------------------------------------------------------
#[repr(C)]
struct BoardObj {
    head: PyObjectHead,
    board: Board,
}

#[inline(always)]
unsafe fn board<'a>(o: Obj) -> &'a mut Board {
    &mut (*(o as *mut BoardObj)).board
}

unsafe fn alloc_board(tp: Obj, b: Board) -> Obj {
    let alloc: unsafe extern "C" fn(Obj, Ssize) -> Obj = std::mem::transmute((api().PyType_GetSlot)(tp, Py_tp_alloc));
    let o = alloc(tp, 0);
    if o.is_null() {
        return null_mut();
    }
    ptr::write(&mut (*(o as *mut BoardObj)).board, b);
    o
}

unsafe extern "C" fn board_new(tp: Obj, args: Obj, kwds: Obj) -> Obj {
    let mut seed: Obj = null_mut();
    if !args.is_null() && (api().PyTuple_Size)(args) > 0 {
        if (api().PyTuple_Size)(args) > 1 {
            return raise(api().exc_type_error, "Board(seed=None) takes at most 1 argument");
        }
        seed = (api().PyTuple_GetItem)(args, 0);
    }
    if !kwds.is_null() {
        let s = (api().PyDict_GetItemString)(kwds, c"seed".as_ptr());
        if !s.is_null() {
            seed = s;
        }
    }
    let b = if seed.is_null() || seed == api().none {
        Board::new()
    } else if is_int(seed) {
        let v = (api().PyLong_AsUnsignedLongLong)(seed);
        if !(api().PyErr_Occurred)().is_null() {
            return null_mut();
        }
        Board::with_seed(v)
    } else {
        return raise(api().exc_type_error, "seed must be an int or None");
    };
    alloc_board(tp, b)
}

unsafe extern "C" fn board_dealloc(o: Obj) {
    let tp = (*(o as *const PyObjectHead)).ob_type;
    ptr::drop_in_place(&mut (*(o as *mut BoardObj)).board);
    let free: unsafe extern "C" fn(Obj) = std::mem::transmute((api().PyType_GetSlot)(tp, Py_tp_free));
    free(o);
    (api().Py_DecRef)(tp);
}

unsafe extern "C" fn board_str(o: Obj) -> Obj {
    py_str(&board(o).to_string())
}

unsafe extern "C" fn board_repr(o: Obj) -> Obj {
    let b = board(o);
    py_str(&format!(
        "Board(positions=({}, {}), walls_left=({}, {}), turn={}, h_walls={}, v_walls={})",
        b.pos[0],
        b.pos[1],
        b.walls[0],
        b.walls[1],
        b.turn,
        b.hmask.count_ones(),
        b.vmask.count_ones()
    ))
}

// ---------------- メソッド ----------------
type FastKw = unsafe extern "C" fn(Obj, *const Obj, Ssize, Obj) -> Obj;
type NoArgs = unsafe extern "C" fn(Obj, Obj) -> Obj;
type OneArg = unsafe extern "C" fn(Obj, Obj) -> Obj;

/// 壁の残数がない・既存の壁と重なる／交差する壁か（Board::make に渡すと盤面が壊れる手）
fn breaks_board(b: &Board, a: u8) -> bool {
    if a < HWALL_BASE {
        return false;
    }
    if b.walls[b.turn as usize] == 0 {
        return true;
    }
    let (vh, vv) = b.valid_wall_masks();
    if a < VWALL_BASE {
        (vh >> (a - HWALL_BASE)) & 1 == 0
    } else {
        (vv >> (a - VWALL_BASE)) & 1 == 0
    }
}

/// make_move(move, check=True)
///
/// check=False は経路の確認（BFS）などを省くが、盤面が壊れる手（breaks_board）は常に弾く。
unsafe extern "C" fn m_make_move(o: Obj, args: *const Obj, nargs: Ssize, kw: Obj) -> Obj {
    let Some(mv) = arg(args, nargs, kw, 0, "move") else {
        return raise(api().exc_type_error, "make_move() missing argument: move");
    };
    let Some(a) = to_action(mv) else { return null_mut() };
    let Some(check) = bool_arg(arg(args, nargs, kw, 1, "check"), true) else { return null_mut() };
    let b = board(o);
    let rejected = if check { !b.is_legal(a) } else { breaks_board(b, a) };
    if rejected {
        let desc = match decode_action(a) {
            Some(Move::Pawn(n)) => format!("('move', {n})"),
            Some(Move::HWall(x, y)) => format!("('hwall', {x}, {y})"),
            Some(Move::VWall(x, y)) => format!("('vwall', {x}, {y})"),
            None => format!("{a}"),
        };
        return raise(api().exc_value_error, &format!("illegal move: {desc}"));
    }
    b.make(a);
    none()
}

/// 直前の手（パスを含む）を取り消す。取り消す手がなければ False。
unsafe extern "C" fn m_undo_move(o: Obj, _: Obj) -> Obj {
    (api().PyBool_FromLong)(board(o).undo() as c_long)
}

unsafe extern "C" fn m_make_pass(o: Obj, _: Obj) -> Obj {
    board(o).make_pass();
    none()
}

unsafe extern "C" fn m_is_legal(o: Obj, mv: Obj) -> Obj {
    let Some(a) = to_action(mv) else { return null_mut() };
    (api().PyBool_FromLong)(board(o).is_legal(a) as c_long)
}

unsafe extern "C" fn m_legal_actions(o: Obj, _: Obj) -> Obj {
    let ml = board(o).legal_actions();
    let s = ml.as_slice();
    let list = (api().PyList_New)(s.len() as Ssize);
    for (i, &a) in s.iter().enumerate() {
        (api().PyList_SetItem)(list, i as Ssize, py_int(a as i64));
    }
    list
}

unsafe extern "C" fn m_legal_moves(o: Obj, _: Obj) -> Obj {
    let ml = board(o).legal_actions();
    let s = ml.as_slice();
    let list = (api().PyList_New)(s.len() as Ssize);
    for (i, &a) in s.iter().enumerate() {
        (api().PyList_SetItem)(list, i as Ssize, move_tuple(a));
    }
    list
}

unsafe extern "C" fn m_count_legal_actions(o: Obj, _: Obj) -> Obj {
    py_int(board(o).count_legal_actions() as i64)
}

unsafe extern "C" fn m_action_mask(o: Obj, _: Obj) -> Obj {
    let mut mask = [0u8; ACTION_COUNT];
    for &a in board(o).legal_actions().as_slice() {
        mask[a as usize] = 1;
    }
    (api().PyBytes_FromStringAndSize)(mask.as_ptr() as *const c_char, ACTION_COUNT as Ssize)
}

unsafe extern "C" fn m_pawn_dest_mask(o: Obj, _: Obj) -> Obj {
    py_u128(board(o).pawn_dest_mask())
}

unsafe fn mask_pair(m: (u64, u64)) -> Obj {
    let a = api();
    py_tuple(&[(a.PyLong_FromUnsignedLongLong)(m.0), (a.PyLong_FromUnsignedLongLong)(m.1)])
}

unsafe extern "C" fn m_legal_wall_masks(o: Obj, _: Obj) -> Obj {
    mask_pair(board(o).legal_wall_masks())
}

unsafe extern "C" fn m_valid_wall_masks(o: Obj, _: Obj) -> Obj {
    mask_pair(board(o).valid_wall_masks())
}

unsafe extern "C" fn m_path_cut_masks(o: Obj, p: Obj) -> Obj {
    let Some(p) = player_arg(p) else { return null_mut() };
    match board(o).path_cut_masks(p) {
        Some(m) => mask_pair(m),
        None => none(),
    }
}

unsafe extern "C" fn m_shortest_path(o: Obj, p: Obj) -> Obj {
    let Some(p) = player_arg(p) else { return null_mut() };
    match board(o).shortest_path(p) {
        Some(d) => py_int(d as i64),
        None => none(),
    }
}

unsafe extern "C" fn m_shortest_path_nodes(o: Obj, p: Obj) -> Obj {
    let Some(p) = player_arg(p) else { return null_mut() };
    match board(o).shortest_path_nodes(p) {
        Some(path) => {
            let list = (api().PyList_New)(path.len() as Ssize);
            for (i, &n) in path.iter().enumerate() {
                (api().PyList_SetItem)(list, i as Ssize, py_int(n as i64));
            }
            list
        }
        None => none(),
    }
}

unsafe extern "C" fn m_winner(o: Obj, _: Obj) -> Obj {
    match board(o).winner() {
        Some(w) => py_int(w as i64),
        None => none(),
    }
}

/// 既存 Python 版と同じ形式: (終局したか, 勝者 or -1)
unsafe extern "C" fn m_is_terminal(o: Obj, _: Obj) -> Obj {
    let (t, w) = match board(o).winner() {
        Some(w) => (1, w as i64),
        None => (0, -1),
    };
    py_tuple(&[(api().PyBool_FromLong)(t), py_int(w)])
}

unsafe extern "C" fn m_repetitions(o: Obj, _: Obj) -> Obj {
    py_int(board(o).repetitions() as i64)
}

/// perft(depth, bulk=False, threads=None)。計算中は GIL を解放する。
unsafe extern "C" fn m_perft(o: Obj, args: *const Obj, nargs: Ssize, kw: Obj) -> Obj {
    let Some(d) = arg(args, nargs, kw, 0, "depth") else {
        return raise(api().exc_type_error, "perft() missing argument: depth");
    };
    let depth = match int_of(d) {
        Some(v @ 0..=16) => v as u32,
        Some(_) => return raise(api().exc_value_error, "depth must be 0..16"),
        None => return raise(api().exc_type_error, "depth must be an int"),
    };
    let Some(bulk) = bool_arg(arg(args, nargs, kw, 1, "bulk"), false) else { return null_mut() };
    let threads = match arg(args, nargs, kw, 2, "threads").filter(|&t| t != api().none) {
        None => default_threads(),
        Some(t) => match int_of(t) {
            Some(v @ 1..=256) => v as usize,
            _ => return raise(api().exc_value_error, "threads must be 1..256"),
        },
    };
    let b = board(o).clone();
    let ts = (api().PyEval_SaveThread)();
    let n = perft_parallel(&b, depth, threads, bulk);
    (api().PyEval_RestoreThread)(ts);
    (api().PyLong_FromUnsignedLongLong)(n)
}

/// search(depth=4, time_ms=None, threads=None) -> (最善手 or None, 評価値, 完了した深さ, ノード数)
///
/// threads=None なら CPU の論理スレッド数で探索する。
unsafe extern "C" fn m_search(o: Obj, args: *const Obj, nargs: Ssize, kw: Obj) -> Obj {
    let depth_arg = arg(args, nargs, kw, 0, "depth");
    let time_arg = arg(args, nargs, kw, 1, "time_ms").filter(|&t| t != api().none);
    let mut lim = Limits { max_depth: 4, time: None, threads: default_threads() };
    if let Some(d) = depth_arg {
        match int_of(d) {
            Some(v @ 1..=60) => lim.max_depth = v as u32,
            _ => return raise(api().exc_value_error, "depth must be 1..60"),
        }
    }
    if let Some(t) = time_arg {
        let ms = (api().PyFloat_AsDouble)(t);
        // inf や巨大な値は Duration にできない（from_secs_f64 はパニックする）
        let time = Duration::try_from_secs_f64(ms / 1000.0).ok().filter(|_| ms > 0.0);
        let Some(time) = time.filter(|_| (api().PyErr_Occurred)().is_null()) else {
            return raise(api().exc_value_error, "time_ms must be a positive finite number");
        };
        lim.time = Some(time);
        if depth_arg.is_none() {
            lim.max_depth = 60;
        }
    }
    if let Some(t) = arg(args, nargs, kw, 2, "threads").filter(|&t| t != api().none) {
        match int_of(t) {
            Some(v @ 1..=256) => lim.threads = v as usize,
            _ => return raise(api().exc_value_error, "threads must be 1..256"),
        }
    }
    let b = board(o).clone();
    let ts = (api().PyEval_SaveThread)();
    let r = {
        let mut g = SEARCHER.lock().unwrap_or_else(|e| e.into_inner());
        g.get_or_insert_with(|| Searcher::new(64)).search(&b, &lim)
    };
    (api().PyEval_RestoreThread)(ts);
    let mv = match r.best {
        Some(a) => move_tuple(a),
        None => none(),
    };
    py_tuple(&[
        mv,
        (api().PyFloat_FromDouble)(r.score as f64 / 100.0),
        py_int(r.depth as i64),
        (api().PyLong_FromUnsignedLongLong)(r.nodes),
    ])
}

/// NN 入力用: (4, 9, 9) の float32 を bytes で返す（自分の駒・相手の駒・横壁・縦壁）
unsafe extern "C" fn m_to_planes(o: Obj, _: Obj) -> Obj {
    let b = board(o);
    let mut planes = [0f32; 4 * 81];
    let t = b.turn as usize;
    planes[b.pos[t] as usize] = 1.0;
    planes[81 + b.pos[t ^ 1] as usize] = 1.0;
    for (k, mut m) in [(2usize, b.hmask), (3, b.vmask)] {
        while m != 0 {
            let w = m.trailing_zeros() as usize;
            planes[k * 81 + (w / 8) * 9 + w % 8] = 1.0;
            m &= m - 1;
        }
    }
    (api().PyBytes_FromStringAndSize)(planes.as_ptr() as *const c_char, (planes.len() * 4) as Ssize)
}

unsafe extern "C" fn m_clone(o: Obj, _: Obj) -> Obj {
    alloc_board(BOARD_TYPE, board(o).clone())
}

unsafe extern "C" fn m_deepcopy(o: Obj, _memo: Obj) -> Obj {
    alloc_board(BOARD_TYPE, board(o).clone())
}

/// from_state(positions, walls_left=(10, 10), turn=0, hmask=0, vmask=0)
unsafe extern "C" fn m_from_state(_: Obj, args: *const Obj, nargs: Ssize, kw: Obj) -> Obj {
    unsafe fn pair(o: Obj) -> Option<(i64, i64)> {
        if !is_tuple(o) || (api().PyTuple_Size)(o) != 2 {
            return None;
        }
        Some((int_of((api().PyTuple_GetItem)(o, 0))?, int_of((api().PyTuple_GetItem)(o, 1))?))
    }
    let bad = || raise(api().exc_value_error, "invalid state");
    let Some(pos) = arg(args, nargs, kw, 0, "positions").and_then(|o| pair(o)) else { return bad() };
    let walls = match arg(args, nargs, kw, 1, "walls_left") {
        None => (10, 10),
        Some(o) => match pair(o) {
            Some(w) => w,
            None => return bad(),
        },
    };
    let turn = match arg(args, nargs, kw, 2, "turn") {
        None => 0,
        Some(o) => match int_of(o) {
            Some(t) => t,
            None => return bad(),
        },
    };
    let mut masks = [0u64; 2];
    for (i, name) in ["hmask", "vmask"].iter().enumerate() {
        if let Some(o) = arg(args, nargs, kw, 3 + i, name) {
            if !is_int(o) {
                return bad();
            }
            masks[i] = (api().PyLong_AsUnsignedLongLong)(o);
            if !(api().PyErr_Occurred)().is_null() {
                return null_mut();
            }
        }
    }
    let ok = (0..81).contains(&pos.0)
        && (0..81).contains(&pos.1)
        && pos.0 != pos.1
        && (0..=MAX_WALLS as i64).contains(&walls.0)
        && (0..=MAX_WALLS as i64).contains(&walls.1)
        && (0..=1).contains(&turn);
    if !ok {
        return bad();
    }
    let b = Board::from_parts(
        [pos.0 as u8, pos.1 as u8],
        [walls.0 as u8, walls.1 as u8],
        turn as u8,
        masks[0],
        masks[1],
    );
    alloc_board(BOARD_TYPE, b)
}

// ---------------- プロパティ ----------------
type Getter = unsafe extern "C" fn(Obj, *mut c_void) -> Obj;

unsafe extern "C" fn g_turn(o: Obj, _: *mut c_void) -> Obj {
    py_int(board(o).turn as i64)
}
unsafe extern "C" fn g_zobrist(o: Obj, _: *mut c_void) -> Obj {
    (api().PyLong_FromUnsignedLongLong)(board(o).hash)
}
unsafe extern "C" fn g_positions(o: Obj, _: *mut c_void) -> Obj {
    let b = board(o);
    py_tuple(&[py_int(b.pos[0] as i64), py_int(b.pos[1] as i64)])
}
unsafe extern "C" fn g_walls_left(o: Obj, _: *mut c_void) -> Obj {
    let b = board(o);
    py_tuple(&[py_int(b.walls[0] as i64), py_int(b.walls[1] as i64)])
}
unsafe fn wall_list(mut m: u64) -> Obj {
    let list = (api().PyList_New)(m.count_ones() as Ssize);
    let mut i = 0;
    while m != 0 {
        let w = m.trailing_zeros() as i64;
        (api().PyList_SetItem)(list, i, py_tuple(&[py_int(w % 8), py_int(w / 8)]));
        i += 1;
        m &= m - 1;
    }
    list
}
unsafe extern "C" fn g_h_walls(o: Obj, _: *mut c_void) -> Obj {
    wall_list(board(o).hmask)
}
unsafe extern "C" fn g_v_walls(o: Obj, _: *mut c_void) -> Obj {
    wall_list(board(o).vmask)
}
unsafe extern "C" fn g_hmask(o: Obj, _: *mut c_void) -> Obj {
    (api().PyLong_FromUnsignedLongLong)(board(o).hmask)
}
unsafe extern "C" fn g_vmask(o: Obj, _: *mut c_void) -> Obj {
    (api().PyLong_FromUnsignedLongLong)(board(o).vmask)
}
unsafe extern "C" fn g_ply(o: Obj, _: *mut c_void) -> Obj {
    py_int(board(o).ply() as i64)
}

// ---------------- モジュール関数 ----------------
unsafe extern "C" fn f_encode_move(_: Obj, mv: Obj) -> Obj {
    match to_action(mv) {
        Some(a) => py_int(a as i64),
        None => null_mut(),
    }
}

/// Board.search の置換表・キラー手・history を消す（Python 版 clear_tt に相当）
unsafe extern "C" fn f_clear_search(_: Obj, _: Obj) -> Obj {
    let ts = (api().PyEval_SaveThread)();
    if let Some(s) = SEARCHER.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        s.clear();
    }
    (api().PyEval_RestoreThread)(ts);
    none()
}

unsafe extern "C" fn f_decode_move(_: Obj, a: Obj) -> Obj {
    match int_of(a) {
        Some(v) if (0..ACTION_COUNT as i64).contains(&v) => move_tuple(v as u8),
        Some(_) => raise(api().exc_value_error, "action out of range (0..208)"),
        None => raise(api().exc_type_error, "action must be an int"),
    }
}

// ----------------------------------------------------------------------
// 定義テーブル
// ----------------------------------------------------------------------
macro_rules! meth {
    ($name:literal, $f:expr, $flags:expr, $doc:literal) => {
        PyMethodDef {
            ml_name: concat!($name, "\0").as_ptr() as *const c_char,
            ml_meth: $f as *const c_void,
            ml_flags: $flags,
            ml_doc: concat!($doc, "\0").as_ptr() as *const c_char,
        }
    };
}

macro_rules! getter {
    ($name:literal, $f:expr, $doc:literal) => {
        PyGetSetDef {
            name: concat!($name, "\0").as_ptr() as *const c_char,
            get: $f as Getter as *const c_void,
            set: null(),
            doc: concat!($doc, "\0").as_ptr() as *const c_char,
            closure: null_mut(),
        }
    };
}

const FK: c_int = METH_FASTCALL | METH_KEYWORDS;

static METHODS: [PyMethodDef; 25] = [
    meth!("make_move", m_make_move as FastKw, FK, "make_move(move, check=True)\n--\n\n手を指す。check=True なら非合法手で ValueError。check=False でも、残数のない壁と重なる壁は ValueError。"),
    meth!("undo_move", m_undo_move as NoArgs, METH_NOARGS, "直前の手（パスを含む）を取り消す。取り消す手がなければ False。"),
    meth!("make_pass", m_make_pass as NoArgs, METH_NOARGS, "パス（ヌルムーブ）。undo_move で取り消せる。"),
    meth!("is_legal", m_is_legal as OneArg, METH_O, "手が合法か。"),
    meth!("legal_actions", m_legal_actions as NoArgs, METH_NOARGS, "全合法手（整数の手番号）のリスト。"),
    meth!("legal_moves", m_legal_moves as NoArgs, METH_NOARGS, "全合法手（('move', n) / ('hwall', x, y) / ('vwall', x, y)）のリスト。"),
    meth!("count_legal_actions", m_count_legal_actions as NoArgs, METH_NOARGS, "全合法手の数。"),
    meth!("action_mask", m_action_mask as NoArgs, METH_NOARGS, "合法手を 209 要素の uint8（bytes）で返す。np.frombuffer(m, np.uint8)。"),
    meth!("pawn_dest_mask", m_pawn_dest_mask as NoArgs, METH_NOARGS, "手番側のコマの移動先（81 ビット整数）。"),
    meth!("legal_wall_masks", m_legal_wall_masks as NoArgs, METH_NOARGS, "合法な壁 (水平, 垂直) の 64 ビットマスク（bit = y*8 + x。残数は見ない）。"),
    meth!("valid_wall_masks", m_valid_wall_masks as NoArgs, METH_NOARGS, "重なり・交差のない壁 (水平, 垂直)（経路は見ない）。"),
    meth!("path_cut_masks", m_path_cut_masks as OneArg, METH_O, "player の最短経路を切る壁 (水平, 垂直)。到達不能なら None。"),
    meth!("shortest_path", m_shortest_path as OneArg, METH_O, "player のゴールまでの最短距離。到達不能なら None。"),
    meth!("shortest_path_nodes", m_shortest_path_nodes as OneArg, METH_O, "player の最短経路のマス列。到達不能なら None。"),
    meth!("winner", m_winner as NoArgs, METH_NOARGS, "勝者（0 / 1）。終局していなければ None。"),
    meth!("is_terminal", m_is_terminal as NoArgs, METH_NOARGS, "(終局したか, 勝者 or -1)。"),
    meth!("repetitions", m_repetitions as NoArgs, METH_NOARGS, "現在の局面がこれまでの手で何回現れたか（初期局面は数えない）。"),
    meth!("perft", m_perft as FastKw, FK, "perft(depth, bulk=False, threads=None)\n--\n\n深さ depth の葉の数。threads=None なら CPU の論理スレッド数。計算中は GIL を解放する。"),
    meth!("search", m_search as FastKw, FK, "search(depth=4, time_ms=None, threads=None)\n--\n\nαβ 探索で最善手を返す: (手 or None, 評価値, 完了した深さ, ノード数)。threads=None なら CPU の論理スレッド数で探索する。置換表は呼び出し間で持ち越す。"),
    meth!("to_planes", m_to_planes as NoArgs, METH_NOARGS, "(4, 9, 9) float32 の bytes。np.frombuffer(p, np.float32).reshape(4, 9, 9)。"),
    meth!("clone", m_clone as NoArgs, METH_NOARGS, "盤面を複製する。"),
    meth!("__copy__", m_clone as NoArgs, METH_NOARGS, ""),
    meth!("__deepcopy__", m_deepcopy as OneArg, METH_O, ""),
    meth!("from_state", m_from_state as FastKw, FK | METH_STATIC, "from_state(positions, walls_left=(10, 10), turn=0, hmask=0, vmask=0)\n--\n\n任意の局面を作る（壁は合法に置かれていること）。"),
    PyMethodDef { ml_name: null(), ml_meth: null(), ml_flags: 0, ml_doc: null() },
];

static GETSET: [PyGetSetDef; 10] = [
    getter!("turn", g_turn, "手番（0 / 1）"),
    getter!("zobrist", g_zobrist, "Zobrist ハッシュ"),
    getter!("positions", g_positions, "(P0 の位置, P1 の位置)"),
    getter!("walls_left", g_walls_left, "(P0 の残り壁, P1 の残り壁)"),
    getter!("h_walls", g_h_walls, "水平壁 [(x, y), ...]"),
    getter!("v_walls", g_v_walls, "垂直壁 [(x, y), ...]"),
    getter!("hmask", g_hmask, "水平壁の 64 ビットマスク"),
    getter!("vmask", g_vmask, "垂直壁の 64 ビットマスク"),
    getter!("ply", g_ply, "指した手の数（undo_move で取り消せる数）"),
    PyGetSetDef { name: null(), get: null(), set: null(), doc: null(), closure: null_mut() },
];

static MODULE_METHODS: [PyMethodDef; 4] = [
    meth!("encode_move", f_encode_move as OneArg, METH_O, "タプル形式の指し手 → 手番号。"),
    meth!("clear_search", f_clear_search as NoArgs, METH_NOARGS, "Board.search の置換表・キラー手・history を消す。"),
    meth!("decode_move", f_decode_move as OneArg, METH_O, "手番号 → タプル形式の指し手。"),
    PyMethodDef { ml_name: null(), ml_meth: null(), ml_flags: 0, ml_doc: null() },
];

static mut MODULE_DEF: PyModuleDef = PyModuleDef {
    m_base: PyModuleDefBase {
        // 静的オブジェクトとして不死（immortal）の参照カウントにしておく
        ob_base: PyObjectHead { ob_refcnt: 3 << 30, ob_type: null_mut() },
        m_init: null(),
        m_index: 0,
        m_copy: null_mut(),
    },
    m_name: c"quoridor_rs".as_ptr(),
    m_doc: c"Quoridor rules engine (Rust bitboard implementation)".as_ptr(),
    m_size: -1,
    m_methods: null(),
    m_slots: null(),
    m_traverse: null(),
    m_clear: null(),
    m_free: null(),
};

/// 読み込み済みの CPython 本体 DLL（python3XX.dll）を探す
unsafe fn find_python_dll() -> *mut c_void {
    for minor in (10..=40).rev() {
        let name = format!("python3{minor}.dll\0");
        let h = GetModuleHandleA(name.as_ptr() as *const c_char);
        if !h.is_null() {
            return h;
        }
    }
    GetModuleHandleA(c"python3.dll".as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn PyInit_quoridor_rs() -> Obj {
    let h = find_python_dll();
    if h.is_null() {
        return null_mut();
    }
    let Some(a) = load_api(h) else { return null_mut() };
    API = Box::into_raw(Box::new(a));
    let a = api();

    // 指し手タプルのキャッシュ
    for i in 0..ACTION_COUNT {
        let t = match decode_action(i as u8).unwrap() {
            Move::Pawn(n) => py_tuple(&[py_str("move"), py_int(n as i64)]),
            Move::HWall(x, y) => py_tuple(&[py_str("hwall"), py_int(x as i64), py_int(y as i64)]),
            Move::VWall(x, y) => py_tuple(&[py_str("vwall"), py_int(x as i64), py_int(y as i64)]),
        };
        MOVE_TUPLES[i] = t;
    }

    let mut slots = [
        PyTypeSlot { slot: Py_tp_new, pfunc: board_new as *const c_void },
        PyTypeSlot { slot: Py_tp_dealloc, pfunc: board_dealloc as *const c_void },
        PyTypeSlot { slot: Py_tp_str, pfunc: board_str as *const c_void },
        PyTypeSlot { slot: Py_tp_repr, pfunc: board_repr as *const c_void },
        PyTypeSlot { slot: Py_tp_methods, pfunc: METHODS.as_ptr() as *const c_void },
        PyTypeSlot { slot: Py_tp_getset, pfunc: GETSET.as_ptr() as *const c_void },
        PyTypeSlot {
            slot: Py_tp_doc,
            pfunc: c"Board(seed=None)\n--\n\nコリドールの盤面。seed を指定すると Zobrist テーブルを seed から作る。"
                .as_ptr() as *const c_void,
        },
        PyTypeSlot { slot: 0, pfunc: null() },
    ];
    let mut spec = PyTypeSpec {
        name: c"quoridor_rs.Board".as_ptr(),
        basicsize: std::mem::size_of::<BoardObj>() as c_int,
        itemsize: 0,
        flags: Py_TPFLAGS_IMMUTABLETYPE,
        slots: slots.as_mut_ptr(),
    };
    let tp = (a.PyType_FromSpec)(&mut spec);
    if tp.is_null() {
        return null_mut();
    }
    BOARD_TYPE = tp;

    let def = ptr::addr_of_mut!(MODULE_DEF);
    (*def).m_methods = MODULE_METHODS.as_ptr();
    let m = (a.PyModule_Create2)(def, PYTHON_ABI_VERSION);
    if m.is_null() {
        return null_mut();
    }
    if (a.PyModule_AddObjectRef)(m, c"Board".as_ptr(), tp) < 0
        || (a.PyModule_AddIntConstant)(m, c"ACTION_COUNT".as_ptr(), ACTION_COUNT as c_long) < 0
        || (a.PyModule_AddIntConstant)(m, c"HWALL_BASE".as_ptr(), HWALL_BASE as c_long) < 0
        || (a.PyModule_AddIntConstant)(m, c"VWALL_BASE".as_ptr(), VWALL_BASE as c_long) < 0
    {
        (a.Py_DecRef)(m);
        return null_mut();
    }
    m
}
