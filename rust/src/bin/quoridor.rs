//! コマンドライン: cargo run --release --bin quoridor -- <サブコマンド>
//!
//!   perft <深さ> [--threads N] [--bulk]
//!   search [--depth D] [--time MS] [--threads N] [手 ...]
//!   selfplay [--games N] [--jobs J] [--openings K] [--seed S] [--a 設定] [--b 設定]
//!   bench [--depth D] [--threads N]
//!
//! 手の表記: 手番号（0〜208）、または m13 / h3,3 / v5,5（コマの移動先・水平壁・垂直壁）
//! selfplay の設定: "depth=4" / "time=100" / "depth=6,threads=4" など（カンマ区切り）

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use quoridor_rs::{decode_action, encode_move, perft_parallel, Board, Limits, Move, Searcher};

fn usage() -> ! {
    eprintln!(
        "usage:\n  quoridor perft <depth> [--threads N] [--bulk]\n  quoridor search [--depth D] [--time MS] [--threads N] [moves...]\n  quoridor selfplay [--games N] [--jobs J] [--openings K] [--seed S] [--a depth=4] [--b time=100]\n  quoridor bench [--depth D] [--threads N]"
    );
    std::process::exit(2)
}

fn parse_move(s: &str) -> Option<u8> {
    if let Ok(a) = s.parse::<u8>() {
        return (a < 209).then_some(a);
    }
    let (kind, rest) = s.split_at(1);
    let nums: Vec<u8> = rest.split(',').filter_map(|x| x.parse().ok()).collect();
    let m = match (kind, nums.as_slice()) {
        ("m", [n]) => Move::Pawn(*n),
        ("h", [x, y]) => Move::HWall(*x, *y),
        ("v", [x, y]) => Move::VWall(*x, *y),
        _ => return None,
    };
    encode_move(m)
}

fn fmt_move(a: u8) -> String {
    match decode_action(a) {
        Some(Move::Pawn(n)) => format!("m{n}"),
        Some(Move::HWall(x, y)) => format!("h{x},{y}"),
        Some(Move::VWall(x, y)) => format!("v{x},{y}"),
        None => "?".into(),
    }
}

struct Args {
    flags: Vec<(String, Option<String>)>,
    rest: Vec<String>,
}

impl Args {
    fn parse(v: &[String], bool_flags: &[&str]) -> Args {
        let mut flags = vec![];
        let mut rest = vec![];
        let mut i = 0;
        while i < v.len() {
            if let Some(name) = v[i].strip_prefix("--") {
                if bool_flags.contains(&name) {
                    flags.push((name.to_string(), None));
                } else {
                    i += 1;
                    flags.push((name.to_string(), v.get(i).cloned()));
                }
            } else {
                rest.push(v[i].clone());
            }
            i += 1;
        }
        Args { flags, rest }
    }
    fn get<T: std::str::FromStr>(&self, name: &str) -> Option<T> {
        self.flags.iter().rev().find(|(n, _)| n == name).and_then(|(_, v)| v.as_ref()?.parse().ok())
    }
    fn has(&self, name: &str) -> bool {
        self.flags.iter().any(|(n, _)| n == name)
    }
}

fn board_from(moves: &[String]) -> Board {
    let mut b = Board::new();
    for s in moves {
        let a = parse_move(s).unwrap_or_else(|| {
            eprintln!("invalid move: {s}");
            std::process::exit(2)
        });
        if !b.is_legal(a) {
            eprintln!("illegal move: {s}");
            std::process::exit(2)
        }
        b.make(a);
    }
    b
}

fn parse_limits(spec: &str) -> Limits {
    let mut l = Limits { max_depth: 4, time: None, threads: 1 };
    for kv in spec.split(',').filter(|s| !s.is_empty()) {
        let (k, v) = kv.split_once('=').unwrap_or_else(|| usage());
        let n: u64 = v.parse().unwrap_or_else(|_| usage());
        match k {
            "depth" => l.max_depth = n as u32,
            "time" => {
                l.time = Some(Duration::from_millis(n));
                if !spec.contains("depth") {
                    l.max_depth = 60;
                }
            }
            "threads" => l.threads = n as usize,
            _ => usage(),
        }
    }
    l
}

fn cmd_perft(a: &Args) {
    let depth: u32 = a.rest.first().and_then(|s| s.parse().ok()).unwrap_or_else(|| usage());
    let threads = a.get("threads").unwrap_or(1);
    let b = board_from(&a.rest[1..]);
    let t = Instant::now();
    let n = perft_parallel(&b, depth, threads, a.has("bulk"));
    let s = t.elapsed().as_secs_f64();
    println!("perft({depth}) = {n}  {s:.3}s  {:.1}M nodes/s  (threads={threads})", n as f64 / s / 1e6);
}

fn cmd_search(a: &Args) {
    let mut lim = Limits { max_depth: a.get("depth").unwrap_or(4), time: None, threads: a.get("threads").unwrap_or(1) };
    if let Some(ms) = a.get::<u64>("time") {
        lim.time = Some(Duration::from_millis(ms));
        if a.get::<u32>("depth").is_none() {
            lim.max_depth = 60;
        }
    }
    let b = board_from(&a.rest);
    let r = Searcher::new(a.get("tt").unwrap_or(64)).search(&b, &lim);
    let s = r.elapsed.as_secs_f64();
    println!(
        "best {}  score {:.2}  depth {}  nodes {}  {:.3}s  {:.2}M nodes/s",
        r.best.map_or("none".into(), fmt_move),
        r.score as f64 / 100.0,
        r.depth,
        r.nodes,
        s,
        r.nodes as f64 / s.max(1e-9) / 1e6
    );
}

fn cmd_bench(a: &Args) {
    // 深さ 3 の自己対局から 8 手ごとの局面を取り、各局面を深さ D で探索してノード数と速度を測る
    let depth = a.get("depth").unwrap_or(6);
    let threads = a.get("threads").unwrap_or(1);
    let mut positions = vec![];
    let mut b = Board::new();
    let mut gen = Searcher::new(16);
    for ply in 0..64 {
        if b.winner().is_some() {
            break;
        }
        if ply % 8 == 0 {
            positions.push(b.clone());
        }
        let mv = gen.search(&b, &Limits { max_depth: 3, time: None, threads: 1 }).best.unwrap();
        b.make(mv);
    }
    let (mut nodes, mut secs) = (0u64, 0.0f64);
    for p in &positions {
        let r = Searcher::new(64).search(p, &Limits { max_depth: depth, time: None, threads });
        println!(
            "  ply {:>2}: best {:6} score {:7.2} depth {} nodes {:>9} {:.3}s",
            p.ply(),
            r.best.map_or("none".into(), fmt_move),
            r.score as f64 / 100.0,
            r.depth,
            r.nodes,
            r.elapsed.as_secs_f64()
        );
        nodes += r.nodes;
        secs += r.elapsed.as_secs_f64();
    }
    println!("bench depth={depth} threads={threads}: {nodes} nodes  {secs:.3}s  {:.2}M nodes/s", nodes as f64 / secs / 1e6);
    for (name, n) in quoridor_rs::stats::snapshot() {
        println!("  {name:24} {n:>12}  ({:.2} / node)", n as f64 / nodes as f64);
    }
}

/// 2 つの設定で対局させる。序盤 K 手をランダムに指し、先後を入れ替えて 2 局ずつ。
fn cmd_selfplay(a: &Args) {
    let games: usize = a.get("games").unwrap_or(20);
    let jobs: usize = a.get("jobs").unwrap_or(1);
    let openings: usize = a.get("openings").unwrap_or(4);
    let seed: u64 = a.get("seed").unwrap_or(1);
    let la = parse_limits(&a.get::<String>("a").unwrap_or("depth=4".into()));
    let lb = parse_limits(&a.get::<String>("b").unwrap_or("depth=4".into()));
    println!("A = {la:?}\nB = {lb:?}\ngames = {games} (openings {openings} random plies), jobs = {jobs}");

    let next = AtomicUsize::new(0);
    let score = Mutex::new((0.0f64, 0.0f64, 0usize, Duration::ZERO, Duration::ZERO));
    let t = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| loop {
                let g = next.fetch_add(1, Ordering::Relaxed);
                if g >= games {
                    return;
                }
                // 序盤（ゲーム番号の偶奇で先後を入れ替え、同じ序盤を 2 局使う）
                let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((g / 2) as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03);
                let mut b = Board::new();
                for _ in 0..openings {
                    let mut m = b.pawn_dest_mask();
                    let n = m.count_ones() as u64;
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    for _ in 0..(rng % n) {
                        m &= m - 1;
                    }
                    b.make(m.trailing_zeros() as u8);
                }
                let a_first = g % 2 == 0;
                let (mut sa, mut sb) = (Searcher::new(32), Searcher::new(32));
                let (mut ta, mut tb) = (Duration::ZERO, Duration::ZERO);
                let mut winner = None;
                for _ in 0..200 {
                    if let Some(w) = b.winner() {
                        winner = Some(w);
                        break;
                    }
                    let a_turn = (b.turn == 0) == a_first;
                    let r = if a_turn { sa.search(&b, &la) } else { sb.search(&b, &lb) };
                    if a_turn { ta += r.elapsed } else { tb += r.elapsed }
                    match r.best {
                        Some(mv) => b.make(mv),
                        None => {
                            winner = Some(b.turn ^ 1);
                            break;
                        }
                    }
                }
                let mut sc = score.lock().unwrap();
                match winner {
                    Some(w) if (w == 0) == a_first => sc.0 += 1.0,
                    Some(_) => sc.1 += 1.0,
                    None => {
                        sc.0 += 0.5;
                        sc.1 += 0.5;
                    }
                }
                sc.2 += 1;
                sc.3 += ta;
                sc.4 += tb;
                if sc.2 % 10 == 0 || sc.2 == games {
                    println!("  {}/{} games: A {} - B {}", sc.2, games, sc.0, sc.1);
                }
            });
        }
    });
    let sc = score.into_inner().unwrap();
    println!(
        "result: A {} - B {}  (A {:.1}%)  think time A {:.1}s B {:.1}s  wall {:.1}s",
        sc.0,
        sc.1,
        100.0 * sc.0 / games as f64,
        sc.3.as_secs_f64(),
        sc.4.as_secs_f64(),
        t.elapsed().as_secs_f64()
    );
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = argv.first() else { usage() };
    let a = Args::parse(&argv[1..], &["bulk"]);
    match cmd.as_str() {
        "perft" => cmd_perft(&a),
        "search" => cmd_search(&a),
        "selfplay" => cmd_selfplay(&a),
        "bench" => cmd_bench(&a),
        _ => usage(),
    }
}
