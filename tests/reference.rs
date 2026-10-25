//! 素朴な参照実装（HashSet と通常の BFS）との突き合わせ・perft の既知値

use std::collections::{HashSet, VecDeque};

use std::hint::black_box;
use std::time::Instant;

use quoridor_rs::{Board, HWALL_BASE, VWALL_BASE};

// ----------------------------------------------------------------------
// 参照実装
// ----------------------------------------------------------------------
#[derive(Clone)]
struct RefState {
    pos: [i32; 2],
    walls: [i32; 2],
    turn: usize,
    hw: HashSet<(i32, i32)>,
    vw: HashSet<(i32, i32)>,
}

fn neighbors(cell: i32, hw: &HashSet<(i32, i32)>, vw: &HashSet<(i32, i32)>) -> Vec<i32> {
    let (c, r) = (cell % 9, cell / 9);
    let mut out = vec![];
    if r > 0 && !hw.contains(&(c, r - 1)) && !hw.contains(&(c - 1, r - 1)) {
        out.push(cell - 9);
    }
    if r < 8 && !hw.contains(&(c, r)) && !hw.contains(&(c - 1, r)) {
        out.push(cell + 9);
    }
    if c > 0 && !vw.contains(&(c - 1, r)) && !vw.contains(&(c - 1, r - 1)) {
        out.push(cell - 1);
    }
    if c < 8 && !vw.contains(&(c, r)) && !vw.contains(&(c, r - 1)) {
        out.push(cell + 1);
    }
    out
}

fn ref_dist(start: i32, goal_row: i32, hw: &HashSet<(i32, i32)>, vw: &HashSet<(i32, i32)>) -> Option<u32> {
    if start / 9 == goal_row {
        return Some(0);
    }
    let mut seen = HashSet::from([start]);
    let mut q = VecDeque::from([(start, 0u32)]);
    while let Some((n, d)) = q.pop_front() {
        for m in neighbors(n, hw, vw) {
            if seen.insert(m) {
                if m / 9 == goal_row {
                    return Some(d + 1);
                }
                q.push_back((m, d + 1));
            }
        }
    }
    None
}

fn ref_wall_ok(h: bool, x: i32, y: i32, hw: &HashSet<(i32, i32)>, vw: &HashSet<(i32, i32)>) -> bool {
    if h {
        !hw.contains(&(x, y)) && !hw.contains(&(x - 1, y)) && !hw.contains(&(x + 1, y)) && !vw.contains(&(x, y))
    } else {
        !vw.contains(&(x, y)) && !vw.contains(&(x, y - 1)) && !vw.contains(&(x, y + 1)) && !hw.contains(&(x, y))
    }
}

fn ref_legal(s: &RefState) -> HashSet<u8> {
    let t = s.turn;
    let (pos, opp) = (s.pos[t], s.pos[1 - t]);
    let mut out = HashSet::new();
    for n in neighbors(pos, &s.hw, &s.vw) {
        if n != opp {
            out.insert(n as u8);
            continue;
        }
        let jump = opp + (opp - pos);
        let on = neighbors(opp, &s.hw, &s.vw);
        if on.contains(&jump) {
            out.insert(jump as u8);
        } else {
            for m in on {
                if m != pos {
                    out.insert(m as u8);
                }
            }
        }
    }
    if s.walls[t] > 0 {
        for h in [true, false] {
            for y in 0..8 {
                for x in 0..8 {
                    if !ref_wall_ok(h, x, y, &s.hw, &s.vw) {
                        continue;
                    }
                    let (mut hw, mut vw) = (s.hw.clone(), s.vw.clone());
                    if h { hw.insert((x, y)); } else { vw.insert((x, y)); }
                    if ref_dist(s.pos[0], 8, &hw, &vw).is_none() || ref_dist(s.pos[1], 0, &hw, &vw).is_none() {
                        continue;
                    }
                    let base = if h { HWALL_BASE } else { VWALL_BASE };
                    out.insert(base + (y * 8 + x) as u8);
                }
            }
        }
    }
    out
}

fn ref_apply(s: &mut RefState, a: u8) {
    let t = s.turn;
    if a < HWALL_BASE {
        s.pos[t] = a as i32;
    } else {
        let (h, w) = if a < VWALL_BASE { (true, a - HWALL_BASE) } else { (false, a - VWALL_BASE) };
        let xy = ((w % 8) as i32, (w / 8) as i32);
        if h { s.hw.insert(xy); } else { s.vw.insert(xy); }
        s.walls[t] -= 1;
    }
    s.turn = 1 - t;
}

// 再現性のある乱数
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[test]
fn matches_reference_on_random_games() {
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    let mut positions = 0;
    for _game in 0..150 {
        let mut b = Board::new();
        let mut s = RefState { pos: [4, 76], walls: [10, 10], turn: 0, hw: HashSet::new(), vw: HashSet::new() };
        let plies = rng.below(70);
        for _ in 0..plies {
            let got: Vec<u8> = b.legal_actions().as_slice().to_vec();
            let got_set: HashSet<u8> = got.iter().copied().collect();
            assert_eq!(got.len(), got_set.len(), "duplicate moves");
            assert_eq!(got_set, ref_legal(&s));
            assert_eq!(b.count_legal_actions() as usize, got.len());
            for p in 0..2 {
                let goal_row = if p == 0 { 8 } else { 0 };
                let d = b.shortest_path(p);
                assert_eq!(d, ref_dist(s.pos[p], goal_row, &s.hw, &s.vw));
                let path = b.shortest_path_nodes(p).unwrap();
                assert_eq!(path.len() as u32 - 1, d.unwrap());
                assert_eq!(path[0] as i32, s.pos[p]);
                assert_eq!(path[path.len() - 1] as i32 / 9, goal_row);
                for w in path.windows(2) {
                    assert!(neighbors(w[0] as i32, &s.hw, &s.vw).contains(&(w[1] as i32)));
                }
            }
            // 個別判定
            for a in 0..=208u8 {
                assert_eq!(b.is_legal(a), got_set.contains(&a), "is_legal({a})");
            }
            // make/undo で完全に元に戻るか
            let snap = (b.open_d, b.open_r, b.hmask, b.vmask, b.pos, b.walls, b.turn, b.hash, b.ply());
            for _ in 0..6 {
                let a = got[rng.below(got.len())];
                b.make(a);
                assert_eq!(b.hash, b.compute_hash());
                b.undo();
                assert_eq!(snap, (b.open_d, b.open_r, b.hmask, b.vmask, b.pos, b.walls, b.turn, b.hash, b.ply()));
            }
            // 壁を多めに選んで進める
            let walls: Vec<u8> = got.iter().copied().filter(|&a| a >= HWALL_BASE).collect();
            let a = if !walls.is_empty() && rng.below(100) < 45 { walls[rng.below(walls.len())] } else { got[rng.below(got.len())] };
            b.make(a);
            ref_apply(&mut s, a);
            assert_eq!(b.hash, b.compute_hash());
            positions += 1;
            if b.winner().is_some() {
                break;
            }
        }
        while b.undo() {}
        assert_eq!((b.hmask, b.vmask, b.pos, b.walls, b.turn), (0, 0, [4, 76], [10, 10], 0));
        assert_eq!(b.hash, Board::new().hash);
    }
    assert!(positions > 3000, "positions = {positions}");
}

#[test]
fn risky_stats_and_consistency() {
    use quoridor_rs::board::cycle_risky_masks;
    // 粗い判定: 盤端か既存の壁に 2 点以上で接する壁
    fn touch_risky_masks(hw: u64, vw: u64) -> (u64, u64) {
        const C0: u64 = 0x0101_0101_0101_0101;
        const C7: u64 = C0 << 7;
        let occ = hw | ((hw << 1) & !C0) | ((hw >> 1) & !C7) | vw | (vw << 8) | (vw >> 8);
        let tl = ((occ << 1) & !C0) | C0;
        let tr = ((occ >> 1) & !C7) | C7;
        let tt = (occ << 8) | 0xFF;
        let tb = (occ >> 8) | (0xFF << 56);
        ((tl & occ) | (tl & tr) | (occ & tr), (tt & occ) | (tt & tb) | (occ & tb))
    }
    let mut rng = Rng(99);
    let (mut coarse, mut exact, mut illegal, mut n) = (0u32, 0u32, 0u32, 0u32);
    for _ in 0..300 {
        let mut b = Board::new();
        for _ in 0..rng.below(60) {
            let ml: Vec<u8> = b.legal_actions().as_slice().to_vec();
            let walls: Vec<u8> = ml.iter().copied().filter(|&a| a >= HWALL_BASE).collect();
            let a = if !walls.is_empty() && rng.below(100) < 45 { walls[rng.below(walls.len())] } else { ml[rng.below(ml.len())] };
            b.make(a);
            if b.winner().is_some() { break; }
            let (vh, vv) = b.valid_wall_masks();
            let (ch, cv) = touch_risky_masks(b.hmask, b.vmask);
            let (eh, ev) = cycle_risky_masks(b.hmask, b.vmask);
            assert_eq!(eh & !ch, 0); assert_eq!(ev & !cv, 0);   // 正確な判定は粗い判定の部分集合
            let (lh, lv) = b.legal_wall_masks();
            // 閉路を作らない壁は必ず合法
            assert_eq!(vh & !eh & !lh, 0); assert_eq!(vv & !ev & !lv, 0);
            coarse += ((vh & ch).count_ones() + (vv & cv).count_ones()) as u32;
            exact += ((vh & eh).count_ones() + (vv & ev).count_ones()) as u32;
            illegal += ((vh & !lh).count_ones() + (vv & !lv).count_ones()) as u32;
            n += 1;
        }
    }
    println!("positions={n} coarse={:.1} exact={:.1} illegal={:.2} per position",
             coarse as f64 / n as f64, exact as f64 / n as f64, illegal as f64 / n as f64);
}

#[test]
fn perft_initial() {
    let mut b = Board::new();
    assert_eq!(b.perft(1), 131);
    assert_eq!(b.perft(2), 16677);
    assert_eq!(b.perft(3), 2062264);
    assert_eq!(b.perft_bulk(3), 2062264);
}

#[test]
fn perft_midgame() {
    // move 13, move 67, hwall(3,3), vwall(5,5), hwall(4,1), move 58
    let mut b = Board::new();
    for a in [13, 67, HWALL_BASE + 27, VWALL_BASE + 45, HWALL_BASE + 12, 58] {
        assert!(b.is_legal(a));
        b.make(a);
    }
    assert_eq!(b.perft(3), 1551223);
    assert_eq!(b.perft_bulk(3), 1551223);
}

#[test]
fn pass_and_repetition() {
    let mut b = Board::new();
    let h0 = b.hash;
    b.make_pass();
    assert_ne!(b.hash, h0);
    b.undo();
    assert_eq!(b.hash, h0);
    // 往復で同じ局面に戻る
    for a in [13, 67, 4, 76] {
        b.make(a);
    }
    assert_eq!(b.hash, h0);
    assert_eq!(b.repetitions(), 1);
    for a in [13, 67, 4, 76] {
        b.make(a);
    }
    assert_eq!(b.repetitions(), 2);
}

#[test]
fn jumps() {
    // 直進ジャンプ
    let b = Board::from_parts([31, 40], [10, 10], 0, 0, 0);
    let m = b.pawn_dest_mask();
    assert!(m >> 49 & 1 == 1 && m >> 40 & 1 == 0);
    // 直進先が水平壁 (3,4) で塞がれている → 斜め
    let b = Board::from_parts([31, 40], [10, 10], 0, 1 << (4 * 8 + 3), 0);
    let m = b.pawn_dest_mask();
    assert!(m >> 49 & 1 == 0 && m >> 39 & 1 == 1 && m >> 41 & 1 == 1);
    // 盤端の相手 → 横のみ
    let b = Board::from_parts([63, 72], [10, 10], 0, 0, 0);
    assert!(b.pawn_dest_mask() >> 73 & 1 == 1);
}

// ----------------------------------------------------------------------
// ベンチマーク（通常のテストでは実行しない）
//   QUORIDOR_POSITIONS=positions.txt cargo test --release --test reference -- --ignored --nocapture
// positions.txt は 1 行 1 局面（初期局面からの手番号をスペース区切り）
// ----------------------------------------------------------------------
fn load(path: Option<String>) -> Vec<Board> {
    let text = match path {
        Some(p) => std::fs::read_to_string(p).expect("positions file"),
        None => String::from("\n13 67 108 190 93 58\n"),
    };
    text.lines()
        .map(|l| {
            let mut b = Board::new();
            for a in l.split_whitespace() {
                b.make(a.parse().unwrap());
            }
            b
        })
        .collect()
}

fn time_per_pos<F: FnMut(&mut Board)>(label: &str, boards: &mut [Board], reps: usize, mut f: F) {
    let t = Instant::now();
    for _ in 0..reps {
        for b in boards.iter_mut() {
            f(b);
        }
    }
    let ns = t.elapsed().as_nanos() as f64 / (reps * boards.len()) as f64;
    println!("  {label:28} {ns:9.1} ns/局面");
}

#[test]
#[ignore]
fn bench() {
    let mut boards = load(std::env::var("QUORIDOR_POSITIONS").ok());
    println!("{} positions", boards.len());

    let first: Vec<u8> = boards.iter().map(|b| b.legal_actions().as_slice()[0]).collect();
    let wall: Vec<Option<u8>> = boards
        .iter()
        .map(|b| b.legal_actions().as_slice().iter().copied().find(|&a| a >= HWALL_BASE))
        .collect();

    let mut i = 0;
    time_per_pos("make+undo (pawn)", &mut boards, 200_000, |b| {
        let a = first[i % first.len()];
        i += 1;
        b.make(black_box(a));
        b.undo();
    });
    let mut i = 0;
    time_per_pos("make+undo (wall)", &mut boards, 200_000, |b| {
        if let Some(a) = wall[i % wall.len()] {
            b.make(black_box(a));
            b.undo();
        }
        i += 1;
    });
    time_per_pos("pawn_dest_mask", &mut boards, 200_000, |b| {
        black_box(b.pawn_dest_mask());
    });
    time_per_pos("shortest_path x2", &mut boards, 200_000, |b| {
        black_box((b.shortest_path(0), b.shortest_path(1)));
    });
    time_per_pos("shortest_path_nodes x2", &mut boards, 50_000, |b| {
        black_box((b.shortest_path_nodes(0), b.shortest_path_nodes(1)));
    });
    time_per_pos("legal_actions (全合法手)", &mut boards, 50_000, |b| {
        black_box(b.legal_actions());
    });

    let mut b = Board::new();
    for (d, bulk) in [(3, false), (3, true), (4, true)] {
        let t = Instant::now();
        let n = if bulk { b.perft_bulk(d) } else { b.perft(d) };
        let s = t.elapsed().as_secs_f64();
        let kind = if bulk { "perft_bulk" } else { "perft" };
        println!("  {kind}({d}) 初期局面 = {n}  {s:.3}s  ({:.1}M nodes/s)", n as f64 / s / 1e6);
    }
}

// ----------------------------------------------------------------------
// 探索
// ----------------------------------------------------------------------
use quoridor_rs::{perft_parallel, Limits, Searcher};
use std::time::Duration;

#[test]
fn search_finds_win_in_one() {
    // P0 が (4,7)、P1 が (4,0) にいる → P0 は (4,8) へ進めば勝ち
    let b = Board::from_parts([67, 4], [10, 10], 0, 0, 0);
    let mut s = Searcher::new(16);
    let r = s.search(&b, &Limits { max_depth: 3, time: None, threads: 1 });
    assert_eq!(r.best, Some(76));
    assert!(r.score >= quoridor_rs::eval::MATE);
}

#[test]
fn search_is_deterministic_and_legal() {
    let mut b = Board::new();
    for a in [13, 67, HWALL_BASE + 27, VWALL_BASE + 45] {
        b.make(a);
    }
    let lim = Limits { max_depth: 4, time: None, threads: 1 };
    let r1 = Searcher::new(16).search(&b, &lim);
    let r2 = Searcher::new(16).search(&b, &lim);
    assert_eq!((r1.best, r1.score, r1.nodes), (r2.best, r2.score, r2.nodes));
    assert!(b.is_legal(r1.best.unwrap()));
    // 複数スレッドでも合法手を返す
    let r = Searcher::new(16).search(&b, &Limits { max_depth: 5, time: None, threads: 4 });
    assert!(b.is_legal(r.best.unwrap()) && r.depth == 5);
    // 時間制限
    let t = std::time::Instant::now();
    let r = Searcher::new(16).search(&b, &Limits { max_depth: 60, time: Some(Duration::from_millis(200)), threads: 2 });
    assert!(t.elapsed() < Duration::from_millis(600), "{:?}", t.elapsed());
    assert!(b.is_legal(r.best.unwrap()));
}

#[test]
fn parallel_perft_matches() {
    let b = Board::new();
    assert_eq!(perft_parallel(&b, 3, 8, false), 2062264);
    assert_eq!(perft_parallel(&b, 3, 8, true), 2062264);
}

#[test]
fn selfplay_finishes() {
    let mut b = Board::new();
    let mut s = Searcher::new(16);
    for _ in 0..200 {
        if b.winner().is_some() {
            break;
        }
        let r = s.search(&b, &Limits { max_depth: 3, time: None, threads: 1 });
        let a = r.best.unwrap();
        assert!(b.is_legal(a));
        b.make(a);
    }
    assert!(b.winner().is_some(), "game did not finish");
}
