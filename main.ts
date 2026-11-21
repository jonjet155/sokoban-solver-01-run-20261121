// Sokoban solver: push-optimal per room, then stitched together.
//
// Structure of every generated level: one hallway (the player's row) with one
// to four small rooms off it, each holding 3-4 boxes and as many goals. A box
// pushed out of a room's door into the hallway can never be pushed back, so
// every box stays in its room and the level's push optimum is the sum of its
// rooms' optima. This solver therefore splits the level into rooms and solves
// each room on its own with an optimal push search, then walks the player
// between rooms along the hallway.
//
// Each room is solved by A* over pushes (a state is the boxes plus the
// player's reachable region; between pushes the player walks freely). The
// heuristic is the cheapest assignment of boxes to goals on push-distance
// (a box's push-distance to a goal is its shortest push path ignoring the
// other boxes, found by pulling backwards from the goal). That heuristic is
// admissible and consistent, so the first solved state A* pops is
// push-optimal for the room. Two prunings keep the search small: a box on a
// square from which no push path reaches any goal is dead, and a box frozen
// inside a 2x2 block of walls and boxes off its goal is dead. If A* runs out
// of its fixed work budget on a room, a greedy best-first search on the same
// heuristic tries to find any solution (a longer solution still scores).
// Every budget is counted in work (expansions), never in wall-clock time, so
// two runs write the same bytes. No clock is read anywhere.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (expansions), never the clock.
const ROOM_ASTAR_BUDGET = 250_000;
const ROOM_GREEDY_BUDGET = 150_000;
const GLOBAL_ASTAR_BUDGET = 400_000;
const GLOBAL_GREEDY_BUDGET = 200_000;

const DIRS = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];

/** A level in XSB text: width, height, walls, goals, boxes, player square. */
function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * For one goal, the push-distance of every square: the fewest pushes that
 * bring a box from that square onto the goal, ignoring the other boxes.
 * Found by pulling backwards from the goal: a box on square y can be pulled
 * to x = y + d (player standing at y - d) exactly when a push from y in
 * direction d lands on x, so reverse the pushes out of the goal.
 */
function pushDistances(w, wall, goal) {
  const size = wall.length;
  const dist = new Int32Array(size).fill(-1);
  dist[goal] = 0;
  const queue = [goal];
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (let k = 0; k < 4; k++) {
      const d = k === 0 ? -w : k === 1 ? w : k === 2 ? -1 : 1;
      const y = x - d;
      if (y < 0 || y >= size || wall[y] || wall[y - d] || dist[y] >= 0) continue;
      dist[y] = dist[x] + 1;
      queue.push(y);
    }
  }
  return dist;
}

/**
 * The level's rooms: cut the hallway row's dead squares (floor squares on
 * the player's row from which no push path reaches a goal — a box there can
 * only slide along the hallway and never come back through a door) out of
 * the floor, and flood-fill what is left. Each part holding boxes is a room.
 * Null when the level is no set of rooms (a box in the hallway, a goal
 * outside every part, or a part whose boxes and goals do not match).
 */
function roomsOf(level) {
  const { w, wall, goals, boxes, player } = level;
  const size = wall.length;
  const live = new Uint8Array(size);
  for (const g of goals) {
    const dist = pushDistances(w, wall, g);
    for (let i = 0; i < size; i++) if (dist[i] >= 0) live[i] = 1;
  }
  const row = Math.floor(player / w);
  const cut = new Uint8Array(size);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(size).fill(-1);
  const rooms = [];
  const stack = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    stack.length = 0;
    stack.push(s);
    while (stack.length > 0) {
      const p = stack.pop();
      for (let k = 0; k < 4; k++) {
        const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
          stack.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) {
    if (part[b] < 0) return null;
    rooms[part[b]].boxes.push(b);
  }
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * A* (or greedy best-first when `greedy`) over pushes for one sub-problem:
 * walls, goals, starting boxes and player on a board w wide. Returns
 * {pushes, moves} or null when the work budget runs out. `budget` counts
 * expanded states, never time.
 */
function searchPushes(w, wall, goals, boxes, player, budget, greedy) {
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  // Push-distance of every square to every goal; a box whose distance to
  // every goal is unreachable is dead, so a state holding one is pruned.
  const dists = goals.map((g) => pushDistances(w, wall, g));
  const live = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    for (const d of dists) if (d[i] >= 0) { live[i] = 1; break; }
  }
  if (boxes.some((b) => !live[b])) return null;

  // Cheapest assignment of boxes to goals on push-distance (brute force over
  // the few permutations, with branch-and-bound pruning). Infinity when no
  // box can reach some goal it must take.
  const heuristic = (bs) => {
    const n = bs.length;
    const used = new Array(n).fill(false);
    let best = Infinity;
    const assign = (i, cost) => {
      if (cost >= best) return;
      if (i === n) { best = cost; return; }
      for (let j = 0; j < n; j++) {
        if (used[j]) continue;
        const d = dists[j][bs[i]];
        if (d < 0) continue;
        used[j] = true;
        assign(i + 1, cost + d);
        used[j] = false;
      }
    };
    assign(0, 0);
    return best;
  };

  // Flood fill from `from` with `occupied` in the way; marks `seenLow` with
  // `stampLow` and returns the region's lowest square (the region's name).
  // NOTE: this uses its own stamp array, separate from the expansion flood
  // fill below: it runs inside the push loop, and sharing the marks would
  // overwrite the expansion's region and falsely reject later pushes.
  const seenLow = new Int32Array(size);
  const floodStack = new Int32Array(size);
  let stampLow = 0;
  const regionLow = (occupied, from) => {
    const s = ++stampLow;
    let low = from;
    let n = 0;
    floodStack[n++] = from;
    seenLow[from] = s;
    while (n > 0) {
      const p = floodStack[--n];
      if (p < low) low = p;
      for (let k = 0; k < 4; k++) {
        const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
        if (!wall[q] && !occupied[q] && seenLow[q] !== s) {
          seenLow[q] = s;
          floodStack[n++] = q;
        }
      }
    }
    return low;
  };

  // A box is frozen in a 2x2 block of walls and boxes with a box off its
  // goal: no push can ever free it, so a state holding one is pruned.
  const frozenBlock = (occupied, t) => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const b = a + 1;
      const c = a + w;
      const d = c + 1;
      if (a < 0 || d >= size) continue;
      if (!(wall[a] || occupied[a])) continue;
      if (!(wall[b] || occupied[b])) continue;
      if (!(wall[c] || occupied[c])) continue;
      if (!(wall[d] || occupied[d])) continue;
      if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) ||
          (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
    }
    return false;
  };

  // State keys: boxes plus the player's region, as one number while it fits
  // in a safe integer, else as a string.
  const useNumberKey = (boxes.length + 1) * Math.log2(size) < 53;
  const keyOf = (bs, low) => {
    if (!useNumberKey) return `${low}|${bs.join(',')}`;
    let k = low;
    for (const b of bs) k = k * size + b;
    return k;
  };

  const solved = (bs) => bs.every((b) => goal[b]);
  const occupied = new Uint8Array(size);
  const startBoxes = boxes.slice().sort((a, b) => a - b);
  if (solved(startBoxes)) return { pushes: 0, moves: '' };

  // States: boxes, player square after the push that made them, parent,
  // the box square the push started from, the push's direction, and the
  // state's key (boxes + region), used to drop stale heap entries.
  const statesBoxes = [startBoxes];
  const statesPlayer = [player];
  const parent = [-1];
  const pushFrom = [-1];
  const pushDir = [-1];
  const stateG = [0];
  const stateKey = [null];
  const best = new Map();
  for (const b of startBoxes) occupied[b] = 1;
  stateKey[0] = keyOf(startBoxes, regionLow(occupied, player));
  best.set(stateKey[0], 0);
  for (const b of startBoxes) occupied[b] = 0;

  // Binary heap on (f, -g, insertion order): lowest f first, deeper first on
  // ties, earliest inserted first after that. Fully deterministic.
  const heap = [];
  let seq = 0;
  const h0 = heuristic(startBoxes);
  if (!Number.isFinite(h0)) return null;
  const heapLess = (a, b) => a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e) => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i], heap[p])) break;
      const t = heap[i]; heap[i] = heap[p]; heap[p] = t;
      i = p;
    }
  };
  const heapPop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1], heap[c])) c++;
        if (!heapLess(heap[c], heap[i])) break;
        const t = heap[i]; heap[i] = heap[c]; heap[c] = t;
        i = c;
      }
    }
    return top;
  };
  heapPush([h0, 0, seq++, 0]);

  // The expansion flood fill's own marks (see regionLow's note above).
  const seen = new Int32Array(size);
  let stamp = 0;

  let expanded = 0;
  let found = -1;
  while (heap.length > 0 && expanded < budget) {
    const [, g, , s] = heapPop();
    if (best.get(stateKey[s]) !== g) continue; // stale: a better path to this state exists
    const bs = statesBoxes[s];
    if (solved(bs)) { found = s; break; }
    expanded++;
    const p = statesPlayer[s];
    for (const b of bs) occupied[b] = 1;
    const here = ++stamp;
    // Reachable region from the player's square, marked with this stamp.
    let rn = 0;
    floodStack[rn++] = p;
    seen[p] = here;
    while (rn > 0) {
      const q = floodStack[--rn];
      for (let k = 0; k < 4; k++) {
        const r = k === 0 ? q - w : k === 1 ? q + w : k === 2 ? q - 1 : q + 1;
        if (!wall[r] && !occupied[r] && seen[r] !== here) {
          seen[r] = here;
          floodStack[rn++] = r;
        }
      }
    }
    for (let bi = 0; bi < bs.length && found < 0; bi++) {
      const b = bs[bi];
      for (let k = 0; k < 4; k++) {
        const d = k === 0 ? -w : k === 1 ? w : k === 2 ? -1 : 1;
        const from = b - d;
        const t = b + d;
        if (seen[from] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        if (frozenBlock(occupied, t)) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const next = bs.slice();
        next[bi] = t;
        next.sort((a, b2) => a - b2);
        const low = regionLow(occupied, b);
        const key = keyOf(next, low);
        occupied[t] = 0;
        occupied[b] = 1;
        const prevG = best.get(key);
        if (prevG !== undefined && prevG <= g + 1) continue;
        const h = heuristic(next);
        if (!Number.isFinite(h)) continue;
        best.set(key, g + 1);
        const idx = statesBoxes.length;
        statesBoxes.push(next);
        statesPlayer.push(b);
        parent.push(s);
        pushFrom.push(b);
        pushDir.push(k);
        stateG.push(g + 1);
        stateKey.push(key);
        heapPush([greedy ? h : g + 1 + h, g + 1, seq++, idx]);
      }
    }
    for (const b of bs) occupied[b] = 0;
  }
  if (found < 0) return null;

  // The path of pushes, then the moves: walk to each push's square, push.
  const path = [];
  for (let s = found; s > 0; s = parent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let curPlayer = player;
  let curBoxes = startBoxes;
  for (const s of path) {
    const nextBoxes = statesBoxes[s];
    const from = pushFrom[s];
    const k = pushDir[s];
    const d = k === 0 ? -w : k === 1 ? w : k === 2 ? -1 : 1;
    for (const b of curBoxes) occupied[b] = 1;
    moves += walkTo(w, wall, occupied, curPlayer, from - d);
    for (const b of curBoxes) occupied[b] = 0;
    moves += DIRS[k][2];
    curPlayer = from;
    curBoxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

/** Shortest walk from `from` to `to` around walls and boxes, lower LURD letters. */
function walkTo(w, wall, occupied, from, to) {
  if (from === to) return '';
  const size = wall.length;
  const prev = new Int32Array(size).fill(-1);
  const how = new Array(size);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = DIRS[k][2];
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/** The player's square after `moves` (LURD, case aside) from `from` on a board w wide. */
function endOf(w, from, moves) {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/** Solve one level: room by room when it splits into rooms, else as a whole. */
function solveLevel(text) {
  const level = parseLevel(text);
  const rooms = roomsOf(level);
  if (rooms === null) {
    const whole = searchPushes(level.w, level.wall, level.goals, level.boxes, level.player, GLOBAL_ASTAR_BUDGET, false)
      ?? searchPushes(level.w, level.wall, level.goals, level.boxes, level.player, GLOBAL_GREEDY_BUDGET, true);
    return whole ? whole.moves : '';
  }
  let player = level.player;
  let moves = '';
  const done = new Set();
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      const src = done.has(other) ? other.goals : other.boxes;
      for (const b of src) wall[b] = 1;
    }
    const solved = searchPushes(level.w, wall, room.goals, room.boxes, player, ROOM_ASTAR_BUDGET, false)
      ?? searchPushes(level.w, wall, room.goals, room.boxes, player, ROOM_GREEDY_BUDGET, true);
    if (solved === null) return '';
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    done.add(room);
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  if (moves.length > 60000) moves = '';
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
