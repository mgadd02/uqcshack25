# modules/pathfinding_algos.py
from __future__ import annotations
from typing import Dict, Tuple, List, Callable, Optional
from heapq import heappush, heappop
import collections
import random
import numpy as np
from time import monotonic

ProgressCB = Optional[Callable[[str, int, str], bool]]
# progress(phase:str, percent:int, note:str) -> return False to cancel

# Allowed moves (no left). Diagonals require both orthogonals free (enforced below).
MOVES: Dict[str, Tuple[int, int]] = {
    "R":  (1,  0),
    "UR": (1, -1),
    "DR": (1,  1),
    "U":  (0, -1),
    "D":  (0,  1),
}

# ---------- Helpers ----------
def _in_bounds(occ: np.ndarray, c: int, r: int) -> bool:
    return 0 <= r < occ.shape[0] and 0 <= c < occ.shape[1]

def _free(occ: np.ndarray, c: int, r: int) -> bool:
    return _in_bounds(occ, c, r) and occ[r, c] == 0

def _neighbors_free(occ: np.ndarray, c: int, r: int):
    """Neighbors that are free, with NO CORNER CUTTING for diagonals."""
    if _free(occ, c+1, r):           # Right
        yield "R", (c+1, r)
    if _free(occ, c, r-1):           # Up
        yield "U", (c, r-1)
    if _free(occ, c, r+1):           # Down
        yield "D", (c, r+1)
    if _free(occ, c+1, r) and _free(occ, c, r-1) and _free(occ, c+1, r-1):  # UR
        yield "UR", (c+1, r-1)
    if _free(occ, c+1, r) and _free(occ, c, r+1) and _free(occ, c+1, r+1):  # DR
        yield "DR", (c+1, r+1)

def _neighbors_soft(occ: np.ndarray, c: int, r: int, obstacle_cost: float, repel: Optional[np.ndarray], repel_w: float):
    """
    Soft neighbors that *can* step into obstacles (costly), discouraging diagonal corner cuts by
    charging partial penalty if an orthogonal is blocked.
    Includes optional repel cost from enemies.
    """
    for m, (dc, dr) in MOVES.items():
        nc, nr = c + dc, r + dr
        if not _in_bounds(occ, nc, nr):
            continue
        step = 1.0
        if occ[nr, nc] != 0:
            step += obstacle_cost
        if m == "UR":
            if not _in_bounds(occ, c+1, r) or not _in_bounds(occ, c, r-1):
                continue
            if not _free(occ, c+1, r): step += obstacle_cost * 0.5
            if not _free(occ, c, r-1): step += obstacle_cost * 0.5
        elif m == "DR":
            if not _in_bounds(occ, c+1, r) or not _in_bounds(occ, c, r+1):
                continue
            if not _free(occ, c+1, r): step += obstacle_cost * 0.5
            if not _free(occ, c, r+1): step += obstacle_cost * 0.5
        if repel is not None:
            step += repel_w * float(repel[nr, nc])
        yield m, (nc, nr), step

def _reconstruct(came, start, goal):
    # Robust: tolerate partial paths
    if goal != start and goal not in came:
        # pick closest reachable key if possible
        if not came:
            return []
        # choose last inserted as best partial
        end = next(reversed(came))
        goal = end
    cur = goal
    out: List[Tuple[str, Tuple[int, int]]] = []
    while cur != start and cur in came:
        pmove, prev = came[cur]
        out.append((pmove, cur))
        cur = prev
    out.reverse()
    return out

def _manhattan(a, b):
    (c1, r1), (c2, r2) = a, b
    return abs(c1 - c2) + abs(r1 - r2)

def _progress(progress: ProgressCB, phase: str, num: int, den: int, note: str = "") -> bool:
    if not progress:
        return True
    try:
        pct = int(min(100, (num / max(1, den)) * 100))
    except Exception:
        pct = 0
    return progress(phase, pct, note)

# ---------- Core planners ----------
def _astar_like(occ, start, goal, progress: ProgressCB, timeout_s: float,
                heuristic=True, repel: Optional[np.ndarray]=None, repel_w: float=0.0):
    if start == goal:
        return []
    g = {start: 0.0}
    came = {}
    pq = []
    heappush(pq, ((_manhattan(start, goal) if heuristic else 0.0), start))
    seen = set()
    max_exp = max(1000, int(occ.size * 5))
    exp = 0
    t0 = monotonic()
    best = start; best_f = _manhattan(start, goal)

    while pq and exp < max_exp:
        if monotonic() - t0 > timeout_s:
            break
        f, cur = heappop(pq); exp += 1
        if cur in seen:
            continue
        seen.add(cur)

        cur_f = g[cur] + (_manhattan(cur, goal) if heuristic else 0.0) + (repel_w * float(repel[cur[1], cur[0]]) if repel is not None else 0.0)
        if cur_f < best_f:
            best_f, best = cur_f, cur

        if not _progress(progress, "Pathfinding", exp, max_exp, f"{exp}/{max_exp}"):
            break

        if cur == goal:
            return _reconstruct(came, start, goal)

        # use soft-neighbors only for repel cost accounting but still forbid stepping into walls here
        for m, nxt in _neighbors_free(occ, *cur):
            base = 1.0
            if repel is not None:
                base += repel_w * float(repel[nxt[1], nxt[0]])
            ng = g[cur] + base
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came[nxt] = (m, cur)
                heappush(pq, (ng + (_manhattan(nxt, goal) if heuristic else 0.0), nxt))
    return _reconstruct(came, start, best)

def astar(occ, start, goal, progress: ProgressCB = None, timeout_s: float = 1.5,
          repel: Optional[np.ndarray]=None, repel_w: float=0.0):
    return _astar_like(occ, start, goal, progress, timeout_s, heuristic=True, repel=repel, repel_w=repel_w)

def dijkstra(occ, start, goal, progress: ProgressCB = None, timeout_s: float = 1.5,
             repel: Optional[np.ndarray]=None, repel_w: float=0.0):
    return _astar_like(occ, start, goal, progress, timeout_s, heuristic=False, repel=repel, repel_w=repel_w)

def greedy_best_first(occ, start, goal, progress: ProgressCB = None, timeout_s: float = 1.0,
                      repel: Optional[np.ndarray]=None, repel_w: float=0.0):
    if start == goal:
        return []
    came = {}
    pq = []
    heappush(pq, (_manhattan(start, goal), start))
    seen = set()
    max_exp = max(800, int(occ.size * 4))
    exp = 0
    t0 = monotonic()
    best = start; best_h = _manhattan(start, goal)
    while pq and exp < max_exp:
        if monotonic() - t0 > timeout_s:
            break
        _, cur = heappop(pq); exp += 1
        if cur in seen:
            continue
        seen.add(cur)
        h = _manhattan(cur, goal)
        if h < best_h:
            best_h, best = h, cur
        if not _progress(progress, "Pathfinding", exp, max_exp, f"{exp}/{max_exp}"):
            break
        if cur == goal:
            return _reconstruct(came, start, goal)
        for m, nxt in _neighbors_free(occ, *cur):
            if nxt not in seen:
                came[nxt] = (m, cur)
                heappush(pq, (_manhattan(nxt, goal) + (repel_w * float(repel[nxt[1], nxt[0]]) if repel is not None else 0.0), nxt))
    return _reconstruct(came, start, best)

def bfs(occ, start, goal, progress: ProgressCB = None, timeout_s: float = 1.0,
        repel: Optional[np.ndarray]=None, repel_w: float=0.0):
    if start == goal:
        return []
    came = {}
    q = collections.deque([start])
    seen = {start}
    max_exp = max(800, int(occ.size * 4))
    exp = 0
    t0 = monotonic()
    best = start; best_h = _manhattan(start, goal)
    while q and exp < max_exp:
        if monotonic() - t0 > timeout_s:
            break
        cur = q.popleft(); exp += 1
        h = _manhattan(cur, goal)
        if h < best_h:
            best_h, best = h, cur
        if not _progress(progress, "Pathfinding", exp, max_exp, f"{exp}/{max_exp}"):
            break
        if cur == goal:
            return _reconstruct(came, start, goal)
        for m, nxt in _neighbors_free(occ, *cur):
            if nxt not in seen:
                seen.add(nxt)
                came[nxt] = (m, cur)
                q.append(nxt)
    return _reconstruct(came, start, best)

# ---------- Soft A* (explicit only) ----------
def soft_astar(occ, start, goal, obstacle_cost: float = 320.0,
               progress: ProgressCB = None, timeout_s: float = 1.5,
               repel: Optional[np.ndarray]=None, repel_w: float=0.0):
    if start == goal:
        return []
    g = {start: 0.0}
    came = {}
    pq = []
    heappush(pq, (_manhattan(start, goal), start))
    seen = set()
    max_exp = max(1200, int(occ.size * 6))
    exp = 0
    t0 = monotonic()
    best = start; best_f = _manhattan(start, goal)
    while pq and exp < max_exp:
        if monotonic() - t0 > timeout_s:
            break
        _, cur = heappop(pq); exp += 1
        if cur in seen:
            continue
        seen.add(cur)
        f = g[cur] + _manhattan(cur, goal) + (repel_w * float(repel[cur[1], cur[0]]) if repel is not None else 0.0)
        if f < best_f:
            best_f, best = f, cur
        if not _progress(progress, "Pathfinding", exp, max_exp, f"{exp}/{max_exp}"):
            break
        if cur == goal:
            return _reconstruct(came, start, goal)
        for m, nxt, step_cost in _neighbors_soft(occ, *cur, obstacle_cost=obstacle_cost, repel=repel, repel_w=repel_w):
            ng = g[cur] + step_cost
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came[nxt] = (m, cur)
                heappush(pq, (ng + _manhattan(nxt, goal), nxt))
    return _reconstruct(came, start, best)

# ---------- Fun planners (now bounded & robust) ----------
def worst_astar(occ, start, goal, chaos: float = 0.15, progress: ProgressCB = None, timeout_s: float = 1.0,
                repel: Optional[np.ndarray]=None, repel_w: float=0.0):
    """Silly priority but bounded, with greedy tail. Returns partial if needed."""
    if start == goal:
        return []
    came = {}
    g = {start: 0.0}
    rnd = random.Random(1337)
    pq = []
    def pri(cur):
        base = -_manhattan(cur, goal) + 0.75 * g[cur]
        if repel is not None:
            base += -0.35 * float(repel[cur[1], cur[0]])  # dislike enemies a bit
        return base + chaos * rnd.random()
    heappush(pq, (pri(start), start))
    seen = set()
    max_exp = max(900, int(occ.size * 3))
    exp = 0
    t0 = monotonic()
    best = start; best_h = _manhattan(start, goal)
    while pq and exp < max_exp:
        if monotonic() - t0 > timeout_s:
            break
        _, cur = heappop(pq); exp += 1
        if cur in seen:
            continue
        seen.add(cur)
        h = _manhattan(cur, goal)
        if h < best_h:
            best_h, best = h, cur
        if not _progress(progress, "Pathfinding", exp, max_exp, f"{exp}/{max_exp}"):
            break
        if cur == goal:
            return _reconstruct(came, start, goal)
        neigh = list(_neighbors_free(occ, *cur))
        rnd.shuffle(neigh)
        for m, nxt in neigh:
            ng = g[cur] + 1.0 + (repel_w * float(repel[nxt[1], nxt[0]]) if repel is not None else 0.0)
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came[nxt] = (m, cur)
                heappush(pq, (pri(nxt), nxt))
    tail = greedy_best_first(occ, best, goal, progress=None, timeout_s=max(0.25, timeout_s - (monotonic()-t0)),
                             repel=repel, repel_w=repel_w)
    for mv, cell in tail:
        came[cell] = (mv, best); best = cell
    return _reconstruct(came, start, best)

def zigzag(occ, start, goal, amplitude_cols: int = 1, progress: ProgressCB = None, timeout_s: float = 0.8,
           repel: Optional[np.ndarray]=None, repel_w: float=0.0):
    """Non-hanging zigzag with rightward bias and greedy finish."""
    if start == goal:
        return []
    came = {}
    cur = start
    steps = 0
    max_steps = max(1000, int(occ.size * 2.0))
    going_up = True
    base_col = start[0]
    t0 = monotonic()
    stall = 0; last_right = start[0]; STALL = max(10, occ.shape[0] // 5)

    def tick():
        nonlocal steps
        steps += 1
        if monotonic() - t0 > timeout_s:
            return False
        return _progress(progress, "Pathfinding", steps, max_steps, f"{steps}/{max_steps}")

    while steps < max_steps and cur != goal:
        if not tick():
            break
        c, r = cur
        if (c - base_col) >= max(1, amplitude_cols):
            base_col = c
            going_up = not going_up
        prefs = [("UR", (c+1, r-1)), ("R", (c+1, r)), ("U", (c, r-1)), ("DR", (c+1, r+1))] if going_up \
                else [("DR", (c+1, r+1)), ("R", (c+1, r)), ("D", (c, r+1)), ("UR", (c+1, r-1))]
        moved = False
        valid = dict(_neighbors_free(occ, c, r))
        # slight repel-aware preference
        prefs.sort(key=lambda pr: (0 if pr[0] not in valid else (repel_w * float(repel[valid[pr[0]][1], valid[pr[0]][0]]) if repel is not None else 0.0)))
        for m, nxt in prefs:
            if m in valid and valid[m] == nxt:
                came[nxt] = (m, cur); cur = nxt; moved = True; break
        if not moved:
            alts = list(_neighbors_free(occ, c, r))
            if not alts:
                break
            # choose least-repel
            alts.sort(key=lambda mn: (repel_w * float(repel[mn[1][1], mn[1][0]]) if repel is not None else 0.0))
            m, nxt = alts[0]
            came[nxt] = (m, cur); cur = nxt
        if cur[0] > last_right:
            last_right = cur[0]; stall = 0
        else:
            stall += 1
        if stall >= STALL:
            break
        if cur[0] >= goal[0] and abs(cur[1] - goal[1]) <= 1:
            break
    tail = greedy_best_first(occ, cur, goal, progress=None, timeout_s=max(0.25, timeout_s - (monotonic()-t0)),
                             repel=repel, repel_w=repel_w)
    for mv, cell in tail:
        came[cell] = (mv, cur); cur = cell
    return _reconstruct(came, start, cur)

def random_greedy(occ, start, goal, right_bias: float = 0.7, seed: int = 42,
                  progress: ProgressCB = None, timeout_s: float = 0.8,
                  repel: Optional[np.ndarray]=None, repel_w: float=0.0):
    """Right-biased random walk with greedy tie-break; bounded with stall guard."""
    if start == goal:
        return []
    rnd = random.Random(seed)
    came = {}
    cur = start
    max_steps = max(1000, int(occ.size * 2.0))
    steps = 0
    t0 = monotonic()
    stall = 0; last_right = start[0]; STALL = max(12, occ.shape[0] // 4)

    while steps < max_steps and cur != goal:
        steps += 1
        if monotonic() - t0 > timeout_s:
            break
        if not _progress(progress, "Pathfinding", steps, max_steps, f"{steps}/{max_steps}"):
            break
        c, r = cur
        neigh = list(_neighbors_free(occ, c, r))
        if not neigh:
            break
        # prefer rightward, then lower repel, then nearer to goal
        neigh.sort(key=lambda m_n: (
            -int(m_n[1][0] > c),
            (repel_w * float(repel[m_n[1][1], m_n[1][0]]) if repel is not None else 0.0),
            _manhattan(m_n[1], goal)
        ))
        top = neigh[:2] if len(neigh) >= 2 else neigh
        if rnd.random() > right_bias and len(top) == 2:
            top = [top[1], top[0]]
        m, nxt = top[0]
        came[nxt] = (m, cur); cur = nxt
        if cur[0] > last_right:
            last_right = cur[0]; stall = 0
        else:
            stall += 1
        if stall >= STALL:
            break
    tail = greedy_best_first(occ, cur, goal, progress=None, timeout_s=max(0.25, timeout_s - (monotonic()-t0)),
                             repel=repel, repel_w=repel_w)
    for mv, cell in tail:
        came[cell] = (mv, cur); cur = cell
    return _reconstruct(came, start, cur)

# ---------- Right-hand wall follower with escape hatches ----------
_DIRS = ["E", "NE", "SE", "N", "S"]
_RHR_ORDER = {
    "E":  ["SE", "E", "NE", "D", "U"],
    "NE": ["E", "NE", "N", "SE", "U"],
    "SE": ["S", "SE", "E", "D", "NE"],
    "N":  ["NE", "N", "E", "U", "SE"],
    "S":  ["SE", "S", "E", "D", "NE"],
}
_VEC = {"E": (1,0), "N": (0,-1), "S": (0,1), "NE": (1,-1), "SE": (1,1)}
_MOVE_LABEL = {"E":"R", "N":"U", "S":"D", "NE":"UR", "SE":"DR"}

def _can_step_sym(occ, c, r, sym_move):
    dc, dr = _VEC[sym_move]
    nc, nr = c + dc, r + dr
    if not _in_bounds(occ, nc, nr) or occ[nr, nc] != 0:
        return False
    if sym_move == "NE":
        return _free(occ, c+1, r) and _free(occ, c, r-1)
    if sym_move == "SE":
        return _free(occ, c+1, r) and _free(occ, c, r+1)
    return True

def _neighbors_heading_order(occ, c, r, heading):
    for sym in _RHR_ORDER[heading]:
        if _can_step_sym(occ, c, r, sym):
            dc, dr = _VEC[sym]
            yield _MOVE_LABEL[sym], (c + dc, r + dr), sym

def _short_astar_burst(occ, start, goal, max_cols_advance=8, repel=None, repel_w=0.0):
    (sc, sr) = start
    (gc, gr) = goal
    tg = (min(sc + max_cols_advance, occ.shape[1]-1, gc), gr)
    g = {start: 0.0}; came = {}; pq = []; seen = set()
    heappush(pq, (_manhattan(start, tg), start))
    expansions = 0; max_exp = int(occ.shape[0] * max_cols_advance * 6 + 200)
    while pq and expansions < max_exp:
        _, cur = heappop(pq); expansions += 1
        if cur in seen: 
            continue
        seen.add(cur)
        if cur == tg:
            return _reconstruct(came, start, cur)
        for m, nxt in _neighbors_free(occ, *cur):
            ng = g[cur] + 1.0 + (repel_w * float(repel[nxt[1], nxt[0]]) if repel is not None else 0.0)
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng; came[nxt] = (m, cur)
                heappush(pq, (ng + _manhattan(nxt, tg), nxt))
    return []

def wall_hugger_simple(occ, start, goal, prefer_down: bool = False, progress: ProgressCB = None, timeout_s: float = 1.0,
                       repel: Optional[np.ndarray]=None, repel_w: float=0.0):
    if start == goal:
        return []
    heading = "E"
    came = {}
    cur = start
    visited = set()
    steps = 0
    max_steps = max(1200, int(occ.size * 3.0))
    t0 = monotonic()
    last_right_col = start[0]
    stall_guard = 0
    STALL_TRIGGER = max(12, occ.shape[0] // 6)
    BURST_COLS = max(4, min(12, occ.shape[1] // 30))

    while steps < max_steps:
        steps += 1
        if monotonic() - t0 > timeout_s:
            break
        if not _progress(progress, "Pathfinding", steps, max_steps, f"{steps}/{max_steps}"):
            break

        state = (cur, heading)
        if state in visited:
            stall_guard = STALL_TRIGGER  # loop → trigger burst
        else:
            visited.add(state)

        c, r = cur
        moved = False
        for mlabel, nxt, new_heading in _neighbors_heading_order(occ, c, r, heading):
            # repel-aware: prefer smaller repel
            rl = repel_w * float(repel[nxt[1], nxt[0]]) if repel is not None else 0.0
            came[nxt] = (mlabel, cur)
            cur = nxt
            heading = new_heading
            moved = True
            break

        if not moved:
            burst = _short_astar_burst(occ, cur, goal, max_cols_advance=BURST_COLS, repel=repel, repel_w=repel_w)
            if not burst:
                break
            for mv, cell in burst:
                came[cell] = (mv, cur); cur = cell
            heading = "E"

        if cur[0] > last_right_col:
            last_right_col = cur[0]; stall_guard = 0
        else:
            stall_guard += 1

        if cur == goal or (cur[0] >= goal[0] and abs(cur[1] - goal[1]) <= 1):
            break

        if stall_guard >= STALL_TRIGGER:
            burst = _short_astar_burst(occ, cur, goal, max_cols_advance=BURST_COLS, repel=repel, repel_w=repel_w)
            stall_guard = 0
            if burst:
                for mv, cell in burst:
                    came[cell] = (mv, cur); cur = cell
                heading = "E"
            else:
                break

    return _reconstruct(came, start, cur)

# ---------- Registry ----------
PLANNERS_BASE = {
    "A*": astar,
    "Dijkstra": dijkstra,
    "Greedy": greedy_best_first,
    "BFS": bfs,
    "WorstA*": worst_astar,
    "ZigZag": zigzag,
    "RandomGreedy": random_greedy,
    "WallHugger": wall_hugger_simple,
}

def plan_cells(
    occ: np.ndarray,
    start: Tuple[int,int],
    goal: Tuple[int,int],
    algo: str = "A*",
    *,
    allow_soft: bool = False,
    obstacle_cost: float = 320.0,
    progress: ProgressCB = None,
    timeout_s: float = 1.5,
    repel: Optional[np.ndarray] = None,
    repel_w: float = 0.0
) -> List[Tuple[str, Tuple[int,int]]]:
    """
    Unified entry.
    - Hard planners (no corner cutting, no stepping into walls).
    - If no path and allow_soft=True, run soft A* with obstacle_cost.
    - repel/repel_w add "enemy buffer" pressure.
    Always returns a (possibly partial) path.
    """
    algo = algo if algo in PLANNERS_BASE else "A*"
    crumb = PLANNERS_BASE[algo](occ, start, goal, progress=progress, timeout_s=timeout_s, repel=repel, repel_w=repel_w)
    if crumb or not allow_soft:
        return crumb
    return soft_astar(occ, start, goal, obstacle_cost=obstacle_cost, progress=progress, timeout_s=timeout_s, repel=repel, repel_w=repel_w)
