# modules/pathfinding_algos.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
from heapq import heappush, heappop
import collections
import random

# Allowed moves: Right, Up, Down, Diagonal Up-Right, Diagonal Down-Right
MOVES: Dict[str, Tuple[int, int]] = {
    "R":  (1,  0),
    "UR": (1, -1),
    "DR": (1,  1),
    "U":  (0, -1),
    "D":  (0,  1),
}

def _in_bounds(occ, c, r):
    return 0 <= r < occ.shape[0] and 0 <= c < occ.shape[1]

def _neighbors(occ, c, r):
    for m, (dc, dr) in MOVES.items():
        nc, nr = c + dc, r + dr
        if _in_bounds(occ, nc, nr) and occ[nr, nc] == 0:
            yield m, (nc, nr)

def _neighbors_soft(occ, c, r, obstacle_cost: float):
    """
    Soft neighbors: allow stepping into obstacles with high extra cost.
    Yields (move, next_cell, step_cost).
    """
    for m, (dc, dr) in MOVES.items():
        nc, nr = c + dc, r + dr
        if not _in_bounds(occ, nc, nr):
            continue
        base = 1.0
        if occ[nr, nc] != 0:
            yield m, (nc, nr), base + obstacle_cost
        else:
            yield m, (nc, nr), base

def _reconstruct(came, start, goal):
    if goal not in came:
        return []
    cur = goal
    out: List[Tuple[str, Tuple[int, int]]] = []
    while cur != start:
        pmove, prev = came[cur]
        out.append((pmove, cur))
        cur = prev
    out.reverse()
    return out

def _manhattan(a, b):
    (c1, r1), (c2, r2) = a, b
    return abs(c1 - c2) + abs(r1 - r2)

# ---------- Classic planners (hard obstacles) ----------
def astar(occ, start, goal):
    g = {start: 0.0}
    came = {}
    pq = []
    heappush(pq, (_manhattan(start, goal), start, ("", start)))
    seen = set()
    while pq:
        _, cur, prev = heappop(pq)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == goal:
            return _reconstruct(came, start, goal)
        for m, nxt in _neighbors(occ, *cur):
            ng = g[cur] + 1.0
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came[nxt] = (m, cur)
                heappush(pq, (ng + _manhattan(nxt, goal), nxt, (m, cur)))
    return []

def dijkstra(occ, start, goal):
    g = {start: 0.0}
    came = {}
    pq = []
    heappush(pq, (0.0, start))
    seen = set()
    while pq:
        cost, cur = heappop(pq)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == goal:
            return _reconstruct(came, start, goal)
        for m, nxt in _neighbors(occ, *cur):
            ng = cost + 1.0
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came[nxt] = (m, cur)
                heappush(pq, (ng, nxt))
    return []

def greedy_best_first(occ, start, goal):
    came = {}
    pq = []
    heappush(pq, (_manhattan(start, goal), start))
    seen = set()
    while pq:
        _, cur = heappop(pq)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == goal:
            return _reconstruct(came, start, goal)
        for m, nxt in _neighbors(occ, *cur):
            if nxt not in seen:
                came[nxt] = (m, cur)
                heappush(pq, (_manhattan(nxt, goal), nxt))
    return []

def bfs(occ, start, goal):
    came = {}
    q = collections.deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        if cur == goal:
            return _reconstruct(came, start, goal)
        for m, nxt in _neighbors(occ, *cur):
            if nxt not in seen:
                seen.add(nxt)
                came[nxt] = (m, cur)
                q.append(nxt)
    return []

# ---------- Soft A* (can pass through obstacles with penalty) ----------
def soft_astar(occ, start, goal, obstacle_cost: float = 100.0):
    g = {start: 0.0}
    came = {}
    pq = []
    heappush(pq, (_manhattan(start, goal), start, ("", start)))
    seen = set()
    while pq:
        _, cur, prev = heappop(pq)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == goal:
            return _reconstruct(came, start, goal)
        for m, nxt, step_cost in _neighbors_soft(occ, *cur, obstacle_cost=obstacle_cost):
            ng = g[cur] + step_cost
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came[nxt] = (m, cur)
                heappush(pq, (ng + _manhattan(nxt, goal), nxt, (m, cur)))
    # return best-so-far even if not reaching goal
    if not g:
        return []
    best = min(g.items(), key=lambda kv: kv[1])[0]
    return _reconstruct(came, start, best)

# ---------- “Funny” / intentionally suboptimal planners ----------
def worst_astar(occ, start, goal, chaos: float = 0.25):
    """
    Purposely bad: prefers *longer* routes and wiggles.
    Priority ~ (-h + alpha*g + small noise). Still terminates at goal if reachable.
    """
    came = {}
    g = {start: 0.0}
    pq = []
    rnd = random.Random(1337)
    def pri(cur):
        return -_manhattan(cur, goal) + 0.75 * g[cur] + chaos * rnd.random()
    heappush(pq, (pri(start), start))
    seen = set()
    while pq:
        _, cur = heappop(pq)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == goal:
            return _reconstruct(came, start, goal)
        # Shuffle neighbors to increase derpiness
        neigh = list(_neighbors(occ, *cur))
        rnd.shuffle(neigh)
        for m, nxt in neigh:
            ng = g[cur] + 1.0
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came[nxt] = (m, cur)
                heappush(pq, (pri(nxt), nxt))
    return []

def zigzag(occ, start, goal, amplitude: int = 1):
    """
    Always tries to go up/down every step to create a sawtooth, then right.
    """
    came = {}
    cur = start
    target_c = goal[0]
    going_up = True
    steps = 0
    max_steps = occ.shape[1] * occ.shape[0] * 4
    while steps < max_steps and cur != goal:
        steps += 1
        c, r = cur
        # try vertical wiggle first
        candidates: List[Tuple[str, Tuple[int,int]]] = []
        if going_up:
            candidates.extend([("UR", (c+1, r-1)), ("U", (c, r-1))])
        else:
            candidates.extend([("DR", (c+1, r+1)), ("D", (c, r+1))])
        candidates.append(("R", (c+1, r)))
        # filter legal
        legal = [(m, nxt) for (m, nxt) in candidates if _in_bounds(occ, *nxt) and occ[nxt[1], nxt[0]] == 0]
        if not legal:
            # try any legal neighbor
            alt = list(_neighbors(occ, c, r))
            if not alt:
                break
            m, nxt = alt[0]
        else:
            m, nxt = legal[0]
        came[nxt] = (m, cur)
        cur = nxt
        # flip direction every 'amplitude' columns advanced
        if m in ("UR","DR","R") and (cur[0] - start[0]) % max(1, amplitude) == 0:
            going_up = not going_up
        if cur[0] >= target_c and abs(cur[1] - goal[1]) <= 1:
            # finish homing in
            break

    # finish with a short greedy to hit exact goal if possible
    tail = greedy_best_first(occ, cur, goal)
    if tail:
        for mv, cell in tail:
            came[cell] = (mv, cur)
            cur = cell
    return _reconstruct(came, start, cur)

def random_greedy(occ, start, goal, right_bias: float = 0.65):
    """
    Random walk with a rightward bias and mild greed toward goal.
    """
    rnd = random.Random()
    came = {}
    cur = start
    max_steps = occ.shape[0] * occ.shape[1] * 4
    steps = 0
    while steps < max_steps and cur != goal:
        steps += 1
        neigh = list(_neighbors(occ, *cur))
        if not neigh:
            break
        # bias: prefer moves that increase column (move right-ish)
        neigh.sort(key=lambda m_n: (-int(m_n[1][0] > cur[0]), _manhattan(m_n[1], goal)))
        if rnd.random() > right_bias:
            rnd.shuffle(neigh)
        m, nxt = neigh[0]
        came[nxt] = (m, cur)
        cur = nxt
    # finish with greedy if close
    tail = greedy_best_first(occ, cur, goal)
    if tail:
        for mv, cell in tail:
            came[cell] = (mv, cur)
            cur = cell
    return _reconstruct(came, start, cur)

def wall_hugger_simple(occ, start, goal, prefer_down: bool = False):
    """
    Super-simple wall follower: try to keep an obstacle 'on one side' while moving right.
    Not a true maze solver; just for fun visuals.
    """
    came = {}
    cur = start
    steps = 0
    max_steps = occ.shape[0] * occ.shape[1] * 6
    # order of preference
    order_up = ["UR", "R", "DR", "U", "D"]
    order_dn = ["DR", "R", "UR", "D", "U"]
    order = order_dn if prefer_down else order_up
    while steps < max_steps and cur != goal:
        steps += 1
        c, r = cur
        moved = False
        for m in order:
            dc, dr = MOVES[m]
            nxt = (c + dc, r + dr)
            if _in_bounds(occ, *nxt) and occ[nxt[1], nxt[0]] == 0:
                came[nxt] = (m, cur)
                cur = nxt
                moved = True
                break
        if not moved:
            # pick any legal
            neigh = list(_neighbors(occ, c, r))
            if not neigh:
                break
            m, nxt = neigh[0]
            came[nxt] = (m, cur)
            cur = nxt
    # try to close in
    tail = greedy_best_first(occ, cur, goal)
    if tail:
        for mv, cell in tail:
            came[cell] = (mv, cur)
            cur = cell
    return _reconstruct(came, start, cur)

# Registry
PLANNERS = {
    "A*": astar,
    "Dijkstra": dijkstra,
    "Greedy": greedy_best_first,
    "BFS": bfs,
    # funny
    "WorstA*": worst_astar,
    "ZigZag": zigzag,
    "RandomGreedy": random_greedy,
    "WallHugger": wall_hugger_simple,
}

def plan_cells(
    occ,
    start: Tuple[int,int],
    goal: Tuple[int,int],
    algo: str = "A*",
    *,
    allow_soft: bool = False,
    obstacle_cost: float = 100.0
) -> List[Tuple[str, Tuple[int,int]]]:
    """
    Primary entry: run selected planner; if no path and allow_soft, run soft A*.
    """
    algo = algo if algo in PLANNERS else "A*"
    crumb = PLANNERS[algo](occ, start, goal)
    if crumb or not allow_soft:
        return crumb
    # Soft fallback: punch through walls at minimum damage
    return soft_astar(occ, start, goal, obstacle_cost=obstacle_cost)
