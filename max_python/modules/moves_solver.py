# modules/moves_solver.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable
import numpy as np
import cv2

ProgressCB = Optional[Callable[[str, int, str], bool]]

ENEMY_RADIUS   = 0.8
X_GROUP_TOL    = 0.6
X_PRE_EPS      = 0.35
X_POST_EPS     = 0.25
STEP_STEEPNESS = 55.0

# keep-away (enemy buffer) defaults for FunnyA*
REPEL_SIGMA_CELLS = 3.0
REPEL_W_FUNNY     = 0.45   # weight added to step cost

@dataclass
class GridSpec:
    dx_units: float = 0.25
    dy_units: float = 0.25
    max_cols: int = 1600
    max_rows: int = 1200

def step_term(k: float, a: float, x_at: float) -> str:
    return f"{k:.4f}/(1+exp(-{a:.0f}*(x+({-x_at:+.4f}))))"

def _diag_params(xs: float, xe: float, m: float) -> Tuple[float,float,float]:
    start, end = (xs, xe) if xs < xe else (xe, xs)
    a = abs(m) / 2.0
    if m >= 0: b, c = -start, -end
    else:      b, c = -end,   -start
    return a, b, c

def _diag_term(x1: float, y1: float, x2: float, y2: float) -> str:
    dx = x2-x1; dy = y2-y1
    if abs(dx)<1e-9 and abs(dy)<1e-9: return "0"
    m  = dy/dx if dx!=0 else 0.0
    a,b,c = _diag_params(x1, x2, m)
    return f"{a:.4f}*(abs(x+({b:+.4f}))-abs(x+({c:+.4f})))"

def moves_to_expression(moves: List[str], start_xy: Tuple[float,float], gs: GridSpec, step_a: float = STEP_STEEPNESS) -> str:
    x,y = start_xy
    parts=[]
    for m in moves:
        if m=="R":
            x += gs.dx_units
        elif m=="U":
            dy = -gs.dy_units; parts.append(step_term(dy, step_a, x)); y += dy
        elif m=="D":
            dy = +gs.dy_units; parts.append(step_term(dy, step_a, x)); y += dy
        elif m=="UR":
            nx,ny = x+gs.dx_units, y-gs.dy_units; parts.append(_diag_term(x,y,nx,ny)); x,y=nx,ny
        elif m=="DR":
            nx,ny = x+gs.dx_units, y+gs.dy_units; parts.append(_diag_term(x,y,nx,ny)); x,y=nx,ny
    expr = " + ".join([p for p in parts if p!="0"]).replace("+-","-").replace("--","+").strip(" +")
    return expr if expr else "0"

def group_enemies_by_x(enemies_xy: List[Tuple[float,float]], start_x: float, x_tol: float=X_GROUP_TOL):
    cand=[p for p in enemies_xy if p[0]>=start_x]
    cand.sort(key=lambda p:p[0])
    cols=[]; cur=[]; cur_x=None
    for x,y in cand:
        if cur_x is None or abs(x-cur_x)<=x_tol:
            cur.append((x,y)); cur_x = x if cur_x is None else (cur_x+x)/2.0
        else:
            cols.append(cur); cur=[(x,y)]; cur_x=x
    if cur: cols.append(cur)
    return cols

def _grid_dims(board, dx, dy):
    cols = int(np.ceil((board.x_range[1]-board.x_range[0]) / dx))
    rows = int(np.ceil((board.y_range[1]-board.y_range[0]) / dy))
    return cols, rows

def _adapt_grid(board, dx, dy, target_cells=100_000):
    cols, rows = _grid_dims(board, dx, dy)
    cells = cols * rows
    if cells <= target_cells:
        return dx, dy, cols, rows
    scale = (cells / target_cells) ** 0.5
    ndx = dx * scale; ndy = dy * scale
    cols, rows = _grid_dims(board, ndx, ndy)
    return ndx, ndy, cols, rows

def make_occ_grid(board, obs_for_planning: np.ndarray, dx_units: float, dy_units: float,
                  max_cols: int, max_rows: int) -> np.ndarray:
    cols, rows = _grid_dims(board, dx_units, dy_units)
    cols = min(cols, max_cols); rows = min(rows, max_rows)
    occ = np.zeros((rows, cols), dtype=np.uint8)
    # r=0 is bottom (y_min). Our visual flips later for Cartesian up.
    for r in range(rows):
        y = board.y_range[0] + r*dy_units
        for c in range(cols):
            x = board.x_range[0] + c*dx_units
            px,py = board.xy_to_px(x,y)
            px = int(np.clip(px,0,board.w-1)); py = int(np.clip(py,0,board.h-1))
            occ[r,c] = 1 if obs_for_planning[py,px] > 0 else 0
    return occ

def xy_to_cell(board, xy: Tuple[float,float], dx_units: float, dy_units: float, occ: np.ndarray) -> Tuple[int,int]:
    x,y = xy
    c = int(np.clip(np.floor((x - board.x_range[0]) / dx_units), 0, occ.shape[1]-1))
    r = int(np.clip(np.floor((y - board.y_range[0]) / dy_units), 0, occ.shape[0]-1))
    return (c,r)

def cell_to_xy(board, cell: Tuple[int,int], dx_units: float, dy_units: float) -> Tuple[float,float]:
    c,r = cell
    x = board.x_range[0] + c*dx_units
    y = board.y_range[0] + r*dy_units
    return (x,y)

# ---------- Enemy repel map (keep-away) ----------
def _gaussian_blur1d(arr: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    if sigma <= 0:
        return arr
    rad = max(1, int(3*sigma))
    x = np.arange(-rad, rad+1)
    w = np.exp(-(x**2) / (2*sigma*sigma))
    w = w / w.sum()
    if axis == 0:
        pad = ((rad, rad), (0, 0))
    else:
        pad = ((0, 0), (rad, rad))
    a = np.pad(arr, pad, mode="edge")
    if axis == 0:
        out = np.zeros_like(arr, dtype=np.float32)
        for i in range(arr.shape[0]):
            out[i,:] = (a[i:i+2*rad+1,:] * w[:,None]).sum(axis=0)
        return out
    else:
        out = np.zeros_like(arr, dtype=np.float32)
        for j in range(arr.shape[1]):
            out[:,j] = (a[:,j:j+2*rad+1] * w[None,:]).sum(axis=1)
        return out

def build_enemy_repel_map(occ: np.ndarray, enemies_cells: List[Tuple[int,int]], sigma_cells: float = REPEL_SIGMA_CELLS) -> np.ndarray:
    if not enemies_cells:
        return np.zeros_like(occ, dtype=np.float32)
    m = np.zeros_like(occ, dtype=np.float32)
    for (c,r) in enemies_cells:
        if 0 <= r < occ.shape[0] and 0 <= c < occ.shape[1]:
            m[r, c] = 1.0
    # cheap separable gaussian blur to turn spikes into a smooth hill
    gx = _gaussian_blur1d(m, sigma_cells, axis=1)
    gy = _gaussian_blur1d(gx, sigma_cells, axis=0)
    # normalize to [0,1]
    if gy.max() > 1e-6:
        gy = gy / float(gy.max())
    return gy.astype(np.float32)

# ---------- Critical grid rendering ----------
def _draw_critical_grid(
    occ_plan: np.ndarray,
    occ_main: np.ndarray,
    path_cells: List[Tuple[int,int]],
    actor_cells: Dict[str, List[Tuple[int,int]]],
    scale: int = 14
) -> np.ndarray:
    H, W = occ_plan.shape
    disp_main = np.flipud(occ_main)
    disp_plan = np.flipud(occ_plan)

    img = np.full((H*scale, W*scale, 3), 255, np.uint8)
    base_mask = disp_main.repeat(scale, 0).repeat(scale, 1) > 0
    tol_bool  = (disp_plan > 0) & (disp_main == 0)
    tol_mask  = tol_bool.repeat(scale, 0).repeat(scale, 1) > 0

    LIGHT_GREEN = (180, 255, 180)
    DARK_GREEN  = (100, 200, 100)
    RED         = (0, 0, 255)
    BLUE        = (255, 0, 0)
    YELLOW      = (0, 215, 255)
    GRID_LINE   = (230, 230, 230)

    img[base_mask] = LIGHT_GREEN
    img[tol_mask]  = DARK_GREEN

    for r in range(0, H*scale, scale):
        cv2.line(img, (0, r), (W*scale-1, r), GRID_LINE, 1)
    for c in range(0, W*scale, scale):
        cv2.line(img, (c, 0), (c, H*scale-1), GRID_LINE, 1)

    def disp_pt(cell):
        c, r = cell
        rr = (H - 1 - r)
        return (c * scale + scale // 2, rr * scale + scale // 2)

    pts = [disp_pt(cr) for cr in path_cells]
    for i in range(1, len(pts)):
        cv2.line(img, pts[i-1], pts[i], RED, 2)

    for cell in actor_cells.get("enemies", []):
        cv2.circle(img, disp_pt(cell), max(2, scale//3), RED, -1)
    for cell in actor_cells.get("allies", []):
        cv2.circle(img, disp_pt(cell), max(2, scale//3), BLUE, -1)
    if actor_cells.get("shooter"):
        cv2.circle(img, disp_pt(actor_cells["shooter"]), max(3, scale//2), YELLOW, 2)

    return img

# ---------- Main moves solver ----------
def solve_moves_on_components(
    *,
    board,
    roi: np.ndarray,
    obs_main: np.ndarray,      # base obstacles+border (tight)
    obs_plan: np.ndarray,      # dilated for safety (planning / tolerance)
    obs_contours,
    actors,
    shooter,
    grid_dx: float,
    grid_dy: float,
    algo: str,
    overlay_path: Optional[str] = None,
    progress: ProgressCB = None,
    targeting_mode: str = "x-first",       # "x-first" | "nearest"
    allow_soft_fallback: bool = True
) -> Dict:
    # Adaptive grid
    dx_units, dy_units, _, _ = _adapt_grid(board, grid_dx, grid_dy, target_cells=100_000)
    cols_cap, rows_cap = GridSpec().max_cols, GridSpec().max_rows

    if progress and not progress("Preparing grid", 5, "adapting resolution"):
        return {"expr": "0"}

    occ_hard = make_occ_grid(board, obs_main, dx_units, dy_units, cols_cap, rows_cap)
    occ_plan = make_occ_grid(board, obs_plan, dx_units, dy_units, cols_cap, rows_cap)

    # Build repel map (enemy keep-away) in grid space
    enemies = [a for a in actors if a.side == "enemy"]
    enemies_cells = [xy_to_cell(board, (a.x, a.y), dx_units, dy_units, occ_hard) for a in enemies]
    repel_map = build_enemy_repel_map(occ_hard, enemies_cells, sigma_cells=REPEL_SIGMA_CELLS)

    # FunnyA*: same as A* but with repel enabled
    algo_name = algo
    repel_w = 0.0
    if algo_name.lower() in ("funny a*", "funny a*".lower(), "funny-a*", "funny"):
        algo_name = "A*"
        repel_w = REPEL_W_FUNNY

    from modules.pathfinding_algos import plan_cells as _plan_cells

    def run_hard(start_xy, goal_xy):
        s = xy_to_cell(board, start_xy, dx_units, dy_units, occ_hard)
        g = xy_to_cell(board, goal_xy,  dx_units, dy_units, occ_hard)
        return _plan_cells(occ_hard, s, g, algo=algo_name, allow_soft=False, progress=progress, timeout_s=1.8,
                           repel=repel_map, repel_w=repel_w)

    def run_soft(start_xy, goal_xy):
        s = xy_to_cell(board, start_xy, dx_units, dy_units, occ_plan)
        g = xy_to_cell(board, goal_xy,  dx_units, dy_units, occ_plan)
        return _plan_cells(occ_plan, s, g, algo="A*", allow_soft=True, obstacle_cost=360.0, progress=progress, timeout_s=1.8,
                           repel=repel_map, repel_w=repel_w)

    enemies_xy = [(a.x, a.y) for a in actors if a.side == "enemy"]

    # Global hard reachability probe (X-priority)
    any_hard_enemy_reachable = False
    if enemies_xy:
        probe_from = (shooter.x, shooter.y)
        for (ex, ey) in sorted(enemies_xy, key=lambda p: p[0]):
            x_pre = ex - X_PRE_EPS
            if run_hard(probe_from, (x_pre, ey)):
                any_hard_enemy_reachable = True
                break

    if progress and not progress("Planning", 10, "choosing targets"):
        return {"expr": "0"}

    # Target order (strict X-first to avoid “can’t go back later”)
    cols = group_enemies_by_x(enemies_xy, start_x=shooter.x, x_tol=X_GROUP_TOL) if enemies_xy else []

    path_cells: List[Tuple[int,int]] = []
    path_moves: List[str] = []
    cur_xy = (shooter.x, shooter.y)
    cur_cell = xy_to_cell(board, cur_xy, dx_units, dy_units, occ_hard)
    path_cells.append(cur_cell)

    def append(crumb):
        nonlocal cur_xy, cur_cell, path_cells, path_moves
        if not crumb:
            return
        path_moves.extend([m for (m,_) in crumb])
        path_cells.extend([cell for (_,cell) in crumb])
        cur_cell = path_cells[-1]
        cur_xy = cell_to_xy(board, cur_cell, dx_units, dy_units)

    if not cols:
        goal_xy = (board.x_range[1] - dx_units, shooter.y)
        c = run_hard(cur_xy, goal_xy)
        if not c and allow_soft_fallback:
            c = run_soft(cur_xy, goal_xy)
        append(c)
    else:
        total_cols = len(cols)
        done_cols = 0
        if any_hard_enemy_reachable:
            for col in cols:
                done_cols += 1
                if progress and not progress("Planning", min(90, 10 + int(75 * done_cols/total_cols)), f"column {done_cols}/{total_cols}"):
                    return {"expr":"0"}
                xcol = float(np.mean([x for (x,_) in col]))
                x_pre, x_post = xcol - X_PRE_EPS, xcol + X_POST_EPS
                pending = col[:]
                while pending:
                    pending.sort(key=lambda p: abs(p[1]-cur_xy[1]))
                    ex, ey = pending.pop(0)
                    c1 = run_hard(cur_xy, (x_pre, ey))
                    if not c1:
                        break
                    append(c1)
                    c2 = run_hard(cur_xy, (x_post, cur_xy[1])); append(c2)
        else:
            for col in cols:
                done_cols += 1
                if progress and not progress("Planning", min(90, 10 + int(75 * done_cols/total_cols)), f"soft approach {done_cols}/{total_cols}"):
                    return {"expr":"0"}
                xcol = float(np.mean([x for (x,_) in col]))
                x_pre, x_post = xcol - X_PRE_EPS, xcol + X_POST_EPS
                avg_y = float(np.mean([y for (_, y) in col]))
                c1 = run_soft(cur_xy, (x_pre, avg_y)); append(c1)
                c2 = run_soft(cur_xy, (x_post, cur_xy[1])); append(c2)

    expr = moves_to_expression(path_moves, (shooter.x, shooter.y), GridSpec(dx_units, dy_units))

    # Overlay (real ROI) with actors + path
    overlay = roi.copy()
    if obs_contours:
        cv2.drawContours(overlay, obs_contours, -1, (0,255,0), 2)
    h,w = obs_main.shape[:2]
    cv2.rectangle(overlay, (0,0), (w-1,h-1), (0,255,0), 2)
    for a in actors:
        if a is shooter:
            cv2.circle(overlay, (a.px,a.py), 18, (0,215,255), 3)
            cv2.putText(overlay, "S", (a.px+10,a.py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,215,255), 2, cv2.LINE_AA)
        else:
            color = (0,0,255) if a.side=="enemy" else (255,0,0)
            label = "E" if a.side=="enemy" else "A"
            cv2.circle(overlay, (a.px,a.py), 6, color, 2)
            cv2.putText(overlay, label, (a.px+8, a.py-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if a.side == "enemy":
            rpx = int((ENEMY_RADIUS / (board.x_range[1]-board.x_range[0])) * board.w)
            cv2.circle(overlay, (a.px,a.py), max(6,rpx), (0,140,255), 1)

    if len(path_cells) >= 2:
        pts_px = [board.xy_to_px(*cell_to_xy(board, c, dx_units, dy_units)) for c in path_cells]
        for i in range(1, len(pts_px)):
            cv2.line(overlay, pts_px[i-1], pts_px[i], (0,0,255), 2)

    # Critical grid (flipped up) with tolerance ring
    allies_cells, enemies_cells2, shooter_cell = [], [], None
    for a in actors:
        cc = xy_to_cell(board, (a.x, a.y), dx_units, dy_units, occ_hard)
        if a is shooter: shooter_cell = cc
        elif a.side == "enemy": enemies_cells2.append(cc)
        else: allies_cells.append(cc)

    crit = _draw_critical_grid(
        occ_plan=occ_plan, occ_main=occ_hard, path_cells=path_cells,
        actor_cells={"allies": allies_cells, "enemies": enemies_cells2, "shooter": shooter_cell},
        scale=14
    )

    return {"expr": (expr if expr.strip() else "0"), "overlay_bgr": overlay, "crit_overlay_bgr": crit}
