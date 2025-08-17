# modules/solver_engine.py
#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Callable
import math
import numpy as np
import cv2

# ---------- Public knobs (bumped padding/tolerance as requested) ----------
DEFAULT_BORDER_PX      = 16
DEFAULT_INFLATE_MAIN   = 10   # hard collision padding
DEFAULT_INFLATE_TOL    = 20   # extra tolerance for planning/visual (dark green)
DEFAULT_MIN_THICK_PX   = 9.0
DEFAULT_MIN_OBS_AREA   = 220
DEFAULT_SHRINK_STEPS   = 6

ENEMY_RADIUS = 0.8

ProgressCB = Optional[Callable[[str, int, str], bool]]

# ---------- Data ----------
@dataclass
class Board:
    x0: int; y0: int; w: int; h: int
    x_range: Tuple[float, float]; y_range: Tuple[float, float]
    def px_to_xy(self, px: float, py: float) -> Tuple[float, float]:
        x_min, x_max = self.x_range; y_min, y_max = self.y_range
        x = x_min + (px / self.w) * (x_max - x_min)
        y = y_max - (py / self.h) * (y_max - y_min)
        return x, y
    def xy_to_px(self, x: float, y: float) -> Tuple[int, int]:
        x_min, x_max = self.x_range; y_min, y_max = self.y_range
        px = int(round((x - x_min) / (x_max - x_min) * self.w))
        py = int(round((y_max - y) / (y_max - y_min) * self.h))
        return px, py

@dataclass
class Actor:
    x: float; y: float; side: str; px: int; py: int; area: int = 0

# ---------- ROI / board ----------
def find_board_roi(img_bgr: np.ndarray) -> Tuple['Board', np.ndarray]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([179, 45, 255]))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Couldn't find a white board region.")
    cnt = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    roi = img_bgr[y:y+h, x:x+w].copy()
    board = Board(x0=x, y0=y, w=w, h=h, x_range=(-25.0, 25.0), y_range=(-15.0, 15.0))
    return board, roi

# ---------- Label-aware small/noisy obstacle filtering ----------
def detect_white_labels(roi_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([179, 120, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        ar = w/max(1,h)
        if 100<=area<=20000 and ar>=1.03:
            boxes.append((x,y,w,h))
    return boxes

def remove_thin_dark_near_labels(obstacles_mask: np.ndarray, roi_bgr: np.ndarray,
                                 pad:int=24, dilate:int=50) -> np.ndarray:
    boxes = detect_white_labels(roi_bgr)
    if not boxes:
        return obstacles_mask
    er = cv2.erode(obstacles_mask, np.ones((3,3), np.uint8), iterations=1)
    thin = cv2.bitwise_and(obstacles_mask, cv2.bitwise_not(er))
    region = np.zeros_like(obstacles_mask)
    for (x,y,w,h) in boxes:
        x0=max(0,x-pad); y0=max(0,y-pad)
        x1=min(obstacles_mask.shape[1]-1, x+w+pad); y1=min(obstacles_mask.shape[0]-1, y+h+pad)
        cv2.rectangle(region,(x0,y0),(x1,y1),255,thickness=cv2.FILLED)
    if dilate>0:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilate+1,2*dilate+1))
        region=cv2.dilate(region,k,iterations=1)
    cleaned = obstacles_mask.copy()
    cleaned[(region>0) & (thin>0)] = 0
    return cleaned

def _keep_by_thickness(shape_mask: np.ndarray, min_thick_px: float, min_area: int) -> np.ndarray:
    num, labels = cv2.connectedComponents(shape_mask)
    keep = np.zeros_like(shape_mask)
    for i in range(1, num):
        comp = np.uint8(labels == i) * 255
        area = int((comp > 0).sum())
        if area < min_area:
            continue
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        per = cv2.arcLength(cnt, True)
        circularity = 4 * math.pi * cv2.contourArea(cnt) / (per * per + 1e-9)
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        thick = float(dist.max())
        if thick >= min_thick_px or (circularity >= 0.60 and area >= max(150, min_area)):
            keep = cv2.bitwise_or(keep, comp)
    return keep

def obstacle_mask_and_contours(roi_bgr: np.ndarray,
                               min_thick_px: float,
                               min_area: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    mask = np.uint8((v < 60) * 255)          # true black fill only
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8), iterations=1)
    mask = remove_thin_dark_near_labels(mask, roi_bgr, pad=24, dilate=50)
    mask = _keep_by_thickness(mask, min_thick_px=min_thick_px, min_area=min_area)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, contours, -1, 255, thickness=cv2.FILLED)
    return clean, contours

def add_border_mask(shape: Tuple[int,int], px:int)->np.ndarray:
    h,w = shape
    border = np.zeros((h,w), dtype=np.uint8)
    cv2.rectangle(border,(0,0),(w-1,h-1),255,thickness=px)
    return border

def inflate(mask: np.ndarray, px:int)->np.ndarray:
    if px<=0: return mask.copy()
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(px,px))
    return cv2.dilate(mask,k,iterations=1)

# ---------- Actors ----------
def detect_actors(roi_bgr: np.ndarray, min_area:int=60)->List[Actor]:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,50,80]), np.array([179,255,255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    num,_,stats,cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out=[]
    for i in range(1,num):
        area=int(stats[i,cv2.CC_STAT_AREA])
        if area<min_area or area>5000: continue
        cx,cy=cent[i]
        out.append(Actor(x=0.0,y=0.0,side="unknown",px=int(cx),py=int(cy),area=area))
    return out

def assign_coords(board: Board, actors: List[Actor])->None:
    for a in actors:
        a.x,a.y = board.px_to_xy(a.px,a.py)

def classify_by_center(actors: List[Actor])->None:
    for a in actors:
        a.side = "ally" if a.x<0 else "enemy"

def red_mask(roi_bgr: np.ndarray)->np.ndarray:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0,60,70]),   np.array([12,255,255]))
    m2 = cv2.inRange(hsv, np.array([168,60,70]), np.array([179,255,255]))
    m  = cv2.bitwise_or(m1,m2)
    m  = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8), iterations=1)
    m  = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=1)
    return m

def choose_shooter_by_red_ring(roi_bgr: np.ndarray, actors: List[Actor])->Optional[Actor]:
    if not actors: return None
    rm = red_mask(roi_bgr)
    if int((rm>0).sum())<20: return None
    best=None; best_score=0.0; R_IN,R_OUT=6,30; NBINS=36
    for a in actors:
        h,w = rm.shape; y,x=a.py,a.px
        y0,y1=max(0,y-R_OUT),min(h,y+R_OUT+1); x0,x1=max(0,x-R_OUT),min(w,x+R_OUT+1)
        sub = rm[y0:y1,x0:x1]
        if sub.size==0: continue
        yy,xx = np.ogrid[y0:y1, x0:x1]
        d2 = (yy-y)**2 + (xx-x)**2
        reds = (sub>0) & (d2<=R_OUT**2) & (d2>=R_IN**2)
        cnt = int(reds.sum())
        if cnt==0: continue
        ys,xs = np.nonzero(reds)
        ang = np.arctan2((ys+y0)-y, (xs+x0)-x)
        bins = ((ang+np.pi)/(2*np.pi)*NBINS).astype(int)
        bins = np.clip(bins,0,NBINS-1)
        cov = np.unique(bins).size/NBINS
        score = cnt*(cov**1.5)
        if cov>=0.25 and score>best_score:
            best_score=score; best=a
    return best

def choose_leftmost(actors: List[Actor])->Optional[Actor]:
    return min(actors, key=lambda t:t.x) if actors else None

# ---------- Entry point used by UI ----------
def solve_from_bgr(
    bgr: np.ndarray,
    *,
    x_range: Tuple[float,float]=(-25.0,25.0),
    y_range: Tuple[float,float]=(-15.0,15.0),
    min_area: int=60,
    overlay_path: Optional[str]=None,
    # moves planner knobs from UI
    grid_dx: float = 0.25,
    grid_dy: float = 0.25,
    algo: str = "A*",
    targeting_mode: str = "x-first",
    allow_soft_fallback: bool = True,
    # padding / tolerance
    inflate_main_px: int = DEFAULT_INFLATE_MAIN,
    inflate_tol_px: int  = DEFAULT_INFLATE_TOL,
    border_px: int       = DEFAULT_BORDER_PX,
    min_obstacle_area: int = DEFAULT_MIN_OBS_AREA,
    min_thickness_px: float = DEFAULT_MIN_THICK_PX,
    shrink_steps: int = DEFAULT_SHRINK_STEPS,
    # progress
    progress_cb: ProgressCB = None
) -> dict:
    board, roi = find_board_roi(bgr)
    board.x_range = x_range; board.y_range = y_range

    if progress_cb and not progress_cb("Detecting obstacles", 2, "thresholds"):
        return {"expr":"0"}
    obs_raw, obs_contours = obstacle_mask_and_contours(
        roi,
        min_thick_px=min_thickness_px,
        min_area=min_obstacle_area
    )

    border_only = add_border_mask(obs_raw.shape, px=border_px)

    # Inflate only obstacles, then OR with border
    obs_hard = cv2.bitwise_or(inflate(obs_raw, inflate_main_px), border_only)
    obs_plan = cv2.bitwise_or(inflate(obs_raw, inflate_tol_px),  border_only)

    # Optional shrink (not used directly, kept for future tuning)
    if shrink_steps > 0:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        cur=obs_raw.copy()
        for _ in range(shrink_steps):
            cur=cv2.erode(cur,k,iterations=1)

    if progress_cb and not progress_cb("Detecting actors", 8, "players & shooter"):
        return {"expr":"0"}

    actors = detect_actors(roi, min_area=min_area)
    assign_coords(board, actors); classify_by_center(actors)
    shooter = choose_shooter_by_red_ring(roi, actors)
    if shooter is None:
        left_allies=[a for a in actors if a.side=="ally"]
        shooter = choose_leftmost(left_allies) or choose_leftmost(actors)
    if shooter is None:
        raise RuntimeError("No actors detected.")

    # ----- Moves-based planner -----
    from modules.moves_solver import solve_moves_on_components as _solve_moves
    res = _solve_moves(
        board=board,
        roi=roi,
        obs_main=obs_hard,           # hard (light green)
        obs_plan=obs_plan,           # tolerant (dark green)
        obs_contours=obs_contours,
        actors=actors,
        shooter=shooter,
        grid_dx=grid_dx,
        grid_dy=grid_dy,
        algo=algo,
        targeting_mode=targeting_mode,
        allow_soft_fallback=allow_soft_fallback,
        overlay_path=overlay_path,
        progress=progress_cb
    )

    if overlay_path and res.get("overlay_bgr") is not None:
        cv2.imwrite(overlay_path, res["overlay_bgr"])

    return res
