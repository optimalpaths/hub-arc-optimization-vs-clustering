"""
Run a small experiment comparing:
- MILP (Gurobi)  [default solve]
- K-medoids heuristic (OD similarity)
- LCS heuristic (route-sequence similarity)

"""

# ============================================================
# Imports
# ============================================================
import os
import time
import copy
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.geo import build_grid_spec, route_sequence_proxy, haversine_m
from src.kmedoids import kmedoids_pam
from src.lcs import cluster_by_lcs
from src.eval import Trip, HubArc, compute_eligibility, greedy_assign
from src.milp import solve_hub_arc_milp


# ============================================================
# SETTINGS (edit if you want)
# ============================================================
DATA_FILE = "april_29.csv"   # same folder as notebook
N_HEAD    = 10000            # first N rows (deterministic)
CAP_OPEN  = 30               # Q
SEED      = 0
WALK_M    = 500.0            # max walk to hubs (meters)
LCS_THR   = 0.15

# MILP controls (these match your previous usage)
TIME_LIMIT_S = 20.0
MIP_GAP      = 0.01
VERBOSE_MILP = False

# If n is large, OD distance is O(n^2). Consider reducing N_HEAD.
OD_DISTANCE_WARN_N = 12000


# ============================================================
# Helpers
# ============================================================
np.random.seed(SEED)

class Timer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = None
        self.dt = None

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.dt = time.time() - self.t0


def safe_mean(x: List[float]) -> float:
    return float(np.mean(x)) if len(x) else 0.0

def safe_quantile(x: List[float], q: float) -> float:
    return float(np.quantile(x, q)) if len(x) else 0.0


def load_trips_head(csv_path: str, n_head: int) -> List[Trip]:
    df = pd.read_csv(csv_path).head(n_head)
    needed = ["pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude","passenger_count"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    trips = [
        Trip(
            o_lat=float(r.pickup_latitude),
            o_lon=float(r.pickup_longitude),
            d_lat=float(r.dropoff_latitude),
            d_lon=float(r.dropoff_longitude),
            passengers=float(r.passenger_count),
        )
        for r in df.itertuples(index=False)
    ]
    return trips


def trips_to_hubarcs(trips: List[Trip], indices: np.ndarray) -> List[HubArc]:
    return [HubArc(trips[i].o_lat, trips[i].o_lon, trips[i].d_lat, trips[i].d_lon) for i in indices.tolist()]


def pairwise_od_distance(trips: List[Trip]) -> np.ndarray:
    """
    Distance between trips in OD space:
      d(i, j) = haversine(origin_i, origin_j) + haversine(dest_i, dest_j)
    """
    n = len(trips)
    if n > OD_DISTANCE_WARN_N:
        print(f"[warn] n={n} => pairwise OD distance is O(n^2). Consider lowering N_HEAD for speed.")

    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        ti = trips[i]
        for j in range(i + 1, n):
            tj = trips[j]
            d = (
                haversine_m(ti.o_lat, ti.o_lon, tj.o_lat, tj.o_lon) +
                haversine_m(ti.d_lat, ti.d_lon, tj.d_lat, tj.d_lon)
            )
            D[i, j] = D[j, i] = d
    return D


def evaluate_solution(
    trips: List[Trip],
    eligible: List[List[int]],
    open_mask: List[bool],
    method_name: str,
    elapsed_s: Optional[float],
) -> Dict:
    """
    Uses greedy_assign to evaluate a chosen set of open hub-arcs.
    Returns consistent summary metrics across methods.
    """
    obj, served_trips = greedy_assign(trips, eligible, open_mask)

    n = len(trips)
    total_pax = sum(t.passengers for t in trips) if n else 0.0
    trip_cov = (served_trips / n) if n else np.nan
    pax_cov  = (obj / total_pax) if total_pax > 0 else np.nan

    return {
        "method": method_name,
        "total_passengers_served": float(obj),
        "trips_served": int(served_trips),
        "trip_coverage": float(trip_cov),
        "passenger_coverage": float(pax_cov),
        "elapsed_s": float(elapsed_s) if elapsed_s is not None else np.nan,
        "num_candidates": int(len(open_mask)),
        "num_open": int(sum(open_mask)),
    }


# ============================================================
# RUN
# ============================================================
print("Notebook folder:", os.getcwd())
print("Dataset file:", os.path.abspath(DATA_FILE))

with Timer("load_trips") as t_load:
    trips = load_trips_head(DATA_FILE, N_HEAD)

n = len(trips)
Q = min(CAP_OPEN, n)
total_pax = sum(t.passengers for t in trips)

print("\n=== Experiment Setup ===")
print(f"DATA_FILE={DATA_FILE}")
print(f"N_HEAD={N_HEAD} => n={n}")
print(f"CAP_OPEN={CAP_OPEN} => Q={Q}")
print(f"WALK_M={WALK_M} | LCS_THR={LCS_THR} | SEED={SEED}")
print(f"MILP: time_limit={TIME_LIMIT_S}s | mip_gap={MIP_GAP} | verbose={VERBOSE_MILP}")
print(f"Load time: {t_load.dt:.2f}s")


# ------------------------------------------------------------
# Build route-sequence proxy for LCS method
# ------------------------------------------------------------
with Timer("build_sequences") as t_seq:
    lats = [t.o_lat for t in trips] + [t.d_lat for t in trips]
    lons = [t.o_lon for t in trips] + [t.d_lon for t in trips]
    grid = build_grid_spec(lats, lons, dlat=0.002, dlon=0.002)
    seqs = [route_sequence_proxy(grid, t.o_lat, t.o_lon, t.d_lat, t.d_lon) for t in trips]

print(f"Sequence proxy time: {t_seq.dt:.2f}s")


# ------------------------------------------------------------
# Heuristic 1: K-medoids (OD distance)
# ------------------------------------------------------------
with Timer("kmedoids_total") as t_kmed:
    with Timer("od_distance") as t_od:
        D_od = pairwise_od_distance(trips)

    with Timer("kmedoids_pam") as t_pam:
        medoids, labels_k = kmedoids_pam(D_od, k=Q, seed=SEED)

    hubs_kmed = trips_to_hubarcs(trips, medoids)

    with Timer("eligibility_kmed") as t_elig_kmed:
        eligible_kmed = compute_eligibility(trips, hubs_kmed, max_walk_m=WALK_M)

    open_mask_kmed = [True] * len(hubs_kmed)

res_kmed = evaluate_solution(
    trips=trips,
    eligible=eligible_kmed,
    open_mask=open_mask_kmed,
    method_name="K-medoids (OD-distance heuristic)",
    elapsed_s=t_kmed.dt
)

print("\n[K-medoids debug]")
print(f"OD distance: {t_od.dt:.2f}s | PAM: {t_pam.dt:.2f}s | eligibility: {t_elig_kmed.dt:.2f}s | total: {t_kmed.dt:.2f}s")
print(f"Chosen medoids: {len(hubs_kmed)}")


# ------------------------------------------------------------
# Heuristic 2: LCS clustering
# ------------------------------------------------------------
with Timer("lcs_total") as t_lcs:
    with Timer("lcs_cluster") as t_cluster:
        labels_lcs = cluster_by_lcs(seqs, threshold=LCS_THR)

    # Representative selection per cluster:
    # pick the trip with highest passenger_count (simple + strong baseline)
    reps = {}
    for i, lab in enumerate(labels_lcs):
        if lab not in reps or trips[i].passengers > trips[reps[lab]].passengers:
            reps[lab] = i

    rep_indices = np.array(list(reps.values()), dtype=int)

    # If more reps than Q, keep top-Q by passengers
    if len(rep_indices) > Q:
        rep_indices = rep_indices[np.argsort([-trips[i].passengers for i in rep_indices])[:Q]]

    # If fewer reps than Q, fill with top remaining passengers
    if len(rep_indices) < Q:
        chosen = set(rep_indices.tolist())
        fill = [i for i in np.argsort([-t.passengers for t in trips]) if i not in chosen]
        rep_indices = np.array(rep_indices.tolist() + fill[:(Q - len(rep_indices))], dtype=int)

    hubs_lcs = trips_to_hubarcs(trips, rep_indices)

    with Timer("eligibility_lcs") as t_elig_lcs:
        eligible_lcs = compute_eligibility(trips, hubs_lcs, max_walk_m=WALK_M)

    open_mask_lcs = [True] * len(hubs_lcs)

res_lcs = evaluate_solution(
    trips=trips,
    eligible=eligible_lcs,
    open_mask=open_mask_lcs,
    method_name="LCS (route-sequence clustering heuristic)",
    elapsed_s=t_lcs.dt
)

print("\n[LCS debug]")
print(f"LCS cluster: {t_cluster.dt:.2f}s | eligibility: {t_elig_lcs.dt:.2f}s | total: {t_lcs.dt:.2f}s")
print(f"Clusters={len(set(labels_lcs))} | reps used={len(hubs_lcs)}")


# ------------------------------------------------------------
# MILP (Gurobi) 
# ------------------------------------------------------------
with Timer("milp_total") as t_milp:
    hubs_all = [HubArc(t.o_lat, t.o_lon, t.d_lat, t.d_lon) for t in trips]
    eligible_all = compute_eligibility(trips, hubs_all, max_walk_m=WALK_M)

    # IMPORTANT: call solve_hub_arc_milp only with supported args.
    # This matches your earlier working pattern.
    milp_res = solve_hub_arc_milp(
        trips,
        eligible_all,
        cap_open=Q,
        time_limit_s=TIME_LIMIT_S,
        mip_gap=MIP_GAP,
        verbose=VERBOSE_MILP
    )

# Build MILP summary row
if milp_res is None:
    res_milp = {
        "method": "MILP (Gurobi)",
        "total_passengers_served": np.nan,
        "trips_served": np.nan,
        "trip_coverage": np.nan,
        "passenger_coverage": np.nan,
        "elapsed_s": float(t_milp.dt) if t_milp.dt is not None else np.nan,
        "num_candidates": len(hubs_all),
        "num_open": Q,
        "status": "skipped / solver unavailable",
    }
else:
    # Your solver returns a dict in your earlier notebook versions.
    # We handle missing keys gracefully.
    served = milp_res.get("served_trips", milp_res.get("trips_served", 0))
    obj    = milp_res.get("objective", milp_res.get("total_passengers_served", np.nan))
    status = milp_res.get("status", "ok")

    res_milp = {
        "method": "MILP (Gurobi)",
        "total_passengers_served": float(obj) if obj is not None else np.nan,
        "trips_served": int(served) if served is not None else np.nan,
        "trip_coverage": float(served / n) if (n and served is not None) else np.nan,
        "passenger_coverage": float(obj / total_pax) if (total_pax and obj is not None) else np.nan,
        "elapsed_s": float(t_milp.dt) if t_milp.dt is not None else np.nan,
        "num_candidates": int(len(hubs_all)),
        "num_open": int(Q),
        "status": str(status),
    }

print("\n[MILP debug]")
print(f"Candidates={len(hubs_all)} | total MILP time: {t_milp.dt:.2f}s")
print(f"Status: {res_milp.get('status')}")


# ============================================================
# Summary Table
# ============================================================
summary = pd.DataFrame([res_kmed, res_lcs, res_milp])

cols = [
    "method",
    "total_passengers_served",
    "passenger_coverage",
    "trips_served",
    "trip_coverage",
    "num_open",
    "num_candidates",
    "elapsed_s",
]
if "status" in summary.columns:
    cols.append("status")

print("\n=== Hub-Arc Ridership Comparison (Summary) ===")
print(f"Dataset: {DATA_FILE} | first_rows={N_HEAD} | n={n} | Q={Q} | walk_m={WALK_M} | lcs_thr={LCS_THR}")
print(summary[cols].to_string(index=False))
