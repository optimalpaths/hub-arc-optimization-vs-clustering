🔗 **Related full-scale study (IEEE):** https://ieeexplore.ieee.org/abstract/document/11029611  
**Scale highlight:** 250,000+ trips / 400,000+ movements, **83.7% trip coverage**, **14 opened connections**, **< 8 min average solve time** via **Benders decomposition**.

---

## Hub-Arc Ride-Sharing Mini-Project  
## MILP (Gurobi) vs. Two Data-Driven Heuristics (K-medoids + LCS)

This repo is a compact, end-to-end mini-project that answers a practical question:

If you can "open" only Q shared hub-to-hub connections, which method chooses the best connections and serves the most riders—an optimization model or data-driven clustering heuristics?

It includes:
- a Gurobi MILP (default solve; full-scale version uses **Benders decomposition** for large instances),
- two "ML-style" baselines:
  - K-medoids on OD similarity (end-point similarity),
  - LCS clustering on route-sequence similarity (overlap),
- a consistent evaluation pipeline (coverage, passengers served, utilization, walking penalty, distance saved).

---

## What this project does

Given a day of trips (pickup/dropoff coordinates + passenger counts), the code:

1. Builds "candidate hub-arcs" (shared connections).
2. Computes which trips are "eligible" for each hub-arc under a walking constraint.
3. Chooses up to Q hub-arcs using "three methods".
4. Assigns trips to chosen hub-arcs and reports metrics + insights.

---

## Core idea (plain English)

A trip can be "served by sharing" if the rider can:
- walk from their "pickup" to the "origin hub", and
- walk from the "destination hub" to their "dropoff",

within a maximum walking threshold ('WALK_M'), while the shared vehicle drives only the hub→hub segment.

So the model is trading off:
- coverage (how many trips/passengers you serve),
- walking inconvenience (access/egress),
- network efficiency (distance saved + utilization of each open hub-arc).

---

## Methods compared

### 1) MILP (Gurobi)
Goal: pick up to Q hub-arcs to maximize total passengers served, subject to eligibility and assignment constraints.

- y(k,l) ∈ {0,1} opens hub-arc k→l
- x(i,k,l) ∈ {0,1} assigns trip i to hub-arc k→l

When it shines
- Best coverage / best objective (usually)
- Principled tradeoffs (directly optimizes the real objective)
- Better utilization patterns (fewer “dead” arcs)

---

### 2) K-medoids heuristic (OD similarity / end-point similarity)
Idea: trips that start and end near each other are likely to benefit from the same shared connection.

Steps:
1. Define a trip-to-trip distance:
   - distance between origins + distance between destinations (haversine)
2. Run K-medoids with k=Q
3. Use the medoid trips as representative hub-arcs
4. Evaluate using the same eligibility + assignment logic

Why this is a strong baseline
- Fast, simple, interpretable
- Produces geographically reasonable connections
- Great for quick prototypes and sanity checks

---

### 3) LCS heuristic (route similarity / overlap)
Idea: even if OD endpoints differ, trips may share a similar "corridor". LCS captures overlap in route sequences.

Steps:
1. Create a lightweight "route sequence proxy" for each trip (grid-based)
2. Compute similarity via "Longest Common Subsequence"
3. Cluster trips by similarity threshold
4. Choose one representative per cluster (demand-aware scoring)
5. Evaluate like the others

Why it’s interesting
- Captures corridor structure (not just endpoints)
- Often finds “backbone” arcs that serve diverse trips
- More robust to noisy OD scatter in dense areas

---

## What you’ll learn from the results (typical insights)

This project is designed to generate insights—not just a scoreboard:

- Optimization vs. heuristics
  - MILP usually wins on served passengers/trips because it directly optimizes the objective under constraints.
  - Heuristics can be close when demand is strongly clustered (clear hotspots/corridors).

- Utilization matters
  - Two methods can serve similar passenger totals but distribute them differently:
    - MILP often concentrates on fewer, high-value arcs.
    - clustering may open arcs that look "representative" but get low utilization.

- Walking constraint is the real lever
  - Tight 'WALK_M' can make many trips ineligible → all methods struggle.
  - Moderate 'WALK_M' often unlocks large gains from a small number of arcs.

- Route similarity can beat OD similarity
  - LCS can outperform K-medoids in settings where trips share corridors but endpoints vary widely.

---

## Repo structure

```text
.
├─ notebooks/
│  └─ comparison_analysis_clean_gurobi.ipynb
├─ src/
│  ├─ geo.py          # grid proxy + haversine helpers
│  ├─ kmedoids.py     # PAM k-medoids
│  ├─ lcs.py          # LCS clustering
│  ├─ eval.py         # eligibility + greedy assignment + metrics
│  └─ milp.py         # Gurobi MILP solver
├─ data/
│  └─ april_29.csv    # local file (not required to commit)
├─ requirements.txt
└─ README.md
```

---

## Setup

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Gurobi
To run the MILP you need:
- gurobipy installed
- a working Gurobi license

If MILP is unavailable, you can still run the two heuristics.

---

## Run the experiment

Open the notebook:

`notebooks/comparison_analysis_clean_gurobi.ipynb`

Edit the top settings:

- DATA_FILE (CSV path)
- N_HEAD (number of rows; deterministic first-N)
- CAP_OPEN (Q)
- WALK_M
- LCS_THR
- TIME_LIMIT_S, MIP_GAP

Then run all cells.

---

## Output and how to interpret it

The notebook prints a comparison table with:

- trips_served / trip_coverage
- total_passengers_served / passenger_coverage
- elapsed time
- optional diagnostics (if enabled in your notebook):
  - utilization distribution across opened arcs
  - distance saved vs baseline
  - walking penalty summaries (avg / p90)

---

## Data format expected

CSV with at least:

- pickup_latitude, pickup_longitude
- dropoff_latitude, dropoff_longitude
- passenger_count

You can use any city trip dataset with these fields.

---

## Limitations and next upgrades

If you want to extend this later:

- Replace the grid proxy with true road-network shortest paths (OSMnx / OSRM).
- Add a parameter sweep (Q, WALK_M, LCS_THR) and export results to results/summary.csv.
- Add more baselines (e.g., k-means on engineered features, spectral clustering on the LCS graph).
- Export the chosen arcs + utilization into a tidy artifact for plotting.

---
