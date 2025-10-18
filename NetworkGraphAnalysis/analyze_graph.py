# NetworkGraphAnalysis/analyze_graph.py
from __future__ import annotations
import json, time, math, sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import networkx as nx
from typing import Dict, Tuple, Iterable

# -----------------------------
# CONFIG (edit if you need)
# -----------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
BASE_ROOT = PROJECT_ROOT.parent
IN_NC_DIR  = BASE_ROOT / "data" / "NetworkConstruction"
IN_NA_DIR  = BASE_ROOT / "data" / "NetworkAnalysis"
OUT_NG_DIR = BASE_ROOT / "data" / "NetworkGraphAnalysis"
OUT_IV_DIR = BASE_ROOT / "data" / "InteractiveVisualization"

OUT_NG_DIR.mkdir(parents=True, exist_ok=True)
OUT_IV_DIR.mkdir(parents=True, exist_ok=True)

# Tag for filenames (you can change to a month like "2001_06")
PERIOD_TAG = "ALL"

# Layout + sampling
RANDOM_SEED = 42
SAMPLE_THRESHOLD = 50_000   # if nodes > threshold, sample (Top-N by PageRank if available else degree)
SAMPLE_TARGET    = 50_000   # keep this many nodes for layout when sampling triggers

# Layout iterations: we run in batches so you see progress
TOTAL_ITER = 300
BATCHES    = 10             # prints "1/10 ... 2/10 ..." etc.
ITERS_PER_BATCH = TOTAL_ITER // BATCHES

# Optional extras
WRITE_TOP_LABELS = True     # writes top-N nodes (by PageRank/degree) for labeling in visuals
TOP_LABELS_N = 200

WRITE_COMPONENT_SIZES = True  # JSON of connected component sizes (for Power BI cards)
WRITE_DEGREE_DISTRIB  = True  # CSV of degree distribution (for histogram)

# -----------------------------
# UTIL
# -----------------------------
def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def log(msg: str):
    print(f"[{ts()}] {msg}", flush=True)

def step(i: int, total: int, msg: str):
    # prints "k/total: msg"
    print(f"[{ts()}] {i}/{total}: {msg}", flush=True)

# -----------------------------
# LOAD GRAPH
# -----------------------------
def load_nodes() -> Iterable[str]:
    nodes_file = IN_NC_DIR / "network_nodes.ndjson"
    with open(nodes_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)["node_id"]
            except Exception:
                # tolerate corrupt rows
                continue

def load_edges() -> Iterable[Tuple[str, str, int]]:
    for ef in sorted(IN_NC_DIR.glob("network_edges_*.csv")):
        df = pd.read_csv(ef, dtype={"source": str, "target": str, "weight": "Int64"})
        # fillna and coerce weight
        df["source"] = df["source"].fillna("").astype(str)
        df["target"] = df["target"].fillna("").astype(str)
        df["weight"] = df["weight"].fillna(0).astype(int)
        for row in df.itertuples(index=False):
            s, t, w = row.source, row.target, int(row.weight)
            if not s or not t or w <= 0:
                continue
            yield s, t, w

def build_graph() -> nx.DiGraph:
    total_steps = 10
    step(1, total_steps, "Loading nodes …")
    G = nx.DiGraph()
    node_count = 0
    for n in load_nodes():
        G.add_node(n)
        node_count += 1
    log(f"Loaded {node_count:,} nodes")

    edge_files = sorted(IN_NC_DIR.glob("network_edges_*.csv"))
    step(2, total_steps, f"Loading {len(edge_files)} edge partitions …")
    row_read = 0
    part_done = 0
    for ef in edge_files:
        df = pd.read_csv(ef, dtype={"source": str, "target": str, "weight": "Int64"})
        df["source"] = df["source"].fillna("").astype(str)
        df["target"] = df["target"].fillna("").astype(str)
        df["weight"] = df["weight"].fillna(0).astype(int)

        # speed: vectorized add_edges_from with weights as attribute dicts
        edges_to_add = [(s, t, {"weight": int(w)}) for s, t, w in df[["source","target","weight"]].itertuples(index=False) if s and t and int(w) > 0]
        G.add_edges_from(edges_to_add)
        row_read += len(df)
        part_done += 1
        log(f"Partition {part_done}/{len(edge_files)}: added {len(edges_to_add):,} edges from {ef.name}")

    # Drop self-loops (layout & some metrics dislike them)
    loops = list(nx.selfloop_edges(G))
    if loops:
        G.remove_edges_from(loops)
        log(f"Removed {len(loops):,} self-loops")

    log(f"Graph built → nodes={G.number_of_nodes():,} edges={G.number_of_edges():,} (read rows={row_read:,})")
    return G

# -----------------------------
# OPTIONAL METRICS (for merge)
# -----------------------------
def load_metrics() -> Dict[str, Dict[str, float]]:
    mfile = IN_NA_DIR / "sna_metrics.csv"
    if not mfile.exists():
        log("No sna_metrics.csv found (optional). Snapshot will include only x,y.")
        return {}
    log(f"Loading metrics from {mfile} …")
    df = pd.read_csv(mfile)
    # Make sure node_id is string
    df["node_id"] = df["node_id"].astype(str)
    metrics = {}
    # keep only known columns if present
    keep_cols = ["indegree","outdegree","betweenness","clustering_coeff","pagerank","eigenvector",
                 "closeness","kcore","community_id","reciprocity"]
    present = [c for c in keep_cols if c in df.columns]
    for row in df[["node_id"] + present].itertuples(index=False):
        d = {}
        node = row[0]
        for i, c in enumerate(present, start=1):
            v = row[i]
            d[c] = float(v) if pd.notna(v) else None
        metrics[node] = d
    log(f"Metrics loaded for {len(metrics):,} nodes.")
    return metrics

# -----------------------------
# SAMPLING
# -----------------------------
def sample_graph(G: nx.DiGraph, metrics: Dict[str, Dict[str, float]]) -> Tuple[nx.DiGraph, bool]:
    n = G.number_of_nodes()
    if n <= SAMPLE_THRESHOLD:
        return G, False

    log(f"Sampling for layout to {SAMPLE_TARGET:,} nodes (of {n:,})")
    # prefer PageRank if available; else out_degree
    if metrics and any("pagerank" in m for m in metrics.values()):
        # rank by pagerank
        rank = sorted(((u, metrics.get(u, {}).get("pagerank", 0.0)) for u in G.nodes()),
                      key=lambda x: x[1], reverse=True)
    else:
        # fallback: out-degree (unique targets)
        rank = sorted(((u, G.out_degree(u)) for u in G.nodes()), key=lambda x: x[1], reverse=True)

    keep = {u for (u, _) in rank[:SAMPLE_TARGET]}
    # add their neighbors to preserve local structure (cap to ~10% extra)
    extra = set()
    cap_extra = max(1, SAMPLE_TARGET // 10)
    for u in list(keep)[:cap_extra]:
        extra.update(v for _, v in G.out_edges(u))
        extra.update(v for v, _ in G.in_edges(u))
        if len(keep) + len(extra) >= SAMPLE_TARGET + cap_extra:
            break
    keep |= extra

    H = G.subgraph(keep).copy()
    log(f"Sampled subgraph → nodes={H.number_of_nodes():,} edges={H.number_of_edges():,}")
    return H, True

# -----------------------------
# LAYOUT with PROGRESS
# -----------------------------
def compute_layout(G: nx.Graph, total_iter: int = TOTAL_ITER, batches: int = BATCHES, seed: int = RANDOM_SEED) -> Dict[str, Tuple[float,float]]:
    # Start from a quick spectral init (fast) then refine with spring in batches
    step(3, 10, "Initializing layout positions …")
    try:
        pos = nx.shell_layout(G)  # very fast initialization
    except Exception:
        pos = None

    # run spring_layout in batches so you see progress
    step(4, 10, f"Computing force-directed layout for {total_iter} iterations in {batches} batches …")
    start = time.time()
    done = 0
    for b in range(1, batches + 1):
        # a smaller 'k' is tighter; auto will scale by 1/sqrt(n)
        pos = nx.spring_layout(
            G,
            pos=pos,
            iterations=ITERS_PER_BATCH,
            seed=seed + b,     # change seed slightly to nudge convergence
            weight="weight"
        )
        done += ITERS_PER_BATCH
        done = min(done, total_iter)
        pct = int(round(100 * done / total_iter))
        log(f"   Batch {b}/{batches} → iterations={done}/{total_iter} ({pct}%)")

    took = time.time() - start
    log(f"Layout finished in {took:.1f}s")
    return pos

# -----------------------------
# WRITE OUTPUTS
# -----------------------------
def write_layout_ndjson(pos: Dict[str, Tuple[float,float]]):
    out_file = OUT_NG_DIR / f"graph_layout_{PERIOD_TAG}.ndjson"
    step(5, 10, f"Writing NDJSON layout → {out_file.name}")
    with open(out_file, "w", encoding="utf-8") as f:
        for u, (x, y) in pos.items():
            f.write(json.dumps({"node_id": u, "x": float(x), "y": float(y)}) + "\n")
    log(f"✅ NDJSON layout → {out_file}")

def write_snapshot_csv(pos: Dict[str, Tuple[float,float]], metrics: Dict[str, Dict[str, float]]):
    out_file = OUT_IV_DIR / f"network_snapshot_{PERIOD_TAG}.csv"
    step(6, 10, f"Writing snapshot CSV → {out_file.name}")
    # build rows
    rows = []
    cols = ["node_id","x","y"]
    # include common metrics if present
    metric_keys = set()
    for m in metrics.values():
        metric_keys.update(k for k in m.keys())
    metric_cols = sorted(metric_keys) if metric_keys else []
    cols.extend(metric_cols)

    for u, (x, y) in pos.items():
        row = {"node_id": u, "x": float(x), "y": float(y)}
        if metrics:
            m = metrics.get(u, {})
            for c in metric_cols:
                row[c] = m.get(c, None)
        rows.append(row)

    pd.DataFrame(rows, columns=cols).to_csv(out_file, index=False)
    log(f"✅ Snapshot CSV → {out_file}")

def write_meta(G_full: nx.Graph, G_used: nx.Graph, sampled: bool, seconds: float):
    out_file = OUT_NG_DIR / f"layout_meta_{PERIOD_TAG}.json"
    step(7, 10, f"Writing layout metadata → {out_file.name}")
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "algorithm": "spring_layout(batch-refined)",
        "iterations_total": TOTAL_ITER,
        "batches": BATCHES,
        "random_seed": RANDOM_SEED,
        "full_graph": {"nodes": G_full.number_of_nodes(), "edges": G_full.number_of_edges()},
        "used_for_layout": {"nodes": G_used.number_of_nodes(), "edges": G_used.number_of_edges()},
        "sampled": sampled,
        "sample_threshold": SAMPLE_THRESHOLD,
        "sample_target": SAMPLE_TARGET,
        "seconds": seconds,
        "period_tag": PERIOD_TAG
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"✅ Layout meta → {out_file}")

# -----------------------------
# OPTIONAL EXTRAS (helpful in Power BI)
# -----------------------------
def write_top_labels(G: nx.Graph, metrics: Dict[str, Dict[str, float]]):
    if not WRITE_TOP_LABELS:
        return
    out_file = OUT_NG_DIR / f"label_nodes_{PERIOD_TAG}.csv"
    step(8, 10, f"Preparing label nodes → {out_file.name}")
    if metrics and any("pagerank" in m for m in metrics.values()):
        scores = [(u, metrics.get(u, {}).get("pagerank", 0.0)) for u in G.nodes()]
    else:
        scores = [(u, G.degree(u)) for u in G.nodes()]  # fallback

    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:TOP_LABELS_N]
    df = pd.DataFrame(top, columns=["node_id", "label_score"])
    df.to_csv(out_file, index=False)
    log(f"✅ Label nodes → {out_file} (top {len(df):,})")

def write_component_sizes(G: nx.Graph):
    if not WRITE_COMPONENT_SIZES:
        return
    out_file = OUT_NG_DIR / f"component_sizes_{PERIOD_TAG}.json"
    step(9, 10, f"Computing connected component sizes → {out_file.name}")
    UG = G.to_undirected()
    sizes = [len(c) for c in nx.connected_components(UG)]
    sizes.sort(reverse=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"component_sizes": sizes}, f, indent=2)
    log(f"✅ Component sizes → {out_file}")

def write_degree_distribution(G: nx.Graph):
    if not WRITE_DEGREE_DISTRIB:
        return
    out_file = OUT_NG_DIR / f"degree_distribution_{PERIOD_TAG}.csv"
    step(10, 10, f"Writing degree distribution → {out_file.name}")
    degs = [d for _, d in G.degree()]
    df = pd.DataFrame({"degree": degs})
    df.to_csv(out_file, index=False)
    log(f"✅ Degree distribution → {out_file}")

# -----------------------------
# MAIN
# -----------------------------
def main():
    t0 = time.time()

    G_full = build_graph()
    metrics = load_metrics()

    # Decide graph for layout (sample if large)
    G_used, sampled = sample_graph(G_full, metrics)

    # Compute layout with batch progress
    pos = compute_layout(G_used, total_iter=TOTAL_ITER, batches=BATCHES, seed=RANDOM_SEED)

    # Write required outputs
    write_layout_ndjson(pos)
    write_snapshot_csv(pos, metrics)
    write_meta(G_full, G_used, sampled, time.time() - t0)

    # Helpful extras (Power BI)
    write_top_labels(G_used, metrics)        # to annotate top influencers in visuals
    write_component_sizes(G_used)            # KPI: largest component size / #components
    write_degree_distribution(G_used)        # histogram source

    log("All done ✅")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user.")
        sys.exit(1)
