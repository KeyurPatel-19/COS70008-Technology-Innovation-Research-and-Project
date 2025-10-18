# NetworkAnalysis/analyze_network.py
from pathlib import Path
import pandas as pd, json
import networkx as nx

# Optional: community detection (pip install python-louvain)
try:
    import community as community_louvain
except ImportError:
    community_louvain = None

PROJECT_ROOT = Path(__file__).parent.resolve()
BASE_ROOT = PROJECT_ROOT.parent
IN_DIR  = BASE_ROOT / "data" / "NetworkConstruction"
OUT_DIR = BASE_ROOT / "data" / "NetworkAnalysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_nodes(nodes_path: Path):
    nodes = []
    with open(nodes_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                nodes.append(json.loads(line)["node_id"])
            except Exception:
                continue
    return nodes

def pick_edge_partitions():
    # Prefer enriched edges if present, else fallback to basic
    enriched = sorted(IN_DIR.glob("network_edges_enriched_*.csv"))
    if enriched:
        print(f"Using enriched edge partitions ({len(enriched)})")
        return enriched, True
    basic = sorted(IN_DIR.glob("network_edges_*.csv"))
    print(f"Using basic edge partitions ({len(basic)})")
    return basic, False

def build_graph(nodes, edge_files):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    added = 0
    for ef in edge_files:
        df = pd.read_csv(ef)
        # Normalize expected columns
        cols = {c.lower(): c for c in df.columns}
        src_col = cols.get("source", "source")
        tgt_col = cols.get("target", "target")
        wt_col  = cols.get("weight", "weight")

        for row in df.itertuples(index=False):
            src = getattr(row, src_col)
            tgt = getattr(row, tgt_col)
            if src == tgt:
                continue  # skip self-loops up front
            try:
                wt = int(getattr(row, wt_col))
            except Exception:
                wt = 1
            G.add_edge(src, tgt, weight=wt)
            added += 1

    # Ensure no self-loops remain (needed for k-core and cleaner metrics)
    G.remove_edges_from(nx.selfloop_edges(G))
    print(f"Graph built → Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} (added {added})")
    return G

def main():
    # ---- load nodes ----
    nodes_file = IN_DIR / "network_nodes.ndjson"
    nodes = load_nodes(nodes_file)
    print(f"Loaded {len(nodes)} nodes")

    # ---- pick edges ----
    edge_files, used_enriched = pick_edge_partitions()
    if not edge_files:
        print(f"No edge partitions found in {IN_DIR}")
        return

    # ---- build graph ----
    G = build_graph(nodes, edge_files)

    # ======================================================
    # ✅ Required metrics
    # ======================================================
    print("Computing required metrics…")
    indeg   = nx.in_degree_centrality(G)
    outdeg  = nx.out_degree_centrality(G)
    betw    = nx.betweenness_centrality(G, k=500, weight="weight", seed=42)  # approx for scale
    cluster = nx.clustering(G.to_undirected(), weight="weight")
    try:
        pr = nx.pagerank(G, weight="weight")  # requires scipy; wrapped in try
    except Exception as e:
        print("⚠️ PageRank failed:", e)
        pr = {}

    # ======================================================
    # ✨ Optional metrics
    # ======================================================
    print("Computing optional metrics…")
    try:
        eigen = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    except Exception as e:
        print("⚠️ Eigenvector centrality failed:", e)
        eigen = {}

    communities = {}
    if community_louvain:
        try:
            part = community_louvain.best_partition(G.to_undirected())
            communities = {n: cid for n, cid in part.items()}
        except Exception as e:
            print("⚠️ Community detection failed:", e)
    else:
        print("⚠️ python-louvain not installed → skipping community detection")

    # ======================================================
    # 🔥 Additional advanced metrics
    # ======================================================
    print("Computing additional metrics…")
    closeness = nx.closeness_centrality(G)
    try:
        kcore_dict = nx.core_number(G.to_undirected())
    except Exception as e:
        print("⚠️ K-core failed:", e)
        kcore_dict = {}

    try:
        edge_betw = nx.edge_betweenness_centrality(G, k=500, weight="weight", seed=42)
    except Exception as e:
        print("⚠️ Edge betweenness failed:", e)
        edge_betw = {}

    # ======================================================
    # Write node metrics
    # ======================================================
    out_csv = OUT_DIR / "sna_metrics.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("node_id,indegree,outdegree,betweenness,clustering_coeff,"
                "pagerank,eigenvector,closeness,community,kcore\n")
        for n in G.nodes():
            f.write(
                f"{n},{indeg.get(n,0)},{outdeg.get(n,0)},{betw.get(n,0)},"
                f"{cluster.get(n,0)},{pr.get(n,0)},{eigen.get(n,0)},"
                f"{closeness.get(n,0)},{communities.get(n,'')},{kcore_dict.get(n,0)}\n"
            )
    print(f"✅ Node metrics → {out_csv}")

    # ======================================================
    # Write edge-level metrics
    # ======================================================
    edge_csv = OUT_DIR / "edge_metrics.csv"
    with open(edge_csv, "w", encoding="utf-8") as f:
        f.write("source,target,weight,edge_betweenness\n")
        for (u, v), val in edge_betw.items():
            wt = G[u][v].get("weight", 1)
            f.write(f"{u},{v},{wt},{val}\n")
    print(f"✅ Edge metrics → {edge_csv}")

    # ======================================================
    # Write global metrics
    # ======================================================
    density = nx.density(G)

    reciprocity = None
    try:
        reciprocity = nx.reciprocity(G) if nx.is_directed(G) else None
    except Exception as e:
        print("⚠️ Reciprocity failed:", e)

    assortativity = None
    try:
        assortativity = nx.degree_assortativity_coefficient(G.to_undirected())
    except Exception as e:
        print("⚠️ Assortativity failed:", e)

    global_json = OUT_DIR / "network_global.json"
    with open(global_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "density": density,
                "reciprocity": reciprocity,
                "assortativity": assortativity,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "using_enriched_edges": used_enriched
            },
            f,
            indent=2,
        )
    print(f"✅ Global metrics → {global_json}")

if __name__ == "__main__":
    main()
