# NetworkAnalysis/analyze_network.py
from pathlib import Path
import pandas as pd, json
import networkx as nx

PROJECT_ROOT = Path(__file__).parent.resolve()
BASE_ROOT = PROJECT_ROOT.parent
IN_DIR  = BASE_ROOT / "data" / "NetworkConstruction"
OUT_DIR = BASE_ROOT / "data" / "NetworkAnalysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # ---- load nodes ----
    nodes_file = IN_DIR / "network_nodes.ndjson"
    nodes = []
    with open(nodes_file, "r", encoding="utf-8") as f:
        for line in f:
            nodes.append(json.loads(line)["node_id"])
    print(f"Loaded {len(nodes)} nodes")

    # ---- load edges (all partitions) ----
    edge_files = list(IN_DIR.glob("network_edges_*.csv"))
    print(f"Loading {len(edge_files)} edge partitions…")

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    edge_count = 0
    for ef in edge_files:
        df = pd.read_csv(ef)
        for row in df.itertuples(index=False):
            G.add_edge(row.source, row.target, weight=int(row.weight))
            edge_count += 1
    print(f"Graph built → Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

    # ---- metrics ----
    indeg = nx.in_degree_centrality(G)
    outdeg = nx.out_degree_centrality(G)
    betw = nx.betweenness_centrality(G, k=500, weight="weight", seed=42)  # approx for scale
    cluster = nx.clustering(G.to_undirected(), weight="weight")
    pr = nx.pagerank(G, weight="weight")

    # ---- write node metrics ----
    out_csv = OUT_DIR / "sna_metrics.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("node_id,indegree,outdegree,betweenness,clustering_coeff,pagerank\n")
        for n in G.nodes():
            f.write(f"{n},{indeg.get(n,0)},{outdeg.get(n,0)},{betw.get(n,0)},{cluster.get(n,0)},{pr.get(n,0)}\n")
    print(f"✅ Node metrics → {out_csv}")

    # ---- write global density ----
    density = nx.density(G)
    with open(OUT_DIR / "network_density.json", "w", encoding="utf-8") as f:
        json.dump({"density": density}, f, indent=2)
    print(f"✅ Density → {OUT_DIR}/network_density.json")

if __name__ == "__main__":
    main()
