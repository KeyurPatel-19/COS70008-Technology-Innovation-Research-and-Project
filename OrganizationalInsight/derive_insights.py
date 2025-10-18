# -*- coding: utf-8 -*-
"""
Organizational Insights (Required + Extras)
------------------------------------------
Inputs:
  - data/SentimentalAnalysis/enriched_emails_{YYYY_MM}.csv
  - data/NetworkAnalysis/sna_metrics.csv
  - data/NetworkConstruction/network_edges_*.csv   (for Louvain communities)

Outputs (required):
  - data/OrganizationalInsight/insights_{YYYY_MM}.csv
  - data/OrganizationalInsight/insight_summary.json

Extras (helpful for Power BI and analysis):
  - data/OrganizationalInsight/influence_roster.csv   (global list of influencers)
  - data/OrganizationalInsight/community_sizes.csv    (community_id, size)
  - data/OrganizationalInsight/anomaly_topk_{YYYY_MM}.csv  (top anomalous actors per month)
"""

from pathlib import Path
import sys, json, math
import pandas as pd
import numpy as np

# ML / Graph
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import networkx as nx
try:
    import community as community_louvain  # package name: python-louvain
    HAS_LOUVAIN = True
except Exception:
    HAS_LOUVAIN = False

# --------- Paths ----------
PROJECT_ROOT = Path(__file__).parent.resolve()
BASE = PROJECT_ROOT.parent
IN_SENT_DIR = BASE / "data" / "SentimentalAnalysis"
IN_NET_DIR  = BASE / "data" / "NetworkAnalysis"
IN_CON_DIR  = BASE / "data" / "NetworkConstruction"
OUT_DIR     = BASE / "data" / "OrganizationalInsight"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = IN_NET_DIR / "sna_metrics.csv"

# --------- Helpers ----------
def qprint(msg):
    print(msg, flush=True)

def safe_mean(series):
    if len(series) == 0:
        return np.nan
    return float(np.nanmean(series))

def month_from_filename(p: Path):
    # enriched_emails_YYYY_MM.csv
    s = p.stem
    return s.replace("enriched_emails_", "")

def load_metrics():
    qprint(f"[INFO] Loading metrics: {METRICS_CSV}")
    m = pd.read_csv(METRICS_CSV)
    # Ensure required columns exist
    needed = {"node_id","indegree","outdegree","betweenness","clustering_coeff","pagerank"}
    missing = needed - set(m.columns)
    if missing:
        raise ValueError(f"sna_metrics.csv missing columns: {missing}")
    m["node_id"] = m["node_id"].astype(str)
    return m.set_index("node_id")

def build_undirected_graph_for_communities():
    qprint("[INFO] Building undirected graph for Louvain communities…")
    G = nx.Graph()
    edge_parts = sorted(IN_CON_DIR.glob("network_edges_*.csv"))
    total_rows = 0
    added = 0
    for i, ef in enumerate(edge_parts, start=1):
        df = pd.read_csv(ef, usecols=["source","target","weight"])
        total_rows += len(df)
        # add weighted edges; if parallel edges exist, sum weights
        for row in df.itertuples(index=False):
            s = str(row.source); t = str(row.target)
            w = float(row.weight) if not (isinstance(row.weight, float) and math.isnan(row.weight)) else 1.0
            if G.has_edge(s,t):
                G[s][t]["weight"] += w
            else:
                G.add_edge(s,t, weight=w)
                added += 1
        qprint(f"  - Loaded partition {i}/{len(edge_parts)} (edges processed so far: {added})")
    qprint(f"[INFO] Graph ready for community detection: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    return G

def compute_louvain_communities():
    if not HAS_LOUVAIN:
        qprint("[WARN] python-louvain not installed; falling back to connected components as communities.")
        G = build_undirected_graph_for_communities()
        comm_map = {}
        cid = 0
        for comp in nx.connected_components(G):
            for n in comp:
                comm_map[n] = cid
            cid += 1
        return comm_map
    else:
        G = build_undirected_graph_for_communities()
        part = community_louvain.best_partition(G, weight="weight", random_state=42)
        # part: dict node -> community_id
        return part

def influence_threshold(pagerank_series: pd.Series, quantile=0.95):
    thr = pagerank_series.quantile(quantile)
    return thr

def month_aggregate(enriched_path: Path):
    """Return per-sender aggregation for this month."""
    month = month_from_filename(enriched_path)
    qprint(f"[{month}] Aggregating per-sender features from {enriched_path.name} …")
    usecols = ["message_id","date","sender","recipients","cc","bcc","subject","body",
               "compound","neg","neu","pos","sentiment_label"]
    # tolerate files that may lack some columns
    try:
        df = pd.read_csv(enriched_path, usecols=usecols)
    except Exception:
        df = pd.read_csv(enriched_path)
        for c in usecols:
            if c not in df.columns:
                df[c] = np.nan

    df["sender"] = df["sender"].astype(str)
    # aggregate by sender
    grp = df.groupby("sender", dropna=False).agg(
        total_emails=("message_id","count"),
        avg_compound=("compound", safe_mean),
        avg_neg=("neg", safe_mean),
        avg_neu=("neu", safe_mean),
        avg_pos=("pos", safe_mean),
    ).reset_index().rename(columns={"sender":"node_id"})
    grp["month"] = month
    return grp

def fit_anomaly_isoforest(feat_df: pd.DataFrame, cols):
    # standardize features
    X = feat_df[cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # fit model
    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )
    iso.fit(Xs)
    score = -iso.decision_function(Xs)  # higher = more anomalous
    label = iso.predict(Xs)             # -1 anomalous, 1 normal
    feat_df["anomaly_score"] = score
    feat_df["anomaly_label"] = (label == -1).astype(int)
    return feat_df

def run_dbscan_flag(feat_df: pd.DataFrame, cols, eps=0.8, min_samples=10):
    X = feat_df[cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if len(Xs) < min_samples:
        feat_df["dbscan_label"] = -99  # not enough points
        return feat_df
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    lab = db.fit_predict(Xs)
    feat_df["dbscan_label"] = lab  # -1 = outlier
    return feat_df

def burnout_probability(feat_df: pd.DataFrame):
    """
    Heuristic burnout probability in [0,1]:
      - higher with high total_emails and high outdegree (overload / broadcast),
      - higher with low avg_compound (more negative tone),
      - tempered by clustering_coeff (supportive local ties often reduce risk).
    """
    # fillna
    for c in ["total_emails","outdegree","avg_compound","clustering_coeff"]:
        if c not in feat_df.columns:
            feat_df[c] = 0.0
        feat_df[c] = feat_df[c].astype(float).fillna(0.0)

    # scale features
    def z(x): 
        if x.std(ddof=0) == 0: 
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - x.mean()) / x.std(ddof=0)

    z_load = z(feat_df["total_emails"]) + z(feat_df["outdegree"])
    z_affect = -z(feat_df["avg_compound"])             # lower sentiment => higher risk
    z_support = -z(feat_df["clustering_coeff"]) * 0.5  # lower clustering => higher risk

    raw = 0.5*z_load + 0.4*z_affect + 0.1*z_support
    # min-max to [0,1]
    mn, mx = raw.min(), raw.max()
    if mx == mn:
        prob = pd.Series(np.zeros(len(raw)), index=raw.index)
    else:
        prob = (raw - mn) / (mx - mn)

    feat_df["burnout_prob"] = prob.astype(float)
    feat_df["burnout_label"] = (feat_df["burnout_prob"] >= 0.8).astype(int)  # tunable threshold
    return feat_df

def main():
    # ---------- Load global SNA metrics ----------
    metrics = load_metrics()  # index=node_id
    # ---------- Communities (once globally) ----------
    comm_map = compute_louvain_communities()
    qprint(f"[INFO] Communities computed for {len(comm_map)} nodes.")

    # Pre-compute global influence threshold (top 5% by PageRank)
    pr_thr = influence_threshold(metrics["pagerank"], quantile=0.95)
    qprint(f"[INFO] Influence PageRank threshold (95th pct): {pr_thr:.6g}")

    # ---------- Iterate months ----------
    enriched_months = sorted(IN_SENT_DIR.glob("enriched_emails_*.csv"))
    if not enriched_months:
        qprint("[ERROR] No enriched email files found.")
        sys.exit(1)

    all_summary = {
        "months": [],
        "total_anomalies": 0,
        "high_burnout": 0,
        "communities": 0,
        "top_influencers": []  # filled after loop (global)
    }
    influencer_global = set()
    community_counter = {}

    for i, fpath in enumerate(enriched_months, start=1):
        month = month_from_filename(fpath)
        qprint(f"\n===== [{i}/{len(enriched_months)}] Month {month} =====")
        all_summary["months"].append(month)

        # per-sender aggregate
        agg = month_aggregate(fpath)

        # join node metrics
        feat = agg.merge(metrics.reset_index(), how="left", on="node_id")

        # add communities & influence flag
        feat["community_id"] = feat["node_id"].map(comm_map).fillna(-1).astype(int)
        feat["influence_flag"] = (feat["pagerank"] >= pr_thr).astype(int)

        # count communities (for summary)
        m_counts = feat["community_id"].value_counts().to_dict()
        for k, v in m_counts.items():
            community_counter[k] = community_counter.get(k, 0) + v

        # anomalies (IsolationForest + DBSCAN)
        iso_cols = ["total_emails","avg_compound","indegree","outdegree","betweenness","clustering_coeff","pagerank"]
        for c in iso_cols:
            if c not in feat.columns: feat[c] = 0.0
            feat[c] = feat[c].astype(float).fillna(0.0)

        feat = fit_anomaly_isoforest(feat, iso_cols)
        feat = run_dbscan_flag(feat, ["total_emails","avg_compound"], eps=0.8, min_samples=10)

        # burnout (heuristic)
        feat = burnout_probability(feat)

        # month summary counters
        m_anom = int(feat["anomaly_label"].sum())
        m_burn = int(feat["burnout_label"].sum())
        all_summary["total_anomalies"] += m_anom
        all_summary["high_burnout"] += m_burn
        influencer_global.update(feat.loc[feat["influence_flag"] == 1, "node_id"].tolist())

        qprint(f"[{month}] Actors={len(feat):,} | anomalies={m_anom:,} | high_burnout={m_burn:,} | influencers(add)={feat['influence_flag'].sum():,}")

        # ------- write per-month insights (REQUIRED) -------
        out_csv = OUT_DIR / f"insights_{month}.csv"
        keep_cols = [
            "node_id","month",
            "total_emails","avg_compound","avg_neg","avg_neu","avg_pos",
            "indegree","outdegree","betweenness","clustering_coeff","pagerank",
            "anomaly_score","anomaly_label","dbscan_label",
            "community_id","influence_flag",
            "burnout_prob","burnout_label"
        ]
        # ensure every column exists
        for c in keep_cols:
            if c not in feat.columns:
                feat[c] = np.nan
        feat[keep_cols].to_csv(out_csv, index=False)
        qprint(f"[{month}] ✅ insights → {out_csv}")

        # ------- write extras: top anomalies per month -------
        topk = feat.sort_values("anomaly_score", ascending=False).head(100)
        topk[["node_id","month","anomaly_score","total_emails","avg_compound","community_id","influence_flag","burnout_prob"]].to_csv(
            OUT_DIR / f"anomaly_topk_{month}.csv", index=False
        )

    # ---------- global extras ----------
    # influence roster
    inf_df = pd.DataFrame({"node_id": sorted(influencer_global)})
    inf_df = inf_df.merge(metrics.reset_index(), how="left", on="node_id")
    inf_df.to_csv(OUT_DIR / "influence_roster.csv", index=False)

    # community sizes
    comm_df = pd.DataFrame(
        sorted(community_counter.items(), key=lambda kv: kv[1], reverse=True),
        columns=["community_id","size"]
    )
    comm_df.to_csv(OUT_DIR / "community_sizes.csv", index=False)

    # summary JSON (REQUIRED)
    all_summary["communities"] = int((pd.Series(list(community_counter.keys())) != -1).sum())
    # top influencers by PageRank
    top_inf = metrics.reset_index().sort_values("pagerank", ascending=False).head(50)["node_id"].tolist()
    all_summary["top_influencers"] = top_inf

    with open(OUT_DIR / "insight_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2)

    qprint("\n======== Done ========")
    qprint(f"✅ Summary → {OUT_DIR/'insight_summary.json'}")
    qprint(f"✅ Influence roster → {OUT_DIR/'influence_roster.csv'}")
    qprint(f"✅ Community sizes → {OUT_DIR/'community_sizes.csv'}")
    qprint("Use these directly in Power BI pages: Burnout, Organizational Insights, Overview.")

if __name__ == "__main__":
    main()
