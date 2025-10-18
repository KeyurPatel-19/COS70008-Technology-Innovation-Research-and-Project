# -*- coding: utf-8 -*-
"""
Module 7 — Interactive Visualization (Power BI data builder)
Required outputs (per month):
  - timeseries_data_{YYYY_MM}.csv     (date, avg_compound, total_emails)
  - burnout_bar_data_{YYYY_MM}.csv    (node_id, burnout_prob, burnout_label, community_id)
  - network_snapshot_data_{YYYY_MM}.csv (node_id, x, y, +metrics +risk)

Extras (per month):
  - influence_trend_{YYYY_MM}.csv
  - community_heatmap_{YYYY_MM}.csv
  - anomaly_log_{YYYY_MM}.csv

Also writes concatenated ALL files for simple Power BI models:
  - timeseries_data_ALL.csv
  - burnout_bar_data_ALL.csv
  - network_snapshot_data_ALL.csv

And a config:
  - visual_config.json
"""

from pathlib import Path
import pandas as pd, json
from datetime import datetime

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).parent.resolve()
BASE         = PROJECT_ROOT.parent
DIR_SENT     = BASE / "data" / "SentimentalAnalysis"
DIR_MET      = BASE / "data" / "NetworkAnalysis"
DIR_INS      = BASE / "data" / "OrganizationalInsight"
DIR_LAYOUT   = BASE / "data" / "NetworkGraphAnalysis"
OUT_DIR      = BASE / "data" / "InteractiveVisualization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg): print(msg, flush=True)

# ---------- Helpers ----------
def tag_from(path: Path, prefix: str) -> str:
    # e.g. sentiment_scores_1997_06.csv -> 1997_06
    return path.stem.replace(prefix, "")

def discover_months():
    # union of months found anywhere (insights, enriched_emails, sentiment_scores)
    months = set()
    months |= {tag_from(p, "insights_") for p in DIR_INS.glob("insights_*.csv")}
    months |= {tag_from(p, "enriched_emails_") for p in DIR_SENT.glob("enriched_emails_*.csv")}
    months |= {tag_from(p, "sentiment_scores_") for p in DIR_SENT.glob("sentiment_scores_*.csv")}
    months = {m for m in months if len(m) == 7 and m[4] == "_"}  # YYYY_MM sanity
    return sorted(months)

def safe_read_csv(path: Path, usecols=None, parse_dates=None):
    try:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates)
    except Exception:
        df = pd.read_csv(path)
        if usecols:
            for u in usecols:
                if u not in df.columns:
                    df[u] = pd.NA
        if parse_dates:
            for c in parse_dates:
                if c in df.columns:
                    try:
                        df[c] = pd.to_datetime(df[c], errors="coerce")
                    except Exception:
                        pass
        return df

def load_layout(tag: str) -> pd.DataFrame:
    # prefer per-month; fallback to ALL or GLOBAL
    candidates = [
        DIR_LAYOUT / f"graph_layout_{tag}.ndjson",
        DIR_LAYOUT / "graph_layout_ALL.ndjson",
        DIR_LAYOUT / "graph_layout_GLOBAL.ndjson",
    ]
    for c in candidates:
        if c.exists():
            return pd.read_json(c, lines=True)  # columns: node_id,x,y
    raise FileNotFoundError("No graph_layout for tag or ALL/GLOBAL.")

def load_metrics() -> pd.DataFrame:
    f = DIR_MET / "sna_metrics.csv"
    if not f.exists():
        log("⚠️  sna_metrics.csv not found. Snapshots will include x,y + insight risk only.")
        return pd.DataFrame({"node_id": []})
    df = pd.read_csv(f)
    df["node_id"] = df["node_id"].astype(str)
    return df

# ---------- Builders (per month) ----------
def build_timeseries(tag: str):
    # Must use enriched_emails_{tag}.csv for the 'date' column
    f = DIR_SENT / f"enriched_emails_{tag}.csv"
    if not f.exists():
        log(f"  ⚠️  timeseries skipped: {f.name} missing")
        return None
    df = safe_read_csv(f, usecols=["message_id","date","compound"], parse_dates=["date"])
    # convert to day
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    ts = df.groupby("date", dropna=True).agg(
        avg_compound=("compound", "mean"),
        total_emails=("message_id", "count")
    ).reset_index()
    out = OUT_DIR / f"timeseries_data_{tag}.csv"
    ts.to_csv(out, index=False)
    log(f"  ✅ {out.name}")
    ts["month"] = tag
    return ts

def build_burnout_bar(tag: str):
    f = DIR_INS / f"insights_{tag}.csv"
    if not f.exists():
        log(f"  ⚠️  burnout bar skipped: {f.name} missing")
        return None
    df = safe_read_csv(f, usecols=["node_id","burnout_prob","burnout_label","community_id"])
    out = OUT_DIR / f"burnout_bar_data_{tag}.csv"
    df.to_csv(out, index=False)
    log(f"  ✅ {out.name}")
    df["month"] = tag
    return df

def build_network_snapshot(tag: str, metrics_df: pd.DataFrame):
    # layout
    try:
        layout = load_layout(tag)  # node_id,x,y
    except FileNotFoundError:
        log(f"  ❌ No layout file for {tag} (and no ALL/GLOBAL fallback). Skipping snapshot.")
        return None
    # insights for this month (to merge risk columns)
    f_ins = DIR_INS / f"insights_{tag}.csv"
    if not f_ins.exists():
        log(f"  ⚠️  snapshot skipped: insights_{tag}.csv missing")
        return None
    ins = safe_read_csv(f_ins, usecols=["node_id","anomaly_score","burnout_prob","community_id","influence_flag"])
    # metrics
    snap = layout.merge(metrics_df, on="node_id", how="left").merge(ins, on="node_id", how="left")
    out = OUT_DIR / f"network_snapshot_data_{tag}.csv"
    snap.to_csv(out, index=False)
    log(f"  ✅ {out.name}")
    snap["month"] = tag
    return snap

# ---------- Extras (per month) ----------
def build_influence_trend(tag: str, metrics_df: pd.DataFrame):
    if "pagerank" not in metrics_df.columns:
        return None
    inf = metrics_df[["node_id","pagerank"]].copy()
    out = OUT_DIR / f"influence_trend_{tag}.csv"
    inf.to_csv(out, index=False)
    log(f"  ✅ {out.name}")
    inf["month"] = tag
    return inf

def build_community_heatmap(tag: str):
    f = DIR_INS / f"insights_{tag}.csv"
    if not f.exists():
        return None
    ins = safe_read_csv(f, usecols=["community_id","avg_compound"])
    heat = ins.groupby("community_id", dropna=True).agg(avg_compound=("avg_compound","mean")).reset_index()
    out = OUT_DIR / f"community_heatmap_{tag}.csv"
    heat.to_csv(out, index=False)
    log(f"  ✅ {out.name}")
    heat["month"] = tag
    return heat

def build_anomaly_log(tag: str, topk: int = 200):
    f = DIR_INS / f"insights_{tag}.csv"
    if not f.exists():
        return None
    ins = safe_read_csv(f, usecols=["node_id","anomaly_score","burnout_prob","community_id"])
    top = ins.sort_values("anomaly_score", ascending=False).head(min(topk, len(ins))).reset_index(drop=True)
    out = OUT_DIR / f"anomaly_log_{tag}.csv"
    top.to_csv(out, index=False)
    log(f"  ✅ {out.name}")
    top["month"] = tag
    return top

# ---------- Visual config ----------
def write_visual_config(months):
    cfg = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "months": months,
        "defaults": {
            "time_series": {"chart_type": "line"},
            "burnout_bar": {"chart_type": "bar"},
            "network_snapshot": {
                "chart_type": "scatter",
                "size_by": "pagerank",
                "color_by": "community_id"
            }
        },
        "filters": ["month","community_id","influence_flag","burnout_label","anomaly_score"]
    }
    with open(OUT_DIR / "visual_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    log("  ✅ visual_config.json")

# ---------- Main ----------
def main():
    log("=== Module 7: Interactive Visualization (Power BI) ===")
    months = discover_months()
    if not months:
        raise SystemExit("No months found (sentiment_scores_*.csv / enriched_emails_*.csv / insights_*.csv).")

    metrics_df = load_metrics()
    all_ts, all_burn, all_snap = [], [], []

    total = len(months)
    for i, m in enumerate(months, start=1):
        log(f"\n[{i}/{total}] Building outputs for {m} …")
        ts   = build_timeseries(m)
        burn = build_burnout_bar(m)
        snap = build_network_snapshot(m, metrics_df)
        build_influence_trend(m, metrics_df)
        build_community_heatmap(m)
        build_anomaly_log(m)

        if ts   is not None: all_ts.append(ts)
        if burn is not None: all_burn.append(burn)
        if snap is not None: all_snap.append(snap)

        log(f"[{i}/{total}] ✅ Done {m}")

    # Concatenate ALL tables for easy BI models
    if all_ts:
        pd.concat(all_ts, ignore_index=True).to_csv(OUT_DIR / "timeseries_data_ALL.csv", index=False)
        log("  ✅ timeseries_data_ALL.csv")
    if all_burn:
        pd.concat(all_burn, ignore_index=True).to_csv(OUT_DIR / "burnout_bar_data_ALL.csv", index=False)
        log("  ✅ burnout_bar_data_ALL.csv")
    if all_snap:
        pd.concat(all_snap, ignore_index=True).to_csv(OUT_DIR / "network_snapshot_data_ALL.csv", index=False)
        log("  ✅ network_snapshot_data_ALL.csv")

    # Visual config once
    write_visual_config(months)

    log("\n🎉 All Power BI inputs are ready in:")
    log(f"   {OUT_DIR}")

if __name__ == "__main__":
    main()
