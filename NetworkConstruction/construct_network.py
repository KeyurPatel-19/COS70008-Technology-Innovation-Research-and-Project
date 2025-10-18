# NetworkConstruction/construct_network.py
from pathlib import Path
import pandas as pd, json, csv, string
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.resolve()
BASE_ROOT = PROJECT_ROOT.parent
IN_DIR  = BASE_ROOT / "data" / "SentimentalAnalysis"     # enriched_emails_{YYYY_MM}.csv
OUT_DIR = BASE_ROOT / "data" / "NetworkConstruction"     # network_edges_{X}.csv, network_nodes.ndjson, network_meta.json
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK = 25_000
PARTITIONS = list(string.ascii_uppercase + string.digits) + ["misc"]

def part_key(email_addr: str) -> str:
    if not email_addr:
        return "misc"
    c = email_addr.strip()[:1].upper()
    return c if c in string.ascii_uppercase + string.digits else "misc"

def parse_list(val: str):
    """Parse semicolon-delimited recipient list into a clean list"""
    if not isinstance(val, str) or not val.strip():
        return []
    return [x for x in (p.strip() for p in val.split(";")) if x]

def main():
    files = sorted(IN_DIR.glob("enriched_emails_*.csv"))
    if not files:
        print(f"No input files in {IN_DIR}")
        return

    # per-partition accumulator
    agg = {p: defaultdict(lambda: {"w":0, "sum_c":0.0, "first":None, "last":None}) for p in PARTITIONS}
    nodes = set()

    usecols = ["date","sender","recipients","cc","bcc","compound"]
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}")
        for chunk in pd.read_csv(f, usecols=usecols, chunksize=CHUNK):
            chunk["date"] = chunk["date"].astype(str)
            for _, row in chunk.iterrows():
                # --- sender ---
                s = str(row.get("sender") or "").strip()
                if not s:
                    continue
                nodes.add(s)

                # --- recipients ---
                rcpts = (
                    parse_list(str(row.get("recipients") or "")) +
                    parse_list(str(row.get("cc") or "")) +
                    parse_list(str(row.get("bcc") or ""))
                )
                if not rcpts:
                    continue

                # --- sentiment + date ---
                try:
                    c = float(row.get("compound") or 0.0)
                except Exception:
                    c = 0.0
                d = str(row.get("date") or "")

                # --- update edges ---
                for t in set(rcpts):
                    t = str(t).strip()
                    if not t:
                        continue
                    nodes.add(t)
                    pk = part_key(s)
                    entry = agg[pk][(s,t)]
                    entry["w"] += 1
                    entry["sum_c"] += c
                    if not entry["first"] or (d and d < entry["first"]):
                        entry["first"] = d
                    if not entry["last"] or (d and d > entry["last"]):
                        entry["last"] = d

    # --- write edges (SPEC: network_edges_{X}.csv) ---
    total_u, total_w = 0, 0
    for p in PARTITIONS:
        outp = OUT_DIR / f"network_edges_{p}.csv"
        with open(outp, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["source","target","weight","mean_compound","first_seen","last_seen"])
            for (src,tgt), e in agg[p].items():
                mean_c = e["sum_c"]/e["w"] if e["w"] else 0.0
                w.writerow([src, tgt, e["w"], f"{mean_c:.6f}", e["first"], e["last"]])
                total_u += 1
                total_w += e["w"]

    # --- write nodes (SPEC: network_nodes.ndjson) ---
    with open(OUT_DIR / "network_nodes.ndjson", "w", encoding="utf-8") as nf:
        for n in sorted(nodes):
            nf.write(json.dumps({"node_id": n, "name": ""}) + "\n")

    # --- write meta (SPEC: network_meta.json with 'partitions') ---
    meta = {
        "partitions": [f"network_edges_{p}.csv" for p in PARTITIONS],
        "total_nodes": len(nodes),
        "total_edges_unique": total_u,
        "total_edges_weighted": total_w
    }
    with open(OUT_DIR / "network_meta.json", "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    print("✅ Edge partitions and metadata written (spec-compliant).")
    print(f"nodes: {meta['total_nodes']} | edges_unique: {meta['total_edges_unique']} | edges_weighted: {meta['total_edges_weighted']}")

if __name__ == "__main__":
    main()
