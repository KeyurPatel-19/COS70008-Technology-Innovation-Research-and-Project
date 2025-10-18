# SentimentalAnalysis/analyze_sentiment.py
from pathlib import Path
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------- paths ---------
PROJECT_ROOT = Path(__file__).parent.resolve()
BASE_ROOT = PROJECT_ROOT.parent
IN_DIR  = BASE_ROOT / "data" / "DataProcessing"         # processed_emails_{YYYY_MM}.csv
OUT_DIR = BASE_ROOT / "data" / "SentimentalAnalysis"    # sentiment_scores_*.csv + enriched_emails_*.csv
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK = 10_000  # rows per chunk; adjust if needed

# --------- sentiment helpers ---------
analyzer = SentimentIntensityAnalyzer()

def score_text(subject: str, body: str):
    text = f"{subject or ''}\n{body or ''}".strip()
    vs = analyzer.polarity_scores(text)
    return vs["compound"], vs["neg"], vs["neu"], vs["pos"]

def label_from_compound(c: float):
    # VADER convention
    if c >= 0.05:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"

# --------- processing ---------
def process_month(csv_path: Path):
    # infer YYYY_MM from filename: processed_emails_2000_07.csv
    stem = csv_path.stem  # processed_emails_2000_07
    month_tag = stem.replace("processed_emails_", "")

    scores_out   = OUT_DIR / f"sentiment_scores_{month_tag}.csv"
    enriched_out = OUT_DIR / f"enriched_emails_{month_tag}.csv"

    wrote_scores_header = False
    wrote_enriched_header = False

    usecols = ["message_id","date","sender","recipients","cc","bcc","subject","body"]

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=CHUNK):
        # ensure expected columns exist
        for col in usecols:
            if col not in chunk.columns:
                chunk[col] = ""

        # compute scores
        comps, negs, neus, poss = [], [], [], []
        for subj, body in zip(chunk["subject"], chunk["body"]):
            c, n, u, p = score_text(subj, str(body))
            comps.append(c); negs.append(n); neus.append(u); poss.append(p)

        chunk["compound"] = comps
        chunk["neg"] = negs
        chunk["neu"] = neus
        chunk["pos"] = poss
        chunk["sentiment_label"] = [label_from_compound(c) for c in comps]

        # write scores-only file
        scores_cols = ["message_id","compound","neg","neu","pos","sentiment_label"]
        chunk[scores_cols].to_csv(
            scores_out, mode="a", index=False, header=not wrote_scores_header
        )
        wrote_scores_header = True

        # write enriched (original + sentiment)
        enriched_cols = usecols + ["compound","neg","neu","pos","sentiment_label"]
        chunk[enriched_cols].to_csv(
            enriched_out, mode="a", index=False, header=not wrote_enriched_header
        )
        wrote_enriched_header = True

def main():
    inputs = sorted((IN_DIR).glob("processed_emails_*.csv"))
    if not inputs:
        print(f"No input files found in {IN_DIR}")
        return

    print(f"Found {len(inputs)} monthly files.")
    for i, csv_path in enumerate(inputs, 1):
        print(f"[{i}/{len(inputs)}] {csv_path.name} → scoring…")
        process_month(csv_path)
    print(f"✅ Done. Outputs written to: {OUT_DIR}")

if __name__ == "__main__":
    main()
