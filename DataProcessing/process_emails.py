from pathlib import Path
import email
from email import policy
from email.utils import parsedate_to_datetime, getaddresses
import csv, json, re


PROJECT_ROOT = Path(__file__).parent.resolve()
BASE_ROOT = PROJECT_ROOT.parent
DATA_ROOT = BASE_ROOT / "data"

INPUT_DIR = DATA_ROOT / "maildir"          # 🔹 entire maildir (all users, all folders)
OUTPUT_DIR = DATA_ROOT / "DataProcessing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = OUTPUT_DIR / "email_index.ndjson"


class DataProcessor:
    """
    Streams and parses raw Maildir emails into:
    - Monthly CSV partitions (message_id, date, sender, recipients, cc, bcc, subject, body)
    - NDJSON index mapping message_id to file path
    """

    def __init__(self, input_dir: Path, output_dir: Path, index_file: Path, base_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.index_file = index_file
        self.base_dir = base_dir

    def find_email_files(self, base_dir: Path):
        """Recursively find all email files under base_dir."""
        if not base_dir.exists() or not base_dir.is_dir():
            print(f"Invalid directory: {base_dir}")
            return []
        files = [p for p in base_dir.rglob('*') if not p.is_dir()]
        print(f"Emails found: {len(files)} files")
        return files

    def extract_body(self, msg):
        """Extract plain text body (concatenate multiple text/plain parts)."""
        body_parts = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body_parts.append(part.get_content().strip())
                    except Exception:
                        continue
        else:
            if msg.get_content_type() == "text/plain":
                body_parts.append(msg.get_content().strip())
        return "\n".join(body_parts)

    def clean_body(self, text: str) -> str:
        """Clean email text for sentiment analysis accuracy."""
        if not text:
            return ""
        text = re.sub(r"^>.*$", "", text, flags=re.MULTILINE)  # remove quoted lines
        text = re.split(r"-{2,}\s*Original Message\s*-{2,}", text, maxsplit=1)[0]
        text = re.sub(r"\n\s*\n", "\n\n", text)  # collapse blank lines
        return text.strip()

    def normalize_date(self, date_str):
        """Convert Date header to ISO8601 and partition key (YYYY_MM)."""
        try:
            dt = parsedate_to_datetime(date_str)
            iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            partition = dt.strftime("%Y_%m")
            return iso, partition
        except Exception:
            return None, None

    def normalize_addresses(self, field_value):
        """Normalize email addresses into lowercase, deduplicated, semicolon-delimited list."""
        if not field_value:
            return ""
        addresses = getaddresses([field_value])
        cleaned = [addr.lower() for _, addr in addresses if addr]
        return ";".join(sorted(set(cleaned)))

    def extract_headers(self, path: Path):
        """Extract and normalize headers and body from one email file."""
        try:
            with open(path, "rb") as f:
                raw = f.read()
            msg = email.message_from_bytes(raw, policy=policy.default)

            iso_date, partition = self.normalize_date(msg.get("Date"))

            return {
                "file": str(path),
                "message_id": (msg.get("Message-ID") or "").strip(),
                "date": iso_date,
                "partition": partition,
                "sender": self.normalize_addresses(msg.get("From")),
                "recipients": self.normalize_addresses(msg.get("To")),
                "cc": self.normalize_addresses(msg.get("Cc")),
                "bcc": self.normalize_addresses(msg.get("Bcc")),
                "subject": (msg.get("Subject") or "").strip(),
                "body": self.clean_body(self.extract_body(msg) or "")
            }
        except Exception as e:
            print(f"Error parsing {path}: {e}")
            return {}

    def write_partitioned_csv(self, record):
        """Append record to monthly partitioned CSV file with fixed header order."""
        partition = record["partition"] or "unknown"

        out_file = self.output_dir / f"processed_emails_{partition}.csv"
        file_exists = out_file.exists()
        fieldnames = ["message_id", "date", "sender", "recipients", "cc", "bcc", "subject", "body"]

        with open(out_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            row = {field: record.get(field, "") for field in fieldnames}
            writer.writerow(row)

    def write_index(self, record):
        """Append one line to NDJSON index file."""
        rel_path = Path(record["file"]).relative_to(self.base_dir)
        rel_path_str = str(rel_path).replace("\\", "/")
        entry = {"message_id": record["message_id"], "path": rel_path_str}
        with open(self.index_file, "a", encoding="utf-8") as f:
            json.dump(entry, f)
            f.write("\n")

    def run(self):
        """Process all emails and write outputs."""
        files = self.find_email_files(self.input_dir)
        count = 0
        for f in files:
            record = self.extract_headers(f)
            if record and record["message_id"]:
                self.write_partitioned_csv(record)
                self.write_index(record)
                count += 1
        print(f"✅ Processed {count} emails into monthly CSV partitions at {self.output_dir}")

    def verify_outputs(self):
        """Check if outputs were created correctly."""
        csv_files = list(Path(self.output_dir).glob("processed_emails_*.csv"))
        print(f"CSV partitions found: {len(csv_files)}")
        if INDEX_FILE.exists():
            with open(self.index_file, encoding="utf-8") as f:
                print(f"Index entries: {len(f.readlines())}")


if __name__ == "__main__":
    processor = DataProcessor(INPUT_DIR, OUTPUT_DIR, INDEX_FILE, DATA_ROOT)
    processor.run()
    processor.verify_outputs()
