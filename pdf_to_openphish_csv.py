# scripts/pdf_to_openphish_csv.py
import re
from pathlib import Path
import csv
import sys

# Use PyPDF2 (or pdfplumber if installed). PyPDF2 is usually available.
try:
    import PyPDF2
except Exception:
    print("Please install PyPDF2: python -m pip install PyPDF2")
    raise

PDF_PATH = Path("data/raw/openphish_labelled.csv.pdf")
OUT_CSV  = Path("data/raw/openphish_labelled.csv")

url_re = re.compile(
    r"""(?:(?:http|https)://|www\.)[^\s'"]+""",
    re.IGNORECASE
)

def extract_text_from_pdf(p):
    txt = []
    with open(p, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                txt.append(page.extract_text() or "")
            except Exception:
                pass
    return "\n".join(txt)

def normalize_url(u):
    u = u.strip()
    # Some lines have trailing punctuation or line breaks; chop them
    u = u.rstrip(".,;:()[]{}<>\"'")
    # ensure scheme
    if u.startswith("www."):
        u = "http://" + u
    return u

def main():
    if not PDF_PATH.exists():
        print("PDF file not found:", PDF_PATH)
        sys.exit(1)
    txt = extract_text_from_pdf(PDF_PATH)
    matches = set()
    for m in url_re.findall(txt):
        m2 = normalize_url(m)
        # ignore obviously invalid short tokens
        if len(m2) < 10:
            continue
        matches.add(m2)
    print(f"Found {len(matches)} candidate URLs")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["url","label"])
        for u in sorted(matches):
            writer.writerow([u, 1])
    print("Wrote", OUT_CSV)

if __name__ == "__main__":
    main()
