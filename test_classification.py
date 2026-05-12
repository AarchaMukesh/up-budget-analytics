import os
import json
from src.extraction.classifier import classify_pdf, classify_batch

# ── Config ────────────────────────────────────────────────
YEAR = "2024-25"
data_dir = f"data/raw/{YEAR}/"
output_dir = "data/metadata/"

# ── Collect PDFs ──────────────────────────────────────────
all_files = [
    os.path.join(data_dir, f)
    for f in os.listdir(data_dir)
    if f.endswith(".pdf")
]

if not all_files:
    print(f"No PDFs found in {data_dir}. Check the path.")
else:
    print(f"Found {len(all_files)} PDFs in {data_dir}")

# ── Classify with Full Metadata ───────────────────────────
print("Starting classification...")
results = {}
for file_path in all_files:
    results[os.path.basename(file_path)] = classify_pdf(
        file_path, return_metadata=True  # saves full metrics, not just label
    )

# ── Save Results ──────────────────────────────────────────
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"doc_classification_{YEAR}.json")

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nClassification complete. Results saved to {output_file}")

# ── Print Summary ─────────────────────────────────────────
print("\n── Results ──────────────────────────────────────────")
for filename, meta in results.items():
    classification = meta["classification"]
    pages = meta["total_pages"]
    label = {
        "digital": "✅ DIGITAL",
        "scanned": "❌ SCANNED",
        "mixed":   "⚠️  MIXED",
        "error":   "🔴 ERROR",
    }.get(classification, classification)
    print(f"{label}  |  {pages:>4} pages  |  {filename}")

# ── Flag anything needing OCR ─────────────────────────────
needs_ocr = [f for f, m in results.items() if m["classification"] in ("scanned", "mixed")]
if needs_ocr:
    print(f"\n⚠️  {len(needs_ocr)} file(s) need OCR treatment:")
    for f in needs_ocr:
        print(f"   → {f}")
else:
    print("\n✅ All files are machine-readable. No OCR needed.")