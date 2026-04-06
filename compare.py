"""
Side-by-side comparison: Traditional RAG vs Vector-less RAG
=============================================================
Runs both scrapers on the same URLs and prints a comparison table.

Requirements:
  pip install anthropic chromadb sentence-transformers beautifulsoup4 requests
"""

import time
from rag_scraper import index_examples, extract_with_rag, EXAMPLE_BANK
from vectorless_scraper import extract_vectorless

TEST_URLS = [
    "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
    "https://books.toscrape.com/catalogue/tipping-the-velvet_999/index.html",
    "https://books.toscrape.com/catalogue/soumission_998/index.html",
]
FIELDS = ["title", "price"]

def run_comparison():
    print("\n  Setting up RAG index...")
    index_examples(EXAMPLE_BANK)

    rows = []
    for url in TEST_URLS:
        book = url.split("/")[-2]
        for field in FIELDS:
            print(f"\n  [{field}] {book}")

            # RAG
            t0  = time.time()
            rag = extract_with_rag(url, field, verbose=False)
            t_rag = time.time() - t0

            # Vector-less
            t0  = time.time()
            vl  = extract_vectorless(url, field, verbose=False)
            t_vl = time.time() - t0

            rows.append({
                "book":        book[:30],
                "field":       field,
                "rag_sel":     rag["selector"],
                "rag_val":     rag["value"] or "—",
                "rag_time":    round(t_rag, 2),
                "vl_sel":      vl["selector"],
                "vl_val":      vl["value"] or "—",
                "vl_time":     round(t_vl, 2),
                "match":       rag["value"] == vl["value"],
            })

    # ── Print table ────────────────────────────────────────────────────────────
    W = 80
    print(f"\n{'═'*W}")
    print(f"{'COMPARISON RESULTS':^{W}}")
    print(f"{'═'*W}")
    print(f"{'Book':<32} {'Field':<6}  {'RAG selector':<22} {'VL selector':<22} Match")
    print(f"{'─'*W}")
    for r in rows:
        match = "✓" if r["match"] else "✗"
        print(f"{r['book']:<32} {r['field']:<6}  {r['rag_sel']:<22} {r['vl_sel']:<22}  {match}")
        print(f"{'':32} {'':6}  val={r['rag_val']:<19} val={r['vl_val']}")
        print(f"{'':32} {'':6}  t={r['rag_time']}s{'':<17} t={r['vl_time']}s")
        print(f"{'─'*W}")

    matched = sum(1 for r in rows if r["match"])
    print(f"\n  Agreement: {matched}/{len(rows)} results matched between methods")
    avg_rag = sum(r["rag_time"] for r in rows) / len(rows)
    avg_vl  = sum(r["vl_time"] for r in rows) / len(rows)
    print(f"  Avg time  — RAG: {round(avg_rag,2)}s   Vector-less: {round(avg_vl,2)}s")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    run_comparison()
