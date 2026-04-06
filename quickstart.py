"""
examples/quickstart.py
----------------------
Minimal working example for both approaches.
Run: python examples/quickstart.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag_scraper import index_examples, extract_with_rag, EXAMPLE_BANK
from src.vectorless_scraper import extract_vectorless

URL = "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"

print("\n── Vector-less (no setup needed) ────────────────────")
for field in ["title", "price"]:
    r = extract_vectorless(URL, field, verbose=False)
    print(f"  {field:6s}: {r['value']}  (selector: {r['selector']})")

print("\n── Traditional RAG ──────────────────────────────────")
index_examples(EXAMPLE_BANK)
for field in ["title", "price"]:
    r = extract_with_rag(URL, field, verbose=False)
    print(f"  {field:6s}: {r['value']}  (selector: {r['selector']})")
