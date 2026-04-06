"""
tests/test_preprocessing.py
----------------------------
Unit tests for DOM pruning — no network or API calls needed.
Run: python -m pytest tests/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag_scraper import preprocess_html
from src.vectorless_scraper import prune_dom

SAMPLE_HTML = """
<html>
<head><title>Shop</title><script>alert(1)</script><style>body{color:red}</style></head>
<body>
  <nav><a href="/">Home</a></nav>
  <header><h2>Store Header</h2></header>
  <main>
    <h1 class="product-title" data-irrelevant="yes">Blue Shirt</h1>
    <p class="price" id="main-price">$29.99</p>
    <div class="description">A great shirt for everyday use.</div>
  </main>
  <footer><p>Copyright 2024</p></footer>
  <script>console.log("noise")</script>
</body>
</html>
"""

def test_prune_removes_scripts():
    result = prune_dom(SAMPLE_HTML)
    assert "<script>" not in result
    assert "alert(1)" not in result

def test_prune_removes_nav_footer():
    result = prune_dom(SAMPLE_HTML)
    assert "<nav>" not in result
    assert "<footer>" not in result

def test_prune_keeps_product_content():
    result = prune_dom(SAMPLE_HTML)
    assert "product-title" in result
    assert "Blue Shirt" in result
    assert "price" in result
    assert "$29.99" in result

def test_preprocess_strips_irrelevant_attrs():
    result = preprocess_html(SAMPLE_HTML)
    assert "data-irrelevant" not in result

def test_prune_respects_max_chars():
    result = prune_dom(SAMPLE_HTML, max_chars=50)
    assert len(result) <= 50

if __name__ == "__main__":
    test_prune_removes_scripts();     print("✓ removes scripts")
    test_prune_removes_nav_footer();  print("✓ removes nav/footer")
    test_prune_keeps_product_content(); print("✓ keeps product content")
    test_preprocess_strips_irrelevant_attrs(); print("✓ strips irrelevant attrs")
    test_prune_respects_max_chars();  print("✓ respects max_chars")
    print("\nAll tests passed.")
