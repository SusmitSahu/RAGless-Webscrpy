"""
Traditional RAG Web Scraper
============================
Extracts structured fields (title, price, etc.) from any product page
using a vector DB of labelled examples + LLM generation.

Pipeline:
  1. Index: store (html_snippet, selector, field) examples as embeddings
  2. Retrieve: embed new page → cosine search for similar past examples
  3. Generate: prompt Claude with retrieved examples → get selector
  4. Apply: run the selector on the live DOM → extract value

Requirements:
  pip install anthropic chromadb sentence-transformers beautifulsoup4 requests
"""

import os
import re
import json
import requests
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_MODEL  = "all-MiniLM-L6-v2"          # local, free, fast
CLAUDE_MODEL = "claude-sonnet-4-20250514"
TOP_K        = 3                            # how many examples to retrieve
MAX_SNIPPET  = 2000                         # chars of page HTML to embed & send

# ── Clients ────────────────────────────────────────────────────────────────────
embedder = SentenceTransformer(EMBED_MODEL)
chroma   = chromadb.Client()                # in-memory; swap for PersistentClient for disk
col      = chroma.get_or_create_collection("selector_examples")
claude   = anthropic.Anthropic()            # reads ANTHROPIC_API_KEY from env


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Example bank
# These are the "documents" we index into the vector DB.
# In production you'd grow this from real labelled scraping runs.
# ══════════════════════════════════════════════════════════════════════════════

EXAMPLE_BANK = [
    # ── title examples ────────────────────────────────────────────────────────
    {
        "field": "title",
        "html":  '<h1 class="product-title">Wireless Headphones Pro</h1>',
        "selector": "h1.product-title",
        "note": "standard h1 with product-title class"
    },
    {
        "field": "title",
        "html":  '<h1 id="productTitle" class="a-size-large">Echo Dot (5th Gen)</h1>',
        "selector": "#productTitle",
        "note": "Amazon-style id selector"
    },
    {
        "field": "title",
        "html":  '<div class="pdp-name"><h1>Blue Cotton Shirt</h1></div>',
        "selector": ".pdp-name h1",
        "note": "h1 nested inside a product detail panel"
    },
    {
        "field": "title",
        "html":  '<span itemprop="name" class="product_name">Leather Wallet</span>',
        "selector": '[itemprop="name"]',
        "note": "schema.org itemprop attribute"
    },
    {
        "field": "title",
        "html":  '<h1 class="title">A Light in the Attic</h1>',
        "selector": "h1.title",
        "note": "books.toscrape style"
    },

    # ── price examples ────────────────────────────────────────────────────────
    {
        "field": "price",
        "html":  '<span class="price">$29.99</span>',
        "selector": "span.price",
        "note": "simple price span"
    },
    {
        "field": "price",
        "html":  '<span id="priceblock_ourprice" class="a-price-whole">$149</span>',
        "selector": "#priceblock_ourprice",
        "note": "Amazon price block"
    },
    {
        "field": "price",
        "html":  '<p class="price_color">£51.77</p>',
        "selector": "p.price_color",
        "note": "books.toscrape price"
    },
    {
        "field": "price",
        "html":  '<div class="product-price" data-price="1499">₹1,499</div>',
        "selector": ".product-price",
        "note": "price in data attribute and text"
    },
    {
        "field": "price",
        "html":  '<span itemprop="price" content="39.95">$39.95</span>',
        "selector": '[itemprop="price"]',
        "note": "schema.org price markup"
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Index
# ══════════════════════════════════════════════════════════════════════════════

def index_examples(examples: list[dict], force_reindex: bool = False) -> None:
    """Embed each example HTML and store in ChromaDB."""
    if col.count() > 0 and not force_reindex:
        print(f"[index] Already indexed {col.count()} examples. Skipping.")
        return

    print(f"[index] Embedding {len(examples)} examples...")
    for i, ex in enumerate(examples):
        embedding = embedder.encode(ex["html"]).tolist()
        col.add(
            ids=[f"ex_{i}"],
            embeddings=[embedding],
            metadatas=[{
                "selector": ex["selector"],
                "field":    ex["field"],
                "note":     ex.get("note", ""),
            }],
            documents=[ex["html"]],
        )
    print(f"[index] Done. {col.count()} examples in DB.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DOM preprocessing
# ══════════════════════════════════════════════════════════════════════════════

REMOVE_TAGS  = {"script", "style", "noscript", "head", "nav",
                "footer", "header", "aside", "svg", "iframe", "img", "picture"}
KEEP_ATTRS   = {"id", "class", "itemprop", "data-price", "content", "href"}

def preprocess_html(raw_html: str, max_chars: int = MAX_SNIPPET) -> str:
    """
    Strip noise from raw HTML and return a compact snippet
    suitable for both embedding and LLM context.
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove noisy tags
    for tag in soup(list(REMOVE_TAGS)):
        tag.decompose()

    # Strip irrelevant attributes
    for tag in soup.find_all(True):
        tag.attrs = {k: v for k, v in tag.attrs.items() if k in KEEP_ATTRS}

    # Collapse whitespace
    text = soup.prettify()
    text = re.sub(r'\n\s*\n', '\n', text)

    return text[:max_chars]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Retrieve
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_examples(snippet: str, field: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the page snippet and retrieve the most similar
    stored examples for the requested field.
    """
    embedding = embedder.encode(snippet).tolist()
    results   = col.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where={"field": field},           # filter by field (title / price)
    )
    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "html":     doc,
            "selector": meta["selector"],
            "note":     meta["note"],
            "score":    round(1 - dist, 3),  # cosine similarity
        })
    return retrieved


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Generate selector via LLM
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(snippet: str, field: str, examples: list[dict]) -> str:
    examples_block = "\n".join(
        f'  HTML: {ex["html"]}\n  → Selector: {ex["selector"]}  ({ex["note"]})'
        for ex in examples
    )
    return f"""You are a CSS selector expert for web scraping.

Here are {len(examples)} examples of HTML snippets and their correct CSS selectors
for extracting '{field}':

{examples_block}

Now look at this real page DOM and generate ONE CSS selector that extracts '{field}'.

Rules:
- Prefer class or id selectors over positional (nth-child) ones
- If you see itemprop="name" or itemprop="price", prefer those
- Return ONLY the CSS selector, nothing else — no explanation, no backticks

Page DOM:
{snippet}"""


def generate_selector(snippet: str, field: str, examples: list[dict]) -> str:
    prompt   = build_prompt(snippet, field, examples)
    response = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=80,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip().strip("`")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Apply selector
# ══════════════════════════════════════════════════════════════════════════════

def apply_selector(raw_html: str, selector: str) -> str | None:
    soup = BeautifulSoup(raw_html, "html.parser")
    el   = soup.select_one(selector)
    if el is None:
        return None
    # Prefer content attribute (schema.org pattern) over text
    return el.get("content") or el.get_text(strip=True) or None


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — one function to call
# ══════════════════════════════════════════════════════════════════════════════

def extract_with_rag(url: str, field: str, verbose: bool = True) -> dict:
    """
    Full RAG pipeline: fetch → preprocess → retrieve → generate → apply.

    Returns:
        {
            "url": ...,
            "field": ...,
            "selector": "p.price_color",
            "value": "£51.77",
            "retrieved_examples": [...],
        }
    """
    if verbose:
        print(f"\n{'═'*55}")
        print(f"  Extracting '{field}' from: {url}")
        print(f"{'═'*55}")

    # Fetch
    resp     = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    raw_html = resp.text

    # Preprocess
    snippet = preprocess_html(raw_html)
    if verbose:
        print(f"[preprocess] DOM reduced to {len(snippet)} chars")

    # Retrieve
    examples = retrieve_examples(snippet, field)
    if verbose:
        print(f"[retrieve]   Top {len(examples)} examples:")
        for ex in examples:
            print(f"             {ex['selector']:30s}  (sim={ex['score']})")

    # Generate
    selector = generate_selector(snippet, field, examples)
    if verbose:
        print(f"[generate]   Selector → {selector}")

    # Apply
    value = apply_selector(raw_html, selector)
    if verbose:
        status = "✓" if value else "✗ not found"
        print(f"[apply]      Value    → {value or status}")

    return {
        "url":               url,
        "field":             field,
        "selector":          selector,
        "value":             value,
        "retrieved_examples": examples,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 1. Build the index once
    index_examples(EXAMPLE_BANK)

    # 2. Target URL (books.toscrape.com — no auth, stable, great for testing)
    TEST_URLS = [
        "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
        "https://books.toscrape.com/catalogue/tipping-the-velvet_999/index.html",
        "https://books.toscrape.com/catalogue/soumission_998/index.html",
    ]

    results = []
    for url in TEST_URLS:
        for field in ["title", "price"]:
            result = extract_with_rag(url, field)
            results.append(result)

    # 3. Print clean summary
    print(f"\n{'═'*55}")
    print("  RESULTS SUMMARY")
    print(f"{'═'*55}")
    for r in results:
        book = r["url"].split("/")[-2]
        print(f"  {r['field']:6s}  {r['selector']:25s}  →  {r['value']}")
        print(f"         ({book})")
