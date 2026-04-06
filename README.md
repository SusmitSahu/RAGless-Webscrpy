# ScrapeShift 🔀

> **Intelligent web scraping that understands pages — not just patterns.**
> A side-by-side implementation of Traditional RAG vs Vector-less RAG for CSS selector generation using Claude.

---

## Why ScrapeShift?

Every production scraper eventually breaks. A site redesigns, an A/B test ships a new layout, and your hard-coded `div.price-block > span:nth-child(2)` returns nothing. The standard fix — update the selector manually — doesn't scale.

ScrapeShift explores two LLM-powered approaches to make selector generation *adaptive*:

| | Traditional RAG | Vector-less RAG |
|---|---|---|
| Infrastructure | ChromaDB + sentence-transformers | None |
| How it works | Retrieves similar past examples, few-shots the LLM | Sends pruned live DOM directly to the LLM |
| Cold-start | ❌ Fails on unseen domains | ✅ Works immediately |
| Staleness | ❌ Index goes out of date silently | ✅ Always sees the current page |
| Setup time | High | Minimal |
| Best for | Known sites with curated examples | Unknown/dynamic domains |

---

## What is Vector-less RAG?

Traditional RAG systems solve the "LLM doesn't know this specific thing" problem by retrieving relevant documents from a vector database and injecting them into the prompt. It works well for text — but breaks down for web scraping:

```
Traditional RAG Problem in Web Scraping
─────────────────────────────────────────

  New page arrives
       │
       ▼
  Embed HTML snippet ──────────────────────────────────────┐
       │                                                    │
       ▼                                                    ▼
  Vector DB (cosine search)                         Stored examples
       │                                         (possibly stale,
       │  retrieves "similar" past selectors      possibly wrong domain)
       ▼
  LLM: "Given THESE examples, generate a selector for THAT page"
       │
       ▼
  Problem: the retrieved examples may be from a completely
  different site structure. The LLM is reasoning from the wrong map.
```

**Vector-less RAG** is the insight that for this domain, *retrieval is the wrong abstraction*. Modern LLMs have large enough context windows to simply read the actual page — if you give them a clean enough version of it.

```
Vector-less RAG: The Core Idea
────────────────────────────────

  New page arrives
       │
       ▼
  Prune DOM ──────────────────────────────────────────────────┐
  (strip scripts, styles,                                     │
   nav, footer, irrelevant attrs)                             │
       │                                                      ▼
       │                                               Compact structural
       │                                               representation
       │                                               (6KB instead of 200KB)
       ▼
  LLM: "Here is the ACTUAL live DOM. Generate a selector for price."
       │
       ▼
  No retrieval. No index. No stale examples.
  The LLM reasons from ground truth.
```

The bottleneck shifts entirely to **DOM pruning quality**. The cleaner you can make the DOM before handing it to the LLM, the better the selector generation becomes.

---

## Pipeline Comparison

![Pipeline comparison: Traditional RAG vs Vector-less RAG](docs/comparison.svg)

| Stage | Traditional RAG | Vector-less RAG |
|---|---|---|
| Infrastructure | ChromaDB + sentence-transformers | None |
| Preprocessing | `preprocess_html()` → 2 000 chars | `prune_dom()` → 6 000 chars |
| Retrieval | Cosine search, top-k examples | — skipped entirely — |
| LLM context | Past (html → selector) examples | Live pruned DOM |
| Cold-start | Fails on unseen domains | Works immediately |
| Staleness | Index goes stale silently | Always sees current DOM |
| Accuracy driver | Quality of example bank | Quality of DOM pruning |

---

## Project Structure

```
scrapeshift/
├── src/
│   ├── __init__.py
│   ├── rag_scraper.py        # Full Traditional RAG pipeline
│   ├── vectorless_scraper.py # Vector-less pipeline
│   └── compare.py            # Side-by-side comparison runner
├── examples/
│   └── quickstart.py         # Minimal working example
├── tests/
│   └── test_preprocessing.py # Unit tests (no API/network needed)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/your-username/scrapeshift.git
cd scrapeshift
```

### 2. Install

```bash
# Vector-less only (minimal)
pip install anthropic beautifulsoup4 requests

# Both methods
pip install -r requirements.txt
```

### 3. Set your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 4. Run

```bash
# Quickstart example (both methods, one URL)
python examples/quickstart.py

# Traditional RAG only
python src/rag_scraper.py

# Vector-less only
python src/vectorless_scraper.py

# Side-by-side comparison table
python src/compare.py
```

---

## Usage as a Library

### Vector-less (no setup)

```python
from src.vectorless_scraper import extract_vectorless

result = extract_vectorless(
    url="https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
    field="price"
)

print(result["value"])     # "£51.77"
print(result["selector"])  # "p.price_color"
print(result["dom_chars"]) # 3241
```

### Traditional RAG

```python
from src.rag_scraper import index_examples, extract_with_rag, EXAMPLE_BANK

# Index once at startup
index_examples(EXAMPLE_BANK)

result = extract_with_rag(
    url="https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
    field="title"
)

print(result["value"])               # "A Light in the Attic"
print(result["selector"])            # "h1"
print(result["retrieved_examples"])  # list of 3 similar past examples with scores
```

### Adding your own examples (RAG)

```python
from src.rag_scraper import EXAMPLE_BANK, index_examples

EXAMPLE_BANK.append({
    "field":    "price",
    "html":     '<div class="my-store-price">$99.00</div>',
    "selector": ".my-store-price",
    "note":     "custom store price container"
})

index_examples(EXAMPLE_BANK, force_reindex=True)
```

---

## Test

```bash
# No API key or network needed
python tests/test_preprocessing.py

# With pytest
pip install pytest
pytest tests/
```

---

## Key Design Decisions

### Why `sentence-transformers` for RAG embeddings?

It runs locally, is free, and produces good semantic similarity for HTML snippets. You can swap it for the Anthropic embeddings API if you prefer fewer dependencies.

### Why ChromaDB in-memory mode?

Zero config for experimentation. Switch to `chromadb.PersistentClient(path="./db")` to persist the index across runs — critical for production use.

### Why `books.toscrape.com` as the test target?

It's a real, public, auth-free site built specifically for scraper testing. It never changes layout, so results are reproducible. No rate limits, no CAPTCHAs.

### The pruning budget

Vector-less sends up to 6,000 chars of pruned DOM; RAG sends 2,000 chars (smaller because the retrieved examples already provide context). Both numbers are tunable via `MAX_DOM_CHARS` / `MAX_SNIPPET`.

---

## When to Use Which

**Use Traditional RAG when:**
- You scrape a fixed, known set of sites repeatedly
- You have (or can build) a labelled example library
- Pages are too large or too noisy even after pruning — examples guide the LLM toward the right DOM region
- You want few-shot learning from manually validated selectors

**Use Vector-less RAG when:**
- You need to scrape unknown or first-seen domains immediately
- You want zero infrastructure (no DB, no embedder)
- Sites redesign frequently (index would go stale)
- You're prototyping and need something working in 10 minutes

---

## License

MIT
