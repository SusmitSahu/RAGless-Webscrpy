"""
Vector-less RAG Web Scraper
============================
Extracts structured fields from any product page by sending the pruned
live DOM directly to Claude — no vector DB, no embeddings, no index.

Pipeline:
  1. Fetch page
  2. Prune DOM (remove noise)
  3. Send pruned DOM to Claude
  4. Apply returned selector

Requirements:
  pip install anthropic beautifulsoup4 requests
"""

import re
import requests
from bs4 import BeautifulSoup
import anthropic

# ── Config ─────────────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_DOM_CHARS = 6000   # chars sent to LLM after pruning

claude = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DOM Pruner
# The smarter this is, the better the LLM performs.
# ══════════════════════════════════════════════════════════════════════════════

REMOVE_TAGS = {"script", "style", "noscript", "head", "nav",
               "footer", "header", "aside", "svg", "iframe",
               "img", "picture", "link", "meta"}
KEEP_ATTRS  = {"id", "class", "itemprop", "data-price", "content", "href"}


def prune_dom(raw_html: str, max_chars: int = MAX_DOM_CHARS) -> str:
    """
    Remove noisy tags, strip irrelevant attributes,
    collapse whitespace, and trim to max_chars.
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove noisy subtrees entirely
    for tag in soup(list(REMOVE_TAGS)):
        tag.decompose()

    # Strip attributes that don't help with selector generation
    for tag in soup.find_all(True):
        tag.attrs = {k: v for k, v in tag.attrs.items() if k in KEEP_ATTRS}

    # Collapse blank lines
    text = soup.prettify()
    text = re.sub(r'\n\s*\n', '\n', text)

    return text[:max_chars]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Generate selector directly from DOM
# ══════════════════════════════════════════════════════════════════════════════

def generate_selector(pruned_dom: str, field: str) -> str:
    prompt = f"""You are a CSS selector expert for web scraping.

Below is a cleaned HTML DOM. Generate ONE CSS selector that reliably
extracts the '{field}' from this page.

Rules:
- Prefer class or id selectors over positional ones (nth-child)
- If you see itemprop="{field}" or itemprop="name"/"price", prefer those
- Return ONLY the CSS selector — no explanation, no backticks, nothing else

DOM:
{pruned_dom}"""

    response = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=80,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip().strip("`")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Apply selector
# ══════════════════════════════════════════════════════════════════════════════

def apply_selector(raw_html: str, selector: str) -> str | None:
    soup = BeautifulSoup(raw_html, "html.parser")
    el   = soup.select_one(selector)
    if el is None:
        return None
    return el.get("content") or el.get_text(strip=True) or None


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def extract_vectorless(url: str, field: str, verbose: bool = True) -> dict:
    """
    Full vector-less pipeline: fetch → prune → generate → apply.

    Returns:
        {
            "url": ...,
            "field": ...,
            "selector": "p.price_color",
            "value": "£51.77",
            "dom_chars": 3241,
        }
    """
    if verbose:
        print(f"\n{'═'*55}")
        print(f"  Extracting '{field}' from: {url}")
        print(f"{'═'*55}")

    resp     = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    raw_html = resp.text

    pruned = prune_dom(raw_html)
    if verbose:
        print(f"[prune]    DOM reduced to {len(pruned)} chars")

    selector = generate_selector(pruned, field)
    if verbose:
        print(f"[generate] Selector → {selector}")

    value = apply_selector(raw_html, selector)
    if verbose:
        status = "✓" if value else "✗ not found"
        print(f"[apply]    Value    → {value or status}")

    return {
        "url":       url,
        "field":     field,
        "selector":  selector,
        "value":     value,
        "dom_chars": len(pruned),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    TEST_URLS = [
        "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
        "https://books.toscrape.com/catalogue/tipping-the-velvet_999/index.html",
        "https://books.toscrape.com/catalogue/soumission_998/index.html",
    ]

    results = []
    for url in TEST_URLS:
        for field in ["title", "price"]:
            result = extract_vectorless(url, field)
            results.append(result)

    print(f"\n{'═'*55}")
    print("  RESULTS SUMMARY")
    print(f"{'═'*55}")
    for r in results:
        book = r["url"].split("/")[-2]
        print(f"  {r['field']:6s}  {r['selector']:25s}  →  {r['value']}")
        print(f"         ({book})")
