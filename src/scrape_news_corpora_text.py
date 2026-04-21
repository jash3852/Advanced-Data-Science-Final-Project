from __future__ import annotations

import argparse
import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INPUT = DATA_DIR / "newsCorpora.csv"
DEFAULT_OUTPUT = DATA_DIR / "newsCorpora_with_text.csv"

CSV_COLUMNS = [
    "id",
    "headline",
    "url",
    "publisher",
    "category",
    "story_id",
    "hostname",
    "timestamp",
]

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

INVALID_TEXT_PATTERNS = (
    re.compile(r"\b404\b", re.IGNORECASE),
    re.compile(r"\baccess denied\b", re.IGNORECASE),
    re.compile(r"\bsubscription required\b", re.IGNORECASE),
    re.compile(r"\bforbidden\b", re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = (
            "Scrape article text for newsCorpora.csv, keep only accessible URLs, "
            "and write an enriched dataset."
        )
    )
    parser.add_argument(
        "--input",
        type = Path,
        default = DEFAULT_INPUT,
        help = "Path to the source newsCorpora.csv file.",
    )
    parser.add_argument(
        "--output",
        type = Path,
        default = DEFAULT_OUTPUT,
        help = "Path for the enriched output CSV file.",
    )
    parser.add_argument(
        "--timeout",
        type = float,
        default = 15.0,
        help = "Request timeout in seconds for each URL.",
    )
    parser.add_argument(
        "--workers",
        type = int,
        default = 8,
        help = "Number of concurrent URL fetches.",
    )
    return parser.parse_args()


def load_news_corpora(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    return pd.read_csv(
        csv_path,
        sep = "\t",
        header = None,
        names = CSV_COLUMNS,
        dtype = str,
        na_filter = False,
        engine = "python",
        on_bad_lines = "skip",
        quoting = csv.QUOTE_NONE,
        escapechar = "\\",
    )


def normalize_url(url: str) -> str:
    cleaned = url.strip()
    cleaned = cleaned.replace(r"\?", "?")
    cleaned = cleaned.replace(r"\&", "&")
    cleaned = cleaned.replace(r"\=", "=")
    cleaned = cleaned.replace(r"\/", "/")
    cleaned = cleaned.strip('"\'')
    return cleaned


def extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "iframe", "svg", "form"]):
        tag.decompose()

    candidates = []
    for selector in ("article", "main", "body"):
        node = soup.find(selector)
        if node is not None:
            text = " ".join(node.stripped_strings)
            if text:
                candidates.append(text)

    if not candidates:
        candidates.append(" ".join(soup.stripped_strings))

    best_text = max(candidates, key = len, default = "")
    best_text = re.sub(r"\s+", " ", best_text).strip()
    return best_text


def looks_valid(text: str) -> bool:
    if len(text.split()) < 40:
        return False

    return not any(pattern.search(text) for pattern in INVALID_TEXT_PATTERNS)


def fetch_text(url: str, timeout: float) -> Optional[str]:
    try:
        response = requests.get(url, headers = REQUEST_HEADERS, timeout = timeout)
        response.raise_for_status()
    except requests.RequestException:
        return None

    if not response.encoding:
        response.encoding = response.apparent_encoding or "utf-8"

    text = extract_article_text(response.text)
    if not text or not looks_valid(text):
        return None

    return text


def scrape_urls(urls: Iterable[str], timeout: float, workers: int) -> Dict[str, str]:
    results: Dict[str, str] = {}

    unique_urls = list(dict.fromkeys(urls))
    if not unique_urls:
        return results

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(fetch_text, url, timeout): url
            for url in unique_urls
        }
        with tqdm(total=len(futures), desc="Scraping URLs", unit="url") as pbar:
            for future in as_completed(futures):
                url = futures[future]
                text = future.result()
                if text:
                    results[url] = text
                pbar.update(1)

    return results


def build_enriched_dataset(df: pd.DataFrame, scraped_text: Dict[str, str]) -> pd.DataFrame:
    enriched = df.copy()
    enriched["url"] = enriched["url"].map(normalize_url)
    enriched = enriched[enriched["url"].astype(bool)]
    enriched = enriched[enriched["url"].isin(scraped_text)]
    enriched = enriched.copy()
    enriched["text_content"] = enriched["url"].map(scraped_text)
    enriched = enriched.dropna(subset=["text_content"])
    enriched = enriched[enriched["text_content"].astype(str).str.strip().astype(bool)]
    return enriched


def main() -> int:
    args = parse_args()

    print(f"Loading dataset from {args.input}...")
    news = load_news_corpora(args.input)
    print(f"Loaded {len(news):,} rows. Normalizing URLs...")
    news["url"] = news["url"].map(normalize_url)
    news = news[news["url"].astype(bool)]
    print(f"Processing {len(news):,} rows with valid URLs.\n")

    scraped_text = scrape_urls(news["url"], timeout=args.timeout, workers=args.workers)
    print(f"\nSuccessfully scraped {len(scraped_text):,} URLs.\n")
    
    print("Building enriched dataset...")
    enriched = build_enriched_dataset(news, scraped_text)

    print(f"Saving {len(enriched):,} rows to {args.output}...")
    args.output.parent.mkdir(parents = True, exist_ok = True)
    enriched.to_csv(args.output, index = False)

    print(
        f"✓ Saved {len(enriched):,} rows with scraped text to {args.output} "
        f"from {len(scraped_text):,} accessible URLs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())