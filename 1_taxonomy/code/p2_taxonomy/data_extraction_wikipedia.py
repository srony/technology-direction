"""
p2_taxonomy/data_extraction.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Wikipedia crawling, seed loading, and taxonomy construction.

Provides:
- ``load_seed_categories`` / ``extract_topic_from_filename`` -- seed helpers
- ``WikiSummaryFetcher``  -- REST-API summary fetcher (01_cleaning version)
- ``WikiLinkExtractor``   -- HTML parser for internal wiki links
- ``wiki_get_category_members`` -- MediaWiki API category members
- ``crawl_from_seeds``    -- BFS category crawler
- ``build_taxonomy``      -- DFS taxonomy tree constructor
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import time
import urllib.parse
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ------------------------------------------------------------------
# Config defaults
# ------------------------------------------------------------------

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_MAX_DEPTH = 4
DEFAULT_MAX_CATEGORIES = 100_000
REQUEST_SLEEP = 0.1

# Used by WikiLinkExtractor -- strip non-body noise from parsed HTML
NON_BODY_SELECTORS = [
    "table.infobox",
    "table.sidebar",
    "div.navbox",
    "div.vertical-navbox",
    "div.reflist",
    "ol.references",
    "div.thumb",
    "div.metadata",
    "div.mw-references-wrap",
    "span.mw-editsection",
    "div.hatnote",
    "div.shortdescription",
    "div.ambox",
    "table.ambox",
]

EXCLUDE_SECTIONS = {
    "see also", "references", "external links", "further reading", "notes",
}


# ------------------------------------------------------------------
# Seed loading + topic extraction
# ------------------------------------------------------------------

def load_seed_categories(path: Path) -> list[str]:
    """Load seed category names from a text file (one per line, ``#`` comments)."""
    seeds: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            seeds.append(line)
    return list(dict.fromkeys(seeds))  # dedupe, preserve order


def extract_topic_from_filename(path: Path) -> str:
    """Extract ``<topic>`` from a filename shaped ``<topic>_seeds.txt``."""
    base = Path(path).name
    if not base.endswith("_seeds.txt"):
        raise ValueError(
            f"Seed filename must follow the pattern <topic>_seeds.txt, got: {base}"
        )
    return base.replace("_seeds.txt", "")


# ------------------------------------------------------------------
# Wikipedia API helpers
# ------------------------------------------------------------------

def wiki_get_category_members(category_title: str) -> list[dict[str, Any]]:
    """Return all members (subcats + pages) of a Wikipedia category."""
    session = requests.Session()
    members: list[dict[str, Any]] = []
    cmcontinue: Optional[str] = None

    while True:
        params: dict[str, str] = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmtype": "subcat|page",
            "cmlimit": "max",
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        resp = session.get(
            WIKI_API_URL,
            params=params,
            headers={"User-Agent": "WikiTaxonomy (srony@356576@gmail.com)"},
        )
        if resp.status_code != 200:
            break

        data = resp.json()
        members.extend(data.get("query", {}).get("categorymembers", []))

        if "continue" in data and "cmcontinue" in data["continue"]:
            cmcontinue = data["continue"]["cmcontinue"]
            time.sleep(REQUEST_SLEEP)
        else:
            break

    return members


# ------------------------------------------------------------------
# WikiSummaryFetcher  (01_cleaning.py version -- richer constructor)
# ------------------------------------------------------------------

class WikiSummaryFetcher:
    """Fetch Wikipedia page summaries via the REST API with caching.

    This is the enhanced version from ``01_cleaning.py`` which exposes
    ``request_sleep``, ``user_agent`` and ``timeout_s`` constructor params.
    """

    SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

    def __init__(
        self,
        cache_path: str = "wiki_description_cache.csv",
        request_sleep: float = 0.1,
        user_agent: str = "TaxonomyBot (srony356576@gmail.com)",
        timeout_s: float = 30.0,
    ):
        self.cache: Dict[str, str] = {}
        self.cache_path = cache_path
        self._dirty = False
        self.request_sleep = float(request_sleep)
        self.user_agent = user_agent
        self.timeout_s = float(timeout_s)

        # Try loading persistent cache from disk
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        title = (row.get("title") or "").strip()
                        desc = row.get("description") or ""
                        if title:
                            self.cache[title] = desc
                print(
                    f"[WikiSummaryFetcher] Loaded {len(self.cache)} cached "
                    f"descriptions from {self.cache_path}"
                )
        except Exception as exc:
            print(f"[WikiSummaryFetcher] Failed to load cache from {self.cache_path}: {exc}")

    # ---- public API ----

    def get_summary(self, title: str) -> str:
        """Return a short summary for *title*; uses cache first."""
        if not title:
            return ""
        key = title.strip()

        if key in self.cache:
            return self.cache[key]

        encoded = urllib.parse.quote(title.replace(" ", "_"))
        url = self.SUMMARY_API.format(encoded)

        try:
            resp = requests.get(
                url,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout_s,
            )
            if resp.status_code != 200:
                summary = ""
            else:
                data = resp.json()
                summary = data.get("extract") or data.get("description") or ""
        except Exception:
            summary = ""

        self.cache[key] = summary
        self._dirty = True

        if self.request_sleep > 0:
            time.sleep(self.request_sleep)
        return summary

    def save_cache(self) -> None:
        """Persist dirty cache to disk."""
        if not self._dirty:
            return
        try:
            with open(self.cache_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["title", "description"])
                writer.writeheader()
                for title, desc in self.cache.items():
                    writer.writerow({"title": title, "description": desc})
            print(
                f"[WikiSummaryFetcher] Saved {len(self.cache)} descriptions "
                f"to {self.cache_path}"
            )
            self._dirty = False
        except Exception as exc:
            print(f"[WikiSummaryFetcher] Failed to save cache: {exc}")


# ------------------------------------------------------------------
# WikiLinkExtractor  (from 01_cleaning.py)
# ------------------------------------------------------------------

class WikiLinkExtractor:
    """Extract internal wiki links from the main body of a page via ``action=parse``."""

    def __init__(
        self,
        language: str = "en",
        user_agent: str = "wiki-link-extractor/1.0 (research use)",
        sleep_s: float = 0.0,
        timeout_s: float = 30.0,
    ):
        self.language = language
        self.sleep_s = float(sleep_s)
        self.timeout_s = float(timeout_s)

        self.api = f"https://{language}.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

        self.cache: Dict[str, List[str]] = {}

    # ---- internal ----

    def _get_json(self, params: Dict[str, Any]) -> Dict[str, Any]:
        max_retries = 8
        base_sleep = self.sleep_s if self.sleep_s > 0 else 0.2

        for attempt in range(max_retries):
            if base_sleep > 0:
                time.sleep(base_sleep)
            try:
                r = self.session.get(self.api, params=params, timeout=self.timeout_s)

                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After")
                    wait = (
                        float(retry_after)
                        if retry_after is not None
                        else min(60.0, (2 ** attempt) * 1.0) + random.uniform(0, 1)
                    )
                    tqdm.write(f"[429] Rate limited. Sleeping {wait:.1f}s ...")
                    time.sleep(wait)
                    continue

                if r.status_code in (500, 502, 503, 504):
                    wait = min(60.0, (2 ** attempt) * 1.0) + random.uniform(0, 1)
                    tqdm.write(f"[{r.status_code}] Server error. Sleeping {wait:.1f}s ...")
                    time.sleep(wait)
                    continue

                if r.status_code != 200:
                    return {}

                ct = (r.headers.get("Content-Type") or "").lower()
                if "json" not in ct:
                    return {}
                try:
                    return r.json()
                except Exception:
                    return {}

            except requests.RequestException as exc:
                wait = min(60.0, (2 ** attempt) * 1.0) + random.uniform(0, 1)
                tqdm.write(f"[WARN] Request error: {exc}. Sleeping {wait:.1f}s ...")
                time.sleep(wait)

        raise RuntimeError("Failed after repeated retries (rate limit or network errors).")

    # ---- public API ----

    def fetch_main_body_link_titles(
        self,
        page_title: str,
        include_lists: bool = False,
    ) -> list[str]:
        """Return unique article titles linked from the main body of *page_title*."""
        if not isinstance(page_title, str) or not page_title.strip():
            return []

        title = page_title.strip()
        cache_key = f"{title}||include_lists={int(include_lists)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        params = {
            "action": "parse",
            "format": "json",
            "page": title,
            "prop": "text",
            "redirects": 1,
            "disableeditsection": 1,
            "formatversion": 2,
        }

        data = self._get_json(params)
        html = (data.get("parse") or {}).get("text")
        if not isinstance(html, str) or not html:
            self.cache[cache_key] = []
            return []

        soup = BeautifulSoup(html, "html.parser")
        parser_output = soup.select_one("div.mw-parser-output")
        if not parser_output:
            self.cache[cache_key] = []
            return []

        for sel in NON_BODY_SELECTORS:
            for tag in parser_output.select(sel):
                tag.decompose()

        extracted: list[str] = []
        exclude_prefixes = (
            "File:", "Special:", "Category:", "Help:",
            "Wikipedia:", "Portal:", "Talk:", "Template:",
        )
        allowed_block_tags = ["p"] + (["ul", "ol"] if include_lists else [])

        for element in parser_output.children:
            if not getattr(element, "name", None):
                continue
            if element.name in ("h2", "h3"):
                headline = element.find("span", class_="mw-headline")
                if headline:
                    htxt = headline.get_text(" ", strip=True).lower()
                    if htxt in EXCLUDE_SECTIONS:
                        break

            if element.name in allowed_block_tags:
                for link in element.find_all("a", href=True):
                    href = link.get("href") or ""
                    link_title = link.get("title")
                    if not link_title:
                        continue
                    if not href.startswith("/wiki/"):
                        continue
                    if any(link_title.startswith(p) for p in exclude_prefixes):
                        continue
                    if "ISBN" in link_title:
                        continue
                    extracted.append(link_title)

        # Dedupe preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for lnk in extracted:
            if lnk not in seen:
                unique.append(lnk)
                seen.add(lnk)

        self.cache[cache_key] = unique
        return unique


# ------------------------------------------------------------------
# BFS crawler
# ------------------------------------------------------------------

def crawl_from_seeds(
    seed_categories: list[str],
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """BFS-crawl Wikipedia categories and return ``(cat_children, cat_pages)``."""
    cat_children: dict[str, list[str]] = defaultdict(list)
    cat_pages: dict[str, list[str]] = defaultdict(list)

    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque()

    for seed in seed_categories:
        visited.add(seed)
        queue.append((seed, 0))

    processed = 0
    pbar = tqdm(total=max_categories, desc="Crawling categories", unit="cat")

    while queue and processed < max_categories:
        cat, depth = queue.popleft()
        if depth > max_depth:
            continue

        processed += 1
        pbar.update(1)

        members = wiki_get_category_members(cat)
        for member in members:
            ns = member.get("ns")
            title = member.get("title")
            if ns == 14:  # subcategory
                if title not in visited:
                    visited.add(title)
                    queue.append((title, depth + 1))
                cat_children[cat].append(title)
            elif ns == 0:  # article page
                cat_pages[cat].append(title)

        time.sleep(REQUEST_SLEEP)

    pbar.close()
    return cat_children, cat_pages


# ------------------------------------------------------------------
# DFS taxonomy construction
# ------------------------------------------------------------------

def _dfs_collect_pages(
    cat: str,
    root_category: str,
    root_label: str,
    cat_children: dict[str, list[str]],
    cat_pages: dict[str, list[str]],
    visited: set[str],
    rows: list[dict[str, str]],
    cat_path: list[str],
) -> None:
    if cat in visited:
        return
    visited.add(cat)

    for page in cat_pages.get(cat, []):
        hierarchy_list = [root_label] + cat_path + [page]
        rows.append({
            "root_technology": root_label,
            "root_category": root_category,
            "page_title": page,
            "category_above_page": cat,
            "category_path": " > ".join(cat_path),
            "depth_from_root_category": len(cat_path) - 1,
            "hierarchy_path": " > ".join(hierarchy_list),
        })

    for child in cat_children.get(cat, []):
        _dfs_collect_pages(
            child, root_category, root_label,
            cat_children, cat_pages, visited, rows,
            cat_path + [child],
        )


def build_taxonomy(
    seed_categories: list[str],
    cat_children: dict[str, list[str]],
    cat_pages: dict[str, list[str]],
) -> list[dict[str, str]]:
    """Build a flat list of taxonomy rows via DFS from each seed."""
    rows: list[dict[str, str]] = []
    for root_cat in tqdm(seed_categories, desc="Building taxonomy per root"):
        root_label = root_cat.replace("Category:", "", 1).strip()
        visited: set[str] = set()
        _dfs_collect_pages(
            root_cat, root_cat, root_label,
            cat_children, cat_pages, visited, rows,
            [root_cat],
        )
    return rows
