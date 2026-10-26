"""
p2_taxonomy/data_extraction_patent_cpc.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Extract 4-character CPC subclass codes with titles and descriptions,
filtered to codes that actually appear in the patent corpus.

Sources:
- CPC title text files  (``CPCTitleList202601/cpc-section-*.txt``)
- CPC scheme XML files  (``CPCSchemeXML202601/cpc-scheme-{CODE}.xml``)
- Patent parquet parts   (``_corpus_cache/{mode}/part_*.parquet``)
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from tqdm import tqdm

from shared.patent_cache import CORPUS_ROOT, _list_part_files, _iter_parts


# ------------------------------------------------------------------
# Default paths
# ------------------------------------------------------------------

_CPC_DATA = Path(
    r"C:\Users\sidha\Dropbox\Work\UNU-MERIT\ST4TE\Task2_1\data\taxonomy\patent_cpc"
)
_TITLE_DIR = _CPC_DATA / "CPCTitleList202601"
_XML_DIR = _CPC_DATA / "CPCSchemeXML202601"


# ------------------------------------------------------------------
# 1) Parse title files  →  {code: full_title_text}
# ------------------------------------------------------------------

def _parse_title_files(title_dir: Path = _TITLE_DIR) -> Dict[str, str]:
    """Parse all ``cpc-section-*_*.txt`` files and return 4-char subclass titles.

    Returns
    -------
    dict[str, str]
        Mapping from 4-character CPC code to full title string.
    """
    titles: Dict[str, str] = {}
    for txt_file in sorted(title_dir.glob("cpc-section-*_*.txt")):
        with open(txt_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                code = parts[0].strip()
                # Keep only 4-character subclass codes (no "/" → not a group)
                if len(code) != 4 or "/" in code:
                    continue
                # Format: CODE \t \t TITLE  (two tabs, middle is empty)
                title_text = parts[-1].strip() if len(parts) >= 2 else ""
                if title_text:
                    titles[code] = title_text
    return titles


# ------------------------------------------------------------------
# 2) Parse XML scheme descriptions  →  {code: {"title": ..., "notes": ...}}
# ------------------------------------------------------------------

def _element_text(el: ET.Element) -> str:
    """Recursively extract plain text from an XML element, stripping all tags."""
    parts: List[str] = []
    if el.text:
        parts.append(el.text)
    for child in el:
        parts.append(_element_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _parse_xml_descriptions(
    xml_dir: Path = _XML_DIR,
    codes: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """Extract descriptive notes from CPC scheme XML files.

    Parameters
    ----------
    xml_dir : Path
        Directory containing ``cpc-scheme-{CODE}.xml`` files.
    codes : set[str] or None
        If given, only parse XML files for these 4-char codes.

    Returns
    -------
    dict[str, str]
        Mapping from 4-char CPC code to concatenated note text
        (empty string if no notes found).
    """
    descriptions: Dict[str, str] = {}
    target_codes = codes if codes is not None else {
        p.stem.replace("cpc-scheme-", "")
        for p in xml_dir.glob("cpc-scheme-????.xml")
    }

    for code in sorted(target_codes):
        xml_path = xml_dir / f"cpc-scheme-{code}.xml"
        if not xml_path.exists():
            descriptions[code] = ""
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            descriptions[code] = ""
            continue

        # Find notes-and-warnings at the top-level classification-item
        note_texts: List[str] = []
        for nw in root.iter("notes-and-warnings"):
            for note in nw.findall("note"):
                if note.get("type") != "note":
                    continue
                for para in note.findall("note-paragraph"):
                    text = _element_text(para).strip()
                    # Collapse whitespace
                    text = re.sub(r"\s+", " ", text)
                    if text:
                        note_texts.append(text)
            # Only use the first notes-and-warnings block (top-level)
            break

        descriptions[code] = " ".join(note_texts)

    return descriptions


# ------------------------------------------------------------------
# 3) Collect unique 4-char CPC codes from patent corpus
# ------------------------------------------------------------------

def _collect_patent_cpcs(mode: str, corpus_root: str = CORPUS_ROOT) -> Set[str]:
    """Stream parquet parts and collect unique 4-char CPC codes.

    Parameters
    ----------
    mode : str
        ``"digital"`` or ``"green"``.
    corpus_root : str
        Root of the corpus cache.

    Returns
    -------
    set[str]
        Unique 4-character CPC codes appearing in the corpus.
    """
    parts_dir = str(Path(corpus_root) / mode)
    parts = _list_part_files(parts_dir)

    all_codes: Set[str] = set()
    for df in tqdm(
        _iter_parts(parts, columns=["cpc_coarse_list"]),
        total=len(parts),
        desc=f"[cpc] scanning {mode} parts",
    ):
        for val in df["cpc_coarse_list"].dropna():
            if not isinstance(val, str) or not val.strip():
                continue
            for code in val.split("|"):
                code = code.strip()
                if len(code) == 4:
                    all_codes.add(code)
    return all_codes


# ------------------------------------------------------------------
# 4) Main orchestrator
# ------------------------------------------------------------------

def extract_cpc_tables(
    output_dir: Path = _CPC_DATA,
    modes: Optional[List[str]] = None,
    title_dir: Path = _TITLE_DIR,
    xml_dir: Path = _XML_DIR,
    corpus_root: str = CORPUS_ROOT,
) -> List[Path]:
    """Extract CPC 4-char code reference tables for the requested patent modes.

    Parameters
    ----------
    output_dir : Path
        Directory to write output CSVs.
    modes : list[str] or None
        Which patent modes to process (default: ``["digital", "green"]``).
    title_dir : Path
        Directory with CPC title text files.
    xml_dir : Path
        Directory with CPC scheme XML files.
    corpus_root : str
        Root of the patent corpus cache.

    Returns
    -------
    list[Path]
        Paths to the created CSV files.
    """
    if modes is None:
        modes = ["digital", "green"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse all titles from txt files (fast, ~14 MB total)
    print("[cpc] Parsing CPC title files …")
    all_titles = _parse_title_files(title_dir)
    print(f"[cpc] Found {len(all_titles)} 4-char subclass titles")

    output_paths: List[Path] = []

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"[cpc] Processing mode={mode}")
        print(f"{'='*50}")

        # Step 2: Collect unique CPC codes from patent corpus
        patent_codes = _collect_patent_cpcs(mode, corpus_root)
        print(f"[cpc] {mode}: {len(patent_codes)} unique 4-char CPC codes in patents")

        # Check for codes missing from title lookup
        missing = patent_codes - set(all_titles.keys())
        if missing:
            print(f"[cpc] WARNING: {len(missing)} patent CPC codes not in title files: "
                  f"{sorted(missing)[:20]}")

        # Step 3: Parse XML descriptions (only for codes we need)
        print(f"[cpc] Parsing XML descriptions for {len(patent_codes)} codes …")
        descriptions = _parse_xml_descriptions(xml_dir, patent_codes)
        n_with_desc = sum(1 for v in descriptions.values() if v)
        print(f"[cpc] {n_with_desc}/{len(patent_codes)} codes have XML descriptions")

        # Step 4: Build output DataFrame
        rows = []
        for code in sorted(patent_codes):
            title_full = all_titles.get(code, "")
            # Clean title: text before first "("
            paren_idx = title_full.find("(")
            title_clean = title_full[:paren_idx].strip() if paren_idx != -1 else title_full

            rows.append({
                "cpc_code": code,
                "title": title_clean,
                "title_full": title_full,
                "description": descriptions.get(code, ""),
            })

        df = pd.DataFrame(rows)
        out_path = output_dir / f"{mode}_cpc_4char.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[cpc] Wrote {len(df)} rows → {out_path}")
        output_paths.append(out_path)

    return output_paths
