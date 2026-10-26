"""
p2_taxonomy/cleaning.py
~~~~~~~~~~~~~~~~~~~~~~~
Taxonomy cleaning pipeline specific to the P2 sub-project.

Re-uses shared helpers (``flag_noise_pages``, ``compile_any``,
``relevance_score``, ``print_stats``) from ``shared.cleaning`` and defines
the P2-specific pattern lists and pipeline functions.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from shared.cleaning import (
    compile_any,
    flag_noise_pages,
    print_stats,
    relevance_score,
)
from shared.utils import (
    checkpoint_path,
    load_checkpoint_if_any,
    write_checkpoint,
)

# ------------------------------------------------------------------
# P2-specific pattern lists (from 01_cleaning.py)
# ------------------------------------------------------------------

TITLE_PARENTHESES_BLOCK = [
    r"\(film\)", r"\(TV series\)", r"\(television series\)", r"\(novel\)",
    r"\(short story\)", r"\(song\)", r"\(album\)", r"\(video game\)",
    r"\(character\)", r"\(comics\)", r"\(band\)", r"\(musician\)",
    r"\(play\)", r"\(opera\)", r"\(musical\)", r"\(manga\)", r"\(anime\)",
    r"\(soundtrack\)", r"\(franchise\)", r"\(person\)", r"\(footballer\)",
    r"ISO",
    r"\(season \d+\)", r"\(disambiguation\)", r"\(website\)",
    r".inc\b", r"inc.\b", r".co\b", r"\.", r"\(surname\)",
]

GENERIC_BLACKLIST = [
    # Entertainment / fiction
    r"\bfilm\b", r"\bmovie\b", r"\btelevision series\b", r"\bTV series\b",
    r"\bnovel\b", r"\bsong\b", r"\balbum\b", r"\bvideo game\b",
    r"\bfictional character\b", r"\bdrama\b", r"\bromantic\b", r"\bcomedy\b",
    r"\bthriller\b", r"\bhorror\b", r"\bsci-fi\b", r"\bscience fiction\b",
    r"\bcomic book\b", r"\banime\b", r"\bmanga\b", r"\bcartoon\b",
    # People / sports
    r"\bfootballer\b", r"\bbasketball\b", r"\bbaseball\b", r"\bathlete\b",
    r"\bpolitician\b", r"\bactor\b", r"\bactress\b", r"\bsinger\b",
    r"\bpoet\b", r"\bwriter\b", r"\bmusician\b", r"\bartist\b", r"\bcoach\b",
    r"\bmanager\b", r"\bplayer\b", r"\bchampionship\b", r"\bleague\b",
    r"\bolympic\b", r"\btournament\b", r"\bbiography\b",
    # Places / geo
    r"\bcity in\b", r"\btown in\b", r"\bvillage in\b", r"\bmunicipality\b",
    r"\bcounty\b", r"\bprovince\b", r"\bdistrict\b", r"\bstate of\b",
    r"\bregion of\b", r"\bmountain\b", r"\briver\b", r"\blake\b", r"\bisland\b",
    r"\bairport\b", r"\brailway station\b", r"\bschool\b", r"\buniversity\b",
    # Humanities / history
    r"\bmythology\b", r"\breligion\b", r"\bphilosophy\b", r"\btheology\b",
    r"\bhistory of\b", r"\btimeline of\b", r"\bbattle of\b", r"\bwar\b",
    r"\bdynasty\b", r"\bempire\b", r"\bmonarchy\b",
    # Biology
    r"\bspecies\b", r"\bgenus\b", r"\bfamily\b", r"\border of\b",
    r"\bmammal\b", r"\breptile\b", r"\binsect\b", r"\bdinosaur\b",
    # Meta / structural
    r"^List of", r"^Category:", r"^Template:", r"^File:", r"^Draft:",
]

GREEN_KEYWORDS = [
    r"climate", r"emission", r"greenhouse", r"\bghg\b", r"carbon", r"decarbon",
    r"footprint", r"net-zero", r"neutrality",
    r"renewable", r"clean energy", r"solar", r"photovoltaic", r"\bpv\b",
    r"wind power", r"wind farm", r"turbine", r"hydroelectric", r"hydropower",
    r"geothermal", r"bioenergy", r"biofuel", r"biomass", r"biogas",
    r"tidal", r"wave energy", r"nuclear fusion", r"hydrogen", r"electrolyzer",
    r"efficien", r"sustainab", r"battery", r"lithium", r"solid-state",
    r"storage", r"grid", r"smart grid", r"microgrid", r"supercapacitor",
    r"waste", r"recycl", r"circular economy", r"remanufacturing", r"upcycl",
    r"biodegrad", r"compost", r"landfill gas",
    r"pollution", r"remediation", r"adaptation", r"mitigation", r"resilien",
    r"conservation", r"ecosystem services", r"reforestation", r"afforestation",
    r"water treatment", r"wastewater", r"desalination", r"sewage",
    r"air quality", r"particulate matter",
    r"electric vehicle", r"\bev\b", r"hybrid vehicle", r"mobility",
    r"public transport", r"modal shift", r"green building", r"retrofit",
    r"hvac", r"insulation", r"passive house", r"precision agriculture",
    r"ccs", r"carbon capture", r"sequestration",
]

DIGITAL_KEYWORDS = [
    r"comput", r"software", r"hardware", r"algorithm", r"programming",
    r"data", r"database", r"information tech", r"\bit\b", r"ict",
    r"network", r"protocol", r"internet", r"web", r"cloud", r"server",
    r"datacenter", r"distributed system", r"edge computing", r"5g", r"6g",
    r"broadband", r"fiber optic", r"satellite", r"wireless",
    r"artificial intelligence", r"\bai\b", r"machine learning", r"\bml\b",
    r"deep learning", r"neural network", r"nlp", r"computer vision",
    r"big data", r"analytics", r"data science", r"predictive",
    r"\biot\b", r"internet of things", r"sensor", r"actuator",
    r"robot", r"robotics", r"automation", r"cybernetic", r"mechatronics",
    r"digital twin", r"simulation", r"virtual reality", r"\bvr\b",
    r"augmented reality", r"\bar\b", r"mixed reality", r"\bxr\b",
    r"cybersecurity", r"encryption", r"cryptography", r"malware",
    r"firewall", r"blockchain", r"distributed ledger", r"smart contract",
    r"privacy",
    r"api", r"microservice", r"containeriz", r"docker", r"kubernetes",
    r"devops", r"saas", r"paas", r"iaas", r"platform", r"interface",
]

# Pre-compiled mega-patterns used by the pipeline
TITLE_PARENS_PAT = compile_any(TITLE_PARENTHESES_BLOCK)
GENERIC_BLACKLIST_PAT = compile_any(GENERIC_BLACKLIST)
GREEN_PAT = compile_any(GREEN_KEYWORDS)
DIGITAL_PAT = compile_any(DIGITAL_KEYWORDS)

# Description-dependent patterns
A_BIO_PAT = re.compile(
    r"\b("
    r"was an?|was the|is an?\s+(american|british|french|german|indian|italian|"
    r"japanese|chinese|canadian|australian)\b.*\b"
    r"(engineer|scientist|inventor|researcher|professor|entrepreneur|"
    r"businessman|businesswoman|politician|author|actor|actress|director|"
    r"musician|artist)"
    r"|born\b|died\b|biograph(y|ical)\b|he was\b|she was\b"
    r")\b",
    re.IGNORECASE,
)

A_ORG_PAT = re.compile(
    r"\b("
    r"is an?\s+(american|global|international)?\s*(company|corporation|"
    r"startup|organisation|organization|association|foundation|institute|"
    r"laboratory|lab|university|college|agency)"
    r"|headquartered\b|subsidiary\b|nonprofit\b|non-profit\b|trade association\b"
    r")\b",
    re.IGNORECASE,
)

A_MEDIA_PAT = re.compile(
    r"\b("
    r"is a(n)?\s+(film|movie|novel|book|short story|tv series|television series|"
    r"episode|comic|manga|anime|album|song|video game|board game|card game)"
    r"|fiction\b|in fiction\b|science fiction\b"
    r")\b",
    re.IGNORECASE,
)

A_EVENT_PAT = re.compile(
    r"\b("
    r"is a(n)?\s+(conference|workshop|symposium|summit|forum|festival|"
    r"competition|tournament|exhibition)"
    r"|held annually\b|held in\b|took place\b"
    r")\b",
    re.IGNORECASE,
)

A_MEME_PAT = re.compile(r"\b(meme|internet meme|viral)\b", re.IGNORECASE)

B_PATH_BAD_PAT = re.compile(
    r"\b("
    # Organizations / companies / industry
    r"companies?|company|vendors?|firms?|startups?|corporations?|enterprises?|"
    r"\bcity\b|providers|sanctuaries|sanctuary|"
    r"organizations?|organisations?|associations?|foundations?|institutes?|"
    r"laboratories?|labs?|manufacturers?|park|museums?|gardens?|"
    # Education / institutions
    r"universities|university|colleges?|schools?|archaeological|"
    # People / biographies / roles
    r"people|persons|biograph(y|ies)|biographies|biography|born|died|births|deaths|"
    r"scientists?|engineers?|inventors?|researchers?|authors?|actors?|actresses?|"
    r"musicians?|artists?|politicians?|theorists?|specialists?|zookeepers|animals?|dogs|"
    r"\w*ists\b|illustrator?|illustrators?|genealog\w*|\w*icians?\b|creators?|"
    r"influencers?|attractions?"
    # Events / competitions
    r"conference|conferences|workshop|workshops|symposium|symposia|symposi(a|um)?|"
    r"summit|summits|forum|forums|festival|festivals|events|parties|"
    r"tournament|tournaments|competition|competitions|championship|championships?|"
    r"league|olympic|sightings|"
    # Media / entertainment / pop culture
    r"fictions?|films?|movies?|books|ebooks?|fictional|websites?|"
    r"television series|television franchises?|television channels?|streaming|"
    r"television shows?|"
    r"comics?|webcomics?|anime|manga|\bcolors?\b|\bart\b|\bmusic\b|songs|letters|"
    r"games\b|memes?|meme|viral|fan|braille|animatronic attractions|critics?|video game|"
    # Lists / meta / classification
    r"list of|lists of|list|lists|related lists|"
    r"overview|introduction|timeline|history|survey|studies|research|"
    r"medical classification|atc codes|journals?|redirects|"
    r"by (city|cities|country|countries|brand|date|continent|origin|fuel|"
    r"method of locomotion)|"
    r"by country|by countries|by brand|by date|by continent|by origin|made in|"
    r"domain names|"
    r"smartphones by operating system|discontinued smartphones?|"
    # Business / market / careers
    r"market|industry|economics|pricing|roi|jobs?|careers?|salary|"
    r"crypto ponzi schemes|scams?|theft|"
    # Education / training
    r"training|course|courses?|certification|certifications?|degree|curriculum|tutorials?|"
    # Governance / policy / legal
    r"strategy|governance|management|policy|regulation|regulations?|ethics|legal|"
    r"censor|censuses|"
    r"standards?|compliance|framework|best practices|\brisk\b|\blaws?\b|safety|advocates|"
    # Application domains / health / govt
    r"business|marketing|finance|healthcare|medicine|education|government|retail|"
    r"manufacturing|fever|"
    # Military / conflict
    r"air force|weapons|\bwar\b|drone strikes|missiles|military radars|"
    # Products / tools / platforms (vendor framing)
    r"products?|services?|solutions?|platforms?|tools?|suite|suites?|"
    # Adoption / implementation / use
    r"applications?|use cases?|case studies|deployment|implementation|adoption|"
    # Transport / vehicles / motorsport
    r"formula one cars|racing series|vehicle series|drivers|"
    r"(light rail|metro|monorail|rail|transportation)\s+stations?|"
    r"radar stations|electrification|"
    # Broadcasting / radio / other station-types
    r"(television|radio|broadcasting|satellite radio|listening|surveillance|"
    r"meteorological|weather|ground|reference|charging|docking)\s+stations?|"
    r"tram stubs|geographic data and information|\bin\b|"
    # Misc slices / physical descriptors
    r"sizes in clothing|body shape|driving licences|human size|"
    r"pioneers?|line of|facial expressions|"
    # Hard-coded non-technology entities
    r"splunk|internet archive|library of congress classification|"
    r"nag hammadi library|medinet madi library|"
    r"historic newspaper digitization projects|commercial digital libraries|"
    r"low-emission zones|afroinsectiphilia|afroinsectivora|afrosoricida|agamidae"
    r")\b",
    re.IGNORECASE,
)

# Noise patterns also used by the matching/clustering scripts
NOISE_TITLE_PATTERNS = [
    r"\bdisambiguation\b",
    r"\bmay refer to\b",
    r"^list of\b",
    r"^outline of\b",
    r"^timeline of\b",
    r"^glossary of\b",
    r"^index of\b",
    r"^comparison of\b",
    r"^history of\b",
    r"^portal:",
    r"^wikipedia:",
    r"^wikimedia",
    r"\bchristmas\b",
]

MORE_BAD_PATH_PATTERNS = [r"\bculture\b"]


# ------------------------------------------------------------------
# Backoff helper (from 01_cleaning.py)
# ------------------------------------------------------------------

MIN_SLEEP = 0.1
MAX_SLEEP = 2.0
BACKOFF_MULT = 1.6
RECOVER_MULT = 0.97
JITTER = 0.05

CHECKPOINT_EVERY = 50


@dataclass
class Backoff:
    """Adaptive sleep timer with exponential back-off on failure."""

    sleep_s: float

    def on_success(self) -> None:
        self.sleep_s = max(MIN_SLEEP, self.sleep_s * RECOVER_MULT)

    def on_failure(self) -> None:
        self.sleep_s = min(MAX_SLEEP, self.sleep_s * BACKOFF_MULT)

    def sleep(self) -> None:
        time.sleep(max(0.0, self.sleep_s + random.uniform(-JITTER, JITTER)))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def category_links_from_hierarchy(hierarchy_path: str, page_title: str) -> str:
    """Extract the list of category links from a hierarchy path string.

    Returns a JSON array of category names (excluding the page itself).
    """
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).casefold()

    pt = norm(page_title)
    parts: List[str] = []
    seen: set[str] = set()
    for raw in (hierarchy_path or "").split(">"):
        item = raw.strip()
        if not item:
            continue
        if norm(item) == pt:
            continue
        if item not in seen:
            seen.add(item)
            parts.append(item)
    return json.dumps(parts, ensure_ascii=False)


def compute_root_similarity(df: pd.DataFrame, model_name: str) -> pd.Series:
    """Compute cosine similarity between each page title and its root technology."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")

    model = SentenceTransformer(model_name)
    roots = df["root_technology"].fillna("").astype(str).tolist()
    texts = df["page_title"].fillna("").astype(str).tolist()

    batch = 64
    root_vecs: list[np.ndarray] = []
    text_vecs: list[np.ndarray] = []
    for i in range(0, len(df), batch):
        root_vecs.append(model.encode(roots[i : i + batch], normalize_embeddings=True))
        text_vecs.append(model.encode(texts[i : i + batch], normalize_embeddings=True))

    rv = np.vstack(root_vecs) if root_vecs else np.zeros((0, 1))
    tv = np.vstack(text_vecs) if text_vecs else np.zeros((0, 1))
    sims = (rv * tv).sum(axis=1)
    return pd.Series(sims, index=df.index, dtype="float64")


# ------------------------------------------------------------------
# Cleaning pipeline (mirrors the staged approach of 01_cleaning.py)
# ------------------------------------------------------------------

def run_cleaning_pipeline(
    input_csv: Path,
    output_csv: Path,
    mode: str = "green",
    include_lists: bool = False,
    sleep: float = 0.1,
    timeout: float = 30.0,
    use_embeddings: bool = False,
    embed_model: str = "AI-Growth-Lab/PatentSBERTa",
    description_cache: str = "wiki_description_cache.csv",
) -> int:
    """Run the full taxonomy cleaning pipeline.

    This mirrors the staged logic from ``01_cleaning.py``:
      1. Title/path pattern filters (no description needed)
      2. Fetch descriptions via Wikipedia REST API
      3. Compute relevance score using title + path + description
      4. Apply description-dependent filters (bio/org/media/event/meme)
      5. Fetch wiki links with adaptive backoff
      6. Write final output

    Returns 0 on success.
    """
    from p2_taxonomy.data_extraction_wikipedia import WikiLinkExtractor, WikiSummaryFetcher

    df_in = pd.read_csv(input_csv)
    prev = print_stats(df_in, "input_loaded")

    required = ["root_technology", "page_title", "hierarchy_path"]
    missing = [c for c in required if c not in df_in.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df_in[required].copy()
    prev = print_stats(df, "after_column_select", prev)

    df["page_title"] = df["page_title"].astype(str)
    df["root_technology"] = df["root_technology"].fillna("").astype(str)
    df["hierarchy_path"] = df["hierarchy_path"].fillna("").astype(str)

    # --- Stage 1: title/path pattern filters ---
    blob_title = df["page_title"].astype(str)
    blob_path = df["hierarchy_path"].astype(str)

    bad_title_parens = blob_title.str.contains(TITLE_PARENS_PAT)
    bad_generic = blob_title.str.contains(GENERIC_BLACKLIST_PAT)
    bad_path = blob_path.str.contains(B_PATH_BAD_PAT)

    keep_s1 = (~bad_title_parens) & (~bad_generic) & (~bad_path)
    df = df.loc[keep_s1].copy()
    prev = print_stats(df, "after_filters_stage1", prev)

    df = df.dropna(subset=["page_title"]).drop_duplicates(
        subset=["page_title"], keep="first"
    ).copy()
    prev = print_stats(df, "after_dedup_page_title", prev)

    # Build output frame
    df_out = pd.DataFrame({
        "root_technology": df["root_technology"],
        "page_title": df["page_title"],
        "hierarchy_path": df["hierarchy_path"],
        "description": "",
        "category_links": [
            category_links_from_hierarchy(h, t)
            for h, t in zip(df["hierarchy_path"], df["page_title"])
        ],
        "relevance_score_regex": np.nan,
        "root_similarity": np.nan,
        "linked_wiki_titles_json": "",
    })

    # Resume from checkpoint
    resume_df = load_checkpoint_if_any(output_csv)
    if resume_df is not None and {"page_title"}.issubset(resume_df.columns):
        for field in [
            "linked_wiki_titles_json", "root_similarity",
            "category_links", "description", "relevance_score_regex",
        ]:
            if field in resume_df.columns:
                tmp = resume_df[["page_title", field]].copy()
                tmp["page_title"] = tmp["page_title"].astype(str)
                df_out = df_out.merge(
                    tmp, on="page_title", how="left",
                    suffixes=("", f"__{field}_old"),
                )
                old = f"{field}__{field}_old"
                if old in df_out.columns:
                    df_out[field] = df_out[old].fillna(df_out[field])
                    df_out.drop(columns=[old], inplace=True)

    # --- Stage 2: fetch descriptions ---
    desc_done = df_out["description"].astype(str).str.strip().str.len() > 0
    to_fetch = df_out.index[~desc_done].tolist()

    if to_fetch:
        fetcher = WikiSummaryFetcher(
            cache_path=description_cache,
            request_sleep=sleep,
            timeout_s=timeout,
        )
        n_since_ckpt = 0
        for idx in tqdm(to_fetch, desc="Fetching descriptions"):
            title_clean = str(df_out.at[idx, "page_title"]).replace("_", " ").strip()
            df_out.at[idx, "description"] = fetcher.get_summary(title_clean)
            n_since_ckpt += 1
            if n_since_ckpt >= CHECKPOINT_EVERY:
                write_checkpoint(df_out, output_csv)
                n_since_ckpt = 0
        fetcher.save_cache()
        write_checkpoint(df_out, output_csv)

    # --- Stage 3: relevance score ---
    df_out["relevance_score_regex"] = [
        relevance_score(
            (t + " " + p + " " + d).lower(),
            mode,
            GREEN_PAT,
            DIGITAL_PAT,
        )
        for t, p, d in zip(
            df_out["page_title"].astype(str),
            df_out["hierarchy_path"].astype(str),
            df_out["description"].fillna("").astype(str),
        )
    ]

    # --- Stage 4: description-dependent filters ---
    blob_all = (
        df_out["page_title"].astype(str)
        + " "
        + df_out["description"].fillna("").astype(str)
    )
    keep_s4 = (
        ~blob_all.str.contains(A_BIO_PAT)
        & ~blob_all.str.contains(A_ORG_PAT)
        & ~blob_all.str.contains(A_MEDIA_PAT)
        & ~blob_all.str.contains(A_EVENT_PAT)
        & ~blob_all.str.contains(A_MEME_PAT)
    )
    prev = int(len(df_out))
    df_out = df_out.loc[keep_s4].copy()
    prev = print_stats(df_out, "after_filters_stage4", prev)

    # --- Embeddings (optional) ---
    if use_embeddings and df_out["root_similarity"].isna().any():
        df_out["root_similarity"] = compute_root_similarity(df_out, embed_model)

    # --- Stage 5: wiki link fetch ---
    extractor = WikiLinkExtractor(
        language="en",
        user_agent="wiki-taxonomy-cleaner/2.1 (srony356576@gmail.com) (research use)",
        sleep_s=sleep,
        timeout_s=timeout,
    )

    done_mask = df_out["linked_wiki_titles_json"].astype(str).str.len() > 0
    to_fetch_links = df_out.index[~done_mask].tolist()

    backoff = Backoff(sleep_s=sleep)
    n_since_ckpt = 0

    for idx in tqdm(to_fetch_links, desc="Fetching Wikipedia links"):
        title = str(df_out.at[idx, "page_title"]).strip()
        if not title:
            df_out.at[idx, "linked_wiki_titles_json"] = "[]"
            backoff.on_success()
        else:
            links = extractor.fetch_main_body_link_titles(title, include_lists=include_lists)
            df_out.at[idx, "linked_wiki_titles_json"] = json.dumps(links, ensure_ascii=False)
            if links:
                backoff.on_success()
            else:
                backoff.on_failure()

        n_since_ckpt += 1
        if n_since_ckpt >= CHECKPOINT_EVERY:
            write_checkpoint(df_out, output_csv)
            n_since_ckpt = 0
        backoff.sleep()

    # Write final output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out = df_out[[
        "root_technology", "page_title", "hierarchy_path", "description",
        "category_links", "relevance_score_regex", "root_similarity",
        "linked_wiki_titles_json",
    ]]
    df_out.to_csv(output_csv, index=False)
    write_checkpoint(df_out, output_csv)
    print_stats(df_out, "final_output")

    return 0
