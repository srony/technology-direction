# technology-direction

A measure of the direction of technological change, substitution or complementarity, for technologies meeting
occupations in industries. Language models do the judging at the three points of its construction where expert judgement
was the bottleneck. The researchers set the design, keep a ledger of every intervention and validate each output against
experts and against employment. The pipeline has run once from end to end, for the digital and the green technologies
diffusing in Europe. This repository holds its outputs, the instrument that produced the direction ratings, and the
scripts that rebuild the validation from the released files.

The full results are in the companion paper, Rony, Ciarli and Nomaler (2026), *Green and Digital Technology: Exposure,
Substitution and Complementarity*, and the calibration of the rating instrument against a panel of some 600 experts in
Rony and Ciarli (2026), *Silicon survey agents*.

## The three steps

1. **Taxonomy.** A crawl of Wikipedia from 27 seed categories, a cleaning cascade, matching against 1.9 million patent
   abstracts, a model acting as judge that decides which pages describe actual technologies, community detection on the
   wiki-link graph, a correction ledger and a second model review of every technology's pages. Result: 10 root domains,
   43 intermediate and 184 base technologies on 6,928 pages. The green technologies are the 66 nodes of the OECD
   ENV-TECH classification.
2. **Exposure.** The citation-weighted semantic similarity between patent text and the task text of occupations and the
   activity text of industries, the same text, embeddings and aggregation in both domains. A model judge reviews the
   largest patent links of every technology and zeroes the cells that rest on a lexical coincidence. Every verdict is
   released.
3. **Direction.** An open-weight model rates substitution and complementarity separately for 11,977 triplets of a base
   technology, a three-digit occupation group and an industry section, under three workplace personas and in five
   draws, 30 ratings per triplet.

## Layout

```
1_taxonomy/
  pages.csv                 the 6,928 pages of the final digital taxonomy with root, intermediate and base technology
  levels.csv                every page of the clustering universe, removed pages marked
  technologies.csv          the 184 digital base technologies with their parents, and the 66 green ENV-TECH nodes
  page_verdicts.csv         the judge's verdict on each of the 11,906 candidate pages (is it a technology, confidence, reasoning)
  ledger/                   the correction passes, the first review round (page plan, off-theme pages) and the second
                            review round (every page moved, renamed, removed or deduplicated, with its item id)
  code/                     the taxonomy pipeline (crawl, clean, match, filter, cluster, tag) and the final-taxonomy scripts
2_exposure/
  release/                  the frozen exposure scores: yearly values per occupation and industry for every base technology,
                            and make_exposure.py, which computes exposure for any window and technology
  link_review_verdicts.csv  the 15,337 judged patent links and the protocol of the review
  cell_zeroing.csv          the entity-technology cells set to zero and the rule that matched each
  code/                     the exposure build, the review and the cleaning scripts
3_direction/
  grid.csv                  the 11,977 rated triplets with their exposures and titles
  ratings.parquet           every answered call: rating, reason, construct, persona, draw
  prompts.md, prompts.py    the system prompt, the personas and the user prompt, verbatim
  rate.py                   the rating instrument, resumable, for any grid
  demo.py                   rates four triplets and prints the answers
  pair_employment_shares.csv  pooled employment shares of industry sections within occupation groups and the reverse
  contents.py               substitution and complementarity content of every occupation and industry for any window
4_validation/
  direction_tables.py       stability, variance decomposition, sign changes across industries, figures
  employment_test.py        the employment test on the public Eurostat table lfsa_egai2d
appendix/                   the technical appendix and the companion paper's draft (preliminary, not for circulation),
                            so that the appendix's pointers to the paper's sections and appendices can be followed
```

## Running it

Python 3.10 or later with `pip install -r requirements.txt`.

```
cd 3_direction
set OPENROUTER_API_KEY=...                     # export on Unix
python demo.py                                  # 24 calls to the open-weight model
python rate.py --grid grid.csv --out my_ratings.csv --limit 20
python contents.py --window 2011 2023           # contents of every occupation and industry
cd ../4_validation
python direction_tables.py                      # reliability and structure of the released ratings
python employment_test.py                       # fetches the Eurostat table on first run
```

`contents.py` and the two validation scripts run on the released files alone and reproduce the numbers of the
companion paper's direction sections. `rate.py` reproduces the rating run for any grid with the same prompts, model
and sampling settings. The exposure release reproduces the exposure scores of the paper for any window.

## What is not here

The patent extract behind the taxonomy matching and the exposure build is licensed and is not redistributed. The
scripts of the two builds are included for inspection and for holders of the data. The two judge steps of the taxonomy
and the link review ran on models whose verdicts are released in full, so everything downstream of them reproduces
exactly. The companion paper also reports the employment test on the restricted EU Labour Force Survey microdata, which
cannot be shared. The public-table version in `4_validation/` is the reproducible one.

## Data terms

The page lists derive from the English Wikipedia and carry its attribution terms (CC BY-SA). The green classification is
the OECD ENV-TECH indicator. Employment data are Eurostat's. The code is released under the MIT licence.
