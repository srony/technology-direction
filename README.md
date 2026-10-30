# technology-direction

A measure of the direction of technological change, substitution or complementarity, for technologies meeting
occupations in industries, rated by a silicon panel of persona-conditioned language-model respondents that is
calibrated against a completed Delphi survey of domain experts. The calibration rebuilds the survey with one prompt
per human response, answered in five draws, and scores the silicon rating distributions against the experts', read
against a uniform-random rater and a resample of the experts' own ratings. The application, in development, rates
substitution and complementarity for 11,977 combinations of a technology, an occupation and an industry with the
calibrated model and settings, in the roles of a chief executive, a manager and a worker, whose ratings are kept
separate, on a taxonomy and an exposure measure that language models helped to build under human direction, and tests
the result against employment change. This repository holds the silicon panel's prompts and code,
the application's outputs and the scripts that rebuild its validation from the released files.

This repository accompanies Rony, Ciarli and Nomaler (2026), *Substitution and Complementarity: From the Firm
Hierarchy*, a paper in progress. The calibration of the silicon panel is in Rony and Ciarli (2026), *Silicon
survey agents*.

## The silicon panel

The human benchmark is a real-time Delphi survey, of which the calibration uses 8,941 ratings given by 597 experts on
304 technologies, 47 industries and 1,066 work activities, on four statements, whether a technology will by 2030 replace humans in a
task in an industry, work alongside them, make the task routine enough to be automated, or make it more efficient.
Each rating carries the respondent's recorded profile. The silicon panel puts the survey's own statement and scale to
the model, one prompt per human response answered in five draws, with the respondent's profile as the persona and the rated technology,
industry and task as the context. Eight models were compared at sixteen decoding configurations on a 500-case sample
and the best, DeepSeek-V3.1, regenerated the full survey at the knee of the similarity-dispersion frontier
(temperature 0.3, top-p 0.3) and at a high-temperature configuration (1.0, 0.75). Agreement is the Earth Mover's
similarity between the human and the silicon rating distributions, read against a uniform-random rater (0) and a
bootstrap resample of the experts' own ratings (1).

## The application

1. **Taxonomy.** A crawl of Wikipedia from 27 seed categories, a cleaning cascade, matching against 1.9 million patent
   abstracts, a model acting as judge that decides which pages describe actual technologies, community detection on the
   wiki-link graph, a correction ledger and a second model review of every technology's pages. The result is 10 root domains,
   43 intermediate and 184 base technologies on 6,928 pages. The green technologies are 66 nodes taken from the OECD
   ENV-TECH classification.
2. **Exposure.** The citation-weighted semantic similarity between patent text and the task text of occupations and the
   activity text of industries, the same text, embeddings and aggregation in both domains. A model judge reviews the
   largest patent links of every technology. It moves patents to the technology they belong to, removes patents that are
   not digital technologies, sets links to zero, and zeroes the cells that rest on a lexical coincidence. Every verdict
   is released.
3. **Direction.** The calibrated model rates substitution and complementarity separately for 11,977 triplets of a base
   technology, a three-digit occupation group and an industry section, in the roles of a chief executive officer
   responsible for the organization's strategy and investment decisions, a manager who runs the team that performs
   the work and an experienced worker who performs it, five times each, 30 ratings per triplet. The ratings of each
   role are kept separate.

## Layout

```
0_silicon_panel/
  prompts.md, prompts.py    the system prompt, the survey prompt and the four statements, verbatim
  survey.py                 regenerates a survey of this design from any case file with a profile, one call per case
                            and draw, resumable
  llm.py                    the call to the model through OpenRouter, key from OPENROUTER_API_KEY
  metrics.py                Earth Mover's and Jensen-Shannon similarity, floor, ceiling, normalization, within-persona
                            variance and entropy, from a ratings file or from pooled counts per statement
  demo.py                   rates four illustrative cases on the four statements and prints the answers
1_taxonomy/
  pages.csv                 the 6,928 pages of the final digital taxonomy with root, intermediate and base technology
  levels.csv                every page of the clustering universe, removed pages marked
  technologies.csv          the 184 digital base technologies with their parents, and the 66 green ENV-TECH nodes
  page_verdicts.csv         the judge's verdict on each of the 11,906 candidate pages (is it a technology, confidence, reasoning)
  ledger/                   the correction passes, the first review round (page plan, off-theme pages) and the second
                            review round (every page moved, renamed, removed or deduplicated, with its item id)
  code/                     the taxonomy pipeline (crawl, clean, match, filter, cluster, tag)
2_exposure/
  release/                  the frozen exposure scores: yearly values per occupation and industry for every base technology,
                            and make_exposure.py, which computes exposure for any window and technology
  link_review_verdicts.csv  the 15,337 judged patent links
  link_review_procedure.md  the procedure of the link review
  cell_zeroing.csv          the entity-technology cells set to zero and the rule that matched each
  code/                     the final-taxonomy scripts, the exposure build, the review and the cleaning scripts
3_direction/
  grid.csv                  the 11,977 rated triplets with their exposures and titles
  ratings.parquet           every call made, with rating, reason, construct, persona and draw. The 359,310 ratings of
                            the grid's triplets, 30 per triplet, are the rows whose triplet is in grid.csv. The file
                            also holds 39,510 calls on triplets outside the final grid and 82 calls without a rating
  prompts.md, prompts.py    the system prompt, the personas and the user prompt, verbatim
  rate.py                   the rating panel's runner, resumable, for any grid
  llm.py                    the call to the model through OpenRouter, key from OPENROUTER_API_KEY
  demo.py                   rates four triplets and prints the answers
  pair_employment_shares.csv  pooled employment shares of industry sections within occupation groups and the reverse
  contents.py               substitution and complementarity content of every occupation and industry for any window
4_validation/
  direction_tables.py       stability, variance decomposition, sign changes across industries, figures
  employment_test.py        the employment test on the public Eurostat table lfsa_egai2d
  eurostat/                 the four Eurostat tables the test uses, as downloaded between July and October 2026
appendix/                   the technical appendix
```

## Running it

Python 3.10 or later with `pip install -r requirements.txt`.

```
cd 0_silicon_panel
set OPENROUTER_API_KEY=...                     # export on Unix
python demo.py                                  # 16 calls to the open-weight model
python survey.py --cases my_cases.csv --out my_ratings.csv --draws 5
python metrics.py --ratings my_ratings.csv --human my_human_ratings.csv
python metrics.py --histograms counts.csv       # per-statement scores from pooled counts alone
cd ../3_direction
python demo.py                                  # 24 calls to the open-weight model
python rate.py --grid grid.csv --out my_ratings.csv --limit 20
python contents.py --window 2011 2023           # contents of every occupation and industry
cd ../4_validation
python direction_tables.py                      # reliability and structure of the released ratings
python employment_test.py                       # reads the included Eurostat tables
```

`survey.py` reruns the calibration prompt on any case file with the columns named in its docstring, and `metrics.py`
scores the result against human ratings, or against pooled counts per statement when no respondent-level file is at
hand. `contents.py` and the two validation scripts run on the released files alone and reproduce the numbers of the
application's direction sections. The employment test reads the Eurostat tables as they were downloaded, so later
revisions by Eurostat do not change its result, and it fetches a table only if the file is missing. The ratings are
sampled at temperature 0.3 through an interface whose hosting provider was not recorded, so a rerun reproduces the
distribution of the ratings, not each rating. The exposure release reproduces the exposure scores of the paper for any
window.

## What is not here

The Delphi survey's respondent-level data, and with them the silicon responses, which are tied to a respondent's
profile and recorded rating, are not released, and neither are the analysis scripts of the calibration, which run on
them. The per-statement rating distributions of the human panel and the aggregated tables of the calibration are
available from the authors and will be added to this repository. Once added, `metrics.py --histograms` reproduces the
per-statement scores of the calibration from the counts. The patent extract behind the taxonomy matching and the exposure build is
licensed and is not redistributed. The scripts of the two builds are included for inspection and for holders of the
data. The two judge steps of the taxonomy and the link review ran on models whose verdicts are released in full, so
the contents and the validation are rebuilt from the released files. The paper also reports the employment test on the
restricted EU Labour Force Survey microdata, which cannot be shared. The public-table version in `4_validation/` is the
reproducible one.

## Use of the ratings

The ratings are assessments made by a language model in the roles of executives, managers and workers. They are not
the views of these groups. The substitution and complementarity contents record an association with past employment,
not a forecast for any worker or firm, and are not guidance on which work to automate.

## Data terms

The page lists derive from the English Wikipedia and carry its attribution terms (CC BY-SA). The green classification is
the OECD ENV-TECH indicator. Employment data are Eurostat's, reused with the source acknowledged as its reuse policy
asks. The code is released under the MIT license.
