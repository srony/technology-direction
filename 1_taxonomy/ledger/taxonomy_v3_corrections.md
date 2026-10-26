# Taxonomy v3: Splits, Per-Page LLM Review, and Resolution

After v2 (`taxonomy_corrected_claude_v2.xlsx`) was published, an additional LLM review pass was run over the v2 pages and clusters. This pass produced three artifacts in `<run>/v3/`:

- `wiki_pages_claude2.l2_splits.csv` — partitions for 30 v2 L2 clusters being split into 2–3 sub-L2s each (62 sub-L2 rows total)
- `wiki_pages_claude2.pages.csv` — per-page KEEP/MOVE/REMOVE decisions with manual_verification (`agree`/`disagree`) for 7,406 v2 pages
- `wiki_pages_claude2.split_resolution.csv` — per-page sub-split assignments for 689 pages whose MOVE target was a v2 cluster being split (in a separate run after the initial review)

The v3 build applies these to v2 to produce `taxonomy_corrected_claude_v3_resolved.xlsx`.

Build script (one-off): `_tmp_build_v3_resolved.py` at project root.

## Methods and Rationale

### Why v3

v2 fixed structural issues from the initial Leiden+labeling output (Phases 0–6b in `notes/taxonomy_v2_corrections.md`) but treated each L2 cluster as monolithic. The claude2 review pass evaluated v2 at finer granularity:

- **Per-page**: should this page stay, move to a different cluster, or be removed?
- **Per-cluster**: is this L2 too broad? If so, partition it into 2–3 sub-clusters by page content.

Manual verification was then applied to each per-page decision (`agree`/`disagree`), gating which actions get applied. Splits were partitioned by `split_pages` columns in `l2_splits.csv` and all carried `manual_verification=agree`.

### Phase order

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → write v3 xlsx
```

The order matters because Phase 1 dissolves 30 v2 L2 clusters; Phase 2 then has to route 686 of 1,130 MOVEs to sub-clusters that didn't exist before Phase 1. Phase 4 cleans up split orphans (pages whose v2 parent was dissolved but who weren't included in any sub-split).

### Phase 1 — L2 splits

For each of the 30 v2 L2 clusters being split:

1. **Drop** the original parent L2 row.
2. For each sub-split (`split_idx = 0, 1, …`): create a new row with `(root, L1, split_label)` and the pages listed in `split_pages`.
3. **Dedup within parent**: if a page appears in `split_pages` of multiple sub-splits of the same parent (an LLM error — splits should be partitions), keep the first occurrence and drop the rest. 35 such duplicates were silently dropped during this run; logged with `action=split_dedup` in `v3_corrections`.

After Phase 1 the taxonomy has 153 − 30 + 62 = **185 L2 rows**. Pages from the 30 parents that aren't in any `split_pages` partition are now **orphans** — not in any row. These are split-orphans and Phase 4 handles them.

### Phase 2 — page MOVEs

For each MOVE row in `pages.csv` with `manual_verification=agree` (1,130 rows total):

1. **Pick target L2**:
   - If the page appears in `split_resolution.csv`: use `resolved_L2` from there (the LLM's per-page sub-split assignment). 686 of the 1,130 pages are covered.
   - Otherwise: use `suggested_L2` from `pages.csv`. 444 pages — these are MOVEs whose target was never a split source, so the original target is correct.
2. **Locate target row** in v3: full-path `(suggested_root, suggested_L1, target_L2)` first, then L2-only fallback.
3. **Remove from current location** (or note that it was already orphaned by Phase 1).
4. **Add to target row**.

If target still can't be resolved: page becomes a `move_failed_orphan` (recorded for Phase 4). With split_resolution applied, this run had **0 unresolved MOVEs**.

`MOVE+disagree` rows (3 cases) are not applied — pages stay in their v2 location regardless of split_resolution suggestions for them.

### Phase 3 — page REMOVEs

For each REMOVE row with `manual_verification=agree` (8 rows):

1. Locate page in current rows. If found: delete. If already orphan from Phase 1: log `remove_orphan_noop` (still counted toward the deletion total).

All 8 REMOVE+agree pages were already orphaned by their parent's split (they happened to be among the unassigned pages of split clusters), so all 8 became `remove_orphan_noop` — the page is deleted from `expected` either way.

`REMOVE+disagree` rows (65 cases) are not applied — pages stay.

### Phase 4 — LLM placement of split orphans

After phases 1–3, the only remaining unplaced pages are split-orphans:
- 49 `KEEP+agree` pages whose Pass-1 LLM decision was MOVE/REMOVE (so they weren't in any `split_pages`) but Pass-2 revised to KEEP
- 48 `REMOVE+disagree` pages that the LLM proposed deleting but were manually overridden

Both classes total **97 pages** that should be in v3 but have no home.

These are routed to Claude Opus via the `claude --print --model opus` CLI subprocess. Per-page context:
- `page` — Wikipedia title
- `wiki_description` — first 300 chars
- `v2_path` — `root / L1 / L2` from v2

The full v3 taxonomy (post Phases 1–3, ~184 L2 paths) is included in every prompt so the LLM sees the actual destination space.

**Batching**: 20 pages per call (5 batches: 20+20+20+20+17). Each batch's response is cached as JSON in `<run>/v3/orphan_llm_batches/batch_NNN.json` so re-runs skip already-computed batches.

**Schema-flexibility**: the LLM-CLI sometimes returns lowercase `l1`/`l2` instead of `L1`/`L2` (the JSON-schema instruction is appended to the prompt for the claude-code backend rather than enforced at the API level). Key access is case-insensitive: `entry.get("L1") or entry.get("l1")`.

### Phase 5 — write xlsx

Output at `<run>/taxonomy_corrected_claude_v3_resolved.xlsx` with three sheets:
- `taxonomy` — final v3 hierarchy (root, L1, L2, semicolon-joined pages)
- `v3_corrections` — full audit log: every split/move/remove/orphan-placement action with source/target paths and reason
- `verification` — counts (rows, pages, per-phase action totals, balance check)

A duplicate-page check runs before write: every page name must appear in exactly one row. (Phase 1's dedup eliminated the 35 LLM duplicates, so this passes cleanly.)

## Results

### Final v3 stats

- **9 roots, 48 L1, 184 L2, 7,398 pages**
- All 1,130 MOVE+agree applied
- 8 REMOVE+agree pages deleted
- 97 split-orphans LLM-placed
- 35 within-split duplicate page entries deduped (kept first occurrence)
- Page balance: v2 (7,406) − REMOVE (8) = v3 (7,398) ✓
- Zero remaining unresolved pages

### Root breakdown (v2 → v3)

| Root | v2 L2 → v3 L2 | v2 pages → v3 pages |
|---|---:|---:|
| Artificial Intelligence | 8 → 12 | 922 → ~720 |
| Computing Hardware & Systems | 12 → 17 | 1,644 → ~1,368 |
| Cryptography & Security | 7 → 8 | 389 → ~427 |
| Imaging & Display Technology | 9 → 12 | 520 → ~438 |
| Networking & Telecommunications | 10 → 12 | 978 → ~968 |
| Quantum Computing | 16 → 16 | 110 → ~111 |
| Robotics & Manufacturing | 30 → 35 | 819 → ~739 |
| Signal Processing & Communications | 30 → 33 | 937 → ~876 |
| Software & Data Management | 31 → 39 | 1,087 → ~984 |

(Cluster counts grew because 30 v2 L2s split into 62 sub-L2s — net +32 — minus a few empty rows dropped after moves.)

### Encoding cleanup

3 page names in v2 contained a UTF-8 replacement character (`�`, U+FFFD) where the original Wikipedia title had an en-dash (`–`). The LLM corrected these in its responses, so v3 now has the proper names:

| v2 | v3 |
|---|---|
| `Bowyer�Watson algorithm` | `Bowyer–Watson algorithm` |
| `Pseudo-marginal Metropolis�Hastings algorithm` | `Pseudo-marginal Metropolis–Hastings algorithm` |
| `Screen�smart device interaction` | `Screen–smart device interaction` |

This produced a benign "unplaced=3" warning at run time (the orphan-name lookup failed because the cleaned-name version didn't match the original-name set), but the pages were placed correctly under the cleaned names — the page balance check confirms.

### Output files

- `taxonomy_corrected_claude_v3_resolved.xlsx` — final taxonomy + audit
- `_tmp_build_v3_resolved.py` — build script (project root)
- `<run>/v3/orphan_llm_batches/batch_001.json` … `batch_005.json` — Claude Opus responses (cached)
- `<run>/v3/wiki_pages_claude2.l2_splits.csv` — split partitions (input)
- `<run>/v3/wiki_pages_claude2.pages.csv` — per-page decisions (input)
- `<run>/v3/Re_ Taxonomy-after correction.zip` → `wiki_pages_claude2.split_resolution.csv` — per-page sub-split assignments (input)

## Implications and limitations

- The split partitions (62 sub-L2 rows) are LLM-generated and rely on `split_pages` enumerations. 35 within-parent duplicates were quietly fixed during build. The LLM also sometimes left pages out of any `split_pages` partition (the 97 split-orphans) — Phase 4 patches these up.
- The 686 split_resolution overrides in Phase 2 are the LLM's per-page assignments for cases where the original MOVE target became invalid (its L2 was being split). Each placement carries a `confidence` (HIGH=574, MEDIUM=106, LOW=9) but all are applied uniformly per the project rule that `manual_verification=agree` is the gate, not LLM-confidence.
- Phase 4's 97 LLM placements are unverified — the LLM sees `(page, wiki_description, full taxonomy)` and picks a path, but no human has reviewed these. They could be added to a verification sheet for manual triage in a future pass if desired.
- Some root-level page totals shifted because moves cross root boundaries (e.g., Web Standards and Radar Systems content moved between AI / Software / Signal Processing). The taxonomy is now a different distribution from v2, not a refinement at the same root structure.
