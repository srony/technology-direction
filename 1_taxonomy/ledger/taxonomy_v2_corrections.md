# Taxonomy v2: Post-Labeling Corrections

After Leiden clustering (step 06) and bottom-up LLM labeling (step 07) produced an initial labeled hierarchy (`taxonomy_grouped_labeled_claude.csv`), several layers of post-hoc correction were applied to produce the publication taxonomy `taxonomy_corrected_claude_v2.xlsx`. This note documents the full correction pipeline and final structure.

Rebuild script (one-off): `_tmp_rebuild_corrected.py` at project root.

Output (winning config): `pipeline/06_cluster/digital/grid_search/v7/sw1.0_tau0.2_hub0_v6_mw8_tauplus/taxonomy_corrected_claude_v2.xlsx`

## Methods and Rationale

### Why a correction layer at all

LLM labeling is sensitive to (a) the bottom-up summarization order, (b) the surface form of the labels generated, and (c) sibling batching at each level. Logical-consistency validation (step 07b) flags violations — a child whose pages don't fit its parent label, sibling clusters whose labels overlap, peripherals that should be assigned/removed — but doesn't auto-fix them. The corrections layer applies fixes in a deterministic order:

- **Phases 0–4**: automated, derived from logical-consistency output.
- **Phase 6a**: manual structural corrections from human review of the labeled taxonomy.
- **Phase 6b**: page-level moves from a per-page LLM review, gated by manual verification.

Two LLM judges run the logical-consistency check in parallel: **claude-judge (CJ)** and **openrouter-judge (OJ)**. CJ applies first; OJ then gap-fills any item CJ didn't touch. This dual-judge pattern lets a stronger model carry the bulk of decisions while a cheaper model fills coverage gaps.

### Phase order

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → write v1 xlsx
                                            ↓
                                       Phase 6a → Phase 6b → write v2 xlsx
```

v1 (`taxonomy_corrected_claude.xlsx`) is the automated-only output (164 rows). v2 (`taxonomy_corrected_claude_v2.xlsx`) is v1 plus manual structural corrections + page moves (153 rows). v2 is the publication artifact.

### Phase 0 — labeler-merge directives

The LLM labeler occasionally emits a label like `"merge into #12"` or `"→ Database Systems"` instead of a real label, when it determines that a cluster should not exist as its own entity. These directives are detected by regex on the L1/L2 label (`merge|→|MERGE|#\d`) and consumed: pages are moved to the indicated target sibling, and the source row is emptied. Must run **before** Phase 1 so that Phase 1's cid-based lookups don't target rows that will later disappear.

### Phase 1 — peripheral resolution

Peripheral pages (those that didn't make it into a core cluster — local peripherals from singleton-demotion at resplit, or global peripherals that couldn't be placed) carry a labeling-time decision: `assign / keep_peripheral / remove`. Phase 1 reads peripheral records from logical-consistency output (CJ then OJ gap-fill) and:

- **assign**: append page to target cluster.
- **keep_peripheral**: leave in place (no-op).
- **remove**: delete page from taxonomy.

OJ entries are only applied to pages CJ didn't already touch.

### Phase 2 — subsumption remediation

For each parent-child violation flagged by the LC check (parent label/technologies don't subsume the child's), the LC remediation sheet provides a `move_to` target. Phase 2 resolves the target (full-path match → L2-only fallback → sibling_b lookup) and reparents. CJ first, OJ gap-fill.

### Phase 3 — sibling exclusivity remediation

For sibling pairs whose labels overlap, the LC remediation sheet provides a merge or move directive. Same matching logic as Phase 2. CJ first, OJ gap-fill.

### Phase 4 — drop empty + sort

After phases 0–3, rows whose `pages` list is empty are dropped. Surviving rows are sorted alphabetically by `(root, L1, L2)` to give a stable diff-friendly ordering. The result is written as `taxonomy_corrected_claude.xlsx` (v1, 164 rows).

### Phase 6a — manual structural corrections (v2)

12 sections of human-reviewed structural fixes. Section labels in code: `phase6_v2_corrections`.

1. **Merge 6 duplicate L2 clusters** — same L2 name appearing twice (e.g., "Operating System Internals" as a 5-page fragment under one L1 and a 358-page main cluster under another). Small fragment merged into large namesake.
2. **Absorb 4 singleton roots** — roots containing only 1 L2 (e.g., "Clock Synchronization", "Automated Cash Handling", "Unix Printing Systems", "Transport Control Systems") merged into existing semantically-matching L1s elsewhere.
3. **Fix 2 questionable placements** — Autonomous Farm Equipment (was under 3D Visualization) → Robotics & Navigation; Unconventional Computing (was under Media Coding) → Computer Hardware Architecture.
4. **Relabel 3 mislabeled clusters**:
   - "Multilevel Security" → "Storage & Virtualization" (90% storage/virtualization pages)
   - "HDR and Tone Mapping" → "Display Technology & Image Capture" (90% display/camera pages)
   - "Concurrency Control" → "Database Systems & Data Processing" (85% database pages)
5. **Split "Imaging, Robotics & Manufacturing" root** into two roots:
   - **Imaging & Display Technology** (L1s: Imaging & Display, 3D Visualization)
   - **Robotics & Manufacturing** (L1s: Robotics & Navigation Systems, Digital Fabrication, Industrial Automation, Physics Simulation, Inspection & Underwater Robotics)
   The split is per-L2: imaging/display/visualization L2s go to the first; fabrication/robotics/sensing/automation L2s go to the second. Sensors and Detection Systems (general-purpose sensors, not imaging) moves to Robotics.
6. **Fix 5 misplaced fragments + Web Standards move**:
   - Thermal Management (12p) → Computer Hardware Architecture
   - Electromagnetic Generation & Propulsion (20p) → Computing/Power Electronics
   - Monte Carlo Particle Transport (3p) → Robotics/Physics Simulation
   - Industrial Safety Interlocks (3p) → Robotics/Robotics & Navigation
   - Smart Television Platforms (12p) → Signal Processing/Digital TV Broadcasting
   - Web Standards (205p) AI → Software & Data Management
7. **Move Radar Systems** (L1 "Radar & Weather Modeling", 321p) AI → Signal Processing & Communications.
8. **Relabel 2 kitchen-sink clusters** (large, irreducibly broad — Wikipedia link density prevents finer splitting):
   - "Hardware Architecture" (751p) → "Computer Hardware & Systems Architecture"
   - "Packet Segmentation & Reassembly" (534p) → "Network Protocols & Infrastructure"
9. **Split "Deep Learning and NLP" (440p)** by keyword into:
   - "Neural Networks & Deep Learning" (319p) — ML architectures, training, optimization
   - "NLP, Speech & Generative AI" (121p) — language models, speech, translation, generative AI, recommenders
   Split is keyword-based on page names (~80 keywords: speech/voice, GPT/BERT/LLM, NLP, generative, recommender, embedding, etc.).
10. **AI L1 regroup (Option B)** — flatten AI root from one giant "AI and Pattern Recognition" L1 (7 L2s) + orphan "Data Management & Processing" L1 into 5 method-based L1s:
    - **Computer Vision & Pattern Recognition** (Computer Vision and Biometrics, OCR)
    - **Natural Language & Generative AI** (NLP, Speech & Generative AI)
    - **Machine Learning** (Neural Networks & Deep Learning, Biodata Clustering & Microarrays, Dependency Networks)
    - **Symbolic AI & Planning** (Symbolic AI and Optimization)
    - **Biosignal Processing** (Neural Interfaces and Biosignal Processing)
11. **Computing Hardware & Systems cleanup**:
    - Rename L1 "Fiber-Optic Communication" → "Data Center Infrastructure" (content was modular data centers, not fiber optics)
    - Move L2 "Graph Reduction" (functional programming evaluation) → Software/Programming Languages & Software Engineering
    - Merge L2 "Liquid Cooling Systems" (3p singleton L1) → "Thermal Management" under Computer Hardware Architecture
12. **Networking & Cryptography split (Option Z)** — split single root into TWO roots:
    - **Cryptography & Security** — L1s: Cryptography (Authentication & Access Control, Cryptography & Encryption, Random Number Generation), Identification & Payment Systems (Barcode, Electronic Payment, RFID, Smart Cards)
    - **Networking & Telecommunications** — L1s: Network Services & Applications (Email, DNS, Participatory Distributed Systems), Internet Protocols & Backbone (Network Protocols & Infrastructure, Network Clock Synchronization), Wireless & Connectivity (Wireless Networking, Digital Contact Tracing, Power-Line Communication), Telephony & Signaling (Telephony & Cellular Networks, Optical Signaling)
    Plus pre-split cleanup: merge 4-page singleton L1 "Knowledge-Based Authentication" into "Authentication & Access Control".

Each correction is logged as a row in the `v2_corrections` sheet of the output xlsx with action/source/target/n_pages/reason fields.

### Phase 6b — page-level moves from per-page LLM review

A separate LLM pass reviewed every page individually and emitted `wiki_pages_claude.pages.csv` with per-page `decision ∈ {KEEP, MOVE, REMOVE}`, `confidence ∈ {HIGH, MEDIUM, LOW}`, and (for MOVE) `suggested_root / suggested_L1 / suggested_L2`.

Application policy:
- **HIGH-confidence MOVE/REMOVE**: applied automatically.
- **MEDIUM/LOW-confidence MOVE/REMOVE**: applied only if the corresponding `(Page, L2)` pair is marked `agree` in `manual_verification_by_confidence.xlsx` (sheets MEDIUM, LOW). KEEP rows and non-agreed MED/LOW rows are skipped.

Target resolution for MOVE: full-path match `(suggested_root, suggested_L1, suggested_L2)` first, then L2-only fallback, then a Phase-6a-relabel redirect (e.g., a page suggested for "Concurrency Control" is redirected to "Database Systems & Data Processing" since Phase 6a renamed that cluster).

Failed moves (target not found, blank suggested_L2) are logged but not applied.

### Phase 5/5b — write outputs

Phase 5 writes v1 (`taxonomy_corrected_claude.xlsx`) with sheets:
- `taxonomy` (root/L1/L2/pages)
- `deleted_peripherals`, `subsumption_applied`, `sibling_exclusivity_applied`, `verification`

Phase 5b writes v2 (`taxonomy_corrected_claude_v2.xlsx`) with sheets:
- `taxonomy` (final hierarchy)
- `v2_corrections` (full audit log of phases 6a + 6b)
- `verification` (row counts, root/L1/L2 totals, pages)

## Results

### Final v2 stats

- **9 roots, 48 L1 clusters, 153 L2 clusters, 7,406 pages**
- 1,839 v2 corrections applied (phases 6a + 6b combined)
- v1 produced 164 L2 rows; v2 collapses to 153 after structural merges and root splits

### Root breakdown

| Root | L1 | L2 | Pages |
|---|---:|---:|---:|
| Artificial Intelligence | 5 | 8 | 922 |
| Computing Hardware & Systems | 5 | 12 | 1,644 |
| Cryptography & Security | 2 | 7 | 389 |
| Imaging & Display Technology | 2 | 9 | 520 |
| Networking & Telecommunications | 4 | 10 | 978 |
| Quantum Computing | 5 | 16 | 110 |
| Robotics & Manufacturing | 5 | 30 | 819 |
| Signal Processing & Communications | 8 | 30 | 937 |
| Software & Data Management | 12 | 31 | 1,087 |

Three roots emerged from this layer that did not exist in the raw clustering output: **Imaging & Display Technology** and **Robotics & Manufacturing** (from the section-5 split of "Imaging, Robotics & Manufacturing"); **Cryptography & Security** and **Networking & Telecommunications** (from the section-12 split of "Networking & Cryptography").

### Kitchen-sink review

22 clusters with >100 pages were reviewed individually (`_tmp_big_clusters.txt` enumerates pages). Findings:
- **Hardware Architecture** (751p) and **Packet Segmentation & Reassembly** (534p) — genuine kitchen-sinks but irreducibly so under Wikipedia's link density. Relabeled to broader names that match their actual breadth (sec. 8).
- **Deep Learning and NLP** (440p) — separable along NN-vs-language axis; split by keyword (sec. 9).
- **Radar Systems** (321p) — coherent topic but misplaced under AI; moved to Signal Processing (sec. 7).
- All other large clusters: thematically coherent, no action.

### Output files

- `taxonomy_corrected_claude_v2.xlsx` — final publication taxonomy
- `taxonomy_corrected_claude.xlsx` — v1 (automated phases only, no Phase 6)
- `_tmp_rebuild_corrected.py` — rebuild script (project root)
- `wiki_pages_claude.pages.csv` — per-page LLM decisions (input to Phase 6b)
- `manual_verification_by_confidence.xlsx` — manual `agree`/`disagree` for MED/LOW-confidence Phase 6b moves
- `taxonomy_grouped_labeled_claude.csv` — source LLM-labeled taxonomy (input to Phase 0)
- `validation/logical_consistency_labels_claude_judge_claude.xlsx` — CJ output (input to phases 1–3)
- `validation/logical_consistency_labels_claude_judge_openrouter.xlsx` — OJ output (input to phases 1–3 gap-fill)

### Implications and limitations

- The 12-section Phase 6a is hand-curated and not reproducible from data alone. It encodes domain judgments (e.g., that Radar belongs in Signal Processing rather than AI) that the raw clustering + LLM labeling missed. Future re-runs of the upstream pipeline would need either (a) re-application of these same corrections or (b) prompt/clustering changes that make them unnecessary.
- The Section 9 keyword split of "Deep Learning and NLP" is a heuristic; pages whose names don't surface NLP/generative keywords (e.g., a transformer paper that just happens to be about translation) may end up in the wrong half. A semantic-similarity split would be more robust but was deferred.
- v3 work (in progress) re-runs the LLM labeling from scratch (`wiki_pages_claude2.*.csv` files in `v3/`) rather than applying further patches on top of v2. This note covers v2 only.
