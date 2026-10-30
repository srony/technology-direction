# LLM link review of the digital exposure (2026-10-04)

Cleaning step of the digital exposure (Paper 2). Applied by `p3_exposure/threshold_experiment/review.py` to the final build (`python -u review.py pb_mpnet_final_yearly pb_mpnet_final_reviewed_yearly`);
verdicts in `link_review_top50.csv` (this folder).

## Unit and selection
A link is (technology node at L2, industry or occupation, patent): the patent's contribution to that industry's
(NACE group) or occupation's (ISCO-4) exposure to the node, i.e. the code's mean over its description statements or
tasks of A x within-year citation weight x gates (the build formula). For every L2 node the 50 largest industry links
and the 50 largest occupation links (by share of the node's arcsinh exposure, filing years 2009-2023) were reviewed:
15,337 links in 181 nodes, carrying a median 44% (industry) and 46% (occupation) of a node's exposure. The link
contributions were recomputed from the final build and checked to add up to every node's exposure (relative error
below 1e-7, L2, L1 and root, both sides).

## Verdicts
- Patent: belongs (an instance of the node's technology) | move (another L2 node's technology; target named from the
  taxonomy outline) | not_a_digital_technology.
- Link: keep (the industry or occupation plausibly produces, uses or is affected by the patent's technology) | zero (the
  connection is only through shared words or generic phrasing). Links are judged on the patent's actual technology,
  independently of whether the patent belongs to the node.

## Procedure
Round 1: nine LLM reviewers (one per packet of L2 nodes) judged the five largest statement/task-patent pairs per node and
side (2,104 pairs). Round 2: 39 LLM reviewers judged the top-50 links per node and side, grouped by patent, with the
node's pages and each patent's title and abstract; 1,585 links judged in round 1 were carried over. Model: Claude
(Anthropic), via Claude Code workflow subagents; prompts below. Spot check: a stratified random sample of 48 verdicts
per round was re-read independently by the coordinating Claude session (not by a person): 47/48 and 45/48 agreement (the
disagreements were links the reviewers kept). No human review of the verdicts yet.

Result: 7,289 links of moved patents (4,021 patents), 627 links of patents that are not a digital technology (302),
6,744 zero links; 27% of links kept.

## Application in the build
After the node exposures are computed and before the off-theme zero list: zero links are removed; a moved patent's
contributions go to the target node (and its L1 and root), except its zero links; a not-digital patent's contributions
are removed. A patent keeps its own page for the task-page gate; only its node changes. Applied to industry and
occupation, L2/L1/root, plain and demeaned values, at every level where the build counts the patent.

## Round-2 prompt (per packet)
You are checking, for a digital-technology taxonomy used in economics research, whether it makes sense that a technology
node is linked to an industry or an occupation THROUGH a given patent. [...] 1) For EVERY patent decide: belongs, move
(give the exact L2 name from the outline), or not_a_digital_technology. 2) For EVERY link decide: keep (the
industry/occupation plausibly produces, uses or is affected by the patent's technology) or zero (the connection is only
through shared words or generic phrasing, e.g. TV 'programme' vs software 'programming', warehouse 'storage' vs data
storage, IT 'trainers' vs machine-learning 'training', guard services vs data protection). Judge links on the patent's
actual technology, independently of whether the patent belongs to this node. Be economically sensible: banks really use
cryptography, telecom operators really run networks, printers really use printing technology.
(Full prompts: workflow scripts digital-taxonomy-pair-audit and top50-link-review, session 2026-10-04.)
