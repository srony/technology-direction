# ST4TE exposure scores (frozen 2026-10-21)

Exposure of occupations and industries to digital and green technologies, measured from patent text. This is the one
exposure measure used in the ST4TE papers.

## Files

| File | Content |
|---|---|
| `exposure_yearly_occupation.parquet` | 403,353 rows: `domain`, `isco4` (ISCO-08 four-digit, 427 occupations), `level`, `technology`, `year`, `value` |
| `exposure_yearly_industry.parquet` | 322,985 rows: `domain`, `nace` (NACE Rev. 2 three-digit, 272 industries), `level`, `technology`, `year`, `value` |
| `technologies.csv` | every technology: `domain`, `level`, `technology` (identifier used in the yearly files), `name`, `parent`, `root` |
| `make_exposure.py` | computes exposure for any window and technology |
| `SHA256SUMS.txt` | checksums of the files |

`value` is the yearly value of an occupation or industry for a base technology, from the patents filed in that year,
before the inverse hyperbolic sine. Rows with no value are zero. The yearly files hold the base technologies only: 184
digital base technologies (`level` = `base`) and 66 green ENV-TECH nodes (`level` = `node`).

Levels. Digital: `base` (184 technologies), `intermediate` (43), `root` (10). Green: `node` (66 ENV-TECH nodes) and
`group` (10 groups). `technologies.csv` gives each base technology its intermediate technology and root (digital) or
its group (green, column `parent`).

## Exposure

The exposure to a base technology over a window of filing years is the inverse hyperbolic sine of its summed yearly
values. Every aggregate is the mean of the base-technology exposures it contains, zeros included: an intermediate
technology or root over its base technologies, a green group over its nodes, and the domain exposure over all base
technologies of the domain (184 digital, 66 green).

    python make_exposure.py --side occupation --domain digital --window 2009 2023 --average
    python make_exposure.py --side occupation --domain digital --window 2018 2023 --technology "Robot Manipulators and Arm Systems"
    python make_exposure.py --side industry --domain green --window 2009 2023 --level group --out green_groups.csv
    python make_exposure.py --list --domain digital --level root

Needs Python 3 with pandas, numpy and pyarrow. The script reproduces the scores of the papers exactly.
