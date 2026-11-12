# CRC3D - compartment-switching analysis of colorectal cancer Hi-C

Code and stamped result files for: "A two-layer architecture of colorectal
cancer 3D-genome evolution: shared epigenetic compartment reprogramming with
patient-specific directionality".

- `src/` core analysis and figure scripts (a9-a30 series; statistics
  pre-registered in script headers, zero gate changes). Download orchestration,
  exploratory probes and project-management utilities are internal and not part
  of this deposit. Data acquisition: a22b_w5_series_inventory.py (GEO inventory
  + iced-matrix download lists) and a16d_allvalidpairs_depth.py (per-bin read
  depth from allValidPairs); Xu et al. companion repo:
  github.com/XiangXuCode/CRC_metastasis_code
- `results/` stamped result CSVs (every number in the manuscript is traceable
  to one of these) and figure files
- `audit/` v1 counterparts of versioned artifacts (v2 files are cited in the
  manuscript; v1 = incident audit only)

Data: GSE133928, GSE63525, GENCODE v19, BIGD HRA000417 (all public).
Python 3; numpy, pandas, scipy, scikit-learn, matplotlib (see requirements.txt).
License: MIT.

Archived on Zenodo: https://doi.org/10.5281/zenodo.24149189
