# MRVA monthly water-table reconstruction

This repository contains the public analysis code for a geophysics-informed
graph neural network reconstruction of monthly water-table depth (WTD) across
the Mississippi River Valley alluvial aquifer (MRVA).

The public workflow covers:

1. horizon-specific GNN training for 1-, 3-, and 6-month WTD changes;
2. monthly WTD reconstruction from January 2011 through December 2023;
3. drought-response metric calculation;
4. physics-guided assignment of three response classes;
5. ExtraTrees prediction of Slow-recovery probability; and
6. generation of the main Figure 2 and Figure 3 analyses.

The manuscript, private working files, intermediate model caches, and
unselected figure notebooks are not part of this code release.

## Repository structure

```text
assets/spatial/       Small spatial overlays used by the public figures
configs/              H1, H3, and H6 model configurations
notebooks/            Public training, reconstruction, analysis, and figure notebooks
src/                  GNN preprocessing, graph, model, training, and reconstruction code
tools/                Command-line tools required by the public workflow
release_data/         Metadata and checksums for the separately archived WTD product
data_manifest.csv     Source and local-path inventory for required input datasets
```

## Environment

Python 3.12 is used by the released notebooks. With
[uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run jupyter lab
```

The exact dependency resolution is recorded in `uv.lock`.

## Required input data

Large and third-party input datasets are not committed to this repository.
Their expected local paths, roles, and source records are listed in
`data_manifest.csv`. Before running the workflow, place each input at its
listed path relative to the repository root.

The three training notebooks generate their own model-ready caches under
`data/train_val_test_inputs/GNN_spacetime/`. These caches are about 6 GB in
the current analysis and should not be version controlled.

## Notebook order

Run the notebooks in this order:

```text
0_GNN_H1.ipynb
0_GNN_H3.ipynb
0_GNN_H6.ipynb
1_MAP_recon.ipynb
2_drought_metrics.ipynb
3_clustering_physics.ipynb
4_regression.ipynb
Fig2.ipynb
Fig3.ipynb
```

The three training notebooks may be run independently. All three ensembles
must be complete before `1_MAP_recon.ipynb` is run.

## Main reconstruction output

The reconstruction notebook writes the canonical monthly product to:

```text
outputs/RECON_MAIN_2011_2023/reconstruction/wtd_reconstructed_matrix.npy
```

Its spatial and temporal coordinates are stored in:

```text
outputs/RECON_MAIN_2011_2023/metadata/grid_lookup.csv
outputs/RECON_MAIN_2011_2023/metadata/month_index.csv
```

Model uncertainty is stored as the nominal 75% prediction-interval radius:

```text
outputs/RECON_MAIN_2011_2023/model_uncertainty/monthly_model_uncertainty_radius_matrix.npy
```

Create the self-describing release product with:

```bash
uv run python tools/export_release_product.py
```

This produces `release_data/mrva_monthly_wtd_2011_2023.nc` and updates
`release_data/SHA256SUMS.txt`. The NetCDF file contains monthly WTD, PI75
uncertainty radius, projected and geographic coordinates, and time metadata.

Read the archived product with:

```python
import xarray as xr

ds = xr.open_dataset("mrva_monthly_wtd_2011_2023.nc")
wtd = ds["water_table_depth"]
pi75 = ds["uncertainty_radius_pi75"]
```

## Reproducibility notes

- WTD is depth below land surface in metres and is positive downward.
- Negative WTD values denote reconstructed water levels above land surface.
- The active MRVA grid contains 87,871 cells at 1-km spacing in EPSG:5070.
- The reconstruction contains 156 monthly fields from 2011-01 to 2023-12.
- Random seeds are fixed at 11, 22, 33, 44, and 55 for each prediction interval.
- GRACE/GRACE-FO values are not used to train the GNN or define response
  classes. They provide an independent regional-scale comparison.
- Notebook execution outputs are intentionally cleared from the public
  versions to remove local paths and machine-specific metadata.

## Data and code availability

Upstream datasets remain governed by their
original providers and licences.
