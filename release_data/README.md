# MRVA monthly WTD data product

The release product is generated from the canonical reconstruction with:

```bash
uv run python tools/export_release_product.py
```

The resulting file is:

```text
mrva_monthly_wtd_2011_2023.nc
```

It contains:

- `water_table_depth(time, cell)`: monthly WTD in metres below land surface;
- `uncertainty_radius_pi75(time, cell)`: nominal 75% prediction-interval radius;
- projected coordinates in EPSG:5070;
- longitude and latitude in EPSG:4326;
- grid, row, column, and cell identifiers; and
- anchor-month and direct-observation metadata.

The NetCDF product covers 156 months from January 2011 through December 2023
and 87,871 active 1-km MRVA grid cells.

WTD is positive downward. Negative values denote reconstructed water levels
above land surface.

`SHA256SUMS.txt` records the checksum of the exact release file. The NetCDF
file is excluded from Git because it should be deposited in a DOI-issuing
data repository.
