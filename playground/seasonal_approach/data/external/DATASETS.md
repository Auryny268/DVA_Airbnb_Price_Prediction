# External Datasets

This folder contains data files sourced from outside the repo that are needed
to re-run the `seasonal_approach` notebooks from scratch.

---

## Files in this folder (tracked in git)

| File | Used in | Source | License |
|------|---------|--------|---------|
| `MTA_Subway_Stations_20260330.csv` | `01_feature_engineering.ipynb` | [MTA Subway Stations — NY Open Data](https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f) | Public Domain |
| `prices_and_months.csv` | `02_stacking.ipynb` | Derived from InsideAirbnb (team preprocessing by Nimit) — see `data/raw/` listing files | InsideAirbnb non-commercial |

---

## Files NOT in this folder (too large — download manually)

### `reviews.csv` — 314 MB

Used in `02_stacking.ipynb` to compute the seasonal index (monthly review volume).

**Download steps:**
1. Go to [https://insideairbnb.com/get-the-data/](https://insideairbnb.com/get-the-data/)
2. Find **New York City, New York, United States** — date **13 February, 2026**
3. Download `reviews.csv` (the uncompressed version, ~315 MB)
4. Place it at: `playground/seasonal_approach/data/external/reviews.csv`

> Note: `reviews.csv` is listed in `.gitignore`. Do not force-add it.

---

## Files already in the repo (`../../data/raw/`)

These are accessed via relative path from within `seasonal_approach` — no download needed after cloning.

| File | Used in | Source |
|------|---------|--------|
| `listings_2025-11-01.csv.gz` | `01_feature_engineering.ipynb` | [InsideAirbnb NYC — Nov 2025](https://insideairbnb.com/get-the-data/) |
| `lirr_stations.csv` | `01_feature_engineering.ipynb` | [MTA LIRR Stations — NY Open Data](https://data.ny.gov/Transportation/MTA-Long-Island-Rail-Road-Stations/vgmf-ypxs) |
| `nyc_facilities.csv` | `01_feature_engineering.ipynb` | [NYC Facilities Database — NYC Open Data](https://data.cityofnewyork.us/City-Government/Facilities-Database/ji82-xba5) |
| `nypd_complaint_data.csv` | `01_feature_engineering.ipynb` | [NYPD Complaint Data Historic — NYC Open Data](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i) |
| `mta_subway_stations.csv` | *(reference only)* | [MTA Subway Stations — NY Open Data](https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f) |

---

## Quick-start checklist

To run all 4 notebooks in order:

- [ ] Clone the repo (gets `data/raw/` files automatically)
- [ ] `pip install -r playground/seasonal_approach/requirements.txt`
- [ ] Download `reviews.csv` and place at `data/external/reviews.csv` (see above)
- [ ] Run `01_feature_engineering.ipynb` → produces `data/feature_matrix.csv`
- [ ] Run `02_stacking.ipynb` → produces `data/stacked_features.csv`, `data/seasonal_index.csv`
- [ ] Run `03_model.ipynb` → produces model + SHAP files
- [ ] Run `04_viz.ipynb` → produces plots
- [ ] `streamlit run app/app.py` to launch the dashboard
