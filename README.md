# NYC Airbnb Price Prediction & Recommendation
**DVA Spring 2026 — Team 005**

Team members: Rishikesh, Auryn, Nimit, Malhar, Harsh, Bret

---

## Project Overview
NYC Airbnb hosts often price listings based on guesswork, while professional tools like 
AirDNA and PriceLabs charge hundreds of dollars per month. This project builds a free, 
open-source alternative: an interactive system that predicts Airbnb listing prices and 
explains what drives them — using only publicly available data from InsideAirbnb NYC.

We combine XGBoost-based price prediction with NLP features derived from guest reviews 
(VADER sentiment, LDA topic modeling) and SHAP-based explainability, surfaced through 
an interactive visualization dashboard.

**Data Sources:**
- [InsideAirbnb NYC](http://insideairbnb.com/get-the-data/) — listings, reviews, calendar, neighbourhood boundaries
- [Kaggle - AirROI pricing data](TBD) — supplementary pricing data
**Visualization:** TBD (Streamlit or similar)  
**Compute:** Georgia Tech ICE-PACE cluster

---

## Data Setup
Data files are **not tracked in this repo** (too large for GitHub).

1. Download the NYC dataset from [InsideAirbnb](http://insideairbnb.com/get-the-data/)
2. Download: `listings.csv`, `reviews.csv`, `calendar.csv`, `neighbourhoods.geojson`
3. Place all files in `data/raw/`

---

## Repo Structure
```
data/
  raw/          # downloaded data files (gitignored)
  processed/    # cleaned/merged outputs (gitignored)
notebooks/      # exploratory and modeling notebooks
src/            # reusable Python modules
app/            # Streamlit dashboard
reports/        # progress report, final report
```

---

## Setup
```bash
git clone https://github.gatech.edu/rdonthula3/team005-airbnb-nyc.git
cd team005-airbnb-nyc
pip install -r requirements.txt
```

---

## Branching
- `main` — stable only, PR required to merge
- `dev` — integration branch, merge your work here first
- Personal branches: `yourname/feature`
