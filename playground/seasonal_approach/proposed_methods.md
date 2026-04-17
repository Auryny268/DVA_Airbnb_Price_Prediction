# Proposed Methods — Seasonal Multi-Month Price Prediction
**Team 005 · CSE 6242 Spring 2026 · Working Draft**

---

## Problem Framing

We want to predict the nightly price of a NYC short-term rental given its features, **and** a target month. A host asking "what should I charge in August vs February?" needs a model that understands seasonal variation. A model trained only on November data will predict the same price regardless of month — which is wrong.

The challenge: InsideAirbnb's price scraper broke on December 1, 2025. We have actual prices for April–November 2025 (8 months) and nothing after.

---

## Decision 1 — Why not SARIMAX

SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) models a time series as a function of its own lagged values, seasonal differences, and external variables. To fit a SARIMA(p,d,q)(P,D,Q)[12] model — where 12 is the seasonal period (months) — you need at minimum **2 full seasonal cycles**, i.e., 24 monthly observations.

We have 8. Fitting a SARIMA[12] model on 8 data points gives fewer observations than the seasonal period itself. The model cannot identify a seasonal pattern; it would either fail to converge or massively overfit to the training data.

**What we do instead:** We treat month as a feature. Rather than modeling price as a time series per listing, we include `month` (integer 4–11) and a `seasonal_index` (float derived from demand data) directly as input features in XGBoost. This lets the model learn seasonal pricing patterns from the 8 months of real data without imposing the structural assumptions of ARIMA.

This is a pragmatic downgrade from the originally proposed approach. We note it as a limitation: with 2+ years of data, SARIMAX would be the superior choice for capturing autocorrelated seasonal structure.

---

## Decision 2 — Reusing the static feature matrix

The spatial and structural features we engineered (subway distance, POI density, crime density, amenities, host stats) describe a listing's **fixed attributes** — they don't change from month to month. A listing's distance to the subway is the same in April and November. Its amenity set is the same.

What *does* change month-to-month is the listed price.

This means we can avoid re-running the expensive spatial pipeline (haversine distance matrices over 34k POIs, crime density counts, etc.) 8 times. Instead:

1. Use the **existing `xgboost_features.csv`** as a static feature table (4,073 STR listings × 57 features)
2. For each of the 8 months, extract just `(listing_id, price)` from the raw snapshot
3. Join price onto the feature table → one copy of the feature matrix per month
4. Stack all 8 → ~32k rows

Each row is a `(listing, month)` observation. The model sees the same structural features for a listing across months, but a different price and a different `month` value.

**Trade-off:** This assumes listing attributes are stable across Apr–Nov 2025, which is mostly true. Hosts do occasionally change amenities or update profiles, but for an 8-month window on NYC compliant STRs, this is an acceptable approximation.

---

## Decision 3 — Why not use Nimit's `prices_and_months.csv` directly

Nimit's preprocessing file has 44,578 unique listings × 12 months = 534,936 rows. The problem:

- **No `min_nights` column.** We cannot filter to short-term rentals (min_nights < 28). The file mixes long-term rentals (monthly leases) into the price data.
- **Median price mismatch confirms contamination.** Nimit's November median is $154; our STR-filtered November median is $222. The $68 gap is entirely explained by including long-term rentals, which price at a lower per-night equivalent.
- **Training a model on mixed data would predict a weighted average of two different markets.** A host pricing a compliant STR ($200+/night) would receive predictions dragged down by LTR listings ($80–$120/night equivalent).

**What we use instead:** We use Nimit's file but filter it down to our 4,073 STR listing IDs (derived from the November 2025 snapshot with min_nights < 28). Since our feature matrix was built with the STR filter applied, any listing ID present in our feature matrix is by definition a short-term rental. This implicitly enforces the STR filter on Nimit's data without needing a `min_nights` column. We additionally cap prices at $2,000/night (see Decision 6) to remove placeholder blocking prices.

---

## Decision 4 — Seasonal index derivation and its limitations

We derive a monthly seasonal demand index from InsideAirbnb review volume data (2019–2024), following the approach used by Gunter et al. (2020) who treat review counts as proxies for occupancy demand. The index for month $m$ is:

$$\text{index}_m = \frac{\bar{R}_m}{\bar{R}_{\text{annual}}}$$

where $\bar{R}_m$ is the mean monthly review count for month $m$ averaged over 2019–2024 (excluding 2020–2021 COVID distortion years).

**Acknowledged limitations:**

1. **Review submission lag.** Reviews are written after checkout. A December stay may generate a review in January, suppressing the December review count and understating December demand.
2. **Review rate heterogeneity.** Holiday travelers may be less likely to leave reviews (short stays, leisure focus), further compressing the apparent December signal.
3. **Review volume ≠ price.** Demand and price are correlated but not identical. High-demand months do not necessarily produce proportionally higher prices if supply also expands.
4. **Not externally validated.** Our computed December index (1.032×) is a self-derived estimate. NYC & Company visitor data and STR industry sources (AirDNA) suggest December premiums of 15–25% over November — substantially higher than our 5.5% estimate. We treat our index as a **conservative lower bound** for December pricing.

We use the seasonal index as a continuous feature in XGBoost, not as a post-hoc price multiplier. This lets the model learn the relationship between the index and price from data, rather than hardcoding a multiplier.

---

## Decision 5 — Handling December in the model

December 2025 prices are unavailable. We handle this in two ways:

1. **In training:** December is excluded from the training set (no labels). The model is trained on months 4–11.
2. **At inference:** To predict a December price for a listing, we pass `month=12` and `seasonal_index=1.032` as inputs. The model extrapolates from the month feature's learned relationship with price. This is an out-of-distribution prediction and should be flagged as such in any output to users.

This is honest: we don't pretend to have December data. We make a conservative estimate and label it as estimated.

---

## Decision 6 — Price cap at $2,000/night

InsideAirbnb's raw data includes listings with prices up to $50,000/night. These are **not real market prices** — they are a well-documented Airbnb host behavior where hosts set an extreme placeholder rate to effectively block their calendar without formally marking dates unavailable. The presence of these values is a platform artifact, not a reflection of the NYC STR market.

**Evidence:**
- 85 unique listings priced at exactly $50,000/night (a suspiciously round and extreme number)
- The price distribution has a natural break between the 97th percentile ($1,595) and 98th percentile ($4,818) — a gap of $3,200 with no listings in between
- Genuine NYC STR listings under Local Law 18 (small apartments with host present) do not command $50,000/night

**Decision:** Cap prices at $2,000/night. This removes 740 rows (2.6% of the stacked dataset) and 85 unique listings. All remaining prices are plausible for the NYC STR market.

**Effect on model:** Without the cap, the mean absolute error on the November test set was $692 — entirely driven by 10 $50,000-priced listings contributing ~$450,000 of absolute error. After capping: test MAE = $45. The median absolute error was $15 both before and after — confirming that the cap only removes artificial outliers, not real market variation.

---

## Decision 7 — Removing InsideAirbnb estimated revenue/occupancy features

InsideAirbnb provides two computed columns:
- `estimated_revenue_l365d` — estimated annual revenue
- `estimated_occupancy_l365d` — estimated occupied nights per year

These are computed by InsideAirbnb's model as:

```
estimated_revenue = current_listing_price × estimated_occupied_nights
```

This is an **exact algebraic function of the target variable (price)**. Verified in data:

```python
(estimated_revenue / estimated_occupancy) == price   # True for 100% of rows, within $1
```

Including either feature gives the model the answer. Both are excluded from training.

**Effect on performance:** Removing them drops R² from 0.934 → 0.915 and raises MdAPE from 7.4% → 10.6%. The model remains strong on legitimate features. More importantly, it is now predicting price from observable listing attributes — which is the actual task.

**Practical note:** At inference time, a host pricing their listing *does not know* their InsideAirbnb estimated revenue — that figure is computed from their current price. Including it would make the model circular: you need a price to get estimated revenue, and you need estimated revenue to get a price.

---

## Full Pipeline (this playground)

```
data/raw/listings_2025-11-01.csv.gz                           [read-only]  ← STR features
data/raw/nyc_facilities.csv, nypd_complaint_data.csv,         [read-only]  ← spatial
         lirr_stations.csv
dataset/subway nyc/MTA_Subway_Stations_20260330.csv           [read-only]  ← subway (new)
        |
        ▼
01_feature_engineering.ipynb
  → playground/seasonal_approach/data/feature_matrix.csv   (4,073 STR listings × 64 cols)

dataset/insideairbnb nyc/nimit preprocessing/prices_and_months.csv  [read-only]  ← prices
dataset/insideairbnb nyc/2026-02 february/reviews.csv               [read-only]  ← seasonal idx
        |
        ▼
02_stacking.ipynb
  → playground/seasonal_approach/data/stacked_features.csv  (28,244 rows × 64 cols)
  → playground/seasonal_approach/data/seasonal_index.csv
        |
        ▼
03_model.ipynb
  → playground/seasonal_approach/data/xgb_model.json
  → playground/seasonal_approach/data/eval_metrics.csv
  → playground/seasonal_approach/data/shap_summary.csv
        |
        ▼
04_evaluation.ipynb                                           [next step]
  → ablation study, baseline comparisons, error analysis
```

---

## Feature additions vs base model

The stacked model uses **59 features**, 28,244 rows across 8 months. Features vs original pipeline:

| Category | Features added | Features removed | Net |
|---|---|---|---|
| Seasonal | `month`, `seasonal_index` | — | +2 |
| Subway (new MTA file) | `subway_lines_500m`, `nearest_subway_cbd`, `nearest_subway_ada` | — | +3 |
| Leakage removed | — | `estimated_revenue_l365d`, `estimated_occupancy_l365d` | −2 |

**Final clean model performance (November 2025 hold-out test):**

| Metric | Value |
|---|---|
| R² (log scale) | **0.915** |
| CV R² (5-fold, mean ± std) | **0.929 ± 0.003** |
| Test MAE ($) | **$45** |
| CV MAE (mean) | **$36** |
| Median absolute error | **$20** |
| MdAPE | **10.4%** |
| Predictions within 20% of true price | **77%** |

---

## Known Limitations and Validity Assessment

### L1 — Static feature matrix (minor, quantified)

The feature matrix was computed from a single November 2025 snapshot. Dynamic features — primarily `number_of_reviews`, `number_of_reviews_ltm`, and `reviews_per_month` — reflect the listing's state as of November. For training rows from April–October, these counts include reviews written between that month and November, making them mildly future-looking relative to the training labels.

**Why this is not a blocker:**
- The feature values are **constant across all months for the same listing** (verified: zero within-listing variation). The model cannot use them to distinguish April from October — it must rely on `month` and `seasonal_index` for temporal variation.
- Removing all three review-count features changes test R² by **0.002** (0.9154 → 0.9135) and MdAPE by **0.4pp** (10.4% → 10.8%). The effect is negligible.
- The **November test evaluation is fully clean**: features and prices are contemporaneous, so the test metrics are not inflated by this issue.

**Report framing:** "The feature matrix was computed from the November 2025 snapshot. Time-invariant features (distance, amenity flags, property type) are unaffected. Dynamic review-count features are mildly future-looking for training months April–October; however, their predictive contribution is minimal (ΔR² < 0.002) and the held-out November test evaluation is unaffected."

---

### L2 — Leaking features (fixed)

`estimated_revenue_l365d` and `estimated_occupancy_l365d` were identified as exact functions of the target variable (`revenue / occupancy = price` for 100% of rows). Both were removed before training. See Decision 7.

---

### L3 — Placeholder price contamination (fixed)

85 listings priced at $50,000/night (host calendar-blocking behavior) were removed via a $2,000/night cap. See Decision 6.

---

### L4 — Single hold-out month

The test set is one month (November 2025, 4,073 rows). While the 5-fold CV on Apr–Oct is stable (R² = 0.929 ± 0.003), a multi-month hold-out would give stronger external validity. This is constrained by data availability — we have no prices for December 2025 onwards (platform change). We report CV metrics alongside the test metric to mitigate this.

---

### L5 — December extrapolation (out-of-distribution)

December is not in the training data. Predicting December prices requires the model to extrapolate beyond its training range for the `month` feature. The seasonal index (1.032) provides a soft signal, but the prediction should be flagged as estimated. See Decision 5.

---

## What this enables for the progress report

- A concrete description of methodology with justified design choices (7 decisions)
- Actual model results: R² = 0.915, MdAPE = 10.4%, 77% within 20% (clean, no leakage)
- An honest treatment of all validity concerns — fixed where possible, quantified where not
- A defensible fallback from SARIMAX with cited limitations
