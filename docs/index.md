---
layout: default
title: NYC Airbnb Price Prediction & Recommendation
description: DVA Spring 2026 · Team 005 · Georgia Tech
---

# NYC Airbnb Price Prediction & Recommendation

**CSE 6242 · Data and Visual Analytics · Spring 2026 · Team 005**

---

## Overview

We are building a machine learning system to predict Airbnb listing prices across New York City and help both hosts and travelers make smarter decisions. Our system combines structured listing features, NLP-derived sentiment from guest reviews, spatial neighborhood context, and seasonal pricing signals into a unified predictive model with interactive visualizations.

---

## Approach

Our pipeline integrates several techniques:

- **XGBoost** — primary price prediction model using structured listing attributes (bedrooms, amenities, host features, location)
- **Sentiment Analysis** — NLP on guest reviews to capture quality signals not reflected in structured data
- **SARIMAX** — seasonal time-series modeling to encode NYC's temporal pricing patterns as a feature
- **Spatial Features** — neighborhood-level embeddings using Moran Eigenvector Spatial Filtering to control for spatial autocorrelation
- **Interactive Heatmaps** — OpenStreetMap-based visualizations showing price variation across NYC neighborhoods with user-defined filters
- **RAG Chatbot** *(bonus)* — retrieval-augmented generation chatbot for natural-language listing recommendations based on user preferences and reviews

**Dataset:** [InsideAirbnb NYC](http://insideairbnb.com/) — listings, reviews, calendar, and neighborhood geospatial data.

---

## Key Features

| Feature | Description |
|---|---|
| Price Prediction | XGBoost model trained on 100k+ NYC listings |
| Review Sentiment | NLP-derived sentiment scores from guest reviews |
| Seasonal Pricing | SARIMAX-estimated seasonality feature |
| Neighborhood Heatmap | Interactive map of price variation by neighborhood |
| Amenity Filtering | Filter listings by amenities, location, and price range |
| RAG Chatbot | Natural-language Q&A for listing recommendations |

---

## Stakeholders & Impact

**Hosts** can estimate a competitive price for their listing based on features, location, and comparable properties — without relying on costly third-party pricing tools.

**Travelers** can identify which NYC neighborhoods offer the best value for their preferred amenities, and use the RAG chatbot to get personalized, natural-language recommendations based on reviews.

---

## Team

| Name | Initials |
|---|---|
| Rishikesh Donthula | RD |
| Harsh Gupta | HG |
| Bret Jacob | BJ |
| Malhar Jadhav | MJ |
| Nimit Sharma | NS |
| Auryn Yamamura | AY |

---

## Timeline

| Task | Members | Weeks 1–2 | Weeks 3–4 | Weeks 5–6 | Weeks 7–8 |
|---|---|:---:|:---:|:---:|:---:|
| Data Preprocessing | AY, NS | ✓ | ✓ | | |
| NLP Development | MJ, HG | ✓ | ✓ | ✓ | ✓ |
| ML Dev (SARIMAX / XGBoost) | BJ, AY, NS, RD | | ✓ | ✓ | |
| Visualization | BJ, AY, NS, RD | | | ✓ | ✓ |
| Integration & Testing | MJ, HG | | | ✓ | ✓ |
| Final Report | All | | | | ✓ |

**Cost:** ~$0 — all data and ML libraries are open source; compute via GT's ICE-PACE cluster; LLM/RAG uses open-source models.

---

## Milestones

**Midterm:** Data processing pipeline + initial XGBoost model trained on full feature set.

**Final:** Complete XGBoost model with selected features, NYC neighborhood price heatmap with filters, and chatbot interface.

---

## Selected References

1. Hong & Yoo (2020) — Spatial variance in Airbnb pricing using Multiscale GWR
2. Gunter et al. (2020) — Spatial panel data modeling of Airbnb demand in NYC
3. Mahyoub et al. (2023) — Airbnb price prediction using machine learning
4. Alharbi (2023) — Sustainable price prediction with sentiment analysis
5. Islam et al. (2022) — LDA + MESF-XGBoost for Airbnb rental price modeling
6. Panahandeh et al. (2024) — XAI-driven analysis of Airbnb price determinants (Dublin)
7. Gao et al. (2024) — RAG survey for large language models

---

*Georgia Tech · CSE 6242 · Spring 2026*
