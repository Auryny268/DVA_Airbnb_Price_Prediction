"""Data loader for the chat tab.

Ported from the airbnb-chat repo. Points at the existing seasonal_approach
dataset so we don't duplicate the 8 MB CSV.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).parent.parent / "data" / "stacked_features.csv"

BOROUGH_COLS = [
    "boro_Bronx",
    "boro_Brooklyn",
    "boro_Manhattan",
    "boro_Queens",
    "boro_Staten Island",
]
PROPERTY_COLS = [
    "prop_Entire home",
    "prop_Hotel room",
    "prop_Other",
    "prop_Private room",
    "prop_Shared room",
]
AMENITY_COLS = [
    "has_wifi",
    "has_kitchen",
    "has_ac",
    "has_gym",
    "has_parking",
    "has_pool",
    "has_washer",
    "has_dryer",
    "has_elevator",
    "has_doorman",
    "has_pets_allowed",
]
ROOM_TYPE_MAP = {1: "Shared room", 2: "Private room", 3: "Entire home", 4: "Hotel room"}

NUMERIC_COLS = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "amenity_count",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "host_response_rate_f",
    "host_acceptance_rate_f",
    "dist_subway_km",
    "subway_lines_500m",
    "dist_lirr_km",
    "lirr_count_1mi",
    "poi_count_500m",
    "poi_count_1km",
    "tourist_poi_500m",
    "crime_count_500m",
    "estimated_occupancy_l365d",
    "estimated_revenue_l365d",
    "price_numeric",
    "log_price",
    "seasonal_index",
]


@st.cache_data(show_spinner="Loading Airbnb data...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["borough"] = df[BOROUGH_COLS].idxmax(axis=1).str.replace("boro_", "", regex=False)
    df["property_type"] = (
        df[PROPERTY_COLS].idxmax(axis=1).str.replace("prop_", "", regex=False)
    )
    df["room_type"] = df["room_type_ord"].map(ROOM_TYPE_MAP).fillna("Unknown")
    return df


@st.cache_data
def listing_level() -> pd.DataFrame:
    """One row per listing — averages price across months, keeps other features as-is."""
    df = load_data()
    agg = df.groupby("id", as_index=False).agg(
        price_numeric=("price_numeric", "mean"),
        log_price=("log_price", "mean"),
    )
    base = df.drop_duplicates("id", keep="first").drop(columns=["price_numeric", "log_price", "month", "seasonal_index"])
    return base.merge(agg, on="id")


SCHEMA_SUMMARY = """
DATASET: stacked_features.csv — 4,014 unique NYC Airbnb listings × 8 months (April–November) = 28,244 rows.

Column groups (all numeric unless stated):
- Identity: id (listing ID)
- Listing: accommodates, bedrooms, beds, bathrooms, room_type_ord (1=Shared,2=Private,3=Entire,4=Hotel), amenity_count, minimum_nights, availability_365
- Reviews: number_of_reviews, number_of_reviews_ltm, reviews_per_month, review_scores_rating (0-5), review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value
- Host flags (0/1): is_superhost, host_identity_verified_f, is_instant_bookable, has_license
- Host rates (0-1): host_response_rate_f, host_acceptance_rate_f, host_response_time_ord
- Amenity flags (0/1): has_wifi, has_kitchen, has_ac, has_gym, has_parking, has_pool, has_washer, has_dryer, has_elevator, has_doorman, has_pets_allowed
- Transit: dist_subway_km, near_subway (0/1 flag, ~within 0.5km), subway_lines_500m, nearest_subway_cbd, nearest_subway_ada (accessible), dist_lirr_km, lirr_count_1mi
- POI/safety: poi_count_500m, poi_count_1km, tourist_poi_500m, crime_count_500m
- Economics: estimated_occupancy_l365d (nights/year), estimated_revenue_l365d (USD/year), price_numeric (USD/night), log_price
- Categorical (friendly, added): borough (Bronx/Brooklyn/Manhattan/Queens/Staten Island), property_type (Entire home/Private room/Shared room/Hotel room/Other), room_type
- One-hots (bool): boro_*, prop_*
- Temporal: month (4-11), seasonal_index

Semantic defaults the assistant should assume unless the user says otherwise:
- "near the subway" → max_subway_km = 0.5 (or use near_subway == 1)
- "well-connected" → subway_lines_500m >= 2
- "safe area" → crime_count_500m below the 25th percentile of the dataset
- "touristy" → tourist_poi_500m >= 10

Important: rows are stacked by month. For listing-level questions (how many, top listings, attribute comparisons), the tools deduplicate by id automatically. For temporal questions, use seasonal_trend.
"""
