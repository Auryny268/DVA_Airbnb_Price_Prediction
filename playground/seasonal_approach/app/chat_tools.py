"""Tool definitions and handlers for the Airbnb chat app.

Each tool returns a JSON-serializable dict. Handlers dedupe by listing `id`
where appropriate; `seasonal_trend` is the only tool that uses the full stacked
frame. Results are capped so they don't blow the context window.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from chat_data import AMENITY_COLS, load_data, listing_level

MAX_ROWS_RETURNED = 20

# ---------- Tool schemas (sent to Claude) ----------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "query_listings",
        "description": (
            "Filter listings by attribute criteria and return the match count plus "
            "summary statistics (price, rating, revenue). Use for questions like "
            "'how many 2-bedroom places in Brooklyn under $200?'. Deduplicates by listing id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "borough": {"type": "string", "enum": ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]},
                "property_type": {"type": "string", "enum": ["Entire home", "Private room", "Shared room", "Hotel room", "Other"]},
                "room_type": {"type": "string", "enum": ["Entire home", "Private room", "Shared room", "Hotel room"]},
                "min_bedrooms": {"type": "number"},
                "max_bedrooms": {"type": "number"},
                "min_accommodates": {"type": "integer"},
                "price_min": {"type": "number"},
                "price_max": {"type": "number"},
                "min_rating": {"type": "number", "description": "Minimum review_scores_rating (0-5)"},
                "is_superhost": {"type": "boolean"},
                "is_instant_bookable": {"type": "boolean"},
                "required_amenities": {
                    "type": "array",
                    "items": {"type": "string", "enum": [c.replace("has_", "") for c in AMENITY_COLS]},
                    "description": "Amenity names (without 'has_' prefix) that MUST be present",
                },
            },
        },
    },
    {
        "name": "get_listing_details",
        "description": "Return all features of a single listing by its id.",
        "input_schema": {
            "type": "object",
            "properties": {"listing_id": {"type": "integer"}},
            "required": ["listing_id"],
        },
    },
    {
        "name": "describe_column",
        "description": (
            "Return summary statistics (count, mean, std, min, quartiles, max) for a numeric column, "
            "or value counts for a categorical column. Optionally filter by borough/property_type first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "Column name, e.g. price_numeric, review_scores_rating"},
                "borough": {"type": "string"},
                "property_type": {"type": "string"},
            },
            "required": ["column"],
        },
    },
    {
        "name": "group_aggregate",
        "description": (
            "Group listings by a dimension and compute an aggregate of a metric. "
            "Use for 'average X by Y' questions. Dedupes by listing id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {"type": "string", "enum": ["borough", "property_type", "room_type", "is_superhost"]},
                "metric": {"type": "string", "description": "Numeric column, e.g. price_numeric, estimated_revenue_l365d"},
                "aggregation": {"type": "string", "enum": ["mean", "median", "sum", "count", "min", "max"], "default": "mean"},
            },
            "required": ["group_by", "metric"],
        },
    },
    {
        "name": "top_n_listings",
        "description": "Return the top N listings ranked by a metric, with optional filters. Default N=10.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "description": "Column to rank by, e.g. estimated_revenue_l365d, review_scores_rating"},
                "n": {"type": "integer", "default": 5, "maximum": 20},
                "ascending": {"type": "boolean", "default": False, "description": "Set true for bottom-N instead of top-N"},
                "borough": {"type": "string"},
                "property_type": {"type": "string"},
                "price_max": {"type": "number"},
                "min_rating": {"type": "number"},
            },
            "required": ["metric"],
        },
    },
    {
        "name": "correlate",
        "description": "Compute Pearson and Spearman correlation between two numeric columns across listings (deduplicated).",
        "input_schema": {
            "type": "object",
            "properties": {
                "col_a": {"type": "string"},
                "col_b": {"type": "string"},
                "borough": {"type": "string"},
            },
            "required": ["col_a", "col_b"],
        },
    },
    {
        "name": "amenity_impact",
        "description": (
            "Compare listings that have a given amenity vs those that don't on price, revenue, rating, "
            "and occupancy. Returns deltas and sample sizes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amenity": {
                    "type": "string",
                    "enum": [c.replace("has_", "") for c in AMENITY_COLS],
                },
                "borough": {"type": "string"},
            },
            "required": ["amenity"],
        },
    },
    {
        "name": "seasonal_trend",
        "description": (
            "Return a metric's value for each month (4-11) using the full stacked data. "
            "Use for 'how does X change over the year?' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "description": "e.g. price_numeric, estimated_occupancy_l365d, seasonal_index"},
                "aggregation": {"type": "string", "enum": ["mean", "median"], "default": "mean"},
                "borough": {"type": "string"},
            },
            "required": ["metric"],
        },
    },
    {
        "name": "location_search",
        "description": (
            "Filter listings by geographic/proximity criteria (subway distance, subway line density, "
            "LIRR access, POI count, tourist POI count, crime level) and return summary stats plus "
            "optional top-N listings. Use for 'near the subway', 'in a safe area', 'walkable to attractions'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_subway_km": {"type": "number", "description": "Maximum distance to nearest subway in km (0.5 ~ 'near the subway')"},
                "min_subway_lines_500m": {"type": "integer", "description": "Minimum number of subway lines within 500m (2+ ~ 'well-connected')"},
                "require_ada_subway": {"type": "boolean", "description": "Only listings where the nearest subway is ADA-accessible"},
                "max_lirr_km": {"type": "number"},
                "min_lirr_count_1mi": {"type": "integer"},
                "min_poi_500m": {"type": "integer"},
                "min_tourist_poi_500m": {"type": "integer", "description": "10+ ~ 'touristy'"},
                "max_crime_500m": {"type": "integer", "description": "Use the dataset's 25th percentile for 'safe area'"},
                "borough": {"type": "string"},
                "price_max": {"type": "number"},
                "min_rating": {"type": "number"},
                "room_type": {"type": "string"},
                "sort_by": {"type": "string", "description": "Column to rank matches by (for top_n)", "default": "estimated_revenue_l365d"},
                "top_n": {"type": "integer", "description": "How many matching listings to return (0 for none)", "default": 0, "maximum": 15},
            },
        },
    },
    {
        "name": "location_impact",
        "description": (
            "Split listings on a proximity threshold (e.g. within vs beyond 0.5 km of a subway) and "
            "compare price, revenue, rating, and occupancy. Answers 'how much does X location feature cost/add?'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "feature": {
                    "type": "string",
                    "enum": ["dist_subway_km", "subway_lines_500m", "dist_lirr_km", "poi_count_500m", "tourist_poi_500m", "crime_count_500m"],
                },
                "threshold": {"type": "number", "description": "Split point; listings <= threshold go in 'group_low', > threshold in 'group_high'"},
                "borough": {"type": "string"},
            },
            "required": ["feature", "threshold"],
        },
    },
]


# ---------- Helpers ----------

def _apply_common_filters(df: pd.DataFrame, *, borough=None, property_type=None, room_type=None,
                          price_max=None, price_min=None, min_rating=None) -> pd.DataFrame:
    if borough:
        df = df[df["borough"] == borough]
    if property_type:
        df = df[df["property_type"] == property_type]
    if room_type:
        df = df[df["room_type"] == room_type]
    if price_max is not None:
        df = df[df["price_numeric"] <= price_max]
    if price_min is not None:
        df = df[df["price_numeric"] >= price_min]
    if min_rating is not None:
        df = df[df["review_scores_rating"] >= min_rating]
    return df


def _summary(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"count": 0}
    return {
        "count": int(len(df)),
        "price_median": round(float(df["price_numeric"].median()), 2),
        "price_mean": round(float(df["price_numeric"].mean()), 2),
        "rating_mean": round(float(df["review_scores_rating"].mean()), 2),
        "revenue_median": round(float(df["estimated_revenue_l365d"].median()), 2),
    }


def _listing_row_to_dict(row: pd.Series, extra_cols: list[str] | None = None) -> dict:
    base = {
        "id": int(row["id"]),
        "borough": row["borough"],
        "property_type": row["property_type"],
        "room_type": row["room_type"],
        "bedrooms": float(row["bedrooms"]),
        "price": round(float(row["price_numeric"]), 2),
        "rating": round(float(row["review_scores_rating"]), 2),
        "revenue_l365d": round(float(row["estimated_revenue_l365d"]), 2),
    }
    for c in extra_cols or []:
        if c in row and c not in base:
            v = row[c]
            base[c] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
    return base


# ---------- Handlers ----------

def query_listings(**kwargs) -> dict:
    df = listing_level()
    df = _apply_common_filters(df, **{k: kwargs.get(k) for k in ["borough", "property_type", "room_type", "price_max", "price_min", "min_rating"]})
    if kwargs.get("min_bedrooms") is not None:
        df = df[df["bedrooms"] >= kwargs["min_bedrooms"]]
    if kwargs.get("max_bedrooms") is not None:
        df = df[df["bedrooms"] <= kwargs["max_bedrooms"]]
    if kwargs.get("min_accommodates") is not None:
        df = df[df["accommodates"] >= kwargs["min_accommodates"]]
    if kwargs.get("is_superhost") is not None:
        df = df[df["is_superhost"] == (1 if kwargs["is_superhost"] else 0)]
    if kwargs.get("is_instant_bookable") is not None:
        df = df[df["is_instant_bookable"] == (1 if kwargs["is_instant_bookable"] else 0)]
    for amenity in kwargs.get("required_amenities", []) or []:
        col = f"has_{amenity}"
        if col in df.columns:
            df = df[df[col] == 1]
    return {"filters_applied": {k: v for k, v in kwargs.items() if v is not None}, "summary": _summary(df)}


def get_listing_details(listing_id: int) -> dict:
    df = listing_level()
    row = df[df["id"] == listing_id]
    if len(row) == 0:
        return {"error": f"Listing {listing_id} not found"}
    r = row.iloc[0]
    amenities = [c.replace("has_", "") for c in AMENITY_COLS if r[c] == 1]
    return {
        "id": int(r["id"]),
        "borough": r["borough"],
        "property_type": r["property_type"],
        "room_type": r["room_type"],
        "accommodates": int(r["accommodates"]),
        "bedrooms": float(r["bedrooms"]),
        "beds": float(r["beds"]),
        "bathrooms": float(r["bathrooms"]),
        "price_avg": round(float(r["price_numeric"]), 2),
        "minimum_nights": int(r["minimum_nights"]),
        "availability_365": int(r["availability_365"]),
        "is_superhost": bool(r["is_superhost"]),
        "is_instant_bookable": bool(r["is_instant_bookable"]),
        "host_response_rate": round(float(r["host_response_rate_f"]), 2),
        "number_of_reviews": int(r["number_of_reviews"]),
        "review_scores_rating": round(float(r["review_scores_rating"]), 2),
        "review_scores_cleanliness": round(float(r["review_scores_cleanliness"]), 2),
        "review_scores_location": round(float(r["review_scores_location"]), 2),
        "amenities": amenities,
        "dist_subway_km": round(float(r["dist_subway_km"]), 3),
        "subway_lines_500m": int(r["subway_lines_500m"]),
        "crime_count_500m": int(r["crime_count_500m"]),
        "poi_count_500m": int(r["poi_count_500m"]),
        "estimated_occupancy_l365d": round(float(r["estimated_occupancy_l365d"]), 1),
        "estimated_revenue_l365d": round(float(r["estimated_revenue_l365d"]), 2),
    }


def describe_column(column: str, borough=None, property_type=None) -> dict:
    df = listing_level()
    df = _apply_common_filters(df, borough=borough, property_type=property_type)
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    s = df[column]
    if pd.api.types.is_numeric_dtype(s):
        q = s.quantile([0.25, 0.5, 0.75])
        return {
            "column": column,
            "count": int(s.count()),
            "mean": round(float(s.mean()), 3),
            "std": round(float(s.std()), 3),
            "min": round(float(s.min()), 3),
            "p25": round(float(q.loc[0.25]), 3),
            "median": round(float(q.loc[0.5]), 3),
            "p75": round(float(q.loc[0.75]), 3),
            "max": round(float(s.max()), 3),
        }
    vc = s.value_counts().head(20).to_dict()
    return {"column": column, "unique": int(s.nunique()), "top_values": {str(k): int(v) for k, v in vc.items()}}


def group_aggregate(group_by: str, metric: str, aggregation: str = "mean") -> dict:
    df = listing_level()
    if group_by not in df.columns or metric not in df.columns:
        return {"error": f"Unknown column: {group_by} or {metric}"}
    agg_func = {"mean": "mean", "median": "median", "sum": "sum", "count": "count", "min": "min", "max": "max"}[aggregation]
    result = df.groupby(group_by)[metric].agg(agg_func).round(3).sort_values(ascending=False)
    return {
        "group_by": group_by,
        "metric": metric,
        "aggregation": aggregation,
        "values": {str(k): float(v) for k, v in result.items()},
    }


def top_n_listings(metric: str, n: int = 5, ascending: bool = False, **filters) -> dict:
    df = listing_level()
    if metric not in df.columns:
        return {"error": f"Column '{metric}' not found"}
    df = _apply_common_filters(df, **{k: filters.get(k) for k in ["borough", "property_type", "price_max", "min_rating"]})
    n = min(n, MAX_ROWS_RETURNED)
    top = df.sort_values(metric, ascending=ascending).head(n)
    return {
        "metric": metric,
        "n": int(len(top)),
        "direction": "bottom" if ascending else "top",
        "listings": [_listing_row_to_dict(row, extra_cols=[metric]) for _, row in top.iterrows()],
    }


def correlate(col_a: str, col_b: str, borough=None) -> dict:
    df = listing_level()
    if borough:
        df = df[df["borough"] == borough]
    if col_a not in df.columns or col_b not in df.columns:
        return {"error": "Unknown column"}
    sub = df[[col_a, col_b]].dropna()
    if len(sub) < 3:
        return {"error": "Not enough data"}
    pearson = float(sub[col_a].corr(sub[col_b], method="pearson"))
    # Compute Spearman as Pearson of ranks — avoids a scipy dependency
    spearman = float(sub[col_a].rank().corr(sub[col_b].rank(), method="pearson"))
    return {
        "col_a": col_a,
        "col_b": col_b,
        "n": int(len(sub)),
        "pearson": round(pearson, 4),
        "spearman": round(spearman, 4),
    }


def amenity_impact(amenity: str, borough=None) -> dict:
    col = f"has_{amenity}"
    df = listing_level()
    if borough:
        df = df[df["borough"] == borough]
    if col not in df.columns:
        return {"error": f"Unknown amenity: {amenity}"}
    with_a = df[df[col] == 1]
    without_a = df[df[col] == 0]

    def stats(d: pd.DataFrame) -> dict:
        return {
            "n": int(len(d)),
            "price_mean": round(float(d["price_numeric"].mean()), 2),
            "rating_mean": round(float(d["review_scores_rating"].mean()), 2),
            "occupancy_mean": round(float(d["estimated_occupancy_l365d"].mean()), 1),
            "revenue_mean": round(float(d["estimated_revenue_l365d"].mean()), 2),
        }

    s_with, s_without = stats(with_a), stats(without_a)
    deltas = {k: round(s_with[k] - s_without[k], 2) for k in s_with if k != "n"}
    return {"amenity": amenity, "with": s_with, "without": s_without, "delta_with_minus_without": deltas}


def seasonal_trend(metric: str, aggregation: str = "mean", borough=None) -> dict:
    df = load_data()  # full stacked data
    if borough:
        df = df[df["borough"] == borough]
    if metric not in df.columns:
        return {"error": f"Unknown metric: {metric}"}
    by_month = df.groupby("month")[metric].agg(aggregation).round(3)
    return {
        "metric": metric,
        "aggregation": aggregation,
        "by_month": {int(k): float(v) for k, v in by_month.items()},
    }


def location_search(**kwargs) -> dict:
    df = listing_level()
    filters = {}
    if (v := kwargs.get("max_subway_km")) is not None:
        df = df[df["dist_subway_km"] <= v]; filters["max_subway_km"] = v
    if (v := kwargs.get("min_subway_lines_500m")) is not None:
        df = df[df["subway_lines_500m"] >= v]; filters["min_subway_lines_500m"] = v
    if kwargs.get("require_ada_subway"):
        df = df[df["nearest_subway_ada"] == 1]; filters["require_ada_subway"] = True
    if (v := kwargs.get("max_lirr_km")) is not None:
        df = df[df["dist_lirr_km"] <= v]; filters["max_lirr_km"] = v
    if (v := kwargs.get("min_lirr_count_1mi")) is not None:
        df = df[df["lirr_count_1mi"] >= v]; filters["min_lirr_count_1mi"] = v
    if (v := kwargs.get("min_poi_500m")) is not None:
        df = df[df["poi_count_500m"] >= v]; filters["min_poi_500m"] = v
    if (v := kwargs.get("min_tourist_poi_500m")) is not None:
        df = df[df["tourist_poi_500m"] >= v]; filters["min_tourist_poi_500m"] = v
    if (v := kwargs.get("max_crime_500m")) is not None:
        df = df[df["crime_count_500m"] <= v]; filters["max_crime_500m"] = v

    df = _apply_common_filters(df, **{k: kwargs.get(k) for k in ["borough", "room_type", "price_max", "min_rating"]})

    result: dict[str, Any] = {"filters_applied": filters, "summary": _summary(df)}
    top_n = int(kwargs.get("top_n") or 0)
    if top_n > 0 and len(df) > 0:
        sort_by = kwargs.get("sort_by") or "estimated_revenue_l365d"
        if sort_by in df.columns:
            top = df.sort_values(sort_by, ascending=False).head(min(top_n, 15))
            result["top_matches"] = [
                _listing_row_to_dict(r, extra_cols=["dist_subway_km", "subway_lines_500m", "crime_count_500m", sort_by])
                for _, r in top.iterrows()
            ]
    return result


def location_impact(feature: str, threshold: float, borough=None) -> dict:
    df = listing_level()
    if borough:
        df = df[df["borough"] == borough]
    if feature not in df.columns:
        return {"error": f"Unknown feature: {feature}"}
    low = df[df[feature] <= threshold]
    high = df[df[feature] > threshold]

    def stats(d: pd.DataFrame) -> dict:
        if len(d) == 0:
            return {"n": 0}
        return {
            "n": int(len(d)),
            "price_mean": round(float(d["price_numeric"].mean()), 2),
            "rating_mean": round(float(d["review_scores_rating"].mean()), 2),
            "occupancy_mean": round(float(d["estimated_occupancy_l365d"].mean()), 1),
            "revenue_mean": round(float(d["estimated_revenue_l365d"].mean()), 2),
        }

    return {
        "feature": feature,
        "threshold": threshold,
        "group_low": {"condition": f"{feature} <= {threshold}", **stats(low)},
        "group_high": {"condition": f"{feature} > {threshold}", **stats(high)},
    }


# ---------- Dispatcher ----------

HANDLERS = {
    "query_listings": query_listings,
    "get_listing_details": get_listing_details,
    "describe_column": describe_column,
    "group_aggregate": group_aggregate,
    "top_n_listings": top_n_listings,
    "correlate": correlate,
    "amenity_impact": amenity_impact,
    "seasonal_trend": seasonal_trend,
    "location_search": location_search,
    "location_impact": location_impact,
}


def execute_tool(name: str, tool_input: dict) -> dict:
    """Dispatch a tool call. Catches exceptions and returns them as error strings."""
    handler = HANDLERS.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return handler(**tool_input)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
