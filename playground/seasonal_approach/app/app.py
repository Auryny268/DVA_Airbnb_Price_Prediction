"""
NYC Airbnb Price Predictor — Interactive Dashboard
Team 005 · CSE 6242 Spring 2026

Run: streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#FF5A5F"
TEAL   = "#00A699"
GRAY   = "#767676"

# ── Data loading ─────────────────────────────────────────────────────────────
DATA = Path(__file__).parent.parent / "data"

@st.cache_data
def load_data():
    preds    = pd.read_csv(DATA / "all_predictions.csv")
    fm       = pd.read_csv(DATA / "feature_matrix.csv")
    shap_pl  = pd.read_csv(DATA / "shap_per_listing.csv")
    shap_sum = pd.read_csv(DATA / "shap_summary.csv")
    si       = pd.read_csv(DATA / "seasonal_index.csv")

    PRICE_CAP = 2000
    fm   = fm[fm["price_numeric"] <= PRICE_CAP].copy()
    preds = preds[preds["pred_price"] <= PRICE_CAP].copy()

    ROOM_MAP = {0:"Shared room", 1:"Hotel room", 2:"Private room", 3:"Entire home/apt"}
    fm["room_type_label"] = fm["room_type_ord"].map(ROOM_MAP)

    MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    preds["room_type_label"] = preds["room_type_ord"].map(ROOM_MAP)
    si["month_name"] = si["month"].map(MONTH_NAMES)

    return preds, fm, shap_pl, shap_sum, si

preds, fm, shap_pl, shap_sum, si = load_data()

REVIEW_COLS = [
    "review_scores_rating", "review_scores_accuracy",
    "review_scores_cleanliness", "review_scores_checkin",
    "review_scores_communication", "review_scores_location",
    "review_scores_value",
]
REVIEW_LABELS = ["Overall", "Accuracy", "Cleanliness",
                 "Check-in", "Communication", "Location", "Value"]

BOROUGHS    = sorted(fm["neighbourhood_group_cleansed"].unique())
ROOM_TYPES  = sorted(fm["room_type_label"].dropna().unique())
<<<<<<< HEAD
MONTH_OPTS = list(range(4, 12)) # April to November only
MONTH_NAMES = {4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov"}
=======
MONTH_OPTS  = list(range(1, 13))
MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
>>>>>>> 0bcfa6eca1db40f3a3f32a9c0497be8dd1c76f6c

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Airbnb_Logo_B%C3%A9lo.svg/2560px-Airbnb_Logo_B%C3%A9lo.svg.png", width=120)
st.sidebar.title("Filters")

selected_month = st.sidebar.selectbox(
    "Month",
    options=MONTH_OPTS,
<<<<<<< HEAD
    index=0,   # April default
=======
    index=10,   # November default
>>>>>>> 0bcfa6eca1db40f3a3f32a9c0497be8dd1c76f6c
    format_func=lambda m: MONTH_NAMES[m],
)
selected_boroughs = st.sidebar.multiselect(
    "Borough", BOROUGHS, default=BOROUGHS
)
<<<<<<< HEAD

# Get neighborhoods only for the selected boroughs
relevant_nbhds = sorted(fm[fm["neighbourhood_group_cleansed"].isin(selected_boroughs)]["neighbourhood_cleansed"].unique())

# Helper: Select All / Clear All
col1, col2 = st.sidebar.columns(2)
select_all = col1.button("Select All")
clear_all  = col2.button("Clear All")

if "neigh_defaults" not in st.session_state:
    st.session_state.neigh_defaults = []

if select_all:
    st.session_state.neigh_defaults = relevant_nbhds
if clear_all:
    st.session_state.neigh_defaults = []

selected_nbhds = st.sidebar.multiselect(
    "Specific Neighborhoods", 
    options=relevant_nbhds,
    default=[], # Default to empty so it doesn't clutter the map unless chosen
    placeholder="All neighborhoods in selected boroughs"
)

=======
>>>>>>> 0bcfa6eca1db40f3a3f32a9c0497be8dd1c76f6c
selected_rooms = st.sidebar.multiselect(
    "Room type", ROOM_TYPES, default=ROOM_TYPES
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** XGBoost · R²=0.915 · MdAPE=10.4%  \n"
    "**Data:** InsideAirbnb NYC · Apr–Nov 2025  \n"
    "**Team 005** · CSE 6242 Spring 2026"
)

# ── Apply filters ─────────────────────────────────────────────────────────────
<<<<<<< HEAD
mask = (
    (preds["month"] == selected_month) &
    (preds["neighbourhood_group_cleansed"].isin(selected_boroughs)) &
    (preds["room_type_label"].isin(selected_rooms))
)

# If the user has picked specific neighborhoods, filter further. 
# If they left it empty, we show all neighborhoods within the selected boroughs.
if selected_nbhds:
    mask &= preds["neighbourhood_cleansed"].isin(selected_nbhds)

map_df = preds[mask].copy()

# Sync the feature matrix filter as well
fm_mask = (fm["neighbourhood_group_cleansed"].isin(selected_boroughs)) & \
          (fm["room_type_label"].isin(selected_rooms))
if selected_nbhds:
    fm_mask &= fm["neighbourhood_cleansed"].isin(selected_nbhds)

fm_filtered = fm[fm_mask].copy()
=======
map_df = preds[
    (preds["month"] == selected_month) &
    (preds["neighbourhood_group_cleansed"].isin(selected_boroughs)) &
    (preds["room_type_label"].isin(selected_rooms))
].copy()

fm_filtered = fm[
    (fm["neighbourhood_group_cleansed"].isin(selected_boroughs)) &
    (fm["room_type_label"].isin(selected_rooms))
].copy()

>>>>>>> 0bcfa6eca1db40f3a3f32a9c0497be8dd1c76f6c
# ── Header ────────────────────────────────────────────────────────────────────
st.title("🗽 NYC Airbnb Price Predictor")
st.caption(
    f"Showing **{len(map_df):,}** listings · "
    f"**{MONTH_NAMES[selected_month]}** · "
    f"{', '.join(selected_boroughs) if len(selected_boroughs) < 5 else 'All boroughs'}"
)

# KPI row
med_pred = map_df["pred_price"].median()
med_actual = map_df["price_numeric"].median()
<<<<<<< HEAD
delta_val = med_pred - med_actual
k1, k2, k3, k4 = st.columns(4)
k1.metric("Listings shown",     f"{len(map_df):,}")
k2.metric("Median predicted",   f"${med_pred:.0f}")
# Update label to show it's specific to the selected month
# TODO: Ask Rishikesh abt how Median actual works
k3.metric(
    label=f"Median Actual ({MONTH_NAMES[selected_month]})", 
    value=f"${med_actual:.0f}",
    delta=f"{delta_val:+.0f} vs Prediction",
    delta_color="inverse" # Red if pred > actual, Green if actual > pred
)
=======
k1, k2, k3, k4 = st.columns(4)
k1.metric("Listings shown",     f"{len(map_df):,}")
k2.metric("Median predicted",   f"${med_pred:.0f}")
k3.metric("Median actual (Nov)",f"${med_actual:.0f}")
>>>>>>> 0bcfa6eca1db40f3a3f32a9c0497be8dd1c76f6c
k4.metric("Seasonal index",     f"{si[si['month']==selected_month]['seasonal_index'].values[0]:.3f}×")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Price Map",
    "🕸️ Listing Spider Chart",
    "📊 SHAP Feature Importance",
    "📋 Data Explorer",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Interactive Price Map
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"Predicted Nightly Prices — {MONTH_NAMES[selected_month]} 2025")

<<<<<<< HEAD
    # Initialize Session State for Camera 
    if "map_view" not in st.session_state:
        st.session_state.map_view = {
            "lat": 40.730,
            "lon": -73.935,
            "zoom": 10
        }
    
    col_map, col_hist = st.columns([3, 1])

    with col_map:
        # Create a log-transformed column for the color mapping only
        map_df["log_pred_price"] = np.log(map_df["pred_price"])
        fig_map = px.scatter_map(
            map_df,
            lat="latitude", lon="longitude",
            color="log_pred_price",     # Map color to the log values
            color_continuous_scale="icefire",
            # Use the values stored in session state
            zoom=st.session_state.map_view["zoom"],
            center={"lat": st.session_state.map_view["lat"], 
                    "lon": st.session_state.map_view["lon"]},
=======
    col_map, col_hist = st.columns([3, 1])

    with col_map:
        fig_map = px.scatter_map(
            map_df,
            lat="latitude", lon="longitude",
            color="pred_price",
            color_continuous_scale="RdYlGn_r",
            zoom=10,
            center={"lat": 40.730, "lon": -73.935},
>>>>>>> 0bcfa6eca1db40f3a3f32a9c0497be8dd1c76f6c
            opacity=0.75,
            size_max=8,
            labels={"pred_price": "Predicted Price ($)"},
            hover_data={
                "latitude": False, "longitude": False,
<<<<<<< HEAD
                "log_pred_price": False,
=======
>>>>>>> 0bcfa6eca1db40f3a3f32a9c0497be8dd1c76f6c
                "pred_price": ":$.0f",
                "price_numeric": ":$.0f",
                "neighbourhood_cleansed": True,
                "room_type_label": True,
            },
            custom_data=["neighbourhood_cleansed","room_type_label",
                         "pred_price","price_numeric"],
        )
<<<<<<< HEAD

        # Manually fix the colorbar ticks so they show $ amounts instead of log decimals
        tick_vals = [50, 100, 200, 500, 1000, 2000]
=======
>>>>>>> 0bcfa6eca1db40f3a3f32a9c0497be8dd1c76f6c
        fig_map.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "Predicted: <b>$%{customdata[2]:.0f}</b><br>"
                "Actual Nov: $%{customdata[3]:.0f}<extra></extra>"
            )
        )
<<<<<<< HEAD

        fig_map.update_layout(
            map_style="carto-positron",
            coloraxis_colorbar=dict(title="Predicted ($)", tickvals=np.log(tick_vals), ticktext=[f"${p}" for p in tick_vals]),
            margin=dict(l=0, r=0, t=0, b=0),
            height=500,
        )
        if selected_nbhds:
            fig_map.update_layout(map_zoom=12) # Zoom in closer for specific neighborhoods
    
        # Catch User Map Interactions 
        # This returns a dictionary of the current state of the chart
        map_event = st.plotly_chart(
            fig_map, 
            use_container_width=True, 
            key="nyc_map",       # Required for state tracking
            on_select="rerun"    # Triggers rerun when map is moved
        )

        # Update camera state based on user movement
        if map_event and "relayout" in map_event:
            rel = map_event["relayout"]
            if "mapbox.center" in rel:
                st.session_state.map_view["lat"] = rel["mapbox.center"]["lat"]
                st.session_state.map_view["lon"] = rel["mapbox.center"]["lon"]
            if "mapbox.zoom" in rel:
                st.session_state.map_view["zoom"] = rel["mapbox.zoom"]


    with col_hist:
        st.markdown("**Price Distribution**")

        # Calculate stats
        mean_val = map_df["pred_price"].mean()
        med_val  = map_df["pred_price"].median()
        # Define transparent colors
        mean_color   = "rgba(0, 166, 153, 0)"  
        median_color = "rgba(51, 51, 51, 0)"   

=======
        fig_map.update_layout(
            map_style="carto-positron",
            coloraxis_colorbar=dict(title="Predicted ($)", tickprefix="$"),
            margin=dict(l=0, r=0, t=0, b=0),
            height=500,
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_hist:
        st.markdown("**Price distribution**")
>>>>>>> 0bcfa6eca1db40f3a3f32a9c0497be8dd1c76f6c
        fig_hist = px.histogram(
            map_df, x="pred_price", nbins=40,
            color_discrete_sequence=[ACCENT],
            labels={"pred_price": "Predicted ($)"},
        )
<<<<<<< HEAD

        # 2. Add Mean Line (Dashed Teal)
        fig_hist.add_vline(
            x=mean_val, line_dash="dash", line_color=mean_color, line_width=2
        )
        
        # 3. Add Median Line (Solid Gray)
        fig_hist.add_vline(
            x=med_val, line_dash="solid", line_color=median_color, line_width=2
        )

        # 4. Dummy traces to force a Legend
        fig_hist.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=mean_color, dash="dash"),
            name=f"Mean: ${mean_val:.0f}"
        ))
        fig_hist.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=median_color, dash="solid"),
            name=f"Median: ${med_val:.0f}"
        ))

        fig_hist.update_layout(
            height=220, margin=dict(l=10,r=10,t=10,b=30),
            showlegend=True, legend=dict(
                x=0.98,              # Almost all the way to the right
                y=0.95,              # Near the top
                xanchor="right",     # Anchors the box from its right edge
                yanchor="top",       # Anchors the box from its top edge
                orientation="h",
                font=dict(size=10)
            ), bargap=0.05,
=======
        fig_hist.update_layout(
            height=220, margin=dict(l=10,r=10,t=10,b=30),
            showlegend=False, bargap=0.05,
>>>>>>> 0bcfa6eca1db40f3a3f32a9c0497be8dd1c76f6c
            xaxis_tickprefix="$",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

<<<<<<< HEAD
        st.markdown("**Seasonal index (Apr–Nov)**")
        
        # 1. Filter the seasonal index DataFrame
        si_filtered = si[(si["month"] >= 4) & (si["month"] <= 11)].copy()
        
        # 2. Update bar colors based on filtered data
        bar_colors = [ACCENT if m == selected_month else "#E0E0E0" for m in si_filtered["month"]]
        
        fig_si = go.Figure(go.Bar(
            x=si_filtered["month_name"], 
            y=si_filtered["seasonal_index"],
            marker_color=bar_colors,
            hovertemplate="Month: %{x}<br>Index: %{y:.3f}<extra></extra>"
        ))
        
        fig_si.add_hline(y=1.0, line_dash="dash", line_color=GRAY)
        
        fig_si.update_layout(
            height=220, 
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis_tickangle=-45, 
            showlegend=False,
            yaxis_title="Index",
            # Ensure the Y-axis has some padding so the 1.0 line is clear
            yaxis=dict(range=[si_filtered["seasonal_index"].min() * 0.9, 
                             si_filtered["seasonal_index"].max() * 1.1])
=======
        st.markdown("**Seasonal index**")
        bar_colors = [ACCENT if m == selected_month else "#E0E0E0" for m in si["month"]]
        fig_si = go.Figure(go.Bar(
            x=si["month_name"], y=si["seasonal_index"],
            marker_color=bar_colors,
        ))
        fig_si.add_hline(y=1.0, line_dash="dash", line_color=GRAY)
        fig_si.update_layout(
            height=220, margin=dict(l=10,r=10,t=10,b=30),
            xaxis_tickangle=-45, showlegend=False,
            yaxis_title="Index",
>>>>>>> 0bcfa6eca1db40f3a3f32a9c0497be8dd1c76f6c
        )
        st.plotly_chart(fig_si, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
<<<<<<< HEAD
# TAB 2 — Spider Chart (Aggregate vs Selection)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Review Score Radar")
    st.caption("Benchmark specific listings against the broader market average.")

    # 1. User Choice: Mean vs Median for the "General" view
    agg_method = st.radio("Aggregate View:", ["Mean", "Median"], horizontal=True)
    
    if agg_method == "Mean":
        baseline_scores = fm_filtered[REVIEW_COLS].mean().tolist()
    else:
        baseline_scores = fm_filtered[REVIEW_COLS].median().tolist()

    # 2. Selection for comparison
    sel_ids = st.multiselect(
        "Select specific listing IDs to overlay (Max 3)",
        options=fm_filtered["id"].astype(str).tolist(),
        max_selections=3,
        help="Compare individual properties to the neighborhood baseline."
    )

    fig_spider = go.Figure()

    # Add the "General" Baseline trace
    fig_spider.add_trace(go.Scatterpolar(
        r=baseline_scores + [baseline_scores[0]],
        theta=REVIEW_LABELS + [REVIEW_LABELS[0]],
        mode='lines',
        line=dict(color=GRAY, width=3, dash='dot'),
        fill='toself',
        fillcolor='rgba(118, 118, 118, 0.1)', # Subtle fill for the "area" of the city
        name=f"Filtered Group {agg_method}"
    ))

    # Add Selected Listings
    colors = [ACCENT, TEAL, "#FFB400"]
    if sel_ids:
        sel_rows = fm_filtered[fm_filtered["id"].astype(str).isin(sel_ids)]
        for i, (_, row) in enumerate(sel_rows.iterrows()):
            scores = [row[c] for c in REVIEW_COLS]
            fig_spider.add_trace(go.Scatterpolar(
                r=scores + [scores[0]],
                theta=REVIEW_LABELS + [REVIEW_LABELS[0]],
                fill="toself",
                fillcolor=colors[i % len(colors)],
                opacity=0.2,
                line=dict(color=colors[i % len(colors)], width=2),
                name=f"Listing {row['id']}",
            ))
    
    fig_spider.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, 
            range=[3, 5], 
            tickfont=dict(color="black")), # Focus on the high-end variance
            angularaxis=dict(tickfont_size=12)
        ),
        height=500,
        margin=dict(t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )
    
    st.plotly_chart(fig_spider, use_container_width=True)

    # 3. Data Table (Only shown if listings are selected)
    if sel_ids:
        st.markdown("### Comparison Detail")
        st.dataframe(
            sel_rows[["id", "neighbourhood_cleansed", "price_numeric"] + REVIEW_COLS]
            .rename(columns={"price_numeric": "Price ($)"})
=======
# TAB 2 — Spider Chart per Listing
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Review Score Radar — Per Listing")
    st.caption(
        "Select up to 5 listings to compare their 7 review dimension scores. "
        "Scores are on a 1–5 scale."
    )

    # Let user pick listings by neighbourhood + search
    neigh_opts = sorted(fm_filtered["neighbourhood_cleansed"].unique())
    sel_neigh  = st.selectbox("Filter by neighbourhood", ["All"] + neigh_opts)

    if sel_neigh != "All":
        neigh_fm = fm_filtered[fm_filtered["neighbourhood_cleansed"] == sel_neigh]
    else:
        neigh_fm = fm_filtered

    # Show a sample table to pick from
    display_cols = ["id","neighbourhood_cleansed","room_type_label","price_numeric"] + REVIEW_COLS
    display_cols = [c for c in display_cols if c in neigh_fm.columns]
    sample = neigh_fm[display_cols].dropna(subset=REVIEW_COLS).head(200)

    sel_ids = st.multiselect(
        "Select listing IDs to plot (up to 5)",
        options=sample["id"].astype(str).tolist(),
        default=sample["id"].astype(str).tolist()[:3],
        max_selections=5,
    )

    if sel_ids:
        sel_rows = fm[fm["id"].astype(str).isin(sel_ids)]
        fig_spider = go.Figure()

        colors = [ACCENT, TEAL, "#FFB400", "#8B5CF6", "#10B981"]
        for i, (_, row) in enumerate(sel_rows.iterrows()):
            scores = [row[c] for c in REVIEW_COLS]
            scores_closed = scores + [scores[0]]   # close the polygon
            labels_closed = REVIEW_LABELS + [REVIEW_LABELS[0]]
            fig_spider.add_trace(go.Scatterpolar(
                r=scores_closed,
                theta=labels_closed,
                fill="toself",
                fillcolor=colors[i % len(colors)],
                opacity=0.25,
                line=dict(color=colors[i % len(colors)], width=2),
                name=f"ID {row['id']} · {row.get('neighbourhood_cleansed','')[:20]}",
            ))

        fig_spider.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[3.5, 5])),
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        st.plotly_chart(fig_spider, use_container_width=True)

        # Show scores table
        st.dataframe(
            sel_rows[["id","neighbourhood_cleansed","room_type_label","price_numeric"] + REVIEW_COLS]
            .rename(columns={"price_numeric":"actual_price ($)"})
>>>>>>> 0bcfa6eca1db40f3a3f32a9c0497be8dd1c76f6c
            .set_index("id")
            .round(2),
            use_container_width=True,
        )
    else:
<<<<<<< HEAD
        st.info("💡 **Pro-tip:** Use the multi-select above to overlay specific properties and see how they deviate from the neighborhood's average scores.")
=======
        st.info("Select at least one listing above.")
>>>>>>> 0bcfa6eca1db40f3a3f32a9c0497be8dd1c76f6c

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SHAP Feature Importance (filterable)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("SHAP Feature Importance")
    st.caption(
        "Mean |SHAP| value = average impact of each feature on the predicted log-price. "
        "Filter by borough or room type to see which features matter most in that segment."
    )

    c1, c2, c3 = st.columns(3)
    shap_boro  = c1.selectbox("Borough filter", ["All"] + BOROUGHS, key="shap_boro")
    shap_room  = c2.selectbox("Room type filter", ["All"] + ROOM_TYPES, key="shap_room")
    shap_top_n = c3.slider("Top N features", 5, 30, 20)

    # Merge SHAP per listing with metadata
    shap_meta = shap_pl.merge(
        fm[["id","neighbourhood_group_cleansed","room_type_label"]],
        on="id", how="left"
    )
    shap_subset = shap_meta.copy()
    if shap_boro != "All":
        shap_subset = shap_subset[shap_subset["neighbourhood_group_cleansed"] == shap_boro]
    if shap_room != "All":
        shap_subset = shap_subset[shap_subset["room_type_label"] == shap_room]

    feat_cols = [c for c in shap_pl.columns if c != "id"]
    if len(shap_subset) < 5:
        st.warning("Too few listings match this filter. Showing global importance.")
        shap_subset = shap_meta

    mean_abs = shap_subset[feat_cols].abs().mean().sort_values(ascending=False)
    top_feats = mean_abs.head(shap_top_n).sort_values()

    def feat_category(name):
        if name in ("month","seasonal_index"):             return "Seasonal"
        if name.startswith("boro_"):                       return "Borough"
        if name.startswith("prop_"):                       return "Property type"
        if "subway" in name or "lirr" in name:             return "Transit"
        if "poi" in name or "crime" in name:               return "Spatial"
        if name in ("is_superhost","host_response_rate_f",
                    "host_acceptance_rate_f","host_response_time_ord",
                    "host_identity_verified_f","has_license"): return "Host"
        if name.startswith("has_") or name == "amenity_count": return "Amenities"
        if "review" in name:                               return "Reviews"
        return "Property"

    CAT_COLORS = {
        "Seasonal":"#FF5A5F", "Borough":"#8B5CF6", "Property type":"#00A699",
        "Transit":"#F59E0B", "Spatial":"#10B981", "Host":"#3B82F6",
        "Amenities":"#EC4899", "Reviews":"#6B7280", "Property":"#1F2937",
    }

    cats   = [feat_category(f) for f in top_feats.index]
    colors = [CAT_COLORS[c] for c in cats]

    fig_shap = go.Figure(go.Bar(
        x=top_feats.values,
        y=top_feats.index,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in top_feats.values],
        textposition="outside",
        textfont_size=10,
        showlegend=False,   # legend built from named dummy traces below
    ))
    fig_shap.update_layout(
        xaxis_title="Mean |SHAP| (impact on log price)",
        yaxis_title="",
        height=max(400, shap_top_n * 22),
        margin=dict(l=10, r=60, t=20, b=40),
        yaxis=dict(tickfont_size=11),
    )

    # Legend by color
    for cat, col in CAT_COLORS.items():
        if cat in cats:
            fig_shap.add_trace(go.Bar(
                x=[None], y=[None], orientation="h",
                marker_color=col, name=cat, showlegend=True,
            ))
    fig_shap.update_layout(legend_title_text="Category",
                           legend=dict(orientation="v", x=1.02))

    n_label = f"{len(shap_subset):,} listings"
    if shap_boro != "All" or shap_room != "All":
        n_label += f" (filtered)"
    st.caption(f"Computed on {n_label}")
    st.plotly_chart(fig_shap, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Data Explorer (Power BI substitute / export)
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Data Explorer")
    st.caption(
        "Browse the full feature dataset. Use filters on the left. "
        "Download as CSV for Power BI or further analysis."
    )

    show_cols = [
        "id", "neighbourhood_cleansed", "neighbourhood_group_cleansed",
        "room_type_label", "price_numeric",
        "accommodates", "bedrooms", "beds", "bathrooms",
        "amenity_count", "is_superhost", "review_scores_rating",
        "dist_subway_km", "poi_count_500m",
    ]
    show_cols = [c for c in show_cols if c in fm_filtered.columns]

    st.dataframe(
        fm_filtered[show_cols]
        .rename(columns={"price_numeric":"actual_price ($)",
                         "neighbourhood_group_cleansed":"borough"})
        .sort_values("actual_price ($)", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

    # Summary stats
    st.markdown("**Summary statistics** (filtered selection)")
    numeric_cols = ["price_numeric","accommodates","bedrooms","review_scores_rating",
                    "dist_subway_km","poi_count_500m","amenity_count"]
    numeric_cols = [c for c in numeric_cols if c in fm_filtered.columns]
    st.dataframe(fm_filtered[numeric_cols].describe().round(2), use_container_width=True)

    # Download
    csv_bytes = fm_filtered[show_cols].to_csv(index=False).encode()
    st.download_button(
        label="⬇️ Download filtered data as CSV",
        data=csv_bytes,
        file_name="nyc_airbnb_filtered.csv",
        mime="text/csv",
    )
