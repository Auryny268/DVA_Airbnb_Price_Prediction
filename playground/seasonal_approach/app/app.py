"""
NYC Airbnb Price Predictor — Interactive Dashboard
Team 005 · CSE 6242 Spring 2026

Run: streamlit run app.py
"""

import ast
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path
from wordcloud import WordCloud

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

    # Topic & sentiment data
    topics_df      = pd.read_csv(DATA / "bertopic_avg_topics_new_2.csv")
    listing_topics = pd.read_csv(DATA / "bertopic_avg_listing_topics_new_2.csv")
    sentiment_df   = pd.read_csv(DATA / "listings_with_determinants.csv")

    # Parse Representation column from string list to actual list
    topics_df["words"] = topics_df["Representation"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    return preds, fm, shap_pl, shap_sum, si, topics_df, listing_topics, sentiment_df

preds, fm, shap_pl, shap_sum, si, topics_df, listing_topics, sentiment_df = load_data()

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
MONTH_OPTS = list(range(4, 12)) # April to November only
MONTH_NAMES = {4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Airbnb_Logo_B%C3%A9lo.svg/2560px-Airbnb_Logo_B%C3%A9lo.svg.png", width=120)
st.sidebar.title("Filters")

selected_month = st.sidebar.selectbox(
    "Month",
    options=MONTH_OPTS,
    index=0,   # April default
    format_func=lambda m: MONTH_NAMES[m],
)
selected_boroughs = st.sidebar.multiselect(
    "Borough", BOROUGHS, default=BOROUGHS
)

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
delta_val = med_pred - med_actual
k1, k2, k3, k4 = st.columns(4)
k1.metric("Listings shown",     f"{len(map_df):,}")
k2.metric("Median predicted",   f"${med_pred:.0f}")
k3.metric(
    label=f"Median Actual ({MONTH_NAMES[selected_month]})",
    value=f"${med_actual:.0f}",
    delta=f"{delta_val:+.0f} vs Prediction",
    delta_color="inverse" # Red if pred > actual, Green if actual > pred
)
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
            opacity=0.75,
            size_max=8,
            labels={"pred_price": "Predicted Price ($)"},
            hover_data={
                "latitude": False, "longitude": False,
                "log_pred_price": False,
                "pred_price": ":$.0f",
                "price_numeric": ":$.0f",
                "neighbourhood_cleansed": True,
                "room_type_label": True,
            },
            custom_data=["neighbourhood_cleansed","room_type_label",
                         "pred_price","price_numeric"],
        )

        # Manually fix the colorbar ticks so they show $ amounts instead of log decimals
        tick_vals = [50, 100, 200, 500, 1000, 2000]
        fig_map.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "Predicted: <b>$%{customdata[2]:.0f}</b><br>"
                "Actual Nov: $%{customdata[3]:.0f}<extra></extra>"
            )
        )

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

        fig_hist = px.histogram(
            map_df, x="pred_price", nbins=40,
            color_discrete_sequence=[ACCENT],
            labels={"pred_price": "Predicted ($)"},
        )

        # Add Mean Line (Dashed Teal)
        fig_hist.add_vline(
            x=mean_val, line_dash="dash", line_color=mean_color, line_width=2
        )

        # Add Median Line (Solid Gray)
        fig_hist.add_vline(
            x=med_val, line_dash="solid", line_color=median_color, line_width=2
        )

        # Dummy traces to force a Legend
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
                x=0.98,
                y=0.95,
                xanchor="right",
                yanchor="top",
                orientation="h",
                font=dict(size=10)
            ), bargap=0.05,
            xaxis_tickprefix="$",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("**Seasonal index (Apr–Nov)**")

        # Filter the seasonal index DataFrame to Apr-Nov
        si_filtered = si[(si["month"] >= 4) & (si["month"] <= 11)].copy()

        # Update bar colors based on filtered data
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
            yaxis=dict(range=[si_filtered["seasonal_index"].min() * 0.9,
                             si_filtered["seasonal_index"].max() * 1.1])
        )
        st.plotly_chart(fig_si, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Listing Deep Dive (Spider Chart + Word Clouds + Sentiment)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Listing Deep Dive — Reviews, Topics & Sentiment")
    st.caption(
        "Select up to 5 listings to compare review scores (radar), "
        "dominant review topics (word cloud), and sentiment breakdown."
    )

    # ── Listing selector ────────────────────────────────────────────────
    neigh_opts = sorted(fm_filtered["neighbourhood_cleansed"].unique())
    sel_neigh  = st.selectbox("Filter by neighbourhood", ["All"] + neigh_opts)

    if sel_neigh != "All":
        neigh_fm = fm_filtered[fm_filtered["neighbourhood_cleansed"] == sel_neigh]
    else:
        neigh_fm = fm_filtered

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
        colors = [ACCENT, TEAL, "#FFB400", "#8B5CF6", "#10B981"]

        # ── 1) Review score radar ───────────────────────────────────────
        st.markdown("#### Review Score Radar")
        fig_spider = go.Figure()
        for i, (_, row) in enumerate(sel_rows.iterrows()):
            scores = [row[c] for c in REVIEW_COLS]
            scores_closed = scores + [scores[0]]
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
            polar=dict(
                radialaxis=dict(visible=True, range=[3.5, 5],
                                tickfont=dict(color="black")),
                angularaxis=dict(tickfont_size=12),
            ),
            height=480,
            margin=dict(t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )
        st.plotly_chart(fig_spider, use_container_width=True)

        # ── 2) Word clouds — top 3 topics per listing ──────────────────
        st.markdown("#### Top 3 Review Topics — Word Clouds")
        # Build topic label & word lookup
        topic_label_map = {}
        topic_words_map = {}
        for _, trow in topics_df.iterrows():
            tid = int(trow["Topic"])
            label = trow["Name"]
            # Clean up the label: "0_host_recommend_comfortable_subway" → "host, recommend, comfortable, subway"
            parts = label.split("_", 1)
            if len(parts) > 1:
                topic_label_map[tid] = parts[1].replace("_", ", ")
            else:
                topic_label_map[tid] = label
            topic_words_map[tid] = trow["words"]

        sel_int_ids = [int(x) for x in sel_ids]
        listing_topic_rows = listing_topics[listing_topics["listing_id"].isin(sel_int_ids)]

        # Pre-build word clouds only for listings that have topic data
        wc_data = []  # list of (caption, image_array)
        for lid_str in sel_ids:
            lid = int(lid_str)
            lt_row = listing_topic_rows[listing_topic_rows["listing_id"] == lid]
            if lt_row.empty:
                continue
            lt = lt_row.iloc[0]
            top_topics = []
            for rank in range(1, 4):
                tid = int(lt[f"top{rank}_topic"])
                prob = float(lt[f"top{rank}_prob"])
                top_topics.append((tid, prob))

            word_freq = {}
            for tid, prob in top_topics:
                words = topic_words_map.get(tid, [])
                for rank, w in enumerate(words):
                    weight = prob * (1.0 / (rank + 1))
                    word_freq[w] = word_freq.get(w, 0) + weight

            if not word_freq:
                continue

            wc = WordCloud(
                width=400, height=280,
                background_color="white",
                colormap="viridis",
                max_words=30,
                prefer_horizontal=0.7,
                relative_scaling=0.5,
            ).generate_from_frequencies(word_freq)

            top_labels = [
                f"T{tid} ({prob:.0%})" for tid, prob in top_topics if prob > 0.001
            ]
            caption = f"**Listing {lid}** — {', '.join(top_labels)}"
            wc_data.append((caption, wc.to_array()))

        if not wc_data:
            st.info("No topic data available for selected listings.")
        else:
            # Render in rows of up to 3, only for listings with data
            WC_IMG_WIDTH = 400
            cols_per_row = min(len(wc_data), 3)
            for batch_start in range(0, len(wc_data), cols_per_row):
                batch = wc_data[batch_start : batch_start + cols_per_row]
                wc_cols = st.columns(cols_per_row)
                for col, (caption, img) in zip(wc_cols, batch):
                    with col:
                        st.caption(caption)
                        st.image(img, width=WC_IMG_WIDTH)

        # ── 3) Sentiment breakdown bar ──────────────────────────────────
        st.markdown("#### Sentiment Breakdown")
        # Left join: ensure all selected listings appear, even without sentiment data
        sel_frame = pd.DataFrame({"listing_id": sel_int_ids})
        sent_rows = sel_frame.merge(
            sentiment_df[["listing_id", "positive_ratio", "neutral_ratio", "negative_ratio", "mean_sentiment"]],
            on="listing_id",
            how="left",
        )
        sent_rows["listing_id"] = sent_rows["listing_id"].astype(str)

        # Flag listings with no sentiment data
        missing_sent = sent_rows[sent_rows["positive_ratio"].isna()]["listing_id"].tolist()
        if missing_sent:
            st.caption(f"No sentiment data for listing(s): {', '.join(missing_sent)}")

        # Fill NaN so they still appear in the chart (as empty bars)
        sent_rows = sent_rows.fillna({"positive_ratio": 0, "neutral_ratio": 0, "negative_ratio": 0})

        # Melt to long format for stacked bar
        sent_long = sent_rows.melt(
            id_vars=["listing_id", "mean_sentiment"],
            value_vars=["positive_ratio", "neutral_ratio", "negative_ratio"],
            var_name="sentiment",
            value_name="ratio",
        )
        label_map = {
            "positive_ratio": "Positive",
            "neutral_ratio": "Neutral",
            "negative_ratio": "Negative",
        }
        color_map = {
            "Positive": "#10B981",
            "Neutral":  "#94A3B8",
            "Negative": ACCENT,
        }
        sent_long["sentiment"] = sent_long["sentiment"].map(label_map)

        # Use short labels for x-axis to keep bars wide
        sent_long["label"] = "ID " + sent_long["listing_id"].astype(str).str[-6:]
        sent_rows["label"] = "ID " + sent_rows["listing_id"].astype(str).str[-6:]

        fig_sent = go.Figure()
        for sentiment_type, color in color_map.items():
            subset = sent_long[sent_long["sentiment"] == sentiment_type]
            fig_sent.add_trace(go.Bar(
                x=subset["label"],
                y=subset["ratio"],
                name=sentiment_type,
                marker_color=color,
            ))
        fig_sent.update_layout(
            barmode="stack",
            bargap=0.5,
            height=350,
            margin=dict(l=10, r=10, t=30, b=40),
            yaxis=dict(title="Proportion", tickformat=".0%", range=[0, 1], fixedrange=True),
            xaxis=dict(title="Listing ID", fixedrange=True),
            legend=dict(title="Sentiment", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            dragmode=False,
        )
        # Add mean_sentiment annotation on each bar group
        for _, srow in sent_rows.iterrows():
            if pd.notna(srow["mean_sentiment"]):
                fig_sent.add_annotation(
                    x=srow["label"],
                    y=1.05,
                    text=f"avg: {srow['mean_sentiment']:.2f}",
                    showarrow=False,
                    font=dict(size=11, color=GRAY),
                )
            else:
                fig_sent.add_annotation(
                    x=srow["label"],
                    y=1.05,
                    text="N/A",
                    showarrow=False,
                    font=dict(size=11, color=GRAY),
                )
        st.plotly_chart(fig_sent, use_container_width=True, config={"displayModeBar": False})

        # ── Review Scores Table ─────────────────────────────────────────
        st.markdown("#### Review Scores")
        st.dataframe(
            sel_rows[["id","neighbourhood_cleansed","room_type_label","price_numeric"] + REVIEW_COLS]
            .rename(columns={"price_numeric": "actual_price ($)"})
            .set_index("id")
            .round(2),
            use_container_width=True,
        )
    else:
        st.info("Select at least one listing above.")

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
