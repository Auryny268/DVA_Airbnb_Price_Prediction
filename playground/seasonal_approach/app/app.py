"""
NYC Airbnb Price Predictor - Interactive Dashboard
Team 005 - CSE 6242 Spring 2026

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

from chat_tab import render_chat_tab

st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT     = "#FF5A5F"
TEAL       = "#00A699"
GRAY       = "#767676"
HIGHLIGHT  = "#6366F1"  # indigo – neutral, no good/bad directional meaning

DATA = Path(__file__).parent.parent / "data"

@st.cache_data
def load_data():
    preds    = pd.read_csv(DATA / "all_predictions.csv")
    fm       = pd.read_csv(DATA / "feature_matrix.csv")
    shap_pl  = pd.read_csv(DATA / "shap_per_listing.csv")
    shap_sum = pd.read_csv(DATA / "shap_summary.csv")
    si       = pd.read_csv(DATA / "seasonal_index.csv")

    PRICE_CAP = 2000
    fm    = fm[fm["price_numeric"] <= PRICE_CAP].copy()
    preds = preds[preds["pred_price"] <= PRICE_CAP].copy()

    ROOM_MAP = {0: "Shared room", 1: "Hotel room", 2: "Private room", 3: "Entire home/apt"}
    fm["room_type_label"]    = fm["room_type_ord"].map(ROOM_MAP)
    preds["room_type_label"] = preds["room_type_ord"].map(ROOM_MAP)

    ALL_MONTHS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                  7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    si["month_name"] = si["month"].map(ALL_MONTHS)

    topics_df      = pd.read_csv(DATA / "bertopic_avg_topics_new_2.csv")
    listing_topics = pd.read_csv(DATA / "bertopic_avg_listing_topics_new_2.csv")
    sentiment_df   = pd.read_csv(DATA / "listings_with_determinants.csv")

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
MONTH_OPTS  = list(range(4, 12))
MONTH_NAMES = {4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Airbnb_Logo_B%C3%A9lo.svg/2560px-Airbnb_Logo_B%C3%A9lo.svg.png",
    width=120,
)
st.sidebar.title("Filters")
st.sidebar.caption("Use these filters to narrow the listings shown across all charts.")

selected_month = st.sidebar.selectbox(
    "Prediction Month",
    options=MONTH_OPTS,
    index=0,
    format_func=lambda m: MONTH_NAMES[m],
    help="Controls which month the model generates price predictions for. Prices are predicted using seasonal demand signals, not separate monthly data snapshots.",
)
selected_boroughs = st.sidebar.multiselect(
    "Borough", BOROUGHS, default=BOROUGHS,
    help="Filters the map, charts, and listings to only the selected boroughs.",
)

relevant_nbhds = sorted(
    fm[fm["neighbourhood_group_cleansed"].isin(selected_boroughs)]["neighbourhood_cleansed"].unique()
)
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
    "Narrow by Neighborhood", options=relevant_nbhds, default=[],
    placeholder="All neighborhoods shown",
    help="Optionally zoom into specific neighborhoods within the selected boroughs.",
)
selected_rooms = st.sidebar.multiselect(
    "Room Type", ROOM_TYPES, default=ROOM_TYPES,
    help="Filters listings by room type across all charts.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** XGBoost - R2=0.915 - MdAPE=10.4% (Nov test set)  \n"
    "**Data:** InsideAirbnb NYC - Apr-Nov 2025  \n"
    "**Team 005** - CSE 6242 Spring 2026"
)

# ── Filters ───────────────────────────────────────────────────────────────────
mask = (
    (preds["month"] == selected_month) &
    (preds["neighbourhood_group_cleansed"].isin(selected_boroughs)) &
    (preds["room_type_label"].isin(selected_rooms))
)
if selected_nbhds:
    mask &= preds["neighbourhood_cleansed"].isin(selected_nbhds)
map_df = preds[mask].copy()

fm_mask = (
    fm["neighbourhood_group_cleansed"].isin(selected_boroughs) &
    fm["room_type_label"].isin(selected_rooms)
)
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

# ── KPI row ───────────────────────────────────────────────────────────────────
med_pred     = map_df["pred_price"].median()
med_actual   = map_df["price_numeric"].median()
delta_val    = med_pred - med_actual
seasonal_idx = si[si["month"] == selected_month]["seasonal_index"].values[0]

static_col, spacer, dynamic_col = st.columns([2, 0.08, 2])

with static_col:
    sc1, sc2 = st.columns(2)
    sc1.metric(
        "Listings Shown", f"{len(map_df):,}",
        help="Number of listings matching your borough, neighborhood, and room type filters. Does not change with the month selector.",
    )
    sc2.metric(
        "Median Listed Price", f"${med_actual:.0f}",
        help="The median calendar price scraped from Airbnb in November 2025. Static - does not vary by month.",
    )

with spacer:
    st.markdown(
        "<div style='border-left:2px solid #E0E0E0; height:80px; margin-top:8px'></div>",
        unsafe_allow_html=True,
    )

with dynamic_col:
    dc1, dc2 = st.columns(2)
    if delta_val > 0:
        delta_label = f"+${abs(delta_val):.0f} seasonal premium"
        delta_help  = (
            f"The model predicts {MONTH_NAMES[selected_month]} commands a ${abs(delta_val):.0f} premium "
            "over the calendar listed price. This reflects higher seasonal demand - not a model error. "
            "The listed price is a static snapshot; the predicted price adjusts for the month."
        )
    else:
        delta_label = f"-${abs(delta_val):.0f} seasonal discount"
        delta_help  = (
            f"The model predicts {MONTH_NAMES[selected_month]} sits ${abs(delta_val):.0f} below "
            "the calendar listed price. This reflects lower seasonal demand for this month - not a model error. "
            "The listed price is a static snapshot; the predicted price adjusts for the month."
        )

    dc1.metric(
        f"Seasonal Price Estimate ({MONTH_NAMES[selected_month]})", f"${med_pred:.0f}",
        delta=delta_label, delta_color="off", help=delta_help,
    )
    dc2.metric(
        "Seasonal Demand Index", f"{seasonal_idx:.3f}x",
        help="A demand multiplier derived from historical review volume. Above 1.0 means higher-than-average demand for this month. Used by the model as a seasonal pricing signal.",
    )

st.caption("Left two tiles reflect your filter selections only. Right two tiles also respond to the month selector.")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Price Map",
    "🕸️ Listing Deep Dive",
    "📊 SHAP Feature Importance",
    "📋 Data Explorer",
    "💬 Ask the Data",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 - Price Map
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if "map_view" not in st.session_state:
        st.session_state.map_view = {"lat": 40.730, "lon": -73.935, "zoom": 10}

    col_map, col_right = st.columns([2.5, 1.5])

    with col_map:
        map_df["log_pred_price"] = np.log(map_df["pred_price"])
        tick_vals = [50, 100, 200, 500, 1000, 2000]
        fig_map = px.scatter_map(
            map_df,
            lat="latitude", lon="longitude",
            color="log_pred_price",
            color_continuous_scale="icefire",
            zoom=st.session_state.map_view["zoom"],
            center={"lat": st.session_state.map_view["lat"],
                    "lon": st.session_state.map_view["lon"]},
            opacity=0.75, size_max=8,
            hover_data={
                "latitude": False, "longitude": False, "log_pred_price": False,
                "pred_price": ":$.0f", "price_numeric": ":$.0f",
                "neighbourhood_cleansed": True, "room_type_label": True,
            },
            custom_data=["neighbourhood_cleansed", "room_type_label", "pred_price", "price_numeric"],
        )
        fig_map.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "Predicted: <b>$%{customdata[2]:.0f}</b><br>"
                "Listed Price: $%{customdata[3]:.0f}<extra></extra>"
            )
        )
        fig_map.update_layout(
            map_style="carto-positron",
            coloraxis_colorbar=dict(
                title="Predicted ($)",
                tickvals=np.log(tick_vals),
                ticktext=[f"${p}" for p in tick_vals],
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=450,
        )
        if selected_nbhds:
            fig_map.update_layout(map_zoom=12)

        with st.container(border=True):
            st.markdown(f"**Predicted Nightly Prices - {MONTH_NAMES[selected_month]} 2025**")
            st.caption(
                "Each dot represents a listing, colored by its predicted nightly price. "
                "Check the legend to see which colors represent higher and lower prices. "
                "Hover over any dot to see the listing's neighborhood, room type, and predicted vs. listed price."
            )
            map_event = st.plotly_chart(fig_map, use_container_width=True, key="nyc_map", on_select="rerun")
        if map_event and "relayout" in map_event:
            rel = map_event["relayout"]
            if "mapbox.center" in rel:
                st.session_state.map_view["lat"] = rel["mapbox.center"]["lat"]
                st.session_state.map_view["lon"] = rel["mapbox.center"]["lon"]
            if "mapbox.zoom" in rel:
                st.session_state.map_view["zoom"] = rel["mapbox.zoom"]

    with col_right:
        # Plot 1: Price Distribution
        mean_val     = map_df["pred_price"].mean()
        med_val      = map_df["pred_price"].median()
        mean_color   = "#00A699"
        median_color = "#F97316"

        fig_hist = px.histogram(
            map_df, x="pred_price", nbins=40,
            color_discrete_sequence=[ACCENT],
            labels={"pred_price": "Predicted ($)"},
        )
        fig_hist.add_vline(x=mean_val, line_dash="dash", line_color=mean_color, line_width=2)
        fig_hist.add_vline(x=med_val, line_dash="solid", line_color=median_color, line_width=2)
        fig_hist.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
            line=dict(color=mean_color, dash="dash"), name=f"Mean: ${mean_val:.0f}"))
        fig_hist.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
            line=dict(color=median_color, dash="solid"), name=f"Median: ${med_val:.0f}"))
        fig_hist.update_layout(
            height=200, margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            legend=dict(x=0.98, y=0.95, xanchor="right", yanchor="top",
                        orientation="v", font=dict(size=9)),
            bargap=0.05, xaxis_tickprefix="$",
        )
        with st.container(border=True):
            st.markdown("**Price Distribution**")
            st.caption(
                "Taller bars mean more listings cluster around that price range. "
                "Use the sidebar filters to see how the distribution shifts."
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("")

        # Plot 2: Seasonal Demand Index
        si_filtered = si[(si["month"] >= 4) & (si["month"] <= 11)].copy()
        bar_colors  = [HIGHLIGHT if m == selected_month else "#E0E0E0" for m in si_filtered["month"]]

        fig_si = go.Figure(go.Bar(
            x=si_filtered["month_name"], y=si_filtered["seasonal_index"],
            marker_color=bar_colors,
            hovertemplate="Month: %{x}<br>Index: %{y:.3f}<extra></extra>",
        ))
        fig_si.add_hline(y=1.0, line_dash="dash", line_color=GRAY)
        fig_si.update_layout(
            height=200, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_tickangle=-45, showlegend=False, yaxis_title="Index",
            yaxis=dict(range=[si_filtered["seasonal_index"].min() * 0.9,
                               si_filtered["seasonal_index"].max() * 1.1]),
        )
        with st.container(border=True):
            st.markdown("**Seasonal Demand Index**")
            st.caption(
                "Values above 1.0 mean higher-than-average demand. "
                "The highlighted bar is your selected month."
            )
            st.plotly_chart(fig_si, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 - Listing Deep Dive
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Listing Deep Dive: Reviews, Topics and Sentiment")
    st.caption(
        "Select up to 5 listings to compare review scores, "
        "dominant review topics, and sentiment breakdown."
    )

    sel_col1, sel_col2 = st.columns([1, 2])
    with sel_col1:
        neigh_opts = sorted(fm_filtered["neighbourhood_cleansed"].unique())
        sel_neigh  = st.selectbox("Filter by Neighborhood", ["All"] + neigh_opts)
    with sel_col2:
        neigh_fm = (
            fm_filtered[fm_filtered["neighbourhood_cleansed"] == sel_neigh]
            if sel_neigh != "All" else fm_filtered
        )
        display_cols = ["id", "neighbourhood_cleansed", "room_type_label", "price_numeric"] + REVIEW_COLS
        display_cols = [c for c in display_cols if c in neigh_fm.columns]
        sample = neigh_fm[display_cols].dropna(subset=REVIEW_COLS).head(200)

        # Show truncated IDs in the selector but store full IDs as values
        all_ids   = sample["id"].astype(str).tolist()
        sel_ids   = st.multiselect(
            "Select Listing IDs to Plot (up to 5)",
            options=all_ids,
            default=all_ids[:3],
            max_selections=5,
        )

    if sel_ids:
        sel_rows = fm[fm["id"].astype(str).isin(sel_ids)]
        colors   = [ACCENT, TEAL, "#FFB400", "#8B5CF6", "#10B981"]

        # Short display ID: last 6 digits. Full ID kept for hover.
        def short(full_id: str) -> str:
            return f"...{full_id[-6:]}"

        left_col, divider_col, right_col = st.columns([1, 0.04, 1])

        with divider_col:
            st.markdown(
                "<div style='border-left:2px solid #E0E0E0; min-height:1000px; margin-top:8px'></div>",
                unsafe_allow_html=True,
            )

        # LEFT: Radar + Review Scores Table
        with left_col:
            st.markdown("#### Review Score Radar")
            st.caption(
                "Each axis represents a review score category. "
                "The outer edge is the maximum score. "
                "Shaded polygons show each listing's relative strengths across all dimensions."
            )
            fig_spider = go.Figure()
            for i, (_, row) in enumerate(sel_rows.iterrows()):
                full_id       = str(row["id"])
                scores        = [row[c] for c in REVIEW_COLS]
                scores_closed = scores + [scores[0]]
                labels_closed = REVIEW_LABELS + [REVIEW_LABELS[0]]
                fig_spider.add_trace(go.Scatterpolar(
                    r=scores_closed, theta=labels_closed,
                    fill="toself",
                    fillcolor=colors[i % len(colors)],
                    opacity=0.25,
                    line=dict(color=colors[i % len(colors)], width=2),
                    name=f"{short(full_id)} - {row.get('neighbourhood_cleansed', '')[:18]}",
                    hovertemplate=(
                        f"Full ID: {full_id}<br>"
                        f"Neighborhood: {row.get('neighbourhood_cleansed', '')}<br>"
                        "%{theta}: %{r:.2f}<extra></extra>"
                    ),
                ))
            fig_spider.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[3.5, 5], tickfont=dict(color="black")),
                    angularaxis=dict(tickfont_size=11),
                ),
                height=400, margin=dict(t=20, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, font=dict(size=10)),
            )
            st.plotly_chart(fig_spider, use_container_width=True)

            st.markdown("#### Review Scores")
            review_display = sel_rows[
                ["id", "neighbourhood_cleansed", "room_type_label", "price_numeric"] + REVIEW_COLS
            ].copy()
            review_display["full_id"]    = review_display["id"].astype(str)
            review_display["id_display"] = review_display["full_id"].apply(short)
            st.dataframe(
                review_display
                .rename(columns={
                    "id_display": "Listing",
                    "full_id": "Full ID",
                    "neighbourhood_cleansed": "Neighborhood",
                    "room_type_label": "Room Type",
                    "price_numeric": "Listed ($)",
                })
                .drop(columns=["id"])
                .set_index("Listing")
                .round(2),
                use_container_width=True,
            )

        # RIGHT: Word Clouds + Sentiment
        with right_col:
            st.markdown("#### Top Review Topics - Word Clouds")
            topic_words_map = {int(r["Topic"]): r["words"] for _, r in topics_df.iterrows()}

            sel_int_ids       = [int(x) for x in sel_ids]
            listing_topic_rows = listing_topics[listing_topics["listing_id"].isin(sel_int_ids)]

            wc_data = []
            for lid_str in sel_ids:
                lid    = int(lid_str)
                lt_row = listing_topic_rows[listing_topic_rows["listing_id"] == lid]
                if lt_row.empty:
                    continue
                lt         = lt_row.iloc[0]
                top_topics = [(int(lt[f"top{r}_topic"]), float(lt[f"top{r}_prob"])) for r in range(1, 4)]
                word_freq  = {}
                for tid, prob in top_topics:
                    for rank, w in enumerate(topic_words_map.get(tid, [])):
                        word_freq[w] = word_freq.get(w, 0) + prob * (1.0 / (rank + 1))
                if not word_freq:
                    continue
                wc = WordCloud(
                    width=380, height=200,
                    background_color="white", colormap="viridis",
                    max_words=30, prefer_horizontal=0.7, relative_scaling=0.5,
                ).generate_from_frequencies(word_freq)
                top_labels = [f"T{tid} ({prob:.0%})" for tid, prob in top_topics if prob > 0.001]
                wc_data.append((lid_str, short(lid_str), ', '.join(top_labels), wc.to_array()))

            if not wc_data:
                st.info("No topic data available for selected listings.")
            else:
                for full_id, short_id, topic_label, img in wc_data:
                    st.markdown(
                        f'<span title="{full_id}" style="cursor:help;font-size:13px;color:#555">'
                        f'<b>{short_id}</b></span>'
                        f'<span style="font-size:13px;color:#555">: {topic_label}</span>',
                        unsafe_allow_html=True,
                    )
                    st.image(img, use_container_width=True)

            st.markdown("#### Sentiment Breakdown")
            sel_frame = pd.DataFrame({"listing_id": sel_int_ids})
            sent_rows = sel_frame.merge(
                sentiment_df[["listing_id", "positive_ratio", "neutral_ratio",
                              "negative_ratio", "mean_sentiment"]],
                on="listing_id", how="left",
            )
            sent_rows["listing_id"] = sent_rows["listing_id"].astype(str)
            missing_sent = sent_rows[sent_rows["positive_ratio"].isna()]["listing_id"].tolist()
            if missing_sent:
                st.caption(f"No sentiment data for listing(s): {', '.join(short(x) for x in missing_sent)}")
            sent_rows = sent_rows.fillna({"positive_ratio": 0, "neutral_ratio": 0, "negative_ratio": 0})

            sent_long = sent_rows.melt(
                id_vars=["listing_id", "mean_sentiment"],
                value_vars=["positive_ratio", "neutral_ratio", "negative_ratio"],
                var_name="sentiment", value_name="ratio",
            )
            label_map = {"positive_ratio": "Positive", "neutral_ratio": "Neutral", "negative_ratio": "Negative"}
            color_map = {"Positive": "#10B981", "Neutral": "#94A3B8", "Negative": ACCENT}
            sent_long["sentiment"] = sent_long["sentiment"].map(label_map)
            sent_long["label"]     = sent_long["listing_id"].apply(short)
            sent_long["full_id"]   = sent_long["listing_id"]
            sent_rows["label"]     = sent_rows["listing_id"].apply(short)
            sent_rows["full_id"]   = sent_rows["listing_id"]

            fig_sent = go.Figure()
            for stype, color in color_map.items():
                subset = sent_long[sent_long["sentiment"] == stype]
                fig_sent.add_trace(go.Bar(
                    x=subset["label"], y=subset["ratio"],
                    name=stype, marker_color=color,
                    customdata=subset["full_id"],
                    hovertemplate=f"Full ID: %{{customdata}}<br>{stype}: %{{y:.1%}}<extra></extra>",
                ))
            fig_sent.update_layout(
                barmode="stack", bargap=0.4, height=280,
                margin=dict(l=10, r=10, t=30, b=40),
                yaxis=dict(title="Proportion", tickformat=".0%", range=[0, 1], fixedrange=True),
                xaxis=dict(title="Listing", fixedrange=True),
                legend=dict(title="Sentiment", orientation="h",
                            yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                dragmode=False,
            )
            for _, srow in sent_rows.iterrows():
                fig_sent.add_annotation(
                    x=srow["label"], y=1.05,
                    text=f"avg: {srow['mean_sentiment']:.2f}" if pd.notna(srow["mean_sentiment"]) else "N/A",
                    showarrow=False, font=dict(size=10, color=GRAY),
                )
            st.plotly_chart(fig_sent, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Select at least one listing above.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 - SHAP Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### SHAP Feature Importance")

    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    shap_boro  = c1.selectbox("Borough", ["All"] + BOROUGHS, key="shap_boro")
    shap_room  = c2.selectbox("Room Type", ["All"] + ROOM_TYPES, key="shap_room")
    shap_top_n = c3.slider("Top N Features", 5, 30, 15)
    shap_mode  = c4.radio("View", ["Absolute Impact", "Directional Impact"], horizontal=False)

    shap_meta = shap_pl.merge(
        fm[["id", "neighbourhood_group_cleansed", "room_type_label"]], on="id", how="left"
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

    mean_abs      = shap_subset[feat_cols].abs().mean().sort_values(ascending=False)
    top_feats     = mean_abs.head(shap_top_n)
    top_feat_names = top_feats.index.tolist()

    n_label = f"{len(shap_subset):,} listings"
    if shap_boro != "All" or shap_room != "All":
        n_label += " (filtered)"

    def feat_category(name):
        if name in ("month", "seasonal_index"):               return "Seasonal"
        if name.startswith("boro_"):                          return "Borough"
        if name.startswith("prop_"):                          return "Property type"
        if "subway" in name or "lirr" in name:                return "Transit"
        if "poi" in name or "crime" in name:                  return "Spatial"
        if name in ("is_superhost", "host_response_rate_f",
                    "host_acceptance_rate_f", "host_response_time_ord",
                    "host_identity_verified_f", "has_license"): return "Host"
        if name.startswith("has_") or name == "amenity_count": return "Amenities"
        if "review" in name:                                  return "Reviews"
        return "Property"

    CAT_COLORS = {
        "Seasonal": "#FF5A5F", "Borough": "#8B5CF6", "Property type": "#00A699",
        "Transit": "#F59E0B", "Spatial": "#10B981", "Host": "#3B82F6",
        "Amenities": "#EC4899", "Reviews": "#6B7280", "Property": "#1F2937",
    }

    if shap_mode == "Absolute Impact":
        st.caption(
            "Mean absolute SHAP value across listings. Longer bars mean the feature has more influence "
            "on the model's price predictions overall, regardless of direction."
        )
        top_feats_sorted = top_feats.sort_values()
        cats   = [feat_category(f) for f in top_feats_sorted.index]
        colors = [CAT_COLORS[c] for c in cats]

        fig_shap = go.Figure(go.Bar(
            x=top_feats_sorted.values, y=top_feats_sorted.index,
            orientation="h", marker_color=colors,
            text=[f"{v:.4f}" for v in top_feats_sorted.values],
            textposition="outside", textfont_size=10, showlegend=False,
        ))
        for cat, col in CAT_COLORS.items():
            if cat in cats:
                fig_shap.add_trace(go.Bar(x=[None], y=[None], orientation="h",
                                          marker_color=col, name=cat, showlegend=True))
        fig_shap.update_layout(
            xaxis_title="Mean |SHAP| (impact on log price)", yaxis_title="",
            height=max(420, shap_top_n * 24),
            margin=dict(l=10, r=80, t=20, b=40),
            yaxis=dict(tickfont_size=11),
            legend_title_text="Category",
            legend=dict(orientation="v", x=1.02),
        )
        st.caption(f"Computed on {n_label}")
        st.plotly_chart(fig_shap, use_container_width=True)

    else:
        st.caption(
            "Each dot is one listing. Right of zero means the feature pushed that listing's price up; "
            "left of zero means it pushed it down. "
            "Dot color shows whether the listing had a high (red) or low (blue) value for that feature."
        )

        # Build normalized feature value lookup using full fm range for consistent color scale
        feat_val_lookup = shap_subset[["id"]].copy().reset_index(drop=True)
        for feat in top_feat_names:
            if feat in fm.columns:
                all_vals = fm[feat].dropna().astype(float)
                mn, mx   = float(all_vals.min()), float(all_vals.max())
                subset_vals = (
                    feat_val_lookup[["id"]]
                    .merge(fm[["id", feat]], on="id", how="left")[feat]
                    .astype(float)
                )
                if mx > mn:
                    feat_val_lookup[feat + "_norm"] = ((subset_vals - mn) / (mx - mn)).clip(0, 1).values
                else:
                    feat_val_lookup[feat + "_norm"] = 0.5
            else:
                feat_val_lookup[feat + "_norm"] = 0.5

        # Merge SHAP values with normalized feature values
        beeswarm_df = shap_subset[["id"] + top_feat_names].merge(feat_val_lookup, on="id", how="left")

        SAMPLE_N = 1500
        if len(beeswarm_df) > SAMPLE_N:
            beeswarm_df = beeswarm_df.sample(SAMPLE_N, random_state=42).reset_index(drop=True)

        def _beeswarm_y(vals, spread=0.42, n_bins=50):
            """Density-aware vertical jitter: denser x-bins get wider vertical spread."""
            n = len(vals)
            if n == 0:
                return np.zeros(0)
            hist, edges = np.histogram(vals, bins=min(n_bins, max(n, 2)))
            max_h = max(hist.max(), 1)
            bin_idx = np.clip(np.digitize(vals, edges[:-1]) - 1, 0, len(hist) - 1)
            jitter = np.zeros(n)
            rng = np.random.default_rng(42)
            for b in range(len(hist)):
                mask = bin_idx == b
                k = int(mask.sum())
                if k == 0:
                    continue
                half = spread * (k / max_h)
                pos = np.linspace(-half, half, k)
                rng.shuffle(pos)
                jitter[mask] = pos
            return jitter

        top_feats_ordered = list(reversed(top_feat_names))
        x_all, y_all, color_all = [], [], []
        for i, feat in enumerate(top_feats_ordered):
            shap_vals  = beeswarm_df[feat].values.astype(float)
            feat_norms = beeswarm_df[feat + "_norm"].fillna(0.5).values.astype(float)
            jitter     = _beeswarm_y(shap_vals)
            x_all.extend(shap_vals.tolist())
            y_all.extend((i + jitter).tolist())
            color_all.extend(feat_norms.tolist())

        fig_bee = go.Figure()
        fig_bee.add_trace(go.Scatter(
            x=x_all, y=y_all, mode="markers",
            marker=dict(
                color=color_all,
                colorscale=[[0, "#3B82F6"], [0.5, "#9333EA"], [1, "#EF4444"]],
                size=4, opacity=0.7,
                colorbar=dict(
                    title=dict(text="Feature<br>Value", side="right"),
                    tickvals=[0, 1], ticktext=["Low", "High"],
                    lenmode="fraction", len=0.4, y=0.5,
                ),
                showscale=True,
            ),
            hovertemplate="SHAP: %{x:.4f}<extra></extra>",
            showlegend=False,
        ))
        fig_bee.add_vline(x=0, line_color="white", line_width=1.5)
        fig_bee.update_layout(
            xaxis=dict(title="SHAP value (impact on predicted log-price)", zeroline=False),
            yaxis=dict(
                tickvals=list(range(len(top_feats_ordered))),
                ticktext=top_feats_ordered,
                tickfont_size=11,
            ),
            height=max(500, shap_top_n * 32),
            margin=dict(l=10, r=100, t=20, b=40),
            plot_bgcolor="#111111",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_gridcolor="#333333",
            yaxis_gridcolor="#333333",
        )
        st.caption(f"Computed on {n_label} (sampled to {min(SAMPLE_N, len(beeswarm_df))} for display)")
        st.plotly_chart(fig_bee, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 - Data Explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### Data Explorer")
    st.caption(
        "Browse the full feature dataset for your current filter selection. "
        "Download as CSV for further analysis."
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
        .rename(columns={"price_numeric": "Listed Price ($)",
                         "neighbourhood_group_cleansed": "Borough"})
        .sort_values("Listed Price ($)", ascending=False)
        .reset_index(drop=True),
        use_container_width=True, height=420,
    )

    st.markdown("**Summary Statistics** (filtered selection)")
    numeric_cols = ["price_numeric", "accommodates", "bedrooms", "review_scores_rating",
                    "dist_subway_km", "poi_count_500m", "amenity_count"]
    numeric_cols = [c for c in numeric_cols if c in fm_filtered.columns]
    st.dataframe(fm_filtered[numeric_cols].describe().round(2), use_container_width=True)

    csv_bytes = fm_filtered[show_cols].to_csv(index=False).encode()
    st.download_button(
        label="Download filtered data as CSV",
        data=csv_bytes,
        file_name="nyc_airbnb_filtered.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 - Ask the Data (Claude chat)
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    render_chat_tab()
