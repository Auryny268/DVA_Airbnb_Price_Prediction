"""
BERTopic — Approach B: Per-comment topic assignment, then aggregate per listing.

Runs BERTopic on individual comments (streamed in batches to manage memory),
assigns each comment a topic, then aggregates topic proportions per listing.

This preserves ALL text and does NOT dilute minority topics.

Outputs:
  - data/bertopic_comments_topics.csv            : topic ID, top words, count
  - data/bertopic_comments_listing_topics.csv    : per-listing topic proportions
  - data/bertopic_comments_model/                : saved model directory
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.lower().strip())


def main():
    parser = argparse.ArgumentParser(
        description="Per-comment BERTopic, aggregate per listing")
    parser.add_argument("--reviews-csv", required=True)
    parser.add_argument("--num-topics", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--min-topic-size", type=int, default=100,
                        help="Higher than default — millions of docs")
    parser.add_argument("--sample-size", type=int, default=500_000,
                        help="Sample size for fitting BERTopic (memory)")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-n-components", type=int, default=5)
    args = parser.parse_args()

    lang = None if args.lang == "all" else args.lang

    # 1. Load embedding model
    log.info("Loading embedding model: %s", args.embedding_model)
    embed_model = SentenceTransformer(args.embedding_model)

    # 2. First pass: collect all comments and listing_ids
    log.info("Pass 1: Reading and filtering comments ...")
    all_listing_ids = []
    all_comments = []
    for chunk in pd.read_csv(
        args.reviews_csv,
        usecols=["listing_id", "clean_comments", "language"],
        chunksize=args.chunksize, dtype=str,
    ):
        if lang:
            chunk = chunk[chunk["language"] == lang]
        chunk = chunk.dropna(subset=["clean_comments"])
        for lid, text in zip(chunk["listing_id"], chunk["clean_comments"]):
            cleaned = clean_text(text)
            if cleaned:
                all_listing_ids.append(lid)
                all_comments.append(cleaned)
        log.info("  %s comments collected ...", f"{len(all_comments):,}")

    total_comments = len(all_comments)
    log.info("Total comments: %s", f"{total_comments:,}")

    # 3. Embed all comments in batches
    log.info("Embedding all comments ...")
    all_embeddings = embed_model.encode(
        all_comments,
        batch_size=args.embedding_batch_size,
        show_progress_bar=True,
    )
    log.info("Embeddings shape: %s", all_embeddings.shape)

    # 4. Fit BERTopic on a sample (full dataset may not fit in UMAP/HDBSCAN memory)
    if total_comments > args.sample_size:
        log.info("Sampling %s comments for model fitting ...",
                 f"{args.sample_size:,}")
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(total_comments, args.sample_size, replace=False)
        sample_comments = [all_comments[i] for i in sample_idx]
        sample_embeddings = all_embeddings[sample_idx]
    else:
        sample_comments = all_comments
        sample_embeddings = all_embeddings
        sample_idx = None

    umap_model = UMAP(
        n_neighbors=args.umap_n_neighbors,
        n_components=args.umap_n_components,
        min_dist=0.0, metric="cosine", random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_topic_size,
        metric="euclidean", prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words="english", min_df=10, max_df=0.5, ngram_range=(1, 2),
    )

    log.info("Fitting BERTopic on %s comments ...", f"{len(sample_comments):,}")
    topic_model = BERTopic(
        embedding_model=embed_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=args.num_topics,
        calculate_probabilities=False,  # skip probs during fit (memory)
        verbose=True,
    )
    topic_model.fit(sample_comments, sample_embeddings)

    n_topics = len(topic_model.get_topic_info()) - 1  # exclude -1
    log.info("Discovered %d topics", n_topics)

    # 5. Save topic info
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(DATA_DIR / "bertopic_comments_topics.csv", index=False)

    print("\n=== Discovered Topics ===")
    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            continue
        print(f"  Topic {row['Topic']} ({row['Count']} docs): {row['Name']}")

    # 6. Transform ALL comments to get topic assignments
    log.info("Assigning topics to all %s comments ...", f"{total_comments:,}")
    # Process in batches to manage memory
    batch_size = 100_000
    all_topics = []
    for start in range(0, total_comments, batch_size):
        end = min(start + batch_size, total_comments)
        batch_comments = all_comments[start:end]
        batch_embeddings = all_embeddings[start:end]
        batch_topics, _ = topic_model.transform(batch_comments, batch_embeddings)
        all_topics.extend(batch_topics)
        log.info("  assigned %s / %s", f"{end:,}", f"{total_comments:,}")

    # 7. Aggregate topic proportions per listing
    log.info("Aggregating topic proportions per listing ...")
    comment_df = pd.DataFrame({
        "listing_id": all_listing_ids,
        "topic": all_topics,
    })

    # Count topic occurrences per listing
    topic_counts = (
        comment_df.groupby(["listing_id", "topic"])
        .size()
        .unstack(fill_value=0)
    )
    # Convert to proportions
    topic_proportions = topic_counts.div(topic_counts.sum(axis=1), axis=0)

    # Rename columns
    topic_proportions.columns = [f"topic_{c}_prop" for c in topic_proportions.columns]
    topic_proportions = topic_proportions.reset_index()

    # Add dominant topic and comment count
    dominant = comment_df.groupby("listing_id")["topic"].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index()
    dominant.columns = ["listing_id", "dominant_topic"]

    counts = comment_df.groupby("listing_id").size().reset_index(name="num_comments")

    listing_df = dominant.merge(counts, on="listing_id").merge(
        topic_proportions, on="listing_id"
    )

    listing_path = DATA_DIR / "bertopic_comments_listing_topics.csv"
    listing_df.to_csv(listing_path, index=False)
    log.info("Per-listing topics saved to %s (%s listings)",
             listing_path, f"{len(listing_df):,}")

    # 8. Save model
    model_dir = DATA_DIR / "bertopic_comments_model"
    topic_model.save(str(model_dir), serialization="safetensors",
                     save_ctfidf=True, save_embedding_model=args.embedding_model)
    log.info("Model saved to %s", model_dir)

    # 9. Summary
    outliers = sum(1 for t in all_topics if t == -1)
    print(f"\nOutlier comments: {outliers:,} ({outliers/total_comments*100:.1f}%)")
    print(f"Topics: {n_topics} | Comments: {total_comments:,} | "
          f"Listings: {len(listing_df):,}")


if __name__ == "__main__":
    main()
