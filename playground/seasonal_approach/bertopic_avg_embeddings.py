"""
BERTopic — Approach A: Per-comment embedding averaging.

Embeds each comment individually (no truncation), averages embeddings
per listing, then runs BERTopic on listing-level averaged embeddings.

Outputs:
  - data/bertopic_avg_topics_new.csv            : topic ID, top words, count
  - data/bertopic_avg_listing_topics_new.csv    : per-listing topic probabilities
  - data/bertopic_avg_model_new/                : saved model directory
"""

import argparse
import logging
import re
from collections import defaultdict
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

# Host names that dominate topic labels — filter from CountVectorizer
HOST_NAME_STOPS = {
    "carlos", "jose", "alicia", "evelyn", "daniel", "david", "maria", "juan",
    "jorge", "luis", "miguel", "ana", "pedro", "ricardo", "antonio", "marco",
    "jessica", "jennifer", "brian", "kevin", "jason", "ryan", "eric", "sean",
    "patrick", "nicole", "ashley", "amanda", "stephanie", "rachel", "lauren",
    "matthew", "andrew", "robert", "william", "richard", "thomas", "steven",
    "timothy", "christopher", "jonathan", "peter", "paul", "george", "edward",
    "henry", "frank", "ray", "roger", "larry", "terry", "keith", "scott",
    "greg", "dennis", "carl", "wayne", "eugene", "ralph", "russell", "louis",
    "philip", "howard", "barry", "fred", "arthur", "albert", "gerald",
    "samuel", "lawrence", "benjamin", "bruce", "harry", "adam", "douglas",
    "leonard", "ernest", "vincent", "phillip", "bobby", "johnny", "billy",
    "albert", "willie", "randy", "howard", "eugene", "carlos", "jose",
    # Airbnb generic
    "airbnb", "bnb", "host", "hosts", "guest", "guests", "stay", "stayed", "staying",
    "place", "apartment", "room", "house", "home", "listing", "rental",
    "bed", "bedroom", "bathroom", "kitchen", "building",
    # Location generic
    "brooklyn", "manhattan", "queens", "bronx", "york", "nyc", "city",
    "street", "avenue", "block", "neighborhood", "area",
    # Functional
    "get", "got", "make", "made", "take", "took", "come", "came", "going",
    "would", "could", "also", "really", "very", "much", "well", "just",
    "even", "back", "still", "right", "thing", "time", "day", "night",
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.lower().strip())


def main():
    parser = argparse.ArgumentParser(
        description="BERTopic with per-comment embedding averaging")
    parser.add_argument("--reviews-csv", required=True)
    parser.add_argument("--num-topics", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--min-topic-size", type=int, default=30)
    parser.add_argument("--min-cluster-size", type=int, default=5,
                        help="HDBSCAN min_cluster_size (lower = more clusters)")
    parser.add_argument("--min-samples", type=int, default=3,
                        help="HDBSCAN min_samples (lower = more permissive)")
    parser.add_argument("--nr-topics", type=int, default=15,
                        help="Merge down to this many topics after fitting")
    parser.add_argument("--reduce-outliers", action=argparse.BooleanOptionalAction,
                        default=True, help="Reassign outliers to nearest topic")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-n-components", type=int, default=5)
    args = parser.parse_args()

    lang = None if args.lang == "all" else args.lang

    # 1. Load embedding model
    log.info("Loading embedding model: %s", args.embedding_model)
    embed_model = SentenceTransformer(args.embedding_model)
    embed_dim = embed_model.get_sentence_embedding_dimension()
    log.info("Embedding dimension: %d", embed_dim)

    # 2. Stream CSV, embed each comment, accumulate per listing
    log.info("Streaming comments, embedding, and averaging per listing ...")
    listing_embed_sums: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(embed_dim, dtype=np.float64)
    )
    listing_counts: dict[str, int] = defaultdict(int)
    listing_texts: dict[str, list[str]] = defaultdict(list)
    total = 0

    for chunk in pd.read_csv(
        args.reviews_csv,
        usecols=["listing_id", "clean_comments", "language"],
        chunksize=args.chunksize, dtype=str,
    ):
        if lang:
            chunk = chunk[chunk["language"] == lang]
        chunk = chunk.dropna(subset=["clean_comments"])

        comments = [clean_text(t) for t in chunk["clean_comments"]]
        valid_mask = [bool(c) for c in comments]
        comments = [c for c, v in zip(comments, valid_mask) if v]
        chunk_ids = [lid for lid, v in zip(chunk["listing_id"], valid_mask) if v]

        if not comments:
            continue

        # Batch embed all comments in this chunk
        embeddings = embed_model.encode(
            comments, batch_size=args.embedding_batch_size, show_progress_bar=False
        )

        for lid, emb, text in zip(chunk_ids, embeddings, comments):
            listing_embed_sums[lid] += emb
            listing_counts[lid] += 1
            # Keep a sample of text for BERTopic's vectorizer (first 5 comments)
            if len(listing_texts[lid]) < 5:
                listing_texts[lid].append(text)

        total += len(comments)
        log.info("  %s comments embedded, %s listings ...",
                 f"{total:,}", f"{len(listing_embed_sums):,}")

    # 3. Compute averaged embeddings
    listing_ids = list(listing_embed_sums.keys())
    avg_embeddings = np.array([
        listing_embed_sums[lid] / listing_counts[lid] for lid in listing_ids
    ])
    # Representative text for BERTopic's c-TF-IDF (concatenated sample)
    documents = [" ".join(listing_texts[lid]) for lid in listing_ids]

    log.info("Averaged embeddings: %s listings, shape %s",
             f"{len(listing_ids):,}", avg_embeddings.shape)

    # 4. Configure and fit BERTopic
    umap_model = UMAP(
        n_neighbors=args.umap_n_neighbors,
        n_components=args.umap_n_components,
        min_dist=0.0, metric="cosine", random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean", prediction_data=True,
    )
    # Combine sklearn English stop words with host names
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    custom_stops = list(ENGLISH_STOP_WORDS | HOST_NAME_STOPS)
    vectorizer_model = CountVectorizer(
        stop_words=custom_stops, min_df=5, max_df=0.5, ngram_range=(1, 2),
    )

    log.info("Fitting BERTopic ...")
    topic_model = BERTopic(
        embedding_model=embed_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=None,  # discover naturally first, then reduce below
        calculate_probabilities=True,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(documents, avg_embeddings)

    n_raw = len(set(topics)) - (1 if -1 in topics else 0)
    outliers_before = sum(1 for t in topics if t == -1)
    log.info("Discovered %d raw topics (%d outliers = %.1f%%)",
             n_raw, outliers_before, outliers_before / len(topics) * 100)

    # Reduce outliers by reassigning to nearest topic via embeddings
    if args.reduce_outliers:
        log.info("Reducing outliers (strategy=embeddings) ...")
        topics = topic_model.reduce_outliers(
            documents, topics, strategy="embeddings", embeddings=avg_embeddings
        )
        topic_model.update_topics(documents, topics=topics,
                                  vectorizer_model=vectorizer_model)
        outliers_after = sum(1 for t in topics if t == -1)
        log.info("Outliers after reduction: %d (%.1f%%)",
                 outliers_after, outliers_after / len(topics) * 100)

    # Merge topics down to target number
    if args.nr_topics and args.nr_topics < len(set(topics)):
        log.info("Reducing to %d topics ...", args.nr_topics)
        topic_model.reduce_topics(documents, nr_topics=args.nr_topics)
        topics = topic_model.topics_
        probs = topic_model.probabilities_

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    log.info("Final topic count: %d", n_topics)

    # 5. Save topic info
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(DATA_DIR / "bertopic_avg_topics_new.csv", index=False)

    print("\n=== Discovered Topics ===")
    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            continue
        print(f"  Topic {row['Topic']} ({row['Count']} docs): {row['Name']}")

    # 6. Per-listing probability matrix
    listing_df = pd.DataFrame({"listing_id": listing_ids, "dominant_topic": topics})
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    for i in range(probs.shape[1]):
        listing_df[f"topic_{i}_prob"] = probs[:, i]

    top3 = np.argsort(-probs, axis=1)[:, :min(3, probs.shape[1])]
    top3_p = np.take_along_axis(probs, top3, axis=1)
    for j in range(top3.shape[1]):
        listing_df[f"top{j+1}_topic"] = top3[:, j]
        listing_df[f"top{j+1}_prob"] = top3_p[:, j]

    listing_df.to_csv(DATA_DIR / "bertopic_avg_listing_topics_new.csv", index=False)
    log.info("Per-listing topics saved (%s listings)", f"{len(listing_df):,}")

    # 7. Save model
    model_dir = DATA_DIR / "bertopic_avg_model_new"
    topic_model.save(str(model_dir), serialization="safetensors",
                     save_ctfidf=True, save_embedding_model=args.embedding_model)
    log.info("Model saved to %s", model_dir)

    outliers = sum(1 for t in topics if t == -1)
    print(f"\nOutliers: {outliers:,} ({outliers/len(topics)*100:.1f}%)")
    print(f"Topics: {n_topics} | Listings: {len(listing_ids):,}")


if __name__ == "__main__":
    main()
