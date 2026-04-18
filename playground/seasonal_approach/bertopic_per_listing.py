"""
BERTopic Topic Modeling — Per-listing with probability distributions.

Concatenates all comments per listing, generates embeddings with
sentence-transformers (GPU-accelerated), then fits BERTopic with
per-topic probability distributions.

Outputs:
  - data/bertopic_topics.csv              : topic ID, top words, count
  - data/bertopic_listing_topics.csv      : per-listing topic probabilities
  - data/bertopic_model/                  : saved model directory
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


# ── helpers ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic cleanup — lowercase, strip extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text.lower().strip())
    return text


def stream_listing_docs(csv_path: str, chunksize: int, lang: str | None):
    """
    Stream the CSV in chunks, concatenate comments per listing_id.
    Returns (listing_ids, documents) as parallel lists.
    """
    log.info("Aggregating comments per listing ...")
    listing_texts: dict[str, list[str]] = defaultdict(list)
    total = 0
    for chunk in pd.read_csv(
        csv_path, usecols=["listing_id", "clean_comments", "language"],
        chunksize=chunksize, dtype=str,
    ):
        if lang:
            chunk = chunk[chunk["language"] == lang]
        for listing_id, text in zip(chunk["listing_id"], chunk["clean_comments"]):
            cleaned = clean_text(text)
            if cleaned:
                listing_texts[listing_id].append(cleaned)
                total += 1
        log.info("  %s comments processed, %s listings ...",
                 f"{total:,}", f"{len(listing_texts):,}")

    listing_ids = list(listing_texts.keys())
    documents = [" ".join(listing_texts[lid]) for lid in listing_ids]
    log.info("Done: %s comments → %s listing documents",
             f"{total:,}", f"{len(documents):,}")
    return listing_ids, documents


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BERTopic on per-listing review comments")
    parser.add_argument("--reviews-csv", required=True, help="Path to reviews CSV")
    parser.add_argument("--num-topics", type=int, default=None,
                        help="Target number of topics (None = auto)")
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--lang", default="en",
                        help="Filter to language (or 'all')")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model name")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--min-topic-size", type=int, default=30,
                        help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-n-components", type=int, default=5)
    parser.add_argument("--max-doc-length", type=int, default=512,
                        help="Truncate listing docs to N words before embedding")
    args = parser.parse_args()

    lang = None if args.lang == "all" else args.lang

    # 1. Build per-listing documents
    listing_ids, documents = stream_listing_docs(
        args.reviews_csv, chunksize=args.chunksize, lang=lang
    )

    # Truncate very long documents (embedding models have token limits)
    documents = [
        " ".join(doc.split()[:args.max_doc_length]) for doc in documents
    ]
    log.info("Documents truncated to max %d words", args.max_doc_length)

    # 2. Generate embeddings (GPU if available, else CPU)
    log.info("Loading embedding model: %s", args.embedding_model)
    embedding_model = SentenceTransformer(args.embedding_model)
    log.info("Generating embeddings for %s documents ...", f"{len(documents):,}")
    embeddings = embedding_model.encode(
        documents,
        batch_size=args.embedding_batch_size,
        show_progress_bar=True,
    )
    log.info("Embeddings shape: %s", embeddings.shape)

    # 3. Configure BERTopic components
    umap_model = UMAP(
        n_neighbors=args.umap_n_neighbors,
        n_components=args.umap_n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_topic_size,
        metric="euclidean",
        prediction_data=True,  # required for probability distributions
    )
    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=5,
        max_df=0.5,
        ngram_range=(1, 2),
    )

    # 4. Fit BERTopic
    log.info("Fitting BERTopic ...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=args.num_topics,  # None = auto, int = reduce to target
        calculate_probabilities=True,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(documents, embeddings)
    # probs shape: (n_docs, n_topics) — probability distribution per document

    log.info("Discovered %d topics (excluding outlier topic -1)",
             len(set(topics)) - (1 if -1 in topics else 0))

    # 5. Save topic info
    topic_info = topic_model.get_topic_info()
    topics_path = DATA_DIR / "bertopic_topics.csv"
    topic_info.to_csv(topics_path, index=False)
    log.info("Topic info saved to %s", topics_path)

    print("\n=== Discovered Topics ===")
    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            continue
        print(f"  Topic {row['Topic']} ({row['Count']} docs): {row['Name']}")

    # 6. Build per-listing probability matrix
    topic_ids_sorted = sorted([t for t in set(topics) if t != -1])
    n_topics = len(topic_ids_sorted)

    if probs.ndim == 1:
        # Edge case: only 1 topic found
        prob_matrix = probs.reshape(-1, 1)
    else:
        prob_matrix = probs  # shape: (n_docs, n_topics)

    listing_df = pd.DataFrame({"listing_id": listing_ids, "dominant_topic": topics})

    # Add probability columns
    for i in range(prob_matrix.shape[1]):
        listing_df[f"topic_{i}_prob"] = prob_matrix[:, i]

    # Add top-3 topics for convenience
    top3_topics = np.argsort(-prob_matrix, axis=1)[:, :3]
    top3_probs = np.take_along_axis(prob_matrix, top3_topics, axis=1)
    listing_df["top1_topic"] = top3_topics[:, 0]
    listing_df["top1_prob"] = top3_probs[:, 0]
    if top3_topics.shape[1] >= 2:
        listing_df["top2_topic"] = top3_topics[:, 1]
        listing_df["top2_prob"] = top3_probs[:, 1]
    if top3_topics.shape[1] >= 3:
        listing_df["top3_topic"] = top3_topics[:, 2]
        listing_df["top3_prob"] = top3_probs[:, 2]

    listing_path = DATA_DIR / "bertopic_listing_topics.csv"
    listing_df.to_csv(listing_path, index=False)
    log.info("Per-listing topics saved to %s  (%s listings)",
             listing_path, f"{len(listing_df):,}")

    # 7. Save model
    model_dir = DATA_DIR / "bertopic_model"
    topic_model.save(str(model_dir), serialization="safetensors",
                     save_ctfidf=True, save_embedding_model=args.embedding_model)
    log.info("Model saved to %s", model_dir)

    # 8. Summary stats
    outlier_count = sum(1 for t in topics if t == -1)
    print(f"\nOutlier listings (no clear topic): {outlier_count:,} "
          f"({outlier_count / len(topics) * 100:.1f}%)")
    print(f"Total topics: {n_topics}")
    print(f"Total listings: {len(listing_ids):,}")


if __name__ == "__main__":
    main()
