"""
LDA Topic Modeling — Approach 1: Train globally, aggregate per listing.

Trains LDA on ALL individual comments (streamed to keep memory low),
then infers per-comment topic distributions and aggregates by listing_id.

Outputs:
  - data/lda_topics_global.csv          : top words per topic
  - data/lda_listing_topics_global.csv  : per-listing mean topic distribution
"""

import argparse
import logging
import os
import re
from pathlib import Path

import pandas as pd
from gensim import corpora, models
from gensim.models import CoherenceModel

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


# ── helpers ──────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer with short-token filter."""
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = re.findall(r"[a-z]{3,}", text.lower())
    return tokens


class ReviewCorpus:
    """Streams bag-of-words vectors from the CSV without loading it all."""

    def __init__(self, csv_path: str, dictionary: corpora.Dictionary,
                 chunksize: int = 50_000, lang: str | None = "en"):
        self.csv_path = csv_path
        self.dictionary = dictionary
        self.chunksize = chunksize
        self.lang = lang

    def __iter__(self):
        for chunk in pd.read_csv(
            self.csv_path, usecols=["clean_comments", "language"],
            chunksize=self.chunksize, dtype=str,
        ):
            if self.lang:
                chunk = chunk[chunk["language"] == self.lang]
            for text in chunk["clean_comments"]:
                tokens = tokenize(text)
                if tokens:
                    yield self.dictionary.doc2bow(tokens)


def build_dictionary(csv_path: str, chunksize: int = 50_000,
                     lang: str | None = "en",
                     no_below: int = 20, no_above: float = 0.5) -> corpora.Dictionary:
    """Stream through CSV once to build a filtered dictionary."""
    log.info("Building dictionary (pass 1) ...")
    dictionary = corpora.Dictionary()
    total = 0
    for chunk in pd.read_csv(
        csv_path, usecols=["clean_comments", "language"],
        chunksize=chunksize, dtype=str,
    ):
        if lang:
            chunk = chunk[chunk["language"] == lang]
        docs = [tokenize(t) for t in chunk["clean_comments"] if isinstance(t, str)]
        dictionary.add_documents(docs)
        total += len(docs)
        log.info("  processed %s docs so far ...", f"{total:,}")
    log.info("Raw vocab size: %s", f"{len(dictionary):,}")
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    log.info("Filtered vocab size: %s", f"{len(dictionary):,}")
    return dictionary


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Global LDA on review comments")
    parser.add_argument("--reviews-csv", required=True, help="Path to reviews CSV")
    parser.add_argument("--num-topics", type=int, default=15)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--lang", default="en", help="Filter to language (or 'all')")
    parser.add_argument("--no-below", type=int, default=20)
    parser.add_argument("--no-above", type=float, default=0.5)
    args = parser.parse_args()

    lang = None if args.lang == "all" else args.lang

    # 1. Build dictionary
    dictionary = build_dictionary(
        args.reviews_csv, chunksize=args.chunksize, lang=lang,
        no_below=args.no_below, no_above=args.no_above,
    )
    dict_path = DATA_DIR / "lda_dictionary_global.dict"
    dictionary.save(str(dict_path))
    log.info("Dictionary saved to %s", dict_path)

    # 2. Build streaming corpus
    corpus = ReviewCorpus(args.reviews_csv, dictionary,
                          chunksize=args.chunksize, lang=lang)

    # 3. Train LDA
    log.info("Training LdaMulticore (k=%d, passes=%d, workers=%d) ...",
             args.num_topics, args.passes, args.workers)
    lda = models.LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=args.num_topics,
        chunksize=args.chunksize,
        passes=args.passes,
        workers=args.workers,
        random_state=42,
    )
    model_path = DATA_DIR / "lda_model_global"
    lda.save(str(model_path))
    log.info("Model saved to %s", model_path)

    # 4. Save topic–word table
    rows = []
    for t in range(args.num_topics):
        top_words = lda.show_topic(t, topn=15)
        rows.append({
            "topic_id": t,
            "top_words": ", ".join(w for w, _ in top_words),
            "weights": ", ".join(f"{p:.4f}" for _, p in top_words),
        })
    topics_df = pd.DataFrame(rows)
    topics_path = DATA_DIR / "lda_topics_global.csv"
    topics_df.to_csv(topics_path, index=False)
    log.info("Topics saved to %s", topics_path)
    print("\n=== Discovered Topics ===")
    for _, row in topics_df.iterrows():
        print(f"  Topic {row.topic_id}: {row.top_words}")

    # 5. Per-listing aggregation (pass 3)
    log.info("Aggregating topic distributions per listing (pass 3) ...")
    records = []
    for chunk in pd.read_csv(
        args.reviews_csv,
        usecols=["listing_id", "clean_comments", "language"],
        chunksize=args.chunksize, dtype=str,
    ):
        if lang:
            chunk = chunk[chunk["language"] == lang]
        for listing_id, text in zip(chunk["listing_id"], chunk["clean_comments"]):
            tokens = tokenize(text)
            if not tokens:
                continue
            bow = dictionary.doc2bow(tokens)
            dist = dict(lda.get_document_topics(bow, minimum_probability=0.0))
            dist["listing_id"] = listing_id
            records.append(dist)

    agg_df = pd.DataFrame(records)
    topic_cols = [c for c in agg_df.columns if c != "listing_id"]
    listing_topics = agg_df.groupby("listing_id")[topic_cols].mean().reset_index()

    # Rename columns
    listing_topics.columns = ["listing_id"] + [
        f"topic_{i}" for i in range(args.num_topics)
    ]
    listing_path = DATA_DIR / "lda_listing_topics_global.csv"
    listing_topics.to_csv(listing_path, index=False)
    log.info("Per-listing topics saved to %s  (%s listings)",
             listing_path, f"{len(listing_topics):,}")

    # 6. Coherence score
    log.info("Computing coherence (c_v) — this reads the corpus again ...")
    coherence_texts = []
    n_sample = 100_000
    for chunk in pd.read_csv(
        args.reviews_csv, usecols=["clean_comments", "language"],
        chunksize=args.chunksize, dtype=str,
    ):
        if lang:
            chunk = chunk[chunk["language"] == lang]
        for text in chunk["clean_comments"]:
            tokens = tokenize(text)
            if tokens:
                coherence_texts.append(tokens)
            if len(coherence_texts) >= n_sample:
                break
        if len(coherence_texts) >= n_sample:
            break
    cm = CoherenceModel(model=lda, texts=coherence_texts,
                        dictionary=dictionary, coherence="c_v")
    score = cm.get_coherence()
    log.info("Coherence (c_v): %.4f", score)
    print(f"\nCoherence (c_v): {score:.4f}")


if __name__ == "__main__":
    main()
