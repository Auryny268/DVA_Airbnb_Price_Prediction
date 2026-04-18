"""
LDA Topic Modeling — Approach 2: Concatenate comments per listing, then train.

Groups all comments by listing_id first (streamed), concatenates them into
one document per listing, then trains LDA on ~N listing-level documents.

Outputs:
  - data/lda_topics_perlisting.csv          : top words per topic
  - data/lda_listing_topics_perlisting.csv  : per-listing topic distribution
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import spacy
from gensim import corpora, models
from gensim.models import CoherenceModel

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# Parts of speech to keep: nouns, adjectives, verbs, adverbs
KEEP_POS = {"NOUN", "ADJ", "VERB", "ADV"}

# Custom stopwords: Airbnb-generic terms, common names, pronouns, etc.
CUSTOM_STOPS = {
    # Airbnb generic
    "airbnb", "host", "hosts", "guest", "guests", "stay", "stayed", "staying",
    "place", "apartment", "room", "house", "home", "listing", "rental",
    "bed", "bedroom", "bathroom", "kitchen", "building",
    # Location generic
    "brooklyn", "manhattan", "queens", "bronx", "york", "nyc", "city",
    "street", "avenue", "block", "neighborhood", "area",
    # Over-common review words
    "great", "nice", "good", "beautiful", "lovely", "amazing", "wonderful",
    "perfect", "excellent", "awesome", "fantastic", "love", "loved",
    "recommend", "recommended", "definitely", "highly", "enjoy", "enjoyed",
    # Functional
    "get", "got", "make", "made", "take", "took", "come", "came", "going",
    "would", "could", "also", "really", "very", "much", "well", "just",
    "even", "back", "still", "right", "thing", "time", "day", "night",
    # Common names in reviews
    "david", "michael", "john", "alex", "chris", "mark", "james", "mary",
    "sarah", "lisa", "anna", "mike", "dan", "tom", "joe", "bob",
}


def load_nlp(model: str = "en_core_web_sm", pos_filter: bool = True):
    """Load spaCy model with only the tokenizer and lemmatizer."""
    log.info("Loading spaCy model: %s (POS filtering: %s)", model, pos_filter)
    disable = ["parser", "ner"]
    if not pos_filter:
        disable.append("tagger")
    nlp = spacy.load(model, disable=disable)
    nlp.max_length = 2_000_000
    return nlp


def tokenize(doc, stops: set[str], pos_filter: bool = True) -> list[str]:
    """Extract lemmatized tokens, optionally filtering by POS and stopwords."""
    tokens = []
    for token in doc:
        if not token.is_stop and token.is_alpha and len(token.lemma_) >= 3 \
                and token.lemma_.lower() not in stops:
            if pos_filter and token.pos_ not in KEEP_POS:
                continue
            tokens.append(token.lemma_.lower())
    return tokens


def stream_listing_docs(csv_path: str, chunksize: int, lang: str | None,
                        nlp, n_process: int = 4,
                        pos_filter: bool = True) -> dict[str, list[str]]:
    """
    Stream the CSV in chunks, lemmatize + filter with spaCy,
    accumulate tokens per listing_id.

    Memory note: This stores one token list per listing (~40-80k listings),
    NOT one per comment (~millions). Typically fits in a few GB of RAM.
    """
    log.info("Aggregating comments per listing (pass 1) ...")
    listing_tokens: dict[str, list[str]] = defaultdict(list)
    total_comments = 0
    for chunk in pd.read_csv(
        csv_path, usecols=["listing_id", "clean_comments", "language"],
        chunksize=chunksize, dtype=str,
    ):
        if lang:
            chunk = chunk[chunk["language"] == lang]
        chunk = chunk.dropna(subset=["clean_comments"])

        texts = chunk["clean_comments"].tolist()
        ids = chunk["listing_id"].tolist()

        # Batch process with spaCy pipe for speed
        for listing_id, doc in zip(ids, nlp.pipe(texts, batch_size=2000, n_process=n_process)):
            tokens = tokenize(doc, CUSTOM_STOPS, pos_filter=pos_filter)
            if tokens:
                listing_tokens[listing_id].extend(tokens)
                total_comments += 1
        log.info("  %s comments processed, %s listings so far ...",
                 f"{total_comments:,}", f"{len(listing_tokens):,}")
    log.info("Done: %s comments → %s listing documents",
             f"{total_comments:,}", f"{len(listing_tokens):,}")
    return listing_tokens


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-listing LDA on review comments")
    parser.add_argument("--reviews-csv", required=True, help="Path to reviews CSV")
    parser.add_argument("--num-topics", type=int, default=15)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--passes", type=int, default=10,
                        help="More passes feasible here (fewer docs)")
    parser.add_argument("--lang", default="en", help="Filter to language (or 'all')")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--n-process", type=int, default=4,
                        help="Number of spaCy parallel processes")
    parser.add_argument("--pos-filter", action=argparse.BooleanOptionalAction,
                        default=True, help="Filter by POS tags (NOUN/ADJ/VERB/ADV)")
    parser.add_argument("--no-below", type=int, default=5)
    parser.add_argument("--no-above", type=float, default=0.5)
    args = parser.parse_args()

    lang = None if args.lang == "all" else args.lang

    # 0. Load spaCy
    nlp = load_nlp(pos_filter=args.pos_filter)

    # 1. Build per-listing documents
    listing_tokens = stream_listing_docs(
        args.reviews_csv, chunksize=args.chunksize, lang=lang, nlp=nlp,
        n_process=args.n_process, pos_filter=args.pos_filter
    )
    listing_ids = list(listing_tokens.keys())
    docs = [listing_tokens[lid] for lid in listing_ids]

    # 2. Build dictionary
    log.info("Building dictionary ...")
    dictionary = corpora.Dictionary(docs)
    log.info("Raw vocab size: %s", f"{len(dictionary):,}")
    dictionary.filter_extremes(no_below=args.no_below, no_above=args.no_above)
    log.info("Filtered vocab size: %s", f"{len(dictionary):,}")
    dict_path = DATA_DIR / "lda_dictionary_perlisting.dict"
    dictionary.save(str(dict_path))

    # 3. Build corpus (in-memory — only ~N_listings docs, manageable)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    log.info("Corpus built: %s documents", f"{len(corpus):,}")

    # 4. Train LDA
    log.info("Training LdaMulticore (k=%d, passes=%d) ...",
             args.num_topics, args.passes)
    lda = models.LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=args.num_topics,
        chunksize=2000,
        passes=args.passes,
        workers=args.workers,
        random_state=42,
    )
    model_path = DATA_DIR / "lda_model_perlisting"
    lda.save(str(model_path))
    log.info("Model saved to %s", model_path)

    # 5. Save topic–word table
    rows = []
    for t in range(args.num_topics):
        top_words = lda.show_topic(t, topn=15)
        rows.append({
            "topic_id": t,
            "top_words": ", ".join(w for w, _ in top_words),
            "weights": ", ".join(f"{p:.4f}" for _, p in top_words),
        })
    topics_df = pd.DataFrame(rows)
    topics_path = DATA_DIR / "lda_topics_perlisting.csv"
    topics_df.to_csv(topics_path, index=False)
    log.info("Topics saved to %s", topics_path)
    print("\n=== Discovered Topics ===")
    for _, row in topics_df.iterrows():
        print(f"  Topic {row.topic_id}: {row.top_words}")

    # 6. Per-listing topic distribution
    records = []
    for lid, bow in zip(listing_ids, corpus):
        dist = dict(lda.get_document_topics(bow, minimum_probability=0.0))
        dist["listing_id"] = lid
        records.append(dist)
    listing_df = pd.DataFrame(records)
    topic_cols = [c for c in listing_df.columns if c != "listing_id"]
    listing_df.columns = ["listing_id"] if len(topic_cols) == 0 else list(listing_df.columns)
    # Rename numeric cols
    rename = {i: f"topic_{i}" for i in range(args.num_topics)}
    listing_df.rename(columns=rename, inplace=True)
    listing_path = DATA_DIR / "lda_listing_topics_perlisting.csv"
    listing_df.to_csv(listing_path, index=False)
    log.info("Per-listing topics saved to %s  (%s listings)",
             listing_path, f"{len(listing_df):,}")

    # 7. Coherence
    log.info("Computing coherence (c_v) ...")
    cm = CoherenceModel(model=lda, texts=docs, dictionary=dictionary,
                        coherence="c_v")
    score = cm.get_coherence()
    log.info("Coherence (c_v): %.4f", score)
    print(f"\nCoherence (c_v): {score:.4f}")


if __name__ == "__main__":
    main()
