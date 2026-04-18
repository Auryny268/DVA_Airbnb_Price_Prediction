"""
BERTopic — Approach B: Per-comment topic assignment, then aggregate per listing.

Runs BERTopic on individual comments (streamed in batches to manage memory),
assigns each comment a topic, then aggregates topic proportions per listing.

This preserves ALL text and does NOT dilute minority topics.

Outputs:
  - data/bertopic_comments_topics_new.csv            : topic ID, top words, count
  - data/bertopic_comments_listing_topics_new.csv    : per-listing topic proportions
  - data/bertopic_comments_model_new/                : saved model directory
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

# Host/person names that dominate topic labels — filter from CountVectorizer
HOST_NAME_STOPS = {
    # Male names (common US + Latin American)
    "aaron", "adam", "adrian", "adrien", "alan", "albert", "alex", "alexander",
    "allen", "andrew", "andy", "anthony", "antonio", "arthur", "barry", "ben",
    "benjamin", "billy", "bob", "bobby", "brad", "bradley", "brandon", "brian",
    "bruce", "carl", "carlos", "charlie", "chris", "christopher", "cliff",
    "craig", "dan", "daniel", "danny", "darren", "dave", "david", "dean",
    "dennis", "derek", "derrick", "doha", "don", "donald", "doug", "douglas",
    "drew", "dustin", "ed", "eddie", "edward", "eli", "elliott", "eric",
    "ernest", "eugene", "evan", "fabio", "frank", "fred", "freddie", "gabriel",
    "gary", "george", "gerald", "glen", "gordon", "greg", "gregory", "harry",
    "henry", "hiroki", "howard", "ian", "jack", "jackson", "jacob", "jake",
    "james", "jamie", "jason", "jay", "jeff", "jeffrey", "jeremy", "jerome",
    "jerry", "jesse", "jim", "jimmy", "joe", "joel", "john", "johnny", "jon",
    "jonathan", "jordan", "jorge", "jose", "joseph", "josh", "joshua", "juan",
    "julian", "julien", "justin", "kai", "keith", "ken", "kenneth", "kevin",
    "kyle", "lance", "larry", "lawrence", "lee", "leo", "leon", "leonard",
    "louis", "luis", "luke", "marc", "marco", "marcus", "mario", "mark",
    "martin", "matt", "matthew", "max", "michael", "miguel", "mike",
    "mohammed", "mohammad", "nathan", "neil", "nelson", "nick", "noah",
    "oliver", "oscar", "otis", "pat", "patrick", "paul", "pedro", "pete",
    "peter", "phil", "philip", "phillip", "pierre", "ralph", "randy", "ray",
    "raymond", "ricardo", "richard", "rick", "rob", "robert", "robin", "rod",
    "roger", "ron", "ronald", "ross", "roy", "russell", "ryan", "sam",
    "samuel", "scott", "sean", "shawn", "shogo", "simon", "stanley", "stefan",
    "stephen", "steve", "steven", "stuart", "ted", "terry", "tim", "timothy",
    "todd", "tom", "tommy", "tony", "travis", "trevor", "troy", "tyler",
    "victor", "vincent", "wade", "walter", "warren", "wayne", "wesley",
    "willie", "zachary",
    # Female names (common US + Latin American)
    "alice", "alicia", "amanda", "amber", "amy", "ana", "andrea", "angela",
    "angie", "ann", "anna", "anne", "annie", "ashley", "barbara", "becky",
    "bella", "bertie", "beth", "betty", "bianca", "bonnie", "brenda",
    "brittany", "carmen", "carol", "caroline", "carolyn", "catherine", "cathy",
    "cheryl", "chioma", "christina", "christine", "cindy", "claire", "claudia",
    "colleen", "courtney", "crystal", "cynthia", "dana", "danielle", "dawn",
    "debbie", "deborah", "denise", "diana", "diane", "donna", "doris",
    "dorothy", "eileen", "elaine", "elena", "elizabeth", "ellen", "emily",
    "emma", "erica", "erin", "eva", "evelyn", "fabiola", "fiona", "florence",
    "frances", "gail", "galina", "gloria", "grace", "hannah", "heather",
    "helen", "holly", "irene", "irin", "jackie", "jane", "janet", "janice",
    "jean", "jeanette", "jenny", "jessica", "jill", "joan", "joanne",
    "jocelyn", "jodie", "joy", "joyce", "judith", "judy", "julia", "julie",
    "june", "kaitlin", "kara", "karen", "kate", "katherine", "kathleen",
    "kathy", "katie", "kay", "kelly", "kim", "kimberly", "kristin", "laura",
    "lauren", "leah", "leeann", "leslie", "leyla", "lily", "linda", "lisa",
    "liz", "lori", "lorraine", "louise", "lucia", "lynn", "maggie",
    "margaret", "maria", "marian", "marie", "marilyn", "marina", "marlene",
    "martha", "mary", "maureen", "megan", "melanie", "melissa", "michelle",
    "mildred", "momoyo", "monica", "nancy", "natalie", "natasha", "nicole",
    "nina", "nora", "norma", "olivia", "pamela", "patricia", "paula", "peggy",
    "penny", "prerna", "rachel", "rebecca", "regina", "renee", "rissala",
    "robin", "rosa", "rose", "rosemary", "ruth", "sally", "samantha",
    "sandra", "sandy", "sara", "sarah", "sharon", "sheila", "shirley",
    "simone", "sofia", "sonia", "sophia", "stacy", "stephanie", "sue",
    "susan", "suzanne", "sylvia", "tammy", "tanya", "teresa", "terri",
    "theresa", "tiffany", "tina", "tracy", "valerie", "vanessa", "veronica",
    "victoria", "virginia", "vivian", "wendy", "yumi", "yulia", "yvonne",
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
        description="Per-comment BERTopic, aggregate per listing")
    parser.add_argument("--reviews-csv", required=True)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--min-cluster-size", type=int, default=10,
                        help="HDBSCAN min_cluster_size")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="HDBSCAN min_samples")
    parser.add_argument("--nr-topics", type=int, default=15,
                        help="Merge down to this many topics after fitting")
    parser.add_argument("--reduce-outliers", action=argparse.BooleanOptionalAction,
                        default=True, help="Reassign outliers to nearest topic")
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
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean", prediction_data=True,
    )
    # Combine sklearn English stop words with host names
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    custom_stops = list(ENGLISH_STOP_WORDS | HOST_NAME_STOPS)
    vectorizer_model = CountVectorizer(
        stop_words=custom_stops, min_df=20, max_df=0.5, ngram_range=(1, 2),
    )

    log.info("Fitting BERTopic on %s comments ...", f"{len(sample_comments):,}")
    topic_model = BERTopic(
        embedding_model=embed_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=None,  # discover naturally first, then reduce below
        calculate_probabilities=False,  # skip probs during fit (memory)
        verbose=True,
    )
    topic_model.fit(sample_comments, sample_embeddings)

    n_raw = len(topic_model.get_topic_info()) - 1
    log.info("Discovered %d raw topics", n_raw)

    # Reduce outliers on the sample fit
    if args.reduce_outliers:
        log.info("Reducing outliers on fit sample (strategy=embeddings) ...")
        sample_topics = topic_model.topics_
        sample_topics = topic_model.reduce_outliers(
            sample_comments, sample_topics,
            strategy="embeddings", embeddings=sample_embeddings
        )
        topic_model.update_topics(sample_comments, topics=sample_topics,
                                  vectorizer_model=vectorizer_model)
        outliers_after = sum(1 for t in sample_topics if t == -1)
        log.info("Sample outliers after reduction: %d (%.1f%%)",
                 outliers_after, outliers_after / len(sample_topics) * 100)

    # Merge topics down to target number
    if args.nr_topics and args.nr_topics < len(set(topic_model.topics_)):
        log.info("Reducing to %d topics ...", args.nr_topics)
        # Swap to relaxed vectorizer before reduce_topics — it internally
        # re-fits c-TF-IDF at each merge step, and min_df=20 crashes when
        # the number of topic-level docs drops below ~40.
        relaxed_vectorizer = CountVectorizer(
            stop_words=custom_stops, min_df=2, max_df=0.95, ngram_range=(1, 2),
        )
        topic_model.vectorizer_model = relaxed_vectorizer
        topic_model.reduce_topics(sample_comments, nr_topics=args.nr_topics)

    n_topics = len(topic_model.get_topic_info()) - 1
    log.info("Final topic count: %d", n_topics)

    # 5. Save topic info
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(DATA_DIR / "bertopic_comments_topics_new.csv", index=False)

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

    listing_path = DATA_DIR / "bertopic_comments_listing_topics_new.csv"
    listing_df.to_csv(listing_path, index=False)
    log.info("Per-listing topics saved to %s (%s listings)",
             listing_path, f"{len(listing_df):,}")

    # 8. Save model
    model_dir = DATA_DIR / "bertopic_comments_model_new"
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
