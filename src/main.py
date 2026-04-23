from src.data_loader import NewsCorporaDataLoader
from src.minhash import NumbaMinHash
import time
import sys
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr


class Tee:
    """Write output to multiple streams simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _pairwise_jaccard_stats(minhash, df, text_column="main_content"):
    """Return average and maximum exact Jaccard statistics for a dataframe slice."""
    if len(df) < 2:
        return None

    texts = df[text_column].fillna("").astype(str).tolist()
    similarities = []
    max_similarity = -1.0
    max_pair = None

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarity = minhash.exact_jaccard(texts[i], texts[j])
            similarities.append(similarity)

            if similarity > max_similarity:
                max_similarity = similarity
                max_pair = (i, j)

    if not similarities:
        return None

    average_similarity = sum(similarities) / len(similarities)
    return {
        "average": average_similarity,
        "maximum": max_similarity,
        "pair": max_pair,
    }


def _article_link(row):
    """Return the row URL when available; otherwise fall back to hostname."""
    url = row.get("url", "")
    if isinstance(url, str) and url.strip():
        return url
    return row.get("hostname", "N/A")


def _build_output_path():
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"main_output_{timestamp}.txt"


def main():
    category_map = {
    "t": "Science and Technology",
    "b": "Business",
    "e": "Entertainment",
    "m": "Health"
    }
    # Initialize the data loader
    data_loader = NewsCorporaDataLoader()

    # Load the entire dataset
    dataset = data_loader.get_dataset()
    print("Full Dataset:")
    print(dataset.head())

    # # Query by category
    # category_data = data_loader.get_dataset_by_category("t")
    # print("\nTechnology Category:")
    # print(category_data.head())

    # # Query by hostname
    # hostname_data = data_loader.get_dataset_by_hostname("www.cnet.com")
    # print("\nHostname www.cnet.com:")
    # print(hostname_data.head())

    # =========================
    # MINHASH PART 
    # =========================

    # make sure your dataset has text
    if "main_content" not in dataset.columns:
        print("\nNo 'main_content' column found. Skipping MinHash.")
        return

    print("\nRunning MinHash...")

    # Initialize MinHash
    minhash = NumbaMinHash(
        num_hashes=100,
        shingle_size=5,
        seed=42
    )
    # Take a SMALL sample first (so it runs fast)
    sample_df = dataset.head(20)

    results = minhash.find_candidate_pairs_in_dataframe(
        sample_df,
        text_column="main_content",
        id_column="id",
        top_k=5,
        exact_check=True,
    )

    print("\nTop Similar Article Pairs:")
    print(results)

    print("\n=========================")
    print("Exact Jaccard Runtime Analysis")
    print("=========================")

    num_iterations = 100
    sample_size = 20

    jaccard_times = []
    minhash_times = []
    speedups = []

    for iteration in range(num_iterations):
        iteration_sample = dataset.sample(sample_size, random_state=42 + iteration)
        texts = iteration_sample["main_content"].fillna("").astype(str).tolist()

        start_time = time.time()
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                _ = minhash.exact_jaccard(texts[i], texts[j])
        jaccard_time = time.time() - start_time
        jaccard_times.append(jaccard_time)

        start_time = time.time()
        _ = minhash.find_candidate_pairs_in_dataframe(
            iteration_sample,
            text_column="main_content",
            id_column="id",
            top_k=5,
            exact_check=False,
        )
        minhash_time = time.time() - start_time
        minhash_times.append(minhash_time)

        if minhash_time > 0:
            speedups.append(jaccard_time / minhash_time)

    avg_jaccard_time = sum(jaccard_times) / len(jaccard_times) if jaccard_times else 0.0
    avg_minhash_time = sum(minhash_times) / len(minhash_times) if minhash_times else 0.0

    print(f"Exact Jaccard average runtime ({num_iterations} runs): {avg_jaccard_time:.4f} seconds")

    print("\n=========================")
    print("MinHash Runtime Analysis")
    print("=========================")

    print(f"MinHash average runtime ({num_iterations} runs): {avg_minhash_time:.4f} seconds")

    print("\n=========================")
    print("Speed Comparison")
    print("=========================")

    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        variance = sum((value - avg_speedup) ** 2 for value in speedups) / len(speedups)
        std_speedup = variance ** 0.5

        print(f"Average Speedup (Jaccard / MinHash): {avg_speedup:.2f}x")
        print(f"Speedup Standard Deviation: {std_speedup:.2f}x")
    else:
        print("Unable to compute speedup (MinHash runtime was zero in all runs).")

    print("\n=========================")
    print("Category-Based Similarity Analysis")
    print("=========================")

    categories = dataset["category"].dropna().unique()

    for cat in categories:
        full_name = category_map.get(cat, "Unknown")
        print(f"\nCategory: {full_name} ({cat})")

        # Take a small sample per category
        cat_df = dataset[dataset["category"] == cat].head(10)

        if len(cat_df) < 2:
            print("Not enough articles.")
            continue

        stats = _pairwise_jaccard_stats(minhash, cat_df)

        if stats:
            max_left, max_right = stats["pair"]
            left_row = cat_df.iloc[max_left]
            right_row = cat_df.iloc[max_right]

            print(f"Average Jaccard Similarity: {stats['average']:.4f}")
            print(f"Max Jaccard Similarity: {stats['maximum']:.4f}")
            print(
                "Max Pair: "
                f"({left_row['id']}, {_article_link(left_row)}) <-> "
                f"({right_row['id']}, {_article_link(right_row)})"
            )
        else:
            print("No valid comparisons.")

    print("\n=========================")
    print("Hostname-Based Similarity Analysis")
    print("=========================")

    hostnames = dataset["hostname"].dropna().unique()

    for hostname in hostnames:
        hostname_df = dataset[dataset["hostname"] == hostname]

        if len(hostname_df) < 2:
            continue

        print(f"\nHostname: {hostname}")

        stats = _pairwise_jaccard_stats(minhash, hostname_df)

        if stats:
            max_left, max_right = stats["pair"]
            left_row = hostname_df.iloc[max_left]
            right_row = hostname_df.iloc[max_right]

            print(f"Average Jaccard Similarity: {stats['average']:.4f}")
            print(f"Max Jaccard Similarity: {stats['maximum']:.4f}")
            print(
                "Max Pair: "
                f"({left_row['id']}, {_article_link(left_row)}) <-> "
                f"({right_row['id']}, {_article_link(right_row)})"
            )
        else:
            print("No valid comparisons.")

    print("\n=========================")
    print("Random Article Similarity (Baseline)")
    print("=========================")

    random_df = dataset.sample(10, random_state=42)
    stats = _pairwise_jaccard_stats(minhash, random_df)

    if stats:
        max_left, max_right = stats["pair"]
        left_row = random_df.iloc[max_left]
        right_row = random_df.iloc[max_right]

        print(f"Average Jaccard Similarity (Random): {stats['average']:.4f}")
        print(f"Max Jaccard Similarity (Random): {stats['maximum']:.4f}")
        print(
            "Max Pair (Random): "
            f"({left_row['id']}, {_article_link(left_row)}) <-> "
            f"({right_row['id']}, {_article_link(right_row)})"
        )
if __name__ == "__main__":
    output_path = _build_output_path()
    with output_path.open("w", encoding="utf-8") as output_file:
        tee_stream = Tee(sys.stdout, output_file)
        with redirect_stdout(tee_stream), redirect_stderr(tee_stream):
            print(f"Saving output log to: {output_path}")
            main()