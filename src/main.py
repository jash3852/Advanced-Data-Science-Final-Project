from src.data_loader import NewsCorporaDataLoader
from src.minhash import NumbaMinHash
import time

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

    # Query by category
    category_data = data_loader.get_dataset_by_category("t")
    print("\nTechnology Category:")
    print(category_data.head())

    # Query by hostname
    hostname_data = data_loader.get_dataset_by_hostname("www.cnet.com")
    print("\nHostname www.cnet.com:")
    print(hostname_data.head())

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

    texts = sample_df["main_content"].fillna("").astype(str).tolist()

    start_time = time.time()

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            _ = minhash.exact_jaccard(texts[i], texts[j])

    end_time = time.time()
    jaccard_time = end_time - start_time

    print(f"Exact Jaccard runtime: {jaccard_time:.4f} seconds")

    print("\n=========================")
    print("MinHash Runtime Analysis")
    print("=========================")

    start_time = time.time()

    results = minhash.find_candidate_pairs_in_dataframe(
        sample_df,
        text_column="main_content",
        id_column="id",
        top_k=5,
        exact_check=False,  
    )

    end_time = time.time()
    minhash_time = end_time - start_time

    print(f"MinHash runtime: {minhash_time:.4f} seconds")

    print("\n=========================")
    print("Speed Comparison")
    print("=========================")

    if jaccard_time > 0:
        speedup = jaccard_time / minhash_time
        print(f"Speedup (Jaccard / MinHash): {speedup:.2f}x faster")
    else:
        print("Jaccard runtime too small to compare.")

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

        texts = cat_df["main_content"].fillna("").astype(str).tolist()

        similarities = []

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = minhash.exact_jaccard(texts[i], texts[j])
                similarities.append(sim)

        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            max_sim = max(similarities)

            print(f"Average Jaccard Similarity: {avg_sim:.4f}")
            print(f"Max Jaccard Similarity: {max_sim:.4f}")
        else:
            print("No valid comparisons.")

    print("\n=========================")
    print("Random Article Similarity (Baseline)")
    print("=========================")

    random_df = dataset.sample(10, random_state=42)
    texts = random_df["main_content"].fillna("").astype(str).tolist()

    similarities = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = minhash.exact_jaccard(texts[i], texts[j])
            similarities.append(sim)

    if similarities:
        avg_sim = sum(similarities) / len(similarities)
        max_sim = max(similarities)

        print(f"Average Jaccard Similarity (Random): {avg_sim:.4f}")
        print(f"Max Jaccard Similarity (Random): {max_sim:.4f}")
if __name__ == "__main__":
    main()