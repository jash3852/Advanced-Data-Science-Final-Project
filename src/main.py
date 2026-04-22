from src.data_loader import NewsCorporaDataLoader

def main():
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

if __name__ == "__main__":
    main()