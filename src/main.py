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

if __name__ == "__main__":
    main()