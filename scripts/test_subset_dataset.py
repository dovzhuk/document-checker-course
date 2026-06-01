from pathlib import Path

from document_checker.dataset import (
    load_dataset_index,
    subset_by_split_and_limit,
)


def main() -> None:
    df = load_dataset_index(Path("artifacts/metrics/dataset_index.csv"))

    print(f"Total samples in index: {len(df)}")
    print("Splits:", df["split"].value_counts().to_dict())

    subset = subset_by_split_and_limit(df, split="train", n_per_label=5, random_state=42)

    print(f"\nSubset size: {len(subset)}")
    print("Subset splits:", subset["split"].value_counts().to_dict())
    print("Subset labels (counts):")
    print(subset["label"].value_counts().sort_index())

    print("\nFirst 5 rows of subset:")
    print(subset.head())


if __name__ == "__main__":
    main()