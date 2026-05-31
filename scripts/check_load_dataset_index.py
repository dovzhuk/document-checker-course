from pathlib import Path

from document_checker.dataset import load_dataset_index


def main() -> None:
    df = load_dataset_index(Path("artifacts/metrics/dataset_index.csv"))

    print(f"Total samples: {len(df)}")
    print("\nSplits:")
    print(df["split"].value_counts())

    print("\nNumber of classes:", df["label"].nunique())


if __name__ == "__main__":
    main()