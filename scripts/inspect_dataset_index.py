from pathlib import Path

import pandas as pd

# pyright: reportMissingModuleSource=false, reportAttributeAccessIssue=false


def main() -> None:
    index_path = Path("artifacts/metrics/dataset_index.csv")
    if not index_path.exists():
        raise FileNotFoundError(f"Dataset index not found: {index_path}")

    df = pd.read_csv(index_path)

    print(f"Total samples: {len(df)}")

    print("\nSplits:")
    print(df["split"].value_counts())

    print("\nNumber of classes:", df["label"].nunique())

    print("\nTop 10 labels by count:")
    print(df["label"].value_counts().head(10))

    print("\nSample paths by first 3 labels:")
    for label in df["label"].unique()[:3]:
        sample_paths = df.loc[df["label"] == label, "path"].head(3).tolist()
        print(f"  {label}:")
        for p in sample_paths:
            print(f"    {p}")


if __name__ == "__main__":
    main()