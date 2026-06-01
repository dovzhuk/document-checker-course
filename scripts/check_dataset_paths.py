from document_checker.dataset import (
    check_dataset_path,
    load_dataset_index,
    subset_by_split_and_limit,
)


def main() -> None:
    df = load_dataset_index()
    subset = subset_by_split_and_limit(df, split="train", n_per_label=2, random_state=42)

    print(f"Subset size for path check: {len(subset)}")

    for _, row in subset.head(10).iterrows():
        result = check_dataset_path(
            path=row["path"],
            label=row["label"],
            split=row["split"],
        )
        print(
            {
                "path": str(result.path),
                "exists": result.exists,
                "is_file": result.is_file,
                "label": result.label,
                "split": result.split,
            }
        )


if __name__ == "__main__":
    main()