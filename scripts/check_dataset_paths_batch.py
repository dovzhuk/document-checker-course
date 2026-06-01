from document_checker.dataset import (
    check_dataset_paths,
    load_dataset_index,
    subset_by_split_and_limit,
)


def main() -> None:
    df = load_dataset_index()
    subset = subset_by_split_and_limit(df, split="train", n_per_label=2, random_state=42)

    checked = check_dataset_paths(subset)

    print(f"Checked rows: {len(checked)}")
    print("\nExists counts:")
    print(checked["exists"].value_counts())

    print("\nIs-file counts:")
    print(checked["is_file"].value_counts())

    print("\nFirst 5 checked rows:")
    print(checked.head())


if __name__ == "__main__":
    main()