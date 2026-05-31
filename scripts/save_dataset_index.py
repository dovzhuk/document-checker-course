from pathlib import Path

from document_checker.dataset import build_dataset_index, samples_to_dataframe


def main() -> None:
    dataset_root = Path("data/raw/rvl-cdip-small-200")
    output_path = Path("artifacts/metrics/dataset_index.csv")

    samples = build_dataset_index(dataset_root)
    df = samples_to_dataframe(samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved dataset index to: {output_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()