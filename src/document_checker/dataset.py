import pandas as pd
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentSample:
    """One document sample from RVL-CDIP-small-200 dataset."""

    path: Path
    label: str
    split: str

@dataclass(frozen=True)
class DatasetPathCheckResult:
    """Validation result for one dataset index row."""

    path: Path
    exists: bool
    is_file: bool
    label: str | None = None
    split: str | None = None   


def scan_split(split_dir: Path) -> list[DocumentSample]:
    """Scan a single split directory (e.g. train or val) and collect samples.

    Expected structure inside split_dir:
        split_dir/
            <label_1>/*.tif
            <label_2>/*.tif
            ...

    Parameters
    ----------
    split_dir : Path
        Path to split directory, e.g. data/raw/rvl-cdip-small-200/train

    Returns
    -------
    list[DocumentSample]
        List of samples with path, label and split name.
    """
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    if not split_dir.is_dir():
        raise NotADirectoryError(f"Split path is not a directory: {split_dir}")

    split_name = split_dir.name
    samples: list[DocumentSample] = []

    # iterate over class directories (labels) inside the split directory
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        label = class_dir.name

        # iterate over all .tif files inside the class directory
        for file_path in sorted(class_dir.glob("*.tif")):
            samples.append(
                DocumentSample(
                    path=file_path,
                    label=label,
                    split=split_name,
                )
            )

    return samples


def build_dataset_index(dataset_root: Path) -> list[DocumentSample]:
    """Build full dataset index from dataset root.

    Expected structure:
        dataset_root/
            train/<label>/*.tif
            val/<label>/*.tif

    Parameters
    ----------
    dataset_root : Path
        Path to dataset root, e.g. data/raw/rvl-cdip-small-200

    Returns
    -------
    list[DocumentSample]
        Combined list of samples from all available splits.
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    samples: list[DocumentSample] = []

    for split_name in ("train", "val"):
        split_dir = dataset_root / split_name
        if split_dir.exists() and split_dir.is_dir():
            split_samples = scan_split(split_dir)
            samples.extend(split_samples)

    samples = sorted(samples, key=lambda sample: (sample.split, sample.label, str(sample.path)))
    return samples


def samples_to_dataframe(samples: list[DocumentSample]) -> pd.DataFrame:
    """Convert a list of DocumentSample into a pandas DataFrame.

    DataFrame columns:
        - path: str, filesystem path to the image
        - label: str, document class
        - split: str, dataset split name (e.g. 'train', 'val')
    """
    records = [
        {"path": str(sample.path), "label": sample.label, "split": sample.split}
        for sample in samples
    ]
    return pd.DataFrame.from_records(records, columns=["path", "label", "split"])


def load_dataset_index(index_path: Path | str = Path("artifacts/metrics/dataset_index.csv")) -> pd.DataFrame:
    """Load dataset index CSV into a pandas DataFrame.

    Parameters
    ----------
    index_path : Path | str, optional
        Path to the dataset index CSV. By default points to
        artifacts/metrics/dataset_index.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: path, label, split.
    """
    index_path = Path(index_path)

    if not index_path.exists():
        raise FileNotFoundError(f"Dataset index not found: {index_path}")

    df = pd.read_csv(index_path)
    expected_columns = {"path", "label", "split"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"Index file is missing columns: {missing}")

    return df


def check_dataset_path(
    path: Path | str,
    label: str | None = None,
    split: str | None = None,
) -> DatasetPathCheckResult:
    """Validate one dataset file path."""

    path = Path(path)

    return DatasetPathCheckResult(
        path=path,
        exists=path.exists(),
        is_file=path.is_file(),
        label=label,
        split=split,
    )


def check_dataset_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Validate dataset paths from an index DataFrame."""

    results = []

    for _, row in df.iterrows():
        result = check_dataset_path(
            path=row["path"],
            label=row.get("label"),
            split=row.get("split"),
        )
        results.append(
            {
                "path": str(result.path),
                "exists": result.exists,
                "is_file": result.is_file,
                "label": result.label,
                "split": result.split,
            }
        )

    return pd.DataFrame(results)

def subset_by_split_and_limit(
    df: pd.DataFrame,
    split: str = "train",
    n_per_label: int | None = None,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Filter dataset index by split and optional per-label limit.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset index with columns: path, label, split.
    split : str, optional
        Split name to keep (e.g. "train", "val"). Default is "train".
    n_per_label : int or None, optional
        If provided, limit to at most n_per_label samples per label for the
        given split. If None, keep all samples for the split.
    random_state : int or None, optional
        Random seed used when sampling per label. Default is 42.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    filtered = df[df["split"] == split].copy()

    if n_per_label is None:
        return filtered

    def _sample_group(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) <= n_per_label:
            return group
        return group.sample(n=n_per_label, random_state=random_state)

    grouped = filtered.groupby("label", group_keys=False)
    subset_parts: list[pd.DataFrame] = []

    for _, group in grouped:
        sampled = _sample_group(group)
        subset_parts.append(sampled)

    subset = pd.concat(subset_parts, ignore_index=True)

    return subset