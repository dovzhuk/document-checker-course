from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentSample:
    path: Path
    label: str
    split: str


def scan_split(split_dir: Path) -> list[DocumentSample]:
    raise NotImplementedError


def build_dataset_index(dataset_root: Path) -> list[DocumentSample]:
    raise NotImplementedError