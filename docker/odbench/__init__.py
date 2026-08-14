"""Stable runtime API made available inside benchmark dataset images."""

from .datasets import dataset_manifest, load_dataset

__all__ = ["dataset_manifest", "load_dataset"]
