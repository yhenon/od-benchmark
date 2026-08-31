"""Build hook that downloads and retains only the CIFAR-10 training split."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path, PurePosixPath


ARCHIVE_URLS = (
    # Byte-for-byte mirror used because the University of Toronto host is often
    # too slow for unattended builds. The canonical checksum is verified below.
    "https://huggingface.co/datasets/xingslong/cifar-10-batches-py/resolve/main/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
)
CANONICAL_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
ARCHIVE_ROOT = PurePosixPath("cifar-10-batches-py")
TRAIN_FILES = {
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    "batches.meta": "5ff9c542aee3614f3951f8cda6e48888",
}
CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(urls: tuple[str, ...], destination: Path) -> None:
    for source_index, url in enumerate(urls):
        try:
            print(f"downloading CIFAR-10 from {url}", flush=True)
            downloaded = 0
            next_report = 16 * 1024 * 1024
            with urllib.request.urlopen(url, timeout=60) as response:
                with destination.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report:
                            print(f"downloaded {downloaded // (1024 * 1024)} MiB", flush=True)
                            next_report += 16 * 1024 * 1024
            return
        except OSError:
            destination.unlink(missing_ok=True)
            if source_index == len(urls) - 1:
                raise
            time.sleep(2)


def main(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)

    with tempfile.TemporaryDirectory() as temporary_directory:
        archive = Path(temporary_directory) / "cifar-10-python.tar.gz"
        download(ARCHIVE_URLS, archive)
        if md5(archive) != ARCHIVE_MD5:
            raise RuntimeError("CIFAR-10 archive checksum mismatch")

        with tarfile.open(archive, mode="r:gz") as bundle:
            members = {PurePosixPath(member.name): member for member in bundle}
            for filename, expected_md5 in TRAIN_FILES.items():
                member_name = ARCHIVE_ROOT / filename
                member = members.get(member_name)
                if member is None or not member.isfile():
                    raise RuntimeError(f"missing regular file: {member_name}")

                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError(f"cannot read: {member_name}")
                output = destination / filename
                with source, output.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if md5(output) != expected_md5:
                    raise RuntimeError(f"checksum mismatch: {filename}")

    manifest = {
        "schema_version": 1,
        "id": "cifar10",
        "display_name": "CIFAR-10",
        "task": "image_classification",
        "default_split": "train",
        "splits": {"train": {"num_examples": 50_000}},
        "classes": list(CLASSES),
        "source": {"url": CANONICAL_URL, "archive_md5": ARCHIVE_MD5},
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    if len(sys.argv) not in {2, 3}:
        raise SystemExit("usage: prepare.py DESTINATION [SOURCE]")
    main(Path(sys.argv[1]))
