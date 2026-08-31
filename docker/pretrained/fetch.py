#!/usr/bin/env python3
"""Fetch the immutable pretrained registry during the trusted image build."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import BinaryIO


ALLOWED_SCHEME = "https"
ALLOWED_HOST = "download.pytorch.org"
CHUNK_BYTES = 1024 * 1024


def copy_and_hash(source: BinaryIO, destination: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    while chunk := source.read(CHUNK_BYTES):
        destination.write(chunk)
        digest.update(chunk)
        total += len(chunk)
    return total, digest.hexdigest()


def fetch_model(model: dict[str, object], output_root: Path) -> None:
    url = str(model["url"])
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != ALLOWED_SCHEME or parsed.hostname != ALLOWED_HOST:
        raise ValueError(f"pretrained URL is outside the allowlist: {url}")

    filename = str(model["filename"])
    if Path(filename).name != filename:
        raise ValueError(f"pretrained filename is not a basename: {filename}")
    expected_bytes = int(model["bytes"])
    expected_sha256 = str(model["sha256"])
    destination = output_root / filename

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", dir=output_root)
    try:
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(url) as response:
            actual_bytes, actual_sha256 = copy_and_hash(response, output)
            output.flush()
            os.fsync(output.fileno())
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{filename}: expected {expected_bytes} bytes, downloaded {actual_bytes}"
            )
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{filename}: expected sha256 {expected_sha256}, got {actual_sha256}"
            )
        os.chmod(temporary_name, 0o444)
        os.replace(temporary_name, destination)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: fetch.py MANIFEST OUTPUT_ROOT")
    manifest_path = Path(sys.argv[1])
    output_root = Path(sys.argv[2])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported pretrained manifest schema")
    output_root.mkdir(parents=True, exist_ok=True)
    for model in manifest["models"]:
        fetch_model(model, output_root)
    destination_manifest = output_root / "manifest.json"
    destination_manifest.write_bytes(manifest_path.read_bytes())
    os.chmod(destination_manifest, 0o444)


if __name__ == "__main__":
    main()
