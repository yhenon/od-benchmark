from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from odbench_outer.tools import _pretrained_context


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "docker" / "pretrained" / "manifest.json"
FETCH_PATH = REPO_ROOT / "docker" / "pretrained" / "fetch.py"


class PretrainedRegistryTests(unittest.TestCase):
    def test_manifest_entries_are_immutable_and_unique(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["torchvision_version"], "0.28.0")
        models = manifest["models"]
        self.assertEqual(len(models), 4)
        self.assertEqual(len({model["id"] for model in models}), len(models))
        self.assertEqual(len({model["filename"] for model in models}), len(models))
        self.assertEqual({model["kind"] for model in models}, {"backbone", "detector"})
        for model in models:
            self.assertRegex(model["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(model["bytes"], 0)
            self.assertTrue(model["url"].startswith("https://download.pytorch.org/models/"))
            self.assertEqual(model["sha256"][:8], model["filename"].rsplit("-", 1)[1][:8])
            if model["kind"] == "backbone":
                self.assertEqual(model["feature_reductions"], [4, 8, 16, 32])
                self.assertEqual(len(model["feature_channels"]), 4)

    def test_stream_copy_reports_digest_and_byte_count(self) -> None:
        specification = importlib.util.spec_from_file_location("fetch_pretrained", FETCH_PATH)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        payload = b"immutable model bytes"
        destination = io.BytesIO()
        size, digest = module.copy_and_hash(io.BytesIO(payload), destination)
        self.assertEqual(size, len(payload))
        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
        self.assertEqual(destination.getvalue(), payload)

    def test_agent_context_advertises_api_without_download_details(self) -> None:
        context = _pretrained_context(REPO_ROOT)
        self.assertTrue(context["available"])
        self.assertTrue(context["offline"])
        self.assertEqual(len(context["models"]), 4)
        self.assertIn("load_backbone", context["api"]["backbone"])
        detector = next(model for model in context["models"] if model["kind"] == "detector")
        self.assertEqual(detector["gflops_320"], 0.583)
        self.assertEqual(detector["normalization"]["mean"], [0.5, 0.5, 0.5])
        self.assertIn("initial sizing", context["note"])
        encoded = json.dumps(context)
        self.assertNotIn("download.pytorch.org", encoded)
        self.assertNotIn("sha256", encoded)

    def test_missing_registry_has_explicit_empty_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _pretrained_context(Path(temporary))
        self.assertFalse(context["available"])
        self.assertEqual(context["models"], [])


if __name__ == "__main__":
    unittest.main()
