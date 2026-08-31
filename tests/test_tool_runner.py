from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docker.odbench import tool_runner


class PatchRunnerTests(unittest.TestCase):
    def test_valid_patch_is_applied(self) -> None:
        patch_text = """diff --git a/result.txt b/result.txt
new file mode 100644
--- /dev/null
+++ b/result.txt
@@ -0,0 +1 @@
+valid
"""
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            with patch.object(tool_runner, "WORKSPACE", workspace):
                result = tool_runner.apply_patch({"patch": patch_text})
            content = (workspace / "result.txt").read_text(encoding="utf-8")

        self.assertTrue(result["applied"])
        self.assertEqual(content, "valid\n")

    def test_structural_warning_rejects_patch_before_writing_files(self) -> None:
        patch_text = """diff --git a/first.txt b/first.txt
new file mode 100644
--- /dev/null
+++ b/first.txt
@@ -0,0 +1 @@
+first
 diff --git a/second.txt b/second.txt
new file mode 100644
--- /dev/null
+++ b/second.txt
@@ -0,0 +1 @@
+second
"""
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            with patch.object(tool_runner, "WORKSPACE", workspace):
                result = tool_runner.apply_patch({"patch": patch_text})
            files = list(workspace.iterdir())

        self.assertFalse(result["applied"])
        self.assertIn("structural warnings", result["stderr"])
        self.assertEqual(files, [])


if __name__ == "__main__":
    unittest.main()
