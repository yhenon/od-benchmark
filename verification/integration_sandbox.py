"""Exercise workspace exec and patch inside the real isolated Docker image."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from odbench_outer.sandbox import Sandbox


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sandbox = Sandbox(
        repo_root=REPO_ROOT,
        container=f"odbench-agent-verify-{uuid.uuid4().hex[:8]}",
        dataset="cifar10",
        image="od-benchmark-agent:cifar10-dev",
        max_command_seconds=5,
    )
    try:
        sandbox.start()
        identity = sandbox.exec("printf 'hello'; printf 'warning' >&2; pwd", 5)
        assert identity["exit_code"] == 0, identity
        assert identity["stdout"] == "hello/workspace\n", identity
        assert identity["stderr"] == "warning", identity
        assert not identity["timed_out"], identity

        patch = '''diff --git a/folder/hello.py b/folder/hello.py
new file mode 100644
--- /dev/null
+++ b/folder/hello.py
@@ -0,0 +1 @@
+print("patched")
'''
        applied = sandbox.apply_patch(patch)
        assert applied["applied"], applied
        executed = sandbox.exec("python folder/hello.py", 5)
        assert executed["stdout"] == "patched\n", executed

        with tempfile.TemporaryDirectory() as copied:
            exported = Path(copied) / "exported"
            sandbox.copy_directory("folder", exported)
            assert (exported / "hello.py").is_file(), list(exported.rglob("*"))

        linked = sandbox.exec("ln -s /etc/passwd folder/leak", 5)
        assert linked["exit_code"] == 0, linked
        with tempfile.TemporaryDirectory() as copied:
            try:
                sandbox.copy_directory("folder", Path(copied) / "unsafe")
            except Exception as error:
                assert "symlink" in str(error), error
            else:
                raise AssertionError("workspace export accepted a symlink")
        removed = sandbox.exec("rm folder/leak", 5)
        assert removed["exit_code"] == 0, removed

        try:
            sandbox.apply_patch(patch)
        except Exception as error:
            assert "already exists" in str(error) or "does not apply" in str(error), error
        else:
            raise AssertionError("duplicate patch was not reported as a tool failure")

        timeout = sandbox.exec("sleep 5", 0.2)
        assert timeout["timed_out"], timeout
        assert timeout["duration_ms"] < 3000, timeout

        isolation = sandbox.exec(
            "python -c 'import os,socket; "
            "assert os.getuid()==10001; "
            "assert not os.path.exists(\"/var/run/docker.sock\"); "
            "s=socket.socket(); s.settimeout(.2); "
            "exec(\"try:\\n s.connect((\\\"1.1.1.1\\\",53))\\nexcept OSError:\\n pass\\nelse:\\n raise AssertionError(\\\"network available\\\")\")'",
            5,
        )
        assert isolation["exit_code"] == 0, isolation
        print("workspace tools: exec=ok patch=ok timeout=ok isolation=ok")
    finally:
        sandbox.stop()


if __name__ == "__main__":
    main()
