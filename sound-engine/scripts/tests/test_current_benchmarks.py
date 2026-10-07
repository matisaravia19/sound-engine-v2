"""Check build provenance without compiling or running audio workloads."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import run_current_benchmarks as benchmarks


class BuildVerificationTests(unittest.TestCase):
    """A recorded stage must execute the binaries produced by its recorded build."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.examples = self.root / "target/release/examples"
        self.examples.mkdir(parents=True)
        self.runner = Mock(output=self.root / "output")
        self.runner.output.mkdir()
        self.binaries = [self.examples / f"{name}.exe" for name in benchmarks.BENCHMARK_EXAMPLES]
        self.binaries.append(self.examples.parent / "sandbox.exe")
        for path in self.binaries:
            path.write_bytes(path.name.encode())
        for name, value in (("ROOT", self.root), ("EXAMPLES", self.examples)):
            patched = patch.object(benchmarks, name, value)
            patched.start()
            self.addCleanup(patched.stop)

    def test_build_excludes_stale_executables_and_includes_playback(self):
        (self.examples / "removed_demo.exe").write_bytes(b"stale")
        benchmarks.build(self.runner)
        recorded = json.loads((self.runner.output / "binaries.json").read_text(encoding="utf-8"))
        self.assertEqual(
            set(recorded), {path.relative_to(self.root).as_posix() for path in self.binaries}
        )
        benchmarks.verify_build(self.runner)
        self.runner.verify_provenance.assert_called_once()

    def test_rebuilt_or_missing_executables_prevent_subsequent_stages(self):
        benchmarks.build(self.runner)
        for path in (self.binaries[0], self.binaries[-1]):
            original = path.read_bytes()
            for replacement in (b"rebuilt", None):
                with self.subTest(binary=path.name, replacement=replacement):
                    if replacement is None:
                        path.unlink()
                    else:
                        path.write_bytes(replacement)
                    with self.assertRaisesRegex(RuntimeError, "Input changed"):
                        benchmarks.verify_build(self.runner)
                    path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
