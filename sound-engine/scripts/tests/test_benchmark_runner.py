"""Regression checks for recorded runs without launching GPU benchmarks."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_runner import ROOT, BenchmarkRun, file_hash, source_hashes, verify_hashes, write_json


class BenchmarkRunTests(unittest.TestCase):
    """Exercise process isolation, record preservation, and failure boundaries."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        memory = patch(
            "benchmark_runner.memory_status",
            return_value={"available_physical": 16 * 1024**3, "available_commit": 16 * 1024**3},
        )
        self.memory = memory.start()
        self.addCleanup(memory.stop)

    def test_output_and_environment_are_isolated_between_runs(self):
        first = BenchmarkRun(self.output / "first")
        second = BenchmarkRun(self.output / "second")
        key = "BENCHMARK_RUN_TEST_VALUE"
        command = [sys.executable, "-c", f"import os; print(os.environ.get('{key}', 'unset'))"]
        with patch.dict(os.environ, {key: "parent"}):
            self.assertEqual(first.run("probe", command, environment={key: "child"}), 0)
            self.assertEqual(second.run("probe", command), 0)
            self.assertEqual(os.environ[key], "parent")
        self.assertEqual((first.output / "probe.stdout").read_text().strip(), "child")
        self.assertEqual((second.output / "probe.stdout").read_text().strip(), "parent")

    def test_completed_records_are_never_overwritten(self):
        runner = BenchmarkRun(self.output)
        record = self.output / "probe.run.json"
        record.write_text("preserved", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Preserving existing run"):
            runner.run("probe", [sys.executable, "-c", "raise SystemExit(0)"])
        self.assertEqual(record.read_text(encoding="utf-8"), "preserved")
        self.memory.assert_not_called()

    def test_insufficient_memory_stops_before_creating_logs(self):
        self.memory.return_value["available_physical"] = 0
        runner = BenchmarkRun(self.output)
        with self.assertRaisesRegex(RuntimeError, "Insufficient headroom"):
            runner.run("probe", [sys.executable, "-c", "raise SystemExit(0)"])
        self.assertEqual(list(self.output.iterdir()), [])

    def test_correctness_failures_remain_recorded_results(self):
        runner = BenchmarkRun(self.output)
        self.assertEqual(runner.run("failure", [sys.executable, "-c", "raise SystemExit(7)"]), 7)
        record = json.loads((self.output / "failure.run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["exit_code"], 7)

    def test_device_failures_are_recorded_before_stopping(self):
        runner = BenchmarkRun(self.output)
        command = [sys.executable, "-c", "import sys; sys.stderr.write('DEVICE_LOST')"]
        with self.assertRaisesRegex(RuntimeError, "resource/device failure"):
            runner.run("failure", command)
        self.assertTrue((self.output / "failure.run.json").is_file())

    def test_provenance_rejects_added_sources(self):
        runner = BenchmarkRun(self.output)
        recorded = {"source.rs": "original"}
        write_json(self.output / "provenance.json", {"source_sha256": recorded})
        with patch("benchmark_runner.source_hashes", return_value=recorded):
            runner.verify_provenance()
        with patch("benchmark_runner.source_hashes", return_value={**recorded, "new.rs": "new"}):
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                runner.verify_provenance()

    def test_input_hashes_reject_changed_and_missing_fixtures(self):
        fixture = self.output / "scene.txt"
        fixture.write_text("original", encoding="utf-8")
        expected = {fixture.name: file_hash(fixture)}
        verify_hashes(self.output, expected)
        fixture.write_text("modified", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Input changed"):
            verify_hashes(self.output, expected)
        fixture.unlink()
        with self.assertRaisesRegex(RuntimeError, "Input changed"):
            verify_hashes(self.output, expected)

    def test_provenance_rejects_build_and_test_input_changes(self):
        runner = BenchmarkRun(self.output)
        write_json(self.output / "provenance.json", {"source_sha256": source_hashes()})
        for name in (
            "vkfft-rs/build_support.rs",
            "vkfft-rs/native/ffi.cpp",
            "vkfft-rs/native/codegen/names.hpp",
            "vkfft-rs/native/scheduler/limits.hpp",
            "vkfft-rs/native/lut/storage.hpp",
            "vkfft-rs/native/vulkan/memory.hpp",
            "vkfft-rs/patches/compiler_cleanup.patch",
            "vkfft-rs/lib/vkfft/vkFFT/vkFFT.h",
            "vkfft-rs/tests/native/vulkan_memory_tests.cpp",
            "acoustic-fixtures/tests/unit/text.rs",
            "sandbox/tests/unit/app/config.rs",
            "sound-engine/tests/unit/engine/direct.rs",
            "sound-engine/scripts/tests/test_benchmark_runner.py",
        ):
            with self.subTest(name=name):
                changed = ROOT / name

                def changed_hash(path):
                    return "modified" if path == changed else file_hash(path)

                with patch("benchmark_runner.file_hash", side_effect=changed_hash):
                    with self.assertRaisesRegex(RuntimeError, "Source changed"):
                        runner.verify_provenance()


if __name__ == "__main__":
    unittest.main()
