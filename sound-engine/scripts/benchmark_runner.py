"""Recorded benchmark execution, resource checks, and source provenance."""

import ctypes
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "target/release/examples"
BUILD_OPTIONS = [
    "--offline",
    "--release",
    "-j",
    "1",
    "--config",
    "profile.release.package.sound-engine.codegen-units=1",
    "-p",
    "sound-engine",
    "--features",
    "playback",
]


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong)
        for name in (
            "total_physical",
            "available_physical",
            "total_commit",
            "available_commit",
            "total_virtual",
            "available_virtual",
            "available_extended",
        )
    ]


def memory_status():
    """Read Windows memory headroom before starting another GPU workload."""
    status = _MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("Cannot check system memory")
    return {name: getattr(status, name) for name, _ in status._fields_}


def file_hash(path):
    """Hash a source, fixture, or executable without loading it all into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_hashes(root, expected):
    """Reject missing or modified inputs from a previously recorded run."""
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or file_hash(path) != digest:
            raise RuntimeError(f"Input changed since this run was recorded: {path}")


def source_hashes():
    """Fingerprint engine sources, tests, build inputs, and benchmark scripts."""
    files = []
    for directory in (
        "acoustic-fixtures/src",
        "sound-engine/src",
        "sound-engine/examples",
        "sound-engine/tests",
        "vkfft-rs/src",
        "sandbox/src",
    ):
        files.extend(path for path in (ROOT / directory).rglob("*") if path.is_file())
    files.extend((ROOT / "sound-engine/scripts").glob("*.py"))
    files.extend(
        ROOT / name
        for name in (
            "Cargo.toml",
            "Cargo.lock",
            "acoustic-fixtures/Cargo.toml",
            "sound-engine/Cargo.toml",
            "sound-engine/build.rs",
            "vkfft-rs/Cargo.toml",
            "vkfft-rs/build.rs",
            "sandbox/Cargo.toml",
            "sound-engine/scripts/pyproject.toml",
            "sound-engine/scripts/uv.lock",
        )
    )
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in sorted(files)}


def write_json(path, data):
    """Write a readable UTF-8 record with a trailing newline."""
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class BenchmarkRun:
    """Own one output directory; preserve completed records and run jobs serially."""

    def __init__(self, output):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)

    def run(self, name, command, timeout=180, *, environment=None):
        """Record a command; stop on timeout/device loss but return ordinary failures."""
        record_path = self.output / f"{name}.run.json"
        if record_path.exists():
            raise RuntimeError(f"Preserving existing run: {record_path}")
        before = memory_status()
        if before["available_physical"] < 4 * 1024**3 or before["available_commit"] < 8 * 1024**3:
            raise RuntimeError(f"Insufficient headroom; no process started: {before}")

        command = [str(argument) for argument in command]
        process_environment = dict(
            os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
        )
        process_environment.update(environment or {})
        print(f"START {name}", flush=True)
        started = time.perf_counter()
        record = {
            "command": command,
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "memory_before": before,
            "timeout_seconds": timeout,
        }
        stdout_path = self.output / f"{name}.stdout"
        stderr_path = self.output / f"{name}.stderr"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=process_environment,
                stdout=stdout,
                stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            try:
                record["exit_code"] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Cargo and native builds spawn children that must stop with the parent.
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True
                )
                process.wait(timeout=15)
                record["exit_code"] = "timeout"

        record["elapsed_seconds"] = time.perf_counter() - started
        record["memory_after"] = memory_status()
        write_json(record_path, record)
        error = stderr_path.read_text(encoding="utf-8", errors="replace").lower()
        resource_failure = any(
            token in error
            for token in (
                "device_lost",
                "out of memory",
                "lnk1102",
                "paging file",
                "1455",
            )
        )
        print(f"END {name}: {record['exit_code']} ({record['elapsed_seconds']:.1f}s)", flush=True)
        if resource_failure or record["exit_code"] == "timeout":
            raise RuntimeError(f"Stopping after resource/device failure: {name}")
        return record["exit_code"]

    def record_provenance(self, model, **extra):
        """Save the inputs of a fresh run before building or executing benchmarks."""
        path = self.output / "provenance.json"
        if path.exists():
            raise RuntimeError(f"Preserving existing provenance: {path}; use a fresh directory")
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
        )
        write_json(
            path,
            {
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "source_sha256": source_hashes(),
                "revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "build_command_prefix": ["cargo", "build", *BUILD_OPTIONS],
                "model": model,
                "memory": memory_status(),
                "gpu": gpu.stdout.strip(),
                **extra,
            },
        )

    def verify_provenance(self):
        """Require the exact recorded source tree, including newly added files."""
        recorded = json.loads((self.output / "provenance.json").read_text(encoding="utf-8"))
        if source_hashes() != recorded["source_sha256"]:
            raise RuntimeError("Source changed since this run was recorded; use a fresh directory")
        return recorded


def hardware_test(target, name=None):
    """Run one ignored integration-test binary with serial Vulkan access."""
    command = ["cargo", "test", *BUILD_OPTIONS, "--test", target]
    if name:
        command.append(name)
    return command + ["--", "--ignored", "--test-threads=1", "--nocapture"]
