"""Run prepared BRAS setups serially with resource guards and provenance."""

import argparse
import json
import shutil
from pathlib import Path

from benchmark_runner import EXAMPLES, ROOT, BenchmarkRun, file_hash, verify_hashes


def prepare_run(base, stage):
    """Record a fresh stage or verify the sources, fixtures, and binary on resume."""
    runner = BenchmarkRun(base / stage)
    binary = EXAMPLES / "bras_extended.exe"
    if not (runner.output / "provenance.json").exists():
        runner.record_provenance(
            "Geometry-aware scalar ray pressure, R1 reflection; no frequency payload",
            binary_sha256=file_hash(binary),
            fixtures_sha256={
                path.relative_to(base).as_posix(): file_hash(path)
                for path in sorted((base / "fixtures").glob("*"))
                if path.is_file()
            },
        )
        shutil.copyfile(
            ROOT / "sound-engine/examples/bras_extended.rs", runner.output / "bras_extended.rs"
        )

    recorded = runner.verify_provenance()
    verify_hashes(base, recorded["fixtures_sha256"])
    verify_hashes(EXAMPLES, {binary.name: recorded["binary_sha256"]})
    return runner


def run_setup(runner, base, stage, setup):
    """Resume successful setups and require a passing smoke run before a full run."""
    name = setup["name"]
    record_path = runner.output / f"{name}.run.json"
    if record_path.exists():
        if json.loads(record_path.read_text(encoding="utf-8"))["exit_code"] != 0:
            raise RuntimeError(f"Previous failure: {name}")
        return

    if stage == "full":
        smoke_path = base / "smoke" / f"{name}.run.json"
        if json.loads(smoke_path.read_text(encoding="utf-8"))["exit_code"] != 0:
            raise RuntimeError(f"Smoke run failed: {name}")

    smoke = stage == "smoke"
    command = [
        EXAMPLES / "bras_extended.exe",
        "--fixture",
        base / "fixtures" / f"{name}.scene",
        "--positions",
        base / "fixtures" / f"{name}.csv",
        "--rays",
        4096 if smoke else 65536,
        "--bounces",
        3 if stage == "realtime" else setup["bounces"],
        "--seconds",
        setup["seconds"],
        "--seeds",
        1 if smoke else 3,
        "--repeats",
        1 if smoke else 5,
        "--output",
        runner.output / name,
    ]
    if stage != "full":
        command += ["--case", setup["pairs"][0]["case"]]
    runner.run(name, command)


def main():
    """Select prepared fixtures and a smoke, full, or realtime benchmark stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["smoke", "full", "realtime"])
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--base", type=Path, default=ROOT / "output/bras-extended-20260909")
    args = parser.parse_args()
    base = args.base.resolve()
    manifest = json.loads((base / "fixtures/manifest.json").read_text(encoding="utf-8"))
    if args.only:
        unknown = set(args.only) - {setup["name"] for setup in manifest}
        if unknown:
            parser.error(f"Unknown setups: {', '.join(sorted(unknown))}")

    runner = prepare_run(base, args.stage)
    for setup in manifest:
        if args.only and setup["name"] not in args.only:
            continue
        if args.stage == "realtime" and setup["bounces"] == 3:
            continue
        run_setup(runner, base, args.stage, setup)


if __name__ == "__main__":
    main()
