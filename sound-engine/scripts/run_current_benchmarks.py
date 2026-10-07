"""Run recorded benchmark stages for the current broadband thesis model."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmark_runner import (
    BUILD_OPTIONS,
    EXAMPLES,
    ROOT,
    BenchmarkRun,
    file_hash,
    hardware_test,
    verify_hashes,
    write_json,
)

BENCHMARK_EXAMPLES = (
    "bras_diffraction",
    "acoustic_method_benchmark",
    "bell_labs_box",
    "convolution_method_benchmark",
    "auralization_realtime_benchmark",
    "spatial_update_realtime_benchmark",
    "btm_realtime_benchmark",
)


def build(runner):
    """Build retained targets and fingerprint the benchmark and playback binaries."""
    runner.record_provenance("Broadband cosine transport; preserved surface suffix; UTD disabled")
    example_options = [argument for name in BENCHMARK_EXAMPLES for argument in ("--example", name)]
    runner.run("build-examples", ["cargo", "build", *BUILD_OPTIONS, *example_options], 300)
    test_build = ["cargo", "test", *BUILD_OPTIONS] + ["--tests", "--no-run"]
    runner.run("build-tests", test_build, 300)
    sandbox_build = ["cargo", "build", "--offline", "--release", "-j", "1", "-p", "sandbox"]
    runner.run("build-sandbox", sandbox_build, 300)
    binaries = [EXAMPLES / f"{name}.exe" for name in BENCHMARK_EXAMPLES]
    binaries.append(ROOT / "target/release/sandbox.exe")
    write_json(
        runner.output / "binaries.json",
        {path.relative_to(ROOT).as_posix(): file_hash(path) for path in binaries},
    )


def verify_build(runner):
    """Reject changed sources or replaced benchmark/playback binaries before a stage."""
    runner.verify_provenance()
    binaries = json.loads((runner.output / "binaries.json").read_text(encoding="utf-8"))
    verify_hashes(ROOT, binaries)


def accuracy(runner):
    """Compare transport against analytical and measured reference scenes."""
    output = runner.output
    runner.run(
        "listener-validation",
        hardware_test("listener_backward_validation", "listener_transport_matches_reference"),
    )
    listing = subprocess.check_output(
        [EXAMPLES / "acoustic_method_benchmark.exe", "--list-cases"], text=True
    )
    (output / "analytical").mkdir(exist_ok=True)
    for case in listing.splitlines():
        runner.run(
            f"acoustic-{case}",
            [
                EXAMPLES / "acoustic_method_benchmark.exe",
                "--case",
                case,
                "--rays",
                "65536",
                "--seed",
                "7",
                "--seed-repeats",
                "3",
                "--warm-repeats",
                "21",
                "--output",
                output / "analytical" / f"{case}.csv",
            ],
        )
    for scene in ["RS5", "RS6"]:
        runner.run(
            f"bras-{scene}",
            [
                EXAMPLES / "bras_diffraction.exe",
                "--scene",
                scene,
                "--rays",
                "65536",
                "--seeds",
                "3",
                "--bounces",
                "3",
                "--radius",
                "1.25",
                "--warmups",
                "3",
                "--repeats",
                "21",
                "--output",
                output / f"bras-{scene}",
            ],
            300,
        )
    runner.run(
        "bell-box",
        [
            EXAMPLES / "bell_labs_box.exe",
            "--rays",
            "65536",
            "--seeds",
            "3",
            "--repeats",
            "21",
            "--output",
            output / "bell-box",
        ],
        300,
    )


def runtime(runner):
    """Measure convolution and spatial update cost after continuity checks."""
    for name in ["sine_convolution_validation", "moving_ir_transition_validation"]:
        runner.run(name, hardware_test(name), 300)
    for name in [
        "convolution_method_benchmark",
        "auralization_realtime_benchmark",
        "spatial_update_realtime_benchmark",
        "btm_realtime_benchmark",
    ]:
        runner.run(name, [EXAMPLES / f"{name}.exe"], 300)


def playback(runner):
    """Capture sandbox playback and frame timing for two ray budgets."""
    output = runner.output
    for label, rays, seconds in [("playground-short", 4096, 10), ("playground-default", 65536, 30)]:
        prefix = output / label
        environment = {
            "SOUND_ENGINE_TIMING_CSV": f"{prefix}-runtime.csv",
            "SOUND_ENGINE_CAPTURE_WAV": f"{prefix}-rendered.wav",
        }
        runner.run(
            label,
            [
                ROOT / "target/release/sandbox.exe",
                f"--rays={rays}",
                "--diffraction-radius=1.25",
                "--bounces=3",
                f"--camera-stress-seconds={seconds}",
                f"--frame-report={prefix}-frames.csv",
            ],
            seconds + 45,
            environment=environment,
        )
        Path(str(prefix) + ".log").write_bytes(
            (output / f"{label}.stdout").read_bytes() + (output / f"{label}.stderr").read_bytes()
        )


def checks(runner):
    """Run CPU checks and the retained reflection integration tests."""
    test = ["cargo", "test", *BUILD_OPTIONS]
    runner.run(
        "cpu-checks",
        test
        + [
            "--lib",
            "--example",
            "acoustic_method_benchmark",
            "--example",
            "bell_labs_box",
            "--example",
            "bras_diffraction",
            "--test",
            "listener_backward_validation",
            "--",
            "--test-threads=1",
        ],
        300,
    )
    runner.run(
        "proxy-gpu-checks",
        test
        + [
            "--test",
            "listener_backward_validation",
            "reflection_tests",
            "--",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ],
        120,
    )


def reports(runner):
    """Generate reports from the recorded measurement and playback runs."""
    output = runner.output
    for scene in ["RS5", "RS6"]:
        runner.run(
            f"report-bras-{scene}",
            [
                sys.executable,
                ROOT / "sound-engine/scripts/bras_reference.py",
                output / f"bras-{scene}",
                "--scene",
                scene,
                "--latency-calibration",
                ROOT / "output/measured-reference-research/source-latency.json",
            ],
        )
    runner.run(
        "report-bell-box",
        [
            sys.executable,
            ROOT / "sound-engine/scripts/bell_labs_box_report.py",
            output / "bell-box",
        ],
    )
    for label in ["playground-short", "playground-default"]:
        if (output / f"{label}-runtime.audio.json").exists():
            runner.run(
                f"report-{label}",
                [
                    sys.executable,
                    ROOT / "sound-engine/scripts/playground_report.py",
                    output / label,
                ],
            )


STAGES = {
    "build": build,
    "accuracy": accuracy,
    "runtime": runtime,
    "playback": playback,
    "checks": checks,
    "reports": reports,
}


def main():
    """Select a benchmark stage and verify its recorded build inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output/benchmarks",
        help="Fresh result directory; completed runs are never overwritten",
    )
    args = parser.parse_args()
    runner = BenchmarkRun(args.output)
    if args.stage != "build":
        verify_build(runner)
    STAGES[args.stage](runner)


if __name__ == "__main__":
    main()
