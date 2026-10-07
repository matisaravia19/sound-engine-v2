"""Run the ray-traced wedge accuracy/radius regression against offline BTM references."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/diffraction-ray-local")
    parser.add_argument("--rays", type=int, default=1_048_576)
    args = parser.parse_args()
    if args.rays <= 0:
        parser.error("--rays must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "sound-engine",
            "--example",
            "acoustic_method_benchmark",
        ],
        cwd=ROOT,
        check=True,
    )
    executable = ROOT / "target/release/examples/acoustic_method_benchmark"
    if not executable.exists():
        executable = executable.with_suffix(".exe")
    summary = {}
    for case in [
        "diffraction_wedge_shadow",
        "diffraction_proxy_radius_020",
        "diffraction_proxy_radius_090",
    ]:
        summary[case] = {}
        for mode in ["proxy"]:
            output = (args.output / f"{case}-{mode}.csv").resolve()
            command = [
                str(executable),
                "--case",
                case,
                "--rays",
                str(args.rays),
                "--seed-repeats",
                "3",
                "--warm-repeats",
                "21",
                "--output",
                str(output),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == 21, "each case must export 3 seeds x 7 frequencies"
            values = [float(r["sampled_complex_relative_error"]) for r in rows]
            assert all(math.isfinite(v) for v in values), "nonfinite response"
            summary[case][mode] = {
                "mean_sampled_complex_error": statistics.mean(values),
                "max_sampled_complex_error": max(values),
                "mean_time_relative_l2": statistics.mean(
                    float(r["time_relative_l2"]) for r in rows
                ),
                "median_query_ms": statistics.median(float(r["warm_gpu_ms"]) for r in rows),
            }
    # Content hashes identify dirty/untracked source as well as the Git revision.
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ["sound-engine/src", "sound-engine/examples"]
        for p in sorted((ROOT / folder).rglob("*"))
        if p.suffix in [".rs", ".glsl"]
    }
    (args.output / "source-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
