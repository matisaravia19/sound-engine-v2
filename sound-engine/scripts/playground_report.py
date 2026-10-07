"""Summarize actual playground frame, render-block and audio callback diagnostics.

Use a shared filename prefix for -runtime.csv, -runtime.audio.json, -frames.csv,
-rendered.wav and .log. WAVs are rendered output before the device queue; they
cannot establish what the audio device actually played during an underrun.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def distribution(values):
    """Return explicitly labeled percentiles without dropping startup outliers."""
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("empty or nonfinite timing series")
    return dict(
        zip(
            ["p50_ms", "p95_ms", "p99_ms", "max_ms"],
            np.percentile(values, [50, 95, 99, 100]).tolist(),
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    args = parser.parse_args()

    def path(suffix):
        return Path(str(args.prefix) + suffix)

    render = np.atleast_1d(np.genfromtxt(path("-runtime.csv"), delimiter=",", names=True))
    frames = np.atleast_1d(np.genfromtxt(path("-frames.csv"), delimiter=",", names=True))
    callbacks = json.loads(path("-runtime.audio.json").read_text())
    rate, wave = wavfile.read(path("-rendered.wav"))
    if wave.ndim != 2 or wave.shape[1] != 2 or not len(wave):
        raise ValueError("expected nonempty stereo rendered output")
    if not np.issubdtype(wave.dtype, np.floating):
        raise ValueError("expected unscaled float rendered output")
    finite = bool(np.isfinite(wave).all())
    if not finite:
        raise ValueError("nonfinite rendered samples")
    worst = int(np.argmax(render["render_ms"]))
    configuration = next(
        (
            line
            for line in path(".log").read_text(encoding="utf-8").splitlines()
            if line.startswith("Playground model=")
        ),
        None,
    )
    if configuration is None:
        raise ValueError("missing playground model/configuration line in run log")
    report = {
        "prefix": str(args.prefix),
        "configuration": configuration,
        "render_blocks": len(render),
        "render": distribution(render["render_ms"]),
        "worst_block": {
            "index": int(render["block"][worst]),
            "elapsed_seconds": float(render["elapsed_seconds"][worst]),
        },
        "graphics_frames": len(frames),
        "graphics": distribution(frames["total_ms"]),
        "audio_callbacks": callbacks,
        "rendered_audio": {
            "sample_rate": rate,
            "frames": len(wave),
            "duration_seconds": len(wave) / rate,
            "all_finite": finite,
            "rms": np.sqrt(np.mean(wave.astype(float) ** 2, axis=0)).tolist(),
            "peak": float(np.max(abs(wave))),
            "samples_outside_unit_range": int(np.sum(abs(wave) > 1)),
            "channel_difference_rms": float(
                np.sqrt(np.mean((wave[:, 0] - wave[:, 1]).astype(float) ** 2))
            ),
        },
        "scope": "One moving-camera playground run. Includes startup timings. Capture precedes device queue. No hard real-time guarantee.",
    }
    report["input_sha256"] = {
        suffix: hashlib.sha256(path(suffix).read_bytes()).hexdigest()
        for suffix in [
            "-runtime.csv",
            "-runtime.audio.json",
            "-frames.csv",
            "-rendered.wav",
            ".log",
        ]
    }
    stages = ["trace_ms", "ir_prepare_ms", "convolution_ms"]
    if all(stage in render.dtype.names for stage in stages):
        report["stages"] = {stage: distribution(render[stage]) for stage in stages}
        report["worst_block"].update({stage: float(render[stage][worst]) for stage in stages})
    if "spatial_updates" in render.dtype.names:
        counts = render["spatial_updates"]
        if np.any(np.diff(counts) < 0):
            raise ValueError("spatial update counter went backwards")
        changed = np.diff(counts, prepend=0) > 0
        report["spatial_updates"] = {
            "ir_replacements": int(counts[-1]),
            "deferred_blocks": int(np.sum(render["spatial_deferred"])),
            "age_scope": "Engine acceptance of the captured pose/source revision to IR crossfade registration; excludes the input mailbox, device queue, waiting for an existing crossfade and the new crossfade duration.",
        }
        if np.any(changed):
            report["spatial_updates"]["request_to_registration"] = distribution(
                render["spatial_update_age_ms"][changed]
            )
        if np.sum(changed) > 1:
            report["spatial_updates"]["replacement_interval"] = distribution(
                np.diff(render["elapsed_seconds"][changed]) * 1000
            )
    path("-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
