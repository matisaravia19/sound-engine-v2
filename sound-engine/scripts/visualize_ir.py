#!/usr/bin/env python3
"""Visualize an impulse response exported by sound_engine::debug::export_ir."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def read_ir(path: Path) -> tuple[dict[str, str], list[float], dict[str, list[float]]]:
    metadata: dict[str, str] = {}
    rows: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as ir_file:
        for line in ir_file:
            if line.startswith("#"):
                key_value = line[1:].strip().split("=", 1)
                if len(key_value) == 2:
                    metadata[key_value[0].strip()] = key_value[1].strip()
                continue
            rows.append(line)

    reader = csv.DictReader(rows)
    if not reader.fieldnames:
        raise ValueError("IR CSV is missing a header row")

    channel_names = ["left", "right"]
    missing_columns = [name for name in ("time_seconds", *channel_names) if name not in reader.fieldnames]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"IR CSV is missing required column(s): {missing}")

    times_ms: list[float] = []
    channels: dict[str, list[float]] = {name: [] for name in channel_names}
    for row in reader:
        times_ms.append(float(row["time_seconds"]) * 1000.0)
        for channel_name in channel_names:
            channels[channel_name].append(float(row[channel_name]))

    return metadata, times_ms, channels


def bin_ir(
    times_ms: list[float],
    channels: dict[str, list[float]],
    bin_size: int,
    mode: str,
) -> tuple[list[float], dict[str, list[float]]]:
    if not channels:
        return times_ms, channels

    first_channel = next(iter(channels.values()))
    binned_times: list[float] = []
    binned_channels: dict[str, list[float]] = {name: [] for name in channels}

    for start in range(0, len(first_channel), bin_size):
        bin_times = times_ms[start : start + bin_size]
        if not bin_times:
            continue

        binned_times.append(sum(bin_times) / len(bin_times))
        for channel_name, amplitudes in channels.items():
            bin_amplitudes = amplitudes[start : start + bin_size]
            if mode == "mean":
                binned_channels[channel_name].append(sum(bin_amplitudes) / len(bin_amplitudes))
            elif mode == "rms":
                energy = sum(sample * sample for sample in bin_amplitudes)
                binned_channels[channel_name].append((energy / len(bin_amplitudes)) ** 0.5)
            elif mode == "peak":
                binned_channels[channel_name].append(max(bin_amplitudes, key=abs))
            else:
                raise ValueError(f"unsupported bin mode {mode}")

    return binned_times, binned_channels


def samples_per_ms(metadata: dict[str, str], times_ms: list[float]) -> float:
    sample_rate = metadata.get("sample_rate")
    if sample_rate:
        return float(sample_rate) / 1000.0
    if len(times_ms) < 2:
        raise ValueError("cannot infer sample rate from fewer than two samples")

    sample_spacing_ms = times_ms[1] - times_ms[0]
    if sample_spacing_ms <= 0.0:
        raise ValueError("cannot infer sample rate from non-increasing sample times")
    return 1.0 / sample_spacing_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir_csv", type=Path, help="CSV file exported by the debug IR exporter")
    parser.add_argument("--output", "-o", type=Path, help="Optional image output path")
    smoothing = parser.add_mutually_exclusive_group()
    smoothing.add_argument("--bin-samples", type=positive_int, help="Plot one aggregate point per N samples")
    smoothing.add_argument("--bin-ms", type=positive_float, help="Plot one aggregate point per time bin in milliseconds")
    parser.add_argument(
        "--bin-mode",
        choices=("rms", "mean", "peak"),
        default="rms",
        help="Aggregation mode used with --bin-samples or --bin-ms",
    )
    args = parser.parse_args()

    metadata, times_ms, channels = read_ir(args.ir_csv)
    sample_rate = metadata.get("sample_rate", "unknown")
    query_id = metadata.get("query_id", "unknown")
    energy = metadata.get("energy", "unknown")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is required to plot IR files. Install it with: python -m pip install matplotlib", file=sys.stderr)
        raise SystemExit(1)

    _, axis = plt.subplots(figsize=(12, 5))
    if args.bin_ms is not None:
        bin_size = max(1, round(args.bin_ms * samples_per_ms(metadata, times_ms)))
        times_ms, channels = bin_ir(times_ms, channels, bin_size, args.bin_mode)
    elif args.bin_samples is not None:
        times_ms, channels = bin_ir(times_ms, channels, args.bin_samples, args.bin_mode)

    axis.plot(times_ms, channels["left"], linewidth=1.0, label="left", color="tab:blue")
    axis.plot(times_ms, channels["right"], linewidth=1.0, label="right", color="tab:orange")
    axis.legend()
    axis.set_title(f"Impulse response query={query_id}, sample_rate={sample_rate} Hz, energy={energy}")
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel(f"Amplitude ({args.bin_mode} bins)" if args.bin_ms or args.bin_samples else "Amplitude")
    axis.grid(True, alpha=0.3)

    if args.output:
        plt.savefig(args.output, dpi=160, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
