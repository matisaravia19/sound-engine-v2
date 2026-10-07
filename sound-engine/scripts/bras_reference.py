"""Compare ray-traced propagation IRs with the calibrated BRAS RS5/RS6 measurements.

The loudspeaker's independently measured on-axis response is a common measurement
operator, not an optimized diffraction gain. No per-pair time shift or gain is fitted.
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
from scipy.signal import butter, fftconvolve, sosfiltfilt

from visualize_ir import read_ir


def source_calibration(directory):
    """Return the measured on-axis FIR scaled to 80 dB SPL at 1 kHz and 2 m."""
    with (directory / "genelec-on-axis.csv").open() as stream:
        row = stream.readline().strip().split(",")
    assert row[0] == "P000T000"
    fir = np.array([float(v) for v in row[1:] if v])
    response_1khz = np.sum(fir * np.exp(-2j * np.pi * 1000 * np.arange(len(fir)) / 44100))
    return fir * (2 * 20e-6 * 10 ** (80 / 20) / abs(response_1khz))


def band_levels(wave, rate, centers):
    """Mean squared transfer magnitude within fixed third-octave bands, in dB re 1 Pa."""
    wave = np.array(wave, copy=True)
    # The same early-response window suppresses later chamber contamination in
    # both signals. This window is fixed before inspecting pairwise errors.
    stop = int(0.04 * rate)
    wave = wave[:stop]
    fade = int(0.003 * rate)
    wave[-fade:] *= np.cos(np.linspace(0, np.pi / 2, fade)) ** 2
    frequency = np.fft.rfftfreq(32768, 1 / rate)
    power = abs(np.fft.rfft(wave, 32768)) ** 2
    return np.array(
        [
            10
            * np.log10(
                max(
                    np.mean(
                        power[(frequency >= f / 2 ** (1 / 6)) & (frequency < f * 2 ** (1 / 6))]
                    ),
                    1e-30,
                )
            )
            for f in centers
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulation", type=Path)
    parser.add_argument("--data", type=Path, default=Path("output/measured-reference-research"))
    parser.add_argument("--scene", choices=["RS5", "RS6"], default="RS5")
    parser.add_argument(
        "--seed", type=int, help="Validate a single update's seed without averaging"
    )
    parser.add_argument(
        "--latency-calibration",
        type=Path,
        help="One fixed latency calibration from a held-out direct-sound control",
    )
    args = parser.parse_args()
    scene = args.scene
    runs_path = args.simulation / f"{scene}-runs.csv"
    run_rows = []
    if runs_path.exists():
        with runs_path.open() as stream:
            run_rows = list(csv.DictReader(stream))
        if any(row.get("reverse", "false") != "false" for row in run_rows):
            raise ValueError("reversed endpoints do not match the measured BRAS configuration")
        obsolete_features = (
            "utd",
            "source_directivity",
            "source_surface_paths",
            "surface_two_edge_paths",
        )
        if any(
            row.get(feature, "false") != "false"
            for row in run_rows
            for feature in obsolete_features
        ):
            raise ValueError(
                "run metadata uses a retired transport model; use its historical report script"
            )
        if len({(row["case"], row["seed"]) for row in run_rows}) != len(run_rows):
            raise ValueError("duplicate case/seed entries in run metadata")
    label = scene if args.seed is None else f"{scene}-seed{args.seed}"
    latency = json.loads(args.latency_calibration.read_text()) if args.latency_calibration else None
    source = source_calibration(args.data)
    centers = np.array([125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600])
    records, waves = [], []
    with h5py.File(args.data / scene / f"{scene}_RIRs.sofa") as measured:
        rate = int(measured["Data.SamplingRate"][0])
        size = len(measured["EmitterID"])
        side = round(np.sqrt(size))
        fig, axes = plt.subplots(
            side, side, figsize=(12, 9), sharex=True, sharey=True, layout="constrained"
        )
        for i in range(size):
            ls, mp = int(measured["EmitterID"][i, 0]), int(measured["ReceiverID"][i, 0])
            name = f"{scene}_LS{ls}_MP{mp}"
            files = sorted(args.simulation.glob(f"{name}-seed*.csv"))
            if args.seed is not None:
                files = [p for p in files if p.name == f"{name}-seed{args.seed}.csv"]
            if run_rows:
                expected_files = {
                    args.simulation / f"{name}-seed{row['seed']}.csv"
                    for row in run_rows
                    if row["case"] == name and (args.seed is None or int(row["seed"]) == args.seed)
                }
                if set(files) != expected_files:
                    raise ValueError(
                        f"ray files disagree with declared case/seed/model metadata for {name}; use a fresh output directory"
                    )
            if not files:
                raise ValueError(f"missing simulation: {name}")
            signals = []
            for file in files:
                metadata, _, channels = read_ir(file)
                assert int(metadata["sample_rate"]) == rate
                signals.append(channels["left"])
            raw = np.mean(signals, axis=0)
            prediction = fftconvolve(raw, source)[: len(raw)]
            if latency:
                advance = int(latency["advance_samples"])
                prediction = np.pad(prediction[advance:], (0, advance))
            actual = measured["Data.IR"][i, 0, :]
            expected, predicted = (
                band_levels(actual, rate, centers),
                band_levels(prediction, rate, centers),
            )
            error = predicted - expected
            sos = butter(4, [100, 1800], fs=rate, btype="bandpass", output="sos")
            size_common = min(len(actual), len(prediction))
            segment = slice(int(0.012 * rate), min(int(0.04 * rate), size_common))
            a, p = sosfiltfilt(sos, actual)[segment], sosfiltfilt(sos, prediction)[segment]
            covered = bool(np.any(raw))
            records.append(
                {
                    "case": name,
                    "band_rmse_db": float(np.sqrt(np.mean(error**2))) if covered else None,
                    "mean_band_bias_db": float(np.mean(error)),
                    "band_error_db": error.tolist(),
                    "seeds": len(files),
                    "raw_ray_peak": float(max(abs(raw))),
                    "coverage": "ray_contribution" if covered else "no_ray_contribution",
                    "waveform_nrmse": float(np.linalg.norm(p - a) / np.linalg.norm(a)),
                    "waveform_correlation": float(np.corrcoef(a, p)[0, 1]) if covered else None,
                    "early_peak_error_db": float(20 * np.log10(max(abs(p)) / max(abs(a))))
                    if covered
                    else None,
                }
            )
            waves.append((name, actual, prediction))
            ax = axes[mp - 1, ls - 1]
            ax.semilogx(centers, expected, "o-", color="black", ms=2, label="Measured")
            if covered:
                ax.semilogx(
                    centers,
                    predicted,
                    "o-",
                    color="#b85b36",
                    ms=2,
                    label="Ray simulation + source calibration",
                )
            else:
                ax.text(
                    0.5,
                    0.12,
                    "NO RAY CONTRIBUTION",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=8,
                    color="#b85b36",
                )
            metric = f"RMSE {records[-1]['band_rmse_db']:.1f} dB" if covered else "unsupported path"
            ax.set_title(f"LS{ls} / MP{mp} · {metric}", fontsize=9)
            ax.grid(alpha=0.2)
            ax.xaxis.set_major_locator(FixedLocator([125, 500, 1600]))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        axes[0, 0].legend(fontsize=6)
        fig.supxlabel("Third-octave center frequency (Hz)")
        fig.supylabel("Transfer magnitude (dB re 1 Pa)")
        fig.suptitle(f"BRAS {scene} · measured pressure · fixed source calibration · 0–40 ms")
        for suffix in ["png", "svg"]:
            fig.savefig(args.simulation / f"{label}-measured-bands.{suffix}", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(side, side, figsize=(12, 9), sharex=True, layout="constrained")
        sos = butter(4, [100, 1800], fs=rate, btype="bandpass", output="sos")
        time = np.arange(len(waves[0][1])) / rate * 1000
        for i, (name, actual, prediction) in enumerate(waves):
            ls, mp = int(measured["EmitterID"][i, 0]), int(measured["ReceiverID"][i, 0])
            ax = axes[mp - 1, ls - 1]
            ax.plot(time, sosfiltfilt(sos, actual), color="black", lw=0.7, label="Measured")
            ax.plot(
                np.arange(len(prediction)) / rate * 1000,
                sosfiltfilt(sos, prediction),
                color="#b85b36",
                lw=0.7,
                label="Simulation",
            )
            ax.set(xlim=(12, 40), title=f"LS{ls} / MP{mp}")
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=7)
        fig.supxlabel("Time (ms)")
        fig.supylabel("Pressure (Pa)")
        timing = (
            f"common {latency['advance_samples']}-sample control calibration"
            if latency
            else "unaligned source FIR"
        )
        fig.suptitle(f"BRAS {scene} · identical 100–1800 Hz display filter · {timing}")
        for suffix in ["png", "svg"]:
            fig.savefig(args.simulation / f"{label}-measured-ir.{suffix}", dpi=160)
        plt.close(fig)
    missing = [r["case"] for r in records if r["coverage"] == "no_ray_contribution"]
    summary = {
        "scene": scene,
        "centers_hz": centers.tolist(),
        "mean_pair_band_rmse_db": float(np.mean([r["band_rmse_db"] for r in records]))
        if not missing
        else None,
        "pairs": records,
        "missing_cases": missing,
        "source_model": "independent measured on-axis FIR, omnidirectional approximation; no per-pair fit",
        "latency_calibration": latency,
    }
    validation = [r for r in records if not latency or r["case"] != latency["control"]]
    summary["validation_mean_pair_band_rmse_db"] = (
        float(np.mean([r["band_rmse_db"] for r in validation])) if not missing else None
    )
    summary["validation_pairs"] = len(validation)
    (args.simulation / f"{label}-comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(scene, "mean pair band RMSE", summary["mean_pair_band_rmse_db"])
    for r in records:
        print(
            r["case"],
            round(r["band_rmse_db"], 2) if r["band_rmse_db"] is not None else "NO CONTRIBUTION",
            round(r["mean_band_bias_db"], 2),
        )


if __name__ == "__main__":
    main()
