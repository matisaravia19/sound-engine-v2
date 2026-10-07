"""Plot CGA02-style comparisons from actual ray-traced Bell Labs Box CSVs.

The paper's measured IRs/calibrations are unavailable. These plots never label
synthetic data as measurements. No simulation is performed by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visualize_ir import read_ir

CASES = {
    "empty": "Empty box: 10 interactions",
    "baffle_proxy": "Baffle: ray-carried diffraction (6 interactions)",
    "baffle_no_diffraction": "Baffle: reflections only (8 interactions)",
}


def lowpass(samples: np.ndarray, rate: int) -> np.ndarray:
    """Zero-phase display filter, unity below 9 kHz and zero above 10 kHz.

    Zero padding prevents wraparound into the plotted IR. Raw exported pressure
    remains authoritative for onset/energy metrics; this filter has pre-ringing.
    """
    size = 4 * len(samples)
    frequency = np.fft.rfftfreq(size, 1 / rate)
    response = 0.5 + 0.5 * np.cos(np.pi * np.clip((frequency - 9000) / 1000, 0, 1))
    return np.fft.irfft(np.fft.rfft(samples, size) * response, size)[: len(samples)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    options = parser.parse_args()
    directory = options.directory
    with (directory / "runs.csv").open() as stream:
        runs = list(csv.DictReader(stream))
    if not runs:
        raise ValueError("runs.csv is empty")
    if len({r["sample_rate"] for r in runs}) != 1:
        raise ValueError("all runs must use the same sample rate")
    rate = int(runs[0]["sample_rate"])
    data, metrics = {}, {}
    for case in CASES:
        records = [r for r in runs if r["case"] == case]
        if not records:
            raise ValueError(f"missing case: {case}")
        waves = []
        for run in records:
            metadata, _, channels = read_ir(directory / f"{case}-seed{run['seed']}.csv")
            if int(metadata["sample_rate"]) != rate:
                raise ValueError("CSV metadata disagrees with runs.csv")
            waves.append(channels["left"])
        waves = np.asarray(waves)
        if not np.isfinite(waves).all():
            raise ValueError(f"nonfinite IR in {case}")
        mean = waves.mean(axis=0)
        data[case] = np.asarray([lowpass(w, rate) for w in waves])
        threshold = np.max(np.abs(mean)) * 1e-4
        onset = np.flatnonzero(np.abs(mean) > threshold)
        early_stop = int(3.0154 / 343 * rate)
        metrics[case] = {
            "seeds": len(waves),
            "onset_ms_at_1e-4_peak": float(onset[0] / rate * 1000) if len(onset) else None,
            "mean_raw_energy_before_baffle_specular_time": float(
                np.mean(np.sum(waves[:, :early_stop] ** 2, axis=1))
            ),
            "relative_seed_rms": float(
                np.sqrt(np.mean((waves - mean) ** 2)) / max(np.sqrt(np.mean(mean**2)), 1e-20)
            ),
            "median_of_run_p50_ms": float(np.median([float(r["p50_ms"]) for r in records])),
            "largest_run_p99_ms": max(float(r["p99_ms"]) for r in records),
        }
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = []

    def save(fig, name):
        fig.savefig(directory / f"{name}.png", dpi=180)
        fig.savefig(directory / f"{name}.svg")
        plt.close(fig)
        figures.append(name)

    # Figure 8: dimensions and coordinates transcribed from the paper.
    fig, axes = plt.subplots(1, 2, figsize=(8, 5), layout="constrained")
    for ax, baffle in zip(axes, [False, True]):
        ax.plot([0, 2.19, 2.19, 0, 0], [0, 0, 3.03, 3.03, 0], "k-")
        source = (1.45, 2.44) if baffle else (1.07, 2.26)
        listener = (1.21, 1.02) if baffle else (1.35, 0.93)
        ax.scatter(*source, label="Source, height 1.35 m", c="#cf5635")
        ax.scatter(*listener, label="Listener, height 1.32 m", c="#26719c")
        if baffle:
            ax.plot([0.93, 2.19], [1.91, 1.91], "k-", lw=3)
            ax.plot(
                [source[0], 0.93, listener[0]],
                [source[1], 1.91, listener[1]],
                "--",
                color="#44856c",
            )
        ax.set(
            xlabel="X (m)",
            ylabel="Z (m)",
            aspect="equal",
            title="With baffle" if baffle else "Empty box",
        )
        ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.06), fontsize=8)
    fig.suptitle("CGA02 Figure 8 geometry · room height 2.42 m")
    save(fig, "figure08_geometry")

    time = np.arange(data["empty"].shape[1]) / rate * 1000
    fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
    for index, wave in enumerate(data["empty"]):
        ax.plot(
            time,
            wave,
            alpha=0.25,
            lw=0.65,
            color="#26719c",
            label="Individual seeds" if index == 0 else None,
        )
    ax.plot(time, data["empty"].mean(axis=0), color="#173f57", lw=0.9, label="Seed mean")
    ax.set(
        xlim=(0, 40),
        xlabel="Time (ms)",
        ylabel="Pressure (1/r source convention)",
        title="Figure 9 counterpart: ray-traced empty box",
    )
    ax.legend()
    save(fig, "figure09_empty_ir")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True, layout="constrained")
    # Published path IDs/arrival times are timing guides, not measured amplitudes.
    guides = [(1, 4.88), (2, 7.92), (3, 7.98), (5, 8.78), (16, 10.82), (25, 11.73), (55, 13.68)]
    for ax, case in zip(axes, list(CASES)[1:]):
        mean = data[case].mean(axis=0)
        ax.plot(time, mean, color="#26719c", lw=0.85)
        for path, arrival in guides:
            ax.axvline(arrival, color="0.7", lw=0.5, linestyle=":")
            if path != 3:
                ax.text(arrival, 0.93, str(path), transform=ax.get_xaxis_transform(), fontsize=8)
        ax.set(xlim=(0, 20), ylabel="Pressure", title=CASES[case])
    axes[-1].set_xlabel("Time (ms); dotted guides: published Table 1 path IDs")
    fig.suptitle("Figure 10 counterparts · identical pressure scale · 10 kHz display bandwidth")
    save(fig, "figure10_baffle_ir")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True, layout="constrained")
    spectra = []
    for case in CASES:
        wave = data[case].mean(axis=0)
        # An explicitly specified STFT makes this reproducible; the paper does
        # not provide its spectrogram window, overlap or color normalization.
        frames = np.lib.stride_tricks.sliding_window_view(np.pad(wave, (128, 128)), 256)[::24]
        spectra.append(np.abs(np.fft.rfft(frames * np.hanning(256), axis=1)).T ** 2)
    reference = max(float(s.max()) for s in spectra)
    for ax, case, spectrum in zip(axes.flat, CASES, spectra):
        db = 10 * np.log10(np.maximum(spectrum / max(reference, 1e-30), 1e-8))
        im = ax.imshow(
            db,
            origin="lower",
            aspect="auto",
            extent=(0, len(time) / rate * 1000, 0, rate / 2),
            vmin=-60,
            vmax=0,
            cmap="magma",
        )
        ax.set(
            xlim=(0, 40 if case == "empty" else 20),
            ylim=(0, 15000),
            xlabel="Time (ms)",
            title=CASES[case].replace(": ", ":\n"),
        )
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=axes, label="Power (dB relative to shared maximum)", shrink=0.8)
    fig.suptitle("Figure 11 counterparts · ray-traced responses · Hann 256, hop 24")
    save(fig, "figure11_spectrograms")

    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    report = [
        "# Bell Labs Box: ray-traced reproduction\n",
        "Source: Tsingos, Carlbom, Elko, Kubli and Funkhouser, *Validation of Acoustical Simulations in the Bell Labs Box* (`cga02.pdf`), Figures 8–11 and Table 1. Citation entries and implementation scope: [references.md](../../references.md).\n",
        "This is a geometry/protocol reproduction, not a measured-waveform replication. The original measured IRs, complex material responses and directional mouth calibration are not available. No paper waveforms have been invented, digitized, fitted or substituted.\n",
        "## Assumptions and differences\n",
        f"- Broadband wall energy absorption: {runs[0]['absorption']} (an assumption, not a value measured in the paper). Omnidirectional unit-pressure point source; no measured mouth filter or air absorption.\n",
        f"- Ray budget {runs[0]['rays']} per query, 48 kHz, 2048 samples, 343 m/s. The paper uses 48 kHz measurements and describes 51.2 kHz simulation/calibration.\n",
        "- The panel is a zero-thickness sheet. The paper's centimeter-rounded plan is used directly; its 1.6758 m and 3.0154 m paths agree to within 1 mm.\n",
        "- Empty room: 10 interactions. Baffle: 6 total interactions, which does **not** enforce the paper's separate limits of 4 reflections and 2 diffractions. Reflection-only ablation: 8 interactions.\n",
        "- Pressure follows ray traversal throughout. Accepted proxy hits advance to the physical edge with an incoming visibility check, physical segment delay and carried attenuation. Uniform launch density is corrected for this projection. A local cosine response is restricted to geometric shadow; BTM is not evaluated by the runtime. Frequency-dependent coefficients and continuous shadow-boundary matching are not implemented.\n",
        "- All plotted simulations come from the Vulkan ray-tracing pipeline. The Python script only reads and plots exported IRs.\n",
        "- Figures use a specified zero-phase 9–10 kHz display filter; its pre-ringing is excluded from raw onset metrics. Source/material calibration is essential to reproduce the paper's bipolar waveforms.\n",
        "- Timing is synchronous host wall time (including submission and query setup), not GPU timestamp time. Cold initialization and readback are separate. Small sample counts and scheduling variation prevent firm speedup claims.\n",
        "## Actual run results\n",
        "| Case | Raw onset (ms) | Energy before baffle reflection time | Seed relative RMS | Median of p50 (ms) | Largest p99 (ms) |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for case, metric in metrics.items():
        report.append(
            f"| {case} | {metric['onset_ms_at_1e-4_peak']} | {metric['mean_raw_energy_before_baffle_specular_time']:.6g} | {metric['relative_seed_rms']:.3g} | {metric['median_of_run_p50_ms']:.3f} | {metric['largest_run_p99_ms']:.3f} |\n"
        )
    report += [
        "\nPublished timing guides: path 1 = 4.88 ms (one diffraction), path 5 = 8.78 ms (one reflection). The significant early contribution in a baffled room is the paper's key qualitative finding. Raw onset uses a relative 1e-4-of-peak threshold; it is not a path identification algorithm.\n",
        "\nThe energy column uses the same baffle path-5 time window for every case, stopping before the sample bin containing 3.0154/343 seconds. It is not the empty room's first reflection time.\n",
        "\nThe paper reports 738 s / 1524 paths for the empty room, 631 s / 1358 paths with diffraction, and 153 s / 307 paths without it. Those timings are historical context, not comparable speedup denominators: hardware, filters, path completeness and order limits differ.\n",
        "\n## Graphs\n",
    ]
    for name in figures:
        report.append(f"\n![{name}]({name}.png)\n")
    report.append(
        "\nRaw per-seed CSVs, runs.csv and metrics.json accompany these plots. Quantitative waveform error against measured data remains unavailable.\n"
    )
    if (directory / "RUN_NOTES.md").exists():
        report.append("\n" + (directory / "RUN_NOTES.md").read_text(encoding="utf-8"))
    (directory / "REPORT.md").write_text("".join(report), encoding="utf-8")
    print(directory / "REPORT.md")


if __name__ == "__main__":
    main()
