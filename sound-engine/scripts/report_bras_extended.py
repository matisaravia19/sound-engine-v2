"""Compare frozen scalar ray IRs against BRAS measurements without fitted gains.

Uses the previously fixed Genelec on-axis FIR and 130-sample advance. Spectra
and room decay are diagnostics of the complete declared model, including its
omnidirectional-source and broadband-material approximations.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt

from bras_reference import source_calibration

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/bras-extended-20260909"
RATE = 44100
BANDS = np.array([125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600])


def levels(wave, seconds):
    wave = np.array(wave[: round(seconds * RATE)], copy=True)
    fade = min(round(0.003 * RATE), len(wave))
    wave[-fade:] *= np.cos(np.linspace(0, np.pi / 2, fade)) ** 2
    nfft = 2 ** int(np.ceil(np.log2(max(32768, len(wave)))))
    frequency = np.fft.rfftfreq(nfft, 1 / RATE)
    power = abs(np.fft.rfft(wave, nfft)) ** 2
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
            for f in BANDS
        ]
    )


def decay(wave, frequency):
    """Return uncorrected octave Schroeder decay and bounded EDT/T20 fits.

    No noise subtraction or extrapolated missing tail is applied. Fits must
    cross their full interval and have R-squared >= .9; these are diagnostics,
    not a claim of full ISO measurement processing.
    """
    sos = butter(
        4, [frequency / np.sqrt(2), frequency * np.sqrt(2)], fs=RATE, btype="bandpass", output="sos"
    )
    p = sosfiltfilt(sos, wave)
    energy = np.cumsum((p * p)[::-1])[::-1]
    if energy[0] <= 0:
        return None, None, None
    db = 10 * np.log10(np.maximum(energy / energy[0], 1e-30))
    time = np.arange(len(db)) / RATE

    def fit(upper, lower):
        cross = np.flatnonzero(db <= upper)
        end = np.flatnonzero(db <= lower)
        if not len(cross) or not len(end):
            return None
        a, b = cross[0], end[0]
        if b - a < 100:
            return None
        x, y = time[a : b + 1], db[a : b + 1]
        slope, intercept = np.polyfit(x, y, 1)
        r2 = 1 - np.sum((y - (slope * x + intercept)) ** 2) / max(
            np.sum((y - y.mean()) ** 2), 1e-30
        )
        return (
            {"seconds": float(-60 / slope), "r2": float(r2), "fit_end_seconds": float(time[b])}
            if slope < 0 and r2 >= 0.9
            else None
        )

    return db, fit(0, -10), fit(-5, -25)


def collect_results(base, manifest, fir, advance):
    """Compare each completed setup against measurements and retain plotting inputs."""
    base = base.resolve()
    rows, groups, all_orientations, plot_data, decays = [], [], [], {}, []
    hashes = {}
    for item in manifest:
        name = item["name"]
        folder = base / "full" / name
        if not folder.exists():
            continue
        run_path = folder / "runs.csv"
        with run_path.open(encoding="utf-8", newline="") as stream:
            run = list(csv.DictReader(stream))
        by_case = {}
        expected = {(p["case"], str(seed)) for p in item["pairs"] for seed in range(7, 10)}
        if {(row["case"], row["seed"]) for row in run} != expected or len(run) != len(expected):
            raise ValueError(f"Expected exactly one run per measured case and seed: {name}")
        for r in run:
            by_case.setdefault(r["case"], []).append(r)
        errors = []
        wave_rows = []
        with h5py.File(ROOT / item["sofa"]) as h:
            window = (
                0.25
                if item["scene"] == "RS3"
                else (0.08 if item["scene"].startswith("CR") else 0.04)
            )
            hashes[item["sofa"]] = hashlib.sha256((ROOT / item["sofa"]).read_bytes()).hexdigest()
            for pair in item["pairs"]:
                case = pair["case"]
                signals = []
                for seed in range(7, 10):
                    p = folder / f"{case}-seed{seed}.f32"
                    key = str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
                    hashes[key] = hashlib.sha256(p.read_bytes()).hexdigest()
                    signals.append(np.fromfile(p, dtype="<f4").astype(float))
                raw = np.mean(signals, axis=0)
                prediction = fftconvolve(raw, fir)[: len(raw)]
                prediction = np.pad(prediction[advance:], (0, advance))
                actual = np.array(h["Data.IR"][tuple(pair["index"])], dtype=float)
                if not (
                    np.all(np.isfinite(prediction))
                    and np.all(np.isfinite(actual))
                    and np.any(actual)
                ):
                    raise ValueError(f"Nonfinite or missing measured response: {case}")
                measured_levels = levels(actual, window)
                predicted_levels = levels(prediction, window)
                covered = bool(np.any(raw[: round(window * RATE)]))
                error = predicted_levels - measured_levels
                rmse = float(np.sqrt(np.mean(error**2))) if covered else None
                stop = min(len(actual), len(prediction), round(window * RATE))
                sos = butter(4, [100, 1800], fs=RATE, btype="bandpass", output="sos")
                aw = sosfiltfilt(sos, actual)[:stop]
                pw = sosfiltfilt(sos, prediction)[:stop]
                peak = float(20 * np.log10(max(abs(pw)) / max(abs(aw)))) if covered else None
                correlation = float(np.corrcoef(aw, pw)[0, 1]) if covered else None
                samples = np.flatnonzero(raw)
                record = {
                    "scene": item["scene"],
                    "setup": name,
                    "case": case,
                    "window_ms": window * 1000,
                    "band_rmse_db": rmse,
                    "peak_error_db": peak,
                    "correlation": correlation,
                    "coverage": covered,
                    "band_error_db": error.tolist() if covered else None,
                    "last_ray_seconds": float(samples[-1] / RATE) if len(samples) else None,
                    "raw_peak": float(max(abs(raw))),
                    "warm_ms": float(
                        np.median([float(r["warm_median_ms"]) for r in by_case[case]])
                    ),
                }
                if item["scene"].startswith("CR"):
                    af = sosfiltfilt(sos, actual)
                    pf = sosfiltfilt(sos, prediction)
                    record["late_energy_error_db"] = float(
                        10
                        * np.log10(
                            np.sum(pf[round(0.08 * RATE) :] ** 2)
                            / np.sum(af[round(0.08 * RATE) :] ** 2)
                        )
                    )
                rows.append(record)
                if rmse is not None:
                    errors.append(rmse)
                wave_rows.append((case, actual, prediction, measured_levels, predicted_levels))
                if item["scene"].startswith("CR"):
                    for orientation in range(h["Data.IR"].shape[0]):
                        index = [orientation, *pair["index"][1:]]
                        measured = h["Data.IR"][tuple(index)]
                        valid = bool(np.all(np.isfinite(measured)) and np.any(measured))
                        err = predicted_levels - levels(measured, window) if valid else None
                        all_orientations.append(
                            {
                                "case": case,
                                "orientation_index": orientation,
                                "measurement_present": valid,
                                "band_rmse_db": float(np.sqrt(np.mean(err**2)))
                                if covered and valid
                                else None,
                            }
                        )
                    for f in [500, 1000]:
                        ad, ae, at = decay(actual, f)
                        pd, pe, pt = decay(prediction, f)
                        decays.append(
                            {
                                "case": case,
                                "setup": name,
                                "frequency_hz": f,
                                "measured_edt": ae,
                                "simulated_edt": pe,
                                "measured_t20": at,
                                "simulated_t20": pt,
                            }
                        )
                # Seed variation is descriptive, not a measured uncertainty estimate.
                seed_errors = []
                for signal in signals:
                    predicted = fftconvolve(signal, fir)[: len(signal)]
                    predicted = np.pad(predicted[advance:], (0, advance))
                    seed_errors.append(
                        float(np.sqrt(np.mean((levels(predicted, window) - measured_levels) ** 2)))
                        if np.any(signal[:stop])
                        else None
                    )
                record["seed_band_rmse_db"] = seed_errors
            plot_data[name] = wave_rows
        group = {
            "setup": name,
            "scene": item["scene"],
            "pairs": len(item["pairs"]),
            "covered": len(errors),
            "mean_band_rmse_db": float(np.mean(errors)) if errors else None,
            "median_band_rmse_db": float(np.median(errors)) if errors else None,
            "median_warm_ms": float(np.median([float(r["warm_median_ms"]) for r in run])),
            "max_warm_ms": max(float(r["warm_max_ms"]) for r in run),
            "bounces": item["bounces"],
            "window_ms": window * 1000,
            "worst_case": max(
                (r for r in rows if r["setup"] == name and r["band_rmse_db"] is not None),
                key=lambda r: r["band_rmse_db"],
                default={},
            ).get("case"),
        }
        groups.append(group)

    result = {
        "groups": groups,
        "pairs": rows,
        "other_orientations": all_orientations,
        "decay": decays,
        "input_sha256": hashes,
    }
    return result, plot_data


def plot_results(result, plot_data, figures):
    """Render spectral errors, waveforms, and room-decay comparisons."""
    rows = result["pairs"]
    groups = result["groups"]
    for scene in ["RS1", "RS2", "RS3", "RS4", "RS7", "CR1", "CR2", "CR3", "CR4"]:
        selected = [r for r in rows if r["scene"] == scene]
        if not selected:
            continue
        fig, ax = plt.subplots(
            figsize=(10, max(3, len(selected) * 0.2 + 1.4)), layout="constrained"
        )
        values = np.array(
            [r["band_error_db"] if r["coverage"] else [np.nan] * len(BANDS) for r in selected]
        )
        im = ax.imshow(
            np.ma.masked_invalid(values), aspect="auto", vmin=-20, vmax=20, cmap="RdBu_r"
        )
        ax.set_xticks(range(len(BANDS)), BANDS)
        ax.set_yticks(
            range(len(selected)), [r["case"].replace(scene + "_", "") for r in selected], fontsize=7
        )
        ax.set_xlabel("Third-octave centre frequency (Hz)")
        ax.set_title(
            f"{scene}: simulated minus measured level (dB)\nGrey rows have no received ray contribution; colour saturates at +/-20 dB",
            fontsize=11,
        )
        ax.set_facecolor(".7")
        fig.colorbar(im, ax=ax, label="Level error (dB)", shrink=0.8)
        fig.savefig(figures / f"{scene}-band-errors.png", dpi=160)
        plt.close(fig)
    for section in ["RS1", "RS2", "RS3", "RS4", "RS7", "rooms"]:
        kind = "rooms" if section == "rooms" else "reference"
        names = [
            g["setup"]
            for g in groups
            if (g["scene"].startswith("CR") if kind == "rooms" else g["scene"] == section)
        ]
        if not names:
            continue
        fig, axes = plt.subplots(
            len(names), 2, figsize=(12, 2.5 * len(names)), layout="constrained", squeeze=False
        )
        for (ax, bx), name in zip(axes, names):
            candidates = plot_data[name]
            chosen = next((v for v in candidates if "_LS2_MP2" in v[0]), candidates[0])
            case, actual, prediction, measured_levels, predicted_levels = chosen
            ax.semilogx(BANDS, measured_levels, "o-", color="black", ms=3, label="Measured")
            ax.semilogx(BANDS, predicted_levels, "o-", color="#176bba", ms=3, label="Scalar rays")
            ax.set_title(case, fontsize=10)
            ax.set_ylabel("Transfer magnitude (dB re 1 Pa)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
            if kind == "rooms":
                for wave, color, label in [
                    (actual, "black", "Measured"),
                    (prediction, "#176bba", "Scalar rays"),
                ]:
                    db, _, _ = decay(wave, 1000)
                    if db is not None:
                        bx.plot(
                            np.arange(len(db))[::100] / RATE, db[::100], color=color, label=label
                        )
                bx.set_ylim(-60, 2)
                bx.set_ylabel("Normalized 1 kHz decay (dB)")
                bx.set_xlabel("Time (s)")
            else:
                seconds = 0.15 if name.startswith("RS3") else 0.04
                sos = butter(4, [100, 1800], fs=RATE, btype="bandpass", output="sos")
                for wave, color in [(actual, "black"), (prediction, "#176bba")]:
                    v = sosfiltfilt(sos, wave)[: round(seconds * RATE)]
                    bx.plot(np.arange(len(v)) / RATE * 1000, v, color=color, lw=0.8)
                bx.set_ylabel("100-1800 Hz pressure (Pa)")
                bx.set_xlabel("Time (ms)")
            bx.grid(alpha=0.2)
            ax.set_xlabel("Frequency (Hz)")
        fig.savefig(figures / f"{section}-comparisons.png", dpi=145)
        fig.savefig(figures / f"{section}-comparisons.svg")
        plt.close(fig)


def write_report(base, manifest, result):
    """Write measured metrics and protocol notes without frozen conclusions."""
    rows = result["pairs"]
    groups = result["groups"]
    decays = result["decay"]
    (base / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (base / "per-position.csv").open("w", newline="") as stream:
        keys = [
            "scene",
            "setup",
            "case",
            "coverage",
            "window_ms",
            "band_rmse_db",
            "peak_error_db",
            "correlation",
            "late_energy_error_db",
            "last_ray_seconds",
            "raw_peak",
            "warm_ms",
        ]
        w = csv.DictWriter(stream, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    text = [
        "# Extended BRAS measured comparison",
        "",
        "All simulation pressure comes from the unchanged scalar ray transport. No acoustic model was tuned during this benchmark extension.",
        "",
        "The table reports the mean of per-position RMS errors across twelve 125-1600 Hz bands. Missing ray coverage is counted separately and excluded from finite dB averages; therefore the mean alone is not a pass criterion.",
        "",
        "| Setup | Measured pairs | Rays received | Mean band error (dB) | Window (ms) | Interactions | Median warm query (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for g in groups:
        error = (
            f"{g['mean_band_rmse_db']:.2f}" if g["mean_band_rmse_db"] is not None else "No coverage"
        )
        text.append(
            f"| {g['setup']} | {g['pairs']} | {g['covered']} | {error} | {g['window_ms']:.0f} | {g['bounces']} | {g['median_warm_ms']:.2f} |"
        )
    text += [
        "",
        "## Coverage and room diagnostics",
        "",
        f"Processed {len(groups)} configurations and {len(rows)} source/receiver pairs. "
        f"{sum(row['coverage'] for row in rows)} pairs received a ray contribution. "
        "Interpret these results with the declared model and measurement limits below.",
        "",
        "| Room setup | Mean late energy error, after 80 ms (dB) | Median last received ray (s) |",
        "|---|---:|---:|",
    ]
    for g in groups:
        if g["scene"].startswith("CR"):
            subset = [r for r in rows if r["setup"] == g["setup"]]
            text.append(
                f"| {g['setup']} | {np.mean([r['late_energy_error_db'] for r in subset]):+.2f} | {np.median([r['last_ray_seconds'] for r in subset]):.3f} |"
            )
    text += [
        "",
        "Late energy uses the same 100-1800 Hz filter on both signals, with no normalization. These are errors of the declared current-model fixtures, not isolated material or reflection-coefficient errors.",
        "",
        "| Setup | Octave (Hz) | Measured median T20 (s), valid fits | Simulated median T20 (s), valid fits |",
        "|---|---:|---:|---:|",
    ]
    for g in groups:
        if not g["scene"].startswith("CR"):
            continue
        for frequency in [500, 1000]:
            d = [v for v in decays if v["setup"] == g["setup"] and v["frequency_hz"] == frequency]
            cells = []
            for key in ["measured_t20", "simulated_t20"]:
                values = [v[key]["seconds"] for v in d if v[key] is not None]
                cells.append(
                    f"{np.median(values):.3f}, {len(values)}/{len(d)}"
                    if values
                    else f"Invalid, 0/{len(d)}"
                )
            text.append(f"| {g['setup']} | {frequency} | {' | '.join(cells)} |")
    text += [
        "",
        "Each median uses its own valid fits; differing valid counts mean these are not paired error statistics. Invalid fits and complete EDT/T20 details remain in the JSON. A plausible single decay slope does not excuse the large absolute-energy error.",
        "",
        "",
        "## Protocol and limits",
        "",
        "- 65,536 rays per query, seeds 7-9; five warm repeats after one cold query. Every GPU process ran serially under the existing memory/timeout guard. Timings include synchronous readback and exclude file export.",
        "- RS1/RS2/RS4/RS7 use three interactions. RS3 and the rooms use 30 to expose repeated-reflection behavior; these are quality diagnostics, not the sandbox default. Separate three-interaction room runs are recorded in `realtime/`.",
        "- Every ray carries one scalar pressure. Source calibration is the independent Genelec on-axis FIR, applied only as a measurement operator. The previous fixed 130-sample advance is reused; no per-scene or per-position alignment/gain fitting.",
        "- The source is omnidirectional in the current runtime. Real Genelec measurements have substantial directionality, especially backward radiation in RS3 and different speaker orientations in the rooms. Full-response errors cannot be attributed solely to the reflection coefficient.",
        "- Broadband absorption/scattering is the arithmetic mean of the supplied 125-1600 Hz coefficients. This deliberately exposes the current scalar model limitation; no separate frequency rays or simulations are introduced. Complex rooms use initial estimates, not coefficients fitted to measured decay.",
        "- Reference-scene positions cover all available monaural Genelec SOFA entries. Room comparisons use all Genelec source/receiver pairs at the first recorded orientation. Other orientations are compared with the same omnidirectional prediction in `comparison.json`; the missing CR2 measurement is flagged. Dodecahedron and binaural measurements are not numerically compared.",
        "- Room meshes are imported from the original BRAS SketchUp files. Room diffraction edges are not authored: these room runs test reflections and scattering. Reference-scene edge geometry uses documented thin-screen/outer-envelope approximations.",
        "- Spectrum windows have the same final 3 ms cosine-squared fade on both signals. Waveform figures use the same 100-1800 Hz filter on both signals. No amplitude normalization is applied to waveform or spectral comparisons.",
        "- Room EDC plots are independently normalized Schroeder integrals. EDT/T20 diagnostics at 500 and 1000 Hz require a complete fitting interval and R-squared >=0.9, with no noise correction or tail extrapolation. Finite interaction counts truncate simulated decay, so these are not certified ISO room-acoustic estimates.",
        "",
        "## Measured reference scenes",
        "",
        *[
            f"![{scene} comparisons](figures/{scene}-comparisons.png)"
            for scene in sorted(
                {group["scene"] for group in groups if not group["scene"].startswith("CR")}
            )
        ],
        "",
        "## Measured rooms",
        "",
        *(
            ["![Room spectra and decay](figures/rooms-comparisons.png)"]
            if any(group["scene"].startswith("CR") for group in groups)
            else []
        ),
        "",
        "## Fixture declarations",
        "",
    ]
    for item in manifest:
        text.append(f"- **{item['name']}**: " + " ".join(item["notes"]))
    text += ["", "## All positions", ""]
    for scene in sorted({group["scene"] for group in groups}):
        text.append(f"![{scene} errors](figures/{scene}-band-errors.png)")
    text += [
        "",
        "## Sources and reproduction",
        "",
        "[Source ledger](../../references/bras-extended-validation.md). BRAS v3: Aspock, Brinkmann, Ackermann, Weinzierl and Vorlander, DOI 10.14279/depositonce-6726.3, CC BY-SA 4.0. Download entries and hashes are in `downloads.json` and `materials-downloads.json`.",
        "",
        "Run `prepare_bras_extended.py`, `run_bras_extended.py smoke`, `run_bras_extended.py full`, `run_bras_extended.py realtime`, then `report_bras_extended.py` with the repository Python environment. The room geometry export prerequisite and SDK version are documented in the source ledger. Frozen raw IRs, commands and provenance remain in `full/`; per-position metrics are in `per-position.csv`.",
        "",
    ]
    (base / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps(groups, indent=2))


def main():
    """Generate an extended BRAS report from existing full benchmark runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=OUT)
    parser.add_argument(
        "--calibration", type=Path, default=ROOT / "output/measured-reference-research"
    )
    args = parser.parse_args()
    manifest = json.loads((args.base / "fixtures/manifest.json").read_text(encoding="utf-8"))
    fir = source_calibration(args.calibration)
    advance = json.loads((args.calibration / "source-latency.json").read_text(encoding="utf-8"))[
        "advance_samples"
    ]
    if advance != 130:
        raise ValueError("This comparison protocol requires the fixed 130-sample calibration")
    result, plot_data = collect_results(args.base, manifest, fir, advance)
    if not result["groups"]:
        raise ValueError("No completed benchmark setups found")
    figures = args.base / "figures"
    figures.mkdir(exist_ok=True)
    plot_results(result, plot_data, figures)
    write_report(args.base, manifest, result)


if __name__ == "__main__":
    main()
