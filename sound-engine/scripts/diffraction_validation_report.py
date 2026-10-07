"""Plot actual radius/seed regression results from validate_diffraction.py."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Preserved ray-transport CSVs with the same cases, rays and seeds",
    )
    args = parser.parse_args()
    summary = json.loads((args.directory / "summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout="constrained")
    report = [
        "# Ray-traced first-order diffraction validation\n\n",
        "All simulated contributions follow listener ray traversal and an accepted proxy intersection. "
        "The BTM quadrature is an independent offline reference, never a simulation fallback.\n\n",
        "| Proxy radius (m) | Mean complex error | Time L2 error | Baseline complex error |\n",
        "|---|---:|---:|---:|\n",
    ]
    for ax, (case, measurements), radius in zip(axes, summary.items(), [0.45, 0.20, 0.90]):
        baseline_error = None
        for mode, label, color in [("proxy", "Ray-carried diffraction", "#287c8e")]:
            with (args.directory / f"{case}-{mode}.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            frequencies = sorted({float(r["frequency_hz"]) for r in rows})
            values = np.array(
                [
                    [
                        float(r["sampled_complex_relative_error"])
                        for r in rows
                        if float(r["frequency_hz"]) == f
                    ]
                    for f in frequencies
                ]
            )
            ax.loglog(
                frequencies, values.mean(axis=1), "o-", color=color, label=label, markersize=3
            )
            ax.fill_between(
                frequencies, values.min(axis=1), values.max(axis=1), color=color, alpha=0.15
            )
            if args.baseline:
                with (args.baseline / f"{case}-{mode}.csv").open() as stream:
                    baseline_rows = list(csv.DictReader(stream))

                def signature(rs):
                    return sorted((r["case"], r["rays"], r["seed"], r["frequency_hz"]) for r in rs)

                if signature(rows) != signature(baseline_rows):
                    raise ValueError(
                        "Baseline cases, ray budgets, seeds and frequencies must match"
                    )
                baseline = np.array(
                    [
                        [
                            float(r["sampled_complex_relative_error"])
                            for r in baseline_rows
                            if float(r["frequency_hz"]) == f
                        ]
                        for f in frequencies
                    ]
                )
                baseline_error = baseline.mean()
                ax.loglog(
                    frequencies,
                    baseline.mean(axis=1),
                    "o--",
                    color="#b85b36",
                    label="Previous ray model",
                    markersize=3,
                )
                ax.fill_between(
                    frequencies,
                    baseline.min(axis=1),
                    baseline.max(axis=1),
                    color="#b85b36",
                    alpha=0.12,
                )
        ax.set(title=f"Radius {radius:.2f} m", xlabel="Frequency (Hz)", ylim=(1e-4, 2))
        ax.set_xticks([125, 500, 2000, 8000], ["125", "500", "2k", "8k"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.15, which="both")
        old = measurements["proxy"]
        baseline_cell = "—" if baseline_error is None else f"{baseline_error:.5f}"
        report.append(
            f"| {radius:.2f} | {old['mean_sampled_complex_error']:.5f} | {old['mean_time_relative_l2']:.5f} | {baseline_cell} |\n"
        )
    axes[0].set_ylabel("Complex relative error against sampled BTM")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"First-order wedge · {rows[0]['rays']} rays · three seeds · shading = seed range")
    fig.savefig(args.directory / "accuracy.png", dpi=180)
    fig.savefig(args.directory / "accuracy.svg")
    plt.close(fig)
    report += [
        "\n![Accuracy](accuracy.png)\n\n",
        "Errors are dimensionless and averaged across the seven frequencies and three seeds. "
        "The sampled reference separates transport error from linear fractional-delay filtering. "
        "Continuous-reference errors are also retained in each CSV. These checks isolate a single "
        "rigid-wedge diffraction contribution; absorbing geometric faces suppress reflection paths.\n\n",
        "This does not validate mixed-order diffraction, impedance-dependent diffraction or exact "
        "shadow-boundary transitions. BTM is used only as an offline reference. Runtime timing is host wall time; "
        "scheduling variability and extra response/visibility work mean accuracy improvement must not "
        "be presented as a demonstrated speedup. Raw timing distributions and source hashes accompany this report.\n",
    ]
    if args.baseline:
        report.append(
            f"\nBaseline data: `{args.baseline.as_posix()}`. Both curves are ray-carried transport; the withdrawn BTM runtime is excluded.\n"
        )
    report.append(
        "\nModel derivation, limitations and sources: [references.md](../../references.md), [BibTeX](../../references/diffraction.bib). Improvements on this single wedge are not general validation of the diffraction coefficient.\n"
    )
    if (args.directory / "RUN_NOTES.md").exists():
        report.append("\n" + (args.directory / "RUN_NOTES.md").read_text(encoding="utf-8"))
    (args.directory / "REPORT.md").write_text("".join(report))


if __name__ == "__main__":
    main()
