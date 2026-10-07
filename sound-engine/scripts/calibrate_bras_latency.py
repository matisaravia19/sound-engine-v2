"""Estimate ONE source-FIR time origin from a reserved unobstructed RS5 control.

The calibration uses no diffraction peak, gain fit, or pair-specific shift. The
control must be excluded from subsequent validation aggregates. This estimates a
nuisance time origin; it does not identify the hardware cause of the offset.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, correlate, correlation_lags, fftconvolve, sosfiltfilt

from bras_reference import source_calibration
from visualize_ir import read_ir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--data", type=Path, default=Path("output/measured-reference-research"))
    args = parser.parse_args()
    files = sorted(args.baseline.glob("RS5_LS3_MP4-seed[0-9]*.csv"))
    assert files, "Missing reserved direct-sound control"
    ray_ir = np.mean([read_ir(p)[2]["left"] for p in files], axis=0)
    prediction = fftconvolve(ray_ir, source_calibration(args.data))
    with h5py.File(args.data / "RS5/RS5_RIRs.sofa") as h:
        index = next(
            i for i in range(16) if h["EmitterID"][i, 0] == 3 and h["ReceiverID"][i, 0] == 4
        )
        measured = h["Data.IR"][index, 0, :]
    sos = butter(4, [100, 1800], fs=44100, btype="bandpass", output="sos")
    window = slice(int(0.015 * 44100), int(0.025 * 44100))
    a, b = sosfiltfilt(sos, measured)[window], sosfiltfilt(sos, prediction)[window]
    lags = correlation_lags(len(b), len(a))
    lag = int(lags[np.argmax(correlate(b, a))])
    assert 0 <= lag < int(0.005 * 44100), "Unexpected source timing; inspect manually"
    result = {
        "advance_samples": lag,
        "sample_rate": 44100,
        "control": "RS5_LS3_MP4",
        "method": "Cross-correlation, 100-1800 Hz fourth-order Butterworth, 15-25 ms; averaged baseline rays. One value shared across all models/scenes/positions.",
        "baseline_files": [str(p) for p in files],
        "limitation": "Residual source-FIR time origin, not a verified hardware latency. Control excluded from validation aggregate.",
    }
    (args.data / "source-latency.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Reserved control calibration:", lag, "samples")


if __name__ == "__main__":
    main()
