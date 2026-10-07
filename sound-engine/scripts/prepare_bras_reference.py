"""Verify downloaded BRAS files, extract measured RIRs, and map SOFA coordinates.

Run download_bras_reference.ps1 first. BRAS v3 is CC BY-SA 4.0; attribution and
citations are in references/measured-validation.md. Extracted names are flattened
deliberately; archive paths are never trusted as filesystem destinations.
"""

import argparse
import csv
import hashlib
import json
import struct
import zipfile
import zlib
from pathlib import Path

import h5py


def verify(path, manifest):
    """Refuse changed bytes so a dataset revision cannot silently change results."""
    entry = next(e for e in manifest["files"] if e["file"] == path.name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("output/measured-reference-research"))
    args = parser.parse_args()
    manifest = json.loads(
        (Path(__file__).resolve().parents[2] / "references/bras-downloads.json").read_text()
    )
    for scene in ["RS5", "RS6"]:
        archive = args.data / f"BRAS_{scene}.zip"
        verify(archive, manifest)
        target = args.data / scene
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if name.endswith((f"{scene}_RIRs.sofa", f"{scene}_RIR.png")):
                    (target / Path(name).name).write_bytes(z.read(name))
        with (
            h5py.File(target / f"{scene}_RIRs.sofa") as sofa,
            (target / "positions.csv").open("w", newline="") as out,
        ):
            writer = csv.writer(out)
            writer.writerow(
                [
                    "case",
                    "source_x",
                    "source_y",
                    "source_z",
                    "listener_x",
                    "listener_y",
                    "listener_z",
                ]
            )
            assert int(sofa["Data.SamplingRate"][0]) == 44100
            for i in range(len(sofa["EmitterID"])):
                ls, mp = int(sofa["EmitterID"][i, 0]), int(sofa["ReceiverID"][i, 0])
                source, receiver = (
                    sofa["EmitterPosition"][0, :, i],
                    sofa["ReceiverPosition"][0, :, i],
                )
                writer.writerow(
                    [f"{scene}_LS{ls}_MP{mp}", *source[[0, 2, 1]], *receiver[[0, 2, 1]]]
                )
    prefix = args.data / "genelec-ir-prefix.bin"
    verify(prefix, manifest)
    data = prefix.read_bytes()
    header = struct.unpack_from("<4s5H3L2H", data)
    assert header[0] == b"PK\x03\x04" and header[3] == 8
    offset = 30 + header[-2] + header[-1]
    decoded = zlib.decompressobj(-15).decompress(data[offset:])
    first = decoded.splitlines()[0]
    assert first.startswith(b"P000T000,")
    assert len([v for v in first.split(b",")[1:] if v]) == 4096
    # Only a complete on-axis row is needed; do not present a partial archive
    # as a complete directivity measurement.
    (args.data / "genelec-on-axis.csv").write_bytes(first + b"\n")
    print("Verified and prepared RS5/RS6 coordinates, RIRs and on-axis source FIR")


if __name__ == "__main__":
    main()
