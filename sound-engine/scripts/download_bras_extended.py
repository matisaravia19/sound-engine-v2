"""Fetch selected original BRAS v3 archive entries with CRC and SHA-256 records.

Range requests avoid downloading unrelated photographs and binaural datasets.
Archive filenames are flattened; no archive path is used as a destination.
"""

import argparse
import hashlib
import io
import json
import re
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "output/measured-reference-research"
OUT = ROOT / "output/bras-extended-20260909"


def fetch(url, byte_range):
    """Require an actual partial response before accepting remote ZIP bytes."""
    request = urllib.request.Request(url, headers={"Range": "bytes=" + byte_range})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise ValueError(f"Range request not honored: {response.status}")
        return response.read(), response.headers["Content-Range"]


def catalogue(url):
    """Read an ordinary ZIP central directory without downloading its payload."""
    tail, content_range = fetch(url, "-65536")
    size = int(content_range.split("/")[-1])
    offset = tail.rfind(b"PK\x05\x06")
    end = struct.unpack_from("<4s4H2IH", tail, offset)
    directory_size, directory_offset = end[5:7]
    directory, _ = fetch(url, f"{directory_offset}-{directory_offset + directory_size - 1}")
    # zipfile corrects offsets for the intentionally omitted local-file region.
    with zipfile.ZipFile(io.BytesIO(directory + tail[offset:])) as archive:
        entries = archive.infolist()
        for entry in entries:
            entry.header_offset += directory_offset
    return size, entries


def download_entry(url, entry):
    """Verify decompression length and ZIP CRC before returning an entry."""
    header, _ = fetch(url, f"{entry.header_offset}-{entry.header_offset + 29}")
    fields = struct.unpack("<4s5H3I2H", header)
    assert fields[0] == b"PK\x03\x04"
    start = entry.header_offset + 30 + fields[-2] + fields[-1]
    payload, _ = fetch(url, f"{start}-{start + entry.compress_size - 1}")
    result = zlib.decompress(payload, -15) if entry.compress_type == 8 else payload
    assert entry.compress_type in (0, 8)
    assert len(result) == entry.file_size and zlib.crc32(result) == entry.CRC
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["RS1", "RS3", "RS2", "RS4", "RS7", "CR2", "CR1", "CR3", "CR4"],
    )
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument(
        "--materials",
        action="store_true",
        help="Fetch the published initial material estimates and reference coefficients",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    html = (DATA / "bras-landing.html").read_text(encoding="utf-8")
    links = dict(re.findall(r'aria-label="Download ([^"]+)"[^>]*href="([^"]+)"', html))
    manifest_path = OUT / "downloads.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"doi": "10.14279/depositonce-6726.3", "license": "CC BY-SA 4.0", "files": []}
    )
    for scene in args.scenes:
        link = links[f"1_scene_descriptions-{scene}.zip"]
        uid = link.split("/")[2]
        url = f"https://api-depositonce.tu-berlin.de/server/api/core/bitstreams/{uid}/content"
        size, entries = catalogue(url)
        print(scene, size, "archive bytes", flush=True)
        records = [
            {"name": e.filename, "size": e.file_size, "compressed": e.compress_size}
            for e in entries
        ]
        (OUT / f"{scene}-catalogue.json").write_text(json.dumps(records, indent=2) + "\n")
        for entry in entries:
            name = entry.filename
            if (
                not name.lower().endswith(
                    (".sofa", ".skp", ".png", ".pdf", ".mat", ".txt", ".dae", ".obj")
                )
                or "/Pictures/" in name
                or "/BRIR" in name
                or "_BRIR" in name
            ):
                continue
            print(" ", name, entry.file_size, flush=True)
            if args.list_only:
                continue
            target = OUT / "data" / scene / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(download_entry(url, entry))
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            record = {
                "scene": scene,
                "url": url,
                "entry": name,
                "bytes": target.stat().st_size,
                "sha256": digest,
                "path": str(target.relative_to(ROOT)),
            }
            prior = next((r for r in manifest["files"] if r["path"] == record["path"]), None)
            if prior:
                assert prior["sha256"] == digest
            else:
                manifest["files"].append(record)
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.materials:
        uid = links["3_surface_descriptions.zip"].split("/")[2]
        url = f"https://api-depositonce.tu-berlin.de/server/api/core/bitstreams/{uid}/content"
        _, entries = catalogue(url)
        records = []
        for entry in entries:
            if not entry.filename.endswith(".csv") or not (
                "initial_estimates/" in entry.filename or "/_csv/mat_" in entry.filename
            ):
                continue
            target = OUT / "materials" / Path(entry.filename).name
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                target.write_bytes(download_entry(url, entry))
            data = target.read_bytes()
            assert len(data) == entry.file_size and zlib.crc32(data) == entry.CRC
            records.append(
                {
                    "url": url,
                    "entry": entry.filename,
                    "path": str(target.relative_to(ROOT)),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        (OUT / "materials-downloads.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
