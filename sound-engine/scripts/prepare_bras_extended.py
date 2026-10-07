"""Prepare BRAS v3 measured positions and declared scalar-model scene fixtures.

Geometry uses source drawings or the original SketchUp mesh, never measured IRs.
Materials use the arithmetic mean of the published 125--1600 Hz coefficients,
matching the scalar runtime's single broadband coefficient without band tracing.
"""

import csv
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/bras-extended-20260909"
DATA = OUT / "data"
FIX = OUT / "fixtures"


class Fixture:
    """Own one text fixture with engine-coordinate triangles and edge proxies."""

    def __init__(self):
        self.lines, self.materials, self.triangles, self.edges = [], {}, [], []

    def material(self, name):
        if name not in self.materials:
            data = np.loadtxt(OUT / "materials" / f"{name}.csv", delimiter=",")
            mask = (data[0] >= 125) & (data[0] <= 1600)
            alpha, scatter = np.clip(data[1:3, mask].mean(axis=1), 0, 1)
            mid = len(self.materials) + 1
            self.materials[name] = {
                "id": mid,
                "absorption": float(alpha),
                "scattering": float(scatter),
            }
            self.lines.append(f"material {mid} {alpha:.9g} {scatter:.9g}")
        return self.materials[name]["id"]

    def tri(self, points, material):
        points = np.array(points)
        mid = self.material(material)
        self.lines.append("tri " + str(mid) + " " + " ".join(f"{v:.9g}" for v in points.ravel()))
        self.triangles.append(points.tolist())

    def quad(self, points, material):
        for indices in [[0, 1, 2], [0, 2, 3]]:
            self.tri(np.array(points)[indices], material)

    def edge(self, a, b, bisector, angle=2 * np.pi):
        d = np.array(bisector, dtype=float)
        d /= np.linalg.norm(d)
        values = [*a, *b, *d, angle, 1.25]
        self.lines.append("edge " + " ".join(f"{v:.9g}" for v in values))
        self.edges.append(values)

    def plate(self, x, y0, y1, z0, z1, material):
        p = np.array([[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]])
        self.quad(p, material)
        for i in range(4):
            a, b = p[i], p[(i + 1) % 4]
            self.edge(a, b, (a + b) / 2 - p.mean(0))

    def floor(self, y=0):
        self.quad([[-20, y, -20], [20, y, -20], [20, y, 20], [-20, y, 20]], "mat_Tiles")

    def block(self, x0, x1, height, z0, z1, y0=0):
        m = "mat_MDF12mm_plane_00deg"
        top = y0 + height
        self.quad([[x0, y0, z0], [x0, top, z0], [x0, top, z1], [x0, y0, z1]], m)
        self.quad([[x1, y0, z0], [x1, top, z0], [x1, top, z1], [x1, y0, z1]], m)
        self.quad([[x0, top, z0], [x1, top, z0], [x1, top, z1], [x0, top, z1]], m)
        for z in [z0, z1]:
            self.quad([[x0, y0, z], [x1, y0, z], [x1, top, z], [x0, top, z]], m)
        for x, s in [(x0, -1), (x1, 1)]:
            self.edge([x, top, z0], [x, top, z1], [s, 1, 0], 1.5 * np.pi)
            for z, t in [(z0, -1), (z1, 1)]:
                self.edge([x, y0, z], [x, top, z], [s, 0, t], 1.5 * np.pi)
        for z, t in [(z0, -1), (z1, 1)]:
            self.edge([x0, top, z], [x1, top, z], [0, 1, t], 1.5 * np.pi)


def geometry(scene, variant):
    f = Fixture()
    notes = []
    if scene == "RS1":
        if variant == "Rigid":
            f.floor()
        elif variant == "Absorbing":
            f.floor(-0.02)
            f.quad(
                [[3.397, 0, 0.881], [7.603, 0, 0.881], [7.603, 0, 5.089], [3.397, 0, 5.089]],
                "mat_RockFonSonarG_45deg",
            )
            notes.append(
                "SOFA heights retained; geometry translated -20 mm to the absorber top reference. Absorber 4.206 x 4.208 m footprint; single 45-degree material curve averaged across bands for every position."
            )
        else:
            f.floor(-0.24)
            for i in range(33):
                x = 3.505 + i * 0.121
                f.block(x, x + 0.12, 0.12 if i % 2 == 0 else 0.24, 0.915, 5.055, y0=-0.24)
            notes.append(
                "SOFA heights retained; diffuser translated -240 mm to its top reference. 33 alternating 120/240 mm profiles, 1 mm gaps, hollow profiles represented by outer surfaces; no transmission."
            )
    elif scene == "RS2":
        size = 1 if variant.startswith("1m") else 2
        m = (
            "mat_RockFonSonarG_45deg"
            if variant.endswith("absorbing")
            else "mat_MDF25mmA_plane_00deg"
        )
        f.plate(0, 1.02 - size / 2, 1.02 + size / 2, -size / 2, size / 2, m)
        notes.append(
            "25 mm plate represented by its front plane and four thin-screen edges; supporting stand omitted as allowed by BRAS documentation. Absorber uses one 45-degree coefficient curve for all angles."
        )
    elif scene == "RS3":
        for x in [0, 10]:
            f.plate(x, 0.02, 2.02, -1, 1, "mat_MDF25mmA_plane_00deg")
        notes.append(
            "Two 2 m plates 10 m apart, front-face thin-screen approximation; stands omitted. Source is directional in the measurement and omnidirectional in this runtime."
        )
    elif scene == "RS4":
        j = json.loads((DATA / scene / f"geometry-{variant}.json").read_text())
        # Preserve surveyed panel offsets/rotations. Select their front faces;
        # source and microphone visualization meshes are deliberately excluded.
        faces = [
            np.array(t["vertices"])
            for t in j["triangles"]
            if t["material"] == "MDF12mm" and np.max(np.abs(np.array(t["vertices"])[:, 0])) < 1e-6
        ]
        assert len(faces) == 18
        boundaries = {}
        for p in faces:
            f.tri(p, "mat_MDF25mmA_plane_00deg")
            for i in range(3):
                a, b = p[i], p[(i + 1) % 3]
                key = tuple(sorted([tuple(np.round(a, 6)), tuple(np.round(b, 6))]))
                boundaries.setdefault(key, []).append(p.mean(0))
        for (a, b), centers in boundaries.items():
            if len(centers) == 1:
                a, b = np.array(a), np.array(b)
                d = (a + b) / 2 - centers[0]
                tangent = (b - a) / np.linalg.norm(b - a)
                d -= np.dot(d, tangent) * tangent
                f.edge(a, b, d)
        assert len(f.edges) == 36
        notes.append(
            "Nine surveyed panel front faces imported from SketchUp; 25 mm thickness approximated by thin screens. Original MDF12mm visual material tag mapped to documented 25 mm MDF material."
        )
    elif scene == "RS7":
        f.floor()
        for i in range(15):
            x = 3.958 + i * 0.34
            f.block(x - 0.06, x + 0.06, 0.24, 2.985 - 2.07, 2.985 + 2.07)
        notes.append(
            "15 hollow blocks represented by outer surfaces; 0.22 m clear spacing. Current runtime permits at most two diffraction events, even when the interaction budget is larger."
        )
    else:
        suffix = variant[-1] if scene == "CR1" else ""
        j = json.loads((DATA / scene / f"geometry{suffix}.json").read_text())
        for t in j["triangles"]:
            material = t["material"]
            if material == "unassigned":
                assert scene == "CR1" and suffix == "1"
                material = "mat_CR1_concrete"
            f.tri(t["vertices"], material)
        notes.append(
            "Original surveyed room geometry and initial_estimates materials; no measured-decay fitted materials. Room diffraction edges not authored: this is a room reflection/scattering test. Genelec directivity omitted; on-axis FIR retained."
        )
        if scene == "CR1":
            notes.append(
                "DoorAngle1 unassigned 17.3028 m2 wall uses concrete, matching the same wall in DoorAngle3."
            )
    return f, notes


def main():
    FIX.mkdir(exist_ok=True)
    manifest = []
    for scene in ["RS1", "RS3", "RS2", "RS4", "RS7", "CR2", "CR1", "CR3", "CR4"]:
        for sofa in sorted((DATA / scene).glob("*RIRs*.sofa")):
            if "Dodecahedron" in sofa.name:
                continue
            variant = sofa.stem.replace(scene + "_RIRs", "").strip("_").replace("_Genelec8020c", "")
            if variant == "Genelec8020c":
                variant = "room"
            if not variant:
                variant = "default"
            name = scene + "_" + variant
            f, notes = geometry(scene, variant)
            (FIX / f"{name}.scene").write_text("\n".join(f.lines) + "\n")
            pairs = []
            with h5py.File(sofa) as h:
                assert int(h["Data.SamplingRate"][0]) == 44100
                if scene.startswith("RS"):
                    for i in range(len(h["EmitterID"])):
                        ls, mp = int(h["EmitterID"][i, 0]), int(h["ReceiverID"][i, 0])
                        pairs.append(
                            {
                                "case": f"{name}_LS{ls}_MP{mp}",
                                "index": [i, 0],
                                "source": h["EmitterPosition"][0, :, i][[0, 2, 1]].tolist(),
                                "receiver": h["ReceiverPosition"][0, :, i][[0, 2, 1]].tolist(),
                            }
                        )
                else:
                    for r in range(h["Data.IR"].shape[1]):
                        for e in range(h["Data.IR"].shape[2]):
                            ls, mp = int(h["EmitterID"][e, 0]), int(h["ReceiverID"][r, 0])
                            pairs.append(
                                {
                                    "case": f"{name}_LS{ls}_MP{mp}",
                                    "index": [0, r, e],
                                    "source": h["EmitterPosition"][e, :, 0][[0, 2, 1]].tolist(),
                                    "receiver": h["ReceiverPosition"][r, :, 0][[0, 2, 1]].tolist(),
                                }
                            )
                samples = h["Data.IR"].shape[-1]
            with (FIX / f"{name}.csv").open("w", newline="") as stream:
                w = csv.writer(stream)
                w.writerow(["case", "sx", "sy", "sz", "rx", "ry", "rz"])
                for p in pairs:
                    w.writerow([p["case"], *p["source"], *p["receiver"]])
            # Full recorded duration is retained; path length is still bounded
            # by the runtime interaction count, a limit reported in decay tests.
            duration = samples / 44100
            manifest.append(
                {
                    "name": name,
                    "scene": scene,
                    "variant": variant,
                    "sofa": str(sofa.relative_to(ROOT)),
                    "seconds": duration,
                    "bounces": 30 if scene.startswith("CR") or scene == "RS3" else 3,
                    "pairs": pairs,
                    "materials": f.materials,
                    "notes": notes,
                    "triangles": len(f.triangles),
                    "edges": len(f.edges),
                }
            )
            print(name, len(pairs), "pairs", len(f.triangles), "triangles", len(f.edges), "edges")
    (FIX / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
