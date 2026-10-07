"""Export surveyed BRAS room meshes using the SketchUp Importer 0.27 bindings.

Run with Blender 4.1's bundled Python 3.11; this does not start Blender or the GPU.
The local SDK package and SHA-256 are recorded in references/bras-extended-validation.md.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sdk = Path("output/bras-extended-20260909/sketchup-sdk").resolve()
    sys.path.insert(0, str(sdk))
    dll_directory = os.add_dll_directory(str(sdk))
    import sketchup

    model = sketchup.Model.from_file(str(args.input.resolve()))
    result, hierarchy = [], []

    def walk(entities, transform, inherited="unassigned", path="root"):
        count = 0
        for face in entities.faces:
            vertices, triangles, _ = face.tessfaces
            vertices = np.array(vertices)
            if not len(vertices):
                continue
            # Bindings return metres, including transform translation. Preserve
            # nested transforms before mapping BRAS Z-up to the engine Y-up.
            vertices = (transform @ np.column_stack((vertices, np.ones(len(vertices)))).T).T[:, :3]
            material = face.material.name if face.material else inherited
            for tri in triangles:
                points = vertices[list(tri)][:, [0, 2, 1]]
                if np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) > 1e-10:
                    result.append({"material": material, "vertices": points.tolist()})
                    count += 1
        hierarchy.append({"path": path, "triangles": count})
        for group in entities.groups:
            if not group.hidden:
                walk(
                    group.entities,
                    transform @ np.array(group.transform),
                    group.material.name if group.material else inherited,
                    path + "/" + group.name,
                )
        for instance in entities.instances:
            if not instance.hidden:
                definition = model.component_definition_as_dict[instance.definition.name]
                walk(
                    definition.entities,
                    transform @ np.array(instance.transform),
                    instance.material.name if instance.material else inherited,
                    path + "/" + definition.name,
                )

    walk(model.entities, np.eye(4))
    args.output.write_text(
        json.dumps(
            {"source": str(args.input), "triangles": result, "hierarchy": hierarchy},
            separators=(",", ":"),
        )
        + "\n"
    )
    p = np.array([v for t in result for v in t["vertices"]])
    print(
        args.input.name,
        len(result),
        "triangles; bounds",
        p.min(axis=0),
        p.max(axis=0),
        "materials",
        sorted({t["material"] for t in result}),
    )
    model.close()
    dll_directory.close()


if __name__ == "__main__":
    main()
