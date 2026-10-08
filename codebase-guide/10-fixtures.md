# Fixtures: controlled scenes and experimental inputs

[Guide index](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/README.md)

`acoustic-fixtures` supplies reusable scene descriptions and source/listener positions. It lets tests, benchmarks, and the sandbox ask the engine the same geometric question without maintaining separate copies of the scene.

For the doorway example, a fixture would specify wall triangles, an opening, explicit diffraction edges, and source/receiver coordinates. The acoustic engine still decides how to propagate sound through that input.

## Where the crate belongs

The crate depends on the engine's public scene types. The sandbox depends on the fixture crate, and the engine uses it as a development dependency. It is not part of the normal engine runtime.

Its [library entry point](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/acoustic-fixtures/src/lib.rs) exposes four groups:

| Module | Purpose |
| --- | --- |
| [geometry](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/acoustic-fixtures/src/geometry.rs) | Basic planes, rooms, wedges, and wall construction. |
| [bras](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/acoustic-fixtures/src/bras.rs) | Geometry used by the RS5 partition and RS6 block comparisons. |
| [bell_box](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/acoustic-fixtures/src/bell_box.rs) | Empty-room and baffle variants of the Bell Labs box reconstruction. |
| [text](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/acoustic-fixtures/src/text.rs) | Validated scene and position-file readers. |

These modules produce ordinary `SceneDescription` values. They do not choose a different propagation algorithm.

## Small scenes make a question easier to isolate

The geometry builders include an empty scene, a single plane, parallel planes, a shoebox, and a diffraction wedge. Their value is interpretability: a plane can isolate a reflection, while a wedge can isolate edge behavior.

A fixture name alone is not enough to identify an experiment. Material parameters, source/listener coordinates, authored edge radius, ray count, and interaction limit all affect the result.

For the doorway, start with direct visibility and obstruction, then add edge interactions and reflections. This makes discrepancies easier to locate than beginning with every transport component enabled.

## Measured geometry still contains modeling choices

The BRAS builders encode two specific approximations documented in the source:

- The RS5 partition is represented by a thin surface at the mid-plane of the panel.
- The RS6 hollow-block assembly is represented by its outer envelope.

Both include a large ground plane. The preparation convention maps BRAS coordinates `(X, Y, Z)` into engine coordinates `(X, Z, Y)`, making engine Y the vertical axis. A source/receiver CSV is expected to have already undergone that conversion.

The Bell Labs box builder uses dimensions of 2.19 × 2.42 × 3.03 meters in engine X/Y/Z order. Its optional baffle is a zero-thickness surface with an explicitly authored exposed edge. The code describes this as a broadband approximation because the original complex wall calibrations are unavailable.

These are statements about the fixture implementation. A thesis discussion of the original experiments should cite and check the original source separately. Reusing published dimensions does not make the implemented material or diffraction model identical to the measured setup.

## Text input is validated before GPU work

`read_scene` accepts material, triangle, and edge records. It builds mesh/object records and passes the result through the engine's CPU scene validation. This shares geometry and material invariants with normal engine use without creating Vulkan state.

`read_positions` returns `PositionPair` values containing a case name and two positions in meters. The table must be nonempty, positions must be finite, and names must be unique and safe as exported filename stems.

Malformed input errors include the path and, for malformed records, the line number. This helps distinguish an invalid exported fixture from a failure in the acoustic calculation.

The reader also recognizes a retired optional edge-scale column in cached files: zero suppresses an edge, while a positive accepted value no longer scales its acoustic response. When reading an old fixture, do not interpret that extra value as a current material or diffraction parameter.

## From external data to a comparison

The preparation/reporting tools live outside the crate:

~~~text
external geometry, coordinates, and recordings
  → preparation/export scripts
  → scene and position files
  → fixture reader
  → engine benchmark and recorded IR
  → report with explicit comparison/calibration choices
~~~

[prepare_bras_reference.py](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/scripts/prepare_bras_reference.py) and [prepare_bras_extended.py](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/scripts/prepare_bras_extended.py) prepare the relevant inputs. [bras_reference.py](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/scripts/bras_reference.py) and [report_bras_extended.py](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/scripts/report_bras_extended.py) compare results. [calibrate_bras_latency.py](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/scripts/calibrate_bras_latency.py) records the source-latency calibration used by applicable comparisons.

External recordings and prepared files are not fetched by constructing an engine. Their presence depends on the local experiment setup.

Keep three things distinct while interpreting a plot: the geometry and approximations supplied as input, the measured recording, and the simulated response. Calibration belongs in the recorded comparison procedure; it should not be hidden in undocumented changes to either signal.

## Evidence to follow

[Text-reader tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/acoustic-fixtures/tests/unit/text.rs) cover malformed records and position invariants. [Bell box tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/acoustic-fixtures/tests/unit/bell_box.rs) check the builder's geometry and edge choices.

Those tests establish properties of the inputs. The acoustic reference tests and measured comparisons described in the next chapter establish different things about the model's output.
