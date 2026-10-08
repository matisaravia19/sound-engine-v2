# Sandbox: driving the engine interactively

[Guide index](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/README.md)

The sandbox is a raylib application around the public engine API. It combines a scene catalog, a movable source and listener, audio playback, and a view of the geometry. It is useful for building intuition and observing transitions; quantitative claims belong with recorded experiments.

For the doorway example, the sandbox supplies the room, draws source and listener markers, and sends movement to the engine. The engine remains responsible for acoustic tracing and convolution.

## Startup has an intentional order

[main.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sandbox/src/main.rs) delegates to [app::run](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sandbox/src/app/mod.rs). Startup parses options, prepares the selected scene, obtains the audio device's sample rate, constructs the engine and audio session, and finally opens the graphical window.

Help, scene listing, and scene validation return before audio-device and window initialization. They are useful for checking configuration independently:

~~~text
cargo run -p sandbox -- --help
cargo run -p sandbox -- --list-scenes
cargo run -p sandbox -- --validate-scenes
~~~

For interactive use, select a WAV explicitly:

~~~text
cargo run -p sandbox --release -- --sound=C:/path/to/recording.wav
~~~

Replace the example path with an existing file. The default sound path points outside the repository, so it should not be treated as a bundled asset.

The default acoustic workload is substantial: stereo output, 1,024-frame blocks, a two-second IR, one million rays, and three interactions. [Configuration](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sandbox/src/app/config.rs) also exposes `--rays=...`, `--bounces=...`, `--diffraction-radius=...`, and other options. Keep those settings with any observations about speed or sound.

## The application is split by responsibility

| Module | What to read it for |
| --- | --- |
| [audio](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sandbox/src/app/audio.rs) | Engine ownership, loading the dry asset, tracked source, playback, and runtime updates. |
| [scene](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sandbox/src/app/scene.rs) | Selected scene, source/listener pair, editable room, and acoustic mode. |
| [listener](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sandbox/src/app/listener.rs) | Listener mode, orientation, and camera-update thresholds. |
| [controls](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sandbox/src/app/controls.rs) | Translating input into changes. |
| [layout](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sandbox/src/app/layout.rs) | Shared rectangles for drawing and hit testing. |
| [render](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sandbox/src/app/render.rs) | Visual presentation. |
| [geometry](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sandbox/src/app/geometry.rs) and [poses](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sandbox/src/app/poses.rs) | Editable room geometry and pose presets. |
| [scenes](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sandbox/src/scenes/mod.rs) | Built-in and file-backed catalog entries. |

The `AudioSession` owns the engine, loaded sound ID, tracked source, and gain. It starts playback with three prefill blocks and a 64 ms maximum queue setting; these are application choices rather than the library's default policy.

A scene change is prepared as a candidate. The application loads that candidate into the engine before accepting the new UI state. A failed load therefore does not leave the display claiming that a different acoustic scene is active.

## Camera movement and listener movement differ

The graphical camera is normally a way to inspect the scene. The acoustic listener is a separate pose shown by a marker. Camera mode explicitly makes the receiver follow the view.

The listener controller supports fixed orientation presets and rotation. Ordinary camera-follow updates are limited by elapsed time and meaningful pose changes: a 50 ms interval, a one-centimeter position threshold, and an orientation dot-product threshold of 0.99996. Explicit controls can bypass the time interval.

It records a camera pose as accepted only after the update is submitted successfully. If submission fails, the same movement can be retried instead of disappearing from the controller's state.

These thresholds reduce unnecessary IR refreshes. They also mean the displayed camera can move slightly before a new acoustic query is sent. Runtime queueing and IR crossfades add their own response times, as explained in the Engine and Auralization chapters.

## What “distance only” means

The alternative playback mode loads an empty acoustic scene while retaining the geometry used for display and the chosen source/listener positions.

The result still contains direct propagation: distance attenuation, travel delay, and the selected spatial output model. Reflections, obstruction, and diffraction from the displayed room are absent because that geometry is not in the acoustic scene.

This is a useful listening comparison, but its name should not be interpreted as bypassing the engine or simply changing a gain knob.

## The scene catalog and inferred edges

Built-in scenes cover the playground, free field, planes, a shoebox, a wedge, and Bell Labs box variants. BRAS entries can use prepared position files; additional scenes are discovered from optional fixture files under the configured benchmark root. The available catalog consequently depends on local data.

The engine accepts **authored diffraction edges**. The sandbox also has a load-time [edge inference helper](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sandbox/src/scenes/diffraction.rs) for suitable triangle geometry.

That helper welds nearby vertices at a one-millimeter scale, handles split edges, and searches for sufficiently long open boundaries and convex creases. It excludes short edges, nearly coplanar bends, and unsupported non-manifold configurations. Its minimum length is 0.20 m and its minimum bend is 20 degrees. Known source/listener air points help orient connected surfaces.

This is a geometric heuristic, not an acoustic theorem. Inspect its output when interpreting a file-backed scene. The core scene module does not automatically infer or move those edges when a mesh changes.

## Reading and testing the application

Start with `app::run`, follow a play action through `controls` into `audio`, then follow a listener move through `listener`. This gives a concrete example of using the facade without first reading rendering details.

[Controller tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sandbox/tests/unit/app/listener.rs), [scene-state tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sandbox/tests/unit/app/scene.rs), and [edge-inference tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sandbox/tests/unit/scenes/diffraction.rs) cover behavior without opening a window.

The camera stress option can produce a frame-time report. Frame time measures graphical application behavior; audio underruns and acoustic refresh age are separate observations. A smooth camera does not establish that audio always met its deadlines.
