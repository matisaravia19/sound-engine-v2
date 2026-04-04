# Sound Engine Design Docs

This folder defines a target architecture for the real-time acoustic ray-traced sound engine.

Reading order:

1. `system-design-overview.md`
2. `public-api.md`
3. `internal-architecture.md`
4. `vulkan-architecture.md`
5. `pipeline-spec.md`
6. `modules/README.md` (detailed per-module implementation plan)

Notes:

- These docs describe the intended architecture, not a strict reflection of current code.
- The design is thesis-first and intentionally minimal, not feature-complete.
- Diffraction is the primary extension point; transmission remains optional.
- Rust module names in examples are proposed names for the next iteration.
