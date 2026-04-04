# Module Specs

Detailed implementation plan per internal module.

1. `api.md`
2. `core.md`
3. `scene.md`
4. `simulation.md`
5. `acoustics.md`
6. `auralization.md`
7. `gpu.md`

Conventions used in these files:

- `pub`: exposed outside crate.
- `pub(crate)`: shared internally.
- private: module-local helpers/state.
- Signatures are design targets and can be adjusted during implementation.
