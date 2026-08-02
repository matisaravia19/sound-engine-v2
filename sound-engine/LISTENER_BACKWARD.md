# Listener-originated acoustic path tracing

The acoustic tracer launches one stable Monte Carlo path per listener ray. The
ray-generation invocation owns the current origin, direction, pressure
throughput, traveled distance, bounce count, initial listener direction, and
random state. Closest-hit and miss shaders only return traversal results, so
the Vulkan pipeline recursion depth remains one.

## Sampling and throughput

Initial directions use a Fibonacci sphere distribution with a deterministic
seeded azimuth rotation. Their estimator weight is `4π / listener_ray_count`;
this normalization is kept separate from path throughput so changing the ray
count changes variance rather than level.

At a surface, reflection energy is `max(0, 1 - absorption - transmission)`.
`scattering` splits it into finite specular and rough lobes, and `roughness`
controls their power-cosine exponents. Transmission uses a finite lobe centered
on straight propagation. Energy fractions are converted to pressure weights
with square roots. Exactly one branch is selected with probability
proportional to its pressure weight, and throughput is updated by:

```text
throughput *= response / branch_probability / direction_pdf
```

Paths stop on a miss, configured bounce or distance limit, invalid sample,
non-finite state, or throughput below the configured threshold. There is no
Russian roulette.

## Source connections and IR deposition

Invocation zero evaluates the direct listener-to-source path once. Every valid
surface or diffraction vertex also traces a surface-only visibility ray to the
point source. Diffraction proxies never occlude visibility.

A completed path deposits the provisional pressure amplitude:

```text
source_amplitude
* path_throughput
* local_source_response
* (4π / listener_ray_count)   // omitted for the explicit direct path
* (1 / total_path_length)
```

Source and listener directivity are currently omnidirectional. Stereo uses the
initial listener-ray direction and the existing equal-power left/right pan.
The floating-point delay is deposited atomically into its two neighboring IR
samples. Contributions stay signed; no energy histogram or square-root
postprocessing is used.

## Diffraction heuristic

Authored diffraction strips and their precomputed edge, bisector, solid-wedge,
and shadow-boundary directions are reused. At a proxy hit, the physical event
is projected to the closest edge point. The valid cross-section interval runs
from the incoming projected direction toward the selected shadow boundary,
with a maximum bend of 90 degrees and a distance-dependent extent.

The detector response uses the normalized triangular radial kernel
`(1 - distance/radius) / radius` and a uniform angular density over the valid
interval. A continuation selects either transparent straight passage or one
sampled diffraction direction. Edge-parallel direction is preserved for the
sampled continuation, while explicit point-source connection tests use the
cross-section interval. These choices are reciprocal for the supported
manually authored wedge cases but remain a heuristic rather than a UTD model.

## Difference from the removed tracer

The former model emitted source-energy rays, recursively spawned children from
closest-hit shaders, and detected a listener volume. The current model treats
listener rays as Monte Carlo samples, explicitly connects every accepted
vertex to the source, retains complete listener-to-source lengths, and writes
the final pressure IR directly during ray tracing.
