//! Analytical references; never used by runtime transport.
use super::*;

/// Defines fixed direct, reflection, room and diffraction validation controls.
pub(super) fn benchmark_cases() -> Vec<BenchmarkCase> {
    let direct_source = Vec3::ZERO;
    let direct_listener_1m = vec3(1.003, 0.217, -0.091);
    let direct_listener_2m = direct_listener_1m.normalize() * 2.0;
    let direct_listener_5m = direct_listener_1m.normalize() * 5.0;

    let plane_source = vec3(1.1, -0.3, 0.2);
    let plane_listener = vec3(2.7, 0.8, -0.4);
    let plane_direct = plane_source.distance(plane_listener) as f64;
    let plane_image = vec3(-plane_source.x, plane_source.y, plane_source.z);
    let plane_reflected = plane_image.distance(plane_listener) as f64;

    let parallel_length = 10.0;
    let parallel_source = vec3(2.1, -0.4, 0.3);
    let parallel_listener = vec3(7.3, 0.6, -0.2);
    let parallel_bounces = 6;

    let room_size = vec3(10.0, 8.0, 6.0);
    let room_source = vec3(2.1, 2.7, 2.2);
    let room_listener = vec3(7.4, 5.9, 3.8);
    let room_bounces = 6;
    // Thirty reflections cover every rigid-room image that can arrive inside
    // this benchmark's 341 ms IR. The test below guards that assumption if the
    // room dimensions, sample count, or speed of sound change.
    let rigid_room_bounces = 30;
    let diffraction_source = vec3(-2.0, 0.0, 3.0);
    let diffraction_listener = vec3(-2.0, 0.0, -3.0);
    let exterior_wedge_angle = 1.5 * PI;
    let diffraction_exact = btm_diffraction_paths(
        diffraction_source,
        diffraction_listener,
        exterior_wedge_angle,
        -DIFFRACTION_EDGE_HALF_LENGTH,
        DIFFRACTION_EDGE_HALF_LENGTH,
        DIFFRACTION_QUADRATURE_SEGMENTS,
    );

    let mut cases = vec![
        BenchmarkCase {
            name: "free_field_1m",
            source: direct_source,
            listener: direct_listener_1m,
            max_bounces: 0,
            scene: empty_scene(),
            exact_paths: vec![ExactPath {
                length_meters: direct_source.distance(direct_listener_1m) as f64,
                pressure_reflection: 1.0,
                _reflection_order: 0,
            }],
        },
        BenchmarkCase {
            name: "free_field_2m",
            source: direct_source,
            listener: direct_listener_2m,
            max_bounces: 0,
            scene: empty_scene(),
            exact_paths: vec![ExactPath {
                length_meters: 2.0,
                pressure_reflection: 1.0,
                _reflection_order: 0,
            }],
        },
        BenchmarkCase {
            name: "free_field_5m",
            source: direct_source,
            listener: direct_listener_5m,
            max_bounces: 0,
            scene: empty_scene(),
            exact_paths: vec![ExactPath {
                length_meters: 5.0,
                pressure_reflection: 1.0,
                _reflection_order: 0,
            }],
        },
        BenchmarkCase {
            name: "rigid_single_plane",
            source: plane_source,
            listener: plane_listener,
            max_bounces: 1,
            scene: single_plane_scene(0.0),
            exact_paths: vec![
                ExactPath {
                    length_meters: plane_direct,
                    pressure_reflection: 1.0,
                    _reflection_order: 0,
                },
                ExactPath {
                    length_meters: plane_reflected,
                    pressure_reflection: 1.0,
                    _reflection_order: 1,
                },
            ],
        },
        BenchmarkCase {
            name: "absorbing_single_plane",
            source: plane_source,
            listener: plane_listener,
            max_bounces: 1,
            scene: single_plane_scene(ROOM_ABSORPTION_ENERGY),
            exact_paths: vec![
                ExactPath {
                    length_meters: plane_direct,
                    pressure_reflection: 1.0,
                    _reflection_order: 0,
                },
                ExactPath {
                    length_meters: plane_reflected,
                    pressure_reflection: ROOM_REFLECTION_PRESSURE,
                    _reflection_order: 1,
                },
            ],
        },
        BenchmarkCase {
            name: "parallel_planes",
            source: parallel_source,
            listener: parallel_listener,
            max_bounces: parallel_bounces,
            scene: parallel_plane_scene(parallel_length, ROOM_ABSORPTION_ENERGY),
            exact_paths: parallel_image_paths(
                parallel_length as f64,
                parallel_source,
                parallel_listener,
                parallel_bounces,
                ROOM_REFLECTION_PRESSURE,
            ),
        },
        BenchmarkCase {
            name: "closed_shoebox",
            source: room_source,
            listener: room_listener,
            max_bounces: room_bounces,
            scene: shoebox_scene(room_size, ROOM_ABSORPTION_ENERGY),
            exact_paths: shoebox_image_paths(
                room_size,
                room_source,
                room_listener,
                room_bounces,
                ROOM_REFLECTION_PRESSURE,
            ),
        },
        BenchmarkCase {
            name: "rigid_shoebox_exact",
            source: room_source,
            listener: room_listener,
            max_bounces: rigid_room_bounces,
            scene: shoebox_scene(room_size, 0.0),
            // For rigid (Neumann) rectangular boundaries, mirroring the
            // monopole in every room cell is an exact time-domain solution of
            // the scalar wave equation. Retaining all images whose arrival is
            // inside the finite IR gives an exact reference for this window.
            exact_paths: rigid_shoebox_image_paths_within_length(
                room_size,
                room_source,
                room_listener,
                MAXIMUM_PATH_LENGTH,
            ),
        },
        BenchmarkCase {
            name: "diffraction_wedge_shadow",
            source: diffraction_source,
            listener: diffraction_listener,
            max_bounces: 1,
            scene: diffraction_wedge_scene(
                exterior_wedge_angle as f32,
                DIFFRACTION_EDGE_HALF_LENGTH as f32,
                0.45,
            ),
            // The Biot-Tolstoy-Medwin secondary-edge-source integral is the
            // exact first-order solution for an infinite rigid wedge. Truncating
            // the integral to the authored edge gives the standard finite-edge
            // first-order reference used by interactive geometric acoustics.
            exact_paths: diffraction_exact.clone(),
        },
        BenchmarkCase {
            name: "diffraction_proxy_radius_020",
            source: diffraction_source,
            listener: diffraction_listener,
            max_bounces: 1,
            scene: diffraction_wedge_scene(
                exterior_wedge_angle as f32,
                DIFFRACTION_EDGE_HALF_LENGTH as f32,
                0.20,
            ),
            exact_paths: diffraction_exact.clone(),
        },
        BenchmarkCase {
            name: "diffraction_proxy_radius_090",
            source: diffraction_source,
            listener: diffraction_listener,
            max_bounces: 1,
            scene: diffraction_wedge_scene(
                exterior_wedge_angle as f32,
                DIFFRACTION_EDGE_HALF_LENGTH as f32,
                0.90,
            ),
            exact_paths: diffraction_exact,
        },
    ];
    // Additional geometries are validation controls, not inputs to the global
    // edge-gain calibration. Keep the original wedge/radius cases unchanged.
    for (name, source, listener) in [
        (
            "diffraction_shallow_shadow",
            vec3(-0.5, 0.0, 4.0),
            vec3(-0.5, 0.0, -4.0),
        ),
        (
            "diffraction_deep_shadow",
            vec3(-2.8, 0.0, 3.0),
            vec3(-2.8, 0.0, -3.0),
        ),
        (
            "diffraction_unequal_distances",
            vec3(-1.0, 0.0, 2.0),
            vec3(-3.0, 0.0, -4.0),
        ),
        (
            "diffraction_axial_offset",
            vec3(-2.0, 1.0, 3.0),
            vec3(-2.0, -1.0, -3.0),
        ),
    ] {
        cases.push(BenchmarkCase {
            name,
            source,
            listener,
            max_bounces: 1,
            scene: diffraction_wedge_scene(
                exterior_wedge_angle as f32,
                DIFFRACTION_EDGE_HALF_LENGTH as f32,
                0.45,
            ),
            exact_paths: btm_diffraction_paths(
                source,
                listener,
                exterior_wedge_angle,
                -DIFFRACTION_EDGE_HALF_LENGTH,
                DIFFRACTION_EDGE_HALF_LENGTH,
                DIFFRACTION_QUADRATURE_SEGMENTS,
            ),
        });
    }
    // Withheld controls exercise reversal, metric scale and shadow directions
    // beyond the old forward-hemisphere cutoff. No parameters are fitted here.
    for (name, source, listener, scale) in [
        (
            "diffraction_unequal_reversed",
            vec3(-3.0, 0.0, -4.0),
            vec3(-1.0, 0.0, 2.0),
            1.0,
        ),
        (
            "diffraction_scale_double",
            diffraction_source * 2.0,
            diffraction_listener * 2.0,
            2.0,
        ),
    ] {
        let half_length = DIFFRACTION_EDGE_HALF_LENGTH * scale;
        cases.push(BenchmarkCase {
            name,
            source,
            listener,
            max_bounces: 1,
            scene: diffraction_wedge_scene(
                exterior_wedge_angle as f32,
                half_length as f32,
                0.45 * scale as f32,
            ),
            exact_paths: btm_diffraction_paths(
                source,
                listener,
                exterior_wedge_angle,
                -half_length,
                half_length,
                DIFFRACTION_QUADRATURE_SEGMENTS,
            ),
        });
    }
    for (name, degrees) in [
        ("diffraction_wedge_210", 210.0_f64),
        ("diffraction_wedge_240", 240.0),
        ("diffraction_wedge_300", 300.0),
        ("diffraction_wedge_330", 330.0),
        ("diffraction_wedge_360", 360.0),
    ] {
        let exterior = degrees.to_radians();
        let source_angle = 0.12 * (exterior - PI);
        let listener_angle = PI + source_angle + 0.83 * (exterior - PI - source_angle);
        let position = |angle: f64, radius: f64, height: f32| {
            let radial_angle = 0.5 * exterior - angle;
            vec3(
                (radius * radial_angle.cos()) as f32,
                height,
                (radius * radial_angle.sin()) as f32,
            )
        };
        let source = position(source_angle, 4.2, -0.7);
        let listener = position(listener_angle, 6.7, 0.4);
        cases.push(BenchmarkCase {
            name,
            source,
            listener,
            max_bounces: 1,
            scene: diffraction_wedge_scene(
                exterior as f32,
                DIFFRACTION_EDGE_HALF_LENGTH as f32,
                0.45,
            ),
            exact_paths: btm_diffraction_paths(
                source,
                listener,
                exterior,
                -DIFFRACTION_EDGE_HALF_LENGTH,
                DIFFRACTION_EDGE_HALF_LENGTH,
                DIFFRACTION_QUADRATURE_SEGMENTS,
            ),
        });
    }
    cases
}

/// Sums continuous-delay path pressures coherently at one frequency.
pub(super) fn exact_transfer(paths: &[ExactPath], frequency_hz: f64) -> Complex {
    let wave_number = 2.0 * PI * frequency_hz / SPEED_OF_SOUND;
    let mut result = Complex::ZERO;
    for path in paths {
        let amplitude = path.pressure_reflection / path.length_meters.max(1.0e-9);
        result += Complex::from_polar(amplitude, -wave_number * path.length_meters);
    }
    result
}

/// Deposits analytical path pressure into the engine sample grid using linear delay weights.
pub(super) fn exact_fractional_ir(paths: &[ExactPath]) -> Vec<f64> {
    let mut samples = vec![0.0; SAMPLE_COUNT as usize];
    for path in paths {
        let sample_position = path.length_meters / SPEED_OF_SOUND * SAMPLE_RATE as f64;
        let base = sample_position.floor() as usize;
        let fraction = sample_position - base as f64;
        let amplitude = path.pressure_reflection / path.length_meters.max(1.0e-9);
        if let Some(sample) = samples.get_mut(base) {
            *sample += amplitude * (1.0 - fraction);
        }
        if let Some(sample) = samples.get_mut(base + 1) {
            *sample += amplitude * fraction;
        }
    }
    samples
}

/// Evaluates the mono pressure transfer function of a downloaded IR.
pub(super) fn transfer_from_samples(samples: &[IrSample], frequency_hz: f64) -> Complex {
    let mut result = Complex::ZERO;
    let angular_step = -2.0 * PI * frequency_hz / SAMPLE_RATE as f64;
    for (index, sample) in samples.iter().enumerate() {
        result += Complex::from_polar(sample.left as f64, angular_step * index as f64);
    }
    result
}

/// Evaluates the sampled analytical reference without narrowing it to f32.
pub(super) fn transfer_from_f64_samples(samples: &[f64], frequency_hz: f64) -> Complex {
    let mut result = Complex::ZERO;
    let angular_step = -2.0 * PI * frequency_hz / SAMPLE_RATE as f64;
    for (index, &sample) in samples.iter().enumerate() {
        result += Complex::from_polar(sample, angular_step * index as f64);
    }
    result
}

/// Compares signed pressure samples and reports norm errors and arrival sparsity.
pub(super) fn time_domain_metrics(measured: &[IrSample], exact: &[f64]) -> TimeDomainMetrics {
    let mut l1_error = 0.0;
    let mut l2_error = 0.0;
    let mut exact_l1 = 0.0;
    let mut exact_l2 = 0.0;
    let mut measured_pressure_sum = 0.0;
    let mut exact_pressure_sum = 0.0;
    let mut measured_peak_abs = 0.0_f64;
    let mut nonzero_samples = 0;
    for (measured, exact) in measured.iter().zip(exact) {
        let measured = measured.left as f64;
        let difference = measured - exact;
        l1_error += difference.abs();
        l2_error += difference * difference;
        exact_l1 += exact.abs();
        exact_l2 += exact * exact;
        measured_pressure_sum += measured;
        exact_pressure_sum += exact;
        measured_peak_abs = measured_peak_abs.max(measured.abs());
        nonzero_samples += usize::from(measured != 0.0);
    }
    TimeDomainMetrics {
        relative_l1: l1_error / exact_l1.max(1.0e-15),
        relative_l2: l2_error.sqrt() / exact_l2.sqrt().max(1.0e-15),
        measured_pressure_sum,
        exact_pressure_sum,
        measured_peak_abs,
        nonzero_samples,
    }
}

/// Enumerates the two-plane image sources up to a reflection-order bound.
pub(super) fn parallel_image_paths(
    room_length: f64,
    source: Vec3,
    listener: Vec3,
    max_order: u32,
    reflection_pressure: f64,
) -> Vec<ExactPath> {
    let mut paths = Vec::new();
    let cells = max_order as i32 + 1;
    for cell in -cells..=cells {
        for parity in 0..=1 {
            let order = (2 * cell - parity).unsigned_abs();
            if order > max_order {
                continue;
            }
            let image_x = 2.0 * cell as f64 * room_length
                + if parity == 0 {
                    source.x as f64
                } else {
                    -(source.x as f64)
                };
            let image = Vec3::new(image_x as f32, source.y, source.z);
            paths.push(ExactPath {
                length_meters: image.distance(listener) as f64,
                pressure_reflection: reflection_pressure.powi(order as i32),
                _reflection_order: order,
            });
        }
    }
    paths
}

/// Enumerates rectangular-room images up to a reflection-order bound.
pub(super) fn shoebox_image_paths(
    room_size: Vec3,
    source: Vec3,
    listener: Vec3,
    max_order: u32,
    reflection_pressure: f64,
) -> Vec<ExactPath> {
    let cells = max_order as i32 + 1;
    let mut paths = Vec::new();
    for nx in -cells..=cells {
        for px in 0..=1 {
            let ox = (2 * nx - px).unsigned_abs();
            for ny in -cells..=cells {
                for py in 0..=1 {
                    let oy = (2 * ny - py).unsigned_abs();
                    for nz in -cells..=cells {
                        for pz in 0..=1 {
                            let oz = (2 * nz - pz).unsigned_abs();
                            let order = ox + oy + oz;
                            if order > max_order {
                                continue;
                            }
                            let image = Vec3::new(
                                image_coordinate(room_size.x, source.x, nx, px),
                                image_coordinate(room_size.y, source.y, ny, py),
                                image_coordinate(room_size.z, source.z, nz, pz),
                            );
                            paths.push(ExactPath {
                                length_meters: image.distance(listener) as f64,
                                pressure_reflection: reflection_pressure.powi(order as i32),
                                _reflection_order: order,
                            });
                        }
                    }
                }
            }
        }
    }
    paths
}

/// Includes every rigid-room image whose arrival lies inside the requested path window.
pub(super) fn rigid_shoebox_image_paths_within_length(
    room_size: Vec3,
    source: Vec3,
    listener: Vec3,
    maximum_length: f64,
) -> Vec<ExactPath> {
    // Each image coordinate is 2*n*L +/- source. One extra cell beyond the
    // simple radius bound accounts for the source/listener offsets within the
    // base room; the final Euclidean-distance check removes that overreach.
    let x_cells = (maximum_length / (2.0 * room_size.x as f64)).ceil() as i32 + 1;
    let y_cells = (maximum_length / (2.0 * room_size.y as f64)).ceil() as i32 + 1;
    let z_cells = (maximum_length / (2.0 * room_size.z as f64)).ceil() as i32 + 1;
    let mut paths = Vec::new();
    for nx in -x_cells..=x_cells {
        for px in 0..=1 {
            let ox = (2 * nx - px).unsigned_abs();
            for ny in -y_cells..=y_cells {
                for py in 0..=1 {
                    let oy = (2 * ny - py).unsigned_abs();
                    for nz in -z_cells..=z_cells {
                        for pz in 0..=1 {
                            let oz = (2 * nz - pz).unsigned_abs();
                            let image = Vec3::new(
                                image_coordinate(room_size.x, source.x, nx, px),
                                image_coordinate(room_size.y, source.y, ny, py),
                                image_coordinate(room_size.z, source.z, nz, pz),
                            );
                            let length = image.distance(listener) as f64;
                            if length <= maximum_length {
                                paths.push(ExactPath {
                                    length_meters: length,
                                    pressure_reflection: 1.0,
                                    _reflection_order: ox + oy + oz,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    paths
}

/// Integrates a finite rigid wedge with midpoint BTM quadrature for offline validation only.
pub(super) fn btm_diffraction_paths(
    source: Vec3,
    listener: Vec3,
    exterior_wedge_angle: f64,
    edge_start: f64,
    edge_end: f64,
    segment_count: usize,
) -> Vec<ExactPath> {
    // This benchmark wedge is aligned to +Y. Its upper face points along
    // the positive half-angle from +X, and angles increase around +Y through the exterior
    // (air) region. Source and receiver radii are therefore measured in XZ.
    let edge_axis = Vec3::Y;
    let half_angle = 0.5 * exterior_wedge_angle;
    let angle_origin = vec3(half_angle.cos() as f32, 0.0, half_angle.sin() as f32);
    let source_radial = vec3(source.x, 0.0, source.z);
    let listener_radial = vec3(listener.x, 0.0, listener.z);
    let source_radius = source_radial.length() as f64;
    let listener_radius = listener_radial.length() as f64;
    let source_angle = wedge_angle(angle_origin, edge_axis, source_radial);
    let listener_angle = wedge_angle(angle_origin, edge_axis, listener_radial);
    let wedge_index = PI / exterior_wedge_angle;
    let segment_length = (edge_end - edge_start) / segment_count as f64;
    let mut paths = Vec::with_capacity(segment_count);

    for segment in 0..segment_count {
        // Midpoint integration avoids evaluating the integrable BTM onset
        // singularity exactly at the shortest source-edge-receiver path.
        let edge_coordinate = edge_start + (segment as f64 + 0.5) * segment_length;
        let source_axial = edge_coordinate - source.y as f64;
        let listener_axial = edge_coordinate - listener.y as f64;
        let source_distance = source_radius.hypot(source_axial);
        let listener_distance = listener_radius.hypot(listener_axial);
        let eta_argument = (source_distance * listener_distance + source_axial * listener_axial)
            / (source_radius * listener_radius);
        let eta = eta_argument.max(1.0).acosh();

        let mut beta_sum = 0.0;
        for source_sign in [-1.0, 1.0] {
            for listener_sign in [-1.0, 1.0] {
                let phi = PI + source_sign * source_angle + listener_sign * listener_angle;
                let numerator = (wedge_index * phi).sin();
                let denominator = (wedge_index * eta).cosh() - (wedge_index * phi).cos();
                beta_sum += numerator / denominator;
            }
        }

        let total_length = source_distance + listener_distance;
        // The published source convention already produces a 1/r direct field,
        // so the BTM coefficient retains its 1/(4*pi) factor.
        let pressure = -wedge_index / (4.0 * PI) * beta_sum * segment_length
            / (source_distance * listener_distance);
        paths.push(ExactPath {
            length_meters: total_length,
            // ExactPath stores the numerator of pressure/length.
            pressure_reflection: pressure * total_length,
            _reflection_order: 1,
        });
    }
    paths
}

/// Measures a radial direction around the oriented edge axis in [0, 2*pi).
pub(super) fn wedge_angle(origin: Vec3, axis: Vec3, direction: Vec3) -> f64 {
    let origin = origin.normalize();
    let direction = direction.normalize();
    let sine = axis.dot(origin.cross(direction)) as f64;
    let cosine = origin.dot(direction) as f64;
    sine.atan2(cosine).rem_euclid(2.0 * PI)
}

/// Maps a source coordinate into a mirrored periodic room cell.
pub(super) fn image_coordinate(length: f32, source: f32, cell: i32, parity: i32) -> f32 {
    2.0 * cell as f32 * length
        + if parity == 0 {
            source
        } else {
            -source
        }
}
