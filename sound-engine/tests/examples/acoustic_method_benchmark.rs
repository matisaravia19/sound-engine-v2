use super::*;

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn scalar_diffraction_tracks_shadow_scale_and_reversal() -> SoundResult<()> {
    let gpu = VkBackend::new()?;
    let names = [
        "diffraction_shallow_shadow",
        "diffraction_deep_shadow",
        "diffraction_unequal_distances",
        "diffraction_unequal_reversed",
        "diffraction_scale_double",
        "diffraction_wedge_240",
        "diffraction_wedge_330",
        "diffraction_wedge_360",
    ];
    for case in benchmark_cases()
        .into_iter()
        .filter(|case| names.contains(&case.name))
    {
        let expected: f64 = case
            .exact_paths
            .iter()
            .map(|path| path.pressure_reflection / path.length_meters)
            .sum();
        for seed in [31, 32] {
            let options = Options {
                rays: 65_536,
                seed,
                seed_repeats: 1,
                warm_repeats: 1,
                case_filter: None,
                output: None,
            };
            let result = trace_case(&gpu, &options, &case)?;
            assert!(
                result
                    .ir
                    .samples
                    .iter()
                    .all(|sample| sample.left.is_finite())
            );
            let measured: f64 = result
                .ir
                .samples
                .iter()
                .map(|sample| sample.left as f64)
                .sum();
            let error = measured / expected - 1.0;
            println!(
                "{} seed={seed} pressure_error_pct={:.4}",
                case.name,
                error * 100.0
            );
            // A pressure gate catches directional zeros, scale errors and
            // large normalization bias without depending on noisy FFT nulls.
            assert!(
                error.abs() < 0.15,
                "{} pressure error exceeds 15%: {error}",
                case.name
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn multiple_reflection_pressure_is_stable_under_rotation() -> SoundResult<()> {
    let gpu = VkBackend::new()?;
    for rotation in [
        glam::Mat4::IDENTITY,
        glam::Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2),
        glam::Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2),
    ] {
        let mut case = benchmark_cases()
            .into_iter()
            .find(|c| c.name == "parallel_planes")
            .unwrap();
        // Rotating the entire scene preserves exact path lengths while
        // changing their alignment with the fixed spherical ray lattice.
        case.source = rotation.transform_point3(case.source);
        case.listener = rotation.transform_point3(case.listener);
        for object in &mut case.scene.objects {
            object.transform = rotation * object.transform;
        }
        let expected: f64 = case
            .exact_paths
            .iter()
            .map(|p| p.pressure_reflection / p.length_meters)
            .sum();
        for seed in [31, 32, 33] {
            let options = Options {
                rays: 65_536,
                seed,
                seed_repeats: 1,
                warm_repeats: 1,
                case_filter: None,
                output: None,
            };
            let result = trace_case(&gpu, &options, &case)?;
            let measured: f64 = result.ir.samples.iter().map(|s| s.left as f64).sum();
            let error = (measured / expected - 1.0).abs();
            println!(
                "rotated parallel planes seed={seed} pressure_error_pct={:.4}",
                error * 100.0
            );
            assert!(
                error < 0.02,
                "six-reflection pressure error must be below 2%, got {error}"
            );
        }
    }
    Ok(())
}

#[test]
fn parallel_image_enumeration_has_direct_and_two_first_reflections() {
    let paths = parallel_image_paths(10.0, vec3(2.0, 0.0, 0.0), vec3(7.0, 0.0, 0.0), 1, 1.0);
    assert_eq!(paths.len(), 3);
    assert_eq!(
        paths
            .iter()
            .filter(|path| path._reflection_order == 0)
            .count(),
        1
    );
    assert_eq!(
        paths
            .iter()
            .filter(|path| path._reflection_order == 1)
            .count(),
        2
    );
}

#[test]
fn shoebox_first_order_has_six_walls_and_direct_path() {
    let paths = shoebox_image_paths(
        vec3(10.0, 8.0, 6.0),
        vec3(2.0, 2.0, 2.0),
        vec3(7.0, 5.0, 3.0),
        1,
        1.0,
    );
    assert_eq!(paths.len(), 7);
    assert_eq!(
        paths
            .iter()
            .filter(|path| path._reflection_order == 0)
            .count(),
        1
    );
    assert_eq!(
        paths
            .iter()
            .filter(|path| path._reflection_order == 1)
            .count(),
        6
    );
}

#[test]
fn rigid_shoebox_ir_window_needs_at_most_thirty_reflections() {
    let paths = rigid_shoebox_image_paths_within_length(
        vec3(10.0, 8.0, 6.0),
        vec3(2.1, 2.7, 2.2),
        vec3(7.4, 5.9, 3.8),
        MAXIMUM_PATH_LENGTH,
    );
    assert!(paths.len() > 1_000);
    assert!(
        paths
            .iter()
            .map(|path| path._reflection_order)
            .max()
            .unwrap()
            <= 30
    );
}

#[test]
fn btm_quadrature_is_converged_at_benchmark_resolution() {
    let source = vec3(-2.0, 0.0, 3.0);
    let listener = vec3(-2.0, 0.0, -3.0);
    let coarse = btm_diffraction_paths(source, listener, 1.5 * PI, -20.0, 20.0, 65_536);
    let fine = btm_diffraction_paths(source, listener, 1.5 * PI, -20.0, 20.0, 131_072);
    let coarse_transfer = exact_transfer(&coarse, 8_000.0);
    let fine_transfer = exact_transfer(&fine, 8_000.0);
    let relative_error =
        (coarse_transfer - fine_transfer).magnitude() / fine_transfer.magnitude().max(1.0e-12);
    assert!(
        relative_error < 1.0e-4,
        "relative quadrature error was {relative_error}"
    );
}

#[test]
fn fractional_reference_preserves_path_pressure() {
    let path = ExactPath {
        length_meters: 1.2345,
        pressure_reflection: 0.75,
        _reflection_order: 0,
    };
    let expected = path.pressure_reflection / path.length_meters;
    let samples = exact_fractional_ir(&[path]);
    assert!((samples.iter().sum::<f64>() - expected).abs() < 1.0e-12);
    assert_eq!(samples.iter().filter(|sample| **sample != 0.0).count(), 2);
}
