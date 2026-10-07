use super::*;

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn audit_native_fft_transfers_match_dft_and_round_trip() {
    let gpu = VkBackend::new().unwrap();
    let fence = OwnedFence::new(gpu.device_context()).unwrap();
    // Cover small radix kernels and the larger plans used by audio convolution.
    for size in [1, 2, 4, 8, 16, 32, 256, 1024, 65_536] {
        let workspace = FftWorkspace::new(&gpu, size, true, fence.handle).unwrap();
        let input: Vec<[f32; 2]> = (0..size)
            .map(|index| [(index % 13) as f32 * 0.01, (index % 7) as f32 * -0.02])
            .collect();
        gpu.memory()
            .upload_typed(&workspace.buffer, &input)
            .unwrap();
        gpu.compute()
            .submit_compute_and_wait(|command| {
                // Both the plan and buffer outlive this synchronous submission.
                unsafe { workspace.plan.append(command) }
                    .map_err(|error| SoundError::external(format!("Forward FFT failed: {error:?}")))
            })
            .unwrap();
        let mut actual = vec![[0.0_f32; 2]; size];
        gpu.memory()
            .download_typed(&workspace.buffer, size, &mut actual)
            .unwrap();
        // Check selected bins against an independent double-precision DFT.
        for bin in [0, 1.min(size - 1), size / 2, size - 1] {
            let mut expected = [0.0_f64; 2];
            for (index, value) in input.iter().enumerate() {
                let phase = -std::f64::consts::TAU * bin as f64 * index as f64 / size as f64;
                let (sin, cos) = phase.sin_cos();
                expected[0] += value[0] as f64 * cos - value[1] as f64 * sin;
                expected[1] += value[0] as f64 * sin + value[1] as f64 * cos;
            }
            for (channel, expected) in expected.iter().enumerate() {
                assert!(
                    (actual[bin][channel] as f64 - expected).abs() < 2e-3 + expected.abs() * 2e-6,
                    "FFT length {size}, bin {bin}: {:?} != {expected}",
                    actual[bin]
                );
            }
        }
        gpu.compute()
            .submit_compute_and_wait(|command| {
                unsafe { workspace.plan.append_inverse(command) }
                    .map_err(|error| SoundError::external(format!("Inverse FFT failed: {error:?}")))
            })
            .unwrap();
        gpu.memory()
            .download_typed(&workspace.buffer, size, &mut actual)
            .unwrap();
        for (actual, expected) in actual.iter().zip(&input) {
            for channel in 0..2 {
                assert!((actual[channel] - expected[channel]).abs() < 2e-4);
            }
        }
    }
}
