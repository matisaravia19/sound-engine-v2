use super::*;
use crate::gpu::resource::OwnedFence;

fn audit_config() -> EngineConfig {
    EngineConfig {
        sound: crate::core::config::SoundConfig {
            output_channels: OutputChannels::Stereo,
            sample_rate: 48_000,
            ir_num_samples: 1024,
        },
        acoustics: crate::core::config::AcousticsConfig {
            listener_ray_count: 1,
            max_bounces: 0,
            maximum_path_length: 100.0,
            minimum_throughput: 0.0,
            sampling_seed: 0,
        },
        cache: Default::default(),
        binaural: Default::default(),
        auralization: crate::core::config::AuralizationConfig {
            block_size: 1024,
        },
    }
}

#[test]
fn fft_dimensions_reject_zero_and_overflow_before_allocating() {
    let mut config = audit_config();
    assert_eq!(validated_fft_size(config).unwrap(), 2048);
    config.auralization.block_size = 0;
    assert!(validated_fft_size(config).is_err());
    config.auralization.block_size = usize::MAX;
    assert!(validated_fft_size(config).is_err());
    config.auralization.block_size = 1024;
    config.sound.ir_num_samples = 0;
    assert!(validated_fft_size(config).is_err());
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn audit_native_fft_transfers_match_dft_and_round_trip() {
    let gpu = VkBackend::new().unwrap();
    let fence = OwnedFence::new(gpu.device_context()).unwrap();
    // Cover small radix kernels and the larger plans used by audio convolution.
    for size in [1, 2, 4, 8, 16, 32, 256, 1024, 65_536] {
        let (plan, buffer) =
            FftConvolver::build_fft_plan(&gpu, size as u64, true, fence.handle).unwrap();
        let input: Vec<[f32; 2]> = (0..size)
            .map(|index| [(index % 13) as f32 * 0.01, (index % 7) as f32 * -0.02])
            .collect();
        gpu.memory().upload_typed(&buffer, &input).unwrap();
        gpu.compute()
            .submit_compute_and_wait(|command| {
                // Both the plan and buffer outlive this synchronous submission.
                unsafe { plan.append(command) }
                    .map_err(|error| SoundError::external(format!("Forward FFT failed: {error:?}")))
            })
            .unwrap();
        let mut actual = vec![[0.0_f32; 2]; size];
        gpu.memory()
            .download_typed(&buffer, size, &mut actual)
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
                unsafe { plan.append_inverse(command) }
                    .map_err(|error| SoundError::external(format!("Inverse FFT failed: {error:?}")))
            })
            .unwrap();
        gpu.memory()
            .download_typed(&buffer, size, &mut actual)
            .unwrap();
        for (actual, expected) in actual.iter().zip(&input) {
            for channel in 0..2 {
                assert!((actual[channel] - expected[channel]).abs() < 2e-4);
            }
        }
        // The unsafe plan borrows its buffer, so destroy it first even in this test.
        drop(plan);
        drop(buffer);
    }
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn audit_batched_convolution_matches_cpu_reference_and_clears_reused_padding() {
    let gpu = Arc::new(VkBackend::new().unwrap());
    let mut convolver = FftConvolver::new(gpu.clone(), audit_config()).unwrap();
    let taps = [
        IrSample::new(0.5, -0.25),
        IrSample::new(0.125, 0.75),
        IrSample::new(-0.25, 0.125),
    ];
    let input: Vec<_> = (0..2048)
        .map(|index| ((index % 17) as f32 - 8.0) * 0.01)
        .collect();
    let buffer = Arc::new(
        gpu.memory()
            .create_storage_buffer(std::mem::size_of_val(&taps) as u64)
            .unwrap(),
    );
    gpu.memory().upload_typed(&buffer, &taps).unwrap();
    let snapshot = GpuIrSnapshot {
        buffer: Some(buffer),
        byte_size: std::mem::size_of_val(&taps),
        sample_rate: 48_000,
        sample_count: taps.len() as u32,
        energy: None,
        scene_version: 1,
        query_id: 0,
    };
    let mut wrong_rate = snapshot.clone();
    wrong_rate.sample_rate = 44_100;
    assert!(
        convolver
            .create_impulse_response_from_gpu_ir(&wrong_rate)
            .is_err()
    );
    for from_gpu in [false, true] {
        // Leave nonzero data throughout the reusable scratch allocation;
        // a short replacement must not inherit the previous IR's FFT tail.
        drop(
            convolver
                .create_impulse_response(&vec![IrSample::new(1.0, 1.0); 1024])
                .unwrap(),
        );
        let ir = if from_gpu {
            convolver
                .create_impulse_response_from_gpu_ir(&snapshot)
                .unwrap()
        } else {
            convolver.create_impulse_response(&taps).unwrap()
        };
        let mut voice = Voice::new(1, 1, 1.0, ir, convolver.tail_size(), convolver.fft_size());
        let mut output = vec![0.0; 2048];
        for (block, samples) in input.chunks_exact(1024).enumerate() {
            convolver
                .process_block(&mut voice, samples, &mut output)
                .unwrap();
            for frame in 0..1024 {
                let index = block * 1024 + frame;
                let mut expected = [0.0; 2];
                for (delay, tap) in taps.iter().enumerate() {
                    if let Some(source) = index.checked_sub(delay) {
                        expected[0] += input[source] * tap.left;
                        expected[1] += input[source] * tap.right;
                    }
                }
                assert!((output[frame * 2] - expected[0]).abs() < 1e-4);
                assert!((output[frame * 2 + 1] - expected[1]).abs() < 1e-4);
            }
        }
    }
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn audit_fft_initialization_and_ir_owner_lifetime() {
    let gpu = Arc::new(VkBackend::new().unwrap());
    let device = Arc::downgrade(&gpu.device_context());
    let mut convolver = FftConvolver::new(gpu.clone(), audit_config()).unwrap();
    let ir = convolver
        .create_impulse_response(&[IrSample::new(1.0, 0.5)])
        .unwrap();
    let mut spectrum = [[0.0_f32; 2]; 8];
    gpu.memory()
        .download_typed(&ir.buffer, 8, &mut spectrum)
        .unwrap();
    for value in spectrum {
        assert!((value[0] - 1.0).abs() < 1e-4 && (value[1] - 0.5).abs() < 1e-4);
    }
    let input = vec![0.0; 1024];
    let mut output = vec![0.0; 2048];
    let mut malformed_voice = Voice::new(1, 1, 1.0, ir.clone(), 0, 1);
    assert!(
        convolver
            .process_block(&mut malformed_voice, &input, &mut output)
            .is_err()
    );
    drop(malformed_voice);
    let undersized_ir = Arc::new(ImpulseResponse {
        buffer: gpu.memory().create_storage_buffer(8).unwrap(),
    });
    let mut malformed_voice = Voice::new(
        1,
        1,
        1.0,
        undersized_ir,
        convolver.tail_size(),
        convolver.fft_size(),
    );
    assert!(
        convolver
            .process_block(&mut malformed_voice, &input, &mut output)
            .is_err()
    );
    drop(malformed_voice);
    drop(convolver);
    assert_eq!(gpu.compute().pipeline_count(), 0);
    for _ in 0..2 {
        let replacement = FftConvolver::new(gpu.clone(), audit_config()).unwrap();
        assert_eq!(gpu.compute().pipeline_count(), 1);
        drop(replacement);
        assert_eq!(gpu.compute().pipeline_count(), 0);
    }
    drop(gpu);
    assert!(device.upgrade().is_some());
    drop(ir);
    assert!(device.upgrade().is_none());
}

#[test]
fn gpu_ir_validation_rejects_missing_buffer() {
    let snapshot = gpu_ir_snapshot(1);

    let error = validate_gpu_ir_snapshot(&snapshot, 8, 48_000)
        .err()
        .unwrap();

    assert_eq!(error.code(), ErrorCode::InvalidArgument);
}

#[test]
fn gpu_ir_validation_rejects_empty_ir() {
    let snapshot = gpu_ir_snapshot(0);

    let error = validate_gpu_ir_snapshot(&snapshot, 8, 48_000)
        .err()
        .unwrap();

    assert_eq!(error.code(), ErrorCode::InvalidArgument);
}

#[test]
fn gpu_ir_validation_rejects_ir_larger_than_linear_capacity() {
    let snapshot = gpu_ir_snapshot(16);

    let error = validate_gpu_ir_snapshot(&snapshot, 8, 48_000)
        .err()
        .unwrap();

    assert_eq!(error.code(), ErrorCode::InvalidArgument);
}

#[test]
fn ir_transition_ramps_every_frame_and_reaches_the_incoming_signal() {
    let mut incoming = [0.0, 0.0, 0.0, 0.0];
    let outgoing = [1.0, -1.0, 1.0, -1.0];
    blend_ir_transition(&mut incoming, &outgoing, 2, 0, 2);

    assert_eq!(incoming, [0.5, -0.5, 0.0, 0.0]);
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn ir_capacity_reserves_padding_and_prevents_circular_wraparound() {
    let gpu = Arc::new(VkBackend::new().unwrap());
    let mut config = audit_config();
    config.auralization.block_size = 4;
    config.sound.ir_num_samples = 5;
    let mut convolver = FftConvolver::new(gpu.clone(), config).unwrap();
    assert_eq!(convolver.fft_size(), 8);

    // Six taps fit in eight FFT bins, but their convolution with four input
    // samples would need nine bins and wrap the last sample into the head.
    let oversized = [IrSample::new(1.0, 0.5); 6];
    assert!(convolver.create_impulse_response(&oversized).is_err());
    let buffer = Arc::new(gpu.memory().create_storage_buffer(48).unwrap());
    gpu.memory().upload_typed(&buffer, &oversized).unwrap();
    let mut snapshot = GpuIrSnapshot {
        buffer: Some(buffer.clone()),
        byte_size: 48,
        sample_rate: config.sound.sample_rate,
        sample_count: 6,
        energy: None,
        scene_version: 1,
        query_id: 0,
    };
    assert!(
        convolver
            .create_impulse_response_from_gpu_ir(&snapshot)
            .is_err()
    );

    let mut taps = [IrSample::new(0.0, 0.0); 5];
    taps[4] = IrSample::new(1.0, 0.5);
    gpu.memory().upload_typed(&buffer, &taps).unwrap();
    snapshot.sample_count = 5;
    snapshot.byte_size = 40;
    for from_gpu in [false, true] {
        let ir = if from_gpu {
            convolver
                .create_impulse_response_from_gpu_ir(&snapshot)
                .unwrap()
        } else {
            convolver.create_impulse_response(&taps).unwrap()
        };
        let mut voice = Voice::new(1, 1, 1.0, ir, convolver.tail_size(), convolver.fft_size());
        let mut output = [0.0; 8];
        convolver
            .process_block(&mut voice, &[0.0, 0.0, 0.0, 1.0], &mut output)
            .unwrap();
        assert!(output.iter().all(|sample| sample.abs() < 1e-5));
        assert!(convolver.flush_tail_block(&mut voice, &mut output).unwrap());
        assert!(output[..6].iter().all(|sample| sample.abs() < 1e-5));
        assert!((output[6] - 1.0).abs() < 1e-5);
        assert!((output[7] - 0.5).abs() < 1e-5);
        assert!(!convolver.flush_tail_block(&mut voice, &mut output).unwrap());
    }
}

fn gpu_ir_snapshot(sample_count: u32) -> GpuIrSnapshot {
    GpuIrSnapshot {
        buffer: None,
        byte_size: sample_count as usize * std::mem::size_of::<IrSample>(),
        sample_rate: 48_000,
        sample_count,
        energy: None,
        scene_version: 1,
        query_id: 0,
    }
}
