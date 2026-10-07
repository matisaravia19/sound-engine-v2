use super::*;

const SOUND_CONFIG: SoundConfig = SoundConfig {
    output_channels: OutputChannels::Stereo,
    sample_rate: 48_000,
    ir_num_samples: 1024,
};
const BLOCK_CONFIG: AuralizationConfig = AuralizationConfig {
    block_size: 1024,
};

#[test]
fn fft_dimensions_reject_zero_and_overflow_before_allocating() {
    let mut sound = SOUND_CONFIG;
    let mut config = BLOCK_CONFIG;
    assert_eq!(validated_fft_size(sound, config).unwrap(), 2048);
    config.block_size = 0;
    assert!(validated_fft_size(sound, config).is_err());
    config.block_size = usize::MAX;
    assert!(validated_fft_size(sound, config).is_err());
    config.block_size = 1024;
    sound.ir_num_samples = 0;
    assert!(validated_fft_size(sound, config).is_err());
    sound.ir_num_samples = 1024;
    sound.sample_rate = 0;
    assert!(validated_fft_size(sound, config).is_err());
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn audit_batched_convolution_matches_cpu_reference_and_clears_reused_padding() {
    let gpu = Arc::new(VkBackend::new().unwrap());
    let mut convolver = FftConvolver::new(gpu.clone(), SOUND_CONFIG, BLOCK_CONFIG).unwrap();
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
    let snapshot =
        GpuIrSnapshot::new(buffer, 48_000, taps.len() as u32, 1, snapshot_query()).unwrap();
    let wrong_rate = GpuIrSnapshot::new(
        snapshot.buffer().clone(),
        44_100,
        taps.len() as u32,
        1,
        snapshot_query(),
    )
    .unwrap();
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
        let mut stream = convolver.create_stream(ir).unwrap();
        let mut output = vec![0.0; 2048];
        for (block, samples) in input.chunks_exact(1024).enumerate() {
            convolver
                .process_block(&mut stream, samples, &mut output)
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
    let mut convolver = FftConvolver::new(gpu.clone(), SOUND_CONFIG, BLOCK_CONFIG).unwrap();
    let ir = convolver
        .create_impulse_response(&[IrSample::new(1.0, 0.5)])
        .unwrap();
    let mut spectrum = [[0.0_f32; 2]; 8];
    gpu.memory()
        .download_typed(&ir.buffer, &mut spectrum)
        .unwrap();
    for value in spectrum {
        assert!((value[0] - 1.0).abs() < 1e-4 && (value[1] - 0.5).abs() < 1e-4);
    }
    let input = vec![0.0; 1024];
    let mut output = vec![0.0; 2048];
    let mut malformed_stream = ConvolutionStream::new(ir.clone(), 0, 1, 2);
    assert!(
        convolver
            .process_block(&mut malformed_stream, &input, &mut output)
            .is_err()
    );
    drop(malformed_stream);
    let undersized_ir = Arc::new(ImpulseResponse {
        buffer: gpu.memory().create_storage_buffer(8).unwrap(),
        sample_rate: SOUND_CONFIG.sample_rate,
    });
    assert!(convolver.create_stream(undersized_ir.clone()).is_err());
    let mut stream = convolver.create_stream(ir.clone()).unwrap();
    assert!(
        convolver
            .update_impulse_response(&mut stream, undersized_ir)
            .is_err()
    );
    assert!(stream.ir_transition.is_none());
    assert!(stream.pending_impulse_response.is_none());
    assert!(Arc::ptr_eq(&stream.current.impulse_response, &ir));
    drop(stream);
    drop(convolver);
    assert_eq!(gpu.compute().pipeline_count(), 0);
    for _ in 0..2 {
        let replacement = FftConvolver::new(gpu.clone(), SOUND_CONFIG, BLOCK_CONFIG).unwrap();
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
fn ir_length_rejects_empty_or_oversized_responses() {
    for count in [0, 16] {
        assert_eq!(
            validate_ir_length(count, 8).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
    }
    assert!(validate_ir_length(8, 8).is_ok());
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn ir_capacity_reserves_padding_and_prevents_circular_wraparound() {
    let gpu = Arc::new(VkBackend::new().unwrap());
    let sound = SoundConfig {
        ir_num_samples: 5,
        ..SOUND_CONFIG
    };
    let config = AuralizationConfig {
        block_size: 4,
    };
    let mut convolver = FftConvolver::new(gpu.clone(), sound, config).unwrap();
    assert_eq!(convolver.fft_size(), 8);

    // Six taps fit in eight FFT bins, but their convolution with four input
    // samples would need nine bins and wrap the last sample into the head.
    let oversized = [IrSample::new(1.0, 0.5); 6];
    assert!(convolver.create_impulse_response(&oversized).is_err());
    let buffer = Arc::new(gpu.memory().create_storage_buffer(48).unwrap());
    gpu.memory().upload_typed(&buffer, &oversized).unwrap();
    let mut snapshot =
        GpuIrSnapshot::new(buffer.clone(), sound.sample_rate, 6, 1, snapshot_query()).unwrap();
    assert!(
        convolver
            .create_impulse_response_from_gpu_ir(&snapshot)
            .is_err()
    );

    let mut taps = [IrSample::new(0.0, 0.0); 5];
    taps[4] = IrSample::new(1.0, 0.5);
    gpu.memory().upload_typed(&buffer, &taps).unwrap();
    snapshot = GpuIrSnapshot::new(buffer, sound.sample_rate, 5, 1, snapshot_query()).unwrap();
    for from_gpu in [false, true] {
        let ir = if from_gpu {
            convolver
                .create_impulse_response_from_gpu_ir(&snapshot)
                .unwrap()
        } else {
            convolver.create_impulse_response(&taps).unwrap()
        };
        let mut stream = convolver.create_stream(ir).unwrap();
        let mut output = [0.0; 8];
        convolver
            .process_block(&mut stream, &[0.0, 0.0, 0.0, 1.0], &mut output)
            .unwrap();
        assert!(output.iter().all(|sample| sample.abs() < 1e-5));
        assert!(
            convolver
                .flush_tail_block(&mut stream, &mut output)
                .unwrap()
        );
        assert!(output[..6].iter().all(|sample| sample.abs() < 1e-5));
        assert!((output[6] - 1.0).abs() < 1e-5);
        assert!((output[7] - 0.5).abs() < 1e-5);
        assert!(
            !convolver
                .flush_tail_block(&mut stream, &mut output)
                .unwrap()
        );
    }
}

fn snapshot_query() -> crate::acoustics::AcousticQuery {
    crate::acoustics::AcousticQuery {
        query_id: 0,
        source: crate::PointSource::new(glam::Vec3::X),
        listener: crate::ListenerPose::default(),
    }
}
