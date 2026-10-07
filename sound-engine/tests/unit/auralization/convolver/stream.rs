use super::*;

#[test]
fn ir_transition_ramps_every_frame_and_reaches_the_incoming_signal() {
    let mut incoming = [0.0, 0.0, 0.0, 0.0];
    let outgoing = [1.0, -1.0, 1.0, -1.0];
    blend_ir_transition(&mut incoming, &outgoing, 2, 0, 2);

    assert_eq!(incoming, [0.5, -0.5, 0.0, 0.0]);
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn reaffirming_the_incoming_filter_cancels_the_queued_replacement() {
    use crate::acoustics::IrSample;
    use crate::auralization::convolver::FftConvolver;
    use crate::core::config::{AuralizationConfig, OutputChannels, SoundConfig};
    use crate::gpu::backend::VkBackend;

    let gpu = Arc::new(VkBackend::new().unwrap());
    let mut convolver = FftConvolver::new(
        gpu,
        SoundConfig {
            sample_rate: 48_000,
            ir_num_samples: 8,
            output_channels: OutputChannels::Mono,
        },
        AuralizationConfig {
            block_size: 4,
        },
    )
    .unwrap();
    let outgoing = convolver
        .create_impulse_response(&[IrSample::new(1.0, 0.0)])
        .unwrap();
    let incoming = convolver
        .create_impulse_response(&[IrSample::new(2.0, 0.0)])
        .unwrap();
    let queued = convolver
        .create_impulse_response(&[IrSample::new(3.0, 0.0)])
        .unwrap();
    let mut stream = convolver.create_stream(outgoing).unwrap();
    convolver
        .update_impulse_response(&mut stream, incoming.clone())
        .unwrap();
    convolver
        .update_impulse_response(&mut stream, queued)
        .unwrap();
    let mut output = [0.0; 4];
    convolver
        .process_block(&mut stream, &[1.0; 4], &mut output)
        .unwrap();
    assert!(stream.pending_impulse_response.is_some());
    let progress = stream.ir_transition.as_ref().unwrap().processed_frames;
    convolver
        .update_impulse_response(&mut stream, incoming.clone())
        .unwrap();
    assert!(stream.pending_impulse_response.is_none());
    assert_eq!(
        stream.ir_transition.as_ref().unwrap().processed_frames,
        progress
    );
    for _ in 0..4 {
        convolver
            .process_block(&mut stream, &[1.0; 4], &mut output)
            .unwrap();
    }
    assert!(stream.ir_transition.is_none());
    assert!(Arc::ptr_eq(&stream.current.impulse_response, &incoming));
    assert!(output.iter().all(|sample| (*sample - 2.0).abs() < 1e-5));
}
