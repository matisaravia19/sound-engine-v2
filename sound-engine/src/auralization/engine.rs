// use crate::auralization::convolver::PartitionedConvolver;
// use crate::auralization::{
//     AudioError, AuralizationConfig, IrSnapshot, SoundBankReader, SoundId, VoiceId, VoiceRenderResult, VoiceState,
//     VoiceTable,
// };
// use std::collections::HashSet;
// use std::sync::Arc;

// pub(crate) struct AuralizationEngine {
//     config: AuralizationConfig,
//     convolver: PartitionedConvolver,
//     active_voices: VoiceTable,
//     next_voice_id: u64,
//     dry_block: Vec<f32>,
//     wet_block: Vec<f32>,
//     finished_voices: Vec<VoiceId>,
// }

// impl AuralizationEngine {
//     pub fn new(config: AuralizationConfig) -> Result<Self, AudioError> {
//         config.validate()?;

//         let block_size = config.block_size;
//         let sample_rate = config.sample_rate;
//         Ok(Self {
//             config,
//             convolver: PartitionedConvolver::new(sample_rate, block_size)?,
//             active_voices: Vec::new(),
//             next_voice_id: 1,
//             dry_block: vec![0.0; block_size],
//             wet_block: vec![0.0; block_size],
//             finished_voices: Vec::new(),
//         })
//     }

//     pub fn start_voice(&mut self, sound: SoundId, ir: Arc<IrSnapshot>) -> VoiceId {
//         let voice_id = self.next_voice_id;
//         self.next_voice_id = self.next_voice_id.saturating_add(1);
//         self.active_voices.push(VoiceState { voice_id, sound, ir });
//         voice_id
//     }

//     pub fn stop_voice(&mut self, voice_id: VoiceId) {
//         self.active_voices.retain(|voice| voice.voice_id != voice_id);
//         self.convolver.forget_voice(voice_id);
//     }

//     pub fn process_block(&mut self, sound_bank: &dyn SoundBankReader, output: &mut [f32]) -> Result<(), AudioError> {
//         let channels = self.config.output_channels.count();
//         let expected_output_len = self.config.block_size * channels;
//         if output.len() != expected_output_len {
//             return Err(std::io::Error::other(format!(
//                 "Output length mismatch: got {}, expected {}",
//                 output.len(),
//                 expected_output_len
//             ))
//             .into());
//         }

//         output.fill(0.0);
//         self.finished_voices.clear();

//         for voice in &self.active_voices {
//             self.dry_block.fill(0.0);
//             let render_state = sound_bank.render_voice_block(voice.sound, voice.voice_id, &mut self.dry_block)?;

//             self.convolver.ensure_ir(voice.voice_id, &voice.ir)?;

//             self.wet_block.fill(0.0);
//             self.convolver
//                 .process_voice(voice.voice_id, &self.dry_block, &mut self.wet_block)?;

//             for frame in 0..self.config.block_size {
//                 let sample = self.wet_block[frame];
//                 let frame_base = frame * channels;
//                 for ch in 0..channels {
//                     output[frame_base + ch] += sample;
//                 }
//             }

//             if matches!(render_state, VoiceRenderResult::Finished) {
//                 self.finished_voices.push(voice.voice_id);
//             }
//         }

//         if !self.finished_voices.is_empty() {
//             let finished_set: HashSet<VoiceId> = self.finished_voices.iter().copied().collect();
//             self.active_voices
//                 .retain(|voice| !finished_set.contains(&voice.voice_id));
//             for &voice in &self.finished_voices {
//                 self.convolver.forget_voice(voice);
//             }
//         }

//         Ok(())
//     }
// }
