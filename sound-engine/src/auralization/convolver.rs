use crate::auralization::{AudioError, IrSnapshot, VoiceId};
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) struct PartitionedConvolver {
    sample_rate: u32,
    block_size: usize,
    voice_ir: HashMap<VoiceId, Arc<IrSnapshot>>,
}

impl PartitionedConvolver {
    pub(crate) fn new(sample_rate: u32, block_size: usize) -> Result<Self, AudioError> {
        if sample_rate == 0 {
            return Err(std::io::Error::other("sample_rate must be > 0").into());
        }

        if block_size == 0 {
            return Err(std::io::Error::other("block_size must be > 0").into());
        }

        Ok(Self {
            sample_rate,
            block_size,
            voice_ir: HashMap::new(),
        })
    }

    pub(crate) fn ensure_ir(&mut self, voice_id: VoiceId, ir: &Arc<IrSnapshot>) -> Result<(), AudioError> {
        if ir.sample_rate != self.sample_rate {
            return Err(std::io::Error::other(format!(
                "IR sample rate mismatch: got {}, expected {}",
                ir.sample_rate, self.sample_rate
            ))
            .into());
        }

        if ir.samples.is_empty() {
            return Err(std::io::Error::other("IR cannot be empty").into());
        }

        self.voice_ir.entry(voice_id).or_insert_with(|| Arc::clone(ir));
        Ok(())
    }

    pub(crate) fn process_voice(
        &mut self,
        voice_id: VoiceId,
        dry_block: &[f32],
        wet_block: &mut [f32],
    ) -> Result<(), AudioError> {
        if dry_block.len() != self.block_size || wet_block.len() != self.block_size {
            return Err(std::io::Error::other(format!(
                "Block size mismatch in convolver: dry {}, wet {}, expected {}",
                dry_block.len(),
                wet_block.len(),
                self.block_size
            ))
            .into());
        }

        if !self.voice_ir.contains_key(&voice_id) {
            return Err(std::io::Error::other(format!("IR not set for voice {}", voice_id)).into());
        }

        // Placeholder: this intentionally bypasses FFT partitioning and frequency-domain
        // multiplication. It keeps the wiring and call pattern in place until the
        // real convolution core is added.
        wet_block.copy_from_slice(dry_block);
        Ok(())
    }

    pub(crate) fn forget_voice(&mut self, voice_id: VoiceId) {
        self.voice_ir.remove(&voice_id);
    }
}
