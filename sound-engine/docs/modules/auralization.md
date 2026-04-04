# `auralization` Module Spec

## Purpose

Render final audio from active voices using partitioned convolution and simple spatial output.

## Types (`pub(crate)`)

```rust
pub(crate) struct AuralizationEngine {
    cfg: AuralizationConfig,
    convolver: PartitionedConvolver,
    active_voices: VoiceTable,
}

pub(crate) struct AuralizationConfig {
    pub sample_rate: u32,
    pub block_size: usize,
    pub output_channels: OutputChannels,
    pub ir_crossfade_ms: f32,
}

pub(crate) struct VoiceState {
    pub voice_id: VoiceId,
    pub sound: SoundId,
    pub gain_linear: f32,
    pub ir_current: Arc<IrSnapshot>,
    pub ir_pending: Option<Arc<IrSnapshot>>,
    pub spatial: SpatialParams,
}

pub(crate) struct SpatialParams {
    pub source_position: [f32; 3],
    pub listener_position: [f32; 3],
    pub listener_orientation_quat: [f32; 4],
}

pub(crate) struct VoiceId(pub u64);
pub(crate) type VoiceTable = Vec<VoiceState>;
```

## Exposed Methods (`pub(crate)`)

```rust
impl AuralizationEngine {
    pub fn new(cfg: AuralizationConfig) -> Result<Self, AudioError>;

    pub fn start_voice(&mut self, sound: SoundId, gain_db: f32) -> VoiceId;
    pub fn stop_voice(&mut self, voice_id: VoiceId);

    pub fn set_voice_ir(&mut self, voice_id: VoiceId, ir: Arc<IrSnapshot>);
    pub fn update_spatial(&mut self, voice_id: VoiceId, spatial: SpatialParams);

    pub fn process_block(
        &mut self,
        sound_bank: &dyn SoundBankReader,
        output: &mut [f32],
    ) -> Result<(), AudioError>;
}
```

Convolution submodule:

```rust
pub(crate) struct PartitionedConvolver;

impl PartitionedConvolver {
    pub fn new(sample_rate: u32, block_size: usize) -> Result<Self, AudioError>;
    pub fn ensure_ir(&mut self, voice_id: VoiceId, ir: &IrSnapshot) -> Result<(), AudioError>;
    pub fn process_voice(
        &mut self,
        voice_id: VoiceId,
        dry_block: &[f32],
        wet_block: &mut [f32],
    ) -> Result<(), AudioError>;
}
```

## Invariants

- Audio path must avoid blocking and heap allocation in steady state.
- `process_block` expects `output.len() == block_size * channels`.
- IR switches use crossfade to avoid clicks.
