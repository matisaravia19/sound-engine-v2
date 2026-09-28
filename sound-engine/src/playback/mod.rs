mod device;
mod player;
mod queue;
mod state;

pub(crate) use player::PlaybackStopHandle;
#[cfg(test)]
pub(crate) use player::TestPlaybackControl;
pub use player::{PlaybackConfig, SoundPlayer};
