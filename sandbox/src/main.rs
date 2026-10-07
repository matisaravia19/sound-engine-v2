//! Interactive spatial-audio playground.
mod app;
mod scenes;

fn main() -> sound_engine::core::error::SoundResult<()> {
    app::run()
}
