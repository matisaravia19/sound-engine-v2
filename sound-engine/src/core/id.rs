use super::error::{SoundError, SoundResult};

/// Allocates a never-reused identifier, reserving `u64::MAX` as exhausted state.
/// Failure leaves the counter unchanged so registries cannot replace live entries.
pub(crate) fn allocate_id(next: &mut u64, resource: &str) -> SoundResult<u64> {
    let following = next.checked_add(1).ok_or_else(|| {
        SoundError::resource_allocation_failed(format!("{resource} identifier space exhausted"))
    })?;
    let id = *next;
    *next = following;
    Ok(id)
}

#[cfg(test)]
#[path = "../../tests/unit/core/id.rs"]
mod tests;
