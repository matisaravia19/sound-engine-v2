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
mod tests {
    use super::*;

    #[test]
    fn exhaustion_cannot_reissue_the_last_identifier() {
        let mut next = u64::MAX - 1;
        assert_eq!(allocate_id(&mut next, "Test").unwrap(), u64::MAX - 1);
        for _ in 0..3 {
            assert!(allocate_id(&mut next, "Test").is_err());
            assert_eq!(next, u64::MAX);
        }
    }
}
