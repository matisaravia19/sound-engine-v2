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
