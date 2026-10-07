use super::*;

#[test]
fn ir_transition_ramps_every_frame_and_reaches_the_incoming_signal() {
    let mut incoming = [0.0, 0.0, 0.0, 0.0];
    let outgoing = [1.0, -1.0, 1.0, -1.0];
    blend_ir_transition(&mut incoming, &outgoing, 2, 0, 2);

    assert_eq!(incoming, [0.5, -0.5, 0.0, 0.0]);
}
