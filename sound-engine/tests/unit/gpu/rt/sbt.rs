use super::*;

#[test]
fn trace_regions_match_group_order_and_omit_missing_kinds() {
    let groups = [
        RtShaderGroupSpec::Miss {
            shader: 0,
        },
        RtShaderGroupSpec::Miss {
            shader: 1,
        },
        RtShaderGroupSpec::Raygen {
            shader: 2,
        },
        RtShaderGroupSpec::ProceduralHit {
            closest_hit_shader: 3,
            intersection_shader: 4,
        },
        RtShaderGroupSpec::TrianglesHit {
            closest_hit_shader: 5,
        },
    ];
    // Pipeline group order is preserved even when the ray-generation
    // record comes after a multi-record miss region.
    for (kind, address, size) in [
        (ShaderGroupKind::Raygen, 0x1080, 64),
        (ShaderGroupKind::Miss, 0x1000, 128),
        (ShaderGroupKind::Hit, 0x10c0, 128),
    ] {
        let region = region_for(&groups, 0x1000, 64, kind);
        assert_eq!(region.device_address, address);
        assert_eq!(region.stride, 64);
        assert_eq!(region.size, size);
    }
    for kind in [ShaderGroupKind::Miss, ShaderGroupKind::Hit] {
        let region = region_for(&groups[2..3], 0x1000, 64, kind);
        assert_eq!(
            (region.device_address, region.stride, region.size),
            (0, 0, 0)
        );
    }
}

#[test]
fn align_up_keeps_aligned_values() {
    assert_eq!(align_up(64, 32), 64);
}

#[test]
fn align_up_rounds_to_next_boundary() {
    assert_eq!(align_up(65, 32), 96);
}
