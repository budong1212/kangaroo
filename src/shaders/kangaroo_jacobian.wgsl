// =============================================================================
// Pollard's Kangaroo Algorithm - GPU Kernel (Pure Jacobian)
// =============================================================================
//
// ALGORITHM CONTRACT:
//   Tame kangaroo i:  position = (start + offset_i) * G,  dist = offset_i
//   Wild  kangaroo i: position = pubkey,                   dist = 0
//
//   After collision (same affine X, different ktype):
//     start + tame_dist = key + wild_dist
//     => key = start + tame_dist - wild_dist
//
// STEP ORDER (critical for dist/point consistency):
//   1. Select jump index from current Jacobian X low byte
//   2. Add jump point  (point advances)
//   3. Add jump dist   (dist advances to match new point)
//   4. Check DP on NEW point X low bits
//   5. If DP: store (Jacobian X, Z, dist) for CPU affine conversion
//
// DP detection uses Jacobian X[0] low bits as a FAST PROXY.
// Because Z changes randomly every step, X_jac low bits are uniformly
// distributed, giving a valid (approximate) DP filter.
// CPU does the true affine X = X_jac * Z_jac^{-2} and uses that for
// collision detection — so false positives just waste a little CPU time.
// =============================================================================

struct Config {
    dp_mask_lo: vec4<u32>,
    dp_mask_hi: vec4<u32>,
    num_kangaroos: u32,
    steps_per_call: u32,
    jump_table_size: u32,
    _padding: u32
}

struct Kangaroo {
    x: array<u32, 8>,
    y: array<u32, 8>,
    z: array<u32, 8>,
    dist: array<u32, 8>,
    ktype: u32,
    is_active: u32,
    _padding: array<u32, 2>
}

struct DistinguishedPoint {
    x: array<u32, 8>,
    z: array<u32, 8>,
    dist: array<u32, 8>,
    ktype: u32,
    kangaroo_id: u32,
}

@group(0) @binding(0) var<uniform>            config:         Config;
@group(0) @binding(1) var<storage, read>      jump_points:    array<AffinePoint, 256>;
@group(0) @binding(2) var<storage, read>      jump_distances: array<array<u32, 8>, 256>;
@group(0) @binding(3) var<storage, read_write> kangaroos:     array<Kangaroo>;
@group(0) @binding(4) var<storage, read_write> dp_buffer:     array<DistinguishedPoint>;
@group(0) @binding(5) var<storage, read_write> dp_count:      atomic<u32>;

// Store DP — saves raw Jacobian (X, Z) + dist; CPU converts to affine X.
fn store_dp(k: Kangaroo, p: JacobianPoint, kangaroo_id: u32) {
    let idx = atomicAdd(&dp_count, 1u);
    if (idx < 65536u) {
        var dp: DistinguishedPoint;
        dp.x          = p.x;
        dp.z          = p.z;
        dp.dist       = k.dist;
        dp.ktype      = k.ktype;
        dp.kangaroo_id = kangaroo_id;
        dp_buffer[idx] = dp;
    }
}

// DP condition: use Jacobian X low 32-bit limb as a fast proxy filter.
// x[0] is least-significant limb (LE layout). Low bits are uniform.
fn is_dp(p: JacobianPoint) -> bool {
    return (p.x[0] & config.dp_mask_lo.x) == 0u;
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let kid = global_id.x;
    if (kid >= config.num_kangaroos) { return; }

    var k: Kangaroo = kangaroos[kid];
    if (k.is_active == 0u) { return; }

    var p: JacobianPoint;
    p.x = k.x;
    p.y = k.y;
    p.z = k.z;

    var dp_stored = false;

    for (var step = 0u; step < config.steps_per_call; step++) {
        // 1. Select jump index from current Jacobian X low byte
        let jump_idx = p.x[0] & 0xFFu;
        let jp = jump_points[jump_idx];
        let jd = jump_distances[jump_idx];

        // 2. Advance point
        p = jac_add_affine(p, jp.x, jp.y);

        // 3. Advance dist to match new point
        k.dist = scalar_add_256(k.dist, jd);

        // 4. Check DP on NEW point (only store one DP per dispatch to avoid
        //    the same kangaroo flooding the buffer after finding a DP)
        if (!dp_stored && is_dp(p)) {
            store_dp(k, p, kid);
            dp_stored = true;
        }
    }

    // Write back updated position and dist
    k.x = p.x;
    k.y = p.y;
    k.z = p.z;
    kangaroos[kid] = k;
}
