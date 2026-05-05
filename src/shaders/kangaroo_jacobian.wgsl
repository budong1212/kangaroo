// =============================================================================
// Pollard's Kangaroo Algorithm - GPU Kernel (Pure Jacobian)
// =============================================================================
//
// ALGORITHM CONTRACT:
//   Tame kangaroo i:  position = (start + offset_i) * G,  dist = offset_i
//   Wild  kangaroo i: position = pubkey + rand_i * G,      dist = rand_i
//
//   After collision (same affine X, different ktype):
//     start + tame_dist = key + wild_dist
//     => key = start + tame_dist - wild_dist
//
// STEP ORDER:
//   1. Select jump index from Jacobian X low byte
//   2. Add jump point  (point advances)
//   3. Add jump dist   (dist advances to match new point)
//   4. Pre-filter: if Jacobian X low byte == 0  (~1/256 steps)
//   5.   Compute affine X = X_jac * Z_jac^{-2}  via fe_inv
//   6.   Check DP on affine X low bits  <-- MUST be affine X
//   7.   If DP: store affine_x in dp.x, fe_zero() in dp.z (sentinel)
//
// WHY affine X is required for DP:
//   Same elliptic-curve point P can have many Jacobian representations
//   (X*t^2, Y*t^3, Z*t) for any t.  Two kangaroos at the same P will have
//   different Jacobian X values if their Z values differ.  Using Jacobian X
//   as the DP trigger means tame and wild NEVER agree on the same DP slot
//   => collision is impossible.  Affine X is unique per point, so both
//   kangaroos agree and the CPU table detects the collision correctly.
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

// dp.x  = affine X (LE u32 limbs)  -- computed in shader via fe_inv
// dp.z  = all-zeros sentinel       -- signals CPU that x is already affine
// dp.dist / dp.ktype / dp.kangaroo_id as before
struct DistinguishedPoint {
    x: array<u32, 8>,
    z: array<u32, 8>,
    dist: array<u32, 8>,
    ktype: u32,
    kangaroo_id: u32,
}

@group(0) @binding(0) var<uniform>             config:         Config;
@group(0) @binding(1) var<storage, read>       jump_points:    array<AffinePoint, 256>;
@group(0) @binding(2) var<storage, read>       jump_distances: array<array<u32, 8>, 256>;
@group(0) @binding(3) var<storage, read_write> kangaroos:      array<Kangaroo>;
@group(0) @binding(4) var<storage, read_write> dp_buffer:      array<DistinguishedPoint>;
@group(0) @binding(5) var<storage, read_write> dp_count:       atomic<u32>;

fn store_dp(k: Kangaroo, affine_x: array<u32, 8>, kangaroo_id: u32) {
    let idx = atomicAdd(&dp_count, 1u);
    if (idx < 65536u) {
        var dp: DistinguishedPoint;
        dp.x           = affine_x;
        dp.z           = fe_zero();  // sentinel: x is already affine
        dp.dist        = k.dist;
        dp.ktype       = k.ktype;
        dp.kangaroo_id = kangaroo_id;
        dp_buffer[idx] = dp;
    }
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
        // 1. Select jump from Jacobian X low byte (uniform even in Jacobian coords)
        let jump_idx = p.x[0] & 0xFFu;
        let jp = jump_points[jump_idx];
        let jd = jump_distances[jump_idx];

        // 2. Advance point
        p = jac_add_affine(p, jp.x, jp.y);

        // 3. Advance dist (now in sync with new point)
        k.dist = scalar_add_256(k.dist, jd);

        // 4. Cheap pre-filter (~1/256 chance) before paying for fe_inv
        if (!dp_stored && (p.x[0] & 0xFFu) == 0u) {
            // 5. Compute affine X = X_jac / Z_jac^2
            let z_inv  = fe_inv(p.z);
            let z_inv2 = fe_mul(z_inv, z_inv);
            let ax     = fe_mul(p.x, z_inv2);

            // 6. True DP test on affine X low bits
            if ((ax[0] & config.dp_mask_lo.x) == 0u) {
                // 7. Store with affine X; z=zero sentinel so CPU skips inversion
                store_dp(k, ax, kid);
                dp_stored = true;
            }
        }
    }

    k.x = p.x;
    k.y = p.y;
    k.z = p.z;
    kangaroos[kid] = k;
}
