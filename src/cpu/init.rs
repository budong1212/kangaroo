//! Kangaroo initialization and jump table generation.
//!
//! # Algorithm contract
//!
//! **Tame** kangaroo `i`:
//!   - position = `(start + offset_i) * G`
//!   - dist     = `offset_i`
//!
//! **Wild** kangaroo `i`:
//!   - position = `pubkey + rand_i * G`  (each wild at a DIFFERENT point)
//!   - dist     = `rand_i`
//!
//! On collision: `key = start + tame_dist - wild_dist`

use crate::convert::{affine_to_gpu, scalar_be_to_limbs};
use crate::crypto::{Point, U256};
use crate::gpu::{GpuAffinePoint, GpuKangaroo};
use anyhow::Result;
use k256::{
    elliptic_curve::ops::{MulByGenerator, Reduce},
    ProjectivePoint, Scalar,
};
use k256::U256 as K256U256;
use rayon::prelude::*;

/// Generate jump table: 256 precomputed affine points with distances.
pub fn generate_jump_table(range_bits: u32) -> (Vec<GpuAffinePoint>, Vec<[u32; 8]>) {
    const TABLE_SIZE: usize = 256;
    let mean_exp = (range_bits / 2).max(8);

    let results: Vec<(GpuAffinePoint, [u32; 8])> = (0..TABLE_SIZE)
        .into_par_iter()
        .map(|i| {
            let mut h = 0x811c9dc5u32;
            h = (h ^ (i as u32)).wrapping_mul(0x01000193);
            h = (h ^ (h >> 16)).wrapping_mul(0x45d9f3b);

            let num_bytes = ((mean_exp + 7) / 8) as usize;
            let mut scalar_be = [0u8; 32];
            for b in 0..num_bytes {
                h = h.wrapping_mul(0x01000193) ^ (b as u32);
                scalar_be[32 - num_bytes + b] = (h >> 24) as u8;
            }
            let rem = mean_exp % 8;
            if rem != 0 {
                scalar_be[32 - num_bytes] &= (1u8 << rem) - 1;
            }
            if scalar_be.iter().all(|&x| x == 0) {
                scalar_be[31] = (i as u8).max(1);
            }

            let uint   = K256U256::from_be_slice(&scalar_be);
            let scalar = Scalar::reduce(uint);
            let point  = ProjectivePoint::mul_by_generator(&scalar).to_affine();
            (affine_to_gpu(&point), scalar_be_to_limbs(&scalar_be))
        })
        .collect();

    let (points, distances) = results.into_iter().unzip();
    tracing::debug!("Jump table: {} entries, mean_exp={}", TABLE_SIZE, mean_exp);
    (points, distances)
}

/// Initialize all kangaroos.
///
/// Tame: evenly spread across [start, start + 2^range_bits) with jitter.
/// Wild: each starts at `pubkey + rand_i * G` with `dist = rand_i`.
///       Every wild has a unique rand_i, so they start at distinct points
///       and their Jacobian X values differ from the very first step.
pub fn initialize_kangaroos(
    pubkey: &Point,
    start: &U256,
    range_bits: u32,
    num_kangaroos: u32,
) -> Result<Vec<GpuKangaroo>> {
    let half       = num_kangaroos / 2;
    let range_size: u128 = if range_bits >= 128 { u128::MAX } else { 1u128 << range_bits };
    let grid_delta = range_size / (half as u128).max(1);
    let pubkey_proj = ProjectivePoint::from(*pubkey);

    tracing::debug!(
        "init_kangaroos: n={} range_bits={} range_size={:#x} grid_delta={:#x}",
        num_kangaroos, range_bits, range_size, grid_delta
    );

    let kangaroos: Vec<GpuKangaroo> = (0..num_kangaroos)
        .into_par_iter()
        .map(|i| {
            if i < half {
                // Tame: position = (start + offset) * G, dist = offset
                let tame_idx = i as u128;
                let grid_pos = tame_idx * grid_delta;
                let jitter   = hash_u128(i, 0xCAFEBABE) % (grid_delta / 2 + 1);
                let offset   = (grid_pos + jitter) % range_size;
                let (point, dist) = tame_init(start, offset);
                let gp = affine_to_gpu(&point);
                GpuKangaroo {
                    x: gp.x, y: gp.y,
                    z: limbs_one(),
                    dist,
                    ktype: 0,
                    is_active: 1,
                    _padding: [0; 2],
                }
            } else {
                // Wild: position = pubkey + rand_i * G, dist = rand_i
                // rand_i is unique per wild kangaroo so each starts at a
                // different point and walks a different path from step 1.
                let wild_idx  = (i - half) as u128;
                let mean_exp  = (range_bits / 2).max(8);
                let rand_base = 1u128 << mean_exp.min(127);
                let rand_jit  = hash_u128(i, 0xDEADBEEF) % rand_base;
                // space wilds evenly then add jitter so they're well separated
                let rand = (wild_idx * rand_base + rand_jit) % range_size;
                let (point, dist) = wild_init(&pubkey_proj, rand);
                let gp = affine_to_gpu(&point);
                GpuKangaroo {
                    x: gp.x, y: gp.y,
                    z: limbs_one(),
                    dist,
                    ktype: 1,
                    is_active: 1,
                    _padding: [0; 2],
                }
            }
        })
        .collect();

    Ok(kangaroos)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Tame init: position = (start + offset) * G, dist = offset as LE limbs.
fn tame_init(start: &U256, offset: u128) -> (k256::AffinePoint, [u32; 8]) {
    let start_uint = K256U256::from_le_slice(start);
    let mut off_le = [0u8; 32];
    off_le[0..16].copy_from_slice(&offset.to_le_bytes());
    let off_uint   = K256U256::from_le_slice(&off_le);
    let scalar     = Scalar::reduce(start_uint.wrapping_add(&off_uint));
    let point      = ProjectivePoint::mul_by_generator(&scalar).to_affine();
    // dist = offset as BE->LE limbs
    let mut off_be = [0u8; 32];
    off_be[16..32].copy_from_slice(&offset.to_be_bytes());
    (point, scalar_be_to_limbs(&off_be))
}

/// Wild init: position = pubkey + rand * G, dist = rand as LE limbs.
fn wild_init(pubkey: &ProjectivePoint, rand: u128) -> (k256::AffinePoint, [u32; 8]) {
    let mut rand_be = [0u8; 32];
    rand_be[16..32].copy_from_slice(&rand.to_be_bytes());
    let rand_scalar = Scalar::reduce(K256U256::from_be_slice(&rand_be));
    let rand_point  = ProjectivePoint::mul_by_generator(&rand_scalar);
    let point       = (pubkey + &rand_point).to_affine();
    (point, scalar_be_to_limbs(&rand_be))
}

/// Deterministic u128 hash for jitter (FNV-1a mix).
fn hash_u128(index: u32, salt: u64) -> u128 {
    let mut h = 0xcbf29ce484222325u64;
    h ^= index as u64;
    h  = h.wrapping_mul(0x100000001b3);
    h ^= salt;
    h  = h.wrapping_mul(0x100000001b3);
    h ^= h >> 33;
    h  = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    let h2 = h.wrapping_mul(0xc4ceb9fe1a85ec53)
             ^ (index as u64).wrapping_mul(0x9e3779b97f4a7c15);
    ((h as u128) << 64) | (h2 as u128)
}

/// Z = 1 in Jacobian (affine point representation).
fn limbs_one() -> [u32; 8] { [1, 0, 0, 0, 0, 0, 0, 0] }
