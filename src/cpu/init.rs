//! Kangaroo initialization and jump table generation.
//!
//! # Algorithm contract
//!
//! **Tame** kangaroo `i`:
//!   - position = `(start + offset_i) * G`
//!   - dist     = `offset_i`  (LE [u32;8] limbs)
//!
//! **Wild** kangaroo `i`:
//!   - position = `pubkey`  (no extra offset)
//!   - dist     = `0`
//!
//! On collision (same affine X, different ktype):
//!   `key = start + tame_dist - wild_dist`
//!
//! Wild kangaroos all start at the same point (pubkey), but their random
//! walks diverge immediately because the jump index is chosen from the
//! Jacobian X low byte, which differs per kangaroo after the first step.

use crate::convert::{affine_to_gpu, scalar_be_to_limbs};
use crate::crypto::{Point, U256};
use crate::gpu::{GpuAffinePoint, GpuKangaroo};
use anyhow::Result;
use k256::{ProjectivePoint, Scalar};
use k256::elliptic_curve::ops::{MulByGenerator, Reduce};
use k256::U256 as K256U256;
use rayon::prelude::*;

/// Generate jump table with 256 precomputed points.
/// Distances are uniformly distributed around 2^(range_bits/2).
pub fn generate_jump_table(range_bits: u32) -> (Vec<GpuAffinePoint>, Vec<[u32; 8]>) {
    const TABLE_SIZE: usize = 256;
    // Mean jump size ~ 2^(range_bits/2 - 1)  (half of sqrt(N))
    let mean_exp = (range_bits / 2).max(8);

    let results: Vec<(GpuAffinePoint, [u32; 8])> = (0..TABLE_SIZE)
        .into_par_iter()
        .map(|i| {
            // FNV-1a style hash for deterministic, well-distributed scalars
            let mut h = 0x811c9dc5u32;
            h = (h ^ (i as u32)).wrapping_mul(0x01000193);
            h = (h ^ (h >> 16)).wrapping_mul(0x45d9f3bu32);

            // Build 32-byte BE scalar with ~mean_exp significant bits
            let num_bytes = ((mean_exp + 7) / 8) as usize;
            let mut scalar_be = [0u8; 32];
            for b in 0..num_bytes {
                h = h.wrapping_mul(0x01000193) ^ (b as u32);
                scalar_be[32 - num_bytes + b] = (h >> 24) as u8;
            }
            // Mask top byte so value < 2^mean_exp
            let rem = mean_exp % 8;
            if rem != 0 {
                scalar_be[32 - num_bytes] &= (1u8 << rem) - 1;
            }
            // Ensure non-zero
            if scalar_be.iter().all(|&x| x == 0) {
                scalar_be[31] = (i as u8).max(1);
            }

            let uint  = K256U256::from_be_slice(&scalar_be);
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
/// Tame kangaroos are spread across [start, start + 2^range_bits).
/// Wild kangaroos all start at `pubkey` with dist = 0.
pub fn initialize_kangaroos(
    pubkey: &Point,
    start: &U256,
    range_bits: u32,
    num_kangaroos: u32,
) -> Result<Vec<GpuKangaroo>> {
    let half = num_kangaroos / 2;
    let range_size: u128 = if range_bits >= 128 { u128::MAX } else { 1u128 << range_bits };
    let grid_delta = range_size / (half as u128).max(1);

    tracing::debug!(
        "init_kangaroos: n={} range_bits={} range_size={:#x} grid_delta={:#x}",
        num_kangaroos, range_bits, range_size, grid_delta
    );

    // Compute affine pubkey for wild kangaroos
    let pubkey_affine = k256::ProjectivePoint::from(*pubkey).to_affine();
    let gpu_pubkey = affine_to_gpu(&pubkey_affine);

    let kangaroos: Vec<GpuKangaroo> = (0..num_kangaroos)
        .into_par_iter()
        .map(|i| {
            let is_tame = i < half;

            if is_tame {
                // Tame: spread evenly across range with small random jitter
                let tame_idx = i;  // 0..half
                let grid_pos = (tame_idx as u128) * grid_delta;
                let jitter   = hash_seed(i, 0xCAFEBABE) % (grid_delta / 2 + 1);
                let offset   = (grid_pos + jitter) % range_size;

                let (point, dist) = init_tame(start, offset);
                let gp = affine_to_gpu(&point);
                GpuKangaroo {
                    x: gp.x, y: gp.y,
                    z: [1, 0, 0, 0, 0, 0, 0, 0],
                    dist,
                    ktype: 0,
                    is_active: 1,
                    _padding: [0; 2],
                }
            } else {
                // Wild: start at pubkey, dist = 0
                GpuKangaroo {
                    x: gpu_pubkey.x, y: gpu_pubkey.y,
                    z: [1, 0, 0, 0, 0, 0, 0, 0],
                    dist: [0; 8],
                    ktype: 1,
                    is_active: 1,
                    _padding: [0; 2],
                }
            }
        })
        .collect();

    Ok(kangaroos)
}

/// FNV-1a based PRNG for deterministic jitter.
fn hash_seed(index: u32, salt: u64) -> u128 {
    let mut h = 0xcbf29ce484222325u64;
    h ^= index as u64;
    h = h.wrapping_mul(0x100000001b3);
    h ^= salt;
    h = h.wrapping_mul(0x100000001b3);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    let h2 = h.wrapping_mul(0xc4ceb9fe1a85ec53)
        ^ (index as u64).wrapping_mul(0x9e3779b97f4a7c15);
    ((h as u128) << 64) | (h2 as u128)
}

/// Tame kangaroo: position = (start + offset) * G, dist = offset.
fn init_tame(start: &U256, offset: u128) -> (k256::AffinePoint, [u32; 8]) {
    // start is LE [u8;32]
    let start_uint = K256U256::from_le_slice(start);

    // offset as LE [u8;32]
    let mut offset_le = [0u8; 32];
    offset_le[0..16].copy_from_slice(&offset.to_le_bytes());
    let offset_uint = K256U256::from_le_slice(&offset_le);

    let sum    = start_uint.wrapping_add(&offset_uint);
    let scalar = Scalar::reduce(sum);
    let point  = ProjectivePoint::mul_by_generator(&scalar).to_affine();

    // dist = offset stored as LE limbs
    // offset fits in 128 bits; place as BE then convert to LE limbs
    let mut offset_be = [0u8; 32];
    offset_be[16..32].copy_from_slice(&offset.to_be_bytes());
    let dist = scalar_be_to_limbs(&offset_be);

    (point, dist)
}
