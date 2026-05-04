//! Kangaroo initialization and jump table generation.

use crate::convert::{affine_to_gpu, scalar_be_to_limbs};
use crate::crypto::{Point, U256};
use crate::gpu::{GpuAffinePoint, GpuKangaroo};
use crate::math::negate_256_be;
use anyhow::Result;
use k256::{ProjectivePoint, Scalar};
use k256::elliptic_curve::ops::{MulByGenerator, Reduce};
use k256::U256 as K256U256;
use rayon::prelude::*;

/// Generate jump table with 256 precomputed points.
/// Jump distances are uniformly distributed in [2^(mean_exp-1), 2^mean_exp).
pub fn generate_jump_table(range_bits: u32) -> (Vec<GpuAffinePoint>, Vec<[u32; 8]>) {
    const TABLE_SIZE: usize = 256;
    let mean_exp = (range_bits / 2).max(8);

    let results: Vec<(GpuAffinePoint, [u32; 8])> = (0..TABLE_SIZE)
        .into_par_iter()
        .map(|i| {
            // FNV-1a hash for deterministic, well-distributed scalars
            let mut h = 0x811c9dc5u32;
            h = (h ^ (i as u32)).wrapping_mul(0x01000193);
            h = (h ^ (h >> 16)).wrapping_mul(0x45d9f3b);

            // Build a 32-byte big-endian scalar of ~mean_exp bits
            let num_bytes = ((mean_exp + 7) / 8) as usize;
            let mut scalar_bytes = [0u8; 32];

            for b in 0..num_bytes {
                h = h.wrapping_mul(0x01000193) ^ (b as u32);
                scalar_bytes[32 - num_bytes + b] = (h >> 24) as u8;
            }

            // Mask the most-significant byte so value < 2^mean_exp
            let rem = mean_exp % 8;
            if rem != 0 {
                scalar_bytes[32 - num_bytes] &= (1u8 << rem) - 1;
            }

            // Ensure non-zero
            if scalar_bytes.iter().all(|&x| x == 0) {
                scalar_bytes[31] = (i as u8).max(1);
            }

            let scalar_uint = K256U256::from_be_slice(&scalar_bytes);
            let scalar = Scalar::reduce(scalar_uint);
            let point = ProjectivePoint::mul_by_generator(&scalar).to_affine();

            (affine_to_gpu(&point), scalar_be_to_limbs(&scalar_bytes))
        })
        .collect();

    let (points, distances) = results.into_iter().unzip();

    tracing::debug!("Jump table generated: {} entries, mean_exp={}", TABLE_SIZE, mean_exp);
    (points, distances)
}

/// Initialize kangaroo positions.
/// Half tame (start at known scalar multiples), half wild (start near pubkey).
pub fn initialize_kangaroos(
    pubkey: &Point,
    start: &U256,
    range_bits: u32,
    num_kangaroos: u32,
) -> Result<Vec<GpuKangaroo>> {
    let half = num_kangaroos / 2;
    let range_size: u128 = if range_bits >= 128 { u128::MAX } else { 1u128 << range_bits };
    let range_middle: u128 = if range_bits >= 128 { u128::MAX / 2 } else { 1u128 << (range_bits - 1) };
    let grid_delta = range_size / (num_kangaroos as u128).max(1);

    tracing::debug!(
        "Kangaroo init: n={} range_bits={} range_size={:#x} grid_delta={:#x}",
        num_kangaroos, range_bits, range_size, grid_delta
    );

    let kangaroos: Vec<GpuKangaroo> = (0..num_kangaroos)
        .into_par_iter()
        .map(|i| {
            let is_tame = i < half;
            let grid_pos = (i as u128) * grid_delta;
            let jitter = hash_seed(i, 0xCAFEBABE) % (grid_delta / 2 + 1);
            let offset = (grid_pos + jitter) % range_size;

            let (point, dist) = if is_tame {
                init_tame_kangaroo_at_offset(start, offset)
            } else {
                init_wild_kangaroo_at_offset(pubkey, offset, range_middle)
            };

            let gpu_point = affine_to_gpu(&point);
            GpuKangaroo {
                x: gpu_point.x,
                y: gpu_point.y,
                z: [1, 0, 0, 0, 0, 0, 0, 0],
                dist,
                ktype: if is_tame { 0 } else { 1 },
                is_active: 1,
                _padding: [0; 2],
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

/// Initialize a tame kangaroo.
/// Position  = (start + offset) * G
/// Distance  = offset  (stored as LE [u32;8] limbs, same format as GPU dist)
fn init_tame_kangaroo_at_offset(start: &U256, offset: u128) -> (k256::AffinePoint, [u32; 8]) {
    // start is LE [u8;32]. Convert to K256U256.
    let start_uint = K256U256::from_le_slice(start);

    // offset as LE [u8;32]
    let mut offset_le = [0u8; 32];
    offset_le[0..16].copy_from_slice(&offset.to_le_bytes());
    let offset_uint = K256U256::from_le_slice(&offset_le);

    let sum = start_uint.wrapping_add(&offset_uint);
    let scalar = Scalar::reduce(sum);
    let point = ProjectivePoint::mul_by_generator(&scalar).to_affine();

    // dist = offset in LE limbs (matches GPU format)
    // Convert offset to BE bytes first, then to LE limbs via scalar_be_to_limbs
    let mut offset_be = [0u8; 32];
    // offset fits in 128 bits; store in big-endian layout
    offset_be[16..32].copy_from_slice(&offset.to_be_bytes());
    let dist = scalar_be_to_limbs(&offset_be);

    (point, dist)
}

/// Initialize a wild kangaroo.
/// Position  = pubkey + centered_offset * G
/// Distance  = centered_offset (stored as LE [u32;8] limbs, signed via 2s-complement)
fn init_wild_kangaroo_at_offset(
    pubkey: &Point,
    raw_offset: u128,
    range_middle: u128,
) -> (k256::AffinePoint, [u32; 8]) {
    let centered: i128 = raw_offset as i128 - range_middle as i128;

    if centered >= 0 {
        let abs = centered as u128;
        let mut be = [0u8; 32];
        be[16..32].copy_from_slice(&abs.to_be_bytes());
        let scalar = Scalar::reduce(K256U256::from_be_slice(&be));
        let offset_point = ProjectivePoint::mul_by_generator(&scalar);
        let wild_point = (*pubkey + offset_point).to_affine();
        let dist = scalar_be_to_limbs(&be);
        (wild_point, dist)
    } else {
        let abs = (-centered) as u128;
        let mut be = [0u8; 32];
        be[16..32].copy_from_slice(&abs.to_be_bytes());
        let scalar = Scalar::reduce(K256U256::from_be_slice(&be));
        let offset_point = ProjectivePoint::mul_by_generator(&scalar);
        let wild_point = (*pubkey - offset_point).to_affine();
        // Negate offset so dist is negative (two's complement BE, then to LE limbs)
        let neg_be = negate_256_be(&be);
        let dist = scalar_be_to_limbs(&neg_be);
        (wild_point, dist)
    }
}
