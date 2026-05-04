//! Distinguished Point hash table for collision detection.
//!
//! # Distance / byte-order contract
//!
//! * GPU `dist` field: LE [u32; 8] limbs  (limbs[0] = least significant 32 bits)
//! * `start` (U256): LE [u8; 32]           (byte 0 = least significant byte)
//!
//! `u32_array_to_le_bytes` converts GPU limbs → LE bytes so that the 256-bit
//! arithmetic in `compute_private_key` (which expects LE bytes) is correct.
//!
//! `verify_key` in crypto::mod expects a **big-endian trimmed** slice, so we
//! reverse the final LE result before returning.

use crate::gpu::GpuDistinguishedPoint;
use dashmap::DashMap;

#[derive(Clone)]
struct StoredDP {
    affine_x: [u8; 32], // 32-byte affine X, big-endian
    dist_le: [u8; 32],  // distance as LE bytes (matches GPU limb layout)
    ktype: u32,
}

/// Thread-safe DP table for collision detection.
pub struct DPTable {
    table: DashMap<u64, Vec<StoredDP>>,
    start_le: [u8; 32], // search range start, LE bytes (U256 format)
}

impl DPTable {
    pub fn new(start: [u8; 32]) -> Self {
        Self { table: DashMap::new(), start_le: start }
    }

    /// Insert a DP and check for a tame/wild collision.
    /// Returns the private key (big-endian, leading-zeros trimmed) if found.
    pub fn insert_and_check(&self, dp: GpuDistinguishedPoint) -> Option<Vec<u8>> {
        // Convert Jacobian (X, Z) → affine X (big-endian bytes)
        let affine_x = jacobian_to_affine_x(&dp.x, &dp.z)?;

        // Convert GPU dist limbs → LE bytes for arithmetic
        let dist_le = limbs_to_le_bytes(&dp.dist);

        // First 8 bytes of affine X (big-endian) as hash-table key
        let hash_key = u64::from_be_bytes(affine_x[0..8].try_into().unwrap());

        if let Some(mut bucket) = self.table.get_mut(&hash_key) {
            for existing in bucket.iter() {
                if existing.affine_x != affine_x {
                    continue; // hash collision, different point
                }
                if existing.ktype == dp.ktype {
                    // same-type collision — different path to same point, skip
                    tracing::debug!(
                        "Same-type DP collision ({}): x={}",
                        if dp.ktype == 0 { "tame" } else { "wild" },
                        hex::encode(&affine_x[..8])
                    );
                    return None;
                }
                // tame ↔ wild collision!
                // Tame: start + tame_dist = point * G
                // Wild: key  + wild_dist  = point * G
                // → key = start + tame_dist − wild_dist
                let key = compute_private_key(
                    &self.start_le,
                    &existing.dist_le,
                    &dist_le,
                    existing.ktype,
                );
                tracing::info!("Collision! key=0x{}", hex::encode(&key));
                return Some(key);
            }
            bucket.push(StoredDP { affine_x, dist_le, ktype: dp.ktype });
        } else {
            self.table.insert(hash_key, vec![StoredDP { affine_x, dist_le, ktype: dp.ktype }]);
        }
        None
    }

    pub fn total_dps(&self) -> usize {
        self.table.iter().map(|e| e.value().len()).sum()
    }

    pub fn count_by_type(&self) -> (usize, usize) {
        let mut tame = 0usize;
        let mut wild = 0usize;
        for entry in &self.table {
            for dp in entry.value() {
                if dp.ktype == 0 { tame += 1; } else { wild += 1; }
            }
        }
        (tame, wild)
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize { self.table.len() }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool { self.table.is_empty() }
}

// ---------------------------------------------------------------------------
// Jacobian → affine X conversion using k256
// ---------------------------------------------------------------------------

fn jacobian_to_affine_x(x_jac: &[u32; 8], z_jac: &[u32; 8]) -> Option<[u8; 32]> {
    use k256::FieldElement;

    if z_jac.iter().all(|&v| v == 0) {
        return None; // point at infinity
    }

    // GPU limbs are LE [u32;8]; convert to BE [u8;32] for k256
    let x_be = limbs_to_be_bytes(x_jac);
    let z_be = limbs_to_be_bytes(z_jac);

    let x_fe = FieldElement::from_bytes(&x_be.into());
    let z_fe = FieldElement::from_bytes(&z_be.into());

    if x_fe.is_none().into() || z_fe.is_none().into() {
        return None;
    }

    let x_fe = x_fe.unwrap();
    let z_fe = z_fe.unwrap();

    let z_inv = z_fe.invert();
    if z_inv.is_none().into() {
        return None;
    }

    // affine_x = X * Z^{-2}
    let z_inv2 = z_inv.unwrap().square();
    let ax = x_fe * z_inv2;

    let mut result = [0u8; 32];
    result.copy_from_slice(&ax.to_bytes());
    Some(result) // big-endian
}

// ---------------------------------------------------------------------------
// Private key recovery
// ---------------------------------------------------------------------------

/// Compute private key from a tame/wild collision.
///
/// All inputs are LE [u8; 32].
/// Returns big-endian bytes with leading zeros trimmed (as expected by verify_key).
fn compute_private_key(
    start_le: &[u8; 32],
    tame_dist_le: &[u8; 32],
    wild_dist_le: &[u8; 32],
    existing_ktype: u32,
) -> Vec<u8> {
    // Determine which dist belongs to tame vs wild
    let (td, wd) = if existing_ktype == 0 {
        (tame_dist_le, wild_dist_le)  // existing=tame, new=wild
    } else {
        (wild_dist_le, tame_dist_le)  // existing=wild, new=tame (swap)
    };

    // diff = tame_dist - wild_dist  (LE 256-bit subtraction)
    let mut diff = [0u8; 32];
    sub_le(td, wd, &mut diff);

    // key = start + diff  (LE 256-bit addition)
    let mut key_le = [0u8; 32];
    add_le(start_le, &diff, &mut key_le);

    // Convert LE → BE and trim leading zeros
    let mut key_be = key_le;
    key_be.reverse();
    let first = key_be.iter().position(|&b| b != 0).unwrap_or(31);
    key_be[first..].to_vec()
}

// ---------------------------------------------------------------------------
// Byte conversion helpers
// ---------------------------------------------------------------------------

/// GPU LE limbs [u32;8] → LE bytes [u8;32]
fn limbs_to_le_bytes(limbs: &[u32; 8]) -> [u8; 32] {
    let mut b = [0u8; 32];
    for (i, &v) in limbs.iter().enumerate() {
        b[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
    }
    b
}

/// GPU LE limbs [u32;8] → BE bytes [u8;32]  (for k256 FieldElement)
fn limbs_to_be_bytes(limbs: &[u32; 8]) -> [u8; 32] {
    let mut b = [0u8; 32];
    for i in 0..8 {
        b[i * 4..(i + 1) * 4].copy_from_slice(&limbs[7 - i].to_be_bytes());
    }
    b
}

// ---------------------------------------------------------------------------
// 256-bit arithmetic on LE byte arrays
// ---------------------------------------------------------------------------

fn add_le(a: &[u8; 32], b: &[u8; 32], out: &mut [u8; 32]) {
    let mut carry = 0u16;
    for i in 0..32 {
        let s = a[i] as u16 + b[i] as u16 + carry;
        out[i] = s as u8;
        carry = s >> 8;
    }
}

fn sub_le(a: &[u8; 32], b: &[u8; 32], out: &mut [u8; 32]) {
    let mut borrow = 0i16;
    for i in 0..32 {
        let d = a[i] as i16 - b[i] as i16 - borrow;
        out[i] = d.rem_euclid(256) as u8;
        borrow = if d < 0 { 1 } else { 0 };
    }
}
