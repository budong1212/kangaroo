//! Distinguished Point hash table for collision detection.
//!
//! # Byte-order contract
//!
//! | Value      | Format stored here         |
//! |------------|----------------------------|
//! | affine X   | BE [u8;32] (from k256)     |
//! | dist       | LE [u8;32] (from GPU limbs)|
//! | start      | LE [u8;32] (U256 type)     |
//!
//! # Collision formula
//!
//! Tame: `(start + tame_dist) * G = P`
//! Wild: `key  * G + wild_dist * G = P`  (wild starts at pubkey, dist=0)
//!
//! Therefore: `key = start + tame_dist - wild_dist`
//!
//! All arithmetic is LE 256-bit. Final result is reversed to BE and
//! leading zeros trimmed before returning (matches `verify_key` expectation).

use crate::gpu::GpuDistinguishedPoint;
use dashmap::DashMap;

#[derive(Clone)]
struct StoredDP {
    affine_x: [u8; 32], // BE
    dist_le:  [u8; 32], // LE
    ktype:    u32,
}

pub struct DPTable {
    table:    DashMap<u64, Vec<StoredDP>>,
    start_le: [u8; 32], // LE
}

impl DPTable {
    pub fn new(start: [u8; 32]) -> Self {
        Self { table: DashMap::new(), start_le: start }
    }

    /// Insert DP and return private key (BE trimmed) if collision found.
    pub fn insert_and_check(&self, dp: GpuDistinguishedPoint) -> Option<Vec<u8>> {
        // Convert Jacobian (X,Z) -> affine X (BE)
        let affine_x = jacobian_to_affine_x(&dp.x, &dp.z)?;

        // GPU dist limbs (LE [u32;8]) -> LE bytes
        let dist_le = limbs_to_le_bytes(&dp.dist);

        // Hash key: first 8 BE bytes of affine X
        let hash_key = u64::from_be_bytes(affine_x[0..8].try_into().unwrap());

        if let Some(mut bucket) = self.table.get_mut(&hash_key) {
            for existing in bucket.iter() {
                if existing.affine_x != affine_x {
                    continue; // hash bucket collision on different point
                }
                if existing.ktype == dp.ktype {
                    tracing::debug!(
                        "same-type DP ({}): x={}",
                        if dp.ktype == 0 { "tame" } else { "wild" },
                        hex::encode(&affine_x[..8])
                    );
                    return None;
                }

                // Tame <-> Wild collision found!
                // Identify which dist is tame and which is wild.
                let (tame_dist, wild_dist) = if existing.ktype == 0 {
                    (&existing.dist_le, &dist_le)   // existing=tame, new=wild
                } else {
                    (&dist_le, &existing.dist_le)   // existing=wild, new=tame
                };

                // key = start + tame_dist - wild_dist  (LE 256-bit)
                let mut tmp = [0u8; 32];
                add_le(&self.start_le, tame_dist, &mut tmp);
                let mut key_le = [0u8; 32];
                sub_le(&tmp, wild_dist, &mut key_le);

                // LE -> BE, trim leading zeros
                let mut key_be = key_le;
                key_be.reverse();
                let first = key_be.iter().position(|&b| b != 0).unwrap_or(31);
                let key = key_be[first..].to_vec();

                tracing::info!("Collision! key=0x{}", hex::encode(&key));
                return Some(key);
            }
            bucket.push(StoredDP { affine_x, dist_le, ktype: dp.ktype });
        } else {
            self.table.insert(
                hash_key,
                vec![StoredDP { affine_x, dist_le, ktype: dp.ktype }],
            );
        }
        None
    }

    pub fn total_dps(&self) -> usize {
        self.table.iter().map(|e| e.value().len()).sum()
    }

    pub fn count_by_type(&self) -> (usize, usize) {
        let (mut tame, mut wild) = (0usize, 0usize);
        for entry in &self.table {
            for dp in entry.value() {
                if dp.ktype == 0 { tame += 1; } else { wild += 1; }
            }
        }
        (tame, wild)
    }

    #[allow(dead_code)] pub fn len(&self)      -> usize { self.table.len() }
    #[allow(dead_code)] pub fn is_empty(&self) -> bool  { self.table.is_empty() }
}

// ---------------------------------------------------------------------------
// Jacobian -> affine X  (k256 field arithmetic)
// ---------------------------------------------------------------------------

fn jacobian_to_affine_x(x_jac: &[u32; 8], z_jac: &[u32; 8]) -> Option<[u8; 32]> {
    use k256::FieldElement;

    if z_jac.iter().all(|&v| v == 0) {
        return None; // point at infinity
    }

    // GPU: LE [u32;8]  ->  BE [u8;32] for k256
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
    // affine_x = X_jac * Z_jac^{-2}
    let z_inv2 = z_inv.unwrap().square();
    let ax = x_fe * z_inv2;

    let mut out = [0u8; 32];
    out.copy_from_slice(&ax.to_bytes());
    Some(out) // BE
}

// ---------------------------------------------------------------------------
// Byte helpers
// ---------------------------------------------------------------------------

/// GPU LE limbs [u32;8] -> LE bytes [u8;32]
fn limbs_to_le_bytes(limbs: &[u32; 8]) -> [u8; 32] {
    let mut b = [0u8; 32];
    for (i, &v) in limbs.iter().enumerate() {
        b[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
    }
    b
}

/// GPU LE limbs [u32;8] -> BE bytes [u8;32]  (for k256)
fn limbs_to_be_bytes(limbs: &[u32; 8]) -> [u8; 32] {
    let mut b = [0u8; 32];
    for i in 0..8 {
        b[i * 4..(i + 1) * 4].copy_from_slice(&limbs[7 - i].to_be_bytes());
    }
    b
}

// ---------------------------------------------------------------------------
// LE 256-bit arithmetic
// ---------------------------------------------------------------------------

fn add_le(a: &[u8; 32], b: &[u8; 32], out: &mut [u8; 32]) {
    let mut carry = 0u16;
    for i in 0..32 {
        let s = a[i] as u16 + b[i] as u16 + carry;
        out[i] = s as u8;
        carry  = s >> 8;
    }
}

fn sub_le(a: &[u8; 32], b: &[u8; 32], out: &mut [u8; 32]) {
    let mut borrow = 0i16;
    for i in 0..32 {
        let d = a[i] as i16 - b[i] as i16 - borrow;
        out[i]  = d.rem_euclid(256) as u8;
        borrow  = if d < 0 { 1 } else { 0 };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn u128_to_le32(v: u128) -> [u8; 32] {
        let mut b = [0u8; 32];
        b[0..16].copy_from_slice(&v.to_le_bytes());
        b
    }

    #[test]
    fn add_le_basic() {
        let a = u128_to_le32(100);
        let b = u128_to_le32(200);
        let mut c = [0u8; 32];
        add_le(&a, &b, &mut c);
        assert_eq!(c[0..2], [44, 1]); // 300 = 0x012c
    }

    #[test]
    fn sub_le_basic() {
        let a = u128_to_le32(500);
        let b = u128_to_le32(200);
        let mut c = [0u8; 32];
        sub_le(&a, &b, &mut c);
        // 300 LE
        let expected = u128_to_le32(300);
        assert_eq!(c, expected);
    }

    #[test]
    fn key_recovery_formula() {
        // start=100, tame_dist=250, wild_dist=80  => key=270
        let start    = u128_to_le32(100);
        let tame_d   = u128_to_le32(250);
        let wild_d   = u128_to_le32(80);

        let mut tmp   = [0u8; 32];
        let mut key_le = [0u8; 32];
        add_le(&start, &tame_d, &mut tmp);
        sub_le(&tmp, &wild_d, &mut key_le);

        let mut key_be = key_le;
        key_be.reverse();
        let first = key_be.iter().position(|&b| b != 0).unwrap_or(31);
        let key_bytes = &key_be[first..];
        let key_val = key_bytes.iter().fold(0u128, |acc, &b| (acc << 8) | b as u128);
        assert_eq!(key_val, 270);
    }
}
