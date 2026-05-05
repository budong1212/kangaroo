//! Distinguished Point hash table for tame/wild collision detection.
//!
//! # Layout contract
//!
//! The GPU shader stores **affine X** in `dp.x` (LE [u32;8]) and sets
//! `dp.z = all-zeros` as a sentinel.  No CPU-side Jacobian inversion needed.
//!
//! # Collision formula
//!
//! ```text
//! Tame: (start + tame_dist) * G = P
//! Wild: (key  + wild_dist)  * G = P
//! => key = start + tame_dist - wild_dist   (LE 256-bit)
//! ```

use crate::gpu::GpuDistinguishedPoint;
use dashmap::DashMap;

#[derive(Clone)]
struct StoredDP {
    affine_x_be: [u8; 32], // affine X in BE (for hash + logging)
    dist_le:     [u8; 32], // distance in LE bytes
    ktype:       u32,
}

pub struct DPTable {
    table:    DashMap<u64, Vec<StoredDP>>,
    start_le: [u8; 32],
}

impl DPTable {
    pub fn new(start: [u8; 32]) -> Self {
        Self { table: DashMap::new(), start_le: start }
    }

    /// Insert a DP and return the private key (BE, leading zeros stripped)
    /// if a tame<->wild collision is detected.
    pub fn insert_and_check(&self, dp: GpuDistinguishedPoint) -> Option<Vec<u8>> {
        // dp.x is affine X as LE [u32;8]  (shader already computed Z^{-2})
        let affine_x_be = le_limbs_to_be_bytes(&dp.x);
        let dist_le     = limbs_to_le_bytes(&dp.dist);

        // Use high 8 bytes of affine X as hash bucket key.
        let hash_key = u64::from_be_bytes(affine_x_be[0..8].try_into().unwrap());

        if let Some(mut bucket) = self.table.get_mut(&hash_key) {
            for existing in bucket.iter() {
                if existing.affine_x_be != affine_x_be {
                    continue; // hash bucket collision on a different point
                }
                if existing.ktype == dp.ktype {
                    // Same-type DP: two kangaroos of same type merged paths.
                    // Harmless — just ignore the duplicate.
                    tracing::debug!(
                        "dup {} DP x_hi={}",
                        if dp.ktype == 0 { "tame" } else { "wild" },
                        hex::encode(&affine_x_be[0..4])
                    );
                    return None;
                }

                // ---- Tame <-> Wild collision ----
                let (tame_dist, wild_dist) = if existing.ktype == 0 {
                    (&existing.dist_le, &dist_le)   // stored=tame, new=wild
                } else {
                    (&dist_le, &existing.dist_le)   // stored=wild, new=tame
                };

                // key = start + tame_dist - wild_dist  (LE 256-bit)
                let mut tmp    = [0u8; 32];
                let mut key_le = [0u8; 32];
                add_le(&self.start_le, tame_dist, &mut tmp);
                sub_le(&tmp, wild_dist, &mut key_le);

                // Convert LE -> BE and strip leading zeros
                let mut key_be = key_le;
                key_be.reverse();
                let first = key_be.iter().position(|&b| b != 0).unwrap_or(31);
                let key = key_be[first..].to_vec();

                tracing::info!(
                    "Collision! tame={} wild={} key=0x{}",
                    hex::encode(tame_dist),
                    hex::encode(wild_dist),
                    hex::encode(&key)
                );
                return Some(key);
            }
            bucket.push(StoredDP { affine_x_be, dist_le, ktype: dp.ktype });
        } else {
            self.table.insert(
                hash_key,
                vec![StoredDP { affine_x_be, dist_le, ktype: dp.ktype }],
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
// Byte helpers
// ---------------------------------------------------------------------------

/// GPU LE [u32;8] -> BE [u8;32]  (for hash key and logging)
fn le_limbs_to_be_bytes(limbs: &[u32; 8]) -> [u8; 32] {
    let mut b = [0u8; 32];
    for i in 0..8 {
        b[i * 4..(i + 1) * 4].copy_from_slice(&limbs[7 - i].to_be_bytes());
    }
    b
}

/// GPU LE [u32;8] -> LE [u8;32]  (for distance arithmetic)
fn limbs_to_le_bytes(limbs: &[u32; 8]) -> [u8; 32] {
    let mut b = [0u8; 32];
    for (i, &v) in limbs.iter().enumerate() {
        b[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
    }
    b
}

// ---------------------------------------------------------------------------
// LE 256-bit integer arithmetic
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
        let d  = a[i] as i16 - b[i] as i16 - borrow;
        out[i] = d.rem_euclid(256) as u8;
        borrow = if d < 0 { 1 } else { 0 };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn le32(v: u128) -> [u8; 32] {
        let mut b = [0u8; 32];
        b[0..16].copy_from_slice(&v.to_le_bytes());
        b
    }

    #[test]
    fn add_sub_roundtrip() {
        let a = le32(0xDEADBEEF_12345678);
        let b = le32(0x11111111_FFFFFFFF);
        let mut s = [0u8; 32];
        let mut r = [0u8; 32];
        add_le(&a, &b, &mut s);
        sub_le(&s, &b, &mut r);
        assert_eq!(r, a);
    }

    #[test]
    fn key_recovery() {
        // start=100  tame_dist=350  wild_dist=80  => key = 370
        let start  = le32(100);
        let tame_d = le32(350);
        let wild_d = le32(80);
        let mut tmp = [0u8; 32];
        let mut key = [0u8; 32];
        add_le(&start, &tame_d, &mut tmp);
        sub_le(&tmp, &wild_d, &mut key);
        key.reverse();
        let first = key.iter().position(|&b| b != 0).unwrap_or(31);
        let val = key[first..].iter().fold(0u128, |a, &b| (a << 8) | b as u128);
        assert_eq!(val, 370);
    }
}
