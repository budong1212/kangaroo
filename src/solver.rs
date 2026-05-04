//! GPU-accelerated Kangaroo solver

use crate::cpu::init::{generate_jump_table, initialize_kangaroos};
use crate::cpu::DPTable;
use crate::crypto::{Point, U256};
use crate::gpu::{
    GpuBuffers, GpuConfig, GpuContext, GpuDistinguishedPoint, GpuKangaroo, KangarooPipeline,
};
use crate::math::create_dp_mask;
use anyhow::Result;
use std::time::Instant;
use tracing::info;

const MAX_DISTINGUISHED_POINTS: u32 = 65_536;
/// Stay well under Windows TDR (default 2s). 20ms gives a large safety margin.
const TARGET_DISPATCH_MS: u128 = 20;

pub struct KangarooSolver {
    ctx: GpuContext,
    pipeline: KangarooPipeline,
    buffers: GpuBuffers,
    dp_table: DPTable,
    total_ops: u64,
    num_kangaroos: u32,
    steps_per_call: u32,
    speed_timer: Instant,
    speed_ops_snapshot: u64,
    pub current_ops_per_sec: f64,
}

impl KangarooSolver {
    pub fn new(
        ctx: GpuContext,
        pubkey: Point,
        start: U256,
        range_bits: u32,
        dp_bits: u32,
        num_kangaroos: u32,
    ) -> Result<Self> {
        info!("Creating pipeline...");
        let pipeline = KangarooPipeline::new(&ctx)?;
        info!("Pipeline created");

        info!("Generating jump table...");
        let (jump_points, jump_distances) = generate_jump_table(range_bits);
        info!("Jump table generated: {} entries", jump_points.len());
        for (i, dist) in jump_distances.iter().enumerate().take(4) {
            info!("Jump dist[{}] = 0x{:08x}_{:08x}", i, dist[1], dist[0]);
        }

        let dp_mask = create_dp_mask(dp_bits);
        let steps_per_call = Self::select_steps_per_call(
            ctx.optimal_steps_per_call(),
            num_kangaroos,
            dp_bits,
            MAX_DISTINGUISHED_POINTS,
        );

        let config = GpuConfig {
            dp_mask_lo: [dp_mask[0], dp_mask[1], dp_mask[2], dp_mask[3]],
            dp_mask_hi: [dp_mask[4], dp_mask[5], dp_mask[6], dp_mask[7]],
            num_kangaroos,
            steps_per_call,
            jump_table_size: 256,
            _padding: 0,
        };
        info!("Config: kangaroos={} steps_per_call={} dp_bits={}", num_kangaroos, steps_per_call, dp_bits);

        info!("Creating GPU buffers...");
        let buffers = GpuBuffers::new(
            &ctx, &pipeline, &config, &jump_points, &jump_distances,
            num_kangaroos, MAX_DISTINGUISHED_POINTS,
        )?;

        info!("Initializing kangaroos...");
        let kangaroos = initialize_kangaroos(&pubkey, &start, range_bits, num_kangaroos)?;
        upload_kangaroos(&ctx, &buffers, &kangaroos)?;
        info!("Kangaroos uploaded");

        let mut solver = Self {
            ctx,
            pipeline,
            buffers,
            dp_table: DPTable::new(start),
            total_ops: 0,
            num_kangaroos,
            steps_per_call,
            speed_timer: Instant::now(),
            speed_ops_snapshot: 0,
            current_ops_per_sec: 0.0,
        };

        solver.calibrate(dp_bits);

        // Write final calibrated config
        let final_config = GpuConfig {
            dp_mask_lo: [dp_mask[0], dp_mask[1], dp_mask[2], dp_mask[3]],
            dp_mask_hi: [dp_mask[4], dp_mask[5], dp_mask[6], dp_mask[7]],
            num_kangaroos,
            steps_per_call: solver.steps_per_call,
            jump_table_size: 256,
            _padding: 0,
        };
        solver.ctx.queue.write_buffer(
            &solver.buffers.config_buffer, 0, bytemuck::bytes_of(&final_config),
        );
        solver.reset_dp_count()?;
        Ok(solver)
    }

    fn select_steps_per_call(
        optimal_steps: u32,
        num_kangaroos: u32,
        dp_bits: u32,
        max_dps: u32,
    ) -> u32 {
        if num_kangaroos == 0 || optimal_steps == 0 {
            return 1;
        }
        // Cap so that one dispatch can't overflow the DP buffer
        let budget = ((max_dps as u128) * 9 / 10).max(1);
        let dp_spacing = 1u128 << dp_bits;
        let allowed = (budget * dp_spacing / num_kangaroos as u128).max(1);
        let capped = allowed.min(u32::MAX as u128) as u32;
        capped.min(optimal_steps)
    }

    pub fn step(&mut self) -> Result<Option<Vec<u8>>> {
        // --- Dispatch GPU compute ---
        let mut enc = self.ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Kangaroo Encoder") }
        );
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Kangaroo Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline.pipeline);
            pass.set_bind_group(0, &self.buffers.bind_group, &[]);
            pass.dispatch_workgroups(self.num_kangaroos.div_ceil(64), 1, 1);
        }
        // Copy dp_count to staging so we can read it on CPU
        enc.copy_buffer_to_buffer(&self.buffers.dp_count_buffer, 0, &self.buffers.staging_buffer, 0, 4);
        self.ctx.queue.submit(Some(enc.finish()));

        self.total_ops += self.num_kangaroos as u64 * self.steps_per_call as u64;

        // Update speed counter
        let elapsed = self.speed_timer.elapsed();
        if elapsed.as_millis() >= 1000 {
            let delta = self.total_ops - self.speed_ops_snapshot;
            self.current_ops_per_sec = delta as f64 / elapsed.as_secs_f64();
            self.speed_ops_snapshot = self.total_ops;
            self.speed_timer = Instant::now();
        }

        // Periodic log
        let ops_per_dispatch = self.num_kangaroos as u64 * self.steps_per_call as u64;
        if self.total_ops % (ops_per_dispatch * 20) < ops_per_dispatch {
            let (tame, wild) = self.dp_table.count_by_type();
            tracing::info!(
                "Ops: {}M | Speed: {:.2} Mop/s | DPs: {} ({} tame, {} wild)",
                self.total_ops / 1_000_000,
                self.current_ops_per_sec / 1_000_000.0,
                self.dp_table.total_dps(), tame, wild
            );
        }

        // --- Read DP count ---
        let dp_count = self.read_dp_count()?;
        if dp_count == 0 {
            return Ok(None);
        }

        // --- Read DPs from GPU ---
        let actual = (dp_count as usize).min(MAX_DISTINGUISHED_POINTS as usize);
        let dp_size = std::mem::size_of::<GpuDistinguishedPoint>();
        let copy_bytes = (actual * dp_size) as u64;

        let mut enc2 = self.ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("DP Readback") }
        );
        enc2.copy_buffer_to_buffer(&self.buffers.dp_buffer, 0, &self.buffers.staging_buffer, 4, copy_bytes);
        self.ctx.queue.submit(Some(enc2.finish()));

        let dps = self.read_dps(actual as u32)?;

        // ALWAYS reset dp_count before processing — avoids double-counting on restart
        self.reset_dp_count()?;

        // --- Check for collisions ---
        for dp in dps {
            if let Some(key) = self.dp_table.insert_and_check(dp) {
                return Ok(Some(key));
            }
        }

        Ok(None)
    }

    pub fn total_operations(&self) -> u64 {
        self.total_ops
    }

    fn read_dp_count(&self) -> Result<u32> {
        let slice = self.buffers.staging_buffer.slice(0..4);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;
        let data = slice.get_mapped_range();
        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        self.buffers.staging_buffer.unmap();
        Ok(count)
    }

    fn read_dps(&self, count: u32) -> Result<Vec<GpuDistinguishedPoint>> {
        let dp_size = std::mem::size_of::<GpuDistinguishedPoint>();
        let total = 4 + count as usize * dp_size;
        let slice = self.buffers.staging_buffer.slice(0..total as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;
        let data = slice.get_mapped_range();
        let dps: Vec<GpuDistinguishedPoint> = data[4..]
            .chunks_exact(dp_size)
            .take(count as usize)
            .map(|c| *bytemuck::from_bytes::<GpuDistinguishedPoint>(c))
            .collect();
        drop(data);
        self.buffers.staging_buffer.unmap();
        Ok(dps)
    }

    fn reset_dp_count(&self) -> Result<()> {
        self.ctx.queue.write_buffer(&self.buffers.dp_count_buffer, 0, &[0u8; 4]);
        Ok(())
    }

    /// Calibrate steps_per_call by timing actual GPU dispatches.
    /// Stays well under Windows TDR (target < 20ms per dispatch).
    fn calibrate(&mut self, dp_bits: u32) {
        let candidates = [16u32, 32, 64, 128, 256, 512, 1024, 2048, 4096];
        let mut best = candidates[0];
        info!("Calibrating GPU (target <{}ms/dispatch)...", TARGET_DISPATCH_MS);

        for &steps in &candidates {
            let capped = Self::select_steps_per_call(
                steps, self.num_kangaroos, dp_bits, MAX_DISTINGUISHED_POINTS,
            );
            if capped < steps { break; }

            let cfg = GpuConfig {
                dp_mask_lo: [0; 4],
                dp_mask_hi: [0; 4],
                num_kangaroos: self.num_kangaroos,
                steps_per_call: steps,
                jump_table_size: 256,
                _padding: 0,
            };
            self.ctx.queue.write_buffer(&self.buffers.config_buffer, 0, bytemuck::bytes_of(&cfg));

            // Warm-up then measure
            self.dispatch_once();
            let t = Instant::now();
            self.dispatch_once();
            let ms = t.elapsed().as_millis();

            info!("  steps={}: {}ms", steps, ms);
            if ms <= TARGET_DISPATCH_MS {
                best = steps;
            } else {
                break;
            }
        }

        self.steps_per_call = best;
        let throughput = self.num_kangaroos as f64 * best as f64 / 1_000_000.0;
        info!("Calibrated: steps_per_call={} ({:.2}M ops/dispatch)", best, throughput);
    }

    fn dispatch_once(&self) {
        let mut enc = self.ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Calibration") }
        );
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Calibration Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline.pipeline);
            pass.set_bind_group(0, &self.buffers.bind_group, &[]);
            pass.dispatch_workgroups(self.num_kangaroos.div_ceil(64), 1, 1);
        }
        self.ctx.queue.submit(Some(enc.finish()));
        self.ctx.device.poll(wgpu::Maintain::Wait);
    }
}

fn upload_kangaroos(ctx: &GpuContext, buffers: &GpuBuffers, kangaroos: &[GpuKangaroo]) -> Result<()> {
    ctx.queue.write_buffer(&buffers.kangaroos_buffer, 0, bytemuck::cast_slice(kangaroos));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{KangarooSolver, MAX_DISTINGUISHED_POINTS};

    #[test]
    fn caps_steps_when_dp_buffer_would_overflow() {
        let steps = KangarooSolver::select_steps_per_call(4_096, 16_384, 8, MAX_DISTINGUISHED_POINTS);
        assert!(steps <= 4_096);
        assert!(steps > 0);
    }

    #[test]
    fn keeps_optimal_when_within_budget() {
        let steps = KangarooSolver::select_steps_per_call(4_096, 4_096, 16, MAX_DISTINGUISHED_POINTS);
        assert_eq!(steps, 4_096);
    }
}
