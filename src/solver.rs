//! GPU-accelerated Kangaroo solver
//!
//! Coordinates GPU compute with CPU collision detection.

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
/// Target dispatch time in milliseconds (stay under TDR threshold)
const TARGET_DISPATCH_MS: u128 = 50;

/// Shared resources for batch mode (pipeline created once, reused)
#[allow(dead_code)]
pub struct SharedResources {
    pub ctx: GpuContext,
    pub pipeline: KangarooPipeline,
}

#[allow(dead_code)]
impl SharedResources {
    pub fn new(ctx: GpuContext) -> Result<Self> {
        let pipeline = KangarooPipeline::new(&ctx)?;
        Ok(Self { ctx, pipeline })
    }
}

/// Main Kangaroo solver
pub struct KangarooSolver {
    ctx: GpuContext,
    pipeline: KangarooPipeline,
    buffers: GpuBuffers,
    dp_table: DPTable,
    total_ops: u64,
    num_kangaroos: u32,
    #[allow(dead_code)]
    steps_per_call: u32,
    /// Track time for speed reporting
    speed_timer: Instant,
    speed_ops_snapshot: u64,
    /// Current measured speed in ops/s (updated every ~1s)
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
        Self::new_internal(ctx, pubkey, start, range_bits, dp_bits, num_kangaroos, true)
    }

    #[allow(dead_code)]
    pub fn new_with_context(
        ctx: &GpuContext,
        pubkey: Point,
        start: U256,
        range_bits: u32,
        dp_bits: u32,
        num_kangaroos: u32,
    ) -> Result<Self> {
        Self::new_internal(ctx.clone(), pubkey, start, range_bits, dp_bits, num_kangaroos, false)
    }

    #[allow(dead_code)]
    pub fn new_with_shared(
        shared: &SharedResources,
        pubkey: Point,
        start: U256,
        range_bits: u32,
        dp_bits: u32,
        num_kangaroos: u32,
    ) -> Result<Self> {
        Self::new_with_pipeline(&shared.ctx, &shared.pipeline, pubkey, start, range_bits, dp_bits, num_kangaroos)
    }

    fn select_steps_per_call(optimal_steps: u32, num_kangaroos: u32, dp_bits: u32, max_dps: u32) -> u32 {
        if num_kangaroos == 0 || optimal_steps == 0 {
            return 0;
        }
        let budgeted_dps = ((max_dps as u128) * 9 / 10).max(1);
        let dp_spacing = 1u128 << dp_bits;
        let num_k = num_kangaroos as u128;
        let allowed_steps = (budgeted_dps.saturating_mul(dp_spacing) / num_k).max(1);
        let capped_steps = allowed_steps.min(u128::from(u32::MAX)) as u32;
        capped_steps.min(optimal_steps)
    }

    #[allow(dead_code)]
    fn new_with_pipeline(
        ctx: &GpuContext,
        pipeline: &KangarooPipeline,
        pubkey: Point,
        start: U256,
        range_bits: u32,
        dp_bits: u32,
        num_kangaroos: u32,
    ) -> Result<Self> {
        let jump_table_size = 256u32;
        let (jump_points, jump_distances) = generate_jump_table(range_bits);
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
            jump_table_size,
            _padding: 0,
        };
        let buffers = GpuBuffers::new(
            ctx, pipeline, &config, &jump_points, &jump_distances, num_kangaroos, MAX_DISTINGUISHED_POINTS,
        )?;
        let kangaroos = initialize_kangaroos(&pubkey, &start, range_bits, num_kangaroos)?;
        upload_kangaroos(ctx, &buffers, &kangaroos)?;
        let pipeline_clone = KangarooPipeline {
            pipeline: pipeline.pipeline.clone(),
            bind_group_layout: pipeline.bind_group_layout.clone(),
        };
        Ok(Self {
            ctx: ctx.clone(),
            pipeline: pipeline_clone,
            buffers,
            dp_table: DPTable::new(start),
            total_ops: 0,
            num_kangaroos,
            steps_per_call,
            speed_timer: Instant::now(),
            speed_ops_snapshot: 0,
            current_ops_per_sec: 0.0,
        })
    }

    fn new_internal(
        ctx: GpuContext,
        pubkey: Point,
        start: U256,
        range_bits: u32,
        dp_bits: u32,
        num_kangaroos: u32,
        verbose: bool,
    ) -> Result<Self> {
        if verbose { info!("Creating pipeline..."); }
        let pipeline = KangarooPipeline::new(&ctx)?;
        if verbose { info!("Pipeline created"); }

        if verbose { info!("Generating jump table..."); }
        let jump_table_size = 256u32;
        let (jump_points, jump_distances) = generate_jump_table(range_bits);
        if verbose {
            info!("Jump table generated: {} entries", jump_table_size);
            for (i, dist) in jump_distances.iter().enumerate().take(4) {
                info!("Jump dist[{}] = 0x{:08x}", i, dist[0]);
            }
        }
        let dp_mask = create_dp_mask(dp_bits);
        if verbose { info!("DP mask created"); }

        let steps_per_call = Self::select_steps_per_call(
            ctx.optimal_steps_per_call(), num_kangaroos, dp_bits, MAX_DISTINGUISHED_POINTS,
        );
        let config = GpuConfig {
            dp_mask_lo: [dp_mask[0], dp_mask[1], dp_mask[2], dp_mask[3]],
            dp_mask_hi: [dp_mask[4], dp_mask[5], dp_mask[6], dp_mask[7]],
            num_kangaroos,
            steps_per_call,
            jump_table_size,
            _padding: 0,
        };
        if verbose { info!("Config created: steps_per_call={}", steps_per_call); }

        if verbose { info!("Creating GPU buffers..."); }
        let buffers = GpuBuffers::new(
            &ctx, &pipeline, &config, &jump_points, &jump_distances, num_kangaroos, MAX_DISTINGUISHED_POINTS,
        )?;
        let kangaroos = initialize_kangaroos(&pubkey, &start, range_bits, num_kangaroos)?;
        upload_kangaroos(&ctx, &buffers, &kangaroos)?;

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

        solver.calibrate(dp_bits, verbose);

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

    pub fn step(&mut self) -> Result<Option<Vec<u8>>> {
        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Kangaroo Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Kangaroo Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline.pipeline);
            pass.set_bind_group(0, &self.buffers.bind_group, &[]);
            let workgroups = self.num_kangaroos.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.buffers.dp_count_buffer, 0, &self.buffers.staging_buffer, 0, 4,
        );
        self.ctx.queue.submit(Some(encoder.finish()));
        self.total_ops += (self.num_kangaroos as u64) * (self.steps_per_call as u64);

        // Update speed measurement every ~1 second
        let elapsed_speed = self.speed_timer.elapsed();
        if elapsed_speed.as_millis() >= 1000 {
            let ops_delta = self.total_ops - self.speed_ops_snapshot;
            self.current_ops_per_sec = ops_delta as f64 / elapsed_speed.as_secs_f64();
            self.speed_ops_snapshot = self.total_ops;
            self.speed_timer = Instant::now();
        }

        // Log progress every ~10M ops
        if self.total_ops % 10_000_000 < (self.num_kangaroos as u64 * self.steps_per_call as u64) {
            let (tame, wild) = self.dp_table.count_by_type();
            let speed_mops = self.current_ops_per_sec / 1_000_000.0;
            tracing::info!(
                "Ops: {}M | Speed: {:.2} Mop/s | DPs: {} ({} tame, {} wild)",
                self.total_ops / 1_000_000,
                speed_mops,
                self.dp_table.total_dps(),
                tame,
                wild
            );
        }

        let dp_count = self.read_dp_count()?;
        if dp_count > 0 {
            let mut encoder2 = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("DP Readback"),
            });
            let dp_size = std::mem::size_of::<GpuDistinguishedPoint>();
            let actual_count = (dp_count as usize).min(MAX_DISTINGUISHED_POINTS as usize);
            let copy_size = (actual_count * dp_size) as u64;
            encoder2.copy_buffer_to_buffer(
                &self.buffers.dp_buffer, 0, &self.buffers.staging_buffer, 4, copy_size,
            );
            self.ctx.queue.submit(Some(encoder2.finish()));
            let dps = self.read_dps(actual_count as u32)?;
            for dp in dps {
                if let Some(key) = self.dp_table.insert_and_check(dp) {
                    return Ok(Some(key));
                }
            }
            self.reset_dp_count()?;
        }
        Ok(None)
    }

    pub fn total_operations(&self) -> u64 {
        self.total_ops
    }

    fn read_dp_count(&self) -> Result<u32> {
        let slice = self.buffers.staging_buffer.slice(0..4);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        // wgpu 0.20: Maintain::Wait blocks until GPU work is done
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
        let total_size = 4 + (count as usize * dp_size);
        let slice = self.buffers.staging_buffer.slice(0..total_size as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        // wgpu 0.20: Maintain::Wait blocks until GPU work is done
        self.ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;
        let data = slice.get_mapped_range();
        let dp_bytes = &data[4..];
        let dps: Vec<GpuDistinguishedPoint> = dp_bytes
            .chunks_exact(dp_size)
            .take(count as usize)
            .map(|chunk| *bytemuck::from_bytes::<GpuDistinguishedPoint>(chunk))
            .collect();
        drop(data);
        self.buffers.staging_buffer.unmap();
        Ok(dps)
    }

    fn reset_dp_count(&self) -> Result<()> {
        self.ctx.queue.write_buffer(&self.buffers.dp_count_buffer, 0, &[0u8; 4]);
        Ok(())
    }

    /// Calibrate steps_per_call by timing increasingly large dispatches.
    /// Extended candidates up to 8192 for AMD/NVIDIA Vulkan which can handle
    /// much more work per dispatch than the old conservative 512 cap.
    fn calibrate(&mut self, dp_bits: u32, verbose: bool) {
        // Extended range: AMD RDNA/GCN cards via Vulkan can do 2048-8192 steps
        // per call well within the 50ms TDR-safe window.
        let candidates = [64u32, 128, 256, 512, 1024, 2048, 4096, 8192];
        let mut best_steps = candidates[0];
        if verbose { info!("Calibrating GPU performance (extended range for AMD/NVIDIA)..."); }
        for &steps in &candidates {
            let max_steps = Self::select_steps_per_call(steps, self.num_kangaroos, dp_bits, MAX_DISTINGUISHED_POINTS);
            if max_steps < steps { break; }
            let config = GpuConfig {
                dp_mask_lo: [0; 4],
                dp_mask_hi: [0; 4],
                num_kangaroos: self.num_kangaroos,
                steps_per_call: steps,
                jump_table_size: 256,
                _padding: 0,
            };
            self.ctx.queue.write_buffer(&self.buffers.config_buffer, 0, bytemuck::bytes_of(&config));
            // Warm-up dispatch (not timed)
            self.dispatch_once();
            let start = Instant::now();
            self.dispatch_once();
            let elapsed_ms = start.elapsed().as_millis();
            if verbose { info!("  steps_per_call={}: {}ms", steps, elapsed_ms); }
            if elapsed_ms <= TARGET_DISPATCH_MS {
                best_steps = steps;
            } else {
                break;
            }
        }
        self.steps_per_call = best_steps;
        if verbose {
            let throughput_mops = (self.num_kangaroos as f64 * best_steps as f64) / 1_000_000.0;
            info!("Calibrated: steps_per_call={} ({:.2}M ops/dispatch)", best_steps, throughput_mops);
        }
    }

    fn dispatch_once(&self) {
        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Calibration Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Calibration Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline.pipeline);
            pass.set_bind_group(0, &self.buffers.bind_group, &[]);
            let workgroups = self.num_kangaroos.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
        // wgpu 0.20: Maintain::Wait blocks until GPU work is done
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
        assert_eq!(steps, 921);
    }

    #[test]
    fn keeps_optimal_when_within_budget() {
        let steps = KangarooSolver::select_steps_per_call(4_096, 4_096, 16, MAX_DISTINGUISHED_POINTS);
        assert_eq!(steps, 4_096);
    }
}
