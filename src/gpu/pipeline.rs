//! Compute pipeline setup

use super::GpuContext;
use anyhow::Result;
use std::sync::Arc;
use tracing::info;
use wgpu::{BindGroupLayout, ComputePipeline};

/// Kangaroo compute pipeline (Clone is cheap - wgpu types are Arc-wrapped)
#[derive(Clone)]
pub struct KangarooPipeline {
    pub pipeline: Arc<ComputePipeline>,
    pub bind_group_layout: Arc<BindGroupLayout>,
}

impl KangarooPipeline {
    /// Create the compute pipeline
    pub fn new(ctx: &GpuContext) -> Result<Self> {
        info!("Loading shader sources...");
        // Use shared shaders from internal gpu_crypto module
        let field = crate::gpu_crypto::shaders::FIELD_WGSL;
        let curve = crate::gpu_crypto::shaders::CURVE_JACOBIAN_WGSL;
        let kangaroo = include_str!("../shaders/kangaroo_jacobian.wgsl");

        info!("Creating shader module...");
        let shader = ctx.create_shader_module("Kangaroo Shader", &[field, curve, kangaroo]);
        info!("Shader module created");

        info!("Creating bind group layout...");
        // Create bind group layout
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Kangaroo Bind Group Layout"),
                    entries: &[
                        // Config (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Jump points (storage, read_only)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Jump distances (storage, read_only)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Kangaroos (storage, read_write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // DP buffer (storage, read_write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // DP count (storage, read_write atomic)
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        info!("Bind group layout created");

        info!("Creating pipeline layout...");
        // wgpu 0.20: PipelineLayoutDescriptor has no immediate_size field
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Kangaroo Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        info!("Pipeline layout created");

        info!("Creating compute pipeline...");
        // wgpu 0.20: entry_point is &str (not Option<&str>), no cache field
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Kangaroo Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            });
        info!("Compute pipeline created");

        Ok(Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
        })
    }
}
