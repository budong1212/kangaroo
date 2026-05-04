//! Kangaroo: Pollard's Kangaroo ECDLP solver using Vulkan/Metal/DX12 compute

mod cli;
mod convert;
mod cpu;
mod crypto;
mod gpu;
mod gpu_crypto;
mod math;
mod solver;

pub use solver::KangarooSolver;
pub use crypto::{full_verify, parse_hex_u256, parse_pubkey, verify_key, Point};
pub use gpu_crypto::GpuContext;

use clap::Parser;
use indicatif::ProgressBar;
use serde::Serialize;
use std::time::Instant;
use tracing::{error, info};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Public key to solve (compressed hex, 33 bytes)
    #[arg(short, long)]
    pubkey: String,

    /// Start of search range (hex, without 0x prefix)
    #[arg(short, long)]
    start: String,

    /// Bit range to search (key is in [start, start + 2^range])
    #[arg(short, long, default_value = "32")]
    range: u32,

    /// Distinguished point bits (auto-calculated if not set)
    #[arg(short, long)]
    dp_bits: Option<u32>,

    /// Number of kangaroos (default: auto based on GPU)
    #[arg(short, long)]
    kangaroos: Option<u32>,

    /// GPU device index
    #[arg(long, default_value = "0")]
    gpu: u32,

    /// Output file for result (hex private key)
    #[arg(short, long)]
    output: Option<String>,

    /// Quiet mode - minimal output, just print found key
    #[arg(short, long)]
    quiet: bool,

    /// Maximum operations before giving up (0 = unlimited)
    #[arg(long, default_value = "0")]
    max_ops: u64,

    /// Use CPU solver instead of GPU (slow, for benchmarking)
    #[arg(long)]
    cpu: bool,

    /// Output benchmark results in JSON format to stdout
    #[arg(long)]
    json: bool,
}

#[derive(Serialize)]
struct BenchmarkResult {
    metric: String,
    value: f64,
    unit: String,
    metadata: Metadata,
}

#[derive(Serialize)]
struct Metadata {
    device: String,
    range_bits: u32,
    algorithm: String,
    total_ops: u64,
    time_seconds: f64,
}

pub fn run_from_args<I, S>(args: I) -> anyhow::Result<()>
where
    I: IntoIterator<Item = S>,
    S: Into<std::ffi::OsString> + Clone,
{
    let args = Args::parse_from(args);
    run(args)
}

pub fn run(args: Args) -> anyhow::Result<()> {
    cli::init_tracing(false, args.quiet || args.json);

    if !args.quiet && !args.json {
        info!("Kangaroo ECDLP Solver");
        info!("=====================");
        info!("Target pubkey: {}", args.pubkey);
        info!("Search range: {} bits from 0x{}", args.range, args.start);
    }

    let pubkey = crypto::parse_pubkey(&args.pubkey)?;
    let start = crypto::parse_hex_u256(&args.start)?;
    let range_bits = args.range;

    if args.cpu {
        if !args.quiet && !args.json {
            info!("Mode: CPU (Software Solver)");
        }
        let dp_bits = args.dp_bits.unwrap_or_else(|| {
            (range_bits / 2).saturating_sub(2).clamp(8, 20)
        });
        if !args.quiet && !args.json {
            info!("DP bits: {}", dp_bits);
        }
        let mut start_be = start;
        start_be.reverse();
        let mut solver = cpu::CpuKangarooSolver::new_full(pubkey.clone(), start_be, range_bits, dp_bits);
        let expected_ops = (1u128 << (range_bits / 2)) as u64;
        let pb = if args.quiet || args.json {
            ProgressBar::hidden()
        } else {
            let pb = ProgressBar::new(expected_ops);
            pb.set_style(cli::default_progress_style_with_msg());
            pb
        };
        let start_time = Instant::now();
        let result = solver.solve(std::time::Duration::from_secs(3600));
        let duration = start_time.elapsed();
        if let Some(private_key) = result {
            pb.finish_with_message("FOUND!");
            let key_hex = hex::encode(&private_key);
            let key_hex_trimmed = key_hex.trim_start_matches('0');
            let key_hex_display = if key_hex_trimmed.is_empty() { "0" } else { key_hex_trimmed };
            if args.json {
                let total_ops = solver.total_ops();
                let time_seconds = duration.as_secs_f64();
                let rate = total_ops as f64 / time_seconds;
                let result = BenchmarkResult {
                    metric: "hash_rate".to_string(),
                    value: rate,
                    unit: "ops/s".to_string(),
                    metadata: Metadata {
                        device: "cpu".to_string(),
                        range_bits,
                        algorithm: "pollard_kangaroo".to_string(),
                        total_ops,
                        time_seconds,
                    },
                };
                println!("{}", serde_json::to_string(&result)?);
            } else if args.quiet {
                println!("{}", key_hex_display);
            } else {
                info!("Private key found: 0x{}", key_hex_display);
                info!("Verification: SUCCESS");
                info!("Total operations: {}", solver.total_ops());
                info!("Time elapsed: {:.2}s", duration.as_secs_f64());
            }
            if let Some(ref output) = args.output {
                std::fs::write(output, &key_hex)?;
            }
            return Ok(());
        } else {
            pb.finish_with_message("TIMEOUT");
            return Err(anyhow::anyhow!("Key not found within timeout"));
        }
    }

    // GPU MODE
    let gpu_context = pollster::block_on(gpu_crypto::GpuContext::new(args.gpu))?;
    let device_name = gpu_context.device_name().to_string();
    if !args.quiet && !args.json {
        info!("GPU: {}", device_name);
        info!("Compute units: {}", gpu_context.compute_units());
    }

    let pubkey = crypto::parse_pubkey(&args.pubkey)?;
    let start = crypto::parse_hex_u256(&args.start)?;
    let range_bits = args.range;
    let num_k = args.kangaroos.unwrap_or(gpu_context.optimal_kangaroos());

    // DP bits: target ~1 DP per kangaroo per dispatch batch.
    // dp_bits controls how often a thread finds a distinguished point.
    // With pure Jacobian DP check (no inversion), we use x[0] low bits.
    // A 20-bit mask means ~1/1M chance per step — reasonable rate.
    // For range_bits >= 64 use 20, for smaller ranges scale down.
    let dp_bits = args.dp_bits.unwrap_or_else(|| {
        // target: each kangaroo hits a DP roughly every 2^dp_bits steps
        // want enough DPs to fill table: ~2 * sqrt(2^range) total
        // dp_bits ~ range/2 / log2(num_k) clamped to [12, 24]
        let auto = (range_bits / 2).saturating_sub(((num_k as f64).log2() as u32) / 2);
        auto.clamp(12, 24)
    });

    if !args.quiet && !args.json {
        info!("DP bits: {} (mask={:#010x})", dp_bits, (1u32 << dp_bits).wrapping_sub(1));
        info!("Kangaroos: {}", num_k);
    }

    let mut solver = solver::KangarooSolver::new(
        gpu_context,
        pubkey.clone(),
        start,
        range_bits,
        dp_bits,
        num_k,
    )?;

    let expected_ops = (1u128 << (range_bits / 2)) as u64;
    let pb = if args.quiet || args.json {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(expected_ops);
        pb.set_style(cli::default_progress_style());
        pb
    };

    if !args.quiet && !args.json {
        info!("Starting search...");
    }

    let max_ops = if args.max_ops == 0 { u64::MAX } else { args.max_ops };
    let start_time = Instant::now();
    let mut last_speed_update = Instant::now();

    loop {
        let result = solver.step()?;
        let total_ops = solver.total_operations();
        pb.set_position(total_ops);

        if last_speed_update.elapsed().as_millis() >= 500 {
            let ops_per_sec = solver.current_ops_per_sec;
            if ops_per_sec > 0.0 {
                let speed_str = if ops_per_sec >= 1_000_000.0 {
                    format!("{:.2} Mop/s", ops_per_sec / 1_000_000.0)
                } else if ops_per_sec >= 1_000.0 {
                    format!("{:.1} Kop/s", ops_per_sec / 1_000.0)
                } else {
                    format!("{:.0} op/s", ops_per_sec)
                };
                pb.set_message(speed_str);
            }
            last_speed_update = Instant::now();
        }

        if let Some(private_key) = result {
            let duration = start_time.elapsed();
            pb.finish_with_message("FOUND!");
            let key_hex = hex::encode(&private_key);
            let key_hex_trimmed = key_hex.trim_start_matches('0');
            let key_hex_display = if key_hex_trimmed.is_empty() { "0" } else { key_hex_trimmed };

            if !crypto::verify_key(&private_key, &pubkey) {
                error!("Verification FAILED - this is a bug!");
                continue;
            }

            if args.json {
                let time_seconds = duration.as_secs_f64();
                let rate = total_ops as f64 / time_seconds;
                let result = BenchmarkResult {
                    metric: "hash_rate".to_string(),
                    value: rate,
                    unit: "ops/s".to_string(),
                    metadata: Metadata {
                        device: device_name,
                        range_bits,
                        algorithm: "pollard_kangaroo".to_string(),
                        total_ops,
                        time_seconds,
                    },
                };
                println!("{}", serde_json::to_string(&result)?);
            } else if args.quiet {
                println!("{}", key_hex_display);
            } else {
                let time_seconds = duration.as_secs_f64();
                let final_speed = total_ops as f64 / time_seconds;
                info!("Private key found: 0x{}", key_hex_display);
                info!("Verification: SUCCESS");
                info!("Total operations: {}", total_ops);
                info!("Time elapsed: {:.2}s", time_seconds);
                info!("Average speed: {:.2} Mop/s", final_speed / 1_000_000.0);
            }

            if let Some(ref output) = args.output {
                std::fs::write(output, &key_hex)?;
                if !args.quiet && !args.json {
                    info!("Result written to: {}", output);
                }
            }
            return Ok(());
        }

        if total_ops >= max_ops {
            pb.finish_with_message("LIMIT REACHED");
            if !args.quiet && !args.json {
                info!("Maximum operations reached ({}) without finding key", max_ops);
            }
            return Err(anyhow::anyhow!("Key not found within {} operations", max_ops));
        }
    }
}
