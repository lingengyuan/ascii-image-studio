//! 亮度色调映射。默认使用分块限幅直方图均衡（CLAHE）。

use anyhow::{Result, ensure};
use clap::ValueEnum;

const SIGNAL_EPSILON: f32 = 1.0e-6;
const TILE_TARGET_PIXELS: u32 = 32;
const MAX_TILES: u32 = 8;
const CLIP_LIMIT: f32 = 1.8;
/// 局部均衡与全局拉伸的混合比例。平滑天空和纸张对局部均衡很敏感。
const CLAHE_STRENGTH: f32 = 0.5;
const FLAT_BIN_RANGE: u8 = 2;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum ToneMap {
    /// 分块限幅直方图均衡，与全局拉伸等权混合以压低平滑区噪点
    #[default]
    Clahe,
    /// 全局 1%–99% 拉伸后加对比度幂指数；保持旧版色调曲线
    Global,
}

pub fn apply(
    luma: &mut [f32],
    width: u32,
    height: u32,
    mode: ToneMap,
    contrast: f32,
) -> Result<()> {
    ensure!(!luma.is_empty(), "输入图片不能为空");
    ensure!(width > 0 && height > 0, "输入图片不能为空");
    ensure!(
        usize::try_from(u64::from(width) * u64::from(height)).ok() == Some(luma.len()),
        "亮度图尺寸不一致"
    );
    ensure!(
        contrast.is_finite() && contrast > 0.0,
        "contrast 必须是有限正数"
    );

    match mode {
        ToneMap::Global => global_stretch(luma, contrast)?,
        ToneMap::Clahe => {
            let global = percentile_stretch(luma);
            let local = clahe_map(luma, width, height);
            for (value, (global, local)) in luma.iter_mut().zip(global.iter().zip(&local)) {
                *value = (global + CLAHE_STRENGTH * (local - global)).clamp(0.0, 1.0);
            }
            apply_gamma(luma, contrast);
        }
    }
    Ok(())
}

fn apply_gamma(luma: &mut [f32], contrast: f32) {
    for value in luma {
        *value = value.powf(contrast).clamp(0.0, 1.0);
    }
}

fn global_stretch(luma: &mut [f32], contrast: f32) -> Result<()> {
    let stretched = percentile_stretch(luma);
    luma.copy_from_slice(&stretched);
    apply_gamma(luma, contrast);
    Ok(())
}

fn percentile_stretch(luma: &[f32]) -> Vec<f32> {
    let mut sorted = luma.to_vec();
    sorted.sort_by(f32::total_cmp);
    let p01 = quantile(&sorted, 0.01);
    let p99 = quantile(&sorted, 0.99);

    luma.iter()
        .map(|value| {
            if p99 - p01 <= SIGNAL_EPSILON {
                *value
            } else {
                ((*value - p01) / (p99 - p01)).clamp(0.0, 1.0)
            }
        })
        .collect()
}

fn quantile(sorted: &[f32], fraction: f64) -> f32 {
    let position = fraction * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let weight = (position - lower as f64) as f32;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

fn clahe_map(luma: &[f32], width: u32, height: u32) -> Vec<f32> {
    let tiles_x = tile_count(width);
    let tiles_y = tile_count(height);
    let luts = tile_luts(luma, width, height, tiles_x, tiles_y);
    let mut output = vec![0.0; luma.len()];

    let last_x = (tiles_x - 1) as f32;
    let last_y = (tiles_y - 1) as f32;
    for y in 0..height {
        let grid_y = ((y as f32 + 0.5) * tiles_y as f32 / height as f32 - 0.5).clamp(0.0, last_y);
        let y0 = grid_y.floor() as u32;
        let y1 = (y0 + 1).min(tiles_y - 1);
        let ty = grid_y - y0 as f32;
        for x in 0..width {
            let grid_x =
                ((x as f32 + 0.5) * tiles_x as f32 / width as f32 - 0.5).clamp(0.0, last_x);
            let x0 = grid_x.floor() as u32;
            let x1 = (x0 + 1).min(tiles_x - 1);
            let tx = grid_x - x0 as f32;

            let index = y as usize * width as usize + x as usize;
            let bin = bin_index(luma[index]);
            let top = lerp(
                luts[tile_index(tiles_x, x0, y0)][bin],
                luts[tile_index(tiles_x, x1, y0)][bin],
                tx,
            );
            let bottom = lerp(
                luts[tile_index(tiles_x, x0, y1)][bin],
                luts[tile_index(tiles_x, x1, y1)][bin],
                tx,
            );
            output[index] = lerp(top, bottom, ty).clamp(0.0, 1.0);
        }
    }
    output
}

fn tile_count(dimension: u32) -> u32 {
    (dimension / TILE_TARGET_PIXELS).clamp(1, MAX_TILES)
}

fn tile_index(tiles_x: u32, x: u32, y: u32) -> usize {
    (y * tiles_x + x) as usize
}

fn bin_index(value: f32) -> usize {
    (value.clamp(0.0, 1.0) * 255.0).round() as usize
}

fn lerp(left: f32, right: f32, weight: f32) -> f32 {
    left + (right - left) * weight
}

fn tile_luts(luma: &[f32], width: u32, height: u32, tiles_x: u32, tiles_y: u32) -> Vec<[f32; 256]> {
    let mut luts = Vec::with_capacity((tiles_x * tiles_y) as usize);
    for ty in 0..tiles_y {
        let y0 = ty * height / tiles_y;
        let y1 = (ty + 1) * height / tiles_y;
        for tx in 0..tiles_x {
            let x0 = tx * width / tiles_x;
            let x1 = (tx + 1) * width / tiles_x;
            luts.push(tile_lut(luma, width, x0, y0, x1, y1));
        }
    }
    luts
}

fn tile_lut(luma: &[f32], width: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> [f32; 256] {
    let total = (x1 - x0) * (y1 - y0);
    let mut histogram = [0_u32; 256];
    let mut min_bin = 255_u8;
    let mut max_bin = 0_u8;

    for y in y0..y1 {
        for x in x0..x1 {
            let bin = bin_index(luma[y as usize * width as usize + x as usize]);
            histogram[bin] += 1;
            min_bin = min_bin.min(bin as u8);
            max_bin = max_bin.max(bin as u8);
        }
    }

    if max_bin.saturating_sub(min_bin) <= FLAT_BIN_RANGE {
        return std::array::from_fn(|index| index as f32 / 255.0);
    }

    clip_histogram(&mut histogram, total, CLIP_LIMIT);
    let mut cumulative = 0_u32;
    let mut lut = [0.0_f32; 256];
    for (index, count) in histogram.into_iter().enumerate() {
        cumulative += count;
        lut[index] = cumulative as f32 / total as f32;
    }
    lut
}

/// 把超过限幅的计数均摊到全部 256 个桶。返回被重分配的总数。
fn clip_histogram(histogram: &mut [u32; 256], total: u32, clip_limit: f32) -> u32 {
    let limit = ((clip_limit * total as f32 / 256.0).round() as u32).max(1);
    let mut excess = 0;
    for count in histogram.iter_mut() {
        if *count > limit {
            excess += *count - limit;
            *count = limit;
        }
    }
    let base = excess / 256;
    let remainder = excess % 256;
    for (index, count) in histogram.iter_mut().enumerate() {
        *count += base + u32::from((index as u32) < remainder);
    }
    excess
}

#[cfg(test)]
mod tests {
    use super::{ToneMap, apply, clip_histogram};

    #[test]
    fn global_mode_keeps_percentile_stretch_and_gamma() {
        let mut luma = vec![0.2, 0.4, 0.6, 0.8];
        apply(&mut luma, 2, 2, ToneMap::Global, 1.0).unwrap();
        assert!((luma[0] - 0.0).abs() < 1.0e-6);
        assert!((luma[3] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn flat_image_survives_clahe_without_level_shift() {
        let mut luma = vec![0.5; 64 * 64];
        apply(&mut luma, 64, 64, ToneMap::Clahe, 1.0).unwrap();
        assert!(luma.iter().all(|value| (*value - 0.5).abs() <= 1.0 / 255.0));
    }

    #[test]
    fn clipping_preserves_pixel_count_and_bounds_bins() {
        let mut histogram = [0_u32; 256];
        histogram[17] = 1024;
        let excess = clip_histogram(&mut histogram, 1024, 2.0);
        assert_eq!(excess, 1016);
        assert_eq!(histogram.iter().sum::<u32>(), 1024);
        assert_eq!(histogram[17], 12);
        assert!(histogram.iter().all(|count| *count <= 12));
    }

    #[test]
    fn tile_boundaries_do_not_create_steps() {
        let width = 256;
        let height = 64;
        let mut luma: Vec<f32> = (0..width * height)
            .map(|index| (index % width) as f32 / (width - 1) as f32)
            .collect();
        let original = luma.clone();
        apply(&mut luma, width, height, ToneMap::Clahe, 1.0).unwrap();

        let mut max_boundary = 0.0_f32;
        let mut max_interior = 0.0_f32;
        for y in 0..height {
            for x in 1..width {
                let index = y as usize * width as usize + x as usize;
                let diff = (luma[index] - luma[index - 1]).abs();
                if x % 32 == 0 {
                    max_boundary = max_boundary.max(diff);
                } else {
                    max_interior = max_interior.max(diff);
                }
            }
        }
        assert!(
            max_boundary <= max_interior * 1.5 + 0.01,
            "boundary={max_boundary}, interior={max_interior}"
        );
        assert!(luma.iter().all(|value| (0.0..=1.0).contains(value)));
        assert_ne!(luma, original);
    }
}
