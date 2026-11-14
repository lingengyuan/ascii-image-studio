use std::sync::OnceLock;

use crate::preprocess::{CELL_HEIGHT, CELL_WIDTH};

pub const SAMPLE_COLUMNS: usize = 5;
pub const SAMPLE_ROWS: usize = 10;
pub const RADIUS_BINS: usize = 5;
pub const ANGLE_BINS: usize = 12;
pub const SAMPLE_COUNT: usize = SAMPLE_COLUMNS * SAMPLE_ROWS;
const HISTOGRAM_BINS: usize = RADIUS_BINS * ANGLE_BINS;
pub const DESCRIPTOR_LEN: usize = SAMPLE_COUNT * HISTOGRAM_BINS;
const CELL_PIXELS: usize = (CELL_WIDTH * CELL_HEIGHT) as usize;

/// 对数半径的最内圈；采样点落在像素中心之间，最近的像素距离为 0.5。
const MIN_RADIUS: f32 = 0.5;

/// 与 `preprocess::edge_maps` 相同的 DoG 尺度。字形覆盖图和图像单元都必须经过
/// 这一组尺度，否则两者的边缘能量在空间上不可比。
const DOG_SIGMA_FINE: f32 = 0.7;
const DOG_SIGMA_COARSE: f32 = 1.6;
const EDGE_EPSILON: f32 = 1.0e-6;

/// 每个采样点、每个像素的直方图归属，索引为 `sample * CELL_PIXELS + pixel`。
/// 只和单元格几何有关，整个进程只构造一次。
static SAMPLE_BINS: OnceLock<Vec<u8>> = OnceLock::new();

/// 单元格内 50 个采样点共用的径向、角向分箱。半径按对数间隔覆盖整个对角线。
pub fn sample_bins() -> &'static [u8] {
    SAMPLE_BINS.get_or_init(build_sample_bins)
}

fn build_sample_bins() -> Vec<u8> {
    let max_radius = (CELL_WIDTH as f32).hypot(CELL_HEIGHT as f32);
    let log_ratio = (max_radius / MIN_RADIUS).ln() / RADIUS_BINS as f32;
    let mut bins = Vec::with_capacity(SAMPLE_COUNT * CELL_PIXELS);
    for sample_row in 0..SAMPLE_ROWS {
        let center_y = (sample_row as f32 + 0.5) * CELL_HEIGHT as f32 / SAMPLE_ROWS as f32;
        for sample_col in 0..SAMPLE_COLUMNS {
            let center_x = (sample_col as f32 + 0.5) * CELL_WIDTH as f32 / SAMPLE_COLUMNS as f32;
            for y in 0..CELL_HEIGHT {
                let dy = y as f32 + 0.5 - center_y;
                for x in 0..CELL_WIDTH {
                    let dx = x as f32 + 0.5 - center_x;
                    let radius = dx.hypot(dy).clamp(MIN_RADIUS, max_radius);
                    let radius_bin =
                        (((radius / MIN_RADIUS).ln() / log_ratio) as usize).min(RADIUS_BINS - 1);
                    let angle = dy.atan2(dx).rem_euclid(std::f32::consts::TAU);
                    let angle_bin =
                        ((angle / std::f32::consts::TAU * ANGLE_BINS as f32) as usize) % ANGLE_BINS;
                    bins.push((radius_bin * ANGLE_BINS + angle_bin) as u8);
                }
            }
        }
    }
    bins
}

/// 单元边缘结果。归一化图只保留分布，绝对幅值单独保留给能量门控。
pub struct CellEdges {
    pub edges: [f32; CELL_PIXELS],
    /// 归一化前的平均 Sobel 幅值，单位与输入亮度相同。
    pub mean_magnitude: f32,
}

/// 10x20 小图上统一的边缘算子：DoG(0.7, 1.6) → Sobel 幅值 → 按最大值归一化。
/// 边界按复制处理，与 `preprocess::edge_maps` 的小图版本一致。
///
/// 字形覆盖图和目标单元都调用这个函数，描述子才处在同一空间尺度上。
/// 归一化会抹掉对比度差异，弱边缘回退必须用 `mean_magnitude` 判断。
pub fn cell_edges(values: &[f32; CELL_PIXELS]) -> CellEdges {
    let fine = gaussian_blur_grid(values, DOG_SIGMA_FINE);
    let coarse = gaussian_blur_grid(values, DOG_SIGMA_COARSE);
    let mut dog = [0.0_f32; CELL_PIXELS];
    for (difference, (fine, coarse)) in dog.iter_mut().zip(fine.iter().zip(&coarse)) {
        *difference = fine - coarse;
    }

    let mut edges = [0.0_f32; CELL_PIXELS];
    let mut max_magnitude = 0.0_f32;
    for y in 0..CELL_HEIGHT as isize {
        for x in 0..CELL_WIDTH as isize {
            let top_left = sample_grid(&dog, x - 1, y - 1);
            let top = sample_grid(&dog, x, y - 1);
            let top_right = sample_grid(&dog, x + 1, y - 1);
            let left = sample_grid(&dog, x - 1, y);
            let right = sample_grid(&dog, x + 1, y);
            let bottom_left = sample_grid(&dog, x - 1, y + 1);
            let bottom = sample_grid(&dog, x, y + 1);
            let bottom_right = sample_grid(&dog, x + 1, y + 1);

            let gx =
                (top_right + 2.0 * right + bottom_right) - (top_left + 2.0 * left + bottom_left);
            let gy =
                (bottom_left + 2.0 * bottom + bottom_right) - (top_left + 2.0 * top + top_right);
            let magnitude = gx.hypot(gy);
            edges[y as usize * CELL_WIDTH as usize + x as usize] = magnitude;
            max_magnitude = max_magnitude.max(magnitude);
        }
    }

    let mean_magnitude = edges.iter().sum::<f32>() / CELL_PIXELS as f32;
    if max_magnitude > EDGE_EPSILON {
        for edge in &mut edges {
            *edge = (*edge / max_magnitude).clamp(0.0, 1.0);
        }
    } else {
        edges = [0.0; CELL_PIXELS];
    }
    CellEdges {
        edges,
        mean_magnitude,
    }
}

fn gaussian_blur_grid(values: &[f32; CELL_PIXELS], sigma: f32) -> [f32; CELL_PIXELS] {
    let radius = (sigma * 3.0).ceil() as isize;
    let mut kernel = Vec::with_capacity((radius * 2 + 1) as usize);
    let mut kernel_sum = 0.0;
    for offset in -radius..=radius {
        let weight = (-0.5 * (offset as f32 / sigma).powi(2)).exp();
        kernel.push(weight);
        kernel_sum += weight;
    }
    for weight in &mut kernel {
        *weight /= kernel_sum;
    }

    let mut horizontal = [0.0_f32; CELL_PIXELS];
    for y in 0..CELL_HEIGHT as isize {
        for x in 0..CELL_WIDTH as isize {
            let mut sum = 0.0;
            for (kernel_index, weight) in kernel.iter().enumerate() {
                let offset = kernel_index as isize - radius;
                sum += sample_grid(values, x + offset, y) * weight;
            }
            horizontal[y as usize * CELL_WIDTH as usize + x as usize] = sum;
        }
    }

    let mut output = [0.0_f32; CELL_PIXELS];
    for y in 0..CELL_HEIGHT as isize {
        for x in 0..CELL_WIDTH as isize {
            let mut sum = 0.0;
            for (kernel_index, weight) in kernel.iter().enumerate() {
                let offset = kernel_index as isize - radius;
                sum += sample_grid(&horizontal, x, y + offset) * weight;
            }
            output[y as usize * CELL_WIDTH as usize + x as usize] = sum;
        }
    }
    output
}

fn sample_grid(values: &[f32; CELL_PIXELS], x: isize, y: isize) -> f32 {
    let clamped_x = x.clamp(0, CELL_WIDTH as isize - 1) as usize;
    let clamped_y = y.clamp(0, CELL_HEIGHT as isize - 1) as usize;
    values[clamped_y * CELL_WIDTH as usize + clamped_x]
}

/// 字形覆盖位图先转成 0..1 亮度，再走统一的单元边缘算子。
pub fn glyph_descriptor(coverage: &[u8]) -> Vec<f32> {
    assert_eq!(coverage.len(), CELL_PIXELS);
    let mut luma = [0.0_f32; CELL_PIXELS];
    for (value, target) in coverage.iter().zip(luma.iter_mut()) {
        *target = f32::from(*value) / 255.0;
    }
    cell_descriptor(&luma)
}

/// 目标单元 10x20 亮度走同一个边缘算子后做对数极坐标直方图。
pub fn cell_descriptor(luma: &[f32; CELL_PIXELS]) -> Vec<f32> {
    descriptor_from_edges(&cell_edges(luma).edges)
}

/// 边缘幅值按采样点分箱累加，再整体除以总和；全零边缘保留零向量。
pub fn descriptor_from_edges(edges: &[f32; CELL_PIXELS]) -> Vec<f32> {
    let bins = sample_bins();
    let mut descriptor = vec![0.0_f32; DESCRIPTOR_LEN];
    for sample in 0..SAMPLE_COUNT {
        let base = sample * CELL_PIXELS;
        let histogram = &mut descriptor[sample * HISTOGRAM_BINS..(sample + 1) * HISTOGRAM_BINS];
        for (pixel, edge) in edges.iter().enumerate() {
            if *edge == 0.0 {
                continue;
            }
            histogram[bins[base + pixel] as usize] += edge;
        }
    }
    let sum: f32 = descriptor.iter().sum();
    if sum > 0.0 {
        for value in &mut descriptor {
            *value /= sum;
        }
    }
    descriptor
}

pub fn l1_distance(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::{
        ANGLE_BINS, CELL_PIXELS, DESCRIPTOR_LEN, HISTOGRAM_BINS, RADIUS_BINS, SAMPLE_COUNT,
        cell_descriptor, cell_edges, descriptor_from_edges, glyph_descriptor, l1_distance,
        sample_bins,
    };
    use crate::font_atlas::FontAtlas;

    fn step_luma(position: usize) -> [f32; CELL_PIXELS] {
        std::array::from_fn(|index| if index % 10 >= position { 1.0 } else { 0.0 })
    }

    fn horizontal_step_luma(position: usize) -> [f32; CELL_PIXELS] {
        std::array::from_fn(|index| if index / 10 >= position { 1.0 } else { 0.0 })
    }

    fn step_coverage(position: usize) -> Vec<u8> {
        let mut coverage = vec![0_u8; CELL_PIXELS];
        for y in 0..20 {
            for x in position..10 {
                coverage[y * 10 + x] = 255;
            }
        }
        coverage
    }

    #[test]
    fn glyph_and_image_paths_produce_the_same_descriptor_for_a_step() {
        let coverage = step_coverage(5);
        let from_coverage = glyph_descriptor(&coverage);
        let from_luma = cell_descriptor(&step_luma(5));
        let distance = l1_distance(&from_coverage, &from_luma);
        assert!(distance < 1.0e-6, "distance={distance}");
        assert!((from_luma.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn a_glyph_bitmap_is_closest_to_its_own_descriptor() {
        let atlas = FontAtlas::load().unwrap();
        let descriptors = atlas.aiss_descriptors();
        let index = ('D' as usize) - 0x20;
        let mut luma = [0.0_f32; CELL_PIXELS];
        for (value, target) in atlas.ascii()[index].bitmap.iter().zip(luma.iter_mut()) {
            *target = f32::from(*value) / 255.0;
        }
        let from_image = cell_descriptor(&luma);
        let self_distance = l1_distance(&from_image, &descriptors[index]);
        let b_distance = l1_distance(&from_image, &descriptors[('B' as usize) - 0x20]);
        assert!(self_distance < 1.0e-6, "self distance={self_distance}");
        assert!(self_distance < b_distance, "B distance={b_distance}");
    }

    #[test]
    fn unified_edge_operator_is_scale_invariant() {
        let luma = step_luma(5);
        let mut scaled = luma;
        for value in &mut scaled {
            *value *= 0.25;
        }
        let base = cell_descriptor(&luma);
        let scaled = cell_descriptor(&scaled);
        let distance = l1_distance(&base, &scaled);
        assert!(distance < 1.0e-6, "distance={distance}");
    }

    #[test]
    fn shifted_step_stays_closer_than_an_orthogonal_step() {
        let base = cell_descriptor(&step_luma(5));
        let orthogonal = cell_descriptor(&horizontal_step_luma(10));
        for shift in [1, 2] {
            let shifted = cell_descriptor(&step_luma(5 - shift));
            let distance = l1_distance(&base, &shifted);
            let unrelated = l1_distance(&base, &orthogonal);
            assert!(
                distance < unrelated,
                "shift={shift} distance={distance} orthogonal={unrelated}"
            );
        }
    }

    #[test]
    fn absolute_edge_energy_scales_with_contrast() {
        let full = cell_edges(&step_luma(5));
        let dim = cell_edges(&std::array::from_fn(|index| step_luma(5)[index] / 255.0));
        assert!(full.mean_magnitude > 0.0);
        assert!(dim.mean_magnitude > 0.0);
        let ratio = full.mean_magnitude / dim.mean_magnitude;
        assert!((ratio - 255.0).abs() < 0.01, "ratio={ratio}");

        let flat = cell_edges(&[0.5; CELL_PIXELS]);
        assert_eq!(flat.mean_magnitude, 0.0);
    }

    #[test]
    fn glyph_descriptors_are_normalized_and_finite() {
        let atlas = FontAtlas::load().unwrap();
        let descriptors = atlas.aiss_descriptors();
        assert_eq!(descriptors.len(), atlas.ascii().len());
        assert_eq!(atlas.ascii()[0].ch, ' ');
        assert!(descriptors[0].iter().all(|value| *value == 0.0));

        for (glyph, descriptor) in atlas.ascii().iter().zip(descriptors) {
            assert_eq!(descriptor.len(), DESCRIPTOR_LEN, "ch={}", glyph.ch);
            assert!(
                descriptor.iter().all(|value| value.is_finite()),
                "ch={}",
                glyph.ch
            );
            let sum: f32 = descriptor.iter().sum();
            if glyph.ch == ' ' {
                assert_eq!(sum, 0.0);
            } else {
                assert!((sum - 1.0).abs() < 1.0e-5, "ch={} sum={sum}", glyph.ch);
            }
        }
    }

    #[test]
    fn pixel_bin_mapping_covers_every_pixel_for_each_sample() {
        let bins = sample_bins();
        assert_eq!(bins.len(), SAMPLE_COUNT * CELL_PIXELS);
        assert_eq!(HISTOGRAM_BINS, RADIUS_BINS * ANGLE_BINS);
        for sample in 0..SAMPLE_COUNT {
            let slice = &bins[sample * CELL_PIXELS..(sample + 1) * CELL_PIXELS];
            assert_eq!(slice.len(), CELL_PIXELS);
            assert!(
                slice.iter().all(|bin| usize::from(*bin) < HISTOGRAM_BINS),
                "sample={sample}"
            );
        }
    }

    #[test]
    fn blank_edge_map_stays_zero() {
        let descriptor = descriptor_from_edges(&[0.0; CELL_PIXELS]);
        assert!(descriptor.iter().all(|value| *value == 0.0));
        let glyph = glyph_descriptor(&[0; CELL_PIXELS]);
        assert!(glyph.iter().all(|value| *value == 0.0));
    }
}
