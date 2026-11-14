use anyhow::{Result, ensure};
use image::DynamicImage;

use crate::tone::{ToneMap, apply as apply_tone_map};

pub const CELL_WIDTH: u32 = 10;
pub const CELL_HEIGHT: u32 = 20;
pub const SHAPE_AREA_WEIGHTS: [f32; 6] = [0.15, 0.15, 0.175, 0.175, 0.175, 0.175];

const BT601_R: f32 = 0.299;
const BT601_G: f32 = 0.587;
const BT601_B: f32 = 0.114;
const SIGNAL_EPSILON: f32 = 1.0e-6;
const DIRECTION_SMOOTH_SIGMA: f32 = 2.0;
const DIRECTION_SMOOTH_MIN_COHERENCE: f32 = 0.25;
const DIRECTION_WEIGHT_EPSILON: f32 = 1.0e-4;

pub struct ProcessedImage {
    pub width: u32,
    pub height: u32,
    luma: Vec<f32>,
    /// 合成到黑底后的原始亮度，只用于前景明度，不参与匹配。
    display_luma: Vec<f32>,
    normal_x: Vec<f32>,
    normal_y: Vec<f32>,
    edge_strength: Vec<f32>,
    colors: Vec<[u8; 3]>,
}

pub struct WorkImage {
    pub width: u32,
    pub height: u32,
    pub luma: Vec<f32>,
    /// 双角度分量（cos 2θ），插值不会因 180° 反相而抵消。
    pub direction_cos2: Vec<f32>,
    /// 双角度分量（sin 2θ）。
    pub direction_sin2: Vec<f32>,
    pub edge_strength: Vec<f32>,
}

pub fn cell_region_bounds(cell_col: u32, cell_row: u32, component: usize) -> (u32, u32, u32, u32) {
    let sub_col = (component % 2) as u32;
    let sub_row = (component / 2) as u32;
    let x0 = cell_col * CELL_WIDTH + sub_col * (CELL_WIDTH / 2);
    let x1 = x0 + CELL_WIDTH / 2;
    let y0 = cell_row * CELL_HEIGHT + sub_row * CELL_HEIGHT / 3;
    let y1 = cell_row * CELL_HEIGHT + (sub_row + 1) * CELL_HEIGHT / 3;
    (x0, y0, x1, y1)
}

pub fn shape_density(shape: [f32; 6]) -> f32 {
    shape
        .into_iter()
        .zip(SHAPE_AREA_WEIGHTS)
        .map(|(value, weight)| value * weight)
        .sum()
}

pub fn preprocess(
    image: &DynamicImage,
    contrast: f32,
    edge_weight: f32,
    tone_map: ToneMap,
) -> Result<ProcessedImage> {
    ensure!(
        contrast.is_finite() && contrast > 0.0,
        "contrast 必须是有限正数"
    );
    ensure!(
        edge_weight.is_finite() && (0.0..=1.0).contains(&edge_weight),
        "edge_weight 必须在 0 到 1 之间"
    );

    let rgba = image.to_rgba8();
    let width = rgba.width();
    let height = rgba.height();
    ensure!(width > 0 && height > 0, "输入图片不能为空");

    let pixel_count = usize::try_from(u64::from(width) * u64::from(height))
        .map_err(|_| anyhow::anyhow!("图片像素数超出可处理范围"))?;
    let mut luma = Vec::with_capacity(pixel_count);
    let mut colors = Vec::with_capacity(pixel_count);

    for pixel in rgba.pixels() {
        let alpha = u16::from(pixel[3]);
        let red = composite_on_black(pixel[0], alpha);
        let green = composite_on_black(pixel[1], alpha);
        let blue = composite_on_black(pixel[2], alpha);
        luma.push(
            (BT601_R * f32::from(red) + BT601_G * f32::from(green) + BT601_B * f32::from(blue))
                / 255.0,
        );
        colors.push([red, green, blue]);
    }
    let display_luma = luma.clone();

    apply_tone_map(&mut luma, width, height, tone_map, contrast)?;

    let (normal_x, normal_y, edge_strength, signed_dog) = if edge_weight > 0.0 {
        edge_maps(&luma, width, height)
    } else {
        let zeroes = vec![0.0; pixel_count];
        (zeroes.clone(), zeroes.clone(), zeroes.clone(), zeroes)
    };

    if edge_weight > 0.0 {
        apply_signed_unsharp(&mut luma, &signed_dog, edge_weight);
    }

    Ok(ProcessedImage {
        width,
        height,
        luma,
        display_luma,
        normal_x,
        normal_y,
        edge_strength,
        colors,
    })
}

fn composite_on_black(channel: u8, alpha: u16) -> u8 {
    ((u16::from(channel) * alpha + 127) / 255) as u8
}

fn edge_maps(luma: &[f32], width: u32, height: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let narrow = gaussian_blur(luma, width, height, 0.7);
    let wide = gaussian_blur(luma, width, height, 1.6);
    let dog: Vec<f32> = narrow
        .iter()
        .zip(wide.iter())
        .map(|(fine, coarse)| fine - coarse)
        .collect();

    let pixel_count = luma.len();
    let mut normal_x = vec![0.0; pixel_count];
    let mut normal_y = vec![0.0; pixel_count];
    let mut magnitudes = vec![0.0; pixel_count];
    let mut max_magnitude = 0.0_f32;

    for y in 0..height as isize {
        for x in 0..width as isize {
            let top_left = sample_clamped(&dog, width, height, x - 1, y - 1);
            let top = sample_clamped(&dog, width, height, x, y - 1);
            let top_right = sample_clamped(&dog, width, height, x + 1, y - 1);
            let left = sample_clamped(&dog, width, height, x - 1, y);
            let right = sample_clamped(&dog, width, height, x + 1, y);
            let bottom_left = sample_clamped(&dog, width, height, x - 1, y + 1);
            let bottom = sample_clamped(&dog, width, height, x, y + 1);
            let bottom_right = sample_clamped(&dog, width, height, x + 1, y + 1);

            let gx =
                (top_right + 2.0 * right + bottom_right) - (top_left + 2.0 * left + bottom_left);
            let gy =
                (bottom_left + 2.0 * bottom + bottom_right) - (top_left + 2.0 * top + top_right);
            let magnitude = gx.hypot(gy);
            let index = y as usize * width as usize + x as usize;
            magnitudes[index] = magnitude;
            max_magnitude = max_magnitude.max(magnitude);
            if magnitude > SIGNAL_EPSILON {
                normal_x[index] = gx / magnitude;
                normal_y[index] = gy / magnitude;
            }
        }
    }

    if max_magnitude > SIGNAL_EPSILON {
        for magnitude in &mut magnitudes {
            *magnitude = (*magnitude / max_magnitude).clamp(0.0, 1.0);
        }
    }

    smooth_edge_directions(&mut normal_x, &mut normal_y, &magnitudes, width, height);

    (normal_x, normal_y, magnitudes, dog)
}

/// 边缘方向以 180° 为周期，直接平均会让相对的两侧互相抵消。
/// 在双角度域按边缘强度加权平滑，再还原角度，就能跨过反相边界。
fn smooth_edge_directions(
    normal_x: &mut [f32],
    normal_y: &mut [f32],
    edge_strength: &[f32],
    width: u32,
    height: u32,
) {
    let cos_double: Vec<f32> = normal_x
        .iter()
        .zip(normal_y.iter())
        .zip(edge_strength)
        .map(|((nx, ny), weight)| (nx * nx - ny * ny) * weight)
        .collect();
    let sin_double: Vec<f32> = normal_x
        .iter()
        .zip(normal_y.iter())
        .zip(edge_strength)
        .map(|((nx, ny), weight)| 2.0 * nx * ny * weight)
        .collect();

    let smooth_cos = gaussian_blur(&cos_double, width, height, DIRECTION_SMOOTH_SIGMA);
    let smooth_sin = gaussian_blur(&sin_double, width, height, DIRECTION_SMOOTH_SIGMA);
    let smooth_weight = gaussian_blur(edge_strength, width, height, DIRECTION_SMOOTH_SIGMA);

    for index in 0..normal_x.len() {
        if smooth_weight[index] <= DIRECTION_WEIGHT_EPSILON {
            continue;
        }
        // 局部相干度低说明没有单一方向，置零比编造一个方向更诚实。
        let coherence = smooth_cos[index].hypot(smooth_sin[index]) / smooth_weight[index];
        if coherence < DIRECTION_SMOOTH_MIN_COHERENCE {
            normal_x[index] = 0.0;
            normal_y[index] = 0.0;
            continue;
        }
        let theta = 0.5 * smooth_sin[index].atan2(smooth_cos[index]);
        normal_x[index] = theta.cos();
        normal_y[index] = theta.sin();
    }
}

/// 有符号 DoG 非锐化掩模：亮侧更亮、暗侧更暗。
///
/// 不能把非负梯度幅值加到亮度上，那会填亮瞳孔、眼线和唇缝。
fn apply_signed_unsharp(luma: &mut [f32], dog: &[f32], edge_weight: f32) {
    for (value, dog) in luma.iter_mut().zip(dog.iter()) {
        *value = (*value + edge_weight * 1.8 * dog).clamp(0.0, 1.0);
    }
}

fn gaussian_blur(values: &[f32], width: u32, height: u32, sigma: f32) -> Vec<f32> {
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

    let mut horizontal = vec![0.0; values.len()];
    for y in 0..height as isize {
        for x in 0..width as isize {
            let mut sum = 0.0;
            for (kernel_index, weight) in kernel.iter().enumerate() {
                let offset = kernel_index as isize - radius;
                sum += sample_clamped(values, width, height, x + offset, y) * weight;
            }
            horizontal[y as usize * width as usize + x as usize] = sum;
        }
    }

    let mut output = vec![0.0; values.len()];
    for y in 0..height as isize {
        for x in 0..width as isize {
            let mut sum = 0.0;
            for (kernel_index, weight) in kernel.iter().enumerate() {
                let offset = kernel_index as isize - radius;
                sum += sample_clamped(&horizontal, width, height, x, y + offset) * weight;
            }
            output[y as usize * width as usize + x as usize] = sum;
        }
    }
    output
}

fn sample_clamped(values: &[f32], width: u32, height: u32, x: isize, y: isize) -> f32 {
    let clamped_x = x.clamp(0, width as isize - 1) as usize;
    let clamped_y = y.clamp(0, height as isize - 1) as usize;
    values[clamped_y * width as usize + clamped_x]
}

impl ProcessedImage {
    /// 暴露最终匹配亮度，供评测脚本导出对照图；不参与匹配。
    pub fn luma(&self) -> &[f32] {
        &self.luma
    }

    /// 平滑后的单位法向，仅供隐藏的方向诊断图读取。
    pub fn edge_normals(&self) -> (&[f32], &[f32]) {
        (&self.normal_x, &self.normal_y)
    }

    /// 归一化边缘强度，仅供隐藏的方向诊断图读取。
    pub fn edge_strength(&self) -> &[f32] {
        &self.edge_strength
    }

    pub fn resample(&self, width: u32, height: u32) -> Result<WorkImage> {
        ensure!(width > 0 && height > 0, "工作图尺寸必须大于 0");
        let pixel_count = usize::try_from(u64::from(width) * u64::from(height))
            .map_err(|_| anyhow::anyhow!("工作图尺寸超出可处理范围"))?;
        // 先换算到双角度域再插值，相邻反相法向才不会互相抵消。
        let direction_cos2: Vec<f32> = self
            .normal_x
            .iter()
            .zip(&self.normal_y)
            .map(|(normal_x, normal_y)| normal_x * normal_x - normal_y * normal_y)
            .collect();
        let direction_sin2: Vec<f32> = self
            .normal_x
            .iter()
            .zip(&self.normal_y)
            .map(|(normal_x, normal_y)| 2.0 * normal_x * normal_y)
            .collect();
        let mut luma = Vec::with_capacity(pixel_count);
        let mut direction_cos2_out = Vec::with_capacity(pixel_count);
        let mut direction_sin2_out = Vec::with_capacity(pixel_count);
        let mut edge_strength = Vec::with_capacity(pixel_count);

        for y in 0..height {
            let source_y = (y as f32 + 0.5) * self.height as f32 / height as f32 - 0.5;
            for x in 0..width {
                let source_x = (x as f32 + 0.5) * self.width as f32 / width as f32 - 0.5;
                let x0 = source_x.floor() as isize;
                let y0 = source_y.floor() as isize;
                let tx = source_x - x0 as f32;
                let ty = source_y - y0 as f32;
                luma.push(bilinear(
                    &self.luma,
                    self.width,
                    self.height,
                    x0,
                    y0,
                    tx,
                    ty,
                ));
                direction_cos2_out.push(bilinear(
                    &direction_cos2,
                    self.width,
                    self.height,
                    x0,
                    y0,
                    tx,
                    ty,
                ));
                direction_sin2_out.push(bilinear(
                    &direction_sin2,
                    self.width,
                    self.height,
                    x0,
                    y0,
                    tx,
                    ty,
                ));
                edge_strength.push(bilinear(
                    &self.edge_strength,
                    self.width,
                    self.height,
                    x0,
                    y0,
                    tx,
                    ty,
                ));
            }
        }

        Ok(WorkImage {
            width,
            height,
            luma,
            direction_cos2: direction_cos2_out,
            direction_sin2: direction_sin2_out,
            edge_strength,
        })
    }

    pub fn cell_highlight_color(&self, cols: u32, rows: u32, col: u32, row: u32) -> [u8; 3] {
        let (x0, y0, x1, y1) = self.cell_bounds(cols, rows, col, row);

        let mut max_luma = 0.0_f32;
        for y in y0..y1 {
            for x in x0..x1 {
                let index = y as usize * self.width as usize + x as usize;
                let color = self.colors[index];
                max_luma = max_luma.max(color_luma(color));
            }
        }
        if max_luma <= SIGNAL_EPSILON {
            return [255, 255, 255];
        }

        let mut sums = [0.0_f64; 3];
        let mut count = 0.0_f64;
        for y in y0..y1 {
            for x in x0..x1 {
                let index = y as usize * self.width as usize + x as usize;
                let color = self.colors[index];
                if color_luma(color) >= max_luma * 0.85 {
                    for (sum, channel) in sums.iter_mut().zip(color) {
                        *sum += f64::from(channel);
                    }
                    count += 1.0;
                }
            }
        }
        sums.map(|sum| (sum / count).round().clamp(0.0, 255.0) as u8)
    }

    /// 单元在原始合成亮度上的平均值，用于调制前景明度。
    pub fn cell_luminance(&self, cols: u32, rows: u32, col: u32, row: u32) -> f32 {
        let (x0, y0, x1, y1) = self.cell_bounds(cols, rows, col, row);
        let mut sum = 0.0_f64;
        let mut count = 0_u64;
        for y in y0..y1 {
            for x in x0..x1 {
                sum += f64::from(self.display_luma[y as usize * self.width as usize + x as usize]);
                count += 1;
            }
        }
        (sum / count.max(1) as f64) as f32
    }

    fn cell_bounds(&self, cols: u32, rows: u32, col: u32, row: u32) -> (u32, u32, u32, u32) {
        let x0 = u64::from(col) * u64::from(self.width) / u64::from(cols);
        let x1 = (u64::from(col + 1) * u64::from(self.width) / u64::from(cols))
            .max(x0 + 1)
            .min(u64::from(self.width));
        let y0 = u64::from(row) * u64::from(self.height) / u64::from(rows);
        let y1 = (u64::from(row + 1) * u64::from(self.height) / u64::from(rows))
            .max(y0 + 1)
            .min(u64::from(self.height));
        (x0 as u32, y0 as u32, x1 as u32, y1 as u32)
    }
}

fn bilinear(
    values: &[f32],
    width: u32,
    height: u32,
    x0: isize,
    y0: isize,
    tx: f32,
    ty: f32,
) -> f32 {
    let top_left = sample_clamped(values, width, height, x0, y0);
    let top_right = sample_clamped(values, width, height, x0 + 1, y0);
    let bottom_left = sample_clamped(values, width, height, x0, y0 + 1);
    let bottom_right = sample_clamped(values, width, height, x0 + 1, y0 + 1);
    let top = top_left * (1.0 - tx) + top_right * tx;
    let bottom = bottom_left * (1.0 - tx) + bottom_right * tx;
    top * (1.0 - ty) + bottom * ty
}

fn color_luma(color: [u8; 3]) -> f32 {
    (BT601_R * f32::from(color[0]) + BT601_G * f32::from(color[1]) + BT601_B * f32::from(color[2]))
        / 255.0
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage};

    use crate::tone::ToneMap;

    use super::{ProcessedImage, gaussian_blur, preprocess, smooth_edge_directions};

    #[test]
    fn bt601_red_and_green_have_expected_luminance() {
        let mut red = GrayImage::new(1, 1);
        red.put_pixel(0, 0, Luma([0]));
        let red = DynamicImage::ImageLuma8(red);
        let processed = preprocess(&red, 1.0, 0.0, ToneMap::Global).unwrap();
        assert!(processed.luma[0].abs() < 1.0e-6);

        let rgb = DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 1, Rgb([255, 0, 0])));
        let processed = preprocess(&rgb, 1.0, 0.0, ToneMap::Global).unwrap();
        assert!((processed.luma[0] - 0.299).abs() < 1.0e-4);
    }

    #[test]
    fn display_luminance_ignores_tone_mapping_and_edges() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 4, Rgb([128, 128, 128])));
        let processed = preprocess(&image, 3.0, 1.0, ToneMap::Clahe).unwrap();
        let expected = 128.0 / 255.0;
        assert!((processed.display_luma[0] - expected).abs() < 1.0e-6);
        assert!((processed.cell_luminance(1, 1, 0, 0) - expected).abs() < 1.0e-6);
        assert!((processed.luma[0] - expected).abs() > 1.0e-3);
    }

    #[test]
    fn separable_gaussian_preserves_constant_signal() {
        let values = vec![0.42; 7 * 5];
        let blurred = gaussian_blur(&values, 7, 5, 1.6);
        assert!(blurred.iter().all(|value| (*value - 0.42).abs() < 1.0e-5));
    }

    /// 24 个交替法向的朴素平均恰好为零（25 个像素无法对半分，余下一个弱像素）；
    /// 双角度平滑后所有像素都应指向 x 轴。
    #[test]
    fn direction_smoothing_keeps_opposing_normals_aligned() {
        let (width, height) = (5, 5);
        let mut normal_x = vec![0.0_f32; (width * height) as usize];
        let mut normal_y = vec![0.0_f32; (width * height) as usize];
        let mut edge_strength = vec![1.0_f32; (width * height) as usize];
        for (index, value) in normal_x.iter_mut().enumerate() {
            *value = if index % 2 == 0 { 1.0 } else { -1.0 };
        }
        let weak = (width * height / 2) as usize;
        normal_x[weak] = 0.0;
        edge_strength[weak] = 0.0;

        smooth_edge_directions(&mut normal_x, &mut normal_y, &edge_strength, width, height);

        for (nx, ny) in normal_x.into_iter().zip(normal_y) {
            // 只查 |nx| 会放过未平滑的反相原值，这里要求统一指向 +x。
            assert!(nx > 0.9, "nx={nx}");
            assert!(ny.abs() < 0.1, "ny={ny}");
        }
    }

    /// 交替的正交法向在双角度域互相抵消，中心区域相干度接近零，应被置零。
    #[test]
    fn checkerboard_orthogonal_normals_are_zeroed_after_smoothing() {
        let (width, height) = (16, 16);
        let mut normal_x = vec![0.0_f32; (width * height) as usize];
        let mut normal_y = vec![0.0_f32; (width * height) as usize];
        let edge_strength = vec![1.0_f32; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) as usize;
                if (x + y) % 2 == 0 {
                    normal_x[index] = 1.0;
                } else {
                    normal_y[index] = 1.0;
                }
            }
        }

        smooth_edge_directions(&mut normal_x, &mut normal_y, &edge_strength, width, height);

        for y in 6..10 {
            for x in 6..10 {
                let index = (y * width + x) as usize;
                assert_eq!(normal_x[index], 0.0, "x={x}, y={y}");
                assert_eq!(normal_y[index], 0.0, "x={x}, y={y}");
            }
        }
    }

    #[test]
    fn resampling_uses_double_angle_domain() {
        // 相邻像素法向接近 +90° 和 -90°，直接插值单位法向会得到近水平向量。
        let angle = 89.8_f32.to_radians();
        let normal_x = angle.cos();
        let normal_y = angle.sin();
        let processed = ProcessedImage {
            width: 2,
            height: 1,
            luma: vec![0.5; 2],
            display_luma: vec![0.5; 2],
            normal_x: vec![normal_x, normal_x],
            normal_y: vec![normal_y, -normal_y],
            edge_strength: vec![1.0; 2],
            colors: vec![[128, 128, 128]; 2],
        };

        let work = processed.resample(3, 1).unwrap();
        let middle = work.direction_cos2[1];
        let sin2 = work.direction_sin2[1];
        assert!(middle < -0.99, "cos2={middle}");
        assert!(sin2.abs() < 0.01, "sin2={sin2}");

        let theta = 0.5 * sin2.atan2(middle);
        assert!(theta.cos().abs() < 0.1, "nx={}", theta.cos());
        assert!(theta.sin() > 0.99, "ny={}", theta.sin());
    }

    #[test]
    fn resampling_preserves_constant_luminance() {
        let image = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            4,
            4,
            image::Rgb([128, 128, 128]),
        ));
        let processed = preprocess(&image, 1.0, 0.0, ToneMap::Global).unwrap();
        let work = processed.resample(20, 40).unwrap();
        assert!(
            work.luma
                .iter()
                .all(|value| (*value - work.luma[0]).abs() < 1.0e-6)
        );
        assert!(work.edge_strength.iter().all(|value| *value == 0.0));
    }
}
