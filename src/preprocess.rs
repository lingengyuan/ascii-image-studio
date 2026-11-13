use anyhow::{Result, ensure};
use image::DynamicImage;

use crate::tone::{ToneMap, apply as apply_tone_map};

pub const CELL_WIDTH: u32 = 10;
pub const CELL_HEIGHT: u32 = 20;
/// 形状特征按 4x8 子块采样，每块 2.5x2.5 像素。
pub const SHAPE_COLS: usize = 4;
pub const SHAPE_ROWS: usize = 8;
pub const SHAPE_DIMS: usize = SHAPE_COLS * SHAPE_ROWS;

const BT601_R: f32 = 0.299;
const BT601_G: f32 = 0.587;
const BT601_B: f32 = 0.114;
const SIGNAL_EPSILON: f32 = 1.0e-6;
const DOG_NARROW_SIGMA: f32 = 0.7;
const DOG_WIDE_SIGMA: f32 = 1.6;
/// 有符号 DoG 的固定增益，把锐化强度与 Sobel 方向特征分开。
const UNSHARP_GAIN: f32 = 1.8;

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
    pub normal_x: Vec<f32>,
    pub normal_y: Vec<f32>,
    pub edge_strength: Vec<f32>,
}

/// 子块在单元格内的相对边界，单位是像素，可能落在半像素上。
pub fn shape_bounds(component: usize) -> (f32, f32, f32, f32) {
    let sub_col = (component % SHAPE_COLS) as f32;
    let sub_row = (component / SHAPE_COLS) as f32;
    let width = CELL_WIDTH as f32 / SHAPE_COLS as f32;
    let height = CELL_HEIGHT as f32 / SHAPE_ROWS as f32;
    let x0 = sub_col * width;
    let y0 = sub_row * height;
    (x0, y0, x0 + width, y0 + height)
}

/// 子块面积相等，平均覆盖率就是单元格的整体墨水密度。
pub fn shape_density(shape: [f32; SHAPE_DIMS]) -> f32 {
    shape.iter().sum::<f32>() / SHAPE_DIMS as f32
}

/// 按像素实际覆盖面积加权的区域均值。子块边界落在半像素上，直接取整会偏 0.5 像素。
pub fn fractional_mean(
    values: &[f32],
    width: usize,
    height: usize,
    bounds: (f32, f32, f32, f32),
) -> f32 {
    let (x0, y0, x1, y1) = bounds;
    let x_start = x0.floor().max(0.0) as usize;
    let x_end = (x1.ceil().min(width as f32) as usize).max(x_start);
    let y_start = y0.floor().max(0.0) as usize;
    let y_end = (y1.ceil().min(height as f32) as usize).max(y_start);

    let mut sum = 0.0;
    let mut weight = 0.0;
    for y in y_start..y_end {
        let overlap_y = (y as f32 + 1.0).min(y1) - (y as f32).max(y0);
        for x in x_start..x_end {
            let overlap_x = (x as f32 + 1.0).min(x1) - (x as f32).max(x0);
            let area = overlap_x * overlap_y;
            if area <= 0.0 {
                continue;
            }
            sum += values[y * width + x] * area;
            weight += area;
        }
    }
    sum / weight.max(f32::EPSILON)
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
        apply_unsharp_mask(&mut luma, &signed_dog, edge_weight);
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
    let narrow = gaussian_blur(luma, width, height, DOG_NARROW_SIGMA);
    let wide = gaussian_blur(luma, width, height, DOG_WIDE_SIGMA);
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

    (normal_x, normal_y, magnitudes, dog)
}

/// 有符号 DoG 非锐化掩模：亮侧更亮、暗侧更暗。
///
/// 不能把非负梯度幅值加到亮度上，那会填亮瞳孔、眼线和唇缝。
fn apply_unsharp_mask(luma: &mut [f32], dog: &[f32], edge_weight: f32) {
    for (value, dog) in luma.iter_mut().zip(dog.iter()) {
        *value = (*value + edge_weight * UNSHARP_GAIN * dog).clamp(0.0, 1.0);
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
    pub fn resample(&self, width: u32, height: u32) -> Result<WorkImage> {
        ensure!(width > 0 && height > 0, "工作图尺寸必须大于 0");
        let pixel_count = usize::try_from(u64::from(width) * u64::from(height))
            .map_err(|_| anyhow::anyhow!("工作图尺寸超出可处理范围"))?;
        let mut luma = Vec::with_capacity(pixel_count);
        let mut normal_x = Vec::with_capacity(pixel_count);
        let mut normal_y = Vec::with_capacity(pixel_count);
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
                normal_x.push(bilinear(
                    &self.normal_x,
                    self.width,
                    self.height,
                    x0,
                    y0,
                    tx,
                    ty,
                ));
                normal_y.push(bilinear(
                    &self.normal_y,
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
            normal_x,
            normal_y,
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

    use super::{apply_unsharp_mask, edge_maps, gaussian_blur, preprocess};

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
    fn signed_unsharp_mask_brightens_light_side_and_darkens_dark_side() {
        let mut luma = vec![0.2_f32; 9];
        luma[4] = 0.9;
        let original = luma.clone();
        let (_, _, _, dog) = edge_maps(&original, 9, 1);
        apply_unsharp_mask(&mut luma, &dog, 1.0);
        assert!(luma[4] >= original[4], "亮侧不应被压暗");
        assert!(luma[2] < original[2], "暗侧应被压暗");
        assert!(luma[6] < original[6], "暗侧应被压暗");
        assert!(
            luma.iter().all(|value| (0.0..=1.0).contains(value)),
            "锐化结果必须保持有界"
        );

        let flat = vec![0.42_f32; 7 * 5];
        let (_, _, _, flat_dog) = edge_maps(&flat, 7, 5);
        assert!(flat_dog.iter().all(|value| value.abs() < 1.0e-6));
    }

    #[test]
    fn separable_gaussian_preserves_constant_signal() {
        let values = vec![0.42; 7 * 5];
        let blurred = gaussian_blur(&values, 7, 5, 1.6);
        assert!(blurred.iter().all(|value| (*value - 0.42).abs() < 1.0e-5));
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
