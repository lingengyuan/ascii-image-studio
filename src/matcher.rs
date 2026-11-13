use std::sync::Arc;

use anyhow::{Result, ensure};
use rayon::prelude::*;

use crate::font_atlas::{BRAILLE_BAYER, BRAILLE_LAYOUT, CharGlyph, FontAtlas, braille_char};
use crate::preprocess::{
    CELL_HEIGHT, CELL_WIDTH, ProcessedImage, SHAPE_DIMS, WorkImage, fractional_mean, shape_bounds,
    shape_density,
};

const DIRECTION_EDGE_THRESHOLD: f32 = 0.55;

#[derive(Clone, Debug)]
pub struct MatchConfig {
    pub cols: u32,
    pub dir_crunch: f32,
    pub edge_weight: f32,
    pub dither: bool,
    pub color: bool,
    pub braille: bool,
    /// 纯文本没有前景明度调制，匹配时按满亮度覆盖计算。
    pub plain_text: bool,
}

#[derive(Clone, Debug)]
pub struct AsciiCell {
    pub ch: char,
    pub fg_color: [u8; 3],
    pub density: f32,
    pub bitmap: Arc<[u8]>,
}

pub struct MatchedArt {
    pub width: u32,
    pub height: u32,
    pub cols: u32,
    pub rows: u32,
    pub cells: Vec<AsciiCell>,
}

pub fn match_image(
    image: &ProcessedImage,
    atlas: &FontAtlas,
    config: &MatchConfig,
) -> Result<MatchedArt> {
    ensure!(config.cols > 0, "宽度必须大于 0");
    ensure!(
        config.dir_crunch.is_finite() && config.dir_crunch > 0.0,
        "dir_crunch 必须是有限正数"
    );
    ensure!(
        config.edge_weight.is_finite() && (0.0..=1.0).contains(&config.edge_weight),
        "edge_weight 必须在 0 到 1 之间"
    );
    ensure!(image.width > 0 && image.height > 0, "输入图片不能为空");

    let rows = output_rows(config.cols, image.width, image.height)?;
    let width = config
        .cols
        .checked_mul(CELL_WIDTH)
        .ok_or_else(|| anyhow::anyhow!("输出图片宽度溢出"))?;
    let height = rows
        .checked_mul(CELL_HEIGHT)
        .ok_or_else(|| anyhow::anyhow!("输出图片高度溢出"))?;
    let cell_count = usize::try_from(u64::from(config.cols) * u64::from(rows))
        .map_err(|_| anyhow::anyhow!("字符网格尺寸溢出"))?;
    let work = image.resample(width, height)?;
    let cells = if config.braille {
        match_braille(image, atlas, config, &work, rows)
    } else if config.dither {
        match_ascii_serial(image, atlas, config, &work, rows, cell_count)
    } else {
        match_ascii_parallel(image, atlas, config, &work, rows)?
    };

    Ok(MatchedArt {
        width,
        height,
        cols: config.cols,
        rows,
        cells,
    })
}

fn output_rows(cols: u32, image_width: u32, image_height: u32) -> Result<u32> {
    let rows = f64::from(cols) * f64::from(image_height) / f64::from(image_width) * 0.5;
    ensure!(rows.is_finite(), "输出行数不是有限值");
    let rows = rows.round().max(1.0);
    ensure!(rows <= f64::from(u32::MAX), "输出行数溢出");
    Ok(rows as u32)
}

fn match_ascii_parallel(
    image: &ProcessedImage,
    atlas: &FontAtlas,
    config: &MatchConfig,
    work: &WorkImage,
    rows: u32,
) -> Result<Vec<AsciiCell>> {
    let rows_of_cells: Vec<Vec<AsciiCell>> = (0..rows)
        .into_par_iter()
        .map(|row| {
            (0..config.cols)
                .map(|col| {
                    build_ascii_cell(
                        image,
                        atlas,
                        config,
                        work,
                        CellPosition { col, row, rows },
                        internal_samples(work, col, row),
                    )
                })
                .collect()
        })
        .collect();
    Ok(rows_of_cells.into_iter().flatten().collect())
}

fn match_ascii_serial(
    image: &ProcessedImage,
    atlas: &FontAtlas,
    config: &MatchConfig,
    work: &WorkImage,
    rows: u32,
    cell_count: usize,
) -> Vec<AsciiCell> {
    let mut cells = Vec::with_capacity(cell_count);
    let mut errors = vec![0.0_f32; cell_count];

    for row in 0..rows {
        for col in 0..config.cols {
            let index = cell_index(config.cols, col, row);
            // 与无抖动路径同一份子块采样；抖动只改密度，不改形状。
            let samples = internal_samples(work, col, row);
            let target_density = (shape_density(samples) + errors[index]).clamp(0.0, 1.0);
            let adjusted = scale_to_density(samples, target_density);
            let cell = build_ascii_cell(
                image,
                atlas,
                config,
                work,
                CellPosition { col, row, rows },
                adjusted,
            );
            diffuse_scalar_error(
                config.cols,
                rows,
                col,
                row,
                target_density - cell.density,
                &mut errors,
            );
            cells.push(cell);
        }
    }
    cells
}

fn build_ascii_cell(
    image: &ProcessedImage,
    atlas: &FontAtlas,
    config: &MatchConfig,
    work: &WorkImage,
    position: CellPosition,
    target: [f32; SHAPE_DIMS],
) -> AsciiCell {
    let terms = MatchingTerms::for_cell(
        image,
        config,
        work,
        position,
        &target,
        atlas.density_scale(),
    );
    let mut glyph = nearest_glyph(atlas.ascii(), &target, &terms);

    if config.edge_weight > 0.0
        && let Some(direction) = edge_direction(work, position.col, position.row)
        && direction.strength >= DIRECTION_EDGE_THRESHOLD
        && let Some(directional) = directional_ascii(direction.normal_x, direction.normal_y)
    {
        glyph = &atlas.ascii()[directional as usize - 0x20];
    }

    AsciiCell {
        ch: glyph.ch,
        fg_color: foreground_color(image, config, position),
        density: shape_density(glyph.shape_vec),
        bitmap: glyph.bitmap.clone(),
    }
}

/// 匹配时使用的亮度与权重。
struct MatchingTerms {
    /// 单元格目标亮度。
    y_mean: f32,
    /// 前景颜色亮度系数；渲染出的字形覆盖率为 alpha * shape。
    alpha: f32,
    /// 形状距离相对密度误差的权重，边缘处提高、平坦处降低。
    shape_weight: f32,
    /// 字形形状密度到 0..1 的缩放，最密字形为 1。
    density_scale: f32,
}

impl MatchingTerms {
    fn for_cell(
        image: &ProcessedImage,
        config: &MatchConfig,
        work: &WorkImage,
        position: CellPosition,
        target: &[f32; SHAPE_DIMS],
        density_scale: f32,
    ) -> Self {
        if config.plain_text {
            let y_mean = target.iter().sum::<f32>() / SHAPE_DIMS as f32;
            return Self {
                y_mean,
                alpha: 1.0,
                shape_weight: config.dir_crunch * edge_gain(work, position.col, position.row),
                density_scale,
            };
        }

        // 与 renderer 的前景颜色用同一份亮度，匹配结果和实际渲染才不会错位。
        let y_mean = image.cell_luminance(config.cols, position.rows, position.col, position.row);
        Self {
            y_mean,
            alpha: foreground_alpha(y_mean),
            shape_weight: config.dir_crunch * edge_gain(work, position.col, position.row),
            density_scale,
        }
    }
}

/// 形状权重按单元格内最强边缘缩放：强边缘保留结构，平坦皮肤压低结构以抑制噪点。
fn edge_gain(work: &WorkImage, col: u32, row: u32) -> f32 {
    let mut peak = 0.0_f32;
    for y in row * CELL_HEIGHT..(row + 1) * CELL_HEIGHT {
        for x in col * CELL_WIDTH..(col + 1) * CELL_WIDTH {
            peak = peak.max(work.edge_strength[y as usize * work.width as usize + x as usize]);
        }
    }
    0.5 + 0.5 * peak.clamp(0.0, 1.0)
}

fn match_braille(
    image: &ProcessedImage,
    atlas: &FontAtlas,
    config: &MatchConfig,
    work: &WorkImage,
    rows: u32,
) -> Vec<AsciiCell> {
    let rows_of_cells: Vec<Vec<AsciiCell>> = (0..rows)
        .into_par_iter()
        .map(|row| {
            (0..config.cols)
                .map(|col| {
                    build_braille_cell(
                        image,
                        atlas,
                        config,
                        CellPosition { col, row, rows },
                        braille_samples(work, col, row),
                        config.dither,
                    )
                })
                .collect()
        })
        .collect();
    rows_of_cells.into_iter().flatten().collect()
}

fn build_braille_cell(
    image: &ProcessedImage,
    atlas: &FontAtlas,
    config: &MatchConfig,
    position: CellPosition,
    samples: [f32; 8],
    ordered_dither: bool,
) -> AsciiCell {
    let mut bits = 0_u8;
    for (bit, sample) in samples.into_iter().enumerate() {
        let threshold = if ordered_dither {
            let (dot_row, dot_col) = BRAILLE_LAYOUT[bit];
            (f32::from(BRAILLE_BAYER[dot_row][dot_col]) + 0.5) / 8.0
        } else {
            0.5
        };
        if sample >= threshold {
            bits |= 1 << bit;
        }
    }
    AsciiCell {
        ch: braille_char(bits),
        fg_color: foreground_color(image, config, position),
        density: bits.count_ones() as f32 / 8.0,
        bitmap: atlas.braille_bitmap(bits),
    }
}

fn nearest_glyph<'a>(
    glyphs: &'a [CharGlyph],
    target: &[f32; SHAPE_DIMS],
    terms: &MatchingTerms,
) -> &'a CharGlyph {
    glyphs
        .iter()
        .min_by(|left, right| {
            glyph_cost(left, target, terms).total_cmp(&glyph_cost(right, target, terms))
        })
        .expect("ASCII atlas is not empty")
}

/// 渲染亮度是 `alpha` 倍的字形覆盖率，按调制后的形状比较才能消除中间调二次压暗。
fn glyph_cost(glyph: &CharGlyph, target: &[f32; SHAPE_DIMS], terms: &MatchingTerms) -> f32 {
    let distance: f32 = target
        .iter()
        .zip(glyph.shape_vec.iter())
        .map(|(target, shape)| (*target - terms.alpha * *shape).powi(2))
        .sum();
    let density = shape_density(glyph.shape_vec) * terms.density_scale;
    terms.shape_weight * distance + 1.5 * (terms.y_mean - terms.alpha * density).powi(2)
}

fn internal_samples(work: &WorkImage, col: u32, row: u32) -> [f32; SHAPE_DIMS] {
    std::array::from_fn(|component| {
        let (x0, y0, x1, y1) = shape_bounds(component);
        let cell_x = (col * CELL_WIDTH) as f32;
        let cell_y = (row * CELL_HEIGHT) as f32;
        mean_region(work, (cell_x + x0, cell_y + y0, cell_x + x1, cell_y + y1))
    })
}

fn braille_samples(work: &WorkImage, col: u32, row: u32) -> [f32; 8] {
    std::array::from_fn(|component| {
        let (dot_row, dot_col) = BRAILLE_LAYOUT[component];
        let x0 = (col * CELL_WIDTH + dot_col as u32 * 5) as f32;
        let y0 = (row * CELL_HEIGHT + dot_row as u32 * 5) as f32;
        mean_region(work, (x0, y0, x0 + 5.0, y0 + 5.0))
    })
}

fn mean_region(work: &WorkImage, bounds: (f32, f32, f32, f32)) -> f32 {
    fractional_mean(
        &work.luma,
        work.width as usize,
        work.height as usize,
        bounds,
    )
}

#[derive(Debug)]
struct EdgeDirection {
    normal_x: f32,
    normal_y: f32,
    strength: f32,
}

fn edge_direction(work: &WorkImage, col: u32, row: u32) -> Option<EdgeDirection> {
    let x0 = col * CELL_WIDTH;
    let y0 = row * CELL_HEIGHT;
    let mut weighted_x = 0.0;
    let mut weighted_y = 0.0;
    let mut strength = 0.0;
    for y in y0..y0 + CELL_HEIGHT {
        for x in x0..x0 + CELL_WIDTH {
            let index = y as usize * work.width as usize + x as usize;
            let edge = work.edge_strength[index];
            weighted_x += work.normal_x[index] * edge;
            weighted_y += work.normal_y[index] * edge;
            strength += edge;
        }
    }
    strength /= (CELL_WIDTH * CELL_HEIGHT) as f32;
    let length = weighted_x.hypot(weighted_y);
    (length > 1.0e-6).then(|| EdgeDirection {
        normal_x: weighted_x / length,
        normal_y: weighted_y / length,
        strength,
    })
}

pub fn directional_ascii(normal_x: f32, normal_y: f32) -> Option<char> {
    if !normal_x.is_finite() || !normal_y.is_finite() || normal_x.hypot(normal_y) <= 1.0e-6 {
        return None;
    }
    let stroke_angle =
        (normal_y.atan2(normal_x) + std::f32::consts::FRAC_PI_2).rem_euclid(std::f32::consts::PI);
    let eighth = std::f32::consts::PI / 8.0;
    Some(if stroke_angle < eighth || stroke_angle >= 7.0 * eighth {
        '-'
    } else if stroke_angle < 3.0 * eighth {
        '\\'
    } else if stroke_angle < 5.0 * eighth {
        '|'
    } else {
        '/'
    })
}

fn foreground_color(
    image: &ProcessedImage,
    config: &MatchConfig,
    position: CellPosition,
) -> [u8; 3] {
    if config.color {
        return image.cell_highlight_color(config.cols, position.rows, position.col, position.row);
    }
    let luminance = image.cell_luminance(config.cols, position.rows, position.col, position.row);
    let level = foreground_alpha(luminance) * 255.0;
    [level.round().clamp(0.0, 255.0) as u8; 3]
}

/// 黑白模式的前景亮度系数。匹配和渲染必须用同一份，否则中间调仍会偏暗。
fn foreground_alpha(luminance: f32) -> f32 {
    0.25 + 0.75 * luminance.clamp(0.0, 1.0).powf(0.85)
}

/// 把 32D 形状缩放到目标密度并保持结构。零向量没有结构，按均匀覆盖填充。
///
/// 缩放只有一个比例，所以分量之间的比值不变。有分量碰到 1.0 时整体停止放大，
/// 达不到的密度留给调用方当标量残差扩散；逐分量截断会改变结构，不能这么做。
fn scale_to_density(shape: [f32; SHAPE_DIMS], target_density: f32) -> [f32; SHAPE_DIMS] {
    let target = target_density.clamp(0.0, 1.0);
    let current = shape_density(shape);
    if current <= 1.0e-6 {
        return [target; SHAPE_DIMS];
    }
    let peak = shape.into_iter().fold(0.0_f32, f32::max);
    let mut scale = target / current;
    if peak > 1.0e-6 {
        scale = scale.min(1.0 / peak);
    }
    shape.map(|value| value * scale)
}

#[derive(Clone, Copy)]
struct CellPosition {
    col: u32,
    row: u32,
    rows: u32,
}

fn cell_index(cols: u32, col: u32, row: u32) -> usize {
    row as usize * cols as usize + col as usize
}

fn diffuse_scalar_error(cols: u32, rows: u32, col: u32, row: u32, error: f32, errors: &mut [f32]) {
    if col + 1 < cols {
        errors[cell_index(cols, col + 1, row)] += error * 7.0 / 16.0;
    }
    if row + 1 < rows {
        if col > 0 {
            errors[cell_index(cols, col - 1, row + 1)] += error * 3.0 / 16.0;
        }
        errors[cell_index(cols, col, row + 1)] += error * 5.0 / 16.0;
        if col + 1 < cols {
            errors[cell_index(cols, col + 1, row + 1)] += error / 16.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgb, RgbImage};

    use super::{
        MatchConfig, MatchedArt, directional_ascii, internal_samples, match_image, mean_region,
        scale_to_density,
    };
    use crate::font_atlas::FontAtlas;
    use crate::preprocess::{SHAPE_DIMS, WorkImage, preprocess, shape_bounds, shape_density};
    use crate::tone::ToneMap;

    fn gradient_image(width: u32, height: u32) -> DynamicImage {
        let mut image = RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let value = ((x + y) * 255 / (width + height - 2).max(1)) as u8;
                image.put_pixel(x, y, Rgb([value, value, value]));
            }
        }
        DynamicImage::ImageRgb8(image)
    }

    fn config(dither: bool) -> MatchConfig {
        MatchConfig {
            cols: 16,
            dir_crunch: 1.4,
            edge_weight: 0.25,
            dither,
            color: false,
            braille: false,
            plain_text: true,
        }
    }

    fn work_image(width: u32, height: u32, luma: Vec<f32>) -> WorkImage {
        let zeros = vec![0.0; luma.len()];
        WorkImage {
            width,
            height,
            luma,
            normal_x: zeros.clone(),
            normal_y: zeros.clone(),
            edge_strength: zeros,
        }
    }

    #[test]
    fn directional_characters_use_stroke_perpendicular_to_gradient() {
        assert_eq!(directional_ascii(1.0, 0.0), Some('|'));
        assert_eq!(directional_ascii(0.0, 1.0), Some('-'));
        assert_eq!(
            directional_ascii(
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2
            ),
            Some('/')
        );
        assert_eq!(
            directional_ascii(
                std::f32::consts::FRAC_1_SQRT_2,
                -std::f32::consts::FRAC_1_SQRT_2
            ),
            Some('\\')
        );
        assert_eq!(directional_ascii(0.0, 0.0), None);
    }

    #[test]
    fn shape_bounds_split_cell_into_four_by_eight_blocks() {
        assert_eq!(shape_bounds(0), (0.0, 0.0, 2.5, 2.5));
        assert_eq!(shape_bounds(3), (7.5, 0.0, 10.0, 2.5));
        assert_eq!(shape_bounds(SHAPE_DIMS - 1), (7.5, 17.5, 10.0, 20.0));
    }

    #[test]
    fn internal_samples_resolve_half_pixel_vertical_positions() {
        // 上半亮、下半暗：8 行子块应逐行分档，旧的 2x3 网格只有 3 档。
        let mut luma = vec![0.0_f32; 10 * 20];
        for y in 0..20 {
            for x in 0..10 {
                luma[y * 10 + x] = if y < 10 { 0.9 } else { 0.1 };
            }
        }
        let work = work_image(10, 20, luma);
        let samples = internal_samples(&work, 0, 0);
        for row in 0..8 {
            let mean = (0..4).map(|col| samples[row * 4 + col]).sum::<f32>() / 4.0;
            let expected = if row < 4 { 0.9 } else { 0.1 };
            assert!((mean - expected).abs() < 1.0e-5, "row={row} mean={mean}");
        }
    }

    #[test]
    fn fractional_mean_weights_partial_pixels() {
        let work = work_image(2, 1, vec![0.0, 1.0]);
        let mean = mean_region(&work, (0.0, 0.0, 1.5, 1.0));
        assert!((mean - 1.0 / 3.0).abs() < 1.0e-6, "mean={mean}");
    }

    #[test]
    fn dithering_changes_matching_and_serial_output_is_deterministic() {
        let image = gradient_image(32, 32);
        let processed = preprocess(&image, 1.0, 0.0, ToneMap::Global).unwrap();
        let atlas = FontAtlas::load().unwrap();
        let plain = match_image(&processed, &atlas, &config(false)).unwrap();
        let dithered = match_image(&processed, &atlas, &config(true)).unwrap();
        let repeated = match_image(&processed, &atlas, &config(true)).unwrap();
        assert!(
            plain
                .cells
                .iter()
                .zip(&dithered.cells)
                .any(|(left, right)| left.ch != right.ch)
        );
        assert!(
            dithered
                .cells
                .iter()
                .zip(&repeated.cells)
                .all(|(left, right)| left.ch == right.ch)
        );
    }

    #[test]
    fn mid_gray_dither_keeps_average_error_in_shape_density_units() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 32, Rgb([128, 128, 128])));
        let processed = preprocess(&image, 1.0, 0.0, ToneMap::Global).unwrap();
        let atlas = FontAtlas::load().unwrap();
        let art = match_image(&processed, &atlas, &config(true)).unwrap();
        let target = 128.0_f32 / 255.0;
        let mean_density =
            art.cells.iter().map(|cell| cell.density).sum::<f32>() / art.cells.len() as f32;
        let error = (mean_density - target).abs();
        assert!(
            error < 0.08,
            "mean error={error}, mean density={mean_density}"
        );
        assert!(mean_density < 0.75);
    }

    #[test]
    fn parallel_matching_is_deterministic() {
        let image = gradient_image(32, 32);
        let processed = preprocess(&image, 1.0, 0.0, ToneMap::Global).unwrap();
        let atlas = FontAtlas::load().unwrap();
        let first = match_image(&processed, &atlas, &config(false)).unwrap();
        let second = match_image(&processed, &atlas, &config(false)).unwrap();
        assert_eq!(
            first.cells.iter().map(|cell| cell.ch).collect::<Vec<_>>(),
            second.cells.iter().map(|cell| cell.ch).collect::<Vec<_>>()
        );
    }

    #[test]
    fn density_scaling_keeps_direction_and_handles_zero_and_saturation() {
        let raw: [f32; SHAPE_DIMS] = std::array::from_fn(|index| 0.1 + 0.02 * index as f32);
        let scaled = scale_to_density(raw, 0.3);
        assert!((shape_density(scaled) - 0.3).abs() < 1.0e-6);
        for component in 1..SHAPE_DIMS {
            assert!((scaled[component] / scaled[0] - raw[component] / raw[0]).abs() < 1.0e-6);
        }

        let zero = scale_to_density([0.0; SHAPE_DIMS], 0.25);
        assert_eq!(zero, [0.25; SHAPE_DIMS]);
        assert!((shape_density(zero) - 0.25).abs() < 1.0e-6);

        let mut saturated_raw = [0.0; SHAPE_DIMS];
        saturated_raw[0] = 0.9;
        let saturated = scale_to_density(saturated_raw, 1.0);
        assert_eq!(saturated[0], 1.0);
        assert!(
            saturated
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        );
    }

    #[test]
    fn saturation_scales_every_component_by_one_factor() {
        let raw: [f32; SHAPE_DIMS] = std::array::from_fn(|index| 0.75 - 0.02 * index as f32);
        let scaled = scale_to_density(raw, 1.0);
        let cap = 1.0 / raw[0];
        for (value, source) in scaled.into_iter().zip(raw) {
            assert!((value - source * cap).abs() < 1.0e-6, "value={value}");
            assert!((0.0..=1.0).contains(&value));
        }
        assert!((scaled[0] - 1.0).abs() < 1.0e-6);
        let reached = shape_density(scaled);
        assert!((reached - shape_density(raw) * cap).abs() < 1.0e-6);
        assert!(reached < 1.0, "饱和后目标密度达不到，残差留给扩散");
    }

    #[test]
    fn modulated_matching_compensates_for_foreground_brightness() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(40, 40, Rgb([128, 128, 128])));
        let processed = preprocess(&image, 1.0, 0.0, ToneMap::Global).unwrap();
        let atlas = FontAtlas::load().unwrap();
        let mut modulated = config(false);
        modulated.plain_text = false;

        let mean_density = |art: &MatchedArt| {
            art.cells.iter().map(|cell| cell.density).sum::<f32>() / art.cells.len() as f32
        };
        let modulated_density = mean_density(&match_image(&processed, &atlas, &modulated).unwrap());
        let plain_density = mean_density(&match_image(&processed, &atlas, &config(false)).unwrap());
        assert!(
            modulated_density > plain_density + 0.05,
            "modulated={modulated_density}, plain={plain_density}"
        );
    }

    #[test]
    fn mono_foreground_uses_cell_luminance_instead_of_white() {
        let atlas = FontAtlas::load().unwrap();
        for (value, expected) in [(0_u8, 64_u8), (128, 170), (255, 255)] {
            let image =
                DynamicImage::ImageRgb8(RgbImage::from_pixel(10, 20, Rgb([value, value, value])));
            let processed = preprocess(&image, 1.0, 0.0, ToneMap::Global).unwrap();
            let art = match_image(&processed, &atlas, &config(false)).unwrap();
            assert!(
                art.cells.iter().all(|cell| cell.fg_color == [expected; 3]),
                "value={value}, fg={:?}",
                art.cells[0].fg_color
            );
        }
    }

    #[test]
    fn braille_known_codepoint_uses_top_left_dot_bit() {
        let mut image = RgbImage::from_pixel(10, 20, Rgb([0, 0, 0]));
        for y in 0..5 {
            for x in 0..5 {
                image.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
        let processed =
            preprocess(&DynamicImage::ImageRgb8(image), 1.0, 0.0, ToneMap::Global).unwrap();
        let atlas = FontAtlas::load().unwrap();
        let mut braille = config(false);
        braille.cols = 1;
        braille.braille = true;
        let art = match_image(&processed, &atlas, &braille).unwrap();
        assert_eq!(art.cells.len(), 1);
        assert_eq!(art.cells[0].ch, '\u{2801}');
    }

    #[test]
    fn braille_dither_uses_bayer_thresholds_per_dot() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(40, 40, Rgb([128, 128, 128])));
        let processed = preprocess(&image, 1.0, 0.0, ToneMap::Global).unwrap();
        let atlas = FontAtlas::load().unwrap();
        let mut dithered = config(true);
        dithered.braille = true;
        let art = match_image(&processed, &atlas, &dithered).unwrap();
        assert!(art.cells.iter().all(|cell| cell.density == 0.5));

        let mut plain = dithered.clone();
        plain.dither = false;
        let plain_art = match_image(&processed, &atlas, &plain).unwrap();
        assert!(plain_art.cells.iter().all(|cell| cell.density == 1.0));
    }

    #[test]
    fn braille_matching_emits_only_real_braille_scalars() {
        let image = gradient_image(32, 32);
        let processed = preprocess(&image, 1.0, 0.0, ToneMap::Global).unwrap();
        let atlas = FontAtlas::load().unwrap();
        let mut braille_config = config(false);
        braille_config.braille = true;
        let art = match_image(&processed, &atlas, &braille_config).unwrap();
        assert!(
            art.cells
                .iter()
                .all(|cell| ('\u{2800}'..='\u{28ff}').contains(&cell.ch))
        );
        assert!(art.cells.iter().any(|cell| cell.ch != '\u{2800}'));
    }
}
