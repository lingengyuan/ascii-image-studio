use std::sync::Arc;

use anyhow::{Result, ensure};
use rayon::prelude::*;

use crate::font_atlas::{BRAILLE_BAYER, BRAILLE_LAYOUT, CharGlyph, FontAtlas, braille_char};
use crate::preprocess::{
    CELL_HEIGHT, CELL_WIDTH, ProcessedImage, WorkImage, cell_region_bounds, shape_density,
};

const DIRECTION_PRIOR_WEIGHT: f32 = 0.04;
const DIRECTION_PRIOR_RAMP_START: f32 = 0.35;
const DIRECTION_PRIOR_RAMP_END: f32 = 0.70;
const DIRECTION_MIN_COHERENCE: f32 = 0.25;
/// 归一化强度会掩盖低对比度图，这里再用原始 Sobel 幅值卡一次绝对下限。
/// 实测固定样张的扩散活跃单元从 0.51 起；12 张 holdout 最高 0.564，取 0.57
/// 让 holdout 全部回退到各向同性 FS，固定样张只关掉 0.51 到 0.57 的窄带。
const DIRECTION_MIN_ABSOLUTE_EDGE: f32 = 0.57;

// 扩散斜坡起点低于选字先验，让中等强度边缘也参与；上端沿用先验。
const DIFFUSION_RAMP_START: f32 = 0.29;
const DIFFUSION_RAMP_END: f32 = DIRECTION_PRIOR_RAMP_END;
const FLOYD_STEINBERG_WEIGHTS: [f32; 4] = [7.0 / 16.0, 3.0 / 16.0, 5.0 / 16.0, 1.0 / 16.0];
const FLOYD_STEINBERG_OFFSETS: [(f32, f32); 4] = [
    (CELL_WIDTH as f32, 0.0),
    (-(CELL_WIDTH as f32), CELL_HEIGHT as f32),
    (0.0, CELL_HEIGHT as f32),
    (CELL_WIDTH as f32, CELL_HEIGHT as f32),
];

const HORIZONTAL_CANDIDATES: &[char] = &['-', '_', '~'];
const BACKSLASH_CANDIDATES: &[char] = &['\\', '\'', '`', ','];
const VERTICAL_CANDIDATES: &[char] = &['|', '!', 'l', 'I', '(', ')'];
const SLASH_CANDIDATES: &[char] = &['/', '(', ')', '\'', '`'];

#[derive(Clone, Debug)]
pub struct MatchConfig {
    pub cols: u32,
    pub dir_crunch: f32,
    pub edge_weight: f32,
    pub dither: bool,
    pub color: bool,
    pub braille: bool,
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
                    let direction = cell_direction(work, config, col, row);
                    build_ascii_cell(
                        image,
                        atlas,
                        config,
                        CellPosition { col, row, rows },
                        enhanced_samples(work, col, row, config.dir_crunch),
                        direction,
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
            // 与无抖动路径同一份方向增强结果；抖动只改密度，不改方向。
            let direction = cell_direction(work, config, col, row);
            let enhanced = enhanced_samples(work, col, row, config.dir_crunch);
            let target_density = (shape_density(enhanced) + errors[index]).clamp(0.0, 1.0);
            let adjusted = scale_to_density(enhanced, target_density);
            let cell = build_ascii_cell(
                image,
                atlas,
                config,
                CellPosition { col, row, rows },
                adjusted,
                direction,
            );
            diffuse_scalar_error(
                config.cols,
                rows,
                col,
                row,
                target_density - cell.density,
                diffusion_weights(direction),
                &mut errors,
            );
            cells.push(cell);
        }
    }
    cells
}

/// `enhanced` 已做过方向增强，这里不再 crunch，抖动路径才能与无抖动路径同形。
fn build_ascii_cell(
    image: &ProcessedImage,
    atlas: &FontAtlas,
    config: &MatchConfig,
    position: CellPosition,
    enhanced: [f32; 6],
    direction: Option<CellDirection>,
) -> AsciiCell {
    let glyph = nearest_glyph(
        atlas.ascii(),
        enhanced,
        direction_prior_from(direction.map(|direction| direction.edge)).as_ref(),
    );

    AsciiCell {
        ch: glyph.ch,
        fg_color: foreground_color(image, config, position),
        density: shape_density(glyph.shape_vec),
        bitmap: glyph.bitmap.clone(),
    }
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
    target: [f32; 6],
    prior: Option<&DirectionPrior>,
) -> &'a CharGlyph {
    glyphs
        .iter()
        .min_by(|left, right| {
            glyph_cost(left, target, prior).total_cmp(&glyph_cost(right, target, prior))
        })
        .expect("ASCII atlas is not empty")
}

fn glyph_cost(glyph: &CharGlyph, target: [f32; 6], prior: Option<&DirectionPrior>) -> f32 {
    let mut cost = distance_squared(glyph.shape_vec, target);
    if let Some(prior) = prior
        && !prior.candidates.contains(&glyph.ch)
    {
        cost += prior.weight;
    }
    cost
}

fn distance_squared(left: [f32; 6], right: [f32; 6]) -> f32 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum()
}

fn dir_crunch(raw: [f32; 6], external: [f32; 6], exponent: f32) -> [f32; 6] {
    std::array::from_fn(|component| {
        let numerator = raw[component].max(0.0);
        let denominator = raw[component].max(external[component]).max(0.0);
        if denominator <= 1.0e-6 {
            0.0
        } else {
            ((numerator / denominator).powf(exponent) * numerator).clamp(0.0, 1.0)
        }
    })
}

/// 方向增强只做一次：无抖动与抖动两条路径都用这个结果选字。
fn enhanced_samples(work: &WorkImage, col: u32, row: u32, exponent: f32) -> [f32; 6] {
    dir_crunch(
        internal_samples(work, col, row),
        external_samples(work, col, row),
        exponent,
    )
}

fn internal_samples(work: &WorkImage, col: u32, row: u32) -> [f32; 6] {
    std::array::from_fn(|component| {
        let bounds = cell_region_bounds(col, row, component);
        mean_region(work, bounds)
    })
}

fn external_samples(work: &WorkImage, col: u32, row: u32) -> [f32; 6] {
    std::array::from_fn(|component| {
        let bounds = external_bounds(work, col, row, component);
        mean_region(work, bounds)
    })
}

fn braille_samples(work: &WorkImage, col: u32, row: u32) -> [f32; 8] {
    std::array::from_fn(|component| {
        let (dot_row, dot_col) = BRAILLE_LAYOUT[component];
        let x0 = col * CELL_WIDTH + dot_col as u32 * 5;
        let y0 = row * CELL_HEIGHT + dot_row as u32 * 5;
        mean_region(work, (x0, y0, x0 + 5, y0 + 5))
    })
}

fn external_bounds(work: &WorkImage, col: u32, row: u32, component: usize) -> (u32, u32, u32, u32) {
    let sub_col = (component % 2) as u32;
    let sub_row = (component / 2) as u32;
    let cell_x = col * CELL_WIDTH;
    let cell_y = row * CELL_HEIGHT;
    let (x0, x1) = if sub_col == 0 {
        (cell_x as isize - 5, cell_x as isize)
    } else {
        (
            (cell_x + CELL_WIDTH) as isize,
            (cell_x + CELL_WIDTH + 5) as isize,
        )
    };
    let (y0, y1) = match sub_row {
        0 => (cell_y as isize - 6, cell_y as isize),
        1 => ((cell_y + 6) as isize, (cell_y + 13) as isize),
        _ => (
            (cell_y + CELL_HEIGHT) as isize,
            (cell_y + CELL_HEIGHT + 7) as isize,
        ),
    };
    let x0 = x0.clamp(0, work.width as isize) as u32;
    let x1 = x1.clamp(0, work.width as isize) as u32;
    let y0 = y0.clamp(0, work.height as isize) as u32;
    let y1 = y1.clamp(0, work.height as isize) as u32;
    if x0 < x1 && y0 < y1 {
        (x0, y0, x1, y1)
    } else {
        cell_region_bounds(col, row, component)
    }
}

fn mean_region(work: &WorkImage, bounds: (u32, u32, u32, u32)) -> f32 {
    let (x0, y0, x1, y1) = bounds;
    let mut sum = 0.0;
    let mut count = 0_u32;
    for y in y0..y1 {
        for x in x0..x1 {
            sum += work.luma[y as usize * work.width as usize + x as usize];
            count += 1;
        }
    }
    sum / count.max(1) as f32
}

#[derive(Clone, Copy, Debug)]
struct EdgeDirection {
    normal_x: f32,
    normal_y: f32,
    strength: f32,
}

#[derive(Clone, Copy, Debug)]
struct CellDirection {
    edge: EdgeDirection,
    /// 绝对强度过门限才允许结构感知扩散；先验仍按归一化强度，与 HEAD 一致。
    diffuses: bool,
}

/// edge_weight 为 0 时不算方向，保持无先验、无扩散偏置的路径。
fn cell_direction(
    work: &WorkImage,
    config: &MatchConfig,
    col: u32,
    row: u32,
) -> Option<CellDirection> {
    if config.edge_weight <= 0.0 {
        return None;
    }
    let edge = edge_direction_components(work, col, row)?;
    Some(CellDirection {
        edge,
        diffuses: absolute_edge_is_real(work, edge.strength),
    })
}

/// 低对比度图的归一化强度仍可能接近 1，用原始 Sobel 幅值判断边缘是否真实。
fn absolute_edge_is_real(work: &WorkImage, strength: f32) -> bool {
    strength * work.edge_scale >= DIRECTION_MIN_ABSOLUTE_EDGE
}

/// 单元内也按双角度聚合。相干度是双角度能量与边缘强度之和的比值：
/// 只有方向一致时它才接近 1，方向互相抵消的低相干单元没有先验。
fn edge_direction_components(work: &WorkImage, col: u32, row: u32) -> Option<EdgeDirection> {
    let x0 = col * CELL_WIDTH;
    let y0 = row * CELL_HEIGHT;
    let mut sum_cos = 0.0;
    let mut sum_sin = 0.0;
    let mut strength_sum = 0.0;
    for y in y0..y0 + CELL_HEIGHT {
        for x in x0..x0 + CELL_WIDTH {
            let index = y as usize * work.width as usize + x as usize;
            let edge = work.edge_strength[index];
            sum_cos += work.direction_cos2[index] * edge;
            sum_sin += work.direction_sin2[index] * edge;
            strength_sum += edge;
        }
    }
    if strength_sum <= 1.0e-6 {
        return None;
    }
    let energy = sum_cos.hypot(sum_sin);
    if energy / strength_sum < DIRECTION_MIN_COHERENCE {
        return None;
    }
    let strength = strength_sum / (CELL_WIDTH * CELL_HEIGHT) as f32;
    let theta = 0.5 * sum_sin.atan2(sum_cos);
    Some(EdgeDirection {
        normal_x: theta.cos(),
        normal_y: theta.sin(),
        strength,
    })
}

#[derive(Clone, Copy)]
struct DirectionPrior {
    candidates: &'static [char],
    weight: f32,
}

/// 方向先验只软性加罚：强边缘才生效，弱边缘仍由 6D 形状决定。
fn direction_prior_from(direction: Option<EdgeDirection>) -> Option<DirectionPrior> {
    let direction = direction?;
    let ramp = ((direction.strength - DIRECTION_PRIOR_RAMP_START)
        / (DIRECTION_PRIOR_RAMP_END - DIRECTION_PRIOR_RAMP_START))
        .clamp(0.0, 1.0);
    let weight = DIRECTION_PRIOR_WEIGHT * ramp;
    if weight <= 0.0 {
        return None;
    }
    let candidates = directional_candidates(direction.normal_x, direction.normal_y)?;
    Some(DirectionPrior { candidates, weight })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StrokeBin {
    Horizontal,
    Backslash,
    Vertical,
    Slash,
}

fn stroke_bin(normal_x: f32, normal_y: f32) -> Option<StrokeBin> {
    if !normal_x.is_finite() || !normal_y.is_finite() || normal_x.hypot(normal_y) <= 1.0e-6 {
        return None;
    }
    let stroke_angle =
        (normal_y.atan2(normal_x) + std::f32::consts::FRAC_PI_2).rem_euclid(std::f32::consts::PI);
    let eighth = std::f32::consts::PI / 8.0;
    Some(if stroke_angle < eighth || stroke_angle >= 7.0 * eighth {
        StrokeBin::Horizontal
    } else if stroke_angle < 3.0 * eighth {
        StrokeBin::Backslash
    } else if stroke_angle < 5.0 * eighth {
        StrokeBin::Vertical
    } else {
        StrokeBin::Slash
    })
}

pub fn directional_ascii(normal_x: f32, normal_y: f32) -> Option<char> {
    Some(match stroke_bin(normal_x, normal_y)? {
        StrokeBin::Horizontal => '-',
        StrokeBin::Backslash => '\\',
        StrokeBin::Vertical => '|',
        StrokeBin::Slash => '/',
    })
}

fn directional_candidates(normal_x: f32, normal_y: f32) -> Option<&'static [char]> {
    Some(match stroke_bin(normal_x, normal_y)? {
        StrokeBin::Horizontal => HORIZONTAL_CANDIDATES,
        StrokeBin::Backslash => BACKSLASH_CANDIDATES,
        StrokeBin::Vertical => VERTICAL_CANDIDATES,
        StrokeBin::Slash => SLASH_CANDIDATES,
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
    let level = (0.25 + 0.75 * luminance.clamp(0.0, 1.0)) * 255.0;
    [level.round().clamp(0.0, 255.0) as u8; 3]
}

/// 把 6D 形状缩放到目标密度并保持方向。零向量没有方向，按面积权重均分。
///
/// 缩放只有一个比例，所以分量之间的比值不变。有分量碰到 1.0 时整体停止放大，
/// 达不到的密度留给调用方当标量残差扩散；逐分量截断会改变方向，不能这么做。
fn scale_to_density(shape: [f32; 6], target_density: f32) -> [f32; 6] {
    let target = target_density.clamp(0.0, 1.0);
    let current = shape_density(shape);
    if current <= 1.0e-6 {
        return [target; 6];
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

/// 强边缘上把 FS 误差沿边缘扩散：垂直边缘的右邻权重趋近 0，水平边缘的下邻权重趋近 0。
fn diffusion_weights(direction: Option<CellDirection>) -> [f32; 4] {
    let Some(direction) = direction.filter(|direction| direction.diffuses) else {
        return FLOYD_STEINBERG_WEIGHTS;
    };
    let edge = direction.edge;
    let ramp = ((edge.strength - DIFFUSION_RAMP_START)
        / (DIFFUSION_RAMP_END - DIFFUSION_RAMP_START))
        .clamp(0.0, 1.0);
    if ramp <= 0.0 {
        return FLOYD_STEINBERG_WEIGHTS;
    }
    let mut weights = FLOYD_STEINBERG_WEIGHTS;
    for (weight, (dx, dy)) in weights.iter_mut().zip(FLOYD_STEINBERG_OFFSETS) {
        let length = dx.hypot(dy);
        let along = 1.0 - ((dx * edge.normal_x + dy * edge.normal_y) / length).powi(2);
        *weight *= (1.0 - ramp) + ramp * along;
    }
    let sum: f32 = weights.iter().sum();
    if sum <= 1.0e-6 {
        return FLOYD_STEINBERG_WEIGHTS;
    }
    weights.map(|weight| weight / sum)
}

fn diffuse_scalar_error(
    cols: u32,
    rows: u32,
    col: u32,
    row: u32,
    error: f32,
    weights: [f32; 4],
    errors: &mut [f32],
) {
    if col + 1 < cols {
        errors[cell_index(cols, col + 1, row)] += error * weights[0];
    }
    if row + 1 < rows {
        if col > 0 {
            errors[cell_index(cols, col - 1, row + 1)] += error * weights[1];
        }
        errors[cell_index(cols, col, row + 1)] += error * weights[2];
        if col + 1 < cols {
            errors[cell_index(cols, col + 1, row + 1)] += error * weights[3];
        }
    }
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgb, RgbImage};

    use super::{
        BACKSLASH_CANDIDATES, CellDirection, CellPosition, DIFFUSION_RAMP_START,
        DIRECTION_PRIOR_WEIGHT, DirectionPrior, EdgeDirection, FLOYD_STEINBERG_WEIGHTS,
        HORIZONTAL_CANDIDATES, MatchConfig, SLASH_CANDIDATES, VERTICAL_CANDIDATES,
        build_ascii_cell, cell_direction, diffuse_scalar_error, diffusion_weights, dir_crunch,
        direction_prior_from, directional_ascii, directional_candidates, distance_squared,
        enhanced_samples, external_bounds, external_samples, internal_samples, match_image,
        mean_region, nearest_glyph, output_rows, scale_to_density,
    };
    use crate::font_atlas::FontAtlas;
    use crate::preprocess::{
        CELL_HEIGHT, CELL_WIDTH, ProcessedImage, WorkImage, cell_region_bounds, preprocess,
        shape_density,
    };
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
            dir_crunch: 2.5,
            edge_weight: 0.25,
            dither,
            color: false,
            braille: false,
        }
    }

    /// 亮暗竖条对齐单元边界：右邻域比内部亮、左邻域更暗，外部样本逐分量不均匀。
    fn striped_image(width: u32, height: u32) -> DynamicImage {
        let mut image = RgbImage::new(width, height);
        for y in 0..height {
            let base = 0.35 + 0.3 * y as f32 / height as f32;
            for x in 0..width {
                let level = if x % CELL_WIDTH < CELL_WIDTH / 2 {
                    0.9
                } else {
                    0.1
                };
                let value = (level * base * 255.0).round() as u8;
                image.put_pixel(x, y, Rgb([value, value, value]));
            }
        }
        DynamicImage::ImageRgb8(image)
    }

    /// 2 像素亮暗交替的竖条，每个工作单元都横跨多条边缘。
    fn vertical_stripes(width: u32, height: u32) -> DynamicImage {
        let mut image = RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let value = if (x / 2) % 2 == 0 { 235 } else { 20 };
                image.put_pixel(x, y, Rgb([value, value, value]));
            }
        }
        DynamicImage::ImageRgb8(image)
    }

    /// 只按顺序拼装现有元件：先 crunch，再等比缩放模长，最后选字。
    fn reference_serial_chars(
        image: &ProcessedImage,
        atlas: &FontAtlas,
        config: &MatchConfig,
    ) -> Vec<char> {
        let rows = output_rows(config.cols, image.width, image.height).unwrap();
        let work = image
            .resample(config.cols * CELL_WIDTH, rows * CELL_HEIGHT)
            .unwrap();
        let mut errors = vec![0.0_f32; (config.cols * rows) as usize];
        let mut chars = Vec::new();
        for row in 0..rows {
            for col in 0..config.cols {
                let index = (row * config.cols + col) as usize;
                let direction = cell_direction(&work, config, col, row);
                let enhanced = enhanced_samples(&work, col, row, config.dir_crunch);
                let target = (shape_density(enhanced) + errors[index]).clamp(0.0, 1.0);
                let cell = build_ascii_cell(
                    image,
                    atlas,
                    config,
                    CellPosition { col, row, rows },
                    scale_to_density(enhanced, target),
                    direction,
                );
                diffuse_scalar_error(
                    config.cols,
                    rows,
                    col,
                    row,
                    target - cell.density,
                    diffusion_weights(direction),
                    &mut errors,
                );
                chars.push(cell.ch);
            }
        }
        chars
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
    fn directional_candidates_follow_the_same_stroke_bins() {
        assert_eq!(
            directional_candidates(0.0, 1.0),
            Some(HORIZONTAL_CANDIDATES)
        );
        assert_eq!(
            directional_candidates(
                std::f32::consts::FRAC_1_SQRT_2,
                -std::f32::consts::FRAC_1_SQRT_2
            ),
            Some(BACKSLASH_CANDIDATES)
        );
        assert_eq!(directional_candidates(1.0, 0.0), Some(VERTICAL_CANDIDATES));
        assert_eq!(
            directional_candidates(
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2
            ),
            Some(SLASH_CANDIDATES)
        );
        assert_eq!(directional_candidates(0.0, 0.0), None);
    }

    #[test]
    fn diffusion_weights_fall_back_to_floyd_steinberg() {
        let base = [7.0 / 16.0, 3.0 / 16.0, 5.0 / 16.0, 1.0 / 16.0];
        assert_eq!(diffusion_weights(None), base);
        let weak = CellDirection {
            edge: EdgeDirection {
                normal_x: 1.0,
                normal_y: 0.0,
                strength: DIFFUSION_RAMP_START * 0.5,
            },
            diffuses: true,
        };
        assert_eq!(diffusion_weights(Some(weak)), base);

        let gated = CellDirection {
            edge: EdgeDirection {
                normal_x: 1.0,
                normal_y: 0.0,
                strength: 1.0,
            },
            diffuses: false,
        };
        assert_eq!(diffusion_weights(Some(gated)), base);
    }

    #[test]
    fn diffusion_weights_follow_a_vertical_edge() {
        let weights = diffusion_weights(Some(CellDirection {
            edge: EdgeDirection {
                normal_x: 1.0,
                normal_y: 0.0,
                strength: 1.0,
            },
            diffuses: true,
        }));
        assert!(weights[0] < 1.0e-6, "right={}", weights[0]);
        assert!(weights[2] > 5.0 / 16.0, "bottom={}", weights[2]);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn diffusion_weights_follow_a_horizontal_edge() {
        let weights = diffusion_weights(Some(CellDirection {
            edge: EdgeDirection {
                normal_x: 0.0,
                normal_y: 1.0,
                strength: 1.0,
            },
            diffuses: true,
        }));
        assert!(weights[2] < 1.0e-6, "bottom={}", weights[2]);
        assert!(weights[0] > 7.0 / 16.0, "right={}", weights[0]);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn zero_weight_prior_does_not_change_the_chosen_glyph() {
        let atlas = FontAtlas::load().unwrap();
        let prior = DirectionPrior {
            candidates: VERTICAL_CANDIDATES,
            weight: 0.0,
        };
        for target in [[0.0; 6], [0.5; 6], [1.0; 6], [0.2, 0.8, 0.4, 0.1, 0.6, 0.3]] {
            assert_eq!(
                nearest_glyph(atlas.ascii(), target, Some(&prior)).ch,
                nearest_glyph(atlas.ascii(), target, None).ch
            );
        }
    }

    #[test]
    fn large_weight_prior_selects_a_directional_candidate() {
        let atlas = FontAtlas::load().unwrap();
        // 6D 距离平方最大为 6，权重 10 足以压过任何非候选字。
        let prior = DirectionPrior {
            candidates: HORIZONTAL_CANDIDATES,
            weight: 10.0,
        };
        let glyph = nearest_glyph(atlas.ascii(), [0.5; 6], Some(&prior));
        assert!(HORIZONTAL_CANDIDATES.contains(&glyph.ch), "ch={}", glyph.ch);
    }

    fn uniform_work_cell(direction_cos2: f32, edge_strength: f32) -> WorkImage {
        let pixel_count = (CELL_WIDTH * CELL_HEIGHT) as usize;
        WorkImage {
            width: CELL_WIDTH,
            height: CELL_HEIGHT,
            edge_scale: 1.0,
            luma: vec![0.0; pixel_count],
            direction_cos2: vec![direction_cos2; pixel_count],
            direction_sin2: vec![0.0; pixel_count],
            edge_strength: vec![edge_strength; pixel_count],
        }
    }

    #[test]
    fn incoherent_cell_has_no_prior() {
        let mut incoherent = uniform_work_cell(1.0, 1.0);
        let half = incoherent.direction_cos2.len() / 2;
        incoherent.direction_cos2[half..].fill(-1.0);
        let cell = cell_direction(&incoherent, &config(false), 0, 0);
        assert!(cell.is_none());
        assert!(direction_prior_from(cell.map(|cell| cell.edge)).is_none());

        // 60/40 反相的绝对和不为零，旧的绝对和检查会误判为有方向。
        let mut mostly_positive = uniform_work_cell(1.0, 1.0);
        let sixty_percent = mostly_positive.direction_cos2.len() * 3 / 5;
        mostly_positive.direction_cos2[sixty_percent..].fill(-1.0);
        assert!(cell_direction(&mostly_positive, &config(false), 0, 0).is_none());

        let coherent = uniform_work_cell(1.0, 1.0);
        let cell = cell_direction(&coherent, &config(false), 0, 0);
        assert!(cell.is_some());
        assert!(direction_prior_from(cell.map(|cell| cell.edge)).is_some());
    }

    #[test]
    fn low_contrast_cell_does_not_diffuse() {
        let mut low_contrast = uniform_work_cell(1.0, 1.0);
        low_contrast.edge_scale = 0.005;
        let cell = cell_direction(&low_contrast, &config(false), 0, 0)
            .expect("归一化强度足够，仍应得到方向");
        assert!(!cell.diffuses);
        assert_eq!(
            diffusion_weights(Some(cell)),
            FLOYD_STEINBERG_WEIGHTS,
            "低对比度单元应回退到各向同性 FS"
        );
        // 先验只按归一化强度计算，低对比度图仍与 HEAD 一样获得先验。
        assert!(direction_prior_from(Some(cell.edge)).is_some());

        let normal_contrast = uniform_work_cell(1.0, 1.0);
        let cell =
            cell_direction(&normal_contrast, &config(false), 0, 0).expect("正常对比度应得到方向");
        assert!(cell.diffuses);
    }

    #[test]
    fn real_weight_prior_allows_non_candidate_shape_to_win() {
        let atlas = FontAtlas::load().unwrap();
        let work = uniform_work_cell(1.0, 1.0);
        let cell = cell_direction(&work, &config(false), 0, 0).expect("相干单元应有方向");
        let prior = direction_prior_from(Some(cell.edge)).expect("相干单元应有先验");
        assert_eq!(prior.candidates, VERTICAL_CANDIDATES);
        assert!(
            (prior.weight - DIRECTION_PRIOR_WEIGHT).abs() < 1.0e-6,
            "weight={}",
            prior.weight
        );

        let target = atlas
            .ascii()
            .iter()
            .find(|glyph| glyph.ch == 'A')
            .expect("atlas contains A")
            .shape_vec;
        assert!(!VERTICAL_CANDIDATES.contains(&'A'));
        assert_eq!(nearest_glyph(atlas.ascii(), target, Some(&prior)).ch, 'A');
    }

    /// 方向先验只软性加罚，形状完全匹配的非候选字 'A' 仍应胜出。
    #[test]
    fn build_ascii_cell_keeps_a_better_non_candidate_shape() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(10, 20, Rgb([128, 128, 128])));
        let processed = preprocess(&image, 1.0, 0.0, ToneMap::default()).unwrap();
        let atlas = FontAtlas::load().unwrap();
        let work = WorkImage {
            width: 10,
            height: 20,
            edge_scale: 1.0,
            luma: vec![0.5; 200],
            direction_cos2: vec![1.0; 200],
            direction_sin2: vec![0.0; 200],
            edge_strength: vec![1.0; 200],
        };
        let config = MatchConfig {
            cols: 1,
            dir_crunch: 2.5,
            edge_weight: 0.25,
            dither: false,
            color: false,
            braille: false,
        };
        let target = atlas
            .ascii()
            .iter()
            .find(|glyph| glyph.ch == 'A')
            .expect("atlas contains A")
            .shape_vec;

        let direction = cell_direction(&work, &config, 0, 0);
        let cell = build_ascii_cell(
            &processed,
            &atlas,
            &config,
            CellPosition {
                col: 0,
                row: 0,
                rows: 1,
            },
            target,
            direction,
        );

        assert_eq!(cell.ch, 'A');
    }

    #[test]
    fn external_samples_use_regions_touching_the_cell_edge() {
        let mut work = WorkImage {
            width: 30,
            height: 40,
            edge_scale: 1.0,
            luma: vec![0.0; 30 * 40],
            direction_cos2: vec![0.0; 30 * 40],
            direction_sin2: vec![0.0; 30 * 40],
            edge_strength: vec![0.0; 30 * 40],
        };
        for y in 6..13 {
            for x in 10..15 {
                work.luma[y * work.width as usize + x] = 0.8;
            }
        }
        let bounds = external_bounds(&work, 0, 0, 3);
        assert_eq!(bounds, (10, 6, 15, 13));
        assert!((mean_region(&work, bounds) - 0.8).abs() < 1.0e-6);

        let edge_work = WorkImage {
            width: 10,
            height: 20,
            edge_scale: 1.0,
            luma: vec![0.3; 10 * 20],
            direction_cos2: vec![0.0; 10 * 20],
            direction_sin2: vec![0.0; 10 * 20],
            edge_strength: vec![0.0; 10 * 20],
        };
        assert_eq!(
            external_bounds(&edge_work, 0, 0, 0),
            cell_region_bounds(0, 0, 0)
        );
    }

    #[test]
    fn external_brightness_suppresses_only_the_matching_direction() {
        let raw = [0.2; 6];
        let mut external = [0.0; 6];
        external[3] = 0.8;
        let enhanced = dir_crunch(raw, external, 2.5);
        assert!(enhanced[3] < enhanced[2]);
        assert!((enhanced[2] - 0.2).abs() < 1.0e-6);
    }

    #[test]
    fn six_dimensional_distance_uses_same_normalized_units() {
        let left = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let right = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        assert_eq!(distance_squared(left, right), 0.0);
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
        let raw = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let scaled = scale_to_density(raw, 0.3);
        assert!((shape_density(scaled) - 0.3).abs() < 1.0e-6);
        for component in 1..6 {
            assert!((scaled[component] / scaled[0] - raw[component] / raw[0]).abs() < 1.0e-6);
        }

        let zero = scale_to_density([0.0; 6], 0.25);
        assert_eq!(zero, [0.25; 6]);
        assert!((shape_density(zero) - 0.25).abs() < 1.0e-6);

        let saturated = scale_to_density([0.9, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
        assert_eq!(saturated[0], 1.0);
        assert!(
            saturated
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        );
    }

    #[test]
    fn saturation_scales_every_component_by_one_factor() {
        let raw = [0.75, 0.5, 0.25, 0.2, 0.1, 0.05];
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
    fn scaling_before_crunch_would_move_the_suppressed_direction() {
        let raw = [0.3, 0.06, 0.3, 0.06, 0.3, 0.06];
        let external = [0.0, 0.3, 0.0, 0.3, 0.0, 0.3];
        let enhanced = dir_crunch(raw, external, 2.5);
        let scaled = scale_to_density(enhanced, 0.35);
        for component in 1..6 {
            let ratio = scaled[component] / scaled[0];
            let reference = enhanced[component] / enhanced[0];
            assert!((ratio - reference).abs() < 1.0e-6, "ratio={ratio}");
        }

        let reordered = dir_crunch(scale_to_density(raw, 0.35), external, 2.5);
        let reordered_ratio = reordered[1] / reordered[0];
        let kept_ratio = scaled[1] / scaled[0];
        assert!(
            reordered_ratio > 2.0 * kept_ratio,
            "先缩放再 crunch 会降低外部抑制：{reordered_ratio} vs {kept_ratio}"
        );
    }

    #[test]
    fn dither_rescales_the_crunched_shape_on_uneven_external_neighborhoods() {
        let image = striped_image(40, 40);
        let processed = preprocess(&image, 1.0, 0.0, ToneMap::Global).unwrap();
        let atlas = FontAtlas::load().unwrap();
        let mut conf = config(true);
        conf.edge_weight = 0.0;

        // 前置条件：确实有分量碰上了更亮的相邻外部，dir_crunch 的阈值会介入。
        let rows = output_rows(conf.cols, processed.width, processed.height).unwrap();
        let work = processed
            .resample(conf.cols * CELL_WIDTH, rows * CELL_HEIGHT)
            .unwrap();
        let suppressed: usize = (0..rows)
            .flat_map(|row| (0..conf.cols).map(move |col| (col, row)))
            .map(|(col, row)| {
                let raw = internal_samples(&work, col, row);
                let external = external_samples(&work, col, row);
                raw.into_iter()
                    .zip(external)
                    .filter(|(inside, outside)| outside > inside)
                    .count()
            })
            .sum();
        assert!(suppressed > 0, "样本没有跨过外部阈值，测不到顺序差异");

        let art = match_image(&processed, &atlas, &conf).unwrap();
        let actual: Vec<char> = art.cells.iter().map(|cell| cell.ch).collect();
        assert_eq!(actual, reference_serial_chars(&processed, &atlas, &conf));
    }

    #[test]
    fn serial_dither_uses_structure_aware_weights() {
        let image = vertical_stripes(80, 40);
        let processed = preprocess(&image, 1.0, 0.25, ToneMap::Global).unwrap();
        let atlas = FontAtlas::load().unwrap();
        let conf = MatchConfig {
            cols: 8,
            dir_crunch: 2.5,
            edge_weight: 0.25,
            dither: true,
            color: false,
            braille: false,
        };

        let art = match_image(&processed, &atlas, &conf).unwrap();
        let actual: Vec<char> = art.cells.iter().map(|cell| cell.ch).collect();
        assert_eq!(actual, reference_serial_chars(&processed, &atlas, &conf));

        let rows = output_rows(conf.cols, processed.width, processed.height).unwrap();
        let work = processed
            .resample(conf.cols * CELL_WIDTH, rows * CELL_HEIGHT)
            .unwrap();
        let anisotropic_cells = (0..rows)
            .flat_map(|row| (0..conf.cols).map(move |col| (col, row)))
            .filter(|&(col, row)| {
                cell_direction(&work, &conf, col, row).is_some_and(|direction| {
                    diffusion_weights(Some(direction)) != FLOYD_STEINBERG_WEIGHTS
                })
            })
            .count();
        assert!(anisotropic_cells > 0, "竖条纹没有触发结构感知扩散");
    }

    #[test]
    fn mono_foreground_uses_cell_luminance_instead_of_white() {
        let atlas = FontAtlas::load().unwrap();
        for (value, expected) in [(0_u8, 64_u8), (128, 160), (255, 255)] {
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
