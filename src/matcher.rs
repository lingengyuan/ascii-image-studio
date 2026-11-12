use std::sync::Arc;

use anyhow::{Result, ensure};
use rayon::prelude::*;

use crate::font_atlas::{BRAILLE_LAYOUT, CharGlyph, FontAtlas, braille_char};
use crate::preprocess::{
    CELL_HEIGHT, CELL_WIDTH, ProcessedImage, WorkImage, cell_region_bounds, shape_density,
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
        match_braille(image, atlas, config, &work, rows, cell_count)
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
                    let raw = internal_samples(work, col, row);
                    build_ascii_cell(
                        image,
                        atlas,
                        config,
                        work,
                        CellPosition { col, row, rows },
                        raw,
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
            let raw = internal_samples(work, col, row);
            let adjusted =
                std::array::from_fn(|component| (raw[component] + errors[index]).clamp(0.0, 1.0));
            let target_density = shape_density(adjusted);
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
    raw: [f32; 6],
) -> AsciiCell {
    let external = external_samples(work, position.col, position.row);
    let enhanced = dir_crunch(raw, external, config.dir_crunch);
    let mut glyph = nearest_glyph(atlas.ascii(), enhanced);

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

fn match_braille(
    image: &ProcessedImage,
    atlas: &FontAtlas,
    config: &MatchConfig,
    work: &WorkImage,
    rows: u32,
    cell_count: usize,
) -> Vec<AsciiCell> {
    if !config.dither {
        let rows_of_cells: Vec<Vec<AsciiCell>> = (0..rows)
            .into_par_iter()
            .map(|row| {
                (0..config.cols)
                    .map(|col| {
                        let samples = braille_samples(work, col, row);
                        build_braille_cell(
                            image,
                            atlas,
                            config,
                            CellPosition { col, row, rows },
                            samples,
                        )
                    })
                    .collect()
            })
            .collect();
        return rows_of_cells.into_iter().flatten().collect();
    }

    let mut cells = Vec::with_capacity(cell_count);
    let mut errors = vec![0.0_f32; cell_count];
    for row in 0..rows {
        for col in 0..config.cols {
            let index = cell_index(config.cols, col, row);
            let samples = braille_samples(work, col, row);
            let adjusted = samples.map(|sample| (sample + errors[index]).clamp(0.0, 1.0));
            let target_density = adjusted.iter().sum::<f32>() / 8.0;
            let cell = build_braille_cell(
                image,
                atlas,
                config,
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

fn build_braille_cell(
    image: &ProcessedImage,
    atlas: &FontAtlas,
    config: &MatchConfig,
    position: CellPosition,
    samples: [f32; 8],
) -> AsciiCell {
    let mut bits = 0_u8;
    for (bit, sample) in samples.into_iter().enumerate() {
        if sample >= 0.5 {
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

fn nearest_glyph(glyphs: &[CharGlyph], target: [f32; 6]) -> &CharGlyph {
    glyphs
        .iter()
        .min_by(|left, right| {
            distance_squared(left.shape_vec, target)
                .total_cmp(&distance_squared(right.shape_vec, target))
        })
        .expect("ASCII atlas is not empty")
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
        image.cell_highlight_color(config.cols, position.rows, position.col, position.row)
    } else {
        [255, 255, 255]
    }
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
        MatchConfig, dir_crunch, directional_ascii, distance_squared, external_bounds, match_image,
        mean_region,
    };
    use crate::font_atlas::FontAtlas;
    use crate::preprocess::{WorkImage, cell_region_bounds, preprocess};

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
    fn external_samples_use_regions_touching_the_cell_edge() {
        let mut work = WorkImage {
            width: 30,
            height: 40,
            luma: vec![0.0; 30 * 40],
            normal_x: vec![0.0; 30 * 40],
            normal_y: vec![0.0; 30 * 40],
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
            luma: vec![0.3; 10 * 20],
            normal_x: vec![0.0; 10 * 20],
            normal_y: vec![0.0; 10 * 20],
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
        let processed = preprocess(&image, 1.0, 0.0).unwrap();
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
        let processed = preprocess(&image, 1.0, 0.0).unwrap();
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
        let processed = preprocess(&image, 1.0, 0.0).unwrap();
        let atlas = FontAtlas::load().unwrap();
        let first = match_image(&processed, &atlas, &config(false)).unwrap();
        let second = match_image(&processed, &atlas, &config(false)).unwrap();
        assert_eq!(
            first.cells.iter().map(|cell| cell.ch).collect::<Vec<_>>(),
            second.cells.iter().map(|cell| cell.ch).collect::<Vec<_>>()
        );
    }

    #[test]
    fn braille_matching_emits_only_real_braille_scalars() {
        let image = gradient_image(32, 32);
        let processed = preprocess(&image, 1.0, 0.0).unwrap();
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
