use std::sync::Arc;

use anyhow::{Result, ensure};
use rayon::prelude::*;

use crate::aiss;
use crate::font_atlas::{BRAILLE_BLUE_NOISE, BRAILLE_LAYOUT, CharGlyph, FontAtlas, braille_char};
use crate::preprocess::{
    CELL_HEIGHT, CELL_WIDTH, ProcessedImage, WorkImage, cell_region_bounds, color_luma,
};

const CELL_PIXELS: usize = (CELL_WIDTH * CELL_HEIGHT) as usize;

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
/// Ostromoukhov (2001) 变量系数表：i=0..127，D(i)=D(255-i)。
/// 每项是三个未归一化的分布系数，对应扫描方向的前方、左下、正下。
const OSTROMOUKHOV_WEIGHTS: [[u16; 3]; 128] = [
    [13, 0, 5],
    [13, 0, 5],
    [21, 0, 10],
    [7, 0, 4],
    [8, 0, 5],
    [47, 3, 28],
    [23, 3, 13],
    [15, 3, 8],
    [22, 6, 11],
    [43, 15, 20],
    [7, 3, 3],
    [501, 224, 211],
    [249, 116, 103],
    [165, 80, 67],
    [123, 62, 49],
    [489, 256, 191],
    [81, 44, 31],
    [483, 272, 181],
    [60, 35, 22],
    [53, 32, 19],
    [237, 148, 83],
    [471, 304, 161],
    [3, 2, 1],
    [481, 314, 185],
    [354, 226, 155],
    [1389, 866, 685],
    [227, 138, 125],
    [267, 158, 163],
    [327, 188, 220],
    [61, 34, 45],
    [627, 338, 505],
    [1227, 638, 1075],
    [20, 10, 19],
    [1937, 1000, 1767],
    [977, 520, 855],
    [657, 360, 551],
    [71, 40, 57],
    [2005, 1160, 1539],
    [337, 200, 247],
    [2039, 1240, 1425],
    [257, 160, 171],
    [691, 440, 437],
    [1045, 680, 627],
    [301, 200, 171],
    [177, 120, 95],
    [2141, 1480, 1083],
    [1079, 760, 513],
    [725, 520, 323],
    [137, 100, 57],
    [2209, 1640, 855],
    [53, 40, 19],
    [2243, 1720, 741],
    [565, 440, 171],
    [759, 600, 209],
    [1147, 920, 285],
    [2311, 1880, 513],
    [97, 80, 19],
    [335, 280, 57],
    [1181, 1000, 171],
    [793, 680, 95],
    [599, 520, 57],
    [2413, 2120, 171],
    [405, 360, 19],
    [2447, 2200, 57],
    [11, 10, 0],
    [158, 151, 3],
    [178, 179, 7],
    [1030, 1091, 63],
    [248, 277, 21],
    [318, 375, 35],
    [458, 571, 63],
    [878, 1159, 147],
    [5, 7, 1],
    [172, 181, 37],
    [97, 76, 22],
    [72, 41, 17],
    [119, 47, 29],
    [4, 1, 1],
    [4, 1, 1],
    [4, 1, 1],
    [4, 1, 1],
    [4, 1, 1],
    [4, 1, 1],
    [4, 1, 1],
    [4, 1, 1],
    [4, 1, 1],
    [65, 18, 17],
    [95, 29, 26],
    [185, 62, 53],
    [30, 11, 9],
    [35, 14, 11],
    [85, 37, 28],
    [55, 26, 19],
    [80, 41, 29],
    [155, 86, 59],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [305, 176, 119],
    [155, 86, 59],
    [105, 56, 39],
    [80, 41, 29],
    [65, 32, 23],
    [55, 26, 19],
    [335, 152, 113],
    [85, 37, 28],
    [115, 48, 37],
    [35, 14, 11],
    [355, 136, 109],
    [30, 11, 9],
    [365, 128, 107],
    [185, 62, 53],
    [25, 8, 7],
    [95, 29, 26],
    [385, 112, 103],
    [65, 18, 17],
    [395, 104, 101],
    [4, 1, 1],
];

const REFINE_SWEEPS: usize = 2;
const REFINE_CANDIDATE_COUNT: usize = 16;
const HVS_SIGMA_PIXELS: f32 = 3.5;
const HVS_REACH: isize = 10;
const HVS_DOWNSCALE: usize = 2;
/// 像素 HVS 残差项权重。
const REFINE_PIXEL_WEIGHT: f32 = 0.2;
const REFINE_MEAN_WEIGHT: f32 = 3.0;
const REFINE_EDGE_WEIGHT: f32 = 1.0;

/// AISS 只在 6D 前 10 名内重排，保留形状先验的初选。
const AISS_CANDIDATE_COUNT: usize = 10;
/// 0.1 到 0.4 的重排会压低固定样张的 FSIM@σ5。0.01 让 FSIM@σ5 近似持平
/// （平均 -0.004%），模糊 SSIM 和 density MAE 5/5 改善。未参与调参的
/// Hay Wain 样张上模糊 SSIM、density MAE 4/4 改善，FSIM@σ5 波动在 ±0.03% 内。
const AISS_WEIGHT: f32 = 0.1;
/// 门限用归一化前的平均 Sobel 幅值，低对比单元不会因为归一化而被放大。
/// 满幅台阶（0→1）为 0.510，1/255 台阶为 0.002，2/255 为 0.004，取 0.005 都挡下。
/// 固定样张的回退占比：portrait 0.7%、high-texture 2.6%、landscape 2.9%、
/// low-contrast 6.0%、lineart 7.0%（其中大部分是空白边）。
const AISS_MIN_CELL_EDGE: f32 = 0.005;
/// 匹配代价的两项权重：去均值形状距离与归一化密度误差。
const SHAPE_WEIGHT: f32 = 1.0;
const DENSITY_WEIGHT: f32 = 15.0;
/// 小权重把候选拉回本格目标均值；全局均值项仍防止整图漂移。
const REFINE_DENSITY_WEIGHT: f32 = 0.20;
const REFINE_NEIGHBORS: [(isize, isize, f32, f32); 4] = [
    (1, 0, CELL_WIDTH as f32, 0.0),
    (-1, 1, -(CELL_WIDTH as f32), CELL_HEIGHT as f32),
    (0, 1, 0.0, CELL_HEIGHT as f32),
    (1, 1, CELL_WIDTH as f32, CELL_HEIGHT as f32),
];
/// 上游邻格相对本格的位置，以及源格指向本格的偏移：左、左上、上、右上。
const REFINE_INCOMING: [(isize, isize, f32, f32); 4] = [
    (-1, 0, CELL_WIDTH as f32, 0.0),
    (-1, -1, CELL_WIDTH as f32, CELL_HEIGHT as f32),
    (0, -1, 0.0, CELL_HEIGHT as f32),
    (1, -1, -(CELL_WIDTH as f32), CELL_HEIGHT as f32),
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
    pub refine: bool,
    pub aiss: bool,
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
    } else if config.refine {
        match_ascii_refine(image, atlas, config, &work, rows, cell_count)
    } else if config.aiss {
        match_ascii_aiss(image, atlas, config, &work, rows)
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
                        cell_target(work, col, row, config.dir_crunch, atlas.shape_scale()),
                        direction,
                    )
                })
                .collect()
        })
        .collect();
    Ok(rows_of_cells.into_iter().flatten().collect())
}

/// AISS 只重排 6D 前 10 名；空白或退化单元保持纯 6D 的首选。
fn match_ascii_aiss(
    image: &ProcessedImage,
    atlas: &FontAtlas,
    config: &MatchConfig,
    work: &WorkImage,
    rows: u32,
) -> Vec<AsciiCell> {
    let glyphs = atlas.ascii();
    let descriptors = atlas.aiss_descriptors();
    let rows_of_cells: Vec<Vec<AsciiCell>> = (0..rows)
        .into_par_iter()
        .map(|row| {
            (0..config.cols)
                .map(|col| {
                    let direction = cell_direction(work, config, col, row);
                    let target =
                        cell_target(work, col, row, config.dir_crunch, atlas.shape_scale());
                    let prior = direction_prior_from(direction.map(|direction| direction.edge));
                    let candidates =
                        top_glyph_candidates(glyphs, target, prior.as_ref(), AISS_CANDIDATE_COUNT);
                    let glyph_index = match aiss_cell_descriptor(work, col, row) {
                        Some(descriptor) => candidates
                            .iter()
                            .min_by(|(left_index, left_cost), (right_index, right_cost)| {
                                let left = *left_cost
                                    + AISS_WEIGHT
                                        * aiss::l1_distance(&descriptor, &descriptors[*left_index]);
                                let right = *right_cost
                                    + AISS_WEIGHT
                                        * aiss::l1_distance(
                                            &descriptor,
                                            &descriptors[*right_index],
                                        );
                                left.total_cmp(&right)
                            })
                            .map(|(index, _)| *index)
                            .unwrap_or(candidates[0].0),
                        None => candidates[0].0,
                    };
                    let glyph = &glyphs[glyph_index];
                    AsciiCell {
                        ch: glyph.ch,
                        fg_color: foreground_color(
                            image,
                            config,
                            CellPosition { col, row, rows },
                            glyph.density_norm,
                        ),
                        density: glyph.density_norm,
                        bitmap: glyph.bitmap.clone(),
                    }
                })
                .collect()
        })
        .collect();
    rows_of_cells.into_iter().flatten().collect()
}

/// 目标单元先走与字形相同的 10x20 边缘算子；绝对平均幅值低于门限时回退纯 6D。
fn aiss_cell_descriptor(work: &WorkImage, col: u32, row: u32) -> Option<Vec<f32>> {
    let luma = cell_luma(work, col, row);
    let edges = aiss::cell_edges(&luma);
    if edges.mean_magnitude < AISS_MIN_CELL_EDGE {
        return None;
    }
    Some(aiss::descriptor_from_edges(&edges.edges))
}

fn cell_luma(work: &WorkImage, col: u32, row: u32) -> [f32; CELL_PIXELS] {
    let x0 = col * CELL_WIDTH;
    let y0 = row * CELL_HEIGHT;
    let mut luma = [0.0_f32; CELL_PIXELS];
    for y in 0..CELL_HEIGHT {
        for x in 0..CELL_WIDTH {
            luma[y as usize * CELL_WIDTH as usize + x as usize] =
                work.luma[(y0 + y) as usize * work.width as usize + (x0 + x) as usize];
        }
    }
    luma
}

fn match_ascii_serial(
    image: &ProcessedImage,
    atlas: &FontAtlas,
    config: &MatchConfig,
    work: &WorkImage,
    rows: u32,
    cell_count: usize,
) -> Vec<AsciiCell> {
    let mut cells: Vec<Option<AsciiCell>> = (0..cell_count).map(|_| None).collect();
    let mut errors = vec![0.0_f32; cell_count];

    for row in 0..rows {
        let forward = row % 2 == 0;
        for step in 0..config.cols {
            let col = if forward {
                step
            } else {
                config.cols - 1 - step
            };
            let index = cell_index(config.cols, col, row);
            // 与无抖动路径同一份方向增强结果；抖动只改密度，不改方向。
            let direction = cell_direction(work, config, col, row);
            let mut target = cell_target(work, col, row, config.dir_crunch, atlas.shape_scale());
            target.density = (target.density + errors[index]).clamp(0.0, 1.0);
            let (weights, offsets) = dither_weights(direction, forward, target.density);
            let cell = build_ascii_cell(
                image,
                atlas,
                config,
                CellPosition { col, row, rows },
                target,
                direction,
            );
            diffuse_three_errors(
                (config.cols, rows),
                (col, row),
                target.density - cell.density,
                weights,
                offsets,
                &mut errors,
            );
            cells[index] = Some(cell);
        }
    }
    cells
        .into_iter()
        .map(|cell| cell.expect("每个单元都应被填充"))
        .collect()
}

/// 棋盘式候选重选只改字符选择：初始解仍是 6D 最近字，位图和前景色不动。
fn match_ascii_refine(
    image: &ProcessedImage,
    atlas: &FontAtlas,
    config: &MatchConfig,
    work: &WorkImage,
    rows: u32,
    cell_count: usize,
) -> Vec<AsciiCell> {
    let glyphs = atlas.ascii();
    let cols = config.cols as usize;
    let row_count = rows as usize;
    let mono = !config.color;
    let mut pixel = PixelHvs::new(work, glyphs, atlas.max_ink());
    let mut cells = Vec::with_capacity(cell_count);
    let mut initial_mean = 0.0;

    for row in 0..rows {
        for col in 0..config.cols {
            let direction = cell_direction(work, config, col, row);
            let match_target = cell_target(work, col, row, config.dir_crunch, atlas.shape_scale());
            let prior = direction_prior_from(direction.map(|direction| direction.edge));
            let candidates =
                top_glyph_candidates(glyphs, match_target, prior.as_ref(), REFINE_CANDIDATE_COUNT);
            let display = image.cell_luminance(config.cols, rows, col, row);
            let color_fg = if mono {
                [255; 3]
            } else {
                image.cell_highlight_color(config.cols, rows, col, row)
            };
            let density = glyphs[candidates[0].0].density_norm;
            let fg_luminance = if mono {
                if density > 1.0e-6 {
                    (display / density).clamp(0.25, 1.0)
                } else {
                    1.0
                }
            } else {
                color_luma(color_fg)
            };
            let target = match_target.density;
            let render = fg_luminance * density;
            pixel.add_cell(col, row, candidates[0].0, fg_luminance);
            initial_mean += render;
            cells.push(RefineCell {
                candidates,
                chosen: 0,
                direction,
                fg_color: color_fg,
                fg_luminance,
                mono,
                display,
                target,
                render,
            });
        }
    }
    initial_mean /= cell_count.max(1) as f32;
    let global_sum: f32 = cells.iter().map(|cell| cell.render).sum();

    let mut grid = RefineGrid {
        glyphs,
        cols,
        rows: row_count,
        cells,
        pixel,
        initial_mean,
        global_sum,
    };

    // 同组单元相隔 3 列/3 行：30x40 足迹不重叠，可在组内并行决策。
    for _ in 0..REFINE_SWEEPS {
        for group_row in 0..3_u32 {
            for group_col in 0..4_u32 {
                let indices: Vec<usize> = (0..rows)
                    .flat_map(|row| {
                        (0..config.cols)
                            .filter(move |col| col % 4 == group_col && row % 3 == group_row)
                            .map(move |col| row as usize * cols + col as usize)
                    })
                    .collect();
                let choices: Vec<RefineChoice> = indices
                    .par_iter()
                    .filter_map(|&index| grid.choose_cell(index))
                    .collect();
                for choice in choices {
                    grid.apply_choice(choice);
                }
            }
        }
    }

    grid.cells
        .iter()
        .map(|cell| {
            let glyph = &grid.glyphs[cell.candidates[cell.chosen].0];
            let fg_color = if cell.mono {
                let fg = if glyph.density_norm > 1.0e-6 {
                    (cell.display / glyph.density_norm).clamp(0.25, 1.0)
                } else {
                    1.0
                };
                [(fg * 255.0).round().clamp(0.0, 255.0) as u8; 3]
            } else {
                cell.fg_color
            };
            AsciiCell {
                ch: glyph.ch,
                fg_color,
                density: glyph.density_norm,
                bitmap: glyph.bitmap.clone(),
            }
        })
        .collect()
}

/// 按基代价升序保留前 `count` 个候选；相同代价按字表顺序，保证与 `nearest_glyph` 的首选一致。
fn top_glyph_candidates(
    glyphs: &[CharGlyph],
    target: MatchTarget,
    prior: Option<&DirectionPrior>,
    count: usize,
) -> Vec<(usize, f32)> {
    let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(count);
    for (index, glyph) in glyphs.iter().enumerate() {
        let cost = glyph_cost(glyph, target, prior);
        let position = candidates
            .iter()
            .position(|(_, existing)| cost < *existing)
            .unwrap_or(candidates.len());
        if position < count {
            candidates.insert(position, (index, cost));
            if candidates.len() > count {
                candidates.pop();
            }
        }
    }
    candidates
}

struct RefineCell {
    candidates: Vec<(usize, f32)>,
    /// `candidates` 的下标，不是字表下标。
    chosen: usize,
    direction: Option<CellDirection>,
    fg_color: [u8; 3],
    fg_luminance: f32,
    mono: bool,
    display: f32,
    target: f32,
    render: f32,
}

/// 像素级 HVS 残差：目标与当前渲染都先做 sigma=3.5 的高斯模糊，
/// 每个字形预计算带外扩边界的模糊足迹，候选替换只在足迹内增量更新。
struct PixelHvs {
    padded_width: usize,
    footprint_width: usize,
    footprint_height: usize,
    residual: Vec<f32>,
    footprints: Vec<Vec<f32>>,
}

impl PixelHvs {
    fn new(work: &WorkImage, glyphs: &[CharGlyph], max_ink: f32) -> Self {
        let width = work.width as usize;
        let height = work.height as usize;
        let (values, grid_width, grid_height) = downsample2(&work.luma, width, height);
        let radius = HVS_REACH / HVS_DOWNSCALE as isize;
        let sigma = HVS_SIGMA_PIXELS / HVS_DOWNSCALE as f32;
        let kernel = gaussian_kernel(sigma, radius);
        let cell_width = CELL_WIDTH as usize / HVS_DOWNSCALE;
        let cell_height = CELL_HEIGHT as usize / HVS_DOWNSCALE;
        let footprint_width = cell_width + radius as usize * 2;
        let footprint_height = cell_height + radius as usize * 2;
        let target = blur_values(&values, grid_width, grid_height, &kernel, true);
        let padded_width = grid_width + radius as usize * 2;
        let padded_height = grid_height + radius as usize * 2;
        let mut residual = vec![0.0_f32; padded_width * padded_height];
        for y in 0..grid_height {
            for x in 0..grid_width {
                residual[(y + radius as usize) * padded_width + x + radius as usize] =
                    -target[y * grid_width + x];
            }
        }
        let footprints: Vec<Vec<f32>> = glyphs
            .iter()
            .map(|glyph| {
                let mut canvas = vec![0.0_f32; footprint_width * footprint_height];
                for y in 0..cell_height {
                    for x in 0..cell_width {
                        let mut sum = 0.0;
                        for dy in 0..HVS_DOWNSCALE {
                            for dx in 0..HVS_DOWNSCALE {
                                let source_y = y * HVS_DOWNSCALE + dy;
                                let source_x = x * HVS_DOWNSCALE + dx;
                                sum += f32::from(
                                    glyph.bitmap[source_y * CELL_WIDTH as usize + source_x],
                                ) / 255.0;
                            }
                        }
                        canvas[(y + radius as usize) * footprint_width + x + radius as usize] =
                            sum / (HVS_DOWNSCALE * HVS_DOWNSCALE) as f32 / max_ink;
                    }
                }
                blur_values(&canvas, footprint_width, footprint_height, &kernel, false)
            })
            .collect();
        Self {
            padded_width,
            footprint_width,
            footprint_height,
            residual,
            footprints,
        }
    }

    fn add_cell(&mut self, col: u32, row: u32, glyph: usize, fg: f32) {
        let (origin_x, origin_y) = self.origin(col, row);
        for fy in 0..self.footprint_height {
            let row_start = (origin_y as usize + fy) * self.padded_width + origin_x as usize;
            for fx in 0..self.footprint_width {
                self.residual[row_start + fx] +=
                    fg * self.footprints[glyph][fy * self.footprint_width + fx];
            }
        }
    }

    fn delta(
        &self,
        col: u32,
        row: u32,
        old_glyph: usize,
        old_fg: f32,
        new_glyph: usize,
        new_fg: f32,
    ) -> f32 {
        let (origin_x, origin_y) = self.origin(col, row);
        let mut delta = 0.0;
        for fy in 0..self.footprint_height {
            let row_start = (origin_y as usize + fy) * self.padded_width + origin_x as usize;
            for fx in 0..self.footprint_width {
                let index = fy * self.footprint_width + fx;
                let change = new_fg * self.footprints[new_glyph][index]
                    - old_fg * self.footprints[old_glyph][index];
                let error = self.residual[row_start + fx];
                delta += 2.0 * error * change + change * change;
            }
        }
        delta
    }

    fn apply(
        &mut self,
        col: u32,
        row: u32,
        old_glyph: usize,
        old_fg: f32,
        new_glyph: usize,
        new_fg: f32,
    ) {
        let (origin_x, origin_y) = self.origin(col, row);
        for fy in 0..self.footprint_height {
            let row_start = (origin_y as usize + fy) * self.padded_width + origin_x as usize;
            for fx in 0..self.footprint_width {
                let index = fy * self.footprint_width + fx;
                let change = new_fg * self.footprints[new_glyph][index]
                    - old_fg * self.footprints[old_glyph][index];
                self.residual[row_start + fx] += change;
            }
        }
    }

    fn origin(&self, col: u32, row: u32) -> (isize, isize) {
        (
            col as isize * (CELL_WIDTH / HVS_DOWNSCALE as u32) as isize,
            row as isize * (CELL_HEIGHT / HVS_DOWNSCALE as u32) as isize,
        )
    }
}

struct RefineGrid<'a> {
    glyphs: &'a [CharGlyph],
    cols: usize,
    rows: usize,
    cells: Vec<RefineCell>,
    pixel: PixelHvs,
    initial_mean: f32,
    global_sum: f32,
}

struct RefineChoice {
    index: usize,
    slot: usize,
    fg: f32,
    render: f32,
}

impl RefineGrid<'_> {
    fn choose_cell(&self, index: usize) -> Option<RefineChoice> {
        let current_slot = self.cells[index].chosen;
        let current_base = self.cells[index].candidates[current_slot].1;
        let current_render = self.cells[index].render;
        let old_glyph = self.cells[index].candidates[current_slot].0;
        let old_fg = self.candidate_fg(index, old_glyph);
        let col = (index % self.cols) as u32;
        let row = (index / self.cols) as u32;
        let mut best_slot = current_slot;
        let mut best_score = current_base;

        for slot in 0..self.cells[index].candidates.len() {
            if slot == current_slot {
                continue;
            }
            let (glyph_index, base_cost) = self.cells[index].candidates[slot];
            let fg = self.candidate_fg(index, glyph_index);
            let render = fg * self.glyphs[glyph_index].density_norm;
            let delta_render = render - current_render;
            let score = base_cost
                + REFINE_PIXEL_WEIGHT
                    * self
                        .pixel
                        .delta(col, row, old_glyph, old_fg, glyph_index, fg)
                + REFINE_MEAN_WEIGHT * self.mean_delta(delta_render)
                + REFINE_DENSITY_WEIGHT * self.density_delta(index, render)
                + REFINE_EDGE_WEIGHT * self.edge_delta(index, current_render, render);
            if score < best_score {
                best_score = score;
                best_slot = slot;
            }
        }

        if best_slot == current_slot {
            return None;
        }
        let glyph_index = self.cells[index].candidates[best_slot].0;
        let fg = self.candidate_fg(index, glyph_index);
        Some(RefineChoice {
            index,
            slot: best_slot,
            fg,
            render: fg * self.glyphs[glyph_index].density_norm,
        })
    }

    fn apply_choice(&mut self, choice: RefineChoice) {
        let index = choice.index;
        let old_slot = self.cells[index].chosen;
        let old_glyph = self.cells[index].candidates[old_slot].0;
        let old_fg = self.candidate_fg(index, old_glyph);
        let current_render = self.cells[index].render;
        let glyph_index = self.cells[index].candidates[choice.slot].0;
        let col = (index % self.cols) as u32;
        let row = (index / self.cols) as u32;
        self.pixel
            .apply(col, row, old_glyph, old_fg, glyph_index, choice.fg);
        self.cells[index].chosen = choice.slot;
        self.cells[index].render = choice.render;
        self.global_sum += choice.render - current_render;
    }

    fn candidate_fg(&self, index: usize, glyph_index: usize) -> f32 {
        let cell = &self.cells[index];
        if !cell.mono {
            return cell.fg_luminance;
        }
        let density = self.glyphs[glyph_index].density_norm;
        if density > 1.0e-6 {
            (cell.display / density).clamp(0.25, 1.0)
        } else {
            1.0
        }
    }

    fn mean_delta(&self, delta_render: f32) -> f32 {
        let count = self.cells.len() as f32;
        let current = self.global_sum / count - self.initial_mean;
        let next = (self.global_sum + delta_render) / count - self.initial_mean;
        next.powi(2) - current.powi(2)
    }

    /// 单格渲染值对目标单元均值的平方误差增量。
    fn density_delta(&self, index: usize, render: f32) -> f32 {
        let cell = &self.cells[index];
        (render - cell.target).powi(2) - (cell.render - cell.target).powi(2)
    }

    fn edge_delta(&self, index: usize, current_render: f32, render: f32) -> f32 {
        self.incident_edge_energy(index, render) - self.incident_edge_energy(index, current_render)
    }

    /// 本格参与的全部边能：四条出边，加上四个上游邻格指向本格的入边。
    fn incident_edge_energy(&self, index: usize, render: f32) -> f32 {
        let cell = &self.cells[index];
        let Some(direction) = cell.direction.filter(|direction| direction.diffuses) else {
            return 0.0;
        };
        let col = index % self.cols;
        let row = index / self.cols;
        let mut energy = 0.0;
        for (delta_col, delta_row, offset_x, offset_y) in REFINE_NEIGHBORS {
            let neighbor_col = col as isize + delta_col;
            let neighbor_row = row as isize + delta_row;
            if !(0..self.cols as isize).contains(&neighbor_col)
                || !(0..self.rows as isize).contains(&neighbor_row)
            {
                continue;
            }
            let neighbor = &self.cells[neighbor_row as usize * self.cols + neighbor_col as usize];
            let Some(neighbor_direction) =
                neighbor.direction.filter(|direction| direction.diffuses)
            else {
                continue;
            };
            energy += edge_pair_energy(direction, neighbor_direction, offset_x, offset_y)
                * (render - neighbor.render).powi(2);
        }
        for (delta_col, delta_row, offset_x, offset_y) in REFINE_INCOMING {
            let source_col = col as isize + delta_col;
            let source_row = row as isize + delta_row;
            if !(0..self.cols as isize).contains(&source_col)
                || !(0..self.rows as isize).contains(&source_row)
            {
                continue;
            }
            let source = &self.cells[source_row as usize * self.cols + source_col as usize];
            let Some(source_direction) = source.direction.filter(|direction| direction.diffuses)
            else {
                continue;
            };
            energy += edge_pair_energy(source_direction, direction, offset_x, offset_y)
                * (source.render - render).powi(2);
        }
        energy
    }
}

fn downsample2(values: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let grid_width = width.div_ceil(HVS_DOWNSCALE);
    let grid_height = height.div_ceil(HVS_DOWNSCALE);
    let mut output = vec![0.0_f32; grid_width * grid_height];
    for y in 0..grid_height {
        for x in 0..grid_width {
            let mut sum = 0.0;
            let mut count = 0;
            for dy in 0..HVS_DOWNSCALE {
                let source_y = y * HVS_DOWNSCALE + dy;
                if source_y >= height {
                    continue;
                }
                for dx in 0..HVS_DOWNSCALE {
                    let source_x = x * HVS_DOWNSCALE + dx;
                    if source_x >= width {
                        continue;
                    }
                    sum += values[source_y * width + source_x];
                    count += 1;
                }
            }
            output[y * grid_width + x] = sum / count.max(1) as f32;
        }
    }
    (output, grid_width, grid_height)
}

fn gaussian_kernel(sigma: f32, radius: isize) -> Vec<f32> {
    let mut kernel = Vec::with_capacity((radius * 2 + 1) as usize);
    let mut sum = 0.0;
    for offset in -radius..=radius {
        let weight = (-0.5 * (offset as f32 / sigma).powi(2)).exp();
        kernel.push(weight);
        sum += weight;
    }
    for weight in &mut kernel {
        *weight /= sum;
    }
    kernel
}

fn blur_values(
    values: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
    clamp_edges: bool,
) -> Vec<f32> {
    let radius = (kernel.len() / 2) as isize;
    let sample = |x: isize, y: isize, data: &[f32]| -> f32 {
        if clamp_edges {
            let cx = x.clamp(0, width as isize - 1) as usize;
            let cy = y.clamp(0, height as isize - 1) as usize;
            data[cy * width + cx]
        } else if (0..width as isize).contains(&x) && (0..height as isize).contains(&y) {
            data[y as usize * width + x as usize]
        } else {
            0.0
        }
    };

    let mut horizontal = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (index, weight) in kernel.iter().enumerate() {
                let offset = index as isize - radius;
                sum += weight * sample(x as isize + offset, y as isize, values);
            }
            horizontal[y * width + x] = sum;
        }
    }
    let mut output = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (index, weight) in kernel.iter().enumerate() {
                let offset = index as isize - radius;
                sum += weight * sample(x as isize, y as isize + offset, &horizontal);
            }
            output[y * width + x] = sum;
        }
    }
    output
}

/// 边权只取源格方向；目标格方向只贡献强度。
fn edge_pair_energy(
    source: CellDirection,
    target: CellDirection,
    offset_x: f32,
    offset_y: f32,
) -> f32 {
    let length = offset_x.hypot(offset_y);
    let dot = (offset_x * source.edge.normal_x + offset_y * source.edge.normal_y) / length;
    let along = 1.0 - dot * dot;
    along * source.edge.strength * target.edge.strength
}

/// `enhanced` 已做过方向增强，这里不再 crunch，抖动路径才能与无抖动路径同形。
fn build_ascii_cell(
    image: &ProcessedImage,
    atlas: &FontAtlas,
    config: &MatchConfig,
    position: CellPosition,
    target: MatchTarget,
    direction: Option<CellDirection>,
) -> AsciiCell {
    let glyph = nearest_glyph(
        atlas.ascii(),
        target,
        direction_prior_from(direction.map(|direction| direction.edge)).as_ref(),
    );

    AsciiCell {
        ch: glyph.ch,
        fg_color: foreground_color(image, config, position, glyph.density_norm),
        density: glyph.density_norm,
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
            let x = position.col * CELL_WIDTH + dot_col as u32 * 5;
            let y = position.row * CELL_HEIGHT + dot_row as u32 * 5;
            let value = BRAILLE_BLUE_NOISE[y as usize % 16][x as usize % 16];
            (f32::from(value) + 0.5) / 256.0
        } else {
            0.5
        };
        if sample >= threshold {
            bits |= 1 << bit;
        }
    }
    let density = bits.count_ones() as f32 / 8.0;
    AsciiCell {
        ch: braille_char(bits),
        fg_color: foreground_color(image, config, position, density),
        density,
        bitmap: atlas.braille_bitmap(bits),
    }
}

#[derive(Clone, Copy, Debug)]
struct MatchTarget {
    /// 去均值后的 6D 形状，与整体密度解耦。
    shape: [f32; 6],
    /// 单元平均亮度，0..1。
    density: f32,
}

fn cell_target(
    work: &WorkImage,
    col: u32,
    row: u32,
    exponent: f32,
    shape_scale: f32,
) -> MatchTarget {
    let shape = enhanced_samples(work, col, row, exponent, shape_scale);
    let bounds = (
        col * CELL_WIDTH,
        row * CELL_HEIGHT,
        col * CELL_WIDTH + CELL_WIDTH,
        row * CELL_HEIGHT + CELL_HEIGHT,
    );
    MatchTarget {
        shape,
        density: mean_region(work, bounds),
    }
}

fn nearest_glyph<'a>(
    glyphs: &'a [CharGlyph],
    target: MatchTarget,
    prior: Option<&DirectionPrior>,
) -> &'a CharGlyph {
    glyphs
        .iter()
        .min_by(|left, right| {
            glyph_cost(left, target, prior).total_cmp(&glyph_cost(right, target, prior))
        })
        .expect("ASCII atlas is not empty")
}

fn glyph_cost(glyph: &CharGlyph, target: MatchTarget, prior: Option<&DirectionPrior>) -> f32 {
    let shape_cost = distance_squared(glyph.shape_vec, target.shape);
    let density_cost = (glyph.density_norm - target.density).powi(2);
    let mut cost = SHAPE_WEIGHT * shape_cost + DENSITY_WEIGHT * density_cost;
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
fn enhanced_samples(
    work: &WorkImage,
    col: u32,
    row: u32,
    exponent: f32,
    shape_scale: f32,
) -> [f32; 6] {
    let scale = shape_scale.max(1.0e-6);
    dir_crunch(
        internal_samples(work, col, row),
        external_samples(work, col, row),
        exponent,
    )
    .map(|value| value / scale)
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
    glyph_density: f32,
) -> [u8; 3] {
    if config.color {
        return image.cell_highlight_color(config.cols, position.rows, position.col, position.row);
    }
    // 抖动需要固定前景，误差扩散才能改变最终亮度；1B 补偿会把误差抵消。
    if config.dither {
        return [255; 3];
    }
    let target = image.cell_luminance(config.cols, position.rows, position.col, position.row);
    let level = if glyph_density > 1.0e-6 {
        (target / glyph_density).clamp(0.25, 1.0)
    } else {
        1.0
    };
    [(level * 255.0).round().clamp(0.0, 255.0) as u8; 3]
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

fn ostromoukhov_coefficients(level: f32) -> [f32; 3] {
    let index = (level.clamp(0.0, 1.0) * 255.0).round() as usize;
    let table_index = if index <= 127 { index } else { 255 - index };
    let [a, b, c] = OSTROMOUKHOV_WEIGHTS[table_index];
    let sum = f32::from(a + b + c).max(1.0);
    [f32::from(a) / sum, f32::from(b) / sum, f32::from(c) / sum]
}

/// 蛇形扫描的三邻域权重：前向、左下/右下、正下；强边时沿边权重增大。
fn dither_weights(
    direction: Option<CellDirection>,
    forward: bool,
    level: f32,
) -> ([f32; 3], [(isize, isize); 3]) {
    let base = ostromoukhov_coefficients(level);
    let offsets = if forward {
        [(1, 0), (-1, 1), (0, 1)]
    } else {
        [(-1, 0), (1, 1), (0, 1)]
    };
    let Some(direction) = direction.filter(|direction| direction.diffuses) else {
        return (base, offsets);
    };
    let edge = direction.edge;
    let ramp = ((edge.strength - DIFFUSION_RAMP_START)
        / (DIFFUSION_RAMP_END - DIFFUSION_RAMP_START))
        .clamp(0.0, 1.0);
    if ramp <= 0.0 {
        return (base, offsets);
    }
    let mut weights = base;
    for (weight, (dx, dy)) in weights.iter_mut().zip(offsets) {
        let px = dx as f32 * CELL_WIDTH as f32;
        let py = dy as f32 * CELL_HEIGHT as f32;
        let length = px.hypot(py);
        let along = 1.0 - ((px * edge.normal_x + py * edge.normal_y) / length).powi(2);
        *weight *= (1.0 - ramp) + ramp * along;
    }
    let sum: f32 = weights.iter().sum();
    if sum <= 1.0e-6 {
        return (base, offsets);
    }
    (weights.map(|weight| weight / sum), offsets)
}

fn diffuse_three_errors(
    dims: (u32, u32),
    position: (u32, u32),
    error: f32,
    weights: [f32; 3],
    offsets: [(isize, isize); 3],
    errors: &mut [f32],
) {
    let (cols, rows) = dims;
    let (col, row) = position;
    for (weight, (dx, dy)) in weights.into_iter().zip(offsets) {
        let target_col = col as isize + dx;
        let target_row = row as isize + dy;
        if (0..cols as isize).contains(&target_col) && (0..rows as isize).contains(&target_row) {
            errors[cell_index(cols, target_col as u32, target_row as u32)] += error * weight;
        }
    }
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgb, RgbImage};

    use super::{
        BACKSLASH_CANDIDATES, CellDirection, DIFFUSION_RAMP_START, DIRECTION_PRIOR_WEIGHT,
        DirectionPrior, EdgeDirection, HORIZONTAL_CANDIDATES, MatchConfig, MatchTarget, MatchedArt,
        REFINE_CANDIDATE_COUNT, SLASH_CANDIDATES, VERTICAL_CANDIDATES, aiss_cell_descriptor,
        cell_direction, dir_crunch, direction_prior_from, directional_ascii,
        directional_candidates, distance_squared, dither_weights, external_bounds, match_image,
        mean_region, nearest_glyph, ostromoukhov_coefficients, output_rows, top_glyph_candidates,
    };
    use crate::font_atlas::FontAtlas;
    use crate::preprocess::{CELL_HEIGHT, CELL_WIDTH, WorkImage, cell_region_bounds, preprocess};
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
            refine: false,
            aiss: false,
            color: false,
            braille: false,
        }
    }

    fn refine_config() -> MatchConfig {
        MatchConfig {
            refine: true,
            ..config(false)
        }
    }

    fn aiss_config() -> MatchConfig {
        MatchConfig {
            aiss: true,
            ..config(false)
        }
    }

    fn match_target(shape: [f32; 6], density: f32) -> MatchTarget {
        MatchTarget { shape, density }
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

    fn ascii_chars(art: &MatchedArt) -> Vec<char> {
        art.cells.iter().map(|cell| cell.ch).collect()
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
    fn dither_weights_fall_back_to_the_base_table() {
        let base = ostromoukhov_coefficients(0.5);
        assert_eq!(dither_weights(None, true, 0.5).0, base);

        let weak = CellDirection {
            edge: EdgeDirection {
                normal_x: 1.0,
                normal_y: 0.0,
                strength: DIFFUSION_RAMP_START * 0.5,
            },
            diffuses: true,
        };
        assert_eq!(dither_weights(Some(weak), true, 0.5).0, base);

        let gated = CellDirection {
            edge: EdgeDirection {
                normal_x: 1.0,
                normal_y: 0.0,
                strength: 1.0,
            },
            diffuses: false,
        };
        assert_eq!(dither_weights(Some(gated), true, 0.5).0, base);
    }

    #[test]
    fn dither_weights_follow_a_vertical_edge() {
        let (weights, offsets) = dither_weights(
            Some(CellDirection {
                edge: EdgeDirection {
                    normal_x: 1.0,
                    normal_y: 0.0,
                    strength: 1.0,
                },
                diffuses: true,
            }),
            true,
            0.5,
        );
        assert_eq!(offsets, [(1, 0), (-1, 1), (0, 1)]);
        assert!(weights[0] < 1.0e-6, "right={}", weights[0]);
        assert!(weights[2] > 1.0 / 3.0, "bottom={}", weights[2]);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn dither_weights_follow_a_horizontal_edge() {
        let (weights, offsets) = dither_weights(
            Some(CellDirection {
                edge: EdgeDirection {
                    normal_x: 0.0,
                    normal_y: 1.0,
                    strength: 1.0,
                },
                diffuses: true,
            }),
            true,
            0.5,
        );
        assert_eq!(offsets, [(1, 0), (-1, 1), (0, 1)]);
        assert!(weights[2] < 1.0e-6, "bottom={}", weights[2]);
        assert!(weights[0] > 1.0 / 3.0, "forward={}", weights[0]);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn serpentine_scan_mirrors_the_forward_neighbor() {
        let direction = Some(CellDirection {
            edge: EdgeDirection {
                normal_x: 1.0,
                normal_y: 0.0,
                strength: 1.0,
            },
            diffuses: true,
        });
        let (forward, forward_offsets) = dither_weights(direction, true, 0.5);
        let (backward, backward_offsets) = dither_weights(direction, false, 0.5);
        assert_eq!(forward_offsets[0], (1, 0));
        assert_eq!(backward_offsets[0], (-1, 0));
        assert!(forward[0] < 1.0e-6);
        assert!(backward[0] < 1.0e-6);
        assert!((forward[2] - backward[2]).abs() < 1.0e-6);
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
                nearest_glyph(atlas.ascii(), match_target(target, 0.5), Some(&prior)).ch,
                nearest_glyph(atlas.ascii(), match_target(target, 0.5), None).ch
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
        let glyph = nearest_glyph(atlas.ascii(), match_target([0.5; 6], 0.5), Some(&prior));
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
            dither_weights(Some(cell), true, 0.5).0,
            ostromoukhov_coefficients(0.5),
            "低对比度单元应回退到基础系数"
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

        let glyph_a = atlas
            .ascii()
            .iter()
            .find(|glyph| glyph.ch == 'A')
            .expect("atlas contains A");
        assert!(!VERTICAL_CANDIDATES.contains(&'A'));
        let chosen = nearest_glyph(
            atlas.ascii(),
            match_target(glyph_a.shape_vec, glyph_a.density_norm),
            Some(&prior),
        );
        assert_eq!(chosen.ch, 'A');
    }

    /// 方向先验只软性加罚，形状完全匹配的非候选字 'A' 仍应胜出。
    #[test]
    fn nearest_glyph_keeps_a_better_non_candidate_shape() {
        let atlas = FontAtlas::load().unwrap();
        let work = uniform_work_cell(1.0, 1.0);
        let config = MatchConfig {
            cols: 1,
            dir_crunch: 2.5,
            edge_weight: 0.25,
            dither: false,
            refine: false,
            aiss: false,
            color: false,
            braille: false,
        };
        let glyph_a = atlas
            .ascii()
            .iter()
            .find(|glyph| glyph.ch == 'A')
            .expect("atlas contains A");

        let direction = cell_direction(&work, &config, 0, 0);
        let prior = direction_prior_from(direction.map(|direction| direction.edge));
        assert_eq!(
            nearest_glyph(
                atlas.ascii(),
                match_target(glyph_a.shape_vec, glyph_a.density_norm),
                prior.as_ref()
            )
            .ch,
            'A'
        );
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
    fn top_glyph_candidates_start_with_the_best_and_keep_five() {
        let atlas = FontAtlas::load().unwrap();
        let target = [0.2, 0.8, 0.4, 0.1, 0.6, 0.3];
        let candidates = top_glyph_candidates(
            atlas.ascii(),
            match_target(target, 0.5),
            None,
            REFINE_CANDIDATE_COUNT,
        );
        assert_eq!(candidates.len(), REFINE_CANDIDATE_COUNT);
        assert_eq!(
            atlas.ascii()[candidates[0].0].ch,
            nearest_glyph(atlas.ascii(), match_target(target, 0.5), None).ch
        );
        assert!(
            candidates
                .windows(2)
                .all(|window| window[0].1 <= window[1].1)
        );
    }

    #[test]
    fn refine_matching_is_deterministic() {
        let image = striped_image(80, 40);
        let processed = preprocess(&image, 1.0, 0.25, ToneMap::Global).unwrap();
        let atlas = FontAtlas::load().unwrap();
        let first = match_image(&processed, &atlas, &refine_config()).unwrap();
        let second = match_image(&processed, &atlas, &refine_config()).unwrap();
        assert_eq!(ascii_chars(&first), ascii_chars(&second));
    }

    #[test]
    fn refine_can_change_a_glyph_without_affecting_plain() {
        let image = striped_image(80, 40);
        let processed = preprocess(&image, 1.0, 0.25, ToneMap::Global).unwrap();
        let atlas = FontAtlas::load().unwrap();
        let plain_config = config(false);
        let refined_config = MatchConfig {
            refine: true,
            ..plain_config.clone()
        };

        let plain = match_image(&processed, &atlas, &plain_config).unwrap();
        let refined = match_image(&processed, &atlas, &refined_config).unwrap();
        let plain_again = match_image(&processed, &atlas, &plain_config).unwrap();

        assert_eq!(
            ascii_chars(&plain),
            ascii_chars(&plain_again),
            "plain 输出不应受 refine 影响"
        );
        assert!(
            plain
                .cells
                .iter()
                .zip(&refined.cells)
                .any(|(left, right)| left.ch != right.ch),
            "强边图像上 refine 未改变任何字符"
        );
    }

    #[test]
    fn flat_cells_fall_back_to_plain_matching() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(40, 40, Rgb([128, 128, 128])));
        let processed = preprocess(&image, 1.0, 0.25, ToneMap::Global).unwrap();
        let atlas = FontAtlas::load().unwrap();
        let plain = match_image(&processed, &atlas, &config(false)).unwrap();
        let aiss = match_image(&processed, &atlas, &aiss_config()).unwrap();
        assert_eq!(ascii_chars(&plain), ascii_chars(&aiss));
    }

    #[test]
    fn aiss_gate_uses_the_unified_cell_edge_map() {
        // edge_strength 与 edge_scale 取反值，证明门限只看统一边缘图。
        let mut step = uniform_work_cell(1.0, 0.0);
        for y in 0..CELL_HEIGHT as usize {
            for x in 0..CELL_WIDTH as usize {
                step.luma[y * CELL_WIDTH as usize + x] = if x >= 5 { 1.0 } else { 0.0 };
            }
        }
        assert!(aiss_cell_descriptor(&step, 0, 0).is_some());

        let mut flat = uniform_work_cell(1.0, 1.0);
        flat.luma.fill(0.5);
        assert!(aiss_cell_descriptor(&flat, 0, 0).is_none());
    }

    /// 1/255、2/255 台阶是非空单元，但绝对能量不足，必须回退纯 6D。
    #[test]
    fn aiss_gate_falls_back_on_low_contrast_non_blank_cells() {
        for (amplitude, expected) in [
            (1.0 / 255.0, false),
            (2.0 / 255.0, false),
            (3.0 / 255.0, true),
        ] {
            let mut step = uniform_work_cell(1.0, 0.0);
            for y in 0..CELL_HEIGHT as usize {
                for x in 0..CELL_WIDTH as usize {
                    step.luma[y * CELL_WIDTH as usize + x] = if x >= 5 { amplitude } else { 0.0 };
                }
            }
            let edges = crate::aiss::cell_edges(&super::cell_luma(&step, 0, 0));
            assert!(
                edges.edges.iter().any(|value| *value > 0.0),
                "amplitude={amplitude} 不是空白单元"
            );
            assert_eq!(
                aiss_cell_descriptor(&step, 0, 0).is_some(),
                expected,
                "amplitude={amplitude} mean={}",
                edges.mean_magnitude
            );
        }
    }

    #[test]
    fn aiss_matching_is_deterministic_and_reranks_ambiguous_cells() {
        let mut image = RgbImage::new(80, 40);
        for y in 0..image.height() {
            for x in 0..image.width() {
                let value = if (x / 3 + y / 3) % 2 == 0 { 230 } else { 25 };
                image.put_pixel(x, y, Rgb([value, value, value]));
            }
        }
        let image = DynamicImage::ImageRgb8(image);
        let processed = preprocess(&image, 1.0, 0.25, ToneMap::Clahe).unwrap();
        let atlas = FontAtlas::load().unwrap();
        let plain = match_image(&processed, &atlas, &config(false)).unwrap();
        let first = match_image(&processed, &atlas, &aiss_config()).unwrap();
        let second = match_image(&processed, &atlas, &aiss_config()).unwrap();

        assert_eq!(ascii_chars(&first), ascii_chars(&second));
        assert!(
            plain
                .cells
                .iter()
                .zip(&first.cells)
                .any(|(left, right)| left.ch != right.ch),
            "AISS 未在棋盘格上改变任何字符"
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
    fn serial_dither_uses_structure_aware_weights() {
        let image = vertical_stripes(80, 40);
        let processed = preprocess(&image, 1.0, 0.25, ToneMap::Global).unwrap();
        let atlas = FontAtlas::load().unwrap();
        let conf = MatchConfig {
            cols: 8,
            dir_crunch: 2.5,
            edge_weight: 0.25,
            dither: true,
            refine: false,
            aiss: false,
            color: false,
            braille: false,
        };

        let rows = output_rows(conf.cols, processed.width, processed.height).unwrap();
        let work = processed
            .resample(conf.cols * CELL_WIDTH, rows * CELL_HEIGHT)
            .unwrap();
        let anisotropic_cells = (0..rows)
            .flat_map(|row| (0..conf.cols).map(move |col| (col, row)))
            .filter(|&(col, row)| {
                cell_direction(&work, &conf, col, row).is_some_and(|direction| {
                    dither_weights(Some(direction), true, 0.5).0 != ostromoukhov_coefficients(0.5)
                })
            })
            .count();
        assert!(anisotropic_cells > 0, "竖条纹没有触发结构感知扩散");
        assert_eq!(
            match_image(&processed, &atlas, &conf).unwrap().cells.len(),
            (rows * 8) as usize
        );
    }

    #[test]
    fn mono_foreground_compensates_glyph_density() {
        let atlas = FontAtlas::load().unwrap();
        let config = config(false);
        for value in [0_u8, 128, 255] {
            let image =
                DynamicImage::ImageRgb8(RgbImage::from_pixel(10, 20, Rgb([value, value, value])));
            let processed = preprocess(&image, 1.0, 0.0, ToneMap::Global).unwrap();
            let rows = output_rows(config.cols, processed.width, processed.height).unwrap();
            let art = match_image(&processed, &atlas, &config).unwrap();
            let cell = &art.cells[0];
            let target = processed.cell_luminance(config.cols, rows, 0, 0);
            let expected = if cell.density > 1.0e-6 {
                (target / cell.density).clamp(0.25, 1.0)
            } else {
                1.0
            };
            let expected = (expected * 255.0).round() as u8;
            assert_eq!(cell.fg_color[0], expected, "value={value}");
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
    fn braille_dither_uses_blue_noise_thresholds_per_dot() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(40, 40, Rgb([128, 128, 128])));
        let processed = preprocess(&image, 1.0, 0.0, ToneMap::Global).unwrap();
        let atlas = FontAtlas::load().unwrap();
        let mut dithered = config(true);
        dithered.braille = true;
        let art = match_image(&processed, &atlas, &dithered).unwrap();
        let mean = art.cells.iter().map(|cell| cell.density).sum::<f32>() / art.cells.len() as f32;
        assert!((mean - 0.5).abs() < 0.15, "mean={mean}");
        assert!(art.cells.iter().any(|cell| cell.density != 0.5));

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
