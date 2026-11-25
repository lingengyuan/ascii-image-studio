use std::sync::{Arc, OnceLock};

use anyhow::{Result, bail};
use fontdue::{Font, FontSettings};

use crate::aiss;
use crate::preprocess::{CELL_HEIGHT, CELL_WIDTH, cell_region_bounds};

const FONT_BYTES: &[u8] = include_bytes!("../assets/JetBrainsMono-Regular.ttf");
const CELL_PIXELS: usize = (CELL_WIDTH * CELL_HEIGHT) as usize;

#[derive(Clone, Debug)]
pub struct CharGlyph {
    pub ch: char,
    pub shape_vec: [f32; 6],
    /// 原始覆盖率，单位是 0..max_ink。
    pub density: f32,
    /// 相对最密字形的覆盖率，0..1。
    pub density_norm: f32,
    pub bitmap: Arc<[u8]>,
}

pub struct FontAtlas {
    ascii: Vec<CharGlyph>,
    blocks: Vec<CharGlyph>,
    sextants: Vec<CharGlyph>,
    braille_bitmaps: Vec<Arc<[u8]>>,
    /// 6 维共用的全局缩放系数；目标是原始子块覆盖率除以它。
    shape_scale: f32,
    /// 单字最大原始覆盖率，用于把 density 映射到 0..1。
    max_ink: f32,
    /// AISS 描述子只在首次使用时构建，plain、dither、refine、braille 路径不付这份成本。
    aiss_descriptors: OnceLock<Vec<Vec<f32>>>,
}

pub const BRAILLE_LAYOUT: [(usize, usize); 8] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (0, 1),
    (1, 1),
    (2, 1),
    (3, 0),
    (3, 1),
];

/// 16x16 void-and-cluster 蓝噪声阈值矩阵；按图像坐标平铺，替代 2x4 Bayer。
pub const BRAILLE_BLUE_NOISE: [[u8; 16]; 16] = [
    [
        0, 217, 35, 55, 247, 1, 199, 90, 68, 209, 2, 114, 33, 241, 188, 87,
    ],
    [
        169, 131, 96, 204, 160, 125, 43, 225, 157, 51, 250, 78, 213, 62, 120, 207,
    ],
    [
        73, 228, 14, 115, 77, 216, 177, 15, 118, 192, 94, 174, 141, 10, 159, 47,
    ],
    [
        31, 147, 183, 238, 26, 53, 101, 143, 233, 24, 132, 40, 237, 184, 92, 243,
    ],
    [
        194, 105, 59, 166, 133, 193, 251, 80, 58, 167, 222, 69, 108, 20, 219, 123,
    ],
    [
        3, 252, 38, 83, 224, 4, 161, 36, 208, 111, 5, 198, 136, 57, 152, 79,
    ],
    [
        179, 126, 214, 153, 106, 66, 122, 182, 91, 242, 154, 84, 253, 170, 211, 45,
    ],
    [
        146, 95, 16, 185, 32, 245, 205, 17, 145, 54, 27, 189, 41, 11, 110, 232,
    ],
    [
        61, 203, 75, 235, 135, 46, 156, 76, 215, 130, 230, 103, 127, 202, 88, 29,
    ],
    [
        246, 162, 49, 117, 172, 93, 226, 112, 22, 178, 71, 168, 56, 227, 140, 186,
    ],
    [
        6, 104, 220, 21, 200, 7, 63, 190, 248, 97, 8, 210, 150, 18, 74, 119,
    ],
    [
        42, 138, 181, 81, 254, 134, 164, 34, 144, 48, 236, 116, 37, 255, 173, 218,
    ],
    [
        89, 239, 64, 158, 39, 107, 234, 85, 121, 201, 163, 82, 191, 102, 60, 155,
    ],
    [
        28, 196, 12, 124, 212, 70, 195, 13, 221, 65, 19, 137, 223, 9, 128, 206,
    ],
    [
        113, 171, 100, 231, 25, 175, 50, 149, 180, 98, 240, 44, 72, 176, 229, 52,
    ],
    [
        249, 67, 148, 187, 86, 139, 109, 244, 30, 129, 165, 197, 151, 99, 23, 142,
    ],
];

impl FontAtlas {
    pub fn load() -> Result<Self> {
        let font = Font::from_bytes(FONT_BYTES, FontSettings::default())
            .map_err(|message| anyhow::anyhow!("字体解析失败: {message}"))?;
        let (size, baseline) = choose_font_size(&font)?;
        let mut ascii = Vec::with_capacity(95);

        for code in 0x20_u8..=0x7e {
            let ch = char::from(code);
            let bitmap = rasterize_centered(&font, ch, size, baseline)?;
            let density = bitmap
                .iter()
                .map(|value| f32::from(*value) / 255.0)
                .sum::<f32>()
                / CELL_PIXELS as f32;
            ascii.push(CharGlyph {
                ch,
                shape_vec: raw_shape_vector(&bitmap),
                density,
                density_norm: 0.0,
                bitmap: Arc::from(bitmap),
            });
        }

        let max_ink = ascii
            .iter()
            .map(|glyph| glyph.density)
            .fold(0.0_f32, f32::max);
        for glyph in &mut ascii {
            glyph.density_norm = glyph.density / max_ink.max(1.0e-6);
        }
        let shape_scale = normalize_shape_vectors(&mut ascii);
        let braille_bitmaps = (0..=255)
            .map(|bits| Arc::<[u8]>::from(render_braille_bitmap(bits)))
            .collect();
        Ok(Self {
            ascii,
            blocks: build_blocks(),
            sextants: build_sextants(),
            braille_bitmaps,
            shape_scale,
            max_ink,
            aiss_descriptors: OnceLock::new(),
        })
    }

    pub fn ascii(&self) -> &[CharGlyph] {
        &self.ascii
    }

    pub fn blocks(&self) -> &[CharGlyph] {
        &self.blocks
    }

    pub fn sextants(&self) -> &[CharGlyph] {
        &self.sextants
    }

    /// 字形向量与目标单元共用的覆盖率缩放系数。
    pub fn shape_scale(&self) -> f32 {
        self.shape_scale
    }

    /// 单字最大原始覆盖率，对应 density_norm = 1。
    pub fn max_ink(&self) -> f32 {
        self.max_ink
    }

    /// 与 `ascii` 下标一一对应的 AISS 描述子，首次调用时才计算。
    pub fn aiss_descriptors(&self) -> &[Vec<f32>] {
        self.aiss_descriptors.get_or_init(|| {
            self.ascii
                .iter()
                .map(|glyph| aiss::glyph_descriptor(&glyph.bitmap))
                .collect()
        })
    }

    #[cfg(test)]
    fn aiss_descriptors_ready(&self) -> bool {
        self.aiss_descriptors.get().is_some()
    }

    pub fn ascii_bitmap(&self, ch: char) -> Arc<[u8]> {
        self.ascii[ch as usize - 0x20].bitmap.clone()
    }

    pub fn braille_bitmap(&self, bits: u8) -> Arc<[u8]> {
        self.braille_bitmaps[bits as usize].clone()
    }
}

const SEXTANTS: [(char, u8); 60] = [
    ('\u{1fb00}', 1),
    ('\u{1fb01}', 2),
    ('\u{1fb02}', 3),
    ('\u{1fb03}', 4),
    ('\u{1fb04}', 5),
    ('\u{1fb05}', 6),
    ('\u{1fb06}', 7),
    ('\u{1fb07}', 8),
    ('\u{1fb08}', 9),
    ('\u{1fb09}', 10),
    ('\u{1fb0a}', 11),
    ('\u{1fb0b}', 12),
    ('\u{1fb0c}', 13),
    ('\u{1fb0d}', 14),
    ('\u{1fb0e}', 15),
    ('\u{1fb0f}', 16),
    ('\u{1fb10}', 17),
    ('\u{1fb11}', 18),
    ('\u{1fb12}', 19),
    ('\u{1fb13}', 20),
    ('\u{1fb14}', 22),
    ('\u{1fb15}', 23),
    ('\u{1fb16}', 24),
    ('\u{1fb17}', 25),
    ('\u{1fb18}', 26),
    ('\u{1fb19}', 27),
    ('\u{1fb1a}', 28),
    ('\u{1fb1b}', 29),
    ('\u{1fb1c}', 30),
    ('\u{1fb1d}', 31),
    ('\u{1fb1e}', 32),
    ('\u{1fb1f}', 33),
    ('\u{1fb20}', 34),
    ('\u{1fb21}', 35),
    ('\u{1fb22}', 36),
    ('\u{1fb23}', 37),
    ('\u{1fb24}', 38),
    ('\u{1fb25}', 39),
    ('\u{1fb26}', 40),
    ('\u{1fb27}', 41),
    ('\u{1fb28}', 43),
    ('\u{1fb29}', 44),
    ('\u{1fb2a}', 45),
    ('\u{1fb2b}', 46),
    ('\u{1fb2c}', 47),
    ('\u{1fb2d}', 48),
    ('\u{1fb2e}', 49),
    ('\u{1fb2f}', 50),
    ('\u{1fb30}', 51),
    ('\u{1fb31}', 52),
    ('\u{1fb32}', 53),
    ('\u{1fb33}', 54),
    ('\u{1fb34}', 55),
    ('\u{1fb35}', 56),
    ('\u{1fb36}', 57),
    ('\u{1fb37}', 58),
    ('\u{1fb38}', 59),
    ('\u{1fb39}', 60),
    ('\u{1fb3a}', 61),
    ('\u{1fb3b}', 62),
];

fn geometric_glyph(ch: char, bitmap: [u8; 200]) -> CharGlyph {
    let density = bitmap
        .iter()
        .map(|value| f32::from(*value) / 255.0)
        .sum::<f32>()
        / 200.0;
    CharGlyph {
        ch,
        shape_vec: raw_shape_vector(&bitmap),
        density,
        density_norm: density,
        bitmap: Arc::from(bitmap.to_vec()),
    }
}

fn fill_rect(bitmap: &mut [u8; 200], x0: usize, x1: usize, y0: usize, y1: usize) {
    for y in y0..y1 {
        for x in x0..x1 {
            bitmap[y * 10 + x] = 255;
        }
    }
}

fn block_bitmap(ch: char) -> Option<[u8; 200]> {
    let mut bitmap = [0_u8; 200];
    match ch {
        ' ' => {}
        '\u{2580}' => fill_rect(&mut bitmap, 0, 10, 0, 10),
        '\u{2581}'..='\u{2588}' => {
            let eighths = ch as u32 - '\u{2580}' as u32;
            let top = 20 - (eighths * 20 / 8) as usize;
            fill_rect(&mut bitmap, 0, 10, top, 20);
        }
        '\u{2589}'..='\u{258f}' => {
            let eighths = 8 - (ch as u32 - '\u{2588}' as u32);
            fill_rect(&mut bitmap, 0, (eighths * 10 / 8) as usize, 0, 20);
        }
        '\u{2590}' => fill_rect(&mut bitmap, 5, 10, 0, 20),
        '\u{2591}' => {
            for y in 0..20 {
                for x in 0..10 {
                    if (x + y) % 4 == 0 {
                        bitmap[y * 10 + x] = 255;
                    }
                }
            }
        }
        '\u{2592}' => {
            for y in 0..20 {
                for x in 0..10 {
                    if (x + y) % 2 == 0 {
                        bitmap[y * 10 + x] = 255;
                    }
                }
            }
        }
        '\u{2593}' => {
            for y in 0..20 {
                for x in 0..10 {
                    if (x + y) % 4 != 0 {
                        bitmap[y * 10 + x] = 255;
                    }
                }
            }
        }
        '\u{2594}' => fill_rect(&mut bitmap, 0, 10, 0, 3),
        '\u{2595}' => fill_rect(&mut bitmap, 9, 10, 0, 20),
        '\u{2596}'..='\u{259f}' => {
            let mask = match ch {
                '\u{2596}' => 4,
                '\u{2597}' => 8,
                '\u{2598}' => 1,
                '\u{2599}' => 13,
                '\u{259a}' => 9,
                '\u{259b}' => 7,
                '\u{259c}' => 11,
                '\u{259d}' => 2,
                '\u{259e}' => 6,
                '\u{259f}' => 14,
                _ => 0,
            };
            if mask & 1 != 0 {
                fill_rect(&mut bitmap, 0, 5, 0, 10);
            }
            if mask & 2 != 0 {
                fill_rect(&mut bitmap, 5, 10, 0, 10);
            }
            if mask & 4 != 0 {
                fill_rect(&mut bitmap, 0, 5, 10, 20);
            }
            if mask & 8 != 0 {
                fill_rect(&mut bitmap, 5, 10, 10, 20);
            }
        }
        _ => return None,
    }
    Some(bitmap)
}

fn build_blocks() -> Vec<CharGlyph> {
    let mut glyphs = Vec::with_capacity(33);
    for code in 0x20_u32..=0x20 {
        glyphs.push(geometric_glyph(char::from_u32(code).unwrap(), [0; 200]));
    }
    for code in 0x2580_u32..=0x259f {
        let ch = char::from_u32(code).unwrap();
        if let Some(bitmap) = block_bitmap(ch) {
            glyphs.push(geometric_glyph(ch, bitmap));
        }
    }
    glyphs
}

/// 斜切楔形块的点位。x 取 0、5、10，y 取 0、1/3、2/3、1 单元格高。
#[derive(Clone, Copy)]
enum WedgePoint {
    UpperLeft,
    UpperMiddleLeft,
    UpperCentre,
    UpperMiddleRight,
    UpperRight,
    LowerLeft,
    LowerMiddleLeft,
    LowerCentre,
    LowerMiddleRight,
    LowerRight,
}

impl WedgePoint {
    fn coords(self) -> (f32, f32) {
        match self {
            Self::UpperLeft => (0.0, 0.0),
            Self::UpperMiddleLeft => (0.0, 20.0 / 3.0),
            Self::UpperCentre => (5.0, 0.0),
            Self::UpperMiddleRight => (10.0, 20.0 / 3.0),
            Self::UpperRight => (10.0, 0.0),
            Self::LowerLeft => (0.0, 20.0),
            Self::LowerMiddleLeft => (0.0, 40.0 / 3.0),
            Self::LowerCentre => (5.0, 20.0),
            Self::LowerMiddleRight => (10.0, 40.0 / 3.0),
            Self::LowerRight => (10.0, 20.0),
        }
    }
}

// 每个字形：填充侧所在的角，斜线起点，斜线终点。名称见 Unicode 18.0。
const WEDGES: [(char, WedgePoint, WedgePoint, WedgePoint); 44] = {
    use WedgePoint::*;
    [
        ('\u{1fb3c}', LowerLeft, LowerMiddleLeft, LowerCentre),
        ('\u{1fb3d}', LowerLeft, LowerMiddleLeft, LowerRight),
        ('\u{1fb3e}', LowerLeft, UpperMiddleLeft, LowerCentre),
        ('\u{1fb3f}', LowerLeft, UpperMiddleLeft, LowerRight),
        ('\u{1fb40}', LowerLeft, UpperLeft, LowerCentre),
        ('\u{1fb41}', LowerRight, UpperMiddleLeft, UpperCentre),
        ('\u{1fb42}', LowerRight, UpperMiddleLeft, UpperRight),
        ('\u{1fb43}', LowerRight, LowerMiddleLeft, UpperCentre),
        ('\u{1fb44}', LowerRight, LowerMiddleLeft, UpperRight),
        ('\u{1fb45}', LowerRight, LowerLeft, UpperCentre),
        ('\u{1fb46}', LowerRight, LowerMiddleLeft, UpperMiddleRight),
        ('\u{1fb47}', LowerRight, LowerCentre, LowerMiddleRight),
        ('\u{1fb48}', LowerRight, LowerLeft, LowerMiddleRight),
        ('\u{1fb49}', LowerRight, LowerCentre, UpperMiddleRight),
        ('\u{1fb4a}', LowerRight, LowerLeft, UpperMiddleRight),
        ('\u{1fb4b}', LowerRight, LowerCentre, UpperRight),
        ('\u{1fb4c}', LowerLeft, UpperCentre, UpperMiddleRight),
        ('\u{1fb4d}', LowerLeft, UpperLeft, UpperMiddleRight),
        ('\u{1fb4e}', LowerLeft, UpperCentre, LowerMiddleRight),
        ('\u{1fb4f}', LowerLeft, UpperLeft, LowerMiddleRight),
        ('\u{1fb50}', LowerLeft, UpperCentre, LowerRight),
        ('\u{1fb51}', LowerLeft, UpperMiddleLeft, LowerMiddleRight),
        ('\u{1fb52}', UpperRight, LowerMiddleLeft, LowerCentre),
        ('\u{1fb53}', UpperRight, LowerMiddleLeft, LowerRight),
        ('\u{1fb54}', UpperRight, UpperMiddleLeft, LowerCentre),
        ('\u{1fb55}', UpperRight, UpperMiddleLeft, LowerRight),
        ('\u{1fb56}', UpperRight, UpperLeft, LowerCentre),
        ('\u{1fb57}', UpperLeft, UpperMiddleLeft, UpperCentre),
        ('\u{1fb58}', UpperLeft, UpperMiddleLeft, UpperRight),
        ('\u{1fb59}', UpperLeft, LowerMiddleLeft, UpperCentre),
        ('\u{1fb5a}', UpperLeft, LowerMiddleLeft, UpperRight),
        ('\u{1fb5b}', UpperLeft, LowerLeft, UpperCentre),
        ('\u{1fb5c}', UpperLeft, LowerMiddleLeft, UpperMiddleRight),
        ('\u{1fb5d}', UpperLeft, LowerCentre, LowerMiddleRight),
        ('\u{1fb5e}', UpperLeft, LowerLeft, LowerMiddleRight),
        ('\u{1fb5f}', UpperLeft, LowerCentre, UpperMiddleRight),
        ('\u{1fb60}', UpperLeft, LowerLeft, UpperMiddleRight),
        ('\u{1fb61}', UpperLeft, LowerCentre, UpperRight),
        ('\u{1fb62}', UpperRight, UpperCentre, UpperMiddleRight),
        ('\u{1fb63}', UpperRight, UpperLeft, UpperMiddleRight),
        ('\u{1fb64}', UpperRight, UpperCentre, LowerMiddleRight),
        ('\u{1fb65}', UpperRight, UpperLeft, LowerMiddleRight),
        ('\u{1fb66}', UpperRight, UpperCentre, LowerRight),
        ('\u{1fb67}', UpperRight, UpperMiddleLeft, LowerMiddleRight),
    ]
};

fn wedge_bitmap(anchor: WedgePoint, start: WedgePoint, end: WedgePoint) -> [u8; 200] {
    let (ax, ay) = anchor.coords();
    let (x0, y0) = start.coords();
    let (x1, y1) = end.coords();
    let anchor_side = (y1 - y0) * (ax - x0) - (x1 - x0) * (ay - y0);
    let mut bitmap = [0_u8; 200];
    for y in 0..20 {
        for x in 0..10 {
            let side = (y1 - y0) * (x as f32 + 0.5 - x0) - (x1 - x0) * (y as f32 + 0.5 - y0);
            if side * anchor_side >= 0.0 {
                bitmap[y * 10 + x] = 255;
            }
        }
    }
    bitmap
}

fn triangle_bitmap(orientation: char) -> [u8; 200] {
    let mut bitmap = [0_u8; 200];
    for y in 0..20 {
        for x in 0..10 {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;
            let inside = match orientation {
                'l' => px <= (py / 2.0).min(10.0 - py / 2.0),
                'r' => px >= 10.0 - (py / 2.0).min(10.0 - py / 2.0),
                'u' => py <= (2.0 * px).min(20.0 - 2.0 * px),
                _ => py >= 20.0 - (2.0 * px).min(20.0 - 2.0 * px),
            };
            if inside {
                bitmap[y * 10 + x] = 255;
            }
        }
    }
    bitmap
}

fn build_sextants() -> Vec<CharGlyph> {
    let mut glyphs = Vec::with_capacity(114);
    glyphs.push(geometric_glyph(' ', [0; 200]));
    for (ch, mask) in SEXTANTS {
        let mut bitmap = [0_u8; 200];
        for bit in 0..6 {
            if mask & (1 << bit) == 0 {
                continue;
            }
            let column = bit % 2;
            let band = bit / 2;
            let x0 = column * 5;
            let y0 = match band {
                0 => 0,
                1 => 6,
                _ => 13,
            };
            let y1 = match band {
                0 => 6,
                1 => 13,
                _ => 20,
            };
            fill_rect(&mut bitmap, x0, x0 + 5, y0, y1);
        }
        glyphs.push(geometric_glyph(ch, bitmap));
    }
    for (ch, anchor, start, end) in WEDGES {
        glyphs.push(geometric_glyph(ch, wedge_bitmap(anchor, start, end)));
    }
    let left = triangle_bitmap('l');
    let upper = triangle_bitmap('u');
    let right = triangle_bitmap('r');
    let lower = triangle_bitmap('d');
    for (ch, triangle) in [
        ('\u{1fb6c}', left),
        ('\u{1fb6d}', upper),
        ('\u{1fb6e}', right),
        ('\u{1fb6f}', lower),
    ] {
        glyphs.push(geometric_glyph(ch, triangle));
    }
    let complement = |triangle: [u8; 200]| {
        let mut bitmap = [0_u8; 200];
        for (value, source) in bitmap.iter_mut().zip(triangle) {
            *value = 255 - source;
        }
        bitmap
    };
    for (ch, triangle) in [
        ('\u{1fb68}', complement(left)),
        ('\u{1fb69}', complement(upper)),
        ('\u{1fb6a}', complement(right)),
        ('\u{1fb6b}', complement(lower)),
    ] {
        glyphs.push(geometric_glyph(ch, triangle));
    }
    let mut full = [0_u8; 200];
    fill_rect(&mut full, 0, 10, 0, 20);
    glyphs.push(geometric_glyph('\u{2588}', full));
    glyphs
}

fn choose_font_size(font: &Font) -> Result<(f32, f32)> {
    for size in (8..=24).rev() {
        let Some(line) = font.horizontal_line_metrics(size as f32) else {
            continue;
        };
        let line_height = line.ascent - line.descent;
        let baseline = (CELL_HEIGHT as f32 - line_height) * 0.5 + line.ascent;
        let fits = (0x20_u8..=0x7e).all(|code| {
            let metrics = font.metrics(char::from(code), size as f32);
            let top = baseline - (metrics.ymin + metrics.height as i32) as f32;
            let bottom = baseline - metrics.ymin as f32;
            metrics.width <= CELL_WIDTH as usize
                && metrics.height <= CELL_HEIGHT as usize
                && top >= -0.01
                && bottom <= CELL_HEIGHT as f32 + 0.01
        });
        if fits {
            return Ok((size as f32, baseline));
        }
    }
    bail!("内嵌字体无法完整放入 {CELL_WIDTH}x{CELL_HEIGHT} 单元格")
}

fn rasterize_centered(font: &Font, ch: char, size: f32, baseline: f32) -> Result<[u8; 200]> {
    let (metrics, source) = font.rasterize(ch, size);
    if metrics.width > CELL_WIDTH as usize || metrics.height > CELL_HEIGHT as usize {
        bail!("字符 {ch:?} 超出字体单元格");
    }

    let mut canvas = [0_u8; CELL_PIXELS];
    if metrics.width == 0 || metrics.height == 0 {
        return Ok(canvas);
    }
    let offset_x = ((CELL_WIDTH as usize - metrics.width) / 2) as isize;
    let offset_y = (baseline - (metrics.ymin + metrics.height as i32) as f32)
        .round()
        .max(0.0) as isize;

    for source_y in 0..metrics.height {
        let target_y = offset_y + source_y as isize;
        if !(0..CELL_HEIGHT as isize).contains(&target_y) {
            continue;
        }
        for source_x in 0..metrics.width {
            let target_x = offset_x + source_x as isize;
            if !(0..CELL_WIDTH as isize).contains(&target_x) {
                continue;
            }
            canvas[target_y as usize * CELL_WIDTH as usize + target_x as usize] =
                source[source_y * metrics.width + source_x];
        }
    }
    Ok(canvas)
}

fn raw_shape_vector(bitmap: &[u8]) -> [f32; 6] {
    let mut shape = [0.0_f32; 6];
    for (component, value) in shape.iter_mut().enumerate() {
        let (x0, y0, x1, y1) = cell_region_bounds(0, 0, component);
        let mut count = 0_u32;
        for y in y0..y1 {
            for x in x0..x1 {
                *value += f32::from(bitmap[y as usize * CELL_WIDTH as usize + x as usize]) / 255.0;
                count += 1;
            }
        }
        *value /= count as f32;
    }
    shape
}

fn normalize_shape_vectors(glyphs: &mut [CharGlyph]) -> f32 {
    let mut maximum = 0.0_f32;
    for glyph in glyphs.iter() {
        for value in glyph.shape_vec {
            maximum = maximum.max(value);
        }
    }
    if maximum > 1.0e-6 {
        for glyph in glyphs {
            for value in &mut glyph.shape_vec {
                *value /= maximum;
            }
        }
    }
    maximum
}

pub fn braille_char(bits: u8) -> char {
    char::from_u32(0x2800 + u32::from(bits)).expect("braille scalar")
}

pub fn render_braille_bitmap(bits: u8) -> [u8; 200] {
    let mut bitmap = [0_u8; CELL_PIXELS];
    for (bit, (row, column)) in BRAILLE_LAYOUT.into_iter().enumerate() {
        if bits & (1 << bit) == 0 {
            continue;
        }
        let center_x = if column == 0 { 2.5 } else { 7.5 };
        let center_y = 2.5 + row as f32 * 5.0;
        for y in 0..CELL_HEIGHT as usize {
            for x in 0..CELL_WIDTH as usize {
                let distance = ((x as f32 + 0.5 - center_x).powi(2)
                    + (y as f32 + 0.5 - center_y).powi(2))
                .sqrt();
                let coverage = (2.4 - distance).clamp(0.0, 1.0);
                let index = y * CELL_WIDTH as usize + x;
                bitmap[index] = bitmap[index].max((coverage * 255.0).round() as u8);
            }
        }
    }
    bitmap
}

#[cfg(test)]
mod tests {
    use super::{BRAILLE_LAYOUT, FontAtlas, braille_char, render_braille_bitmap};
    use crate::aiss::glyph_descriptor;
    use crate::preprocess::{CELL_HEIGHT, CELL_WIDTH};

    #[test]
    fn ascii_atlas_has_all_characters_and_no_clipping() {
        let atlas = FontAtlas::load().unwrap();
        assert_eq!(atlas.ascii().len(), 95);
        assert_eq!(atlas.ascii()[0].ch, ' ');
        assert_eq!(atlas.ascii().last().unwrap().ch, '~');
        assert!(
            atlas
                .ascii()
                .iter()
                .all(|glyph| glyph.bitmap.len() == (CELL_WIDTH * CELL_HEIGHT) as usize)
        );
    }

    #[test]
    fn shape_components_share_one_global_scale() {
        let atlas = FontAtlas::load().unwrap();
        let space = &atlas.ascii()[0];
        assert_eq!(space.shape_vec, [0.0; 6]);
        assert_eq!(space.density, 0.0);

        let maximum = atlas
            .ascii()
            .iter()
            .flat_map(|glyph| glyph.shape_vec)
            .fold(0.0_f32, f32::max);
        assert!((maximum - 1.0).abs() < 1.0e-5);

        // 顶部子块本身更稀疏，共享缩放后不应再被放大到 1。
        let top_max = atlas
            .ascii()
            .iter()
            .map(|glyph| glyph.shape_vec[0])
            .fold(0.0_f32, f32::max);
        assert!(top_max < 0.9, "top_max={top_max}");
    }

    #[test]
    fn glyph_vectors_represent_distinct_shapes() {
        let atlas = FontAtlas::load().unwrap();
        let slash = &atlas.ascii()[('/' as usize) - 0x20];
        let underscore = &atlas.ascii()[('_' as usize) - 0x20];
        let at = &atlas.ascii()[('@' as usize) - 0x20];
        assert_ne!(slash.shape_vec, underscore.shape_vec);
        assert_ne!(at.shape_vec, slash.shape_vec);
        assert!(at.density > slash.density.min(underscore.density));
    }

    #[test]
    fn aiss_descriptors_are_built_on_first_use_and_match_the_bitmaps() {
        let atlas = FontAtlas::load().unwrap();
        assert!(!atlas.aiss_descriptors_ready());
        let index = ('D' as usize) - 0x20;
        let expected = glyph_descriptor(&atlas.ascii()[index].bitmap);
        assert!(!atlas.aiss_descriptors_ready());
        assert_eq!(atlas.aiss_descriptors()[index], expected);
        assert!(atlas.aiss_descriptors_ready());
    }

    #[test]
    fn geometric_charsets_have_complete_masks() {
        let atlas = FontAtlas::load().unwrap();
        assert_eq!(atlas.blocks().len(), 33);
        assert_eq!(atlas.sextants().len(), 114);
        for charset in [atlas.blocks(), atlas.sextants()] {
            assert!(charset.iter().all(|glyph| glyph.bitmap.len() == 200));
            let distinct: std::collections::HashSet<&[u8]> =
                charset.iter().map(|glyph| glyph.bitmap.as_ref()).collect();
            assert_eq!(distinct.len(), charset.len(), "charset={}", charset[1].ch);
        }
        let full = atlas
            .sextants()
            .iter()
            .find(|glyph| glyph.ch == '\u{2588}')
            .unwrap();
        assert!((full.density - 1.0).abs() < 1.0e-6);
        assert!(atlas.blocks()[0].bitmap.iter().all(|value| *value == 0));
    }

    #[test]
    fn diagonal_wedges_fill_the_named_side() {
        let atlas = FontAtlas::load().unwrap();
        let glyph = |ch| {
            atlas
                .sextants()
                .iter()
                .find(|glyph| glyph.ch == ch)
                .unwrap()
        };
        let bottom_left = 19 * CELL_WIDTH as usize;
        let bottom_right = 19 * CELL_WIDTH as usize + CELL_WIDTH as usize - 1;

        let lower_left = glyph('\u{1fb40}');
        assert_eq!(lower_left.bitmap[bottom_left], 255);
        assert_eq!(lower_left.bitmap[0], 0);

        let upper_right = glyph('\u{1fb56}');
        assert_eq!(upper_right.bitmap[bottom_left], 0);
        assert_eq!(upper_right.bitmap[0], 255);

        let corner = glyph('\u{1fb47}');
        assert_eq!(corner.bitmap[bottom_right], 255);
        assert_eq!(corner.bitmap[0], 0);
    }

    #[test]
    fn braille_bits_follow_unicode_dot_numbering() {
        let expected_pixels = [
            (2, 2),
            (2, 7),
            (2, 12),
            (7, 2),
            (7, 7),
            (7, 12),
            (2, 17),
            (7, 17),
        ];
        for (bit, ((row, column), (x, y))) in
            BRAILLE_LAYOUT.into_iter().zip(expected_pixels).enumerate()
        {
            assert_eq!((x, y), (2 + column * 5, 2 + row * 5));
            let bitmap = render_braille_bitmap(1 << bit);
            assert!(bitmap[y * CELL_WIDTH as usize + x] > 0);
        }
        assert_eq!(braille_char(0x81), '\u{2881}');
        assert_eq!(braille_char(0xff), '\u{28ff}');
    }

    #[test]
    fn known_braille_codepoints_use_right_top_and_left_bottom_layout() {
        assert_eq!(braille_char(0x08), '\u{2808}');
        let right_top = render_braille_bitmap(0x08);
        assert!(right_top[2 * CELL_WIDTH as usize + 7] > 0);
        assert_eq!(right_top[2 * CELL_WIDTH as usize + 2], 0);

        assert_eq!(braille_char(0x40), '\u{2840}');
        let left_bottom = render_braille_bitmap(0x40);
        assert!(left_bottom[17 * CELL_WIDTH as usize + 2] > 0);
        assert_eq!(left_bottom[17 * CELL_WIDTH as usize + 7], 0);
    }

    #[test]
    fn braille_bitmap_has_real_two_by_four_cell_density() {
        let full = render_braille_bitmap(0xff);
        let empty = render_braille_bitmap(0);
        assert_eq!(empty, [0; 200]);
        assert!(full.iter().filter(|coverage| **coverage > 128).count() >= 32);
    }
}
