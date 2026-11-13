use std::sync::Arc;

use anyhow::{Result, bail};
use fontdue::{Font, FontSettings};

use crate::preprocess::{CELL_HEIGHT, CELL_WIDTH, cell_region_bounds};

const FONT_BYTES: &[u8] = include_bytes!("../assets/JetBrainsMono-Regular.ttf");
const CELL_PIXELS: usize = (CELL_WIDTH * CELL_HEIGHT) as usize;

#[derive(Clone, Debug)]
pub struct CharGlyph {
    pub ch: char,
    pub shape_vec: [f32; 6],
    pub density: f32,
    pub bitmap: Arc<[u8]>,
}

pub struct FontAtlas {
    ascii: Vec<CharGlyph>,
    braille_bitmaps: Vec<Arc<[u8]>>,
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

/// 2x4 Bayer 阈值矩阵。行、列分别对应盲文点位行和列。
pub const BRAILLE_BAYER: [[u8; 2]; 4] = [[0, 4], [2, 6], [1, 5], [3, 7]];

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
                bitmap: Arc::from(bitmap),
            });
        }

        normalize_shape_vectors(&mut ascii);
        let braille_bitmaps = (0..=255)
            .map(|bits| Arc::<[u8]>::from(render_braille_bitmap(bits)))
            .collect();
        Ok(Self {
            ascii,
            braille_bitmaps,
        })
    }

    pub fn ascii(&self) -> &[CharGlyph] {
        &self.ascii
    }

    pub fn ascii_bitmap(&self, ch: char) -> Arc<[u8]> {
        self.ascii[ch as usize - 0x20].bitmap.clone()
    }

    pub fn braille_bitmap(&self, bits: u8) -> Arc<[u8]> {
        self.braille_bitmaps[bits as usize].clone()
    }
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

fn normalize_shape_vectors(glyphs: &mut [CharGlyph]) {
    let mut maxima = [0.0_f32; 6];
    for glyph in glyphs.iter() {
        for (maximum, value) in maxima.iter_mut().zip(glyph.shape_vec) {
            *maximum = maximum.max(value);
        }
    }
    for glyph in glyphs {
        for (value, maximum) in glyph.shape_vec.iter_mut().zip(maxima) {
            if maximum > 1.0e-6 {
                *value /= maximum;
            }
        }
    }
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
    fn space_is_empty_and_shape_components_are_normalized_by_dimension() {
        let atlas = FontAtlas::load().unwrap();
        let space = &atlas.ascii()[0];
        assert_eq!(space.shape_vec, [0.0; 6]);
        assert_eq!(space.density, 0.0);

        for component in 0..6 {
            let maximum = atlas
                .ascii()
                .iter()
                .map(|glyph| glyph.shape_vec[component])
                .fold(0.0_f32, f32::max);
            assert!((maximum - 1.0).abs() < 1.0e-5);
        }
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
