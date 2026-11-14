use std::path::Path;
use std::process::ExitCode;

use anyhow::{Context, Result};
use ascii_image_studio::cli::{Cli, resolve_width, terminal_columns};
use ascii_image_studio::font_atlas::FontAtlas;
use ascii_image_studio::matcher::{MatchConfig, match_image};
use ascii_image_studio::preprocess::{ProcessedImage, preprocess};
use ascii_image_studio::renderer::{OutputKind, render};
use clap::Parser;
use image::{GrayImage, ImageFormat, Luma, Rgb, RgbImage};

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("错误: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    let output_kind = OutputKind::from_path(cli.output.as_deref())?;
    let terminal = if cli.output.is_none() {
        terminal_columns()
    } else {
        None
    };
    let width = resolve_width(cli.width, cli.output.as_deref(), terminal);
    let image = image::open(&cli.input_image)
        .with_context(|| format!("无法读取输入图片 {}", cli.input_image.display()))?;
    let processed = preprocess(&image, cli.contrast, cli.edge_weight, cli.tone_map)?;
    if let Some(path) = &cli.dump_processed {
        dump_processed(path, &processed)?;
    }
    if let Some(path) = &cli.dump_direction {
        dump_direction(path, &processed)?;
    }
    let atlas = FontAtlas::load().context("无法初始化内嵌字体")?;
    let config = MatchConfig {
        cols: width,
        dir_crunch: cli.dir_crunch,
        edge_weight: cli.edge_weight,
        dither: cli.dither,
        refine: cli.refine,
        color: cli.color,
        braille: cli.braille,
    };
    let art = match_image(&processed, &atlas, &config)?;
    render(&art, output_kind, cli.output.as_deref())
}

/// 评测需要最终亮度，放在匹配前导出；正常渲染路径不受影响。
fn dump_processed(path: &Path, processed: &ProcessedImage) -> Result<()> {
    let mut image = GrayImage::new(processed.width, processed.height);
    for (pixel, value) in image.pixels_mut().zip(processed.luma()) {
        *pixel = Luma([(value.clamp(0.0, 1.0) * 255.0).round() as u8]);
    }
    image
        .save_with_format(path, ImageFormat::Png)
        .with_context(|| format!("无法写入预处理图 {}", path.display()))
}

/// 方向诊断图：R/G 是双角度余弦/正弦，B 是边缘强度；正常渲染不读取。
fn dump_direction(path: &Path, processed: &ProcessedImage) -> Result<()> {
    let (normal_x, normal_y) = processed.edge_normals();
    let strength = processed.edge_strength();
    let mut image = RgbImage::new(processed.width, processed.height);
    for (index, pixel) in image.pixels_mut().enumerate() {
        let nx = normal_x[index];
        let ny = normal_y[index];
        let cos_double = nx * nx - ny * ny;
        let sin_double = 2.0 * nx * ny;
        *pixel = Rgb([
            ((cos_double + 1.0) * 0.5 * 255.0).round().clamp(0.0, 255.0) as u8,
            ((sin_double + 1.0) * 0.5 * 255.0).round().clamp(0.0, 255.0) as u8,
            (strength[index].clamp(0.0, 1.0) * 255.0).round() as u8,
        ]);
    }
    image
        .save_with_format(path, ImageFormat::Png)
        .with_context(|| format!("无法写入方向图 {}", path.display()))
}
