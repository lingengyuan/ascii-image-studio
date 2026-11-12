use std::process::ExitCode;

use anyhow::{Context, Result};
use ascii_image_studio::cli::Cli;
use ascii_image_studio::font_atlas::FontAtlas;
use ascii_image_studio::matcher::{MatchConfig, match_image};
use ascii_image_studio::preprocess::preprocess;
use ascii_image_studio::renderer::{OutputKind, render};
use clap::Parser;

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
    let image = image::open(&cli.input_image)
        .with_context(|| format!("无法读取输入图片 {}", cli.input_image.display()))?;
    let processed = preprocess(&image, cli.contrast, cli.edge_weight)?;
    let atlas = FontAtlas::load().context("无法初始化内嵌字体")?;
    let config = MatchConfig {
        cols: cli.width,
        dir_crunch: cli.dir_crunch,
        edge_weight: cli.edge_weight,
        dither: cli.dither,
        color: cli.color,
        braille: cli.braille,
    };
    let art = match_image(&processed, &atlas, &config)?;
    render(&art, output_kind, cli.output.as_deref())
}
