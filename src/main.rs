use std::process::ExitCode;

use anyhow::{Context, Result};
use ascii_image_studio::cli::{Cli, resolve_width, terminal_columns};
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
    let terminal = if cli.output.is_none() {
        terminal_columns()
    } else {
        None
    };
    let width = resolve_width(cli.width, cli.output.as_deref(), terminal);
    let image = image::open(&cli.input_image)
        .with_context(|| format!("无法读取输入图片 {}", cli.input_image.display()))?;
    let processed = preprocess(&image, cli.contrast, cli.edge_weight, cli.tone_map)?;
    let atlas = FontAtlas::load().context("无法初始化内嵌字体")?;
    let config = MatchConfig {
        cols: width,
        dir_crunch: cli.dir_crunch,
        edge_weight: cli.edge_weight,
        dither: cli.dither,
        color: cli.color,
        braille: cli.braille,
        plain_text: output_kind == OutputKind::Txt,
    };
    let art = match_image(&processed, &atlas, &config)?;
    render(&art, output_kind, cli.output.as_deref())
}
