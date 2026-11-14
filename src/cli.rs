use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use clap::Parser;

use crate::tone::ToneMap;

pub const DEFAULT_WIDTH: u32 = 160;

#[derive(Debug, Parser)]
#[command(
    name = "ascii-image-studio",
    version,
    about = "将图片转换为黑底 ASCII 或盲文字符画"
)]
pub struct Cli {
    /// 输入图片路径
    pub input_image: PathBuf,

    /// 输出字符列数；终端输出默认取窗口宽度，文件输出默认 160
    #[arg(short = 'w', long, value_parser = positive_u32)]
    pub width: Option<u32>,

    /// 输出文件路径；支持 .txt、.ans 和黑底 .png
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    /// 对比度幂指数，作用于色调映射后的亮度
    #[arg(short = 'c', long, default_value_t = 1.3, value_parser = positive_finite_f32)]
    pub contrast: f32,

    /// 亮度色调映射：clahe 为局部均衡，global 为全局拉伸（旧版色调曲线）
    #[arg(long, value_enum, default_value = "clahe")]
    pub tone_map: ToneMap,

    /// 外部邻域方向锐化指数
    #[arg(short = 'd', long, default_value_t = 2.5, value_parser = positive_finite_f32)]
    pub dir_crunch: f32,

    /// 有符号 DoG 锐化权重；0 表示关闭
    #[arg(short = 'e', long, default_value_t = 0.25, value_parser = unit_interval_finite_f32)]
    pub edge_weight: f32,

    /// 启用抖动：ASCII 用 Floyd-Steinberg，盲文用 2x4 Bayer 有序抖动
    #[arg(long)]
    pub dither: bool,

    /// 保留原图高亮像素颜色
    #[arg(long)]
    pub color: bool,

    /// 使用 2x4 盲文点阵字符
    #[arg(long)]
    pub braille: bool,

    /// 仅供评测导出最终亮度，正常渲染不使用
    #[arg(long, hide = true)]
    pub dump_processed: Option<PathBuf>,

    /// 仅供诊断导出平滑后的方向场，正常渲染不使用
    #[arg(long, hide = true)]
    pub dump_direction: Option<PathBuf>,
}

/// 显式宽度优先；文件输出固定 160；只有终端输出才使用窗口列数。
pub fn resolve_width(
    explicit: Option<u32>,
    output: Option<&Path>,
    terminal_columns: Option<u32>,
) -> u32 {
    if let Some(width) = explicit {
        return width;
    }
    if output.is_some() {
        return DEFAULT_WIDTH;
    }
    terminal_columns.unwrap_or(DEFAULT_WIDTH).max(1)
}

/// 只在标准输出连接真实终端时查询窗口列数。
pub fn terminal_columns() -> Option<u32> {
    if !std::io::stdout().is_terminal() {
        return None;
    }
    terminal_size::terminal_size().map(|(width, _)| u32::from(width.0).max(1))
}

fn positive_u32(value: &str) -> Result<u32, String> {
    let parsed = value
        .parse::<u32>()
        .map_err(|_| "宽度必须是大于 0 的整数".to_owned())?;
    if parsed == 0 {
        return Err("宽度必须大于 0".to_owned());
    }
    Ok(parsed)
}

fn positive_finite_f32(value: &str) -> Result<f32, String> {
    let parsed = parse_finite(value)?;
    if parsed <= 0.0 {
        return Err("参数必须大于 0".to_owned());
    }
    Ok(parsed)
}

fn unit_interval_finite_f32(value: &str) -> Result<f32, String> {
    let parsed = parse_finite(value)?;
    if !(0.0..=1.0).contains(&parsed) {
        return Err("参数必须在 0 到 1 之间".to_owned());
    }
    Ok(parsed)
}

fn parse_finite(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| "参数必须是浮点数".to_owned())?;
    if !parsed.is_finite() {
        return Err("参数必须是有限浮点数".to_owned());
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use clap::Parser;

    use super::{Cli, DEFAULT_WIDTH, resolve_width};
    use crate::tone::ToneMap;

    #[test]
    fn defaults_match_cli_contract() {
        let cli = Cli::try_parse_from(["tool", "input.png"]).unwrap();
        assert_eq!(cli.width, None);
        assert_eq!(cli.contrast, 1.3);
        assert_eq!(cli.tone_map, ToneMap::Clahe);
        assert_eq!(cli.dir_crunch, 2.5);
        assert_eq!(cli.edge_weight, 0.25);
    }

    #[test]
    fn width_resolution_prefers_explicit_then_terminal_then_file_default() {
        let file = Path::new("out.txt");
        assert_eq!(resolve_width(Some(40), None, Some(200)), 40);
        assert_eq!(resolve_width(Some(40), Some(file), Some(200)), 40);
        assert_eq!(resolve_width(None, None, Some(88)), 88);
        assert_eq!(resolve_width(None, None, None), DEFAULT_WIDTH);
        assert_eq!(resolve_width(None, Some(file), Some(200)), DEFAULT_WIDTH);
        assert_eq!(resolve_width(None, Some(file), None), DEFAULT_WIDTH);
        assert_eq!(resolve_width(None, None, Some(0)), 1);
    }

    #[test]
    fn rejects_zero_and_non_finite_values() {
        for args in [
            vec!["tool", "input.png", "--width", "0"],
            vec!["tool", "input.png", "--tone-map", "unknown"],
            vec!["tool", "input.png", "--contrast", "0"],
            vec!["tool", "input.png", "--contrast", "NaN"],
            vec!["tool", "input.png", "--dir-crunch", "inf"],
            vec!["tool", "input.png", "--edge-weight", "-0.1"],
            vec!["tool", "input.png", "--edge-weight", "1.1"],
            vec!["tool", "input.png", "--edge-weight", "NaN"],
        ] {
            assert!(Cli::try_parse_from(args).is_err());
        }
    }
}
