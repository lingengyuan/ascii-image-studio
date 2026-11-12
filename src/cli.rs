use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "ascii-image-studio",
    version,
    about = "将图片转换为黑底 ASCII 或盲文字符画"
)]
pub struct Cli {
    /// 输入图片路径
    pub input_image: PathBuf,

    /// 输出字符列数
    #[arg(short = 'w', long, default_value_t = 120, value_parser = positive_u32)]
    pub width: u32,

    /// 输出文件路径；支持 .txt、.ans 和黑底 .png
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    /// 全局对比度幂指数
    #[arg(short = 'c', long, default_value_t = 1.3, value_parser = positive_finite_f32)]
    pub contrast: f32,

    /// 外部邻域方向锐化指数
    #[arg(short = 'd', long, default_value_t = 2.5, value_parser = positive_finite_f32)]
    pub dir_crunch: f32,

    /// DoG 与 Sobel 边缘权重；0 表示关闭
    #[arg(short = 'e', long, default_value_t = 0.25, value_parser = unit_interval_finite_f32)]
    pub edge_weight: f32,

    /// 启用 Floyd-Steinberg 误差扩散
    #[arg(long)]
    pub dither: bool,

    /// 保留原图高亮像素颜色
    #[arg(long)]
    pub color: bool,

    /// 使用 2x4 盲文点阵字符
    #[arg(long)]
    pub braille: bool,
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
    use clap::Parser;

    use super::Cli;

    #[test]
    fn defaults_match_cli_contract() {
        let cli = Cli::try_parse_from(["tool", "input.png"]).unwrap();
        assert_eq!(cli.width, 120);
        assert_eq!(cli.contrast, 1.3);
        assert_eq!(cli.dir_crunch, 2.5);
        assert_eq!(cli.edge_weight, 0.25);
    }

    #[test]
    fn rejects_zero_and_non_finite_values() {
        for args in [
            vec!["tool", "input.png", "--width", "0"],
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
