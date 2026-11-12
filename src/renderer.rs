use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use image::{ImageFormat, Rgb, RgbImage};

use crate::matcher::MatchedArt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputKind {
    Terminal,
    Txt,
    Ans,
    Png,
}

impl OutputKind {
    pub fn from_path(path: Option<&Path>) -> Result<Self> {
        let Some(path) = path else {
            return Ok(Self::Terminal);
        };
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase);
        match extension.as_deref() {
            Some("txt") => Ok(Self::Txt),
            Some("ans") => Ok(Self::Ans),
            Some("png") => Ok(Self::Png),
            _ => bail!("输出文件扩展名必须是 .txt、.ans 或 .png"),
        }
    }
}

pub fn render(art: &MatchedArt, kind: OutputKind, output: Option<&Path>) -> Result<()> {
    match kind {
        OutputKind::Terminal => {
            let ansi = render_ansi(art);
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(ansi.as_bytes()).context("无法写入终端")?;
            stdout.flush().context("无法刷新终端输出")?;
        }
        OutputKind::Txt => {
            let path = output.expect("text output requires a path");
            let mut writer = BufWriter::new(
                File::create(path)
                    .with_context(|| format!("无法创建输出文件 {}", path.display()))?,
            );
            writer
                .write_all(render_plain(art).as_bytes())
                .with_context(|| format!("无法写入输出文件 {}", path.display()))?;
            writer.flush().context("无法刷新输出文件")?;
        }
        OutputKind::Ans => {
            let path = output.expect("ansi output requires a path");
            let mut writer = BufWriter::new(
                File::create(path)
                    .with_context(|| format!("无法创建输出文件 {}", path.display()))?,
            );
            writer
                .write_all(render_ansi(art).as_bytes())
                .with_context(|| format!("无法写入输出文件 {}", path.display()))?;
            writer.flush().context("无法刷新输出文件")?;
        }
        OutputKind::Png => {
            let path = output.expect("png output requires a path");
            render_png(art)?
                .save_with_format(path, ImageFormat::Png)
                .with_context(|| format!("无法写入 PNG 文件 {}", path.display()))?;
        }
    }
    Ok(())
}

pub fn render_plain(art: &MatchedArt) -> String {
    let mut output = String::new();
    for row in 0..art.rows {
        for col in 0..art.cols {
            output.push(art.cells[cell_index(art, col, row)].ch);
        }
        output.push('\n');
    }
    output
}

pub fn render_ansi(art: &MatchedArt) -> String {
    let mut output = String::from("\u{1b}[48;2;0;0;0m");
    for row in 0..art.rows {
        for col in 0..art.cols {
            let cell = &art.cells[cell_index(art, col, row)];
            if cell.ch == ' ' {
                output.push(' ');
            } else {
                write!(
                    output,
                    "\u{1b}[38;2;{};{};{}m{}",
                    cell.fg_color[0], cell.fg_color[1], cell.fg_color[2], cell.ch
                )
                .expect("writing to String cannot fail");
            }
        }
        output.push('\n');
    }
    output.push_str("\u{1b}[0m");
    output
}

pub fn render_png(art: &MatchedArt) -> Result<RgbImage> {
    let mut image = RgbImage::from_pixel(art.width, art.height, Rgb([0, 0, 0]));
    ensure!(
        art.cells.len() == usize::try_from(u64::from(art.cols) * u64::from(art.rows))?,
        "内部字符网格尺寸不一致"
    );
    ensure!(
        image.len() == usize::try_from(u64::from(art.width) * u64::from(art.height))? * 3,
        "输出 PNG 尺寸溢出"
    );

    for row in 0..art.rows {
        for col in 0..art.cols {
            let cell = &art.cells[cell_index(art, col, row)];
            for y in 0..cell.bitmap.len() / 10 {
                for x in 0..10 {
                    let coverage = u16::from(cell.bitmap[y * 10 + x]);
                    if coverage == 0 {
                        continue;
                    }
                    let color = cell
                        .fg_color
                        .map(|channel| ((u16::from(channel) * coverage + 127) / 255) as u8);
                    image.put_pixel(col * 10 + x as u32, row * 20 + y as u32, Rgb(color));
                }
            }
        }
    }
    Ok(image)
}

fn cell_index(art: &MatchedArt, col: u32, row: u32) -> usize {
    row as usize * art.cols as usize + col as usize
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::OutputKind;

    #[test]
    fn output_extensions_are_case_insensitive_and_unknown_extensions_fail() {
        assert_eq!(
            OutputKind::from_path(Some(Path::new("output.TXT"))).unwrap(),
            OutputKind::Txt
        );
        assert_eq!(
            OutputKind::from_path(Some(Path::new("output.Ans"))).unwrap(),
            OutputKind::Ans
        );
        assert_eq!(
            OutputKind::from_path(Some(Path::new("output.PnG"))).unwrap(),
            OutputKind::Png
        );
        assert!(OutputKind::from_path(Some(Path::new("output.jpg"))).is_err());
        assert!(OutputKind::from_path(Some(Path::new("output"))).is_err());
    }
}
