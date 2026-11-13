use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

use image::{DynamicImage, Rgb, RgbImage, Rgba, RgbaImage};

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ascii-image-studio-{name}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_ascii-image-studio")
}

fn run(args: &[&str]) -> Output {
    Command::new(bin()).args(args).output().unwrap()
}

fn write_gradient(path: &Path) {
    let mut image = RgbImage::new(40, 24);
    for y in 0..image.height() {
        for x in 0..image.width() {
            let red = (x * 255 / image.width().saturating_sub(1)) as u8;
            let green = (y * 255 / image.height().saturating_sub(1)) as u8;
            image.put_pixel(x, y, Rgb([red, green, 96]));
        }
    }
    DynamicImage::ImageRgb8(image).save(path).unwrap();
}

fn path_text(path: &Path) -> &str {
    path.to_str().unwrap()
}

fn strip_ansi(input: &str) -> String {
    let mut output = String::new();
    let mut chars = input.chars();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            for next in chars.by_ref() {
                if next == 'm' {
                    break;
                }
            }
        } else {
            output.push(ch);
        }
    }
    output
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "status={:?}\nstdout={}\nstderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cli_smoke_covers_terminal_txt_ans_and_png() {
    let temp = TempDir::new("formats");
    let input = temp.path("input.png");
    write_gradient(&input);

    let terminal = run(&[
        path_text(&input),
        "--width",
        "12",
        "--contrast",
        "1.3",
        "--edge-weight",
        "0.25",
        "--color",
    ]);
    assert_success(&terminal);
    let terminal = String::from_utf8(terminal.stdout).unwrap();
    assert!(terminal.contains("\u{1b}[48;2;0;0;0m"));
    assert!(terminal.contains("\u{1b}[38;2;"));
    assert!(terminal.ends_with("\u{1b}[0m"));

    let txt = temp.path("plain.TXT");
    let txt_output = run(&[
        path_text(&input),
        "--width",
        "12",
        "--output",
        path_text(&txt),
    ]);
    assert_success(&txt_output);
    let text = fs::read_to_string(&txt).unwrap();
    assert!(!text.contains('\u{1b}'));
    assert_eq!(text.lines().count(), 4);

    let ans = temp.path("color.Ans");
    let ans_output = run(&[
        path_text(&input),
        "--width",
        "12",
        "--color",
        "--output",
        path_text(&ans),
    ]);
    assert_success(&ans_output);
    let ansi = fs::read_to_string(&ans).unwrap();
    assert!(ansi.contains("\u{1b}[48;2;0;0;0m"));
    assert!(ansi.contains("\u{1b}[38;2;"));

    let png = temp.path("mono.PNG");
    let png_output = run(&[
        path_text(&input),
        "--width",
        "12",
        "--output",
        path_text(&png),
    ]);
    assert_success(&png_output);
    let rendered = image::open(&png).unwrap().to_rgb8();
    assert_eq!(rendered.dimensions(), (120, 80));
    assert!(
        rendered
            .pixels()
            .all(|pixel| pixel[0] == pixel[1] && pixel[1] == pixel[2])
    );
    assert!(rendered.pixels().any(|pixel| *pixel != Rgb([0, 0, 0])));

    let color_png = temp.path("color.png");
    let color_png_output = run(&[
        path_text(&input),
        "--width",
        "12",
        "--color",
        "--output",
        path_text(&color_png),
    ]);
    assert_success(&color_png_output);
    let colored = image::open(&color_png).unwrap().to_rgb8();
    assert!(
        colored
            .pixels()
            .any(|pixel| pixel[0] != pixel[1] || pixel[1] != pixel[2])
    );
}

#[test]
fn cli_smoke_covers_braille_dither_and_transparent_input() {
    let temp = TempDir::new("algorithms");
    let input = temp.path("input.png");
    write_gradient(&input);

    let braille_txt = temp.path("braille.txt");
    let braille = run(&[
        path_text(&input),
        "--width",
        "16",
        "--contrast",
        "1",
        "--edge-weight",
        "0",
        "--braille",
        "--output",
        path_text(&braille_txt),
    ]);
    assert_success(&braille);
    let braille_text = fs::read_to_string(&braille_txt).unwrap();
    assert!(
        braille_text
            .chars()
            .filter(|ch| *ch != '\n')
            .all(|ch| ('\u{2800}'..='\u{28ff}').contains(&ch))
    );
    assert!(
        braille_text
            .chars()
            .any(|ch| ch != '\u{2800}' && ch != '\n')
    );

    let braille_dithered_txt = temp.path("braille-dithered.txt");
    assert_success(&run(&[
        path_text(&input),
        "--width",
        "16",
        "--contrast",
        "1",
        "--edge-weight",
        "0",
        "--braille",
        "--dither",
        "--output",
        path_text(&braille_dithered_txt),
    ]));
    assert_ne!(
        fs::read(&braille_txt).unwrap(),
        fs::read(&braille_dithered_txt).unwrap()
    );

    let braille_png = temp.path("braille.png");
    let braille_png_output = run(&[
        path_text(&input),
        "--width",
        "16",
        "--contrast",
        "1",
        "--edge-weight",
        "0",
        "--braille",
        "--output",
        path_text(&braille_png),
    ]);
    assert_success(&braille_png_output);
    let braille_image = image::open(&braille_png).unwrap().to_rgb8();
    assert!(braille_image.pixels().any(|pixel| *pixel != Rgb([0, 0, 0])));

    let plain = temp.path("plain.txt");
    let dithered = temp.path("dithered.txt");
    assert_success(&run(&[
        path_text(&input),
        "--width",
        "32",
        "--contrast",
        "1",
        "--edge-weight",
        "0",
        "--output",
        path_text(&plain),
    ]));
    assert_success(&run(&[
        path_text(&input),
        "--width",
        "32",
        "--contrast",
        "1",
        "--edge-weight",
        "0",
        "--dither",
        "--output",
        path_text(&dithered),
    ]));
    assert_ne!(fs::read(&plain).unwrap(), fs::read(&dithered).unwrap());

    let transparent = temp.path("transparent.png");
    RgbaImage::from_pixel(8, 8, Rgba([255, 255, 255, 0]))
        .save(&transparent)
        .unwrap();
    let transparent_txt = temp.path("transparent.txt");
    assert_success(&run(&[
        path_text(&transparent),
        "--width",
        "8",
        "--contrast",
        "1",
        "--edge-weight",
        "0",
        "--output",
        path_text(&transparent_txt),
    ]));
    assert!(
        fs::read_to_string(transparent_txt)
            .unwrap()
            .chars()
            .all(|ch| ch == ' ' || ch == '\n')
    );
}

#[test]
fn cli_accepts_jpeg_and_webp_inputs() {
    let temp = TempDir::new("decoders");
    let png = temp.path("source.png");
    write_gradient(&png);
    let source = image::open(&png).unwrap();

    for extension in ["jpg", "webp"] {
        let input = temp.path(&format!("input.{extension}"));
        source.save(&input).unwrap();
        let output = temp.path(&format!("out-{extension}.txt"));
        assert_success(&run(&[
            path_text(&input),
            "--width",
            "12",
            "--contrast",
            "1",
            "--edge-weight",
            "0",
            "--output",
            path_text(&output),
        ]));
        let text = fs::read_to_string(&output).unwrap();
        assert_eq!(text.lines().count(), 4, "extension={extension}");
        assert!(
            text.lines().all(|line| line.chars().count() == 12),
            "extension={extension}"
        );
    }
}

#[test]
fn cli_smoke_covers_constant_and_empty_inputs() {
    let temp = TempDir::new("degenerate");
    let white = temp.path("white.png");
    RgbImage::from_pixel(8, 8, Rgb([255, 255, 255]))
        .save(&white)
        .unwrap();
    let output = run(&[
        path_text(&white),
        "--width",
        "8",
        "--contrast",
        "1.3",
        "--edge-weight",
        "0",
    ]);
    assert_success(&output);
    assert!(!output.stdout.is_empty());

    let empty = temp.path("empty.png");
    fs::write(&empty, []).unwrap();
    let empty_output = run(&[path_text(&empty)]);
    assert_eq!(empty_output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&empty_output.stderr).contains("无法读取输入图片"));
}

#[test]
fn cli_rejects_invalid_values_extensions_and_overflow() {
    let temp = TempDir::new("errors");
    let input = temp.path("input.png");
    write_gradient(&input);

    let invalid_values = [
        vec![path_text(&input), "--width", "0"],
        vec![path_text(&input), "--tone-map", "unknown"],
        vec![path_text(&input), "--contrast", "0"],
        vec![path_text(&input), "--contrast", "NaN"],
        vec![path_text(&input), "--dir-crunch", "inf"],
        vec![path_text(&input), "--edge-weight", "-0.1"],
        vec![path_text(&input), "--edge-weight", "1.1"],
        vec![path_text(&input), "--edge-weight", "NaN"],
    ];
    for args in invalid_values {
        assert_eq!(run(&args).status.code(), Some(2));
    }

    let unknown = temp.path("output.jpg");
    let unknown_output = run(&[path_text(&input), "--output", path_text(&unknown)]);
    assert_eq!(unknown_output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&unknown_output.stderr).contains("扩展名"));

    let overflow_output = run(&[path_text(&input), "--width", "4294967295"]);
    assert_eq!(overflow_output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&overflow_output.stderr).contains("溢出"));
}

#[test]
fn cli_default_width_is_used_only_for_terminal_output() {
    let temp = TempDir::new("width");
    let input = temp.path("input.png");
    write_gradient(&input);

    let file_output = temp.path("default.txt");
    assert_success(&run(&[
        path_text(&input),
        "--output",
        path_text(&file_output),
    ]));
    assert!(
        fs::read_to_string(&file_output)
            .unwrap()
            .lines()
            .all(|line| line.chars().count() == 120)
    );

    let narrow_output = temp.path("narrow.txt");
    assert_success(&run(&[
        path_text(&input),
        "--width",
        "40",
        "--output",
        path_text(&narrow_output),
    ]));
    assert!(
        fs::read_to_string(&narrow_output)
            .unwrap()
            .lines()
            .all(|line| line.chars().count() == 40)
    );

    // 测试进程的 stdout 是管道，不是终端，应退回 120 列。
    let terminal = run(&[path_text(&input), "--contrast", "1", "--edge-weight", "0"]);
    assert_success(&terminal);
    let plain = strip_ansi(&String::from_utf8(terminal.stdout).unwrap());
    assert_eq!(plain.lines().next().unwrap().chars().count(), 120);
}

#[test]
fn cli_tone_map_modes_differ_on_low_contrast_input() {
    let temp = TempDir::new("tone");
    let input = temp.path("low.png");
    let mut image = RgbImage::new(48, 48);
    for y in 0..image.height() {
        for x in 0..image.width() {
            let value = (100 + (x + y) * 50 / 92) as u8;
            image.put_pixel(x, y, Rgb([value, value, value]));
        }
    }
    DynamicImage::ImageRgb8(image).save(&input).unwrap();

    let global = temp.path("global.txt");
    let local = temp.path("local.txt");
    for (path, mode) in [(&global, "global"), (&local, "clahe")] {
        assert_success(&run(&[
            path_text(&input),
            "--width",
            "24",
            "--contrast",
            "1",
            "--edge-weight",
            "0",
            "--tone-map",
            mode,
            "--output",
            path_text(path),
        ]));
    }
    assert_ne!(fs::read(&global).unwrap(), fs::read(&local).unwrap());
}

#[test]
fn cli_mono_png_uses_cell_luminance_for_foreground() {
    let temp = TempDir::new("foreground");
    let input = temp.path("gray.png");
    RgbImage::from_pixel(20, 20, Rgb([128, 128, 128]))
        .save(&input)
        .unwrap();
    let output = temp.path("gray-out.png");
    assert_success(&run(&[
        path_text(&input),
        "--width",
        "4",
        "--contrast",
        "1",
        "--edge-weight",
        "0",
        "--tone-map",
        "global",
        "--output",
        path_text(&output),
    ]));
    let rendered = image::open(&output).unwrap().to_rgb8();
    assert!(
        rendered
            .pixels()
            .all(|pixel| pixel[0] == pixel[1] && pixel[1] == pixel[2])
    );
    assert_eq!(rendered.pixels().map(|pixel| pixel[0]).max(), Some(160));
}

#[test]
fn cli_handles_single_row_and_single_column_inputs() {
    let temp = TempDir::new("aspect");
    let wide = temp.path("wide.png");
    RgbImage::from_pixel(200, 1, Rgb([120, 120, 120]))
        .save(&wide)
        .unwrap();
    let wide_output = temp.path("wide-out.png");
    assert_success(&run(&[
        path_text(&wide),
        "--width",
        "20",
        "--contrast",
        "1",
        "--edge-weight",
        "0",
        "--output",
        path_text(&wide_output),
    ]));
    assert_eq!(
        image::open(&wide_output).unwrap().to_rgb8().dimensions(),
        (200, 20)
    );

    let tall = temp.path("tall.png");
    RgbImage::from_pixel(1, 200, Rgb([120, 120, 120]))
        .save(&tall)
        .unwrap();
    let tall_output = temp.path("tall-out.png");
    assert_success(&run(&[
        path_text(&tall),
        "--width",
        "2",
        "--contrast",
        "1",
        "--edge-weight",
        "0",
        "--output",
        path_text(&tall_output),
    ]));
    assert_eq!(
        image::open(&tall_output).unwrap().to_rgb8().dimensions(),
        (20, 4000)
    );
}

#[test]
fn cli_braille_dither_changes_uniform_gray_output() {
    let temp = TempDir::new("braille-dither");
    let input = temp.path("gray.png");
    RgbImage::from_pixel(20, 20, Rgb([128, 128, 128]))
        .save(&input)
        .unwrap();
    let plain = temp.path("plain.txt");
    let dithered = temp.path("dithered.txt");
    for (path, extra) in [(&plain, None), (&dithered, Some("--dither"))] {
        let mut args = vec![
            path_text(&input),
            "--width",
            "4",
            "--contrast",
            "1",
            "--edge-weight",
            "0",
            "--tone-map",
            "global",
            "--braille",
        ];
        if let Some(extra) = extra {
            args.push(extra);
        }
        args.extend(["--output", path_text(path)]);
        assert_success(&run(&args));
    }
    assert_ne!(fs::read(&plain).unwrap(), fs::read(&dithered).unwrap());
    let plain_text = fs::read_to_string(&plain).unwrap();
    assert!(
        plain_text
            .chars()
            .filter(|ch| *ch != '\n')
            .all(|ch| ch == '\u{28ff}')
    );
}
