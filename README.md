# ASCII Image Studio

将图片转换为黑底、等宽字符画。字符由字体 6D 形状向量匹配，并用 DoG、Sobel 和外部邻域方向增强保留轮廓。

## 构建

```bash
cargo build --release
```

可执行文件位于 `target/release/ascii-image-studio`。字体已内嵌，运行时不依赖系统字体。

## 使用

```bash
./target/release/ascii-image-studio photo.jpg
```

```bash
./target/release/ascii-image-studio photo.jpg --width 100 --color
```

```bash
./target/release/ascii-image-studio photo.jpg --dither --output portrait.txt
```

```bash
./target/release/ascii-image-studio photo.jpg --braille --output portrait.png
```

不传 `--output` 时输出到终端。常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-w, --width` | `120` | 输出字符列数 |
| `-c, --contrast` | `1.3` | 全局对比度幂指数 |
| `-d, --dir-crunch` | `2.5` | 外部邻域方向锐化指数 |
| `-e, --edge-weight` | `0.25` | 边缘增强权重，范围为 0 到 1 |
| `--dither` | 关闭 | 使用 Floyd-Steinberg 误差扩散 |
| `--color` | 关闭 | 保留高亮区域颜色 |
| `--braille` | 关闭 | 使用 2x4 盲文点阵匹配 |

## 输出

- `.txt`：纯文本，不包含 ANSI 转义序列。
- `.ans`：24-bit ANSI，黑底前景色。
- `.png`：黑底无损图片，宽高为字符列数乘 10、行数乘 20。
- 终端：24-bit ANSI，黑底前景色。

透明像素先合成到黑底。盲文 PNG 使用标准 2x4 点阵位图绘制，不依赖字体盲文字形。

## 字体许可

内嵌字体为 [JetBrains Mono Regular](https://github.com/JetBrains/JetBrainsMono)，按 SIL Open Font License 1.1 使用。许可证见 `assets/OFL.txt`。
