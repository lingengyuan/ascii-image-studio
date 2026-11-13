#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image=${1:?"用法: scripts/compare-grid.sh <图片> [输出目录]"}
bin=${BIN:-"$repo_root/target/release/ascii-image-studio"}
out=${2:-"$repo_root/output/compare/$(basename "${image%.*}")"}

if [[ ! -x "$bin" ]]; then
  echo "缺少可执行文件 $bin，请先运行 cargo build --release" >&2
  exit 1
fi
if ! command -v python3 >/dev/null; then
  echo "拼图需要 python3" >&2
  exit 1
fi
if ! python3 -c 'import PIL' >/dev/null 2>&1; then
  echo "拼图需要 Python Pillow：python3 -m pip install pillow" >&2
  exit 1
fi

mkdir -p "$out"
manifest="$out/manifest.tsv"
: >"$manifest"

for dir_crunch in 1 2.5 4; do
  for edge_weight in 0 0.25 0.5; do
    png="$out/d${dir_crunch}_e${edge_weight}.png"
    "$bin" "$image" -w 80 -d "$dir_crunch" -e "$edge_weight" -o "$png"
    printf '%s\t%s\t%s\n' "$dir_crunch" "$edge_weight" "$png" >>"$manifest"
  done
done

python3 - "$manifest" "$out/grid.png" "$image" <<'PY'
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

manifest, grid_path, source = sys.argv[1:4]
entries = []
for line in Path(manifest).read_text().splitlines():
    dir_crunch, edge_weight, path = line.split("\t")
    entries.append((dir_crunch, edge_weight, Image.open(path).convert("RGB")))

font = ImageFont.load_default(size=18)
label_height = 26
gap = 8
columns = 3
width = max(image.width for _, _, image in entries)
height = max(image.height for _, _, image in entries)
rows = (len(entries) + columns - 1) // columns
header = 34

sheet = Image.new(
    "RGB",
    (columns * width + (columns + 1) * gap, header + rows * (height + label_height) + (rows + 1) * gap),
    (14, 14, 14),
)
draw = ImageDraw.Draw(sheet)
draw.text(
    (gap, 8),
    f"{Path(source).name} · width=80 · default contrast=1.1 / clahe",
    fill=(235, 235, 235),
    font=font,
)

for index, (dir_crunch, edge_weight, image) in enumerate(entries):
    column = index % columns
    row = index // columns
    x = gap + column * (width + gap)
    y = header + gap + row * (height + label_height + gap)
    draw.text(
        (x, y),
        f"-d {dir_crunch}    -e {edge_weight}",
        fill=(220, 220, 220),
        font=font,
    )
    sheet.paste(image, (x, y + label_height))

sheet.save(grid_path)
print(f"拼图: {grid_path}")
PY
