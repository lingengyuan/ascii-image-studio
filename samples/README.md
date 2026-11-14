# 评测样张

固定样张用于 `scripts/eval.py`。替换样张会改变基线，需同步更新 `docs/eval-baseline.md`。

| 文件 | 来源 | 许可 | 用途 |
| --- | --- | --- | --- |
| `portrait.jpg` | [Migrant Mother](https://commons.wikimedia.org/wiki/File:Migrant_Mother_%28LOC_fsa.8b29516%29.jpg) | Public domain | 人像，中间调与皮肤细节 |
| `lineart.jpg` | [Main facade Cathedral of Burgos 1850 Villaamil](https://commons.wikimedia.org/wiki/File:Main_facade_Cathedral_of_Burgos_1850_Villaamil.jpg) | Public domain | 线稿，密集线条 |
| `high-texture.png` | `docs/landscape-original.jpg` 的草地与树木裁切 | CC0 1.0 | 高纹理，草地与树枝 |
| `low-contrast.png` | `docs/landscape-original.jpg` 压缩对比度 | CC0 1.0 | 低对比 |
| `docs/landscape-original.jpg` | [Backlit Margarita Island Sunset](https://commons.wikimedia.org/wiki/File:Backlit_Margarita_Island_Sunset_in_Las_Guevaras,_Venezuela_CaptureNX2.jpg) | CC0 1.0 | 逆光风景 |

生成方法：

- `high-texture.png`：对 `docs/landscape-original.jpg` 裁切 `(0, int(h*0.45), int(w*0.70), h)`，即左下草地与树木区域。
- `low-contrast.png`：把 `docs/landscape-original.jpg` 转为灰度，按 `mean + 0.45 * (gray - mean)` 压缩对比度。
