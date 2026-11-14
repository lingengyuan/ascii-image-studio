# 评测基线

- 提交：`5794e59d2df80f079c2540a5da10d5ad99b2fdd4`
- 日期：2026-11-14
- 二进制：`target/release/ascii-image-studio`
- 二进制 SHA-256：`18aa5d5d44d5c3ea45785c593baf0a0068ec6ee97ce878e7464f36293c987901`
- 环境：Python 3.12.13；numpy 2.4.3；scipy 1.17.1；Pillow 12.1.1；opencv-python 4.13.0.92
- 工作区：有未提交改动
- 固定宽度：120 列；模式：plain、dither、refine
- 样张：landscape-original.jpg、portrait.jpg、lineart.jpg、high-texture.png、low-contrast.png

## 结果

| sample | mode | SSIM | SSIM@σ5 | FSIM | FSIM@σ5 | Sobel corr | DoG corr | density diff | density MAE | runtime s | deterministic |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| landscape-original | plain | 0.023166 | 0.273260 | 0.497786 | 0.689457 | -0.038073 | 0.019645 | 0.322584 | 0.322584 | 0.039 | yes |
| landscape-original | dither | 0.024289 | 0.285387 | 0.503197 | 0.687221 | -0.037624 | 0.020330 | 0.321788 | 0.321788 | 0.044 | yes |
| landscape-original | refine | 0.023355 | 0.275808 | 0.497086 | 0.694205 | -0.035884 | 0.019533 | 0.321901 | 0.321901 | 0.043 | yes |
| portrait | plain | 0.031167 | 0.260729 | 0.639088 | 0.685865 | 0.078691 | 0.015524 | 0.301756 | 0.301756 | 0.130 | yes |
| portrait | dither | 0.032111 | 0.274844 | 0.645322 | 0.685571 | 0.074685 | 0.014277 | 0.300863 | 0.300865 | 0.140 | yes |
| portrait | refine | 0.031363 | 0.262624 | 0.640373 | 0.689962 | 0.078982 | 0.015831 | 0.301310 | 0.301310 | 0.142 | yes |
| lineart | plain | 0.009372 | 0.320006 | 0.535852 | 0.676009 | -0.145069 | 0.020939 | 0.455081 | 0.455081 | 0.078 | yes |
| lineart | dither | 0.009517 | 0.324017 | 0.536840 | 0.672786 | -0.146800 | 0.020571 | 0.453065 | 0.453065 | 0.084 | yes |
| lineart | refine | 0.009355 | 0.321034 | 0.535616 | 0.679792 | -0.144192 | 0.020666 | 0.454687 | 0.454687 | 0.084 | yes |
| high-texture | plain | 0.034220 | 0.212225 | 0.554248 | 0.704197 | 0.100534 | 0.029923 | 0.239853 | 0.239853 | 0.026 | yes |
| high-texture | dither | 0.036406 | 0.233402 | 0.562920 | 0.702414 | 0.099835 | 0.032080 | 0.237838 | 0.237838 | 0.028 | yes |
| high-texture | refine | 0.034239 | 0.214433 | 0.554736 | 0.709048 | 0.101320 | 0.029557 | 0.239399 | 0.239399 | 0.027 | yes |
| low-contrast | plain | 0.014511 | 0.283114 | 0.504660 | 0.718422 | -0.025951 | 0.018112 | 0.310953 | 0.310953 | 0.037 | yes |
| low-contrast | dither | 0.015417 | 0.295757 | 0.501365 | 0.712957 | -0.027016 | 0.020717 | 0.309951 | 0.309951 | 0.042 | yes |
| low-contrast | refine | 0.014616 | 0.283981 | 0.503437 | 0.722252 | -0.024779 | 0.018414 | 0.310554 | 0.310554 | 0.042 | yes |

## 指标定义

- 对照目标是预处理后的最终亮度（色调映射和有符号 DoG 之后），不是原图。
- 预处理图保存为 8-bit PNG，再用 OpenCV 缩放到渲染图尺寸：缩小时用 `INTER_AREA`，放大时用 `INTER_LINEAR`。
- SSIM：11x11 Gaussian 窗，sigma=1.5，C1=0.01²，C2=0.03²；只统计 valid 区域（四周各裁 5 px），取 SSIM 图均值。
- SSIM@σ5、FSIM@σ5：两幅图先做 Gaussian 模糊，sigma=5.0，再计算同名指标。
- FSIM 是 FeatureSIM.m 的逐函数移植，包括自动降采样、Butterworth 低通和噪声补偿。
- FSIM 降采样：F = max(1, round(min(rows, cols) / 256))；两图各自做 F×F 均值滤波（零填充 same 卷积），再按 0::F 取两个轴的采样。
- FSIM 相位一致性 phasecong2：4 尺度、4 方向 log-Gabor；minWaveLength=6，mult=2，sigmaOnf=0.55，dThetaOnSigma=1.2，k=2.0，epsilon=1e-4。
- FSIM 频域：完整 fft2/ifft2；Butterworth 低通 cutoff=0.45、order=15 乘到每个 log-Gabor 滤波器；角度 spread 用 atan2 的角距。
- FSIM 噪声：用最小尺度响应平方的中值估计噪声功率；T = EstNoiseEnergy + k·EstNoiseEnergySigma，再除以 1.7；Energy = max(Energy - T, 0)，PC = EnergyAll / AnAll。
- FSIM 梯度：dx=[3 0 -3; 10 0 -10; 3 0 -3]/16，dy=[3 10 3; 0 0 0; -3 -10 -3]/16，same 卷积后取幅值。
- FSIM 公式：S_PC=(2·PC1·PC2+T1)/(PC1²+PC2²+T1)，T1=0.85；S_G=(2·G1·G2+T2)/(G1²+G2²+T2)，T2=160；FSIM=Σ(S_PC·S_G·max(PC1,PC2))/Σmax(PC1,PC2)。
- FSIM 在 PC 权重和为 0 时：两图完全相同返回 1.0，否则返回 0.0。
- Sobel corr：目标图与渲染图 Sobel 梯度幅值图的 Pearson 相关系数。
- DoG corr：目标图与渲染图 DoG 图（Gaussian σ=1.0 减 σ=2.0）的 Pearson 相关系数。
- density diff：两图全局均值之差的绝对值。
- density MAE：按 10x20 像素块取均值后，120 x rows 单元网格的平均绝对误差。
- runtime s：3 次渲染耗时的中位数；计时不含 `--dump-processed`。
- deterministic：导出运行加 3 次计时运行，4 份渲染 PNG 的 SHA-256 完全一致。

## 限制

- 渲染图是字符覆盖的栅格化结果，绝对指标不代表照片级质量，只能在同一套样张、宽度和参数下比较。
- 未模糊的 SSIM、Sobel corr、DoG corr 受字符栅格纹理主导，数值接近 0；判断结构改善主要看 SSIM@σ5、FSIM@σ5 和密度误差。
- `--refine` 的验收口径以 SSIM@σ5、FSIM@σ5 和 density MAE 为准；未模糊 SSIM/FSIM 仅作回归调试，允许微小下降。
- 只实现灰度 FSIM，没有实现彩色 FSIMc；彩色信息不进入指标。
- 卷积边界仍可能与 MATLAB 逐位不同：MATLAB `conv2` 的边界与 OpenCV 零填充在边界像素上可能相差 1 像素；本脚本的 FSIM 用 full 卷积后按 MATLAB `same` 规则裁剪，仍不保证与 MATLAB 逐位一致。
- 预处理图为 8-bit，量化误差已计入指标。
- 运行时间受本机负载影响；更换样张、宽度或默认参数会改变基线。
