#!/usr/bin/env python3
"""ASCII Image Studio P0 评测基线。

固定样张、宽度和模式，重复渲染后计算渲染图与预处理目标图之间的指标。
目标图来自 --dump-processed：色调映射和有符号 DoG 之后的最终亮度，不是原图。

FSIM 逐函数移植自 FeatureSIM.m（Zhang et al., IEEE TIP 2011）：
自动降采样、phasecong2、Butterworth 低通、噪声补偿和 1/16 Scharr 梯度。
SSIM 使用 valid 区域。指标只用于同一套样张和参数下的回归比较。

上游来源与许可：
FeatureSIM.m, https://cslinzhang.github.io/home/FSIM/Files/FeatureSIM.m
Copyright (c) 2010 Lin Zhang, Lei Zhang, Xuanqin Mou and David Zhang. All Rights Reserved.
Permission to use, copy, or modify this software and its documentation for educational
and research purposes only and without fee is hereby granted, provided that this
copyright notice and the original authors' names appear on all copies and supporting
documentation. This program shall not be used, rewritten, or adapted as the basis of a
commercial software or hardware product without first obtaining permission of the
authors. The authors make no representations about the suitability of this software for
any purpose. It is provided "as is" without express or implied warranty.

phasecong2 is embedded in FeatureSIM.m.
Copyright (c) 1996-2009 Peter Kovesi
School of Computer Science & Software Engineering
The University of Western Australia
http://www.csse.uwa.edu.au/
Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, subject to the following conditions: The above copyright notice and
this permission notice shall be included in all copies or substantial portions of the
Software. The software is provided "as is", without warranty of any kind.

lowpassfilter is embedded in FeatureSIM.m.
Copyright (c) 1999 Peter Kovesi
School of Computer Science & Software Engineering
The University of Western Australia
http://www.csse.uwa.edu.au/
Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, subject to the following conditions: The above copyright notice and
this permission notice shall be included in all copies or substantial portions of the
Software. The Software is provided "as is", without warranty of any kind.

This port is not covered by the repository MIT license. See scripts/NOTICE-eval.md.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = ROOT / "target" / "eval"
BASELINE_DOC = ROOT / "docs" / "eval-baseline.md"

WIDTH = 120
CELL_WIDTH = 10
CELL_HEIGHT = 20
HVS_SIGMA = 5.0

SSIM_SIGMA = 1.5
SSIM_WINDOW = 11
SSIM_C1 = 0.01**2
SSIM_C2 = 0.03**2

FSIM_T1 = 0.85
FSIM_T2 = 160.0
FSIM_NSCALE = 4
FSIM_NORIENT = 4
FSIM_MIN_WAVELENGTH = 6.0
FSIM_MULT = 2.0
FSIM_SIGMA_ONF = 0.55
FSIM_DTHETA_ON_SIGMA = 1.2
FSIM_K = 2.0
FSIM_EPSILON = 1.0e-4
FSIM_LOWPASS_CUTOFF = 0.45
FSIM_LOWPASS_ORDER = 15
FSIM_DX = np.array([[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]]) / 16.0
FSIM_DY = np.array([[3.0, 10.0, 3.0], [0.0, 0.0, 0.0], [-3.0, -10.0, -3.0]]) / 16.0
FSIM_WEIGHT_EPSILON = 1.0e-12

SAMPLES = [
    ROOT / "docs" / "landscape-original.jpg",
    ROOT / "samples" / "portrait.jpg",
    ROOT / "samples" / "lineart.jpg",
    ROOT / "samples" / "high-texture.png",
    ROOT / "samples" / "low-contrast.png",
]
MODES = [("plain", []), ("dither", ["--dither"]), ("refine", ["--refine"])]
METRIC_KEYS = [
    "ssim",
    "ssim_hvs",
    "fsim",
    "fsim_hvs",
    "sobel_corr",
    "dog_corr",
    "density_diff",
    "density_mae",
]

_SSIM_AXIS = np.exp(
    -((np.arange(SSIM_WINDOW, dtype=np.float64) - (SSIM_WINDOW - 1) / 2.0) ** 2)
    / (2.0 * SSIM_SIGMA**2)
)
_SSIM_KERNEL = np.outer(_SSIM_AXIS, _SSIM_AXIS)
_SSIM_KERNEL /= _SSIM_KERNEL.sum()


def run_checked(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)


def ensure_binary() -> Path:
    """每次先构建，避免脚本悄悄评测旧二进制；路径跟随 CARGO_TARGET_DIR。"""
    subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=ROOT, check=True)
    result = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    target_directory = Path(json.loads(result.stdout)["target_directory"])
    executable = "ascii-image-studio.exe" if sys.platform == "win32" else "ascii-image-studio"
    binary = target_directory / "release" / executable
    if not binary.is_file():
        raise RuntimeError(f"cargo 构建后未找到二进制: {binary}")
    return binary


def render_once(
    binary: Path,
    sample: Path,
    mode_flags: list[str],
    output: Path,
    dump_processed: Path | None = None,
) -> None:
    command = [
        str(binary),
        str(sample),
        "-w",
        str(WIDTH),
        "-o",
        str(output),
        *mode_flags,
    ]
    if dump_processed is not None:
        command += ["--dump-processed", str(dump_processed)]
    run_checked(command)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": metadata.version("numpy"),
        "scipy": metadata.version("scipy"),
        "pillow": metadata.version("Pillow"),
        "opencv_python": metadata.version("opencv-python"),
    }


def load_gray(path: Path) -> np.ndarray:
    """返回 0..255 灰度图；FeatureSIM.m 的输入动态范围是 0..255。"""
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64)


def hvs_blur(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(
        image, (0, 0), sigmaX=HVS_SIGMA, sigmaY=HVS_SIGMA, borderType=cv2.BORDER_REFLECT
    )


def conv2_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """MATLAB conv2(..., 'same')：full 卷积后每轴从起点裁掉 floor(size/2)。"""
    full = signal.convolve2d(image, kernel, mode="full")
    offset_y = kernel.shape[0] // 2
    offset_x = kernel.shape[1] // 2
    return full[offset_y : offset_y + image.shape[0], offset_x : offset_x + image.shape[1]]


def ssim(reference: np.ndarray, compared: np.ndarray) -> float:
    if reference.shape[0] < SSIM_WINDOW or reference.shape[1] < SSIM_WINDOW:
        raise ValueError("SSIM 需要至少 11x11 的输入")

    def valid_filter(image: np.ndarray) -> np.ndarray:
        filtered = cv2.filter2D(image, cv2.CV_64F, _SSIM_KERNEL, borderType=cv2.BORDER_CONSTANT)
        radius = SSIM_WINDOW // 2
        return filtered[radius:-radius, radius:-radius]

    mu_ref = valid_filter(reference)
    mu_cmp = valid_filter(compared)
    mu_ref_sq = mu_ref * mu_ref
    mu_cmp_sq = mu_cmp * mu_cmp
    mu_ref_cmp = mu_ref * mu_cmp

    sigma_ref_sq = valid_filter(reference * reference) - mu_ref_sq
    sigma_cmp_sq = valid_filter(compared * compared) - mu_cmp_sq
    sigma_ref_cmp = valid_filter(reference * compared) - mu_ref_cmp

    numerator = (2.0 * mu_ref_cmp + SSIM_C1) * (2.0 * sigma_ref_cmp + SSIM_C2)
    denominator = (mu_ref_sq + mu_cmp_sq + SSIM_C1) * (sigma_ref_sq + sigma_cmp_sq + SSIM_C2)
    return float(np.mean(numerator / denominator, dtype=np.float64))


def matlab_round(value: float) -> int:
    return int(math.floor(value + 0.5))


def frequency_range(size: int) -> np.ndarray:
    if size % 2:
        return np.arange(-(size - 1) / 2.0, (size - 1) / 2.0 + 1.0) / (size - 1)
    return np.arange(-size / 2.0, size / 2.0) / size


def lowpass_filter(rows: int, cols: int, cutoff: float, order: int) -> np.ndarray:
    x, y = np.meshgrid(frequency_range(cols), frequency_range(rows))
    radius = np.sqrt(x * x + y * y)
    return np.fft.ifftshift(1.0 / (1.0 + (radius / cutoff) ** (2 * order)))


def phase_congruency(image: np.ndarray) -> np.ndarray:
    rows, cols = image.shape
    image_fft = np.fft.fft2(image)

    x, y = np.meshgrid(frequency_range(cols), frequency_range(rows))
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan2(-y, x)
    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)
    radius[0, 0] = 1.0
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    lowpass = lowpass_filter(rows, cols, FSIM_LOWPASS_CUTOFF, FSIM_LOWPASS_ORDER)
    log_sigma_onf = math.log(FSIM_SIGMA_ONF)
    log_gabor: list[np.ndarray] = []
    for scale in range(FSIM_NSCALE):
        wavelength = FSIM_MIN_WAVELENGTH * FSIM_MULT**scale
        center_frequency = 1.0 / wavelength
        radial = np.exp(
            -((np.log(radius / center_frequency)) ** 2) / (2.0 * log_sigma_onf**2)
        )
        radial *= lowpass
        radial[0, 0] = 0.0
        log_gabor.append(radial)

    theta_sigma = math.pi / FSIM_NORIENT / FSIM_DTHETA_ON_SIGMA
    spreads: list[np.ndarray] = []
    for orientation in range(FSIM_NORIENT):
        angle = orientation * math.pi / FSIM_NORIENT
        ds = sintheta * math.cos(angle) - costheta * math.sin(angle)
        dc = costheta * math.cos(angle) + sintheta * math.sin(angle)
        dtheta = np.abs(np.arctan2(ds, dc))
        spreads.append(np.exp(-(dtheta**2) / (2.0 * theta_sigma**2)))

    zero = np.zeros((rows, cols), dtype=np.float64)
    energy_all = zero.copy()
    an_all = zero.copy()

    for orientation in range(FSIM_NORIENT):
        sum_e = zero.copy()
        sum_o = zero.copy()
        sum_an = zero.copy()
        energy = zero.copy()
        responses: list[np.ndarray] = []
        ifft_filters: list[np.ndarray] = []
        em_n = 0.0

        for scale in range(FSIM_NSCALE):
            filt = log_gabor[scale] * spreads[orientation]
            ifft_filter = np.real(np.fft.ifft2(filt)) * math.sqrt(rows * cols)
            ifft_filters.append(ifft_filter)
            response = np.fft.ifft2(image_fft * filt)
            responses.append(response)
            amplitude = np.abs(response)
            sum_an += amplitude
            sum_e += np.real(response)
            sum_o += np.imag(response)
            if scale == 0:
                em_n = float(np.sum(filt**2))

        x_energy = np.sqrt(sum_e**2 + sum_o**2) + FSIM_EPSILON
        mean_e = sum_e / x_energy
        mean_o = sum_o / x_energy

        for scale in range(FSIM_NSCALE):
            even = np.real(responses[scale])
            odd = np.imag(responses[scale])
            energy += even * mean_e + odd * mean_o - np.abs(even * mean_o - odd * mean_e)

        median_e2n = float(np.median(np.abs(responses[0]) ** 2))
        mean_e2n = -median_e2n / math.log(0.5)
        noise_power = mean_e2n / em_n

        est_sum_an2 = zero.copy()
        for scale in range(FSIM_NSCALE):
            est_sum_an2 += ifft_filters[scale] ** 2
        est_sum_aiaj = zero.copy()
        for first in range(FSIM_NSCALE - 1):
            for second in range(first + 1, FSIM_NSCALE):
                est_sum_aiaj += ifft_filters[first] * ifft_filters[second]

        est_noise_energy2 = 2.0 * noise_power * float(np.sum(est_sum_an2)) + (
            4.0 * noise_power * float(np.sum(est_sum_aiaj))
        )
        tau = math.sqrt(est_noise_energy2 / 2.0) if est_noise_energy2 >= 0.0 else math.nan
        est_noise_energy = tau * math.sqrt(math.pi / 2.0)
        est_noise_energy_sigma = math.sqrt((2.0 - math.pi / 2.0) * tau**2)
        threshold = (est_noise_energy + FSIM_K * est_noise_energy_sigma) / 1.7

        energy = np.maximum(energy - threshold, zero)
        energy_all += energy
        an_all += sum_an

    result_pc = np.zeros_like(energy_all)
    # 常量区域 AnAll 为 0；MATLAB 这里得到 NaN，显式记为 PC=0。
    np.divide(energy_all, an_all, out=result_pc, where=an_all != 0)
    return result_pc


def fsim_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gradient_x = conv2_same(image, FSIM_DX)
    gradient_y = conv2_same(image, FSIM_DY)
    return np.sqrt(gradient_x**2 + gradient_y**2)


def downsample(image: np.ndarray) -> np.ndarray:
    rows, cols = image.shape
    factor = max(1, matlab_round(min(rows, cols) / 256.0))
    kernel = np.full((factor, factor), 1.0 / (factor * factor), dtype=np.float64)
    averaged = conv2_same(image, kernel)
    return averaged[0::factor, 0::factor]


def fsim(reference: np.ndarray, compared: np.ndarray) -> float:
    downsampled_reference = downsample(reference)
    downsampled_compared = downsample(compared)
    pc_reference = phase_congruency(downsampled_reference)
    pc_compared = phase_congruency(downsampled_compared)
    gradient_reference = fsim_gradient_magnitude(downsampled_reference)
    gradient_compared = fsim_gradient_magnitude(downsampled_compared)

    similarity_pc = (2.0 * pc_reference * pc_compared + FSIM_T1) / (
        pc_reference * pc_reference + pc_compared * pc_compared + FSIM_T1
    )
    similarity_gradient = (2.0 * gradient_reference * gradient_compared + FSIM_T2) / (
        gradient_reference * gradient_reference
        + gradient_compared * gradient_compared
        + FSIM_T2
    )
    weight = np.maximum(pc_reference, pc_compared)
    weight_sum = float(np.sum(weight, dtype=np.float64))
    if weight_sum <= FSIM_WEIGHT_EPSILON:
        # PC 全零时权重无定义：相同图视为完全相似，否则视为不相似。
        return 1.0 if np.array_equal(reference, compared) else 0.0
    return float(
        np.sum(similarity_pc * similarity_gradient * weight, dtype=np.float64) / weight_sum
    )


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    gray = image * 255.0
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(gradient_x, gradient_y)


def difference_of_gaussians(image: np.ndarray) -> np.ndarray:
    fine = cv2.GaussianBlur(image, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT)
    coarse = cv2.GaussianBlur(image, (0, 0), 2.0, borderType=cv2.BORDER_REFLECT)
    return fine - coarse


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = left.ravel().astype(np.float64)
    right = right.ravel().astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    if denominator <= 1.0e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def density_metrics(reference: np.ndarray, compared: np.ndarray) -> tuple[float, float]:
    height, width = compared.shape
    if width != WIDTH * CELL_WIDTH or height % CELL_HEIGHT != 0:
        raise RuntimeError(f"渲染图尺寸不符合预期: {width}x{height}")
    rows = height // CELL_HEIGHT
    reference_cells = reference.reshape(rows, CELL_HEIGHT, WIDTH, CELL_WIDTH).mean(axis=(1, 3))
    compared_cells = compared.reshape(rows, CELL_HEIGHT, WIDTH, CELL_WIDTH).mean(axis=(1, 3))
    density_diff = abs(float(reference.mean(dtype=np.float64) - compared.mean(dtype=np.float64)))
    density_mae = float(np.mean(np.abs(reference_cells - compared_cells), dtype=np.float64))
    return density_diff, density_mae


def evaluate(binary: Path, sample: Path, mode_name: str, mode_flags: list[str]) -> dict:
    sample_name = sample.stem
    output_dir = EVAL_ROOT / sample_name / mode_name
    output_dir.mkdir(parents=True, exist_ok=True)
    render_path = output_dir / "render.png"
    processed_path = output_dir / "processed.png"

    render_once(binary, sample, mode_flags, render_path, processed_path)
    hashes = [sha256(render_path)]

    runtimes: list[float] = []
    for index in range(3):
        timed_path = output_dir / f"render_run{index + 1}.png"
        started = time.perf_counter()
        render_once(binary, sample, mode_flags, timed_path)
        runtimes.append(time.perf_counter() - started)
        hashes.append(sha256(timed_path))

    deterministic = len(set(hashes)) == 1

    reference = load_gray(processed_path)
    compared = load_gray(render_path)
    shrinking = reference.shape[0] >= compared.shape[0] and reference.shape[1] >= compared.shape[1]
    reference = cv2.resize(
        reference,
        (compared.shape[1], compared.shape[0]),
        interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
    )

    blurred_reference = hvs_blur(reference)
    blurred_compared = hvs_blur(compared)
    reference_unit = reference / 255.0
    compared_unit = compared / 255.0
    density_diff, density_mae = density_metrics(reference_unit, compared_unit)

    metrics = {
        "ssim": ssim(reference_unit, compared_unit),
        "ssim_hvs": ssim(blurred_reference / 255.0, blurred_compared / 255.0),
        "fsim": fsim(reference, compared),
        "fsim_hvs": fsim(blurred_reference, blurred_compared),
        "sobel_corr": pearson(sobel_magnitude(reference_unit), sobel_magnitude(compared_unit)),
        "dog_corr": pearson(
            difference_of_gaussians(reference_unit), difference_of_gaussians(compared_unit)
        ),
        "density_diff": density_diff,
        "density_mae": density_mae,
    }

    return {
        "sample": sample_name,
        "mode": mode_name,
        "metrics": metrics,
        "runtime_s": statistics.median(runtimes),
        "deterministic": deterministic,
        "sha256": hashes[0],
    }


def run_self_checks() -> None:
    kernel = np.ones((2, 2), dtype=np.float64)
    impulse = np.arange(1.0, 10.0).reshape(3, 3)
    expected = np.array([[12.0, 16.0, 9.0], [24.0, 28.0, 15.0], [15.0, 17.0, 9.0]])
    assert np.array_equal(conv2_same(impulse, kernel), expected), "conv2 same 裁剪错误"

    generator = np.random.default_rng(20261114)
    image = generator.uniform(0.0, 255.0, size=(520, 520))
    unit = image / 255.0
    assert abs(ssim(unit, unit) - 1.0) <= 1.0e-6, "相同图 SSIM 不是 1"
    assert abs(fsim(image, image) - 1.0) <= 1.0e-6, "相同图 FSIM 不是 1"

    constant = np.full((64, 64), 128.0)
    other_constant = np.full((64, 64), 200.0)
    assert abs(ssim(constant / 255.0, constant / 255.0) - 1.0) <= 1.0e-6, "常量图 SSIM 不是 1"
    assert abs(fsim(constant, constant) - 1.0) <= 1.0e-6, "常量图 FSIM 不是 1"
    assert fsim(constant, other_constant) == 0.0, "不同常量图 FSIM 应为 0"
    assert math.isfinite(ssim(constant / 255.0, other_constant / 255.0)), "常量图 SSIM 非有限"


def metrics_identical(left: dict[str, float], right: dict[str, float]) -> bool:
    for key in METRIC_KEYS:
        left_value = left[key]
        right_value = right[key]
        if not (math.isfinite(left_value) and math.isfinite(right_value)):
            return False
        if abs(left_value - right_value) > 1.0e-9:
            return False
    return True


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=False, capture_output=True, text=True
    )
    return result.stdout.strip()


def write_results(
    rows: list[dict],
    commit: str,
    date: str,
    binary: Path,
    binary_sha: str,
    environment: dict[str, str],
) -> None:
    payload = {
        "commit": commit,
        "date": date,
        "width": WIDTH,
        "hvs_sigma": HVS_SIGMA,
        "binary": display_path(binary),
        "binary_sha256": binary_sha,
        "environment": environment,
        "samples": [path.name for path in SAMPLES],
        "modes": [name for name, _ in MODES],
        "rows": rows,
    }
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    (EVAL_ROOT / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def format_value(value: float) -> str:
    return f"{value:.6f}"


def write_baseline_doc(
    rows: list[dict],
    commit: str,
    date: str,
    dirty: bool,
    binary: Path,
    binary_sha: str,
    environment: dict[str, str],
) -> None:
    lines = [
        "# 评测基线",
        "",
        f"- 提交：`{commit}`",
        f"- 日期：{date}",
        f"- 二进制：`{display_path(binary)}`",
        f"- 二进制 SHA-256：`{binary_sha}`",
        f"- 环境：Python {environment['python']}；numpy {environment['numpy']}；"
        f"scipy {environment['scipy']}；Pillow {environment['pillow']}；"
        f"opencv-python {environment['opencv_python']}",
        f"- 工作区：{'有未提交改动' if dirty else '干净'}",
        f"- 固定宽度：{WIDTH} 列；模式：{'、'.join(name for name, _ in MODES)}",
        f"- 样张：{'、'.join(path.name for path in SAMPLES)}",
        "",
        "## 结果",
        "",
        "| sample | mode | SSIM | SSIM@σ5 | FSIM | FSIM@σ5 | Sobel corr | DoG corr | density diff | density MAE | runtime s | deterministic |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            "| {sample} | {mode} | {ssim} | {ssim_hvs} | {fsim} | {fsim_hvs} | "
            "{sobel_corr} | {dog_corr} | {density_diff} | {density_mae} | {runtime:.3f} | {deterministic} |".format(
                sample=row["sample"],
                mode=row["mode"],
                ssim=format_value(metrics["ssim"]),
                ssim_hvs=format_value(metrics["ssim_hvs"]),
                fsim=format_value(metrics["fsim"]),
                fsim_hvs=format_value(metrics["fsim_hvs"]),
                sobel_corr=format_value(metrics["sobel_corr"]),
                dog_corr=format_value(metrics["dog_corr"]),
                density_diff=format_value(metrics["density_diff"]),
                density_mae=format_value(metrics["density_mae"]),
                runtime=row["runtime_s"],
                deterministic="yes" if row["deterministic"] else "no",
            )
        )

    lines += [
        "",
        "## 指标定义",
        "",
        "- 对照目标是预处理后的最终亮度（色调映射和有符号 DoG 之后），不是原图。",
        "- 预处理图保存为 8-bit PNG，再用 OpenCV 缩放到渲染图尺寸：缩小时用 `INTER_AREA`，放大时用 `INTER_LINEAR`。",
        "- SSIM：11x11 Gaussian 窗，sigma=1.5，C1=0.01²，C2=0.03²；只统计 valid 区域（四周各裁 5 px），取 SSIM 图均值。",
        f"- SSIM@σ5、FSIM@σ5：两幅图先做 Gaussian 模糊，sigma={HVS_SIGMA:.1f}，再计算同名指标。",
        "- FSIM 是 FeatureSIM.m 的逐函数移植，包括自动降采样、Butterworth 低通和噪声补偿。",
        "- FSIM 降采样：F = max(1, round(min(rows, cols) / 256))；两图各自做 F×F 均值滤波（零填充 same 卷积），再按 0::F 取两个轴的采样。",
        "- FSIM 相位一致性 phasecong2：4 尺度、4 方向 log-Gabor；minWaveLength=6，mult=2，sigmaOnf=0.55，dThetaOnSigma=1.2，k=2.0，epsilon=1e-4。",
        "- FSIM 频域：完整 fft2/ifft2；Butterworth 低通 cutoff=0.45、order=15 乘到每个 log-Gabor 滤波器；角度 spread 用 atan2 的角距。",
        "- FSIM 噪声：用最小尺度响应平方的中值估计噪声功率；T = EstNoiseEnergy + k·EstNoiseEnergySigma，再除以 1.7；Energy = max(Energy - T, 0)，PC = EnergyAll / AnAll。",
        "- FSIM 梯度：dx=[3 0 -3; 10 0 -10; 3 0 -3]/16，dy=[3 10 3; 0 0 0; -3 -10 -3]/16，same 卷积后取幅值。",
        "- FSIM 公式：S_PC=(2·PC1·PC2+T1)/(PC1²+PC2²+T1)，T1=0.85；S_G=(2·G1·G2+T2)/(G1²+G2²+T2)，T2=160；FSIM=Σ(S_PC·S_G·max(PC1,PC2))/Σmax(PC1,PC2)。",
        "- FSIM 在 PC 权重和为 0 时：两图完全相同返回 1.0，否则返回 0.0。",
        "- Sobel corr：目标图与渲染图 Sobel 梯度幅值图的 Pearson 相关系数。",
        "- DoG corr：目标图与渲染图 DoG 图（Gaussian σ=1.0 减 σ=2.0）的 Pearson 相关系数。",
        "- density diff：两图全局均值之差的绝对值。",
        "- density MAE：按 10x20 像素块取均值后，120 x rows 单元网格的平均绝对误差。",
        "- runtime s：3 次渲染耗时的中位数；计时不含 `--dump-processed`。",
        "- deterministic：导出运行加 3 次计时运行，4 份渲染 PNG 的 SHA-256 完全一致。",
        "",
        "## 限制",
        "",
        "- 渲染图是字符覆盖的栅格化结果，绝对指标不代表照片级质量，只能在同一套样张、宽度和参数下比较。",
        "- 未模糊的 SSIM、Sobel corr、DoG corr 受字符栅格纹理主导，数值接近 0；判断结构改善主要看 SSIM@σ5、FSIM@σ5 和密度误差。",
        "- 只实现灰度 FSIM，没有实现彩色 FSIMc；彩色信息不进入指标。",
        "- 卷积边界仍可能与 MATLAB 逐位不同：MATLAB `conv2` 的边界与 OpenCV 零填充在边界像素上可能相差 1 像素；本脚本的 FSIM 用 full 卷积后按 MATLAB `same` 规则裁剪，仍不保证与 MATLAB 逐位一致。",
        "- 预处理图为 8-bit，量化误差已计入指标。",
        "- 运行时间受本机负载影响；更换样张、宽度或默认参数会改变基线。",
        "",
    ]
    BASELINE_DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    started = time.perf_counter()
    try:
        run_self_checks()
    except AssertionError as error:
        print(f"自检失败: {error}", file=sys.stderr)
        return 1
    print("自检: conv2/SSIM/FSIM 恒等性与常量图处理通过")

    missing = [str(path.relative_to(ROOT)) for path in SAMPLES if not path.exists()]
    if missing:
        print(f"缺少样张: {', '.join(missing)}", file=sys.stderr)
        return 1

    binary = ensure_binary()
    commit = git_value("rev-parse", "HEAD") or "unknown"
    date = datetime.date.today().isoformat()
    dirty = bool(git_value("status", "--porcelain"))
    environment = dependency_versions()

    rows: list[dict] = []
    nonfinite = False
    total = len(SAMPLES) * len(MODES)
    for sample in SAMPLES:
        for mode_name, mode_flags in MODES:
            row = evaluate(binary, sample, mode_name, mode_flags)
            rows.append(row)
            metrics = row["metrics"]
            print(
                f"[{len(rows):>2}/{total}] {row['sample']:<18} {row['mode']:<6} "
                f"SSIM={metrics['ssim']:.4f} FSIM={metrics['fsim']:.4f} "
                f"density_mae={metrics['density_mae']:.4f} "
                f"runtime={row['runtime_s']:.3f}s det={'yes' if row['deterministic'] else 'no'}"
            )
            for key, value in metrics.items():
                if not math.isfinite(value):
                    print(
                        f"错误: 指标非有限值 sample={row['sample']} mode={row['mode']} "
                        f"{key}={value}",
                        file=sys.stderr,
                    )
                    nonfinite = True

    if nonfinite:
        return 1

    binary_sha = sha256(binary)
    write_results(rows, commit, date, binary, binary_sha, environment)
    write_baseline_doc(rows, commit, date, dirty, binary, binary_sha, environment)

    exit_code = 0
    nondeterministic = [row for row in rows if not row["deterministic"]]
    if nondeterministic:
        print("存在非确定性渲染:", file=sys.stderr)
        for row in nondeterministic:
            print(f"  {row['sample']} {row['mode']}", file=sys.stderr)
        exit_code = 1

    by_sample: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_sample.setdefault(row["sample"], {})[row["mode"]] = row
    identical_samples = []
    for sample_name, modes in by_sample.items():
        if "plain" in modes and "dither" in modes:
            plain = modes["plain"]["metrics"]
            dither = modes["dither"]["metrics"]
            if metrics_identical(plain, dither):
                identical_samples.append(sample_name)
    if len(identical_samples) == len(by_sample) and by_sample:
        print("plain 与 dither 在所有样张上的指标完全相同，评测无效。", file=sys.stderr)
        exit_code = 1

    elapsed = time.perf_counter() - started
    print(f"结果: {EVAL_ROOT / 'results.json'}")
    print(f"基线: {BASELINE_DOC}")
    print(f"脚本耗时: {elapsed:.1f}s")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
