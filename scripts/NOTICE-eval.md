# 评测脚本的上游许可

`scripts/eval.py` 的 FSIM 指标是以下 MATLAB 代码的 Python 移植。移植部分不适用仓库根目录的 MIT 许可，适用下列上游条款。

## FeatureSIM.m

来源：<https://cslinzhang.github.io/home/FSIM/Files/FeatureSIM.m>

参考文件 SHA-256：`4b23dc2881c71dfa996e40fd84b9b946f3934b36ecfb3ee4dfbafa5505bcbba9`

Copyright (c) 2010 Lin Zhang, Lei Zhang, Xuanqin Mou and David Zhang. All Rights Reserved.

Permission to use, copy, or modify this software and its documentation for educational and research purposes only and without fee is hereby granted, provided that this copyright notice and the original authors' names appear on all copies and supporting documentation. This program shall not be used, rewritten, or adapted as the basis of a commercial software or hardware product without first obtaining permission of the authors. The authors make no representations about the suitability of this software for any purpose. It is provided "as is" without express or implied warranty.

## phasecong2

来源：`FeatureSIM.m` 内置的 `phasecong2` 函数。

Copyright (c) 1996-2009 Peter Kovesi
School of Computer Science & Software Engineering
The University of Western Australia
<http://www.csse.uwa.edu.au/>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. The software is provided "as is", without warranty of any kind.

## lowpassfilter

来源：`FeatureSIM.m` 内置的 `lowpassfilter` 函数。

Copyright (c) 1999 Peter Kovesi
School of Computer Science & Software Engineering
The University of Western Australia
<http://www.csse.uwa.edu.au/>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. The Software is provided "as is", without warranty of any kind.

## 适用范围

- 上述条款只覆盖 `scripts/eval.py` 中的 FSIM 移植及本文件。
- 仓库其余代码按 `Cargo.toml` 中的 MIT 许可发布。
- 商用前需取得 FeatureSIM 作者的许可。
