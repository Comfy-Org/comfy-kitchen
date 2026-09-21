# comfy-kitchen 本地编译一键安装

`install.bat` 用于在 Windows + ROCm 机器上从本仓库源码**正确编译**并安装
comfy-kitchen（当前环境: Python 3.13 + torch 2.13.0+rocm10.0.0 + pip rocm-sdk 10.0）。

## 用法

双击 `install.bat`，或在命令行运行：

```bat
install.bat
```

脚本会自动完成: 检查 Python / PyTorch(ROCm) / cmake / ninja → 安装 nanobind →
编译 HIP 后端（自动检测当前 GPU 架构，如 gfx1103）→ 安装 → 自检
（HIP 后端是否生效 + int8 GEMM 冒烟测试）。

可选环境变量：

| 变量 | 说明 |
|---|---|
| `PYTHON` | python 可执行文件（默认 `python`） |
| `COMFY_HIP_ARCHS` | 指定目标 GPU 架构，如 `gfx1103`（默认自动检测） |

例如：

```bat
set COMFY_HIP_ARCHS=gfx1103
install.bat
```

## 关键点：为什么必须 `--no-build-isolation`

脚本里的安装命令是：

```bat
python -m pip install . --no-build-isolation --no-deps --force-reinstall
```

`--no-build-isolation` **不能去掉**。原因：

- pip 默认会先建一个隔离的构建环境，里面**没有** pip 安装的 rocm-sdk
  （`_rocm_sdk_devel`）也**没有** torch。
- setup.py 在这种环境下检测不到 ROCm 编译器，会打印
  “No ROCm compiler detected; skipping HIP backend”，然后**静默装出一个
  纯 Python 包**（无任何编译产物）。
- 于是 int8 GEMM 退化成 triton/eager 后端。本机实测: eager 约 1.5 TFLOPS，
  比编译后的 HIP WMMA 内核（约 7-10 TFLOPS）**慢约 5 倍**。

加上 `--no-build-isolation` 后，编译直接复用当前环境里的 rocm-sdk 和 torch，
会自动为当前 GPU 编译 HIP WMMA 内核（`backends/hip/_C.abi3.pyd`）。

## 验证

脚本最后一步会运行 `install_verify.py`（从仓库目录之外运行，避免源码树遮蔽
site-packages 导致误判）。正确安装的表现为：

```
[OK] HIP backend active (WMMA: True)
int8_linear 4096x2048x2048: ~4.8 ms (7.1 TFLOPS)
```

## 基准测试（重要）

**基准脚本必须在仓库目录之外运行**。在仓库目录内运行会 import 到源码树
（没有编译产物），int8 会走 triton 回退，得到“偏慢”的错误结论：

```bat
cd /d C:\Build\benchmark_attn
python benchmark_gemm.py
```

## 本机性能优化（核显专用）

本仓库源码已包含针对低 WGP 核显（本机 Radeon 780M = 6 WGP）的 int8 GEMM 调优，
`install.bat` 安装的就是调优后的版本：

1. **tile 启发式**（`comfy_kitchen/backends/hip/gemm_wmma.h`）：WGP ≤ 8 的设备
   上，K ≥ 2048 时改用 16-warp 128×128、BKB=128 的块——6 WGP 上块内隐藏
   延迟比块间交错更有效；K < 2048 保留 8-warp BKB=64。dGPU（WGP > 8）行为不变。
   受控交错 A/B 实测（同会话背靠背，27 个真实形状 ×2 次重复）：
   **int8_linear 整体 -2.9%**，Anima 各形状 -1% ~ -14%（adaln -11~-14%、
   mlp2 -3.5%、qkv/o -2~-4%），SDXL 无超噪声回退。
2. **量化核单次读取**（`comfy_kitchen/backends/hip/ops/quantize_int8.hip`）：
   rowwise 量化原来要读 x 两遍（一遍算 absmax、一遍量化），现在 K ≤ 12288 时
   寄存器暂存只读一遍，减少访存与指令（核显带宽有限，带宽敏感用例收益明显）。
3. **运行时调优旋钮**：设置环境变量 `COMFY_KITCHEN_WMMA_TILE` 可强制指定
   tile（0=自动(调优后)，1=128×128 BKB128 16w，2=128×128 BKB128 8w，
   3/4=128×128 BKB64 16w/8w，5/6=64×64 BKB128/64 4w，7=64×128 8w，
   8=128×64 8w，9=128×128 BKB192 8w，10=旧启发式）。例如：

   ```bat
   set COMFY_KITCHEN_WMMA_TILE=1
   python benchmark_gemm.py
   ```

   用 `C:\Build\benchmark_attn\bench_gemm_sweep2.py` 可以快速扫各模式的
   27 个真实形状（约 50 秒/模式）。

正确性已用 `C:\Build\benchmark_attn\check_int8_correct.py` 校验
（量化 q/scale 与参考一致，int8_linear 端到端相对误差 ≈1%，为 int8 本身误差）。

## 恢复官方包

```bat
python -m pip install comfy-kitchen==0.2.35 --force-reinstall --no-deps
```

## 结论

本机实测：**正确编译的本地包与官方包的 int8 GEMM 性能完全一致**（HIP WMMA
内核，约 7-10 TFLOPS）。之前“本地编译更慢”的现象来自安装产物缺失编译后端
（纯 Python 包），而不是源码本身。
