# Edge Inference Benchmark Lab

Measuring what precision reduction actually buys you on an NVIDIA Jetson Orin Nano.

FP32, TF32, FP16 and INT8 inference of the same model, on the same board, with pinned clocks, percentile latencies, and accuracy measured on a fixed, disjoint evaluation split. The goal is a reusable harness, not a one-off number.

**Status: early, in progress.** Latency and accuracy are measured for four precisions. Energy, thermals and per-layer analysis are not done yet. Numbers below are real and reproducible; the conclusions are provisional.

---

## Headline result

Same model, same board, same measurement method. Batch 1, CUDA graphs enabled, clocks pinned, 25 W power mode.

| Precision | Latency p50 | p99 | Throughput | Top-1 | Δ Top-1 | Engine | Activation mem |
|---|---|---|---|---|---|---|---|
| FP32 (`--noTF32`) | 1.793 ms | 1.803 ms | 573 qps | 58.61% | — | 14.40 MB | 7.02 MB |
| TF32 | 1.332 ms | 1.341 ms | 777 qps | 58.57% | −0.04 | 14.35 MB | 7.02 MB |
| FP16 | 0.795 ms | 0.812 ms | 1327 qps | 58.59% | −0.02 | 7.46 MB | 3.51 MB |
| INT8 | 0.626 ms | 0.633 ms | 1712 qps | 57.83% | −0.78 | 5.65 MB | 1.83 MB |

Accuracy is top-1 on 9000 ImageNetV2 images, 95% CI ±1.02%.

**The gap between theory and measurement:**

| | FP32 | TF32 | FP16 | INT8 |
|---|---|---|---|---|
| Theoretical peak vs FP32 | 1× | 4× | 8× | 16× |
| Measured speedup | 1.00× | 1.36× | 2.32× | 2.86× |
| **Fraction realised** | — | **34%** | **29%** | **18%** |

Sixteen times the arithmetic throughput delivers 2.86× the inference rate, and the fraction realised *falls* as precision drops. MobileNetV2's depthwise convolutions sit far inside the memory-bound region of the roofline — the arithmetic was never the bottleneck.

Two checks rule out the easy explanations for INT8's modest 1.23× over FP16. Coverage is near-total (54 of 57 output tensors are INT8), and layout-conversion overhead is negligible (2 reformat layers versus FP16's 1). What remains is memory bandwidth and the fixed costs that don't scale with precision.

## Other findings so far

**FP32 is ambiguous and the ambiguity is worth 1.34×.** TensorRT enables TF32 on Ampere by default, so an unflagged "FP32" baseline is silently running reduced-precision math on Tensor Cores. Which baseline you pick moves your headline speedup by 34%.

**CUDA Graphs are worth more than a precision step.** Collapsing 87 kernel launches into one submission gave 1.11× at FP32 rising to 1.28× at INT8 — and 20% of what `trtexec` reports as "GPU Compute Time" without graphs turns out to be launch bubbles between kernels, not computation.

**Pinning clocks mostly helps the CPU.** `jetson_clocks` improved GPU compute by 3.9% but cut kernel-launch (enqueue) time by 40%, because the frequency governor handles bursty submission work badly.

**Precision changes which algorithm gets selected.** At FP16, TensorRT uses GEMM kernels for 18 of the 1×1 pointwise convolutions. At INT8 it uses direct convolution for all 52 — the autotuner abandoned the GEMM formulation entirely.

## Platform

| | |
|---|---|
| Board | Jetson Orin Nano 8 GB Developer Kit (P3767-0005, Tegra234, sm_87) |
| JetPack | 7.2.1 — L4T r39.2.0, Ubuntu 24.04, kernel 6.8.12-tegra |
| CUDA / TensorRT / cuDNN | 13.2.86 / 10.16.2.10 / 9.20.0.46 |
| Python | 3.12.3 |
| Power mode | 25 W (GPU 918 MHz, EMC 3199 MHz = 102 GB/s), `jetson_clocks` applied |
| Quantisation | NVIDIA TensorRT Model Optimizer 0.46.0, explicit Q/DQ, entropy calibration |

Model: `mobilenetv2-12.onnx` from the ONNX model zoo, opset 12, batch 1, 224×224.

## Reproducing

```bash
# environment report — capture before anything else
bash scripts/00_env_report.sh | tee docs/env/$(date -u +%Y%m%dT%H%M%SZ).md

# data (1.26 GB, not in this repo)
mkdir -p data && cd data
wget https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz
tar xf imagenetv2-matched-frequency.tar.gz && cd ..

# split — must print the fingerprints below
python scripts/make_split.py
#   calib  1000  214e64efb5dc5f8f
#   eval   9000  ecdf27521cf9967e

# build and benchmark
sudo nvpmodel -m <25W mode id> && sudo jetson_clocks
bash scripts/build_all.sh

# accuracy
python scripts/eval_onnx.py                              # ORT CPU reference
python scripts/eval_trt.py models/mobilenetv2_fp32.plan
```

Every result should be traceable to a git SHA, an engine hash, and a split fingerprint. If a fingerprint doesn't match, the data changed and the numbers aren't comparable.

## Layout

```
framecost/     # the harness: preprocessing, TensorRT runner, telemetry
scripts/       # entry points: env report, split, quantise, build, evaluate
docs/          # ramp-up notes, methodology, environment captures, backlog
models/        # ONNX and .plan files (gitignored — build artifacts)
data/          # dataset (gitignored); split_v1.json is committed
results/       # logs and exported traces
```

## Caveats

**ImageNetV2, not ImageNet val.** Absolute accuracy runs ~13 points below published ImageNet validation figures because ImageNetV2 is a deliberately harder set. All comparisons here are deltas on one fixed split, which is what
the question requires, but don't compare these absolute numbers to a paper.

**The INT8 accuracy drop is not yet statistically established.** 0.78 points sits inside the ±1.02% confidence interval. Because both arms use identical images, a paired test (McNemar's) is the right instrument and hasn't been run yet.

**Isolated inference latency, not pipeline latency.** In a Python evaluation loop dominated by JPEG decode and preprocessing, the 2.32× FP16 speedup showed up as roughly 9% end to end.

**One board, one unit.** Nothing here establishes part-to-part variation.

**Super mode required a firmware update.** The JetPack 7.2 installer misdetects the Orin Nano Developer Kit and omits the 25 W and MAXN SUPER power profiles, capping the board at 15 W / 624 MHz / 68 GB/s. Editing nvpmodel config does not fix it — the clock ceilings come from the bootloader. Check your actual clocks rather than trusting
`nvpmodel -q`.

## Not done yet

- Energy per inference (mJ/frame) and the 7 W / 15 W / 25 W power sweep
- Sustained thermal behaviour and throttling
- Per-layer profiling and the roofline scatter plot
- Calibration method and calibration-set-size sweep
- McNemar's paired significance test
- A second model (ResNet-18 for contrast, YOLO for detection)
- 2:4 structured sparsity

## Background

Built alongside MIT 6.5940 *TinyML and Efficient Deep Learning* (Fall 2024). The course covers the accuracy side of quantisation with simulated arithmetic; this repo is the performance side, with kernels that actually execute in INT8.

## Licence

MIT.