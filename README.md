# Edge Inference Benchmark Lab

Measuring what precision reduction actually buys you on an NVIDIA Jetson Orin Nano.

FP32, TF32, FP16 and INT8 inference of the same model, on the same board, with pinned clocks, percentile latencies, accuracy on a fixed disjoint split, energy per frame from the onboard rails and a wall meter, and a per-layer roofline.

**Article:** davidvincze.com/posts/edge-inference-benchmark-lab/

---

## Results

Batch 1, CUDA graphs enabled, clocks pinned, 25 W power mode.

| Precision | Latency p50 | p99 | Throughput | Top-1 | Δ Top-1 | Engine | Activation mem |
|---|---|---|---|---|---|---|---|
| FP32 (`--noTF32`) | 1.793 ms | 1.803 ms | 573 qps | 58.61% | — | 14.40 MB | 7.02 MB |
| TF32 | 1.332 ms | 1.341 ms | 777 qps | 58.57% | −0.04 | 14.34 MB | 7.02 MB |
| FP16 | 0.795 ms | 0.812 ms | 1327 qps | 58.59% | −0.02 | 7.35 MB | 3.51 MB |
| INT8 | 0.626 ms | 0.633 ms | 1712 qps | 57.83% | −0.78 | 5.68 MB | 1.83 MB |

Top-1 on 9000 ImageNetV2 images, 95% CI ±1.02%. The Tensor Cores offer 4× the FP32 arithmetic rate at TF32, 8× at FP16 and 16× at INT8, but the fraction realised falls as precision drops: 34%, 29%, 19%. Sixteen times the arithmetic buys three times the inference rate, because the arithmetic was never the bottleneck.

### Energy per frame, DVFS, by power mode

| Mode | GPU max | FP16 | INT8 |
|---|---:|---:|---:|
| 15 W | 612 MHz | **10.10 mJ** | 6.73 mJ |
| 25 W | 918 MHz | 10.75 mJ | **6.53 mJ** |
| MAXN_SUPER | 1020 MHz | 10.90 mJ | 6.58 mJ |

Measured on the board's input rail; idle baseline 3.52 W, all six runs starting at 46–47 °C.

**The optimal power mode is a property of the board and the precision together, not of the board alone.** FP16 is most efficient at 15 W — 62% of MAXN's throughput for 57% of its power. INT8 inverts it and wants 25 W: it has already cut dynamic power by 39%, so the 3.5 W static baseline dominates, and stretching a frame from 0.65 ms to 0.88 ms costs more than the lower clock saves. Deploying INT8 at the power mode that was optimal for FP16 is the wrong call.

### Three more

**Every layer is memory-bound except in FP32.** FP32 is the only precision with compute-bound layers — sixteen pointwise convolutions, 44% of the FLOPs in 27% of the runtime. Engaging the Tensor Cores moves the ridge past the entire workload and it never comes back, which is why FP32 → TF32 buys 1.30× and every later step tracks traffic instead. → [roofline](results/profiles/summary-276801ac2b6a.md)

**An unflagged FP32 baseline is worth 1.36×.** TensorRT enables TF32 on Ampere by default, so "FP32" without `--noTF32` is silently reduced-precision math on Tensor Cores. → [latency](results/run/20260920T102737Z_bff3273/summary.md)

**Depthwise convolutions cost 8× more per FLOP than pointwise**, carrying 6.9% of the arithmetic in 32% of the runtime. The operation that makes MobileNet cheap in multiplications returns a third of that saving as memory traffic. → [roofline](results/profiles/summary-276801ac2b6a.md)

## Measurements

| Campaign | What it establishes | Data |
|---|---|---|
| Latency, FP32/TF32/FP16 | Build commands, percentile latencies, CUDA Graph and spin-wait arms | [summary](results/run/20260919T211117Z_7eb0b09/summary.md) |
| Latency + accuracy, INT8 | Quantisation, the four-precision tables, ImageNetV2 evaluation | [summary](results/run/20260920T102737Z_bff3273/summary.md) |
| Energy, pinned vs unpinned | Clock policy against mJ/frame, rail and wall power | [summary](results/energy/summary-e3e0867e26c9.md) |
| Energy, 15 W / 25 W / MAXN_SUPER | Energy per frame across power modes, FP16 and INT8 | [summary](results/energy/summary-6fc2633c8be2.md) |
| Per-layer roofline | Arithmetic intensity, ridge, achieved bandwidth, ten runs across four precisions | [summary](results/profiles/summary-276801ac2b6a.md) |
| Measured arithmetic ceiling | GEMM peak, FP32-accumulate question | [summary](results/peak_gemm/summary-a74615cea83b.md) |

Each summary carries the commands that produced it, the raw logs and the interpretation. Every result traces to a git SHA, an engine hash and a split fingerprint — `calib 1000 214e64efb5dc5f8f`, `eval 9000 ecdf27521cf9967e`. If `python scripts/make_split.py` prints anything else, the data changed and the numbers are not comparable.

## Platform

| | |
|---|---|
| Board | Jetson Orin Nano 8 GB Developer Kit (P3767-0005, Tegra234, sm_87) |
| JetPack | 7.2.1 — L4T r39.2.1, Ubuntu 24.04, kernel 6.8.12-tegra |
| CUDA / TensorRT / cuDNN | 13.2.86 / 10.16.2.10 / 9.20.0.46 |
| Python | 3.12.3 |
| Power mode | 25 W (GPU 918 MHz, EMC 3199 MHz = 102 GB/s), `jetson_clocks` applied |
| Quantisation | NVIDIA Model Optimizer 0.46.0, explicit Q/DQ, entropy calibration, 512 images |
| Power meter | WM03-DE, ±2%, between the PSU and the socket |

Model: `mobilenetv2-12.onnx` from the ONNX model zoo, opset 12, batch 1, 224×224.

Environment captures are in [`docs/env/`](docs/env/) — including the [mismatched one](docs/env/env-jetson-orin-20260904T174022Z-pre-firmware-misdetected.md), where the installer had capped the board at 624 MHz with 15W power mode. Please refer to the [updated one](docs/env/env-jetson-orin-20260919T113300Z.md).
## Layout

```
framecost/     the harness
  env.py         environment capture, git provenance, suitability checks
  preprocess.py  PIL path, matched to the torchvision reference
  trt_runner.py  TensorRT execution
  telemetry.py   tegrastats sampler, energy accounting
  roofline.py    per-layer FLOPs, bytes, arithmetic intensity, the plot

scripts/       entry points
  00_env_report.sh   environment report
  make_split.py      fixed calibration / evaluation split
  quantize_int8.py   explicit Q/DQ via ModelOpt PTQ
  eval_onnx.py       ONNX Runtime CPU accuracy reference
  eval_trt.py        TensorRT accuracy
  measure_energy.py  energy per frame under telemetry
  profile_model.py   per-layer profile and roofline
  peak_gemm.py       measured arithmetic ceiling

results/       one directory per run, plus a summary per campaign
docs/env/      environment captures
models/        ONNX and .plan files (gitignored except the quantised ONNX)
data/          dataset (gitignored); split_v1.json is committed
```

## Background

Built alongside MIT 6.5940 *TinyML and Efficient Deep Learning*. The course covers the accuracy side of quantisation with simulated arithmetic; this repo is the performance side, with kernels that actually execute in INT8.

## Licence

MIT.
