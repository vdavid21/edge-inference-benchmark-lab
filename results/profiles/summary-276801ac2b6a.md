# MobileNetV2 Per-Layer Roofline, FP32 / TF32 / FP16 / INT8 at 25 W

## Objective

Place every layer of the engine on a roofline — arithmetic intensity against attained throughput — to find out which ceiling actually limits this model at each precision. The latency chapter showed the gains tracking bytes rather than flops; this run tests that inference directly, one point per layer, per precision.

Ten runs at 25 W with clocks pinned: FP32 ×2, TF32 ×2, FP16 ×3 ungraphed plus one with CUDA Graph, INT8 ×2.

## Metrics of interest

- Peak compute and memory bandwidth (the two ceilings)
- Ridge point, where they cross
- Arithmetic intensity per layer
- Layers below the ridge (memory-bound)
- Achieved memory bandwidth per layer
- Time and FLOP share by layer kind

AI — arithmetic intensity, FLOP per byte of DRAM traffic.
Ridge — peak compute / bandwidth. Left of it a layer is memory-bound, right of it
compute-bound.
Achieved bandwidth — layer bytes / layer time. The useful efficiency measure for a
memory-bound kernel; percent-of-compute-peak is not.

---

## 1) Pin the clocks

```bash
sudo jetson_clocks
```

The script refuses to run unpinned unless `--allow-dvfs` or explicit rates are given:
a roofline assumes fixed ceilings, and under DVFS they move with demand.

---

## 2) Profile each engine

```bash
python scripts/profile_model.py models/mobilenetv2_fp32.plan \
    --onnx models/mobilenetv2-12.onnx --precision fp32 --no-cuda-graph
python scripts/profile_model.py models/mobilenetv2_tf32.plan \
    --onnx models/mobilenetv2-12.onnx --precision tf32 --no-cuda-graph
python scripts/profile_model.py models/mobilenetv2_fp16.plan \
    --onnx models/mobilenetv2-12.onnx --precision fp16 --no-cuda-graph
python scripts/profile_model.py models/mobilenetv2_int8.plan \
    --onnx models/mobilenetv2_int8_entropy_n512.onnx --precision int8 --no-cuda-graph
```

Which issues, per run:

```text
trtexec --loadEngine=<plan> --shapes=input:1x3x224x224 \
  --warmUp=2000 --duration=30 --noDataTransfers \
  --profilingVerbosity=detailed --dumpProfile --separateProfileRun \
  --exportProfile=<run_dir>/profile.json --useSpinWait
```

The one graphed run, `20260924T212858Z`, was issued without `--no-cuda-graph`, which
adds `--useCudaGraph` to the line above.

Notes:

- `--separateProfileRun` gives a Performance summary from an *unprofiled* benchmark run
  and a per-layer table from a *profiled* one, in the same process. Per-layer times are
  scaled so their sum matches the clean GPU compute time.
- `--noDataTransfers` so GPU Compute Time covers only the kernels the per-layer sum
  also covers.
- `--precision` selects the compute roof and the byte model. It is an analysis
  parameter, not read from the engine, so it must match what the plan was built as.
- `--onnx` must be the graph the engine was built from, not the original model. Layer
  geometry is joined to TensorRT layer names by ONNX node name and index, and the INT8
  graph has 318 nodes against the base model's 105, so the indices do not correspond.
  The INT8 runs therefore take `mobilenetv2_int8_entropy_n512.onnx`. Passing the base
  model would still match some names and return wrong geometry rather than an error.
- Clocks are read from tegrastats during the run, not from the configured ceiling. All
  ten runs measured 904–907 MHz against a nominal 918 MHz, EMC 3199 MHz.

---

## 3) Repeat for variance

Each configuration was run at least twice from a separate process invocation.

---

## Results

> Ten runs, one commit, all at 25 W pinned. Bandwidth is 102.4 GB/s throughout
> (LPDDR5, 3199 MHz, 128-bit, DDR).

### Ceilings and position

| Precision | Peak | Ridge | AI median | Memory-bound | Achieved BW (median) | Achieved BW (best) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 1.85 T/s | 18.1 | 13.24 | **36 / 52** | 28.9 GB/s | 64.2 GB/s |
| TF32 | 7.41 T/s | 72.4 | 13.24 | 52 / 52 | **38.3 GB/s** | 80.3 GB/s |
| FP16 | 14.82 T/s | 144.8 | 26.47 | 52 / 52 | 29.7 GB/s | 82.8 GB/s |
| INT8 | 29.71 T/s | 290.2 | 52.95 | 52 / 52 | 17.6 GB/s | 74.5 GB/s |

Peaks are dense Ampere SM rates at the measured 905 MHz: 256 FLOP/SM/clk on the CUDA cores for FP32, 1024 / 2048 / 4096 on the Tensor Cores, times 8 SMs. 
Cross-check: 4096 × 8 × 1020 MHz = 33.4 TOPS dense = the advertised 67 TOPS sparse, and the same arithmetic gives the 40 TOPS quoted for the 625 MHz original Orin Nano.

### Precision steps

| Step | Measured | Theoretical | Captured |
| --- | ---: | ---: | ---: |
| FP32 → TF32 | 1.296× | 4× | 32.4% |
| TF32 → FP16 | 1.581× | 2× | — |
| FP16 → INT8 | 1.236× | 2× | — |
| FP32 → INT8 | 2.533× | 16× | 15.8% |

Mean GPU compute time, ungraphed: FP32 1.9219 ms, TF32 1.4829 ms, FP16 0.9378 ms, INT8 0.7588 ms.

### Layer kinds (FP16, ungraphed)

| Kind | n | Share of FLOPs | Share of time | Cost per FLOP |
| --- | ---: | ---: | ---: | ---: |
| depthwise | 17 | 6.9% | 32.2% | 4.7× |
| pointwise | 34 | 89.1% | 51.2% | 0.57× |
| conv | 1 | 3.6% | 2.7% | 0.75× |

### Interpretation

- FP32 is the only precision with compute-bound layers: 16 of 52, all pointwise,carrying 44.3% of the model's FLOPs in 27.0% of its runtime. Its roof is the CUDA cores at 1.85 TFLOP/s, putting the ridge at just 18.1 FLOP/byte.
- Engaging the Tensor Cores ends that permanently. TF32 lifts the roof 4×, the ridge moves to 72.4, and every layer falls to the memory-bound side. FP16 and INT8 push the ridge further while the workload's intensity rises only in step: the most intense layer reaches 40% of the ridge in both.
- This explains the sweep. FP32 → TF32 buys 1.296× because roughly a third of the layers were genuinely arithmetic-limited — the one step where more FLOP/s is the right medicine. Every step after it is byte reduction, which is why TF32 → FP16 and FP16 → INT8 track the halving of traffic rather than the doubling of arithmetic.
- Memory-bound is not the same as bandwidth-saturated. Achieved bandwidth is non-monotonic — 28.9 → 38.3 → 29.7 → 17.6 GB/s — peaking at TF32. The best layers reach 74–81% of peak; the median reaches under a third. The typical layer is limited by neither ceiling: it is too small to saturate anything, and what bounds it is launch latency and occupancy. That is consistent with CUDA Graph buying 1.26×, which a bandwidth-saturated GPU could not give.
- Even the compute-bound FP32 layers only reach 25–34% of their 1.85 T/s roof. Above the ridge means the memory ceiling is no longer binding, not that the ALUs are busy.
- Depthwise convolutions cost 8× more per FLOP than pointwise, and hold the lowest intensity in the network (1.73 for `Conv_7 + Clip_8`). The depthwise separable convolution buys an 8–9× reduction in multiplications and returns about a third of it as memory traffic.
- `Reformatting CopyNode for Input Tensor 0` is the largest non-compute item in the FP16 engine at 8.5% of profiled time. CUDA Graph removes three quarters of it, 93.2 µs to 27.2 µs — part of the graph's 1.26× is this one node, not launch overhead in general.
- Instrumentation overhead rises monotonically as kernels shorten: 5.6% FP32, 9.4% TF32, 17.0% FP16, 24.2% INT8. INT8 per-layer times therefore carry the largest correction.
- Cross-check against the latency chapter, which measured end-to-end throughput rather than GPU compute time: FP32 → TF32 1.296 here against 1.297 there, FP32 → INT8 2.533 against 2.528. Two independent methods.

### Reproducibility

- Repeated configurations agree to 0.07–0.32% on benchmark GPU compute time.
- Across the three FP16 repeats, per-layer times vary by a median of 1.1%, p90 4.6%, worst 8.4% — the worst being a 6 µs cast where 0.5 µs of jitter is 8%.
- AI, FLOP and byte counts are bit-identical across repeats, as they must be: they come from the ONNX graph, not the clock.
- 52 of 53 ONNX compute nodes are matched to TensorRT layers in every run. The unmatched 0.43% of graph FLOPs is folded into myelin kernels whose names carry no node reference; those are reported as `unattributed` (4.7–8.1% of time) rather than silently dropped.

---

## Run inventory

| Run | Precision | CUDA Graph | GPU compute | Instr. overhead | Use |
| --- | --- | --- | ---: | ---: | --- |
| 20260924T212858Z | FP16 | yes | 0.7441 ms | 58.6% | **benchmark only** |
| 20260924T215609Z | FP16 | no | 0.9375 ms | 17.3% | full |
| 20260925T172020Z | FP16 | no | 0.9395 ms | 16.8% | full |
| 20260925T172315Z | FP16 | no | 0.9365 ms | 17.0% | full |
| 20260924T215942Z | INT8 | no | 0.7598 ms | 24.9% | full |
| 20260925T173301Z | INT8 | no | 0.7578 ms | 23.5% | full |
| 20260925T173619Z | FP32 | no | 1.9199 ms | 5.8% | full |
| 20260925T195311Z | FP32 | no | 1.9238 ms | 5.4% | full |
| 20260925T195811Z | TF32 | no | 1.4834 ms | 9.4% | full |
| 20260925T200231Z | TF32 | no | 1.4824 ms | 9.5% | full |

**Do not use the per-layer data from `20260924T212858Z`.** Profiling and CUDA Graph do
not compose: the profile pass does not get the graph's benefit, so the scale factor ends
up reconciling a benchmark run that is 1.26× faster with a profile run that is 7.5%
slower, and compresses every layer by 35% to make the sum match. Identical kernels
report 1300 G/s there against 1033 G/s in the ungraphed runs. Its benchmark number is
sound and is the source of the 1.26× CUDA Graph figure and the reformat comparison.

## Provenance

All ten runs are from commit `276801ac2b6a`, which contains the profiler. Run `20260924T212858Z` records `dirty: false`; every later run records `dirty: true`, caused only by `results/profiles/` being untracked output that the first run created. No source
file changed during the campaign.

Note that `framecost/env.py` treats a dirty tree as disqualifying for published numbers.
That check cannot discriminate here, because the script dirties the tree by writing its
own results. It should exclude the results directory.

Engine and ONNX hashes are recorded per run in `result.json` (`engine_sha256_16`,
`onnx_sha256_16`). Two source graphs were used: `554077ddef9336f6`
(`mobilenetv2-12.onnx`, for FP32/TF32/FP16) and `ef760b3fc2909577`
(`mobilenetv2_int8_entropy_n512.onnx`, for INT8).

---
