# Measured Arithmetic Ceiling, FP32 / TF32 / FP16 at 25 W

## Objective

The roofline ceilings in `framecost/roofline.py` are derived from Ampere SM rates, not
measured. FP16 is the one that could be wrong by 2×: consumer GA10x halves FP16
throughput when accumulating in FP32, and whether Orin's GA10B does the same was
unverified. If it does, the FP16 ridge is 72 rather than 145 and every FP16
"% of peak" figure in the roofline chapter is out by a factor of two.

A large square GEMM is deeply compute-bound — the opposite regime from MobileNetV2 —
so its attained throughput is limited by arithmetic and nothing else.

## Metrics of interest

- Attained TFLOP/s per precision, against the derived ceiling
- Which kernel TensorRT actually selected, and its accumulate type
- Board power and clock during the timed region

---

## 1) Pin the clocks

```bash
sudo jetson_clocks
```

## 2) Run twice

```bash
python scripts/peak_gemm.py --sizes 2048 4096
python scripts/peak_gemm.py --sizes 2048 4096
```

FP32 is the control: it runs on the CUDA cores with no Tensor Core and no accumulate
ambiguity, so if it reaches a sensible fraction of its ceiling the harness is sound.
The verdict is one-sided by design — exceeding the halved ceiling falsifies it, while
falling short proves nothing, since any inefficiency produces the same result.

---

## Results

> N = 2048, mean of two runs. Derived ceilings at the pinned 918 MHz, 8 SMs.

| Precision | Ceiling | Attained | % of ceiling | Kernel accumulate |
| --- | ---: | ---: | ---: | --- |
| FP32 | 1.88 T/s | 1.50 | 79.6% | — (CUDA cores) |
| TF32 | 7.52 T/s | 4.01 | 53.3% | FP32 |
| FP16 | 15.04 T/s | **9.97** | 66.3% | **FP16** |

Reproducibility at N = 2048 is 0.5% between runs. Both runs returned
`verdict: no_penalty`, `throttle_status: clean`, no timing cache, telemetry 6/6.

### Verdict

The halved FP16 ceiling would be **7.52 T/s**. FP16 attained **9.97**, exceeding it by
33%. A roof cannot be beaten, so the FP32-accumulate penalty does not apply to the
kernel measured. FP32 passed its control at 79.6%.

Strictly this proves that FP16 *with FP16 accumulate* is not halved, because that is
the kernel TensorRT chose:

```
sm80_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x256x32_...
                              ^^^ accumulate
```

That turns out to be the relevant mode. MobileNetV2's own FP16 engine uses FP16
accumulate for its implicit-GEMM convolutions:

```
sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x32x32_...
```

So `fp16 = 2048 FLOP/SM/clk` is correct for the engine the roofline chapter analyses,
and the FP16 ridge of 144.8 stands. Whether Orin penalises FP32 accumulate in general
remains untested and is moot here.

### Power

The dense GEMM is the heaviest load in this project. At 25 W pinned it drew **21 W**
sustained against MobileNetV2 FP16 inference at 13.8 W, with Tj at 65 °C and the GPU
clock never dropping below 903 MHz of its 918 MHz ceiling. No throttling, so the
inference workload has substantial thermal and current headroom.

---

## Not reported: N = 4096

Both runs also covered N = 4096 and its numbers are in the run directories, but they do
not reproduce and are excluded from any claim. Run `20260926T111728Z` was the first run
ever to need those files, so it generated 34–67 MB of ONNX immediately before each arm;
run `20260926T112839Z` found them already present. The first run's 4096 arms show
depressed memory utilisation (FP32 23.0% against 43.8%) and ran 1.4–1.9× slower. TF32
additionally picked a different tile size in each run, which confounds tactic variance
with the generation effect.

N = 2048 is unaffected — its files are 8–16 MB and it reproduces to 0.5% — and it
settles the question on its own, so the discrepancy was not chased further.

`make_matmul` should be moved ahead of the measurement loop so the first run at a new
size pays generation cost outside any timed region.

---
