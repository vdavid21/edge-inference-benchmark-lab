# MobileNetV2 TensorRT Accuracy Benchmark

## Objective

This benchmark measures the inference accuracy and performance of the MobileNetV2 model at FP32, TF32, FP16, and INT8 precision. The current run uses the benchmarking and INT8 quantization tooling from commit `bff3273`, and the FP32/TF32/FP16 performance baselines are retained from the earlier benchmark run in `20260919T211117Z_7eb0b09` for comparison.

## Metrics of interest

- Top-1 accuracy
- Top-5 accuracy
- Throughput
- Median latency
- Latency p99
- GPU compute median
- Enqueue median
- Enqueue / compute ratio
- Throughput speedup vs. FP32
- Latency speedup vs. FP32

---

## 1) Download and prepare the dataset

```bash
wget https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz

tar xf imagenetv2-matched-frequency.tar.gz
rm imagenetv2-matched-frequency.tar.gz
ls imagenetv2-matched-frequency-format-val | head -5
```

Use the preprocessing script on the extracted dataset:

```bash
python preprocess.py ~/eibl/data/imagenetv2-matched-frequency-format-val
```

Note:

- The `split_fingerprint` confirms that two runs used the same image set.
- The preprocessing path uses PIL, not OpenCV. The two resize pipelines produce slightly different pixel values, and the torchvision reference accuracy used the PIL path. Mixing them can falsely appear as a 1–2% quantization loss when it is actually a resize mismatch.

---

## 2) Generate the calibration and evaluation split

```bash
python scripts/make_split.py
```

Confirm that the printed fingerprints match before continuing.

---

## 3) Run the FP32 accuracy baseline on ONNX Runtime CPU

This is an independent implementation, so agreement between ONNX Runtime and the TensorRT FP32 engine later is a strong validation that the TensorRT engine is correct.

```bash
python scripts/eval_onnx.py
```

Note:

- Activate the environment if needed:

```bash
source ~/venvs/eibl/bin/activate
```

- Pin the GPU clocks to max for comparable results:

```bash
sudo jetson_clocks
sudo jetson_clocks --show | grep -E 'GPU|EMC'
```

---

## 4) Run the FP32 accuracy baseline in TensorRT

Reuse the existing `mobilenetv2_fp32.plan` built with `--noTF32` from the earlier benchmark in `20260919T211117Z_7eb0b09`.

---

## 5) Smoke-test the TensorRT runner

```bash
python framecost/trt_runner.py models/mobilenetv2_fp32.plan
```

Expect:

- Output shape: `(1, 1000)`
- Logit range roughly within `+/-10`

---

## 6) Run the TensorRT accuracy evaluation for FP32

```bash
python scripts/eval_trt.py models/mobilenetv2_fp32.plan
```

---

## 7) Run the TF32 and FP16 accuracy evaluations

```bash
python scripts/eval_trt.py models/mobilenetv2_tf32.plan
python scripts/eval_trt.py models/mobilenetv2_fp16.plan
```

---

## 8) Run the INT8 quantization pipeline

```bash
python scripts/quantize_int8.py
```

---

## 9) Build the INT8 TensorRT engine

```bash
trtexec \
  --onnx=models/mobilenetv2_int8_entropy_n512.onnx \
  --saveEngine=models/mobilenetv2_int8.plan \
  --minShapes=input:1x3x224x224 \
  --optShapes=input:1x3x224x224 \
  --maxShapes=input:1x3x224x224 \
  --int8 --fp16 \
  --warmUp=2000 --duration=30 \
  --profilingVerbosity=detailed \
  --exportTimes=results/times_int8.json \
  --exportLayerInfo=results/layers_int8.json \
  2>&1 | tee results/build_int8.log
```

---

## 10) Run the INT8 plan with CUDA Graph

```bash
trtexec --loadEngine=models/mobilenetv2_int8.plan \
  --shapes=input:1x3x224x224 --warmUp=2000 --duration=30 --useCudaGraph \
  --exportTimes=results/times_int8_graph.json \
  --exportLayerInfo=results/layers_int8.json \
  2>&1 | tee results/bench_int8_graph.log
```

---

## 11) Inspect the layer formats and reformats

```bash
python -c "
import json, collections
L = json.load(open('results/layers_int8.json'))
L = L['Layers'] if isinstance(L, dict) else L
print('total', len(L))
for k, v in collections.Counter(l.get('LayerType', '?') for l in L).most_common():
    print(f'{v:4d}  {k}')
c = collections.Counter()
for l in L:
    for o in l.get('Outputs', []):
        c[o.get('Format/Datatype', '?')] += 1
print()
for k, v in c.most_common():
    print(f'{v:4d}  {k}')
"
```

---

## 12) Run the INT8 accuracy evaluation

```bash
python scripts/eval_trt.py models/mobilenetv2_int8.plan
```

---

## Results

### Accuracy summary

| Precision | Top-1 | Top-5 | Top-1 Δ vs. FP32 |
| --- | ---: | ---: | ---: |
| FP32 | 58.61% | 81.11% | — |
| TF32 | 58.57% | 81.08% | -0.04 pts |
| FP16 | 58.59% | 81.04% | -0.02 pts |
| INT8 | 57.83% | 80.47% | -0.78 pts |

### Performance summary without CUDA Graph

| Metric | FP32 | TF32 | FP16 | INT8 | INT8 / FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Throughput (qps) | 518.057 | 671.866 | 1047.750 | 1309.840 | 2.528x |
| Median latency (ms) | 1.9805 | 1.5342 | 0.9902 | 0.8086 | 0.408x |
| Latency p99 (ms) | 1.9895 | 1.5430 | 1.0061 | 0.8164 | 0.410x |
| GPU compute median (ms) | 1.9277 | 1.4844 | 0.9439 | 0.7607 | 0.395x |
| Enqueue median (ms) | 0.5459 | 0.5264 | 0.5830 | 0.5371 | 0.984x |
| Enqueue / compute ratio | 0.2833 | 0.3545 | 0.6177 | 0.7060 | 2.493x |
| Throughput speedup vs. FP32 | 1.000x | 1.297x | 2.022x | 2.528x | — |
| Latency speedup vs. FP32 | 1.000x | 1.291x | 2.000x | 2.449x | — |

### Performance summary with CUDA Graph


| Metric | FP32 | TF32 | FP16 | INT8 | INT8 / FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Throughput (qps) | 573.177 | 777.122 | 1327.200 | 1711.850 | 2.987x |
| Median latency (ms) | 1.7930 | 1.3320 | 0.7949 | 0.6260 | 0.349x |
| Latency p99 (ms) | 1.8027 | 1.3408 | 0.8123 | 0.6328 | 0.351x |
| GPU compute median (ms) | 1.7402 | 1.2832 | 0.7490 | 0.5820 | 0.334x |
| Enqueue median (ms) | 0.0254 | 0.0098 | 0.0098 | 0.0078 | 0.308x |
| Enqueue / compute ratio | 0.0146 | 0.0076 | 0.0130 | 0.0134 | 0.920x |
| Throughput speedup vs. FP32 | 1.000x | 1.356x | 2.316x | 2.987x | — |
| Latency speedup vs. FP32 | 1.000x | 1.346x | 2.256x | 2.864x | — |

### Interpretation

- FP16 remains effectively free from an accuracy perspective while significantly improving latency and throughput.
- INT8 keeps high performance with a small accuracy loss, around 0.78 points of Top-1 on this dataset.
- CUDA Graph reduces the host-side overhead substantially in the INT8 run, improving the median latency from about 0.809 ms to about 0.626 ms.
- Throughput speedup exceeds latency speedup, and the gap widens as the kernels get shorter: trtexec overlaps the H2D and D2H copies with compute across iterations, so the roughly 0.04 ms of transfer per query is hidden. That overhead is a larger share of a 0.58 ms INT8 kernel than of a 1.74 ms FP32 one, which is why INT8 reaches 2.987x on throughput but 2.864x on median latency, while TF32 shows almost no divergence (1.356x vs 1.346x).
- The current run confirms the expected pattern: the INT8 graph-enabled configuration is the fastest measured variant in this benchmark set.

---