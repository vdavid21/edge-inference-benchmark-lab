# MobileNetV2 FP16, INT8 Energy Measurement on 15W, 25W, MAXN_SUPER

## Objective

The goal is to benchmark model throughput and power consumption with different power modes. With my current Jetson Developer Kit I can measure 15W, 25W and MAXN_SUPER, the 7W is not available. As we concluded that unpinned dvfs configuration is more energy efficient than the pinned GPU clock, therefore all the tests are executed in dvfs mode.

## Metrics of interest

- GPU max clock
- Throughput
- P_board
- P_wall
- Energy per frame
- Tj max

P_board — what VDD_IN measures: the DC input to the devkit after the barrel jack,
sensed by the carrier-board INA3221, so it covers module *and* carrier loads.
P_wall — what the WM03-DE measures at the socket, including PSU loss.

---

## 1) Run the measurement script for 120 seconds

```bash
python scripts/measure_energy.py models/mobilenetv2_int8.plan --duration 120
```

---

## 2) Give time to cool down

```bash
sleep 300
```

---

## 3) Change power configuration and repeat it for different model plans

---

## Results

> Six runs, all starting at 46–47 °C, all with idle baselines within 1% of each other.

### FP16

| Mode | GPU max | Throughput | P_board | P_wall | Gap | mJ/frame (P_board) | mJ/frame (P_wall) | Tj max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 W | 612 MHz | 889.8 qps | 8.76 W | 11.65 W | 2.89 W | **10.10** | **13.43** | 54.7 °C |
| 25 W | 918 MHz | 1320.0 qps | 13.80 W | 17.75 W | 3.95 W | 10.75 | 13.83 | 61.5 °C |
| MAXN_SUPER | 1020 MHz | 1439.2 qps | 15.31 W | 19.70 W | 4.39 W | 10.90 | 14.02 | 63.3 °C |
| Idle | — | — | 3.52 W | 5.25 W | 1.73 W | — | — | 46.5 °C |

### INT8

| Mode | GPU max | Throughput | P_board | P_wall | Gap | mJ/frame (P_board) | mJ/frame (P_wall) | Tj max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 W | 612 MHz | 1141.7 qps | 7.49 W | 10.15 W | 2.66 W | 6.73 | 9.11 | 52.9 °C |
| 25 W | 918 MHz | 1537.4 qps | 9.75 W | 12.95 W | 3.20 W | **6.53** | 8.67 | 56.7 °C |
| MAXN_SUPER | 1020 MHz | 1662.8 qps | 10.68 W | 13.95 W | 3.27 W | 6.58 | **8.60** | 57.8 °C |

### Interpretation

- FP16 behaves as predicted: 15 W is the most efficient: 62% of MAXN's throughput for 57% of the power, giving 7.3% better energy per frame. Slower is more efficient.
- INT8 inverts it: 25 W is the optimum on P_board, and MAXN_SUPER at the wall. 15 W is the worst of the three. That's the opposite of the FP16 ordering. The difference between 25 W and MAXN_SUPER is very small, under 1%, which can be inside noise. But the direction of the shift is systematic, because the fixed 1.73 W is paid per second and so it penalises slower configurations.
- The reason is that at 15 W the GPU is capped at 612 MHz, but the 3.5 W static baseline is still paid for every frame and INT8 frames take 0.88 ms instead of 0.65 ms. By slowing down, the static share of each frame's energy grows. FP16 is dynamic-power-dominated so lowering clocks wins; INT8 already cut dynamic power by 39%, so static overhead now dominates and stretching the frame costs more than the clock reduction saves.
- The practical statement: the optimal power mode depends on the precision you deploy.

---
