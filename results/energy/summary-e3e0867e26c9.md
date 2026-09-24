# MobileNetV2 FP16 Energy Measurement

## Objective

This test run measures energy consumption of the MobileNetV2 FP16 engine, first with unpinned GPU clocks and then with clocks pinned to max. During the test the instantaneous power consumption is measured at idle and under load.

## Metrics of interest

- Throughput
- Total energy per frame
- P_board under load
- P_board at idle
- P_wall at idle
- P_wall / P_board ratio

P_board — what VDD_IN measures: the DC input to the devkit after the barrel jack,
sensed by the carrier-board INA3221, so it covers module *and* carrier loads.
P_wall — what the WM03-DE measures at the socket, including PSU loss.

---

## 1) Activate the virtual environment and enable sudo

```bash
source ~/venvs/eibl/bin/activate
sudo -v
```

---

## 2) Run the energy measurement on FP16 (unpinned)

```bash
python scripts/measure_energy.py models/mobilenetv2_fp16.plan --duration 120
```

---

## 3) Cool down and pin the clocks

```bash
sleep 300
sudo jetson_clocks
```

Note:

- Let the board cool back down before the second arm.

---

## 4) Rerun the energy measurement on FP16 (pinned)

```bash
python scripts/measure_energy.py models/mobilenetv2_fp16.plan --duration 120
```

---

## Results

### Unpinned vs. pinned

| Metric | Unpinned | Pinned | Δ |
| --- | ---: | ---: | ---: |
| Throughput | 1320.0 qps | 1329.5 qps | +0.7% |
| P_board load | 13.797 W | 14.001 W | +1.5% |
| Total mJ/frame | 10.753 | 10.788 | +0.3% |
| P_board idle | 3.534 W | 5.827 W | +65% |
| P_wall idle | 5.25 W | 8.15 W | +55% |

### P_wall / P_board ratio

| Metric | P_board | P_wall | Gap | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Idle unpinned | 3.53 W | 5.25 W | 1.72 W | 1.49× |
| Idle pinned | 5.83 W | 8.15 W | 2.32 W | 1.40× |
| Load unpinned | 13.80 W | 17.75 W | 3.95 W | 1.29× |
| Load pinned | 14.00 W | 17.95 W | 3.95 W | 1.28× |

### Interpretation

- Pinning clocks costs 2.29 W at idle and buys 0.7% throughput.
- Under continuous load it's a wash — 0.7% more work for 1.5% more power, so energy per frame is marginally worse pinned. The entire cost lands at idle, and the wall meter confirms it independently: 2.9 W at the socket for a board doing nothing.
- It means 13.8 mJ/frame at the wall, against 10.8 mJ/frame from tegrastats.

---
