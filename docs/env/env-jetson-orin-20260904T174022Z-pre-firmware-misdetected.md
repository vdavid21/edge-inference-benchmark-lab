# Jetson environment report

- generated: 2026-09-04T17:40:22Z
- **Disclaimer note: During environment setup it was noticed that the JetPack 7.2 installer misdetected the Orin Nano Super Developer Kit and omited the 25 W and MAXN SUPER power profiles, capping the board at 15 W / 624 MHz. Firmware update including upgrading the bootleader fixed the problem.** 

## Board identity

```
NVIDIA Jetson Orin Nano Developer Kit
--- L4T / JetPack ---
# R39 (release), REVISION: 2.0, GCID: 45755727, BOARD: generic, EABI: aarch64, DATE: Mon Jun  1 09:28:48 PM UTC 2026
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
nvidia-l4t-core	39.2.0-20260601141651
--- SoC ---
Tegra
NVIDIA Jetson Orin Nano Developer Kit
--- module part number ---
  (unavailable)

```

## OS / kernel / CPU

```
Distributor ID:	Ubuntu
Description:	Ubuntu 24.04.4 LTS
Release:	24.04
Codename:	noble
Linux jetson-orin 6.8.12-1021-tegra #1 SMP PREEMPT Mon Jun  1 13:25:46 PDT 2026 aarch64 aarch64 aarch64 GNU/Linux
--- CPU ---
Architecture:                            aarch64
CPU(s):                                  6
On-line CPU(s) list:                     0-5
Model name:                              Cortex-A78AE
CPU(s) scaling MHz:                      100%
CPU max MHz:                             1510.4000
CPU min MHz:                             115.2000
NUMA node0 CPU(s):                       0-5
--- online cores ---
0-5
```

## Memory

```
               total        used        free      shared  buff/cache   available
Mem:           7.3Gi       1.9Gi       3.8Gi       7.3Mi       1.8Gi       5.4Gi
Swap:             0B          0B          0B
--- swap ---
```

## JetPack component versions

```
--- apt: nvidia-jetpack ---
  (unavailable)
--- CUDA ---
  (unavailable)
  (unavailable)
--- TensorRT ---
  (unavailable)
--- cuDNN ---
  (unavailable)
--- VPI / OpenCV ---
libopencv-calib3d406 
libopencv-calib3d406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-contrib406 
libopencv-contrib406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-core-dev 4.6.0+dfsg-13.1ubuntu1
libopencv-core4.5 
libopencv-core406 
libopencv-core406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-dnn-dev 4.6.0+dfsg-13.1ubuntu1
libopencv-dnn406 
libopencv-dnn406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-features2d406 
libopencv-features2d406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-flann-dev 4.6.0+dfsg-13.1ubuntu1
libopencv-flann406 
libopencv-flann406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-highgui406 
libopencv-highgui406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-imgcodecs-dev 4.6.0+dfsg-13.1ubuntu1
libopencv-imgcodecs406 
libopencv-imgcodecs406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-imgproc-dev 4.6.0+dfsg-13.1ubuntu1
libopencv-imgproc406 
libopencv-imgproc406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-ml-dev 4.6.0+dfsg-13.1ubuntu1
libopencv-ml406 
libopencv-ml406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-objdetect406 
libopencv-objdetect406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-photo-dev 4.6.0+dfsg-13.1ubuntu1
libopencv-photo406 
libopencv-photo406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-shape-dev 4.6.0+dfsg-13.1ubuntu1
libopencv-shape406 
libopencv-shape406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-stitching406 
libopencv-stitching406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-video-dev 4.6.0+dfsg-13.1ubuntu1
libopencv-video406 
libopencv-video406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-viz-dev 4.6.0+dfsg-13.1ubuntu1
libopencv-viz406 
libopencv-viz406t64 4.6.0+dfsg-13.1ubuntu1
  (unavailable)
```

## jetson_release (if jetson-stats installed)

```
  jetson-stats not installed.
  Install with:  sudo pip3 install -U jetson-stats && sudo reboot
```

## Power mode and clocks

```
--- current nvpmodel mode ---
NV Power Mode: 15W
0

--- available modes ---
< POWER_MODEL ID=0 NAME=15W >
< POWER_MODEL ID=1 NAME=7W >

--- jetson_clocks state ---
SOC family:tegra234  Machine:NVIDIA Jetson Orin Nano Developer Kit
Online CPUs: 0-5, Offline CPUs: 
cpu0:  Governor=schedutil MinFreq=729600 MaxFreq=1510400 CurrentFreq=1420800 IdleStates: WFI=1 c7=1 
cpu1:  Governor=schedutil MinFreq=729600 MaxFreq=1510400 CurrentFreq=1420800 IdleStates: WFI=1 c7=1 
cpu2:  Governor=schedutil MinFreq=729600 MaxFreq=1510400 CurrentFreq=1510400 IdleStates: WFI=1 c7=1 
cpu3:  Governor=schedutil MinFreq=729600 MaxFreq=1510400 CurrentFreq=1510400 IdleStates: WFI=1 c7=1 
cpu4:  Governor=schedutil MinFreq=729600 MaxFreq=1510400 CurrentFreq=729600 IdleStates: WFI=1 c7=1 
cpu5:  Governor=schedutil MinFreq=729600 MaxFreq=1510400 CurrentFreq=1036800 IdleStates: WFI=1 c7=1 
GPU MinFreq=306000000 MaxFreq=624750000 CurrentFreq=306000000
Active GPU TPCs: 4
EMC MinFreq=204000000 MaxFreq=2133000000 CurrentFreq=2133000000
FAN Dynamic Speed Control=nvfancontrol hwmon0_pwm1=62
FAN Dynamic Speed Control=nvfancontrol hwmon0_pwm1_enable=1
NV Power Mode: 15W
```

## Thermal zones (idle)

```
cpu-thermal            48.0 C
gpu-thermal            48.0 C
cv0-thermal            0.0 C
cv1-thermal            0.0 C
cv2-thermal            0.0 C
soc0-thermal           45.9 C
soc1-thermal           46.9 C
soc2-thermal           46.8 C
tj-thermal             48.2 C
```

## Power rails (idle) — INA3221

```
--- discovered hwmon paths ---

--- tegrastats, 5 samples @ 1s ---
09-04-2026 19:40:46 RAM 1803/7486MB (lfb 8x4MB) CPU [66%@1510,2%@1510,0%@1510,1%@1510,33%@729,0%@729] EMC_FREQ 1%@2133 GR3D_FREQ 4%@[305] NVDEC0 off NVJPG0 off NVJPG1 off VIC off OFA off APE 200 cpu@48C/48C soc2@46.906C/46.906C soc0@46.031C/46.031C gpu@48.125C/48.125C tj@48.125C/48.125C soc1@46.906C/46.906C VDD_IN 4832mW/4832mW/4832mW VDD_CPU_GPU_CV 1236mW/1236mW/1236mW VDD_SOC 1437mW/1437mW/1437mW
09-04-2026 19:40:47 RAM 1803/7486MB (lfb 8x4MB) CPU [64%@729,0%@729,0%@729,0%@729,0%@729,0%@729] EMC_FREQ 5%@665 GR3D_FREQ 12%@[306] NVDEC0 off NVJPG0 off NVJPG1 off VIC off OFA off APE 200 cpu@47.562C/48C soc2@46.75C/46.906C soc0@45.843C/46.031C gpu@48C/48.125C tj@48C/48.125C soc1@46.937C/46.937C VDD_IN 4093mW/4463mW/4832mW VDD_CPU_GPU_CV 802mW/1019mW/1236mW VDD_SOC 1243mW/1340mW/1437mW
09-04-2026 19:40:48 RAM 1802/7486MB (lfb 8x4MB) CPU [8%@729,2%@729,20%@729,1%@729,0%@729,0%@729] EMC_FREQ 4%@665 GR3D_FREQ 2%@[305] NVDEC0 off NVJPG0 off NVJPG1 off VIC off OFA off APE 200 cpu@47.468C/48C soc2@46.656C/46.906C soc0@45.906C/46.031C gpu@47.718C/48.125C tj@47.718C/48.125C soc1@47C/47C VDD_IN 3703mW/4209mW/4832mW VDD_CPU_GPU_CV 602mW/880mW/1236mW VDD_SOC 1127mW/1269mW/1437mW
09-04-2026 19:40:49 RAM 1802/7486MB (lfb 8x4MB) CPU [8%@729,0%@729,22%@729,0%@729,0%@729,0%@729] EMC_FREQ 4%@665 GR3D_FREQ 2%@[305] NVDEC0 off NVJPG0 off NVJPG1 off VIC off OFA off APE 200 cpu@47.593C/48C soc2@46.718C/46.906C soc0@45.906C/46.031C gpu@47.656C/48.125C tj@47.656C/48.125C soc1@46.968C/47C VDD_IN 3663mW/4073mW/4832mW VDD_CPU_GPU_CV 562mW/801mW/1236mW VDD_SOC 1127mW/1234mW/1437mW
```

## Storage

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/mmcblk0p1  116G   42G   69G  38% /
NAME           SIZE TYPE MOUNTPOINT
loop0           16M loop 
mmcblk0      119.2G disk 
├─mmcblk0p1  117.7G part /
├─mmcblk0p2    128M part 
├─mmcblk0p3    768K part 
├─mmcblk0p4   31.6M part 
├─mmcblk0p5    128M part 
├─mmcblk0p6    768K part 
├─mmcblk0p7   31.6M part 
├─mmcblk0p8    100M part 
├─mmcblk0p9    512K part 
├─mmcblk0p10    64M part /boot/efi
├─mmcblk0p11   100M part 
├─mmcblk0p12   512K part 
├─mmcblk0p13    64M part 
├─mmcblk0p14   400M part 
└─mmcblk0p15 479.5M part 
```

## Python environment

```
--- interpreter ---
Python 3.12.3
/usr/bin/python3
VIRTUAL_ENV=<none>

--- key packages ---
  torch                NOT INSTALLED (ModuleNotFoundError)
  torchvision          NOT INSTALLED (ModuleNotFoundError)
  tensorrt             NOT INSTALLED (ModuleNotFoundError)
  onnx                 NOT INSTALLED (ModuleNotFoundError)
  onnxruntime          NOT INSTALLED (ModuleNotFoundError)
  onnxsim              NOT INSTALLED (ModuleNotFoundError)
  onnx_graphsurgeon    NOT INSTALLED (ModuleNotFoundError)
  polygraphy           NOT INSTALLED (ModuleNotFoundError)
  modelopt             NOT INSTALLED (ModuleNotFoundError)
  numpy                1.26.4
  cv2                  NOT INSTALLED (ModuleNotFoundError)
```

## PyTorch CUDA smoke test (proves a real kernel exists for this SM)

```
  FAILED: ModuleNotFoundError No module named 'torch'
```

## TensorRT presence

```
--- trtexec ---
  (unavailable)
  /usr/src/tensorrt/bin/trtexec NOT found

--- python bindings ---
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'tensorrt'
```

## Docker (optional, for jetson-containers)

```
Docker version 29.1.3, build 29.1.3-0ubuntu3~24.04.2

  (unavailable)
```

## Summary checklist

Fill this in after reading the output above:

- [ ] Module variant (Orin Nano 4GB / 8GB / Super): 
- [ ] JetPack version:
- [ ] TensorRT version:
- [ ] CUDA version:
- [ ] Python version:
- [ ] Current nvpmodel mode:
- [ ] Super mode (25W / MAXN SUPER) available:
- [ ] PyTorch installed and CUDA-functional:
- [ ] trtexec present:
- [ ] Booting from SD card or NVMe SSD:
- [ ] Free disk space:

_end of report_
