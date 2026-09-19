# Jetson environment report

- generated: 2026-09-19T11:33:00Z

## Board identity

```
NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
--- L4T / JetPack ---
# R39 (release), REVISION: 2.1, GCID: 46758480, BOARD: generic, EABI: aarch64, DATE: Fri Aug  7 05:54:22 AM UTC 2026
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
nvidia-l4t-core	39.2.1-20260806224157
--- SoC ---
Tegra
NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
--- module part number ---
  (unavailable)

```

## OS / kernel / CPU

```
Distributor ID:	Ubuntu
Description:	Ubuntu 24.04.4 LTS
Release:	24.04
Codename:	noble
Linux jetson-orin 6.8.12-1021-tegra #1 SMP PREEMPT Thu Aug  6 21:56:04 PDT 2026 aarch64 aarch64 aarch64 GNU/Linux
--- CPU ---
Architecture:                            aarch64
CPU(s):                                  6
On-line CPU(s) list:                     0-5
Model name:                              Cortex-A78AE
CPU(s) scaling MHz:                      72%
CPU max MHz:                             1728.0000
CPU min MHz:                             115.2000
NUMA node0 CPU(s):                       0-5
--- online cores ---
0-5
```

## Memory

```
               total        used        free      shared  buff/cache   available
Mem:           7.3Gi       2.9Gi       1.6Gi        30Mi       3.0Gi       4.4Gi
Swap:             0B          0B          0B
--- swap ---
```

## JetPack component versions

```
--- apt: nvidia-jetpack ---
nvidia-jetpack	7.2.1-b49
--- CUDA ---
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Fri_May_08_10:52:52_AM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.86
Build cuda_13.2.r13.2/compiler.37953736_0
{
   "cuda" : {
      "name" : "CUDA SDK",
      "version" : "13.2.2"
   },
   "cuda_cccl" : {
      "name" : "CUDA C++ Core Compute Libraries",
      "version" : "13.2.86"
   },
   "cuda_crt" : {
      "name" : "CUDA crt Compiler for CUDA applications",
      "version" : "13.2.86"
   },
   "cuda_ctadvisor" : {
      "name" : "CUDA Compile Time Advisor",
      "version" : "13.2.86"
   },
   "cuda_cudart" : {
      "name" : "CUDA Runtime (cudart)",
      "version" : "13.2.86"
   },
   "cuda_culibos" : {
      "name" : "CUDA DEV culibos is a Math Libraries",
      "version" : "13.2.86"
   },
   "cuda_cuobjdump" : {
      "name" : "cuobjdump",
      "version" : "13.2.86"
   },
   "cuda_cupti" : {
      "name" : "CUPTI",
      "version" : "13.2.86"
   },
   "cuda_cuxxfilt" : {
      "name" : "CUDA cu++ filt",
      "version" : "13.2.86"
   },
   "cuda_gdb" : {
      "name" : "CUDA GDB",
      "version" : "13.2.86"
   },
   "cuda_nvcc" : {
      "name" : "CUDA NVCC",
      "version" : "13.2.86"
   },
   "cuda_nvdisasm" : {
      "name" : "CUDA nvdisasm",
      "version" : "13.2.86"
   },
   "cuda_nvml_dev" : {
      "name" : "CUDA NVML Headers",
      "version" : "13.2.86"
   },
   "cuda_nvprune" : {
      "name" : "CUDA nvprune",
      "version" : "13.2.86"
   },
   "cuda_nvrtc" : {
      "name" : "CUDA NVRTC",
      "version" : "13.2.86"
   },
   "cuda_nvtx" : {
      "name" : "CUDA NVTX",
      "version" : "13.2.86"
   },
   "cuda_profiler_api" : {
      "name" : "CUDA Profiler API",
      "version" : "13.2.86"
   },
   "cuda_sandbox_dev" : {
      "name" : "NVIDIA Sandbox Utils",
      "version" : "13.2.86"
   },
   "cuda_sanitizer_api" : {
      "name" : "CUDA Compute Sanitizer API",
      "version" : "13.2.87"
   },
   "libcublas" : {
      "name" : "CUDA cuBLAS",
      "version" : "13.4.1.3"
   },
   "libcudla" : {
      "name" : "CUDA cuDLA",
      "version" : "13.2.86"
   },
   "libcufft" : {
      "name" : "CUDA cuFFT",
      "version" : "12.2.0.57"
   },
   "libcufile" : {
      "name" : "GPUDirect Storage (cufile)",
      "version" : "1.17.1.22"
   },
   "libcurand" : {
      "name" : "CUDA cuRAND",
      "version" : "10.4.2.66"
   },
   "libcusolver" : {
      "name" : "CUDA cuSOLVER",
      "version" : "12.2.0.11"
   },
   "libcusparse" : {
      "name" : "CUDA cuSPARSE",
      "version" : "12.7.10.12"
   },
   "libnpp" : {
      "name" : "CUDA NPP",
      "version" : "13.1.0.59"
   },
   "libnvfatbin" : {
      "name" : "Fatbin interaction library",
      "version" : "13.2.86"
   },
   "libnvjitlink" : {
      "name" : "JIT Linker Library",
      "version" : "13.2.86"
   },
   "libnvjpeg" : {
      "name" : "CUDA nvJPEG",
      "version" : "13.1.0.59"
   },
   "libnvptxcompiler" : {
      "name" : "CUDA PTX compiler",
      "version" : "13.2.86"
   },
   "libnvvm" : {
      "name" : "NVVM",
      "version" : "13.2.86"
   },
   "nsight_compute" : {
      "name" : "Nsight Compute",
      "version" : "2026.1.1.2"
   },
   "nsight_systems" : {
      "name" : "Nsight Systems",
      "version" : "2025.6.3.541"
   },
   "nvidia_driver" : {
      "name" : "NVIDIA Linux Driver",
      "version" : "595.71.05"
   },
   "nvidia_fs" : {
      "name" : "NVIDIA file-system",
      "version" : "2.28.4"
   }
}
--- TensorRT ---
libnvinfer-bin 10.16.2.10-1+cuda13.2
libnvinfer-bin-cuda-13.2 
libnvinfer-dev 10.16.2.10-1+cuda13.2
libnvinfer-dev-cuda-13.2 
libnvinfer-dispatch-dev 10.16.2.10-1+cuda13.2
libnvinfer-dispatch-dev-cuda-13.2 
libnvinfer-dispatch10 10.16.2.10-1+cuda13.2
libnvinfer-dispatch10-cuda-13.2 
libnvinfer-headers-dev 10.16.2.10-1+cuda13.2
libnvinfer-headers-dev-cuda-13.2 
libnvinfer-headers-plugin-dev 10.16.2.10-1+cuda13.2
libnvinfer-headers-plugin-dev-cuda-13.2 
libnvinfer-headers-python-plugin-dev 10.16.2.10-1+cuda13.2
libnvinfer-headers-python-plugin-dev-cuda-13.2 
libnvinfer-lean-dev 10.16.2.10-1+cuda13.2
libnvinfer-lean-dev-cuda-13.2 
libnvinfer-lean10 10.16.2.10-1+cuda13.2
libnvinfer-lean10-cuda-13.2 
libnvinfer-plugin-dev 10.16.2.10-1+cuda13.2
libnvinfer-plugin-dev-cuda-13.2 
libnvinfer-plugin10 10.16.2.10-1+cuda13.2
libnvinfer-plugin10-cuda-13.2 
libnvinfer-plugin6 
libnvinfer-plugin7 
libnvinfer-safe-headers-dev 10.16.2.10-1+cuda13.2
libnvinfer-safe-headers-dev-cuda-13.2 
libnvinfer-vc-plugin-dev 10.16.2.10-1+cuda13.2
libnvinfer-vc-plugin-dev-cuda-13.2 
libnvinfer-vc-plugin10 10.16.2.10-1+cuda13.2
libnvinfer-vc-plugin10-cuda-13.2 
libnvinfer10 10.16.2.10-1+cuda13.2
libnvinfer10-cuda-13.2 
libnvinfer6 
libnvinfer7 
tensorrt 10.16.2.10-1+cuda13.2
tensorrt-cuda-13.2 
tensorrt-libs 10.16.2.10-1+cuda13.2
tensorrt-libs-cuda-13.2 
--- cuDNN ---
libcudnn8-dev 
libcudnn9 
libcudnn9-cuda-13 9.20.0.46-1
libcudnn9-dev 
libcudnn9-dev-cuda-12 
libcudnn9-dev-cuda-13 9.20.0.46-1
libcudnn9-headers 
libcudnn9-headers-cuda-13 9.20.0.46-1
libcudnn9-jit 
libcudnn9-jit-dev 
libcudnn9-jit-static 
libcudnn9-samples 9.20.0.46-1
--- VPI / OpenCV ---
libopencv 4.8.0-4-g18251aa
libopencv-calib3d-dev 
libopencv-calib3d4.0 
libopencv-calib3d406 
libopencv-calib3d406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-contrib406 
libopencv-contrib406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-core-dev 
libopencv-core4.0 
libopencv-core4.5 
libopencv-core406 
libopencv-core406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-dev 4.8.0-4-g18251aa
libopencv-dnn-dev 
libopencv-dnn4.0 
libopencv-dnn406 
libopencv-dnn406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-features2d-dev 
libopencv-features2d4.0 
libopencv-features2d406 
libopencv-features2d406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-flann-dev 
libopencv-flann4.0 
libopencv-flann406 
libopencv-flann406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-gapi-dev 
libopencv-gapi4.0 
libopencv-highgui-dev 
libopencv-highgui4.0 
libopencv-highgui406 
libopencv-highgui406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-imgcodecs-dev 
libopencv-imgcodecs4.0 
libopencv-imgcodecs406 
libopencv-imgcodecs406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-imgproc-dev 
libopencv-imgproc4.0 
libopencv-imgproc406 
libopencv-imgproc406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-ml-dev 
libopencv-ml4.0 
libopencv-ml406 
libopencv-ml406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-objdetect-dev 
libopencv-objdetect4.0 
libopencv-objdetect406 
libopencv-objdetect406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-photo-dev 
libopencv-photo4.0 
libopencv-photo406 
libopencv-photo406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-python 4.8.0-4-g18251aa
libopencv-samples 4.8.0-4-g18251aa
libopencv-shape406 
libopencv-shape406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-stitching-dev 
libopencv-stitching4.0 
libopencv-stitching406 
libopencv-stitching406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-ts-dev 
libopencv-ts4.0 
libopencv-video-dev 
libopencv-video4.0 
libopencv-video406 
libopencv-video406t64 4.6.0+dfsg-13.1ubuntu1
libopencv-videoio-dev 
libopencv-videoio4.0 
libopencv-viz406 
libopencv-viz406t64 4.6.0+dfsg-13.1ubuntu1
vpi-dev 
vpi2-dev 
vpi4-dev 4.1.4
vpi4-python-src 4.1.4
vpi4-samples 4.1.4
```

## jetson_release (if jetson-stats installed)

```
  jetson-stats not installed.
  Install with:  sudo pip3 install -U jetson-stats && sudo reboot
```

## Power mode and clocks

```
--- current nvpmodel mode ---
NV Power Mode: 25W
1

--- available modes ---
< POWER_MODEL ID=0 NAME=15W >
< POWER_MODEL ID=1 NAME=25W >
< POWER_MODEL ID=2 NAME=MAXN_SUPER >

--- jetson_clocks state ---
SOC family:tegra234  Machine:NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
Online CPUs: 0-5, Offline CPUs: 
cpu0:  Governor=schedutil MinFreq=729600 MaxFreq=1344000 CurrentFreq=960000 IdleStates: WFI=1 c7=1 
cpu1:  Governor=schedutil MinFreq=729600 MaxFreq=1344000 CurrentFreq=960000 IdleStates: WFI=1 c7=1 
cpu2:  Governor=schedutil MinFreq=729600 MaxFreq=1344000 CurrentFreq=1344000 IdleStates: WFI=1 c7=1 
cpu3:  Governor=schedutil MinFreq=729600 MaxFreq=1344000 CurrentFreq=1344000 IdleStates: WFI=1 c7=1 
cpu4:  Governor=schedutil MinFreq=729600 MaxFreq=1344000 CurrentFreq=1344000 IdleStates: WFI=1 c7=1 
cpu5:  Governor=schedutil MinFreq=729600 MaxFreq=1344000 CurrentFreq=1267200 IdleStates: WFI=1 c7=1 
GPU MinFreq=306000000 MaxFreq=918000000 CurrentFreq=306000000
Active GPU TPCs: 4
EMC MinFreq=204000000 MaxFreq=3199000000 CurrentFreq=665600000
FAN Dynamic Speed Control=nvfancontrol hwmon0_pwm1=60
FAN Dynamic Speed Control=nvfancontrol hwmon0_pwm1_enable=1
NV Power Mode: 25W
```

## Thermal zones (idle)

```
cpu-thermal            46.0 C
gpu-thermal            46.7 C
cv0-thermal            0.0 C
cv1-thermal            0.0 C
cv2-thermal            0.0 C
soc0-thermal           46.2 C
soc1-thermal           46.3 C
soc2-thermal           44.8 C
tj-thermal             47.0 C
```

## Power rails (idle) — INA3221

```
--- discovered hwmon paths ---

--- tegrastats, 5 samples @ 1s ---
09-19-2026 13:33:02 RAM 2850/7486MB (lfb 33x4MB) CPU [2%@729,0%@729,0%@729,0%@729,1%@729,0%@729] EMC_FREQ 5%@665 GR3D_FREQ 0%@[305] NVDEC0 off NVJPG0 off NVJPG1 off VIC off OFA off APE 200 cpu@45.812C/45.812C soc2@44.843C/44.843C soc0@46.093C/46.093C gpu@46.843C/46.843C tj@46.843C/46.843C soc1@46.281C/46.281C VDD_IN 3582mW/3582mW/3582mW VDD_CPU_GPU_CV 523mW/523mW/523mW VDD_SOC 1127mW/1127mW/1127mW
09-19-2026 13:33:03 RAM 2845/7486MB (lfb 33x4MB) CPU [1%@729,0%@729,0%@729,0%@729,0%@729,0%@729] EMC_FREQ 3%@665 GR3D_FREQ 0%@[305] NVDEC0 off NVJPG0 off NVJPG1 off VIC off OFA off APE 200 cpu@45.375C/45.812C soc2@44.687C/44.843C soc0@46.25C/46.25C gpu@46.562C/46.843C tj@46.562C/46.843C soc1@46.281C/46.281C VDD_IN 3462mW/3522mW/3582mW VDD_CPU_GPU_CV 442mW/483mW/523mW VDD_SOC 1127mW/1127mW/1127mW
09-19-2026 13:33:04 RAM 2840/7486MB (lfb 33x4MB) CPU [3%@729,0%@729,1%@729,0%@729,0%@729,0%@729] EMC_FREQ 3%@665 GR3D_FREQ 0%@[305] NVDEC0 off NVJPG0 off NVJPG1 off VIC off OFA off APE 200 cpu@45.687C/45.812C soc2@44.687C/44.843C soc0@46.093C/46.25C gpu@46.781C/46.843C tj@46.781C/46.843C soc1@46.25C/46.281C VDD_IN 3462mW/3502mW/3582mW VDD_CPU_GPU_CV 442mW/469mW/523mW VDD_SOC 1127mW/1127mW/1127mW
09-19-2026 13:33:05 RAM 2837/7486MB (lfb 33x4MB) CPU [1%@729,1%@729,0%@729,0%@729,0%@729,0%@729] EMC_FREQ 3%@665 GR3D_FREQ 0%@[306] NVDEC0 off NVJPG0 off NVJPG1 off VIC off OFA off APE 200 cpu@45.437C/45.812C soc2@44.718C/44.843C soc0@46.312C/46.312C gpu@46.5C/46.843C tj@46.5C/46.843C soc1@46.25C/46.281C VDD_IN 3462mW/3492mW/3582mW VDD_CPU_GPU_CV 442mW/462mW/523mW VDD_SOC 1127mW/1127mW/1127mW
```

## Storage

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/mmcblk0p1  116G   67G   43G  62% /
NAME           SIZE TYPE MOUNTPOINT
loop0            4K loop /snap/bare/5
loop1         61.9M loop /snap/core24/1644
loop2        248.4M loop /snap/firefox/8862
loop3        552.9M loop /snap/gnome-46-2404/154
loop4         91.7M loop /snap/gtk-common-themes/1535
loop5        188.2M loop /snap/mesa-2404/1836
loop6         43.4M loop /snap/snapd/27709
loop7        214.7M loop /snap/thunderbird/1241
loop8           16M loop 
loop9        215.2M loop /snap/thunderbird/1261
loop10       249.7M loop /snap/firefox/8926
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
/home/vdavid21/venvs/eibl/bin/python3
VIRTUAL_ENV=/home/vdavid21/venvs/eibl

--- key packages ---
  torch                2.14.0+cpu
  tensorrt             10.16.2.10
  onnx                 1.22.0
  onnxruntime          1.29.0
  onnxslim             0.1.96
  onnx_graphsurgeon    0.6.1
  polygraphy           0.53.4
  modelopt             0.46.0
  numpy                2.5.3
```

## PyTorch CUDA smoke test (proves a real kernel exists for this SM)

```
  torch             2.14.0+cpu
  cuda available    False
```

## TensorRT presence

```
--- trtexec ---
  found: /usr/src/tensorrt/bin/trtexec
&&&& RUNNING TensorRT.trtexec [TensorRT v101602] [b10] # /usr/src/tensorrt/bin/trtexec --help
=== Model Options ===
  --onnx=<file>               ONNX model

--- python bindings ---
  tensorrt 10.16.2.10
```

## Docker (optional, for jetson-containers)

```
Docker version 29.8.0, build 88096ef

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
