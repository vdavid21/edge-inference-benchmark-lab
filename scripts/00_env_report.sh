#!/usr/bin/env bash
# Edge Inference Benchmark Lab — environment report
# Captures everything needed to make a benchmark result reproducible.
# Usage:  bash 00_env_report.sh | tee env-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ).md
# Some sections need sudo; it will prompt once and degrade gracefully if refused.

set -uo pipefail

hr() { printf '\n## %s\n\n' "$1"; }
try() { "$@" 2>/dev/null || echo "  (unavailable)"; }

echo "# Jetson environment report"
echo
echo "- generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "- hostname:  $(hostname)"
echo "- user:      $(whoami)"

hr "Board identity"
echo '```'
try cat /proc/device-tree/model | tr -d '\0'; echo
echo "--- L4T / JetPack ---"
try cat /etc/nv_tegra_release
try dpkg-query --show nvidia-l4t-core
echo "--- SoC ---"
try cat /sys/devices/soc0/family
try cat /sys/devices/soc0/machine
echo "--- module part number ---"
try cat /proc/device-tree/nvidia,dtsfilename | tr -d '\0'; echo
echo '```'

hr "OS / kernel / CPU"
echo '```'
try lsb_release -a
uname -a
echo "--- CPU ---"
try lscpu | grep -E 'Model name|Architecture|CPU\(s\)|MHz'
echo "--- online cores ---"
try cat /sys/devices/system/cpu/online
echo '```'

hr "Memory"
echo '```'
try free -h
echo "--- swap ---"
try swapon --show
echo '```'

hr "JetPack component versions"
echo '```'
echo "--- apt: nvidia-jetpack ---"
try dpkg-query --show nvidia-jetpack
echo "--- CUDA ---"
try nvcc --version
try cat /usr/local/cuda/version.json
echo "--- TensorRT ---"
try dpkg-query -W -f='${Package} ${Version}\n' 'libnvinfer*' 'tensorrt*'
echo "--- cuDNN ---"
try dpkg-query -W -f='${Package} ${Version}\n' 'libcudnn*'
echo "--- VPI / OpenCV ---"
try dpkg-query -W -f='${Package} ${Version}\n' 'vpi*' 'libopencv*'
echo '```'

hr "jetson_release (if jetson-stats installed)"
echo '```'
if command -v jetson_release >/dev/null 2>&1; then
  jetson_release
else
  echo "  jetson-stats not installed."
  echo "  Install with:  sudo pip3 install -U jetson-stats && sudo reboot"
fi
echo '```'

hr "Power mode and clocks"
echo '```'
echo "--- current nvpmodel mode ---"
try sudo nvpmodel -q
echo
echo "--- available modes ---"
try grep -E '^< POWER_MODEL' /etc/nvpmodel.conf
echo
echo "--- jetson_clocks state ---"
try sudo jetson_clocks --show
echo '```'

hr "Thermal zones (idle)"
echo '```'
for z in /sys/devices/virtual/thermal/thermal_zone*; do
  [ -e "$z/type" ] || continue
  t=$(cat "$z/temp" 2>/dev/null)
  printf '%-22s %s C\n' "$(cat "$z/type")" "$(awk -v v="$t" 'BEGIN{printf "%.1f", v/1000}')"
done
echo '```'

hr "Power rails (idle) — INA3221"
echo '```'
echo "--- discovered hwmon paths ---"
try find /sys/bus/i2c/drivers/ina3221 -name 'in*_input' -o -name 'curr*_input' 2>/dev/null | head -40
echo
echo "--- tegrastats, 5 samples @ 1s ---"
if command -v tegrastats >/dev/null 2>&1; then
  ( sudo tegrastats --interval 1000 & echo $! > /tmp/.ts.pid ) 2>/dev/null
  sleep 5
  sudo kill "$(cat /tmp/.ts.pid)" 2>/dev/null
  rm -f /tmp/.ts.pid
else
  echo "  tegrastats not found"
fi
echo '```'

hr "Storage"
echo '```'
try df -h /
try lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
echo '```'

hr "Python environment"
echo '```'
echo "--- interpreter ---"
try python3 --version
try which python3
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-<none>}"
echo
echo "--- key packages ---"
python3 - <<'PY' 2>/dev/null || echo "  python3 introspection failed"
import importlib, sys
mods = ["torch","tensorrt","onnx","onnxruntime",
        "onnxslim","onnx_graphsurgeon","polygraphy","modelopt","numpy"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:20s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:20s} NOT INSTALLED ({type(e).__name__})")
PY
echo '```'

hr "PyTorch CUDA smoke test (proves a real kernel exists for this SM)"
echo '```'
python3 - <<'PY' 2>&1 || echo "  torch smoke test failed"
try:
    import torch
    print("  torch            ", torch.__version__)
    print("  cuda available   ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  device           ", torch.cuda.get_device_name(0))
        print("  capability       ", torch.cuda.get_device_capability(0))
        print("  cuda runtime     ", torch.version.cuda)
        print("  cudnn            ", torch.backends.cudnn.version())
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        torch.cuda.synchronize()
        print("  fp32 matmul ok   ", float((a @ b).sum()))
        ah, bh = a.half(), b.half()
        torch.cuda.synchronize()
        print("  fp16 matmul ok   ", float((ah @ bh).float().sum()))
except Exception as e:
    print("  FAILED:", type(e).__name__, e)
PY
echo '```'

hr "TensorRT presence"
echo '```'
echo "--- trtexec ---"
if [ -x /usr/src/tensorrt/bin/trtexec ]; then
  echo "  found: /usr/src/tensorrt/bin/trtexec"
  /usr/src/tensorrt/bin/trtexec --help 2>&1 | head -3
else
  try command -v trtexec
  echo "  /usr/src/tensorrt/bin/trtexec NOT found"
fi
echo
echo "--- python bindings ---"
python3 -c "import tensorrt as t; print('  tensorrt', t.__version__)" 2>&1 | head -3
echo '```'

hr "Docker (optional, for jetson-containers)"
echo '```'
try docker --version
try docker info --format '{{.DefaultRuntime}}'
echo '```'

hr "Summary checklist"
cat <<'EOF'
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
EOF
echo
echo "_end of report_"
