# Environment probe: what does the Kaggle T4 machine actually provide?
# Runs after the bootstrap (T4 check, env.json). Costs a few GPU minutes.
log("probe start")
report = {}
for cmd in [
    "nvidia-smi",
    "nvcc --version",
    "nsys --version",
    "ncu --version",
    "g++ --version | head -1",
    "which zstd tar sqlite3 cmake make",
    "nproc",
    "lscpu | grep -E 'Model name|Thread|Core|Socket|MHz|L3'",
    "free -g",
    "df -h /kaggle/working /tmp",
    "pip list 2>/dev/null | grep -i -E '^(cupy|numpy|numba|zstandard|scipy|pandas|pybind11) '",
    "ls /usr/local/cuda/include/cub | head -3",
    "ls /usr/local/cuda/bin | tr '\\n' ' '",
    "ls /opt/nvidia 2>/dev/null",
    "cat /proc/sys/kernel/perf_event_paranoid",
    "python -c \"import cupy; p=cupy.cuda.runtime.getDeviceProperties(0); print({k:(v if not isinstance(v,bytes) else v.decode()) for k,v in p.items() if k in ('name','multiProcessorCount','totalGlobalMem','sharedMemPerMultiprocessor','clockRate','memoryClockRate','memoryBusWidth','l2CacheSize','major','minor')})\"",
]:
    rc, out, err = sh(cmd)
    print(f"$ {cmd}\n{out}\n{err}\n", flush=True)
    report[cmd] = {"rc": rc, "out": out[-2000:], "err": err[-500:]}

# nvcc smoke test: compile and run a device-to-device copy kernel vs cudaMemcpy.
bw_src = r'''
#include <cstdio>
#include <cuda_runtime.h>
__global__ void copy4(const int4* __restrict__ a, int4* __restrict__ b, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) b[i] = a[i];
}
int main() {
    size_t bytes = 1ull << 30; size_t n = bytes / sizeof(int4);
    int4 *a, *b; cudaMalloc(&a, bytes); cudaMalloc(&b, bytes); cudaMemset(a, 1, bytes);
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    for (int w = 0; w < 3; ++w) copy4<<<40*8, 256>>>(a, b, n);
    cudaDeviceSynchronize();
    float best = 1e9f;
    for (int r = 0; r < 10; ++r) { cudaEventRecord(s); copy4<<<40*8, 256>>>(a, b, n); cudaEventRecord(e); cudaEventSynchronize(e); float ms; cudaEventElapsedTime(&ms, s, e); if (ms < best) best = ms; }
    printf("copy kernel: %.3f ms -> %.1f GB/s (read+write)\n", best, 2.0 * bytes / best / 1e6);
    best = 1e9f;
    for (int r = 0; r < 10; ++r) { cudaEventRecord(s); cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice); cudaEventRecord(e); cudaEventSynchronize(e); float ms; cudaEventElapsedTime(&ms, s, e); if (ms < best) best = ms; }
    printf("cudaMemcpy D2D: %.3f ms -> %.1f GB/s (read+write)\n", best, 2.0 * bytes / best / 1e6);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("device: %s SMs=%d sharedPerSM=%zu memClock=%d kHz bus=%d bits -> theoretical %.1f GB/s\n", p.name, p.multiProcessorCount, p.sharedMemPerMultiprocessor, p.memoryClockRate, p.memoryBusWidth, 2.0 * p.memoryClockRate * 1e3 * p.memoryBusWidth / 8 / 1e9);
    return 0;
}
'''
open("/kaggle/working/bw.cu", "w").write(bw_src)
rc, out, err = sh("cd /kaggle/working && nvcc -O3 -arch=sm_75 -o bw bw.cu && ./bw")
print("$ nvcc bw.cu && ./bw\n", out, "\n", err, flush=True)
report["bw"] = {"rc": rc, "out": out, "err": err[-500:]}

# nsys smoke test on a tiny CuPy workload, then the kernel summary report.
smoke = "import cupy as cp; x = cp.arange(1<<24, dtype=cp.float32); [ (x*2.0).sum() for _ in range(5) ]; cp.cuda.Device().synchronize()"
rc, out, err = sh(f"cd /kaggle/working && nsys profile --trace=cuda,nvtx --stats=false -o {OUT}/raw/probe_nsys python -c \"{smoke}\" 2>&1 | tail -3")
print("$ nsys profile ...\n", out, err, flush=True)
rc2, out2, err2 = sh(f"nsys stats --report cuda_gpu_kern_sum --format csv {OUT}/raw/probe_nsys.nsys-rep 2>&1 | head -8")
print("$ nsys stats ...\n", out2, err2, flush=True)
rc3, out3, err3 = sh(f"nsys export --type sqlite -o {OUT}/raw/probe_nsys.sqlite {OUT}/raw/probe_nsys.nsys-rep 2>&1 | tail -2 && python -c \"import sqlite3; c=sqlite3.connect('{OUT}/raw/probe_nsys.sqlite'); print([r[0] for r in c.execute('select name from sqlite_master where type=\\'table\\' and name like \\'CUPTI%\\'')])\"")
print("$ nsys export sqlite ...\n", out3, err3, flush=True)
report["nsys"] = {"profile": (rc, out, err[-300:]), "stats": (rc2, out2, err2[-300:]), "sqlite": (rc3, out3, err3[-300:])}

# ncu once: expected to fail on profiling permissions in a container.
rc, out, err = sh("ncu --version >/dev/null 2>&1 && ncu --metrics gpu__time_duration.sum python -c \"import cupy as cp; (cp.arange(1<<20)*2).sum(); cp.cuda.Device().synchronize()\" 2>&1 | tail -4")
print("$ ncu ...\n", out, err, flush=True)
report["ncu"] = {"rc": rc, "out": out[-800:], "err": err[-300:]}

json.dump(report, open(f"{OUT}/probe.json", "w"), indent=2)
finish("complete", {"probe": True})
