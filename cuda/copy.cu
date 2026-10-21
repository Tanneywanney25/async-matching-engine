// cuda/copy.cu -- device-to-device copy kernel used to measure the achievable
// memory bandwidth ceiling in the same job as the feature kernels.
// 128-bit loads and stores in a grid-stride loop; n counts int4 elements.
// Device code only (NVRTC through cupy.RawModule, -arch=sm_75).

extern "C" __global__ void copy_int4(const int4* __restrict__ src, int4* __restrict__ dst, long long n) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += step) {
        dst[i] = src[i];
    }
}

// Read-only streaming kernel: the ceiling for read-dominated kernels. A copy splits DRAM time
// between reads and writes (and pays the bus turnaround), so a kernel that mostly reads can beat
// it; this measures the read side alone. Each thread folds its loads into one word and writes it
// once (grid * block * 4 bytes of writes, negligible), so the loads cannot be optimized away.
extern "C" __global__ void read_int4(const int4* __restrict__ src, long long n, unsigned int* __restrict__ sink) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long step = (long long)gridDim.x * blockDim.x;
    unsigned int acc = 0u;
    for (long long i = tid; i < n; i += step) {
        const int4 v = src[i];
        acc ^= (unsigned int)(v.x ^ v.y ^ v.z ^ v.w);
    }
    sink[tid] = acc;
}
