// cuda/layout.cu -- Phase 4 layout profiling: the same book-level kernels in array-of-structures
// and structure-of-arrays form, profiled under Nsight Systems by bench/profile_layout.py.
//
// Kernels (one thread per snapshot each):
//   spread        ask_px[0] - bid_px[0]
//   depth         sum of the 50 sizes per side
//   imbalance     top-10 (B - A) / (B + A)
//   cumdepth      per-snapshot inclusive scan of the 50 sizes per side (the cumulative depth a
//                 market order walks); AoS writes record-major output [i * 50 + l], SoA writes
//                 level-major output [l * n + i], each in its own layout.
// In the AoS form a warp's 32 threads touch 32 records 800 bytes apart on every load; in the SoA
// form they touch 32 consecutive 4-byte words. The arithmetic is identical.
// Device code only (NVRTC through cupy.RawModule, -arch=sm_75).

#define LAYOUT_LEVELS 50
#define LAYOUT_TOP 10

struct LayoutRec {
    int bid_px[LAYOUT_LEVELS];
    float bid_sz[LAYOUT_LEVELS];
    int ask_px[LAYOUT_LEVELS];
    float ask_sz[LAYOUT_LEVELS];
};

extern "C" __global__ void layout_spread_aos(const LayoutRec* __restrict__ r, long long n, int* __restrict__ out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = r[i].ask_px[0] - r[i].bid_px[0];
}

extern "C" __global__ void layout_depth_aos(const LayoutRec* __restrict__ r, long long n,
                                            float* __restrict__ db, float* __restrict__ da) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float b = 0.0f, a = 0.0f;
    for (int l = 0; l < LAYOUT_LEVELS; ++l) { b += r[i].bid_sz[l]; a += r[i].ask_sz[l]; }
    db[i] = b;
    da[i] = a;
}

extern "C" __global__ void layout_imbalance_aos(const LayoutRec* __restrict__ r, long long n, float* __restrict__ out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float b = 0.0f, a = 0.0f;
    for (int l = 0; l < LAYOUT_TOP; ++l) { b += r[i].bid_sz[l]; a += r[i].ask_sz[l]; }
    const float t = b + a;
    out[i] = t > 0.0f ? (b - a) / t : 0.0f;
}

extern "C" __global__ void layout_cumdepth_aos(const LayoutRec* __restrict__ r, long long n,
                                               float* __restrict__ cb, float* __restrict__ ca) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float b = 0.0f, a = 0.0f;
    for (int l = 0; l < LAYOUT_LEVELS; ++l) {
        b += r[i].bid_sz[l];
        a += r[i].ask_sz[l];
        cb[i * LAYOUT_LEVELS + l] = b;
        ca[i * LAYOUT_LEVELS + l] = a;
    }
}

extern "C" __global__ void layout_spread_soa(const int* __restrict__ bid_px, const int* __restrict__ ask_px,
                                             long long n, int* __restrict__ out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ask_px[i] - bid_px[i];
}

extern "C" __global__ void layout_depth_soa(const float* __restrict__ bid_sz, const float* __restrict__ ask_sz,
                                            long long n, float* __restrict__ db, float* __restrict__ da) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float b = 0.0f, a = 0.0f;
    for (int l = 0; l < LAYOUT_LEVELS; ++l) { b += bid_sz[l * n + i]; a += ask_sz[l * n + i]; }
    db[i] = b;
    da[i] = a;
}

extern "C" __global__ void layout_imbalance_soa(const float* __restrict__ bid_sz, const float* __restrict__ ask_sz,
                                                long long n, float* __restrict__ out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float b = 0.0f, a = 0.0f;
    for (int l = 0; l < LAYOUT_TOP; ++l) { b += bid_sz[l * n + i]; a += ask_sz[l * n + i]; }
    const float t = b + a;
    out[i] = t > 0.0f ? (b - a) / t : 0.0f;
}

extern "C" __global__ void layout_cumdepth_soa(const float* __restrict__ bid_sz, const float* __restrict__ ask_sz,
                                               long long n, float* __restrict__ cb, float* __restrict__ ca) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float b = 0.0f, a = 0.0f;
    for (int l = 0; l < LAYOUT_LEVELS; ++l) {
        b += bid_sz[l * n + i];
        a += ask_sz[l * n + i];
        cb[l * n + i] = b;
        ca[l * n + i] = a;
    }
}
