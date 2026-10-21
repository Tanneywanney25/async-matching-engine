// cuda/precision.cu -- Phase 4 precision study: where float32 prefix sums lose the window.
//
// A rolling-window sum computed as P[i] - P[i - w] from a float32 prefix P cancels two numbers
// that grow with i, so its relative error grows with the length of the stream. These kernels
// produce the window sums W[i] (w = window, W[i] = P[i] for i < w) and the prefix P[i] for:
//
//   * float32 naive: one global float32 scan (cuda/scan.cu), then window_diff_local with a single
//     chunk, so W = L[i] - L[i - w] is a float32 subtraction of two large prefixes;
//   * Kahan float32: kahan_chunks_f32 runs one thread per chunk of `chunk` elements, a serial
//     compensated (Kahan) prefix in float32, writing the running sum L and compensation C; chunk
//     totals are carried in float64. True prefix ~= offset[chunk] + (L - C);
//   * block-rebased float32: a float32 scan local to each 1024-element block (block_local_scan_f32)
//     relative to the block's start, with block totals reduced in float64 (block_totals_f64) and
//     carried in float64;
//   * float64: a float64 scan and window_diff_f64.
//
// For two indices in the same chunk the window is formed from the small local values in float32;
// across chunks it is formed in float64 from offset + local. Device code only (NVRTC, sm_75).

#define PREC_BLOCK 1024
#define PREC_THREADS 256
#define PREC_FULL_MASK 0xffffffffu

extern "C" __global__ void kahan_chunks_f32(const float* __restrict__ x, float* __restrict__ L,
                                            float* __restrict__ C, double* __restrict__ chunk_tot,
                                            long long n, int chunk) {
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long start = c * chunk;
    if (start >= n) return;
    long long end = start + chunk;
    if (end > n) end = n;
    float sum = 0.0f, comp = 0.0f;
    for (long long i = start; i < end; ++i) {
        const float y = x[i] - comp;
        const float t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        L[i] = sum;
        C[i] = comp;
    }
    chunk_tot[c] = (double)sum - (double)comp;
}

// float32 inclusive scan local to each block (no carry); same scheme as warpscan_f32.
extern "C" __global__ void __launch_bounds__(PREC_THREADS)
block_local_scan_f32(const float* __restrict__ in, float* __restrict__ out, long long n) {
    __shared__ float s[PREC_BLOCK];
    __shared__ float warp_tot[PREC_THREADS / 32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const long long base = (long long)blockIdx.x * PREC_BLOCK;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int idx = k * PREC_THREADS + tid;
        s[idx] = (base + idx < n) ? in[base + idx] : 0.0f;
    }
    __syncthreads();
    float v0 = s[4 * tid], v1 = s[4 * tid + 1], v2 = s[4 * tid + 2], v3 = s[4 * tid + 3];
    v1 += v0; v2 += v1; v3 += v2;
    const float tsum = v3;
    float xs = tsum;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const float y = __shfl_up_sync(PREC_FULL_MASK, xs, off);
        if (lane >= off) xs += y;
    }
    if (lane == 31) warp_tot[warp] = xs;
    __syncthreads();
    if (warp == 0) {
        float w = (lane < PREC_THREADS / 32) ? warp_tot[lane] : 0.0f;
#pragma unroll
        for (int off = 1; off < PREC_THREADS / 32; off <<= 1) {
            const float y = __shfl_up_sync(PREC_FULL_MASK, w, off);
            if (lane >= off) w += y;
        }
        if (lane < PREC_THREADS / 32) warp_tot[lane] = w;
    }
    __syncthreads();
    const float prefix = (xs - tsum) + (warp > 0 ? warp_tot[warp - 1] : 0.0f);
    s[4 * tid] = v0 + prefix; s[4 * tid + 1] = v1 + prefix; s[4 * tid + 2] = v2 + prefix; s[4 * tid + 3] = v3 + prefix;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int idx = k * PREC_THREADS + tid;
        if (base + idx < n) out[base + idx] = s[idx];
    }
}

// Exact-as-possible block totals: float32 inputs summed in float64.
extern "C" __global__ void __launch_bounds__(PREC_THREADS)
block_totals_f64(const float* __restrict__ in, double* __restrict__ tot, long long n) {
    __shared__ double warp_sum[PREC_THREADS / 32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const long long base = (long long)blockIdx.x * PREC_BLOCK;
    double acc = 0.0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const long long idx = base + k * PREC_THREADS + tid;
        if (idx < n) acc += (double)in[idx];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(PREC_FULL_MASK, acc, off);
    if (lane == 0) warp_sum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        double t = 0.0;
        for (int w = 0; w < PREC_THREADS / 32; ++w) t += warp_sum[w];
        tot[blockIdx.x] = t;
    }
}

// W[i] and P[i] from local float32 prefixes L (optional compensation C) and float64 chunk offsets.
extern "C" __global__ void window_diff_local(const float* __restrict__ L, const float* __restrict__ C,
                                             const double* __restrict__ offset, long long n, long long chunk,
                                             int window, int has_comp, double* __restrict__ W,
                                             double* __restrict__ P) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long bi = i / chunk;
    const float ci = has_comp ? C[i] : 0.0f;
    const double pi = offset[bi] + (double)(L[i] - ci);
    P[i] = pi;
    const long long j = i - window;
    if (j < 0) {
        W[i] = pi;
        return;
    }
    const long long bj = j / chunk;
    const float cj = has_comp ? C[j] : 0.0f;
    if (bi == bj) {
        W[i] = (double)((L[i] - L[j]) - (ci - cj));  // float32 arithmetic on local values
    } else {
        W[i] = pi - (offset[bj] + (double)(L[j] - cj));
    }
}

extern "C" __global__ void window_diff_f64(const double* __restrict__ P, long long n, int window,
                                           double* __restrict__ W) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    W[i] = (i >= window) ? P[i] - P[i - window] : P[i];
}
