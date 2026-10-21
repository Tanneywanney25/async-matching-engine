// cuda/ddscan.cu -- double-double (compensated float64) inclusive prefix scan.
//
// Each prefix is an unevaluated sum hi + lo of two doubles (about 106 significant bits), built
// with Knuth's TwoSum, so a rolling window formed as P[i] - P[j] keeps its precision even when
// P has grown to 1e12 times the window (a dust trade next to a day of volume). Every input used
// here is exactly representable in float64 (int ticks x float32 size, or a float32), so the
// result is accurate to about 2^-106 of the prefix.
//
// Structure (any n): dd_block_scan scans 1024-element blocks (256 threads x 4) with warp shuffles
// of both words and writes each block's total; dd_block_offsets turns the block totals into
// exclusive offsets with one thread (a serial compensated pass, tens of microseconds for
// thousands of blocks); dd_add_offsets adds them. The exclusive scans take the previous lane's
// inclusive value by shuffle, never by subtraction, so no step cancels.
// Additions only: no multiply, so FMA contraction cannot change TwoSum.
// Device code only (NVRTC through cupy.RawModule).

#define DD_BLOCK 1024
#define DD_THREADS 256
#define DD_FULL_MASK 0xffffffffu

__device__ __forceinline__ void dd_add(double ah, double al, double bh, double bl, double& rh, double& rl) {
    const double s = ah + bh;
    const double v = s - ah;
    double e = (ah - (s - v)) + (bh - v);
    e += al + bl;
    rh = s + e;
    rl = e - (rh - s);
}

extern "C" __global__ void __launch_bounds__(DD_THREADS)
dd_block_scan(const double* __restrict__ x, double* __restrict__ hi, double* __restrict__ lo,
              double* __restrict__ bhi, double* __restrict__ blo, long long n) {
    __shared__ double sh[DD_BLOCK];
    __shared__ double wh[DD_THREADS / 32], wl[DD_THREADS / 32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const long long base = (long long)blockIdx.x * DD_BLOCK;
#pragma unroll
    for (int k = 0; k < 4; ++k) {  // coalesced striped load
        const int idx = k * DD_THREADS + tid;
        sh[idx] = (base + idx < n) ? x[base + idx] : 0.0;
    }
    __syncthreads();
    double h[4], l[4];
    h[0] = sh[4 * tid];
    l[0] = 0.0;
#pragma unroll
    for (int k = 1; k < 4; ++k) dd_add(h[k - 1], l[k - 1], sh[4 * tid + k], 0.0, h[k], l[k]);
    // inclusive warp scan of the thread totals (h[3], l[3])
    double ih = h[3], il = l[3];
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const double oh = __shfl_up_sync(DD_FULL_MASK, ih, off);
        const double ol = __shfl_up_sync(DD_FULL_MASK, il, off);
        if (lane >= off) dd_add(oh, ol, ih, il, ih, il);
    }
    // exclusive within the warp: the previous lane's inclusive value
    double eh = __shfl_up_sync(DD_FULL_MASK, ih, 1);
    double el = __shfl_up_sync(DD_FULL_MASK, il, 1);
    if (lane == 0) { eh = 0.0; el = 0.0; }
    if (lane == 31) { wh[warp] = ih; wl[warp] = il; }
    __syncthreads();
    if (warp == 0) {
        double sh_ = (lane < DD_THREADS / 32) ? wh[lane] : 0.0;
        double sl_ = (lane < DD_THREADS / 32) ? wl[lane] : 0.0;
#pragma unroll
        for (int off = 1; off < DD_THREADS / 32; off <<= 1) {
            const double oh = __shfl_up_sync(DD_FULL_MASK, sh_, off);
            const double ol = __shfl_up_sync(DD_FULL_MASK, sl_, off);
            if (lane >= off) dd_add(oh, ol, sh_, sl_, sh_, sl_);
        }
        if (lane < DD_THREADS / 32) { wh[lane] = sh_; wl[lane] = sl_; }
    }
    __syncthreads();
    double ph = eh, pl = el;  // this thread's exclusive offset inside the block
    if (warp > 0) dd_add(wh[warp - 1], wl[warp - 1], eh, el, ph, pl);
    __syncthreads();  // sh is reused below
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        double rh, rl;
        dd_add(ph, pl, h[k], l[k], rh, rl);
        const long long idx = base + 4 * tid + k;
        if (idx < n) { hi[idx] = rh; lo[idx] = rl; }
    }
    if (tid == 0) { bhi[blockIdx.x] = wh[DD_THREADS / 32 - 1]; blo[blockIdx.x] = wl[DD_THREADS / 32 - 1]; }
}

// In place: block totals -> exclusive block offsets, one thread, compensated.
extern "C" __global__ void dd_block_offsets(double* __restrict__ bhi, double* __restrict__ blo, long long nb) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double ah = 0.0, al = 0.0;
    for (long long b = 0; b < nb; ++b) {
        const double th = bhi[b], tl = blo[b];
        bhi[b] = ah;
        blo[b] = al;
        dd_add(ah, al, th, tl, ah, al);
    }
}

extern "C" __global__ void dd_add_offsets(double* __restrict__ hi, double* __restrict__ lo,
                                          const double* __restrict__ ohi, const double* __restrict__ olo, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long b = i / DD_BLOCK;
    double rh, rl;
    dd_add(ohi[b], olo[b], hi[i], lo[i], rh, rl);
    hi[i] = rh;
    lo[i] = rl;
}

// W[i] = P[i] - P[i - window] (W[i] = P[i] for i < window). hi_i - hi_j is exact when the two
// prefixes are within a factor of two (Sterbenz), which is the cancelling case.
extern "C" __global__ void window_diff_dd(const double* __restrict__ hi, const double* __restrict__ lo,
                                          long long n, int window, double* __restrict__ W) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i < window) { W[i] = hi[i] + lo[i]; return; }
    const long long j = i - window;
    W[i] = (hi[i] - hi[j]) + (lo[i] - lo[j]);
}
