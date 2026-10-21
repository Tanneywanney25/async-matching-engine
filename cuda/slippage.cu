// cuda/slippage.cu -- Phase 5: the execution-cost surface (showcase kernel).
//
// For every snapshot, both sides and SLIPPAGE_SIZES log-spaced notional order
// sizes: average fill price from walking the book, slippage in basis points
// against mid, and the fraction filled within the recorded levels.
//
// Device code only. Compiled by NVRTC through cupy.RawModule with
// -arch=sm_75 --std=c++17 -default-device (bench.common.raw_module); no host
// code, no <cstdio>, no CUB.
//
// Conventions (bench/ref_slippage.py mirrors them exactly):
//   * prices are int32 ticks, sizes float32, arrays are level-major:
//     px[level * n_stride + i] is level `level` of snapshot `i`;
//   * side 0 = buy order walking the asks, side 1 = sell order walking the bids;
//   * targets[product * SLIPPAGE_SIZES + k] is order k's notional in tick units
//     (usd / tick_usd[product]), computed once on the host in float64;
//   * cumulative quantity and cumulative notional (px_ticks * sz) are float64.
//     (float32 quantity would put ~0.7 tick of rounding into an average price
//     near 11,000,000 ticks, far outside the 1e-3 tick verification bound.)
//   * a level with sz == 0 or px == 0 (padding for a missing level) ends the walk;
//   * when cum_notional_prev + px * sz >= target the order fills at this level
//     with (target - cum_notional_prev) / px of extra quantity (ties fill);
//   * avg price = filled notional / filled quantity;
//     fill fraction = filled notional / target, exactly 1.0f when fully filled;
//   * outputs are order-major: out[((i * 2) + side) * SLIPPAGE_SIZES + k] for
//     chunk-local snapshot i:
//       slip_bps        = +(avg - mid) / mid * 1e4 for buys, -(avg - mid) / mid * 1e4
//                         for sells (positive = worse than mid), float32
//       fill_frac       = filled notional / target, float32
//       avg_offset_ticks= avg - mid in ticks, signed the same way for both sides
//       levels_used     = number of levels touched, uint8
//     mid = (best bid + best ask) / 2 in ticks; when either best price is 0 the
//     mid is undefined and slip/offset are NaN. A side with no liquidity gives
//     NaN slip/offset, fill fraction 0 and levels_used 0.
//
// Indexing: n_stride and i0 are 64-bit so a 1.3M x 50 archive never overflows
// a flat index; per-chunk counts fit in int.

#ifndef SLIPPAGE_LEVELS
#define SLIPPAGE_LEVELS 50
#endif
#define SLIPPAGE_SIZES 64
#define SLIP_ORDERS_PER_SNAPSHOT (2 * SLIPPAGE_SIZES)
#define SLIP_NAIVE_BLOCK 256
#define SLIP_WARPS_PER_BLOCK 8
#define SLIP_WARP_BLOCK (SLIP_WARPS_PER_BLOCK * 32)
#define SLIP_FULL_MASK 0xffffffffu
#define SLIP_SHARED_LEVELS 64  // SLIPPAGE_LEVELS rounded up to a multiple of 32

__device__ __forceinline__ float slip_nan() { return __int_as_float(0x7fffffff); }

// Write one order's four outputs. `o` is the flat order index in the chunk.
__device__ __forceinline__ void slip_store(float* __restrict__ slip_bps,
                                           float* __restrict__ fill_frac,
                                           float* __restrict__ avg_offset,
                                           unsigned char* __restrict__ levels_used,
                                           unsigned long long o, int side, bool mid_ok,
                                           double mid, double target,
                                           double filled_notional, double filled_qty,
                                           int used, bool full) {
    float s = slip_nan();
    float off = slip_nan();
    float ff = full ? 1.0f : (float)(filled_notional / target);
    if (filled_qty > 0.0 && mid_ok) {
        const double avg = filled_notional / filled_qty;
        const double d = avg - mid;
        off = (float)d;
        s = (float)((side == 0 ? d : -d) / mid * 1e4);
    }
    slip_bps[o] = s;
    fill_frac[o] = ff;
    avg_offset[o] = off;
    levels_used[o] = (unsigned char)used;
}

// ---------------------------------------------------------------------------
// Naive: one thread per (snapshot, side, size). Thread t handles order
// t = ((i * 2) + side) * 64 + k, so a warp is 32 sizes of one snapshot side.
// Every lane reads the same level address (a broadcast load) but the walk
// length depends on the size, so the warp runs as long as its deepest order:
// that is the divergence the warp-cooperative kernel removes.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(SLIP_NAIVE_BLOCK)
slippage_naive(const int* __restrict__ bid_px, const float* __restrict__ bid_sz,
               const int* __restrict__ ask_px, const float* __restrict__ ask_sz,
               const unsigned char* __restrict__ product,
               const double* __restrict__ targets,
               long long n_stride, long long i0, int n_chunk,
               float* __restrict__ slip_bps, float* __restrict__ fill_frac,
               float* __restrict__ avg_offset, unsigned char* __restrict__ levels_used) {
    const long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    const long long n_orders = (long long)n_chunk * SLIP_ORDERS_PER_SNAPSHOT;
    if (tid >= n_orders) return;

    const long long i = tid / SLIP_ORDERS_PER_SNAPSHOT;              // chunk-local snapshot
    const int rem = (int)(tid - i * SLIP_ORDERS_PER_SNAPSHOT);        // 0 .. 127
    const int side = rem / SLIPPAGE_SIZES;                            // 0 buy, 1 sell
    const int k = rem - side * SLIPPAGE_SIZES;                        // 0 .. 63
    const unsigned long long col = (unsigned long long)(i0 + i);      // global snapshot index
    const unsigned long long stride = (unsigned long long)n_stride;

    const int* px = (side == 0) ? ask_px : bid_px;
    const float* sz = (side == 0) ? ask_sz : bid_sz;
    const int bb = bid_px[col];
    const int ba = ask_px[col];
    const bool mid_ok = (bb > 0) && (ba > 0);
    const double mid = 0.5 * ((double)bb + (double)ba);
    const double target = targets[(int)product[col] * SLIPPAGE_SIZES + k];

    double cum_n = 0.0;   // cumulative notional, ticks * size
    double cum_q = 0.0;   // cumulative quantity
    int used = 0;
    bool full = false;
    for (int l = 0; l < SLIPPAGE_LEVELS; ++l) {
        const unsigned long long idx = (unsigned long long)l * stride + col;
        const int p = px[idx];
        const float s = sz[idx];
        if (!(s > 0.0f) || p <= 0) break;                 // padding level ends the walk
        const double notional = (double)p * (double)s;    // exact: 24-bit * 24-bit mantissas
        if (cum_n + notional >= target) {
            cum_q += (target - cum_n) / (double)p;
            cum_n = target;
            used = l + 1;
            full = true;
            break;
        }
        cum_n += notional;
        cum_q += (double)s;
        used = l + 1;
    }
    slip_store(slip_bps, fill_frac, avg_offset, levels_used, (unsigned long long)tid, side,
               mid_ok, mid, target, cum_n, cum_q, used, full);
}

// ---------------------------------------------------------------------------
// Warp-cooperative: one warp per snapshot. For each side in turn, lane l loads
// levels l and l + 32 (when < 50), the warp builds the inclusive cumulative
// notional and quantity with __shfl_up_sync, writes them to shared memory, and
// every lane resolves sizes l and l + 32 by binary search over the cumulative
// notional: no data-dependent walk, every lane does the same six steps.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(SLIP_WARP_BLOCK)
slippage_warp(const int* __restrict__ bid_px, const float* __restrict__ bid_sz,
              const int* __restrict__ ask_px, const float* __restrict__ ask_sz,
              const unsigned char* __restrict__ product,
              const double* __restrict__ targets,
              long long n_stride, long long i0, int n_chunk,
              float* __restrict__ slip_bps, float* __restrict__ fill_frac,
              float* __restrict__ avg_offset, unsigned char* __restrict__ levels_used) {
    __shared__ double s_cn[SLIP_WARPS_PER_BLOCK][SLIP_SHARED_LEVELS];  // cumulative notional
    __shared__ double s_cq[SLIP_WARPS_PER_BLOCK][SLIP_SHARED_LEVELS];  // cumulative quantity
    __shared__ int s_px[SLIP_WARPS_PER_BLOCK][SLIP_SHARED_LEVELS];     // level prices (ticks)

    const int lane = threadIdx.x & 31;
    const int wib = threadIdx.x >> 5;
    const long long i = (long long)blockIdx.x * SLIP_WARPS_PER_BLOCK + wib;   // chunk-local snapshot
    if (i >= (long long)n_chunk) return;   // warp-uniform: the whole warp leaves together

    const unsigned long long col = (unsigned long long)(i0 + i);
    const unsigned long long stride = (unsigned long long)n_stride;
    const int prod = (int)product[col];
    const int bb = bid_px[col];
    const int ba = ask_px[col];
    const bool mid_ok = (bb > 0) && (ba > 0);
    const double mid = 0.5 * ((double)bb + (double)ba);

    double* cn = s_cn[wib];
    double* cq = s_cq[wib];
    int* spx = s_px[wib];

    for (int side = 0; side < 2; ++side) {
        const int* px = (side == 0) ? ask_px : bid_px;
        const float* sz = (side == 0) ? ask_sz : bid_sz;

        // 1. Load: lane l takes level l; lanes 0..17 also take level l + 32.
        const int p0 = px[(unsigned long long)lane * stride + col];
        const float s0 = sz[(unsigned long long)lane * stride + col];
        int p1 = 0;
        float s1 = 0.0f;
        if (lane + 32 < SLIPPAGE_LEVELS) {
            p1 = px[(unsigned long long)(lane + 32) * stride + col];
            s1 = sz[(unsigned long long)(lane + 32) * stride + col];
        }
        const bool v0 = (s0 > 0.0f) && (p0 > 0);
        const bool v1 = (s1 > 0.0f) && (p1 > 0);

        // 2. m = number of leading valid levels (the walk stops at the first padding level).
        const unsigned m0 = __ballot_sync(SLIP_FULL_MASK, v0);
        const unsigned m1 = __ballot_sync(SLIP_FULL_MASK, v1);
        int m;
        if (m0 != SLIP_FULL_MASK) {
            m = __ffs(~m0) - 1;                 // first invalid level among 0..31
        } else {
            m = 32 + (__ffs(~m1) - 1);          // lanes >= 18 have v1 == false, so m <= 50
        }

        // 3. Per-level notional and quantity, zero beyond m.
        double n0 = (lane < m) ? (double)p0 * (double)s0 : 0.0;
        double q0 = (lane < m) ? (double)s0 : 0.0;
        double n1 = (lane + 32 < m) ? (double)p1 * (double)s1 : 0.0;
        double q1 = (lane + 32 < m) ? (double)s1 : 0.0;

        // 4. Inclusive scan across lanes for levels 0..31, then 32..49 offset by the total.
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const double tn = __shfl_up_sync(SLIP_FULL_MASK, n0, off);
            const double tq = __shfl_up_sync(SLIP_FULL_MASK, q0, off);
            if (lane >= off) {
                n0 += tn;
                q0 += tq;
            }
        }
        const double tot_n = __shfl_sync(SLIP_FULL_MASK, n0, 31);
        const double tot_q = __shfl_sync(SLIP_FULL_MASK, q0, 31);
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const double tn = __shfl_up_sync(SLIP_FULL_MASK, n1, off);
            const double tq = __shfl_up_sync(SLIP_FULL_MASK, q1, off);
            if (lane >= off) {
                n1 += tn;
                q1 += tq;
            }
        }
        n1 += tot_n;
        q1 += tot_q;

        // 5. Publish the cumulative arrays to this warp's shared slice.
        __syncwarp();   // the previous side's binary searches are finished before overwriting
        cn[lane] = n0;
        cq[lane] = q0;
        spx[lane] = p0;
        if (lane + 32 < SLIPPAGE_LEVELS) {
            cn[lane + 32] = n1;
            cq[lane + 32] = q1;
            spx[lane + 32] = p1;
        }
        __syncwarp();

        // 6. Each lane resolves sizes k = lane and k = lane + 32.
#pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            const int k = lane + rep * 32;
            const double target = targets[prod * SLIPPAGE_SIZES + k];
            // smallest j in [0, m) with cn[j] >= target; j == m when the side runs out
            int lo = 0;
            int hi = m;
            while (lo < hi) {
                const int mid_idx = (lo + hi) >> 1;
                if (cn[mid_idx] >= target) {
                    hi = mid_idx;
                } else {
                    lo = mid_idx + 1;
                }
            }
            const int j = lo;
            double fn, fq;
            int used;
            bool full;
            if (j < m) {
                const double pn = (j > 0) ? cn[j - 1] : 0.0;
                const double pq = (j > 0) ? cq[j - 1] : 0.0;
                fq = pq + (target - pn) / (double)spx[j];
                fn = target;
                used = j + 1;
                full = true;
            } else {
                fn = (m > 0) ? cn[m - 1] : 0.0;
                fq = (m > 0) ? cq[m - 1] : 0.0;
                used = m;
                full = false;
            }
            const unsigned long long o =
                ((unsigned long long)i * 2ull + (unsigned long long)side) * SLIPPAGE_SIZES +
                (unsigned long long)k;
            slip_store(slip_bps, fill_frac, avg_offset, levels_used, o, side, mid_ok, mid, target,
                       fn, fq, used, full);
        }
    }
}
