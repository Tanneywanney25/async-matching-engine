// cuda/rolling.cu -- Phase 4: rolling windows built on inclusive prefix scans.
//
// All inputs are in "segment order": the host stable-sorts trades (and snapshots) by product
// once, so product p occupies positions [seg_start[p], seg_end[p]) in receive-time order, and
// one global inclusive scan P serves every product: the sum over positions lo..hi is
// P[hi] - P[lo - 1] (P[-1] = 0) even across segment boundaries.
//
//   vwap_trade_window   the last `window` trades of the same product including trade i, exactly
//                       metrics.rolling_vwap: sum(px * sz) / sum(sz), in ticks
//   vwap_time_window    trades of the snapshot's product with receive time in (t - W, t], sampled
//                       at every snapshot time t by binary search; NaN when the window is empty
//   book_features_sorted spread (ticks) and top-10 imbalance per snapshot, written in segment order
//   window_mean          mean of a scanned series over the snapshots of the same product with time
//                       in (t - W, t] (t = the snapshot's own time)
//
// Device code only (NVRTC through cupy.RawModule, -arch=sm_75).

__device__ __forceinline__ double rolling_nan() { return __longlong_as_double(0x7ff8000000000000ll); }

// first index in [lo, hi) with ts[idx] > t (upper bound)
__device__ __forceinline__ long long upper_bound_ll(const long long* __restrict__ ts, long long lo, long long hi,
                                                    long long t) {
    while (lo < hi) {
        const long long mid = (lo + hi) >> 1;
        if (ts[mid] <= t) lo = mid + 1; else hi = mid;
    }
    return lo;
}

extern "C" __global__ void vwap_trade_window(const double* __restrict__ P, const double* __restrict__ S,
                                             const unsigned char* __restrict__ prod_sorted,
                                             const long long* __restrict__ seg_start, long long n, int window,
                                             double* __restrict__ vwap) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long s = seg_start[prod_sorted[i]];
    long long lo = i - window;          // the window is lo+1 .. i
    if (lo < s - 1) lo = s - 1;
    const double num = P[i] - (lo >= 0 ? P[lo] : 0.0);
    const double den = S[i] - (lo >= 0 ? S[lo] : 0.0);
    vwap[i] = den > 0.0 ? num / den : rolling_nan();
}

extern "C" __global__ void vwap_time_window(const long long* __restrict__ snap_ts,
                                            const unsigned char* __restrict__ snap_prod, long long n_snap,
                                            const long long* __restrict__ trade_ts_sorted,
                                            const double* __restrict__ P, const double* __restrict__ S,
                                            const long long* __restrict__ seg_start,
                                            const long long* __restrict__ seg_end, long long window_ns,
                                            double* __restrict__ out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_snap) return;
    const int p = snap_prod[i];
    const long long t = snap_ts[i];
    const long long s = seg_start[p];
    const long long e = seg_end[p];
    const long long hi = upper_bound_ll(trade_ts_sorted, s, e, t) - 1;            // last trade with ts <= t
    const long long lo = upper_bound_ll(trade_ts_sorted, s, e, t - window_ns);    // first with ts > t - W
    if (hi < lo) {
        out[i] = rolling_nan();
        return;
    }
    const double num = P[hi] - (lo > 0 ? P[lo - 1] : 0.0);
    const double den = S[hi] - (lo > 0 ? S[lo - 1] : 0.0);
    out[i] = den > 0.0 ? num / den : rolling_nan();
}

extern "C" __global__ void book_features_sorted(const int* __restrict__ bid_px, const float* __restrict__ bid_sz,
                                                const int* __restrict__ ask_px, const float* __restrict__ ask_sz,
                                                const long long* __restrict__ order, long long n, long long n_stride,
                                                double* __restrict__ spread_sorted,
                                                double* __restrict__ imbalance_sorted) {
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const long long i = order[k];
    const int bb = bid_px[i];
    const int ba = ask_px[i];
    spread_sorted[k] = (bb > 0 && ba > 0) ? (double)(ba - bb) : 0.0;
    float b = 0.0f, a = 0.0f;
#pragma unroll
    for (int l = 0; l < 10; ++l) {
        b += bid_sz[(long long)l * n_stride + i];
        a += ask_sz[(long long)l * n_stride + i];
    }
    const float tot = b + a;
    imbalance_sorted[k] = tot > 0.0f ? (double)((b - a) / tot) : 0.0;
}

extern "C" __global__ void window_mean(const long long* __restrict__ ts_sorted,
                                       const unsigned char* __restrict__ prod_sorted,
                                       const long long* __restrict__ seg_start, const double* __restrict__ P,
                                       long long n, long long window_ns, double* __restrict__ out) {
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const long long s = seg_start[prod_sorted[k]];
    const long long t = ts_sorted[k];
    const long long lo = upper_bound_ll(ts_sorted, s, k + 1, t - window_ns);  // first with ts > t - W
    const double sum = P[k] - (lo > 0 ? P[lo - 1] : 0.0);
    out[k] = sum / (double)(k - lo + 1);
}
