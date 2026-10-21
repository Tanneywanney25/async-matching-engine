// cuda/ofi.cu -- Phase 6: multi-level order flow imbalance as a stencil kernel.
//
// Cont, Kukanov and Stoikov (2014), "The price impact of order book events",
// extended from the best level to K levels. Between consecutive snapshots
// t-1 and t of the same product, the level-k term is
//
//   e_k,t = 1{Pb_k,t >= Pb_k,t-1} qb_k,t - 1{Pb_k,t <= Pb_k,t-1} qb_k,t-1
//         - 1{Pa_k,t <= Pa_k,t-1} qa_k,t + 1{Pa_k,t >= Pa_k,t-1} qa_k,t-1
//
// and OFI_t = sum_k e_k,t. Positive OFI = net buying pressure at the book.
//
// The kernel is a stencil over per-product contiguous index arrays: `order`
// is a stable argsort of snap_product (built once on the host), so positions
// [start_p, end_p) hold product p's snapshots in time order. Thread `pos`
// reads snapshot order[pos] (t) and order[pos - 1] (t-1) through the index
// array. The first snapshot of each product (pos == 0 or a product change
// between pos - 1 and pos) gets 0. Output is written in sorted position order,
// so ofi_out[pos] belongs to snapshot order[pos].
//
// Missing levels are stored as px = 0, sz = 0. A missing level is given the
// worst possible price for its side before the comparison: 0 is already the
// worst bid, and OFI_ASK_MISSING (INT_MAX) is used for a missing ask. With
// that convention a level that disappears counts as liquidity removed and a
// level that appears counts as liquidity added, on both sides alike
// (bench/ref_ofi.py applies the same rule).
//
// Device code only (NVRTC via bench.common.raw_module). Level-major arrays:
// a[level * n + i] with n the stride passed by the host (the host may upload
// only the first `levels` rows, the stride stays n). 64-bit flat indices. The
// 4 * K level terms are accumulated in float64 and stored as float32.

#define OFI_BLOCK 256
#define OFI_ASK_MISSING 2147483647

extern "C" __global__ void __launch_bounds__(OFI_BLOCK)
ofi_stencil(const int* __restrict__ bid_px, const float* __restrict__ bid_sz,
            const int* __restrict__ ask_px, const float* __restrict__ ask_sz,
            const unsigned char* __restrict__ product, const int* __restrict__ order,
            long long n, int levels, float* __restrict__ ofi_out) {
    const long long pos = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (pos >= n) return;
    const long long i = (long long)order[pos];
    if (pos == 0) {
        ofi_out[pos] = 0.0f;
        return;
    }
    const long long j = (long long)order[pos - 1];
    if (product[i] != product[j]) {   // first snapshot of this product
        ofi_out[pos] = 0.0f;
        return;
    }
    const unsigned long long un = (unsigned long long)n;
    const unsigned long long ui = (unsigned long long)i;
    const unsigned long long uj = (unsigned long long)j;
    double acc = 0.0;
    for (int k = 0; k < levels; ++k) {
        const unsigned long long oi = (unsigned long long)k * un + ui;
        const unsigned long long oj = (unsigned long long)k * un + uj;
        const int bpi = bid_px[oi];
        const int bpj = bid_px[oj];
        const float bsi = bid_sz[oi];
        const float bsj = bid_sz[oj];
        int api = ask_px[oi];
        int apj = ask_px[oj];
        const float asi = ask_sz[oi];
        const float asj = ask_sz[oj];
        if (api <= 0) api = OFI_ASK_MISSING;   // missing ask = worst (highest) price
        if (apj <= 0) apj = OFI_ASK_MISSING;
        double e = 0.0;
        if (bpi >= bpj) e += (double)bsi;
        if (bpi <= bpj) e -= (double)bsj;
        if (api <= apj) e -= (double)asi;
        if (api >= apj) e += (double)asj;
        acc += e;
    }
    ofi_out[pos] = (float)acc;
}
