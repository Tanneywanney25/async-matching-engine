// cuda/features.cu -- Phase 3: per-snapshot book features and the optimization ladder.
//
// Outputs per snapshot i (bench/ref_features.py defines them identically):
//   spread_ticks[i]  int32  ask_px[0] - bid_px[0]; 0 when either side is empty
//   spread_usd[i]    f32    spread_ticks * tick_usd[product[i]]
//   depth_bid[i]     f32    sum of bid_sz over the 50 levels (padding levels hold 0)
//   depth_ask[i]     f32    same for the asks
//   imbalance10[i]   f32    (B10 - A10) / (B10 + A10) over the top 10 levels, 0 when the
//                           total is 0 (exactly metrics.book_imbalance with levels=10)
//   microprice_off[i] f32   microprice - mid in ticks, microprice = (bb*as0 + ba*bs0)/(bs0+as0),
//                           mid = (bb + ba)/2, which equals 0.5*spread*(bs0 - as0)/(bs0 + as0);
//                           0 when a side is empty or both top sizes are 0. The offset keeps
//                           float32 precision (an absolute microprice near 11,000,000 ticks
//                           would have a float32 spacing of one tick).
// Sums run level 0..49 in float32 in that order, the same order as the C++ baseline.
//
// Layouts:
//   AoS  (v0): one FeatRec of 800 bytes per snapshot; thread i reads record i, so a warp's
//              32 loads of one field are 800 bytes apart (uncoalesced).
//   SoA  (v1..v4): level-major arrays, a[level * n_stride + i]; a warp reads 32 consecutive
//              snapshots of one level (coalesced). n_stride is n rounded up to a multiple
//              of 4 so every row starts 16-byte aligned for the 128-bit loads of v3/v4.
//
// Device code only (NVRTC through cupy.RawModule, -arch=sm_75): no includes, no host code.

#define FEAT_LEVELS 50
#define FEAT_TOP 10
#define FEAT_BLOCK_MAX 1024

struct FeatRec {
    int bid_px[FEAT_LEVELS];
    float bid_sz[FEAT_LEVELS];
    int ask_px[FEAT_LEVELS];
    float ask_sz[FEAT_LEVELS];
};

__device__ __forceinline__ float feat_imbalance(float b10, float a10) {
    const float tot = b10 + a10;
    return (tot > 0.0f) ? (b10 - a10) / tot : 0.0f;
}

__device__ __forceinline__ int feat_spread(int bb, int ba) {
    return (bb > 0 && ba > 0) ? (ba - bb) : 0;
}

__device__ __forceinline__ float feat_micro(int bb, int ba, float bs0, float as0) {
    const float tot = bs0 + as0;
    if (!(bb > 0 && ba > 0) || !(tot > 0.0f)) return 0.0f;
    return 0.5f * (float)(ba - bb) * ((bs0 - as0) / tot);
}

// ---------------------------------------------------------------------------
// v0: array-of-structures, one kernel per metric
// ---------------------------------------------------------------------------

extern "C" __global__ void v0_spread(const FeatRec* __restrict__ recs,
                                     const unsigned char* __restrict__ product,
                                     const float* __restrict__ tick_usd, long long n,
                                     int* __restrict__ spread_ticks, float* __restrict__ spread_usd) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int s = feat_spread(recs[i].bid_px[0], recs[i].ask_px[0]);
    spread_ticks[i] = s;
    spread_usd[i] = (float)s * tick_usd[product[i]];
}

extern "C" __global__ void v0_depth(const FeatRec* __restrict__ recs, long long n,
                                    float* __restrict__ depth_bid, float* __restrict__ depth_ask) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float db = 0.0f, da = 0.0f;
#pragma unroll 10
    for (int l = 0; l < FEAT_LEVELS; ++l) {
        db += recs[i].bid_sz[l];
        da += recs[i].ask_sz[l];
    }
    depth_bid[i] = db;
    depth_ask[i] = da;
}

extern "C" __global__ void v0_imbalance(const FeatRec* __restrict__ recs, long long n,
                                        float* __restrict__ imbalance10) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float b = 0.0f, a = 0.0f;
#pragma unroll
    for (int l = 0; l < FEAT_TOP; ++l) {
        b += recs[i].bid_sz[l];
        a += recs[i].ask_sz[l];
    }
    imbalance10[i] = feat_imbalance(b, a);
}

extern "C" __global__ void v0_microprice(const FeatRec* __restrict__ recs, long long n,
                                         float* __restrict__ micro) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    micro[i] = feat_micro(recs[i].bid_px[0], recs[i].ask_px[0], recs[i].bid_sz[0], recs[i].ask_sz[0]);
}

// ---------------------------------------------------------------------------
// v1: structure-of-arrays, one kernel per metric
// ---------------------------------------------------------------------------

extern "C" __global__ void v1_spread(const int* __restrict__ bid_px, const int* __restrict__ ask_px,
                                     const unsigned char* __restrict__ product,
                                     const float* __restrict__ tick_usd, long long n,
                                     int* __restrict__ spread_ticks, float* __restrict__ spread_usd) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int s = feat_spread(bid_px[i], ask_px[i]);  // level 0 row
    spread_ticks[i] = s;
    spread_usd[i] = (float)s * tick_usd[product[i]];
}

extern "C" __global__ void v1_depth(const float* __restrict__ bid_sz, const float* __restrict__ ask_sz,
                                    long long n, long long n_stride,
                                    float* __restrict__ depth_bid, float* __restrict__ depth_ask) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float db = 0.0f, da = 0.0f;
#pragma unroll 10
    for (int l = 0; l < FEAT_LEVELS; ++l) {
        const long long idx = (long long)l * n_stride + i;
        db += bid_sz[idx];
        da += ask_sz[idx];
    }
    depth_bid[i] = db;
    depth_ask[i] = da;
}

extern "C" __global__ void v1_imbalance(const float* __restrict__ bid_sz, const float* __restrict__ ask_sz,
                                        long long n, long long n_stride, float* __restrict__ imbalance10) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float b = 0.0f, a = 0.0f;
#pragma unroll
    for (int l = 0; l < FEAT_TOP; ++l) {
        const long long idx = (long long)l * n_stride + i;
        b += bid_sz[idx];
        a += ask_sz[idx];
    }
    imbalance10[i] = feat_imbalance(b, a);
}

extern "C" __global__ void v1_microprice(const int* __restrict__ bid_px, const float* __restrict__ bid_sz,
                                         const int* __restrict__ ask_px, const float* __restrict__ ask_sz,
                                         long long n, float* __restrict__ micro) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    micro[i] = feat_micro(bid_px[i], ask_px[i], bid_sz[i], ask_sz[i]);  // level 0 rows
}

// ---------------------------------------------------------------------------
// v2: structure-of-arrays, every metric fused into one pass
// ---------------------------------------------------------------------------

extern "C" __global__ void v2_fused(const int* __restrict__ bid_px, const float* __restrict__ bid_sz,
                                    const int* __restrict__ ask_px, const float* __restrict__ ask_sz,
                                    const unsigned char* __restrict__ product,
                                    const float* __restrict__ tick_usd, long long n, long long n_stride,
                                    int* __restrict__ spread_ticks, float* __restrict__ spread_usd,
                                    float* __restrict__ depth_bid, float* __restrict__ depth_ask,
                                    float* __restrict__ imbalance10, float* __restrict__ micro) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float db = 0.0f, da = 0.0f, b10 = 0.0f, a10 = 0.0f;
#pragma unroll
    for (int l = 0; l < FEAT_TOP; ++l) {
        const long long idx = (long long)l * n_stride + i;
        const float bs = bid_sz[idx];
        const float as = ask_sz[idx];
        b10 += bs;
        a10 += as;
        db += bs;
        da += as;
    }
#pragma unroll 10
    for (int l = FEAT_TOP; l < FEAT_LEVELS; ++l) {
        const long long idx = (long long)l * n_stride + i;
        db += bid_sz[idx];
        da += ask_sz[idx];
    }
    const int bb = bid_px[i];
    const int ba = ask_px[i];
    const int s = feat_spread(bb, ba);
    spread_ticks[i] = s;
    spread_usd[i] = (float)s * tick_usd[product[i]];
    depth_bid[i] = db;
    depth_ask[i] = da;
    imbalance10[i] = feat_imbalance(b10, a10);
    micro[i] = feat_micro(bb, ba, bid_sz[i], ask_sz[i]);
}

// ---------------------------------------------------------------------------
// v3 / v4: fused, 128-bit loads, four consecutive snapshots per thread
// ---------------------------------------------------------------------------

// One group g covers snapshots 4g .. 4g+3. Rows are 16-byte aligned because n_stride % 4 == 0
// and the allocations are 256-byte aligned, so the float4 / int4 / uchar4 casts are legal.
// Outputs are padded to n_stride elements; the padding columns read zeros and write zeros.
__device__ __forceinline__ void feat_group4(const int* __restrict__ bid_px, const float* __restrict__ bid_sz,
                                            const int* __restrict__ ask_px, const float* __restrict__ ask_sz,
                                            const unsigned char* __restrict__ product,
                                            const float* __restrict__ tick_usd, long long n_stride, long long g,
                                            int* __restrict__ spread_ticks, float* __restrict__ spread_usd,
                                            float* __restrict__ depth_bid, float* __restrict__ depth_ask,
                                            float* __restrict__ imbalance10, float* __restrict__ micro) {
    const float4* bsz4 = reinterpret_cast<const float4*>(bid_sz);
    const float4* asz4 = reinterpret_cast<const float4*>(ask_sz);
    const long long row4 = n_stride >> 2;  // float4 elements per level row
    float4 db = make_float4(0.f, 0.f, 0.f, 0.f), da = db, b10 = db, a10 = db;
    float4 bs0 = db, as0 = db;
#pragma unroll
    for (int l = 0; l < FEAT_TOP; ++l) {
        const float4 b = bsz4[(long long)l * row4 + g];
        const float4 a = asz4[(long long)l * row4 + g];
        if (l == 0) { bs0 = b; as0 = a; }
        b10.x += b.x; b10.y += b.y; b10.z += b.z; b10.w += b.w;
        a10.x += a.x; a10.y += a.y; a10.z += a.z; a10.w += a.w;
        db.x += b.x; db.y += b.y; db.z += b.z; db.w += b.w;
        da.x += a.x; da.y += a.y; da.z += a.z; da.w += a.w;
    }
#pragma unroll 10
    for (int l = FEAT_TOP; l < FEAT_LEVELS; ++l) {
        const float4 b = bsz4[(long long)l * row4 + g];
        const float4 a = asz4[(long long)l * row4 + g];
        db.x += b.x; db.y += b.y; db.z += b.z; db.w += b.w;
        da.x += a.x; da.y += a.y; da.z += a.z; da.w += a.w;
    }
    const int4 bb = reinterpret_cast<const int4*>(bid_px)[g];  // level 0 row
    const int4 ba = reinterpret_cast<const int4*>(ask_px)[g];
    const uchar4 pr = reinterpret_cast<const uchar4*>(product)[g];
    int4 s;
    s.x = feat_spread(bb.x, ba.x);
    s.y = feat_spread(bb.y, ba.y);
    s.z = feat_spread(bb.z, ba.z);
    s.w = feat_spread(bb.w, ba.w);
    float4 su = make_float4((float)s.x * tick_usd[pr.x], (float)s.y * tick_usd[pr.y],
                            (float)s.z * tick_usd[pr.z], (float)s.w * tick_usd[pr.w]);
    float4 im = make_float4(feat_imbalance(b10.x, a10.x), feat_imbalance(b10.y, a10.y),
                            feat_imbalance(b10.z, a10.z), feat_imbalance(b10.w, a10.w));
    float4 mp = make_float4(feat_micro(bb.x, ba.x, bs0.x, as0.x), feat_micro(bb.y, ba.y, bs0.y, as0.y),
                            feat_micro(bb.z, ba.z, bs0.z, as0.z), feat_micro(bb.w, ba.w, bs0.w, as0.w));
    reinterpret_cast<int4*>(spread_ticks)[g] = s;
    reinterpret_cast<float4*>(spread_usd)[g] = su;
    reinterpret_cast<float4*>(depth_bid)[g] = db;
    reinterpret_cast<float4*>(depth_ask)[g] = da;
    reinterpret_cast<float4*>(imbalance10)[g] = im;
    reinterpret_cast<float4*>(micro)[g] = mp;
}

extern "C" __global__ void v3_fused_vec4(const int* __restrict__ bid_px, const float* __restrict__ bid_sz,
                                         const int* __restrict__ ask_px, const float* __restrict__ ask_sz,
                                         const unsigned char* __restrict__ product,
                                         const float* __restrict__ tick_usd, long long n_stride,
                                         int* __restrict__ spread_ticks, float* __restrict__ spread_usd,
                                         float* __restrict__ depth_bid, float* __restrict__ depth_ask,
                                         float* __restrict__ imbalance10, float* __restrict__ micro) {
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (n_stride >> 2)) return;
    feat_group4(bid_px, bid_sz, ask_px, ask_sz, product, tick_usd, n_stride, g,
                spread_ticks, spread_usd, depth_bid, depth_ask, imbalance10, micro);
}

extern "C" __global__ void v4_fused_vec4_gs(const int* __restrict__ bid_px, const float* __restrict__ bid_sz,
                                            const int* __restrict__ ask_px, const float* __restrict__ ask_sz,
                                            const unsigned char* __restrict__ product,
                                            const float* __restrict__ tick_usd, long long n_stride,
                                            int* __restrict__ spread_ticks, float* __restrict__ spread_usd,
                                            float* __restrict__ depth_bid, float* __restrict__ depth_ask,
                                            float* __restrict__ imbalance10, float* __restrict__ micro) {
    const long long groups = n_stride >> 2;
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x; g < groups; g += step) {
        feat_group4(bid_px, bid_sz, ask_px, ask_sz, product, tick_usd, n_stride, g,
                    spread_ticks, spread_usd, depth_bid, depth_ask, imbalance10, micro);
    }
}
