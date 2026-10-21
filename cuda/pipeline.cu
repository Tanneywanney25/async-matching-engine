// cuda/pipeline.cu -- Phase 7: the fused feature kernel of the three-stream pipeline.
//
// The pipeline (bench/bench_pipeline.py) splits the archive into K chunks, copies
// chunk i+1 host-to-device on a copy-in stream while this kernel runs on chunk i
// on the compute stream and chunk i-1 is copied back on a copy-out stream. One
// chunk is one contiguous block per array: inside a chunk the arrays keep the
// archive level-major layout, a[level * n + i] with n = snapshots in the chunk
// (bench_pipeline.py builds that chunk-major pinned copy once on the host).
//
// Per snapshot the kernel produces the ladder feature set (bench/CONTRACT.md):
//   spread_ticks      = best ask - best bid in ticks (int32); 0 when either side is empty
//   depth_bid/ask     = sum of the sizes of all `levels` levels of that side (float32)
//   imbalance10       = (bid_qty - ask_qty) / (bid_qty + ask_qty) over the top 10 levels,
//                       0 when the total is 0 (metrics.py book_imbalance, levels=10)
//   microprice_offset = microprice - mid in ticks, where
//                       microprice = (bid_px0 * ask_sz0 + ask_px0 * bid_sz0) / (bid_sz0 + ask_sz0),
//                       which simplifies exactly to 0.5 * spread * (bid_sz0 - ask_sz0) / (bid_sz0 + ask_sz0);
//                       0 when a side is empty or both top sizes are 0. The offset form
//                       keeps float32 precision: a microprice near 11,000,000 ticks would
//                       have a float32 spacing of 1 tick, the offset is O(spread).
// It also checks every book it reads: bids strictly decreasing and asks strictly
// increasing across present levels (px > 0 and sz > 0). A violation adds 1 to
// `bad_books`. The archive validator guarantees 0; the counter exists so that the
// int16 variant's on-device price decode of all 50 levels is observable.
//
// Two entry points share the feature code through a template:
//   pipeline_features_i32  prices as int32 ticks (the archive encoding);
//   pipeline_features_i16  prices as int16 tick offsets from the best price of the
//                          side (stretch: transfer-aware encoding). px = best + offset,
//                          offset == PIPELINE_I16_MISSING (-32768) marks a missing
//                          level (px = 0). Any |offset| > 32767 makes the host fall
//                          back to the int32 layout for that whole chunk.
//
// Device code only: compiled by NVRTC through cupy.RawModule with
// -arch=sm_75 --std=c++17 -default-device (bench.common.raw_module); no host
// code, no includes, no CUB. One thread per snapshot, 256-thread blocks; a warp
// reads 32 consecutive snapshots of one level, so every load is coalesced.
// `n` is the chunk length (fits int: a chunk is at most the whole archive,
// about 1.3M); flat indices are size_t.

#ifndef PIPELINE_BLOCK
#define PIPELINE_BLOCK 256
#endif
#define PIPELINE_IMB_LEVELS 10
#define PIPELINE_I16_MISSING (-32768)

// Price access for the int32 encoding: the level price is stored directly.
struct PipelinePxI32 {
    const int* px;
    __device__ __forceinline__ int best(size_t /*col*/) const { return 0; }  // unused
    __device__ __forceinline__ int load(size_t idx, int /*best*/) const { return px[idx]; }
};

// Price access for the int16 encoding: px = best + offset, sentinel = missing.
struct PipelinePxI16 {
    const short* off;
    const int* best_px;
    __device__ __forceinline__ int best(size_t col) const { return best_px[col]; }
    __device__ __forceinline__ int load(size_t idx, int best) const {
        const short o = off[idx];
        return (o == (short)PIPELINE_I16_MISSING) ? 0 : (best + (int)o);
    }
};

template <typename BidPx, typename AskPx>
__device__ __forceinline__ void pipeline_features_body(
        const BidPx bid, const float* __restrict__ bid_sz,
        const AskPx ask, const float* __restrict__ ask_sz,
        const int n, const int levels,
        int* __restrict__ spread_ticks, float* __restrict__ depth_bid,
        float* __restrict__ depth_ask, float* __restrict__ imbalance10,
        float* __restrict__ microprice_offset, unsigned int* __restrict__ bad_books) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t stride = (size_t)n;
    const size_t col = (size_t)i;
    // For the int16 encoding the best price of each side is read once per thread
    // (a coalesced int32 load); the int32 variant ignores it.
    const int bid_best = bid.best(col);
    const int ask_best = ask.best(col);

    float db = 0.0f, da = 0.0f;      // depth over all levels
    float b10 = 0.0f, a10 = 0.0f;    // depth over the top PIPELINE_IMB_LEVELS levels
    int bb = 0, ba = 0;              // best prices (level 0)
    float bs0 = 0.0f, as0 = 0.0f;    // best sizes (level 0)
    int prev_b = 0x7fffffff;         // last present bid price (bids strictly decrease)
    int prev_a = 0;                  // last present ask price (asks strictly increase)
    bool ok = true;

    for (int l = 0; l < levels; ++l) {
        const size_t idx = (size_t)l * stride + col;
        const int bp = bid.load(idx, bid_best);
        const float bs = bid_sz[idx];
        const int ap = ask.load(idx, ask_best);
        const float as = ask_sz[idx];
        if (l == 0) { bb = bp; ba = ap; bs0 = bs; as0 = as; }
        db += bs;
        da += as;
        if (l < PIPELINE_IMB_LEVELS) { b10 += bs; a10 += as; }
        if (bs > 0.0f && bp > 0) { if (!(bp < prev_b)) ok = false; prev_b = bp; }
        if (as > 0.0f && ap > 0) { if (!(ap > prev_a)) ok = false; prev_a = ap; }
    }

    const bool both = (bb > 0) && (ba > 0);
    const int spread = both ? (ba - bb) : 0;
    const float tot10 = b10 + a10;
    const float imb = (tot10 > 0.0f) ? (b10 - a10) / tot10 : 0.0f;
    const float tot0 = bs0 + as0;
    const float mp = (both && tot0 > 0.0f) ? 0.5f * (float)spread * ((bs0 - as0) / tot0) : 0.0f;

    spread_ticks[col] = spread;
    depth_bid[col] = db;
    depth_ask[col] = da;
    imbalance10[col] = imb;
    microprice_offset[col] = mp;
    if (!ok) atomicAdd(bad_books, 1u);
}

extern "C" __global__ void __launch_bounds__(PIPELINE_BLOCK)
pipeline_features_i32(const int* __restrict__ bid_px, const float* __restrict__ bid_sz,
                      const int* __restrict__ ask_px, const float* __restrict__ ask_sz,
                      int n, int levels,
                      int* __restrict__ spread_ticks, float* __restrict__ depth_bid,
                      float* __restrict__ depth_ask, float* __restrict__ imbalance10,
                      float* __restrict__ microprice_offset, unsigned int* __restrict__ bad_books) {
    PipelinePxI32 bid; bid.px = bid_px;
    PipelinePxI32 ask; ask.px = ask_px;
    pipeline_features_body(bid, bid_sz, ask, ask_sz, n, levels, spread_ticks, depth_bid,
                           depth_ask, imbalance10, microprice_offset, bad_books);
}

extern "C" __global__ void __launch_bounds__(PIPELINE_BLOCK)
pipeline_features_i16(const int* __restrict__ best_bid, const int* __restrict__ best_ask,
                      const short* __restrict__ bid_off, const short* __restrict__ ask_off,
                      const float* __restrict__ bid_sz, const float* __restrict__ ask_sz,
                      int n, int levels,
                      int* __restrict__ spread_ticks, float* __restrict__ depth_bid,
                      float* __restrict__ depth_ask, float* __restrict__ imbalance10,
                      float* __restrict__ microprice_offset, unsigned int* __restrict__ bad_books) {
    PipelinePxI16 bid; bid.off = bid_off; bid.best_px = best_bid;
    PipelinePxI16 ask; ask.off = ask_off; ask.best_px = best_ask;
    pipeline_features_body(bid, bid_sz, ask, ask_sz, n, levels, spread_ticks, depth_bid,
                           depth_ask, imbalance10, microprice_offset, bad_books);
}
