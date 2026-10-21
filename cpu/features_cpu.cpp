// cpu/features_cpu.cpp -- the C++ baselines for the Phase 3 feature kernels.
//
// Built in the GPU job, on the same machine, with
//   g++ -O3 -march=native -fopenmp -shared -fPIC -o build/libfeatures_cpu.so cpu/features_cpu.cpp
// and called through ctypes on the same level-major NumPy arrays the GPU reads.
//
// threads == 1 runs a plain serial loop (the single-threaded baseline quoted in bullet 1);
// threads > 1 splits the snapshots into blocks with an OpenMP parallel for.
//
// The loop is level-outer, snapshot-inner inside a block of snapshots, which is the
// cache-friendly order for level-major data and lets the compiler vectorise the inner loop
// with AVX2; per snapshot the float32 sums run level 0..49 in order, exactly like the GPU.
// The outputs and their definitions match cuda/features.cu and bench/ref_features.py.

#include <cstdint>
#include <cstring>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int kTop = 10;
constexpr int64_t kBlock = 2048;

void features_block(const int32_t* bid_px, const float* bid_sz, const int32_t* ask_px,
                    const float* ask_sz, const uint8_t* product, const float* tick_usd,
                    int64_t n, int32_t levels, int64_t i0, int64_t i1,
                    int32_t* spread_ticks, float* spread_usd, float* depth_bid, float* depth_ask,
                    float* imbalance10, float* micro) {
    const int64_t m = i1 - i0;
    float db[kBlock], da[kBlock], b10[kBlock], a10[kBlock];
    std::memset(db, 0, sizeof(float) * m);
    std::memset(da, 0, sizeof(float) * m);
    for (int32_t l = 0; l < levels; ++l) {
        const float* bs = bid_sz + (int64_t)l * n + i0;
        const float* as = ask_sz + (int64_t)l * n + i0;
        for (int64_t k = 0; k < m; ++k) {
            db[k] += bs[k];
            da[k] += as[k];
        }
        if (l == kTop - 1) {
            std::memcpy(b10, db, sizeof(float) * m);
            std::memcpy(a10, da, sizeof(float) * m);
        }
    }
    for (int64_t k = 0; k < m; ++k) {
        const int64_t i = i0 + k;
        const int32_t bb = bid_px[i];
        const int32_t ba = ask_px[i];
        const bool both = bb > 0 && ba > 0;
        const int32_t s = both ? ba - bb : 0;
        spread_ticks[i] = s;
        spread_usd[i] = (float)s * tick_usd[product[i]];
        depth_bid[i] = db[k];
        depth_ask[i] = da[k];
        const float tot = b10[k] + a10[k];
        imbalance10[i] = tot > 0.0f ? (b10[k] - a10[k]) / tot : 0.0f;
        const float bs0 = bid_sz[i];
        const float as0 = ask_sz[i];
        const float tot0 = bs0 + as0;
        micro[i] = (both && tot0 > 0.0f) ? 0.5f * (float)s * ((bs0 - as0) / tot0) : 0.0f;
    }
}

}  // namespace

extern "C" int features_cpu(const int32_t* bid_px, const float* bid_sz, const int32_t* ask_px,
                            const float* ask_sz, const uint8_t* product, const float* tick_usd,
                            int64_t n, int32_t levels, int32_t threads,
                            int32_t* spread_ticks, float* spread_usd, float* depth_bid,
                            float* depth_ask, float* imbalance10, float* micro) {
    if (levels < kTop) return -1;
    const int64_t blocks = (n + kBlock - 1) / kBlock;
    if (threads <= 1) {
        for (int64_t b = 0; b < blocks; ++b) {
            const int64_t i0 = b * kBlock;
            features_block(bid_px, bid_sz, ask_px, ask_sz, product, tick_usd, n, levels, i0,
                           std::min(n, i0 + kBlock), spread_ticks, spread_usd, depth_bid, depth_ask,
                           imbalance10, micro);
        }
        return 1;
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t b = 0; b < blocks; ++b) {
        const int64_t i0 = b * kBlock;
        features_block(bid_px, bid_sz, ask_px, ask_sz, product, tick_usd, n, levels, i0,
                       std::min(n, i0 + kBlock), spread_ticks, spread_usd, depth_bid, depth_ask,
                       imbalance10, micro);
    }
    return threads;
#else
    return -2;
#endif
}

extern "C" int features_cpu_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
