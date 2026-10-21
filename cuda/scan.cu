// cuda/scan.cu -- Phase 4: inclusive prefix scans in shared memory.
//
// Two block-level designs, both scanning SCAN_ELEMS = 1024 elements per block and writing the
// block's total to block_sums[blockIdx.x]; the host scans block_sums recursively with the same
// kernel and adds the scanned totals back with scan_add_offsets_*, so any n works.
//
//   blelloch_*   work-efficient Blelloch (up-sweep / down-sweep) scan, 512 threads, two elements
//                per thread, shared-memory indices padded by (i >> SHIFT) so the tree's strided
//                accesses fall into different banks (SHIFT 5 for 4-byte, 4 for 8-byte elements).
//   warpscan_*   256 threads x 4 elements: a coalesced striped load through shared memory, a serial
//                scan of each thread's four elements, an inclusive warp scan of the thread totals
//                with __shfl_up_sync, then a scan of the eight warp totals by warp 0.
//
// One single-pass design, the decoupled look-back of Merrill and Garland (2016):
//
//   lookback_*   one pass over the data. Each block takes the next tile from an atomic counter
//                (tiles are claimed in order, so every predecessor of a tile belongs to a block that
//                has already started: CUDA gives blocks no forward-progress guarantee, so spinning
//                on a tile whose block may not be resident yet could deadlock), scans its
//                256 x 8 elements, publishes its aggregate with status A, then warp 0 walks back over
//                32 predecessors at a time, adding aggregates until it reaches a tile whose
//                inclusive prefix is published (status P), publishes its own inclusive prefix and
//                broadcasts the exclusive one to the block. Status words and values live in separate
//                arrays (a double and a status do not fit one 64-bit word, and 128-bit accesses are
//                not single-copy atomic in the PTX memory model): values are written first and the
//                status with st.release.gpu; readers load the status with ld.acquire.gpu and only then
//                the value, through L2 (__ldcg). lookback_init resets the statuses and the counter.
//
// Device code only (NVRTC through cupy.RawModule, -arch=sm_75). 64-bit global indices.

#define SCAN_ELEMS 1024
#define BLELLOCH_THREADS 512
#define WARPSCAN_THREADS 256
#define WARPSCAN_ITEMS 4
#define SCAN_FULL_MASK 0xffffffffu

template <typename T, int SHIFT>
__device__ __forceinline__ void blelloch_body(const T* __restrict__ in, T* __restrict__ out,
                                              T* __restrict__ block_sums, long long n) {
    __shared__ T temp[SCAN_ELEMS + (SCAN_ELEMS >> SHIFT) + 1];
    const int tid = threadIdx.x;
    const long long base = (long long)blockIdx.x * SCAN_ELEMS;
    const int ai = tid;
    const int bi = tid + SCAN_ELEMS / 2;
    const int pai = ai + (ai >> SHIFT);
    const int pbi = bi + (bi >> SHIFT);
    const T a = (base + ai < n) ? in[base + ai] : T(0);
    const T b = (base + bi < n) ? in[base + bi] : T(0);
    temp[pai] = a;
    temp[pbi] = b;
    int offset = 1;
    for (int d = SCAN_ELEMS >> 1; d > 0; d >>= 1) {  // up-sweep (reduce)
        __syncthreads();
        if (tid < d) {
            int x = offset * (2 * tid + 1) - 1;
            int y = offset * (2 * tid + 2) - 1;
            x += x >> SHIFT;
            y += y >> SHIFT;
            temp[y] += temp[x];
        }
        offset <<= 1;
    }
    const int last = (SCAN_ELEMS - 1) + ((SCAN_ELEMS - 1) >> SHIFT);
    __syncthreads();
    if (tid == 0) {
        block_sums[blockIdx.x] = temp[last];
        temp[last] = T(0);
    }
    for (int d = 1; d < SCAN_ELEMS; d <<= 1) {  // down-sweep
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int x = offset * (2 * tid + 1) - 1;
            int y = offset * (2 * tid + 2) - 1;
            x += x >> SHIFT;
            y += y >> SHIFT;
            const T t = temp[x];
            temp[x] = temp[y];
            temp[y] += t;
        }
    }
    __syncthreads();
    // temp holds the exclusive scan; adding the element makes it inclusive
    if (base + ai < n) out[base + ai] = temp[pai] + a;
    if (base + bi < n) out[base + bi] = temp[pbi] + b;
}

template <typename T>
__device__ __forceinline__ void warpscan_body(const T* __restrict__ in, T* __restrict__ out,
                                              T* __restrict__ block_sums, long long n) {
    __shared__ T s[SCAN_ELEMS];
    __shared__ T warp_tot[WARPSCAN_THREADS / 32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const long long base = (long long)blockIdx.x * SCAN_ELEMS;
#pragma unroll
    for (int k = 0; k < WARPSCAN_ITEMS; ++k) {  // striped, coalesced load
        const int idx = k * WARPSCAN_THREADS + tid;
        s[idx] = (base + idx < n) ? in[base + idx] : T(0);
    }
    __syncthreads();
    T v0 = s[WARPSCAN_ITEMS * tid + 0];
    T v1 = s[WARPSCAN_ITEMS * tid + 1];
    T v2 = s[WARPSCAN_ITEMS * tid + 2];
    T v3 = s[WARPSCAN_ITEMS * tid + 3];
    v1 += v0;
    v2 += v1;
    v3 += v2;
    const T tsum = v3;
    T x = tsum;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const T y = __shfl_up_sync(SCAN_FULL_MASK, x, off);
        if (lane >= off) x += y;
    }
    if (lane == 31) warp_tot[warp] = x;
    __syncthreads();
    if (warp == 0) {
        T w = (lane < WARPSCAN_THREADS / 32) ? warp_tot[lane] : T(0);
#pragma unroll
        for (int off = 1; off < WARPSCAN_THREADS / 32; off <<= 1) {
            const T y = __shfl_up_sync(SCAN_FULL_MASK, w, off);
            if (lane >= off) w += y;
        }
        if (lane < WARPSCAN_THREADS / 32) warp_tot[lane] = w;
    }
    __syncthreads();
    const T prefix = (x - tsum) + (warp > 0 ? warp_tot[warp - 1] : T(0));  // exclusive, this thread
    s[WARPSCAN_ITEMS * tid + 0] = v0 + prefix;
    s[WARPSCAN_ITEMS * tid + 1] = v1 + prefix;
    s[WARPSCAN_ITEMS * tid + 2] = v2 + prefix;
    s[WARPSCAN_ITEMS * tid + 3] = v3 + prefix;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < WARPSCAN_ITEMS; ++k) {
        const int idx = k * WARPSCAN_THREADS + tid;
        if (base + idx < n) out[base + idx] = s[idx];
    }
    if (tid == 0) block_sums[blockIdx.x] = warp_tot[WARPSCAN_THREADS / 32 - 1];
}

// Adds the scanned block totals: element i of block b gets sums_inclusive[b - 1] (b > 0).
template <typename T>
__device__ __forceinline__ void add_offsets_body(T* __restrict__ out, const T* __restrict__ sums_inclusive,
                                                 long long n) {
    const long long b = blockIdx.x;
    if (b == 0) return;
    const T off = sums_inclusive[b - 1];
    const long long base = b * SCAN_ELEMS;
    for (int k = threadIdx.x; k < SCAN_ELEMS; k += blockDim.x) {
        if (base + k < n) out[base + k] += off;
    }
}

extern "C" __global__ void __launch_bounds__(BLELLOCH_THREADS)
blelloch_f64(const double* __restrict__ in, double* __restrict__ out, double* __restrict__ block_sums, long long n) {
    blelloch_body<double, 4>(in, out, block_sums, n);
}
extern "C" __global__ void __launch_bounds__(BLELLOCH_THREADS)
blelloch_f32(const float* __restrict__ in, float* __restrict__ out, float* __restrict__ block_sums, long long n) {
    blelloch_body<float, 5>(in, out, block_sums, n);
}
extern "C" __global__ void __launch_bounds__(WARPSCAN_THREADS)
warpscan_f64(const double* __restrict__ in, double* __restrict__ out, double* __restrict__ block_sums, long long n) {
    warpscan_body<double>(in, out, block_sums, n);
}
extern "C" __global__ void __launch_bounds__(WARPSCAN_THREADS)
warpscan_f32(const float* __restrict__ in, float* __restrict__ out, float* __restrict__ block_sums, long long n) {
    warpscan_body<float>(in, out, block_sums, n);
}
extern "C" __global__ void scan_add_offsets_f64(double* __restrict__ out, const double* __restrict__ sums, long long n) {
    add_offsets_body<double>(out, sums, n);
}
extern "C" __global__ void scan_add_offsets_f32(float* __restrict__ out, const float* __restrict__ sums, long long n) {
    add_offsets_body<float>(out, sums, n);
}

// --------------------------------------------------------------------------- //
// Single-pass decoupled look-back scan
// --------------------------------------------------------------------------- //

#define LB_THREADS 256
#define LB_ITEMS 8
#define LB_TILE (LB_THREADS * LB_ITEMS)
#define LB_WARPS (LB_THREADS / 32)
#define LB_WARP_ELEMS (32 * LB_ITEMS)
#define LB_PAD(i) ((i) + ((i) >> 3))  // one pad slot per 8 doubles: blocked reads hit distinct banks
#define LB_STATUS_X 0                 // invalid: nothing published yet
#define LB_STATUS_A 1                 // aggregate (this tile alone) available
#define LB_STATUS_P 2                 // inclusive prefix (all tiles up to this one) available

__device__ __forceinline__ int lb_load_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void lb_store_release(int* p, int v) {
    asm volatile("st.release.gpu.b32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

template <typename T>
__device__ __forceinline__ T lb_warp_sum(T v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(SCAN_FULL_MASK, v, off);
    return v;
}

// Exclusive prefix of `tile` from its predecessors' published statuses (called by all of warp 0).
template <typename T>
__device__ __forceinline__ T lb_look_back(const int* status, const T* aggregate, const T* inclusive, long long tile, int lane) {
    T prefix = T(0);
    long long pred = tile - 1 - lane;  // lane 0 looks at the nearest predecessor
    while (true) {
        int st = pred >= 0 ? lb_load_acquire(status + pred) : LB_STATUS_P;  // before tile 0: an empty prefix
        while (__any_sync(SCAN_FULL_MASK, st == LB_STATUS_X)) {
            if (st == LB_STATUS_X) {
                __nanosleep(64);
                st = lb_load_acquire(status + pred);
            }
        }
        T v = T(0);
        if (pred >= 0) v = (st == LB_STATUS_P) ? __ldcg(inclusive + pred) : __ldcg(aggregate + pred);
        const unsigned done = __ballot_sync(SCAN_FULL_MASK, st == LB_STATUS_P);
        if (done) {
            const int first = __ffs(done) - 1;  // nearest tile with an inclusive prefix
            prefix += lb_warp_sum(lane <= first ? v : T(0));
            return prefix;
        }
        prefix += lb_warp_sum(v);  // 32 aggregates, none of them conclusive: step back a window
        pred -= 32;
    }
}

template <typename T>
__device__ __forceinline__ void lookback_body(const T* __restrict__ in, T* __restrict__ out, int* __restrict__ status,
                                              T* __restrict__ aggregate, T* __restrict__ inclusive,
                                              unsigned int* __restrict__ tile_counter, long long n) {
    __shared__ T s[LB_WARPS][LB_PAD(LB_WARP_ELEMS)];
    __shared__ T warp_tot[LB_WARPS];
    __shared__ T block_prefix;
    __shared__ unsigned int tile_id;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) tile_id = atomicAdd(tile_counter, 1u);
    __syncthreads();
    const long long tile = tile_id;
    const long long wbase = tile * LB_TILE + (long long)warp * LB_WARP_ELEMS;
    T* ws = s[warp];
#pragma unroll
    for (int k = 0; k < LB_ITEMS; ++k) {  // striped, coalesced load into this warp's slice
        const int i = k * 32 + lane;
        ws[LB_PAD(i)] = (wbase + i < n) ? in[wbase + i] : T(0);
    }
    __syncwarp();
    T v[LB_ITEMS];
#pragma unroll
    for (int j = 0; j < LB_ITEMS; ++j) v[j] = ws[LB_PAD(lane * LB_ITEMS + j)];  // blocked: 8 consecutive per thread
#pragma unroll
    for (int j = 1; j < LB_ITEMS; ++j) v[j] += v[j - 1];
    T x = v[LB_ITEMS - 1];
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const T y = __shfl_up_sync(SCAN_FULL_MASK, x, off);
        if (lane >= off) x += y;
    }
    T lane_excl = __shfl_up_sync(SCAN_FULL_MASK, x, 1);  // shuffled, not x - total, so nothing cancels
    if (lane == 0) lane_excl = T(0);
    if (lane == 31) warp_tot[warp] = x;
    __syncthreads();
    if (warp == 0) {
        T w = lane < LB_WARPS ? warp_tot[lane] : T(0);
#pragma unroll
        for (int off = 1; off < LB_WARPS; off <<= 1) {
            const T y = __shfl_up_sync(SCAN_FULL_MASK, w, off);
            if (lane >= off) w += y;
        }
        const T tile_total = __shfl_sync(SCAN_FULL_MASK, w, LB_WARPS - 1);
        const T warp_excl = __shfl_up_sync(SCAN_FULL_MASK, w, 1);
        if (lane < LB_WARPS) warp_tot[lane] = lane == 0 ? T(0) : warp_excl;  // now exclusive per warp
        T excl = T(0);
        if (tile == 0) {
            if (lane == 0) {
                inclusive[0] = tile_total;
                lb_store_release(status, LB_STATUS_P);
            }
        } else {
            if (lane == 0) {
                aggregate[tile] = tile_total;
                lb_store_release(status + tile, LB_STATUS_A);
            }
            excl = lb_look_back(status, aggregate, inclusive, tile, lane);
            if (lane == 0) {
                inclusive[tile] = excl + tile_total;
                lb_store_release(status + tile, LB_STATUS_P);
            }
        }
        if (lane == 0) block_prefix = excl;
    }
    __syncthreads();
    const T prefix = block_prefix + warp_tot[warp] + lane_excl;
#pragma unroll
    for (int j = 0; j < LB_ITEMS; ++j) ws[LB_PAD(lane * LB_ITEMS + j)] = prefix + v[j];
    __syncwarp();
#pragma unroll
    for (int k = 0; k < LB_ITEMS; ++k) {  // striped, coalesced store
        const int i = k * 32 + lane;
        if (wbase + i < n) out[wbase + i] = ws[LB_PAD(i)];
    }
}

extern "C" __global__ void lookback_init(int* __restrict__ status, unsigned int* __restrict__ tile_counter, long long num_tiles) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < num_tiles; i += stride) status[i] = LB_STATUS_X;
    if (blockIdx.x == 0 && threadIdx.x == 0) *tile_counter = 0u;
}

extern "C" __global__ void __launch_bounds__(LB_THREADS)
lookback_f64(const double* __restrict__ in, double* __restrict__ out, int* __restrict__ status, double* __restrict__ aggregate,
             double* __restrict__ inclusive, unsigned int* __restrict__ tile_counter, long long n) {
    lookback_body<double>(in, out, status, aggregate, inclusive, tile_counter, n);
}

extern "C" __global__ void __launch_bounds__(LB_THREADS)
lookback_f32(const float* __restrict__ in, float* __restrict__ out, int* __restrict__ status, float* __restrict__ aggregate,
             float* __restrict__ inclusive, unsigned int* __restrict__ tile_counter, long long n) {
    lookback_body<float>(in, out, status, aggregate, inclusive, tile_counter, n);
}
