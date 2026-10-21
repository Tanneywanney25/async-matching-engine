// cuda/scan_cub.cu -- cub::DeviceScan::InclusiveSum behind a C ABI, for ctypes.
//
// Compiled in the GPU job (CUB headers ship with the CUDA toolkit, NVRTC cannot use them):
//   nvcc -O3 -arch=sm_75 -Xcompiler -fPIC -shared -o build/libscan_cub.so cuda/scan_cub.cu
// Pointers are device pointers owned by CuPy (array.data.ptr); `stream` is a cudaStream_t
// (0 = the legacy default stream CuPy uses unless told otherwise). The temporary storage is
// allocated once per size and reused, so after one warmup call nothing is allocated in a
// timed region.

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

namespace {
void* g_temp = nullptr;
size_t g_temp_bytes = 0;

template <typename T>
int inclusive_sum(const T* in, T* out, long long n, void* stream) {
    size_t need = 0;
    cudaError_t err = cub::DeviceScan::InclusiveSum(nullptr, need, in, out, (int)n, (cudaStream_t)stream);
    if (err != cudaSuccess) return (int)err;
    if (need > g_temp_bytes) {
        if (g_temp) cudaFree(g_temp);
        err = cudaMalloc(&g_temp, need);
        if (err != cudaSuccess) return (int)err;
        g_temp_bytes = need;
    }
    err = cub::DeviceScan::InclusiveSum(g_temp, g_temp_bytes, in, out, (int)n, (cudaStream_t)stream);
    return (int)err;
}
}  // namespace

extern "C" int cub_inclusive_sum_f64(const double* in, double* out, long long n, void* stream) {
    return inclusive_sum<double>(in, out, n, stream);
}

extern "C" int cub_inclusive_sum_f32(const float* in, float* out, long long n, void* stream) {
    return inclusive_sum<float>(in, out, n, stream);
}

extern "C" int cub_sync() { return (int)cudaDeviceSynchronize(); }
