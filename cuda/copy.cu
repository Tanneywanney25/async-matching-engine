// cuda/copy.cu -- device-to-device copy kernel used to measure the achievable
// memory bandwidth ceiling in the same job as the feature kernels.
// 128-bit loads and stores in a grid-stride loop; n counts int4 elements.
// Device code only (NVRTC through cupy.RawModule, -arch=sm_75).

extern "C" __global__ void copy_int4(const int4* __restrict__ src, int4* __restrict__ dst, long long n) {
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += step) {
        dst[i] = src[i];
    }
}
