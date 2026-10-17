// EN605.617 Module 6: CUDA streams and events
// Adapted from course module6/stream_example.cu: two input arrays,
// array-addition kernel concept, pinned host allocations, asynchronous H2D/D2H,
// stream creation/synchronization and CUDA event timing. Extended to use a
// non-trivial transform, two-stream comparison, varied launches and validation.
// Compare the SAME numerical transform with a single stream and two streams.
// CUDA events time kernels within each stream; a CPU steady clock times the
// complete H2D -> kernel -> D2H pipeline, including final synchronization.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) do {                                             \
    cudaError_t status = (call);                                          \
    if (status != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(status));                              \
        std::exit(EXIT_FAILURE);                                         \
    }                                                                     \
} while (0)

constexpr int kIterations = 24;
constexpr int kChunks = 2;
constexpr int kTrials = 2;     // At least two runs of each distinct variant.
constexpr int kValidationSamples = 128;

__host__ __device__ inline float transform_value(float x) {
    for (int i = 0; i < kIterations; ++i) {
        x = fmaf(x, 0.99991f, 0.000123f);
        x = sqrtf(x * x + 0.01f);
        x = x / (1.0f + 0.0001f * x);
    }
    return x;
}

// Starting with the instructor's arrayAddition pattern, sum two arrays and
// transform each sum so the kernel performs non-trivial numerical work.
__global__ void add_transform_kernel(const float* input_a, const float* input_b,
                                     float* output, size_t count) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = transform_value(input_a[i] + input_b[i]);
}

struct Buffers {
    float* host_a = nullptr;
    float* host_b = nullptr;
    float* host_out = nullptr;
    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_out = nullptr;
    size_t count = 0;

    explicit Buffers(size_t n) : count(n) {
        size_t bytes = n * sizeof(float);
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_a), bytes));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_b), bytes));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_out), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_a), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_b), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_out), bytes));
        for (size_t i = 0; i < n; ++i) {
            host_a[i] = 0.1f + static_cast<float>(i % 1024) * 0.0005f;
            host_b[i] = 0.2f + static_cast<float>(i % 256) * 0.0003f;
        }
    }
    ~Buffers() {
        if (device_out) cudaFree(device_out);
        if (device_b) cudaFree(device_b);
        if (device_a) cudaFree(device_a);
        if (host_out) cudaFreeHost(host_out);
        if (host_b) cudaFreeHost(host_b);
        if (host_a) cudaFreeHost(host_a);
    }
    Buffers(const Buffers&) = delete;
    Buffers& operator=(const Buffers&) = delete;
};

struct TrialResult {
    double pipeline_ms = 0;
    float kernel_ms[kChunks] = {0, 0};
};

TrialResult run_trial(Buffers& b, int threads, bool parallel) {
    cudaStream_t streams[kChunks] = {nullptr, nullptr};
    cudaEvent_t start[kChunks] = {nullptr, nullptr};
    cudaEvent_t stop[kChunks] = {nullptr, nullptr};
    for (int i = 0; i < kChunks; ++i) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&start[i]));
        CUDA_CHECK(cudaEventCreate(&stop[i]));
    }

    const size_t chunk = b.count / kChunks;
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < kChunks; ++i) {
        // Sequential mode queues both chunks on stream 0; parallel uses 0 and 1.
        cudaStream_t stream = streams[parallel ? i : 0];
        const size_t offset = static_cast<size_t>(i) * chunk;
        const size_t count = (i == kChunks - 1) ? b.count - offset : chunk;
        const size_t bytes = count * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(b.device_a + offset, b.host_a + offset,
                                   bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(b.device_b + offset, b.host_b + offset,
                                   bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(start[i], stream));
        int blocks = static_cast<int>((count + threads - 1) / threads);
        add_transform_kernel<<<blocks, threads, 0, stream>>>(b.device_a + offset,
                                                      b.device_b + offset,
                                                      b.device_out + offset, count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop[i], stream));
        CUDA_CHECK(cudaMemcpyAsync(b.host_out + offset, b.device_out + offset,
                                   bytes, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    if (parallel) CUDA_CHECK(cudaStreamSynchronize(streams[1]));
    auto end = std::chrono::steady_clock::now();

    TrialResult result;
    result.pipeline_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    for (int i = 0; i < kChunks; ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&result.kernel_ms[i], start[i], stop[i]));
        CUDA_CHECK(cudaEventDestroy(start[i]));
        CUDA_CHECK(cudaEventDestroy(stop[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    return result;
}

bool validate(const Buffers& b) {
    const size_t step = std::max<size_t>(1, b.count / kValidationSamples);
    for (size_t i = 0; i < b.count; i += step) {
        const float expected = transform_value(b.host_a[i] + b.host_b[i]);
        const float actual = b.host_out[i];
        const float tolerance = 1e-3f + 1e-4f * fabsf(expected);
        if (!std::isfinite(actual) || fabsf(actual - expected) > tolerance) {
            fprintf(stderr, "FAIL at %zu: GPU=%f CPU=%f\n", i, actual, expected);
            return false;
        }
    }
    return true;
}

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA GPU found. Select a GPU runtime in Colab.\n");
        return EXIT_FAILURE;
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaSetDevice(0));
    printf("GPU: %s | async engines: %d | concurrent kernels: %s\n",
           prop.name, prop.asyncEngineCount, prop.concurrentKernels ? "yes" : "no");
    printf("Pipeline = async H2D of two arrays + addition/%d-step transform + async D2H\n", kIterations);
    printf("Full pipeline uses CPU steady-clock timing; individual kernels use CUDA events.\n");
    printf("Mode,Elements,Threads,BlocksPerChunk,Trial,PipelineMs,Kernel0Ms,Kernel1Ms,Validation\n");

    const std::vector<size_t> sizes = {1u << 16, 1u << 20, 1u << 22};
    const std::vector<int> thread_options = {128, 256, 512};
    for (size_t size : sizes) {
        Buffers b(size);  // pinned host memory enables genuine asynchronous transfers
        for (int threads : thread_options) {
            if (threads > prop.maxThreadsPerBlock) continue;
            size_t max_chunk = size - size / kChunks;
            int blocks = static_cast<int>((max_chunk + threads - 1) / threads);
            if (blocks > prop.maxGridSize[0]) continue;
            // Warm up kernels and runtime, excluded from timed trials.
            run_trial(b, threads, false);
            run_trial(b, threads, true);
            for (int mode = 0; mode < 2; ++mode) {
                bool concurrent = mode == 1;
                for (int trial = 1; trial <= kTrials; ++trial) {
                    TrialResult r = run_trial(b, threads, concurrent);
                    bool pass = validate(b);
                    printf("%s,%zu,%d,%d,%d,%.4f,%.4f,%.4f,%s\n",
                           concurrent ? "TwoStreams" : "OneStream", size, threads,
                           blocks, trial, r.pipeline_ms, r.kernel_ms[0],
                           r.kernel_ms[1], pass ? "PASS" : "FAIL");
                    if (!pass) return EXIT_FAILURE;
                }
            }
        }
    }
    printf("All configurations validated. Two streams may or may not be faster on this GPU.\n");
    return EXIT_SUCCESS;
}
