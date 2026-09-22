//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error__ = (call);                                           \
        if (error__ != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(error__)          \
                      << " (" << __FILE__ << ':' << __LINE__ << ")\n";              \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                      \
    } while (0)

constexpr std::size_t DATA_SIZE = 8ULL * 1024ULL * 1024ULL;
constexpr int REPEATS = 10;

__host__ __device__ inline float branchless_op(float x) {
    float y = x * 1.0001f + 0.25f;
    y = y * y * 0.00001f + y;
    y = y * 0.9999f - 0.125f;
    return y;
}

// The two paths contain similar amounts of arithmetic. Alternating signs in
// the input make adjacent GPU lanes choose opposite paths, causing divergence.
__host__ __device__ inline float branching_op(float x) {
    if (x >= 0.0f) {
        float y = x * 1.0001f + 0.25f;
        y = y * y * 0.00001f + y;
        y = y * 0.9999f - 0.125f;
        return y;
    } else {
        float y = x * 0.9999f - 0.25f;
        y = y * y * -0.00001f + y;
        y = y * 1.0001f + 0.125f;
        return y;
    }
}

__global__ void branchless_kernel(const float* input, float* output,
                                  std::size_t n) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t i = index; i < n; i += stride) {
        output[i] = branchless_op(input[i]);
    }
}

__global__ void branching_kernel(const float* input, float* output,
                                 std::size_t n) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t i = index; i < n; i += stride) {
        // branching_op contains the if/else branch executed by each GPU thread.
        output[i] = branching_op(input[i]);
    }
}

template <typename Operation>
double time_cpu(const std::vector<float>& input, std::vector<float>& output,
                Operation operation) {
    double total_ms = 0.0;
    for (int repeat = 0; repeat < REPEATS; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = operation(input[i]);
        }
        const auto stop = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(stop - start).count();
    }
    return total_ms / REPEATS;
}

double time_gpu(bool use_branching_kernel, int blocks, int threads_per_block,
                const float* device_input, float* device_output, std::size_t n) {
    // Warm-up removes one-time CUDA context/JIT costs from the measurement.
    if (use_branching_kernel) {
        branching_kernel<<<blocks, threads_per_block>>>(device_input, device_output, n);
    } else {
        branchless_kernel<<<blocks, threads_per_block>>>(device_input, device_output, n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int repeat = 0; repeat < REPEATS; ++repeat) {
        if (use_branching_kernel) {
            branching_kernel<<<blocks, threads_per_block>>>(device_input, device_output, n);
        } else {
            branchless_kernel<<<blocks, threads_per_block>>>(device_input, device_output, n);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(total_ms) / REPEATS;
}

bool results_match(const std::vector<float>& expected,
                   const std::vector<float>& actual) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float tolerance = 1e-4f * std::max(1.0f, std::fabs(expected[i]));
        if (std::fabs(expected[i] - actual[i]) > tolerance) {
            std::cerr << "Mismatch at " << i << ": CPU=" << expected[i]
                      << ", GPU=" << actual[i] << '\n';
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    // Professor-provided command-line argument handling.
    int totalThreads = (1 << 20);
    int blockSize = 256;

    if (argc >= 2) {
        totalThreads = atoi(argv[1]);
    }
    if (argc >= 3) {
        blockSize = atoi(argv[2]);
    }

    // Additional validation before division or launching a CUDA kernel.
    if (totalThreads <= 0 || blockSize <= 0 || blockSize > 1024) {
        fprintf(stderr,
                "Error: totalThreads must be positive and blockSize must be "
                "between 1 and 1024.\n");
        return EXIT_FAILURE;
    }

    int numBlocks = totalThreads / blockSize;

    // Professor-provided validation and rounding behavior.
    if (totalThreads % blockSize != 0) {
        ++numBlocks;
        totalThreads = numBlocks * blockSize;

        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", totalThreads);
    }

    try {

        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        std::cout << "GPU: " << properties.name << '\n'
                  << "Data elements: " << DATA_SIZE << '\n'
                  << "Launch: " << numBlocks << " blocks x " << blockSize
                  << " threads (" << numBlocks * blockSize
                  << " actual threads)\n";

        std::vector<float> input(DATA_SIZE);
        std::vector<float> cpu_output(DATA_SIZE);
        std::vector<float> gpu_output(DATA_SIZE);
        std::mt19937 generator(42);
        std::uniform_real_distribution<float> magnitude(0.01f, 100.0f);
        for (std::size_t i = 0; i < DATA_SIZE; ++i) {
            // Alternation creates maximally mixed warps for the branching test.
            input[i] = (i % 2 == 0) ? magnitude(generator) : -magnitude(generator);
        }

        float *device_input = nullptr, *device_output = nullptr;
        const std::size_t bytes = DATA_SIZE * sizeof(float);
        CUDA_CHECK(cudaMalloc(&device_input, bytes));
        CUDA_CHECK(cudaMalloc(&device_output, bytes));
        CUDA_CHECK(cudaMemcpy(device_input, input.data(), bytes,
                              cudaMemcpyHostToDevice));

        const double cpu_branchless =
            time_cpu(input, cpu_output, [](float x) { return branchless_op(x); });
        const double gpu_branchless = time_gpu(false, numBlocks,
                                               blockSize, device_input,
                                               device_output, DATA_SIZE);
        CUDA_CHECK(cudaMemcpy(gpu_output.data(), device_output, bytes,
                              cudaMemcpyDeviceToHost));
        const bool branchless_valid = results_match(cpu_output, gpu_output);

        const double cpu_branching =
            time_cpu(input, cpu_output, [](float x) { return branching_op(x); });
        const double gpu_branching = time_gpu(true, numBlocks,
                                              blockSize, device_input,
                                              device_output, DATA_SIZE);
        CUDA_CHECK(cudaMemcpy(gpu_output.data(), device_output, bytes,
                              cudaMemcpyDeviceToHost));
        const bool branching_valid = results_match(cpu_output, gpu_output);

        CUDA_CHECK(cudaFree(device_input));
        CUDA_CHECK(cudaFree(device_output));

        std::ofstream csv("results.csv");
        csv << "method,branching,time_ms\n" << std::fixed << std::setprecision(6)
            << "CPU,No," << cpu_branchless << '\n'
            << "GPU,No," << gpu_branchless << '\n'
            << "CPU,Yes," << cpu_branching << '\n'
            << "GPU,Yes," << gpu_branching << '\n';

        std::cout << std::fixed << std::setprecision(3)
                  << "CPU branchless: " << cpu_branchless << " ms\n"
                  << "GPU branchless: " << gpu_branchless << " ms\n"
                  << "CPU branching:  " << cpu_branching << " ms\n"
                  << "GPU branching:  " << gpu_branching << " ms\n"
                  << "Validation: "
                  << ((branchless_valid && branching_valid) ? "PASS" : "FAIL")
                  << '\n';
        return (branchless_valid && branching_valid) ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
