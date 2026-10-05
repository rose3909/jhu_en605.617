/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Adapted from JHU-EP-Intro2GPU/EN605.617/module5; see SOURCES.md.
#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using u32 = unsigned int;
static_assert(sizeof(u32) == 4, "This example needs 32-bit unsigned int");
__constant__ u32 const_data_gpu[3];

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) throw std::runtime_error( \
        std::string(#call) + ": " + cudaGetErrorString(err)); \
} while (0)

// register.cu's per-thread d_tmp, extended with runtime coefficients.
__host__ __device__ u32 transform(u32 value, u32 scale, u32 offset, u32 mask) {
    u32 d_tmp = value;
    d_tmp = d_tmp * scale + offset;  // unsigned arithmetic wraps modulo 2^32
    return d_tmp ^ mask;
}

// Generalizes shared_memory2.cu's dynamicReverse to multiple blocks and
// a partial final block. All three specializations perform the SAME operation.
template<bool UseShared, bool UseConstant>
__global__ void memory_kernel(const u32* input, u32* output,
                              const u32* global_coeff, int n) {
    extern __shared__ u32 tile[];
    const int t = threadIdx.x;
    const int base = blockIdx.x * blockDim.x;
    const int remaining = n - base;
    const int valid = remaining < static_cast<int>(blockDim.x)
                    ? remaining : static_cast<int>(blockDim.x);
    if (UseShared) {
        if (t < valid) tile[t] = input[base + t];
        // EVERY thread reaches this barrier, even in a partial block.
        __syncthreads();
    }
    if (t < valid) {
        // The private scalar is a register candidate; ptxas reports allocation.
        const u32 value = UseShared ? tile[valid - t - 1]
                                   : input[base + valid - t - 1];
        const u32 scale = UseConstant ? const_data_gpu[0] : global_coeff[0];
        const u32 offset = UseConstant ? const_data_gpu[1] : global_coeff[1];
        const u32 mask = UseConstant ? const_data_gpu[2] : global_coeff[2];
        u32 d_tmp = transform(value, scale, offset, mask);
        output[base + t] = d_tmp;
    }
}

static int positive_int(const char* text, const char* name) {
    errno = 0;
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (errno || end == text || *end || value < 1 || value > INT_MAX)
        throw std::runtime_error(std::string(name) + " must be a positive integer <= INT_MAX");
    return static_cast<int>(value);
}

template<bool Shared, bool Constant>
static void launch(int blocks, int threads, const u32* input, u32* output,
                   const u32* coeff, int n) {
    const size_t shared_bytes = Shared ? threads * sizeof(u32) : 0;
    memory_kernel<Shared, Constant><<<blocks, threads, shared_bytes>>>(input, output, coeff, n);
    CUDA_CHECK(cudaGetLastError());
}

template<bool Shared, bool Constant>
static float benchmark(const char* name, int blocks, int threads, int repeats,
                       const u32* input, u32* output, const u32* coeff, int n,
                       const std::vector<u32>& expected, std::ofstream& csv) {
    cudaFuncAttributes attrs{};
    CUDA_CHECK(cudaFuncGetAttributes(&attrs, memory_kernel<Shared, Constant>));
    for (int i = 0; i < 5; ++i) launch<Shared, Constant>(blocks, threads, input, output, coeff, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r)
        launch<Shared, Constant>(blocks, threads, input, output, coeff, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::vector<u32> result(n); // HOST array for returned device results
    CUDA_CHECK(cudaMemcpy(result.data(), output, n * sizeof(u32), cudaMemcpyDeviceToHost));
    size_t mismatches = 0;
    for (int i = 0; i < n; ++i) if (result[i] != expected[i]) ++mismatches;
    const float average = elapsed / repeats;
    std::printf("%-18s average=%10.6f ms  registers/thread=%d local/thread=%zu bytes  %s (%zu mismatches)\n",
                name, average, attrs.numRegs, attrs.localSizeBytes,
                mismatches ? "FAIL" : "PASS", mismatches);
    csv << name << ',' << n << ',' << threads << ',' << blocks << ','
        << repeats << ',' << average << ',' << attrs.numRegs << ','
        << attrs.localSizeBytes << ',' << mismatches << '\n';
    if (!csv) throw std::runtime_error("Could not write results CSV");
    if (mismatches) throw std::runtime_error("GPU result does not match CPU reference");
    return average;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::puts("Usage: ./assignment.exe [totalThreads=1048576] [blockSize=256] [repeats=100] [results.csv]");
        return 0;
    }
    try {
        if (argc > 5) throw std::runtime_error("Too many arguments; use --help");
        // Retains the Module 5 assignment.c argument order. One element per
        // requested thread; ceil division adds inactive padding threads.
        const int n = argc > 1 ? positive_int(argv[1], "totalThreads") : (1 << 20);
        const int threads = argc > 2 ? positive_int(argv[2], "blockSize") : 256;
        const int repeats = argc > 3 ? positive_int(argv[3], "repeats") : 100;
        const std::string csv_path = argc > 4 ? argv[4] : "results.csv";
        int devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (!devices) throw std::runtime_error("No CUDA GPU. In Colab select a GPU runtime.");
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0])
            throw std::runtime_error("blockSize exceeds this GPU's thread limit");
        const int blocks = 1 + (n - 1) / threads;
        if (blocks > prop.maxGridSize[0]) throw std::runtime_error("Grid exceeds GPU limit");
        if (threads * sizeof(u32) > prop.sharedMemPerBlock)
            throw std::runtime_error("Shared-memory request exceeds GPU limit");
        std::printf("GPU: %s\nElements/requested threads: %d | Threads/block: %d | Blocks: %d | Launched threads: %lld\n",
                    prop.name, n, threads, blocks, static_cast<long long>(blocks) * threads);
        std::printf("Repeats: %d | Shared tile: %zu bytes/block | Constant coefficients: 12 bytes\n",
                    repeats, threads * sizeof(u32));
        const u32 coeff[3] = {2u, 17u, 0x55555555u}; // HOST coefficients
        std::vector<u32> input(n), expected(n);      // HOST input/reference arrays
        for (int i = 0; i < n; ++i) input[i] = static_cast<u32>(i) * 2654435761u + 12345u;
        for (int base = 0; base < n; ) {
            const int valid = std::min(threads, n - base);
            for (int t = 0; t < valid; ++t)
                expected[base + t] = transform(input[base + valid - t - 1], coeff[0], coeff[1], coeff[2]);
            base += valid;
        }
        u32 *d_input = nullptr, *d_output = nullptr, *d_coeff = nullptr;
        const size_t bytes = n * sizeof(u32);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_coeff), sizeof(coeff)));
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_coeff, coeff, sizeof(coeff), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(const_data_gpu, coeff, sizeof(coeff)));
        std::ofstream csv(csv_path);
        if (!csv) throw std::runtime_error("Cannot open CSV output");
        csv << "kernel,elements,threads_per_block,blocks,repeats,average_ms,registers_per_thread,local_bytes_per_thread,mismatches\n";
        float g = benchmark<false, false>("global", blocks, threads, repeats, d_input, d_output, d_coeff, n, expected, csv);
        float c = benchmark<false, true>("constant", blocks, threads, repeats, d_input, d_output, d_coeff, n, expected, csv);
        float s = benchmark<true, true>("shared_constant", blocks, threads, repeats, d_input, d_output, d_coeff, n, expected, csv);
        std::printf("Global/constant time ratio: %.3f | Constant/shared time ratio: %.3f\n", g / c, c / s);
        std::printf("All variants match CPU reference. CSV: %s\n", csv_path.c_str());
        CUDA_CHECK(cudaFree(d_coeff));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_input));
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Error: %s\n", error.what());
        // Releases partially allocated device resources on an error path.
        cudaDeviceReset();
        return EXIT_FAILURE;
    }
}
