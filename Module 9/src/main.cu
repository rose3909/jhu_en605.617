#include "cpu_reference.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

void check_cuda(cudaError_t result, const char* file, int line) {
  if (result == cudaSuccess) return;
  throw std::runtime_error(std::string(file) + ":" +
      std::to_string(line) + ": " + cudaGetErrorString(result));
}

void check_cublas(cublasStatus_t result, const char* file, int line) {
  if (result == CUBLAS_STATUS_SUCCESS) return;
  throw std::runtime_error(std::string(file) + ":" +
      std::to_string(line) + ": cuBLAS error " +
      std::to_string(static_cast<int>(result)));
}

#define CUDA_OK(call) check_cuda((call), __FILE__, __LINE__)
#define BLAS_OK(call) check_cublas((call), __FILE__, __LINE__)

struct GpuState {
  float* embeddings = nullptr;
  float* similarities = nullptr;
  cublasHandle_t handle = nullptr;
  ~GpuState() {
    if (handle) cublasDestroy(handle);
    if (similarities) cudaFree(similarities);
    if (embeddings) cudaFree(embeddings);
  }
};

struct EligiblePair {
  const float* scores;
  int records;
  float threshold;
  __host__ __device__ bool operator()(int id) const {
    int a = id / records;
    int b = id % records;
    return a < b && scores[id] >= threshold;
  }
};

struct GatherScore {
  const float* scores;
  __host__ __device__ float operator()(int id) const {
    return scores[id];
  }
};

void compute_similarity(GpuState& gpu, int records, int dimensions) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // Each record is one column of a dimension x records matrix.
  // Its transpose times itself gives pairwise cosine similarities.
  BLAS_OK(cublasSgemm(gpu.handle, CUBLAS_OP_T, CUBLAS_OP_N,
      records, records, dimensions, &alpha, gpu.embeddings,
      dimensions, gpu.embeddings, dimensions, &beta,
      gpu.similarities, records));
}

size_t select_and_rank(GpuState& gpu, int records, float threshold,
                       thrust::device_vector<int>& ids,
                       thrust::device_vector<float>& scores) {
  int cells = records * records;
  auto first = thrust::make_counting_iterator(0);
  auto last = thrust::copy_if(thrust::device, first, first + cells,
      ids.begin(), EligiblePair{gpu.similarities, records, threshold});
  size_t count = static_cast<size_t>(last - ids.begin());
  thrust::transform(thrust::device, ids.begin(), ids.begin() + count,
      scores.begin(), GatherScore{gpu.similarities});
  thrust::sort_by_key(thrust::device, scores.begin(),
      scores.begin() + count, ids.begin(), thrust::greater<float>());
  return count;
}

MatchResult gpu_resolve(const DataSet& data, float threshold) {
  GpuState gpu;
  int n = data.records;
  size_t feature_bytes = data.embeddings.size() * sizeof(float);
  size_t matrix_bytes = static_cast<size_t>(n) * n * sizeof(float);
  size_t capacity = static_cast<size_t>(n) * (n - 1) / 2;
  CUDA_OK(cudaMalloc(&gpu.embeddings, feature_bytes));
  CUDA_OK(cudaMalloc(&gpu.similarities, matrix_bytes));
  BLAS_OK(cublasCreate(&gpu.handle));
  thrust::device_vector<int> ids(capacity);
  thrust::device_vector<float> scores(capacity);
  CUDA_OK(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();
  CUDA_OK(cudaMemcpy(gpu.embeddings, data.embeddings.data(),
      feature_bytes, cudaMemcpyHostToDevice));
  compute_similarity(gpu, n, data.dimensions);
  size_t count = select_and_rank(gpu, n, threshold, ids, scores);
  std::vector<int> host_ids(count);
  std::vector<float> host_scores(count);
  thrust::copy(ids.begin(), ids.begin() + count, host_ids.begin());
  thrust::copy(scores.begin(), scores.begin() + count,
               host_scores.begin());
  CUDA_OK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  MatchResult output;
  output.milliseconds =
      std::chrono::duration<double, std::milli>(end - start).count();
  for (size_t i = 0; i < count; ++i) {
    output.matches.push_back({host_ids[i], host_scores[i]});
  }
  return output;
}

int verify_matches(const DataSet& data, const MatchResult& gpu,
                   const MatchResult& cpu, float threshold) {
  std::unordered_map<int, float> reference;
  for (const Match& m : cpu.matches) reference[m.id] = m.score;
  int problems = 0;
  for (const Match& m : gpu.matches) {
    int a = m.id / data.records;
    int b = m.id % data.records;
    float expected = cpu_similarity(data, a, b);
    if (a >= b || std::abs(expected - m.score) > 0.0002f) {
      ++problems;
    }
    if (!reference.count(m.id) &&
        std::abs(expected - threshold) > 0.0002f) ++problems;
    reference.erase(m.id);
  }
  for (const auto& item : reference) {
    if (std::abs(item.second - threshold) > 0.0002f) ++problems;
  }
  for (size_t i = 1; i < gpu.matches.size(); ++i) {
    if (gpu.matches[i].score > gpu.matches[i - 1].score + 0.00001f) {
      ++problems;
    }
  }
  return problems;
}

void print_metrics(const DataSet& data, const MatchResult& gpu,
                   const MatchResult& cpu, int errors) {
  int tp = true_positive_count(data, gpu.matches);
  int actual = actual_match_count(data);
  double precision = gpu.matches.empty() ? 0.0 :
      static_cast<double>(tp) / gpu.matches.size();
  double recall = static_cast<double>(tp) / actual;
  double speedup = cpu.milliseconds / gpu.milliseconds;
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "CPU_ms=" << cpu.milliseconds
            << " GPU_end_to_end_ms=" << gpu.milliseconds
            << " speedup=" << speedup << "x\n";
  std::cout << "candidates=" << gpu.matches.size()
            << " true_positives=" << tp
            << " expected_duplicate_pairs=" << actual << "\n";
  std::cout << "precision=" << precision << " recall=" << recall
            << " validation=" << (errors ? "FAIL" : "PASS") << "\n";
  for (size_t i = 0; i < std::min<size_t>(3, gpu.matches.size()); ++i) {
    const Match& match = gpu.matches[i];
    int a = match.id / data.records;
    int b = match.id % data.records;
    std::cout << "  pair=" << a << "," << b
              << " similarity=" << match.score
              << " same_entity=" << (data.entity_ids[a] ==
                                     data.entity_ids[b]) << "\n";
  }
}

void print_gpu_name() {
  int device = 0;
  cudaDeviceProp props{};
  CUDA_OK(cudaGetDevice(&device));
  CUDA_OK(cudaGetDeviceProperties(&props, device));
  std::cout << "GPU=" << props.name << "\n";
}

int run_case(int records, int dimensions, int copies, float noise,
             float threshold, unsigned seed) {
  if (records < 2 || records > 4096 || copies < 2 ||
      threshold < -1.0f || threshold > 1.0f ||
      records % copies != 0) {
    throw std::invalid_argument("Unsupported records or threshold");
  }
  DataSet data = generate_data(records, dimensions, copies, noise,
                               seed);
  std::cout << "records=" << records << " dimensions=" << dimensions
            << " copies=" << copies << " noise=" << noise
            << " threshold=" << threshold << " seed=" << seed << "\n";
  auto cpu = cpu_resolve(data, threshold);
  auto gpu = gpu_resolve(data, threshold);
  int errors = verify_matches(data, gpu, cpu, threshold);
  print_metrics(data, gpu, cpu, errors);
  return errors ? 1 : 0;
}

int main(int argc, char** argv) {
  try {
    if (argc != 1 && argc != 6 && argc != 7) {
      std::cerr << "Usage: entity_match [records dims copies noise "
                   "threshold [seed]]\n";
      return 2;
    }
    int records = argc > 1 ? std::stoi(argv[1]) : 512;
    int dimensions = argc > 1 ? std::stoi(argv[2]) : 64;
    int copies = argc > 1 ? std::stoi(argv[3]) : 4;
    float noise = argc > 1 ? std::stof(argv[4]) : 0.10f;
    float threshold = argc > 1 ? std::stof(argv[5]) : 0.45f;
    unsigned seed = argc == 7 ? std::stoul(argv[6]) : 42;
    print_gpu_name();
    return run_case(records, dimensions, copies, noise,
                    threshold, seed);
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << "\n";
    return 1;
  }
}
