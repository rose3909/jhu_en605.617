#include "../src/cpu_reference.hpp"
#include <cmath>
#include <iostream>

int main() {
  const auto data = generate_data(128, 32, 4, 0.08f, 42);
  const auto result = cpu_resolve(data, 0.8f);
  const int true_positives = true_positive_count(data, result.matches);
  const int actual = actual_match_count(data);
  if (data.records != 128 || actual != 192 ||
      true_positives > actual || result.matches.empty()) return 1;
  for (int r = 0; r < data.records; ++r) {
    if (std::abs(cpu_similarity(data, r, r) - 1.0f) > 1e-4f) {
      return 2;
    }
  }
  std::cout << "CPU dataset/reference smoke test PASS\n";
  std::cout << "matches=" << result.matches.size()
            << " true_positives=" << true_positives
            << " expected_pairs=" << actual << "\n";
  return 0;
}
