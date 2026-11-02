#ifndef ENTITY_CPU_REFERENCE_HPP
#define ENTITY_CPU_REFERENCE_HPP

#include "data.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

struct Match {
  int id;
  float score;
};

struct MatchResult {
  std::vector<Match> matches;
  double milliseconds = 0.0;
};

inline float cpu_similarity(const DataSet& data, int a, int b) {
  float sum = 0.0f;
  int d = data.dimensions;
  for (int k = 0; k < d; ++k) {
    sum += data.embeddings[a * d + k] *
           data.embeddings[b * d + k];
  }
  return sum;
}

inline bool by_score(const Match& lhs, const Match& rhs) {
  if (lhs.score != rhs.score) return lhs.score > rhs.score;
  return lhs.id < rhs.id;
}

inline MatchResult cpu_resolve(const DataSet& data, float threshold) {
  auto start = std::chrono::steady_clock::now();
  MatchResult output;
  int n = data.records;
  for (int a = 0; a < n; ++a) {
    for (int b = a + 1; b < n; ++b) {
      float score = cpu_similarity(data, a, b);
      if (score >= threshold) {
        output.matches.push_back({a * n + b, score});
      }
    }
  }
  std::sort(output.matches.begin(), output.matches.end(), by_score);
  auto end = std::chrono::steady_clock::now();
  output.milliseconds =
      std::chrono::duration<double, std::milli>(end - start).count();
  return output;
}

inline int actual_match_count(const DataSet& data) {
  int entities = data.records / data.copies;
  return entities * data.copies * (data.copies - 1) / 2;
}

inline int true_positive_count(const DataSet& data,
                               const std::vector<Match>& matches) {
  int true_positives = 0;
  for (const Match& match : matches) {
    int a = match.id / data.records;
    int b = match.id % data.records;
    true_positives += (data.entity_ids[a] == data.entity_ids[b]);
  }
  return true_positives;
}

#endif
