#ifndef ENTITY_DATA_HPP
#define ENTITY_DATA_HPP

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

struct DataSet {
  int records;
  int dimensions;
  int copies;
  std::vector<float> embeddings;
  std::vector<int> entity_ids;
};

inline void normalize(float* row, int dimensions) {
  double norm = 0.0;
  for (int k = 0; k < dimensions; ++k) {
    norm += static_cast<double>(row[k]) * row[k];
  }
  norm = std::sqrt(norm);
  for (int k = 0; k < dimensions; ++k) {
    row[k] = static_cast<float>(row[k] / norm);
  }
}

inline std::vector<float> make_prototypes(int entities, int dimensions,
                                          std::mt19937& rng) {
  std::normal_distribution<float> gaussian(0.0f, 1.0f);
  std::vector<float> data(entities * dimensions);
  for (int e = 0; e < entities; ++e) {
    float* current = data.data() + e * dimensions;
    for (int k = 0; k < dimensions; ++k) {
      float unique = gaussian(rng);
      bool lookalike = (e % 8 == 1);
      float shared = lookalike ? data[(e - 1) * dimensions + k]
                               : 0.0f;
      current[k] = lookalike ? 0.94f * shared + 0.34f * unique
                            : unique;
    }
    normalize(current, dimensions);
  }
  return data;
}

inline void shuffle_records(DataSet& data, std::mt19937& rng) {
  std::vector<int> order(data.records);
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<float> shuffled(data.embeddings.size());
  std::vector<int> labels(data.records);
  for (int i = 0; i < data.records; ++i) {
    int original = order[i];
    labels[i] = data.entity_ids[original];
    for (int k = 0; k < data.dimensions; ++k) {
      shuffled[i * data.dimensions + k] =
          data.embeddings[original * data.dimensions + k];
    }
  }
  data.embeddings.swap(shuffled);
  data.entity_ids.swap(labels);
}

inline DataSet generate_data(int records, int dimensions, int copies,
                             float noise, unsigned seed) {
  if (records < 2 || dimensions < 2 || copies < 2 ||
      records % copies != 0 || noise < 0.0f) {
    throw std::invalid_argument("Invalid dataset parameters");
  }
  std::mt19937 rng(seed);
  std::normal_distribution<float> gaussian(0.0f, 1.0f);
  int entities = records / copies;
  auto bases = make_prototypes(entities, dimensions, rng);
  DataSet data{records, dimensions, copies,
               std::vector<float>(records * dimensions),
               std::vector<int>(records)};
  for (int r = 0; r < records; ++r) {
    int entity = r / copies;
    data.entity_ids[r] = entity;
    float* row = data.embeddings.data() + r * dimensions;
    for (int k = 0; k < dimensions; ++k) {
      row[k] = bases[entity * dimensions + k] + noise * gaussian(rng);
    }
    normalize(row, dimensions);
  }
  shuffle_records(data, rng);
  return data;
}

#endif
