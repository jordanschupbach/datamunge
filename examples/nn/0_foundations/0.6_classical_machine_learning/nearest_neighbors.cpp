// CMD: just run-nn nearest_neighbors
// k-nearest-neighbor classification and regression with nn ops.
//
// Port of tflow's nearest_neighbors (model in common/neighbors.hpp, distances
// in common/kernels.hpp). Form pairwise squared distances, select the nearest
// indices, and aggregate uniform or inverse-distance votes/targets. Check every
// returned neighbor against an independent stable host sort on nonlinear ring
// data. Compare held-out classification/regression errors, and exercise
// equal-distance ties, equal-class votes, duplicate points and exact-match
// weighting. The Q-by-N-by-D distance temporary is appropriate for this small
// example; production-scale data needs chunking or a dedicated nearest-neighbor
// index. Data come from the same seeded streams as the TF version; indices and
// predicted classes are int64 (TF int32). No files are written.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common/checks.hpp"
#include "common/neighbors.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

void Edges() {
  std::vector<float> values{-1, 0, 1, 0, 0, 1, 0, -1};
  const Tensor x       = nn::from_vector(values, {4, 2});
  const Tensor query   = nn::from_vector(std::vector<float>{0, 0}, {1, 2});
  const Tensor classes = nn::from_vector(std::vector<std::int64_t>{1, 0, 1, 0});
  const Tensor targets = nn::from_vector(std::vector<float>{2, 6, 10, 20});
  ml::NearestNeighbors tied(x, query, 2, false);
  const Tensor         tie_class = tied.Classify(classes, 2), tie_regression = tied.Regress(targets);
  values[0] = 0;
  values[2] = 0;
  const Tensor         duplicated = nn::from_vector(values, {4, 2});
  ml::NearestNeighbors exact(duplicated, query, 3, true);
  const Tensor         exact_regression = exact.Regress(targets);
  const auto           indices          = tied.indices.to_vector<std::int64_t>();
  const auto           weights          = exact.weights.to_vector<float>();
  check::Require(indices[0] == 0 && indices[1] == 1, "distance ties use training index");
  check::Require(tie_class.to_vector<std::int64_t>()[0] == 0, "vote ties use lowest class");
  check::Near(tie_regression.to_vector<float>()[0], 4, 1e-7, "uniform neighbor regression");
  check::Near(weights[0], .5, 1e-7, "first duplicate weight");
  check::Near(weights[1], .5, 1e-7, "second duplicate weight");
  check::Near(weights[2], 0, 1e-7, "nonmatch ignored with exact matches");
  check::Near(exact_regression.to_vector<float>()[0], 4, 1e-7, "duplicate regression is their mean");
  check::Throws([&] { ml::NearestNeighbors invalid(duplicated, query, 5, false); }, "k larger than dataset rejected");
  std::cout << "  distance/class ties, exact duplicates, regression and invalid k passed\n";
}

void Rings() {
  constexpr int                         n = 400, q = 160, k = 5;
  std::mt19937                          rng(73);
  std::normal_distribution<float>       noise(0, .06);
  std::uniform_real_distribution<float> angle(0, 2 * std::acos(-1.f));
  std::vector<float>                    train(n * 2), query(q * 2), targets(n);
  std::vector<std::int64_t>             labels(n);
  for (int i = 0; i < n + q; ++i) {
    const float a = angle(rng), radius = (i % 2 ? 1.5f : .6f) + noise(rng);
    auto&       points = i < n ? train : query;
    const int   row    = i < n ? i : i - n;
    points[row * 2]     = radius * std::cos(a);
    points[row * 2 + 1] = radius * std::sin(a);
    if (i < n) {
      labels[i]  = i % 2;
      targets[i] = radius;
    }
  }
  const Tensor train_t = nn::from_vector(train, {n, 2}), query_t = nn::from_vector(query, {q, 2});
  const Tensor labels_t = nn::from_vector(labels), targets_t = nn::from_vector(targets);
  ml::NearestNeighbors uniform(train_t, query_t, k, false), weighted(train_t, query_t, k, true);
  const auto prediction          = uniform.Classify(labels_t, 2).to_vector<std::int64_t>();
  const auto weighted_prediction = weighted.Classify(labels_t, 2).to_vector<std::int64_t>();
  const auto regression          = weighted.Regress(targets_t).to_vector<float>();
  const auto indices             = uniform.indices.to_vector<std::int64_t>();
  const auto distances           = uniform.squared_distances.to_vector<float>();
  int        correct = 0, weighted_correct = 0;
  double     regression_mse = 0, max_distance_error = 0;
  for (int i = 0; i < q; ++i) {
    std::vector<std::pair<double, int>> reference;
    for (int j = 0; j < n; ++j) {
      const double dx = double(query[i * 2]) - train[j * 2];
      const double dy = double(query[i * 2 + 1]) - train[j * 2 + 1];
      reference.emplace_back(dx * dx + dy * dy, j);
    }
    std::sort(reference.begin(), reference.end());
    double denominator = 0, numerator = 0;
    for (int j = 0; j < k; ++j) {
      check::Require(indices[i * k + j] == reference[j].second, "neighbor index agrees with host sort");
      const double distance = distances[i * k + j];
      max_distance_error    = std::max(max_distance_error, std::abs(distance - reference[j].first));
      check::Near(distance, reference[j].first, 2e-6, "squared distance agrees with direct subtraction");
      const double weight = 1 / std::sqrt(reference[j].first);
      denominator += weight;
      numerator += weight * targets[reference[j].second];
    }
    check::Near(regression[i], numerator / denominator, 3e-5, "weighted regression agrees with direct distances");
    correct += prediction[i] == i % 2;
    weighted_correct += weighted_prediction[i] == i % 2;
    const double radius = std::hypot(query[i * 2], query[i * 2 + 1]);
    regression_mse += std::pow(regression[i] - radius, 2) / q;
  }
  check::Require(correct > .98 * q && weighted_correct > .98 * q && regression_mse < .005,
                 "held-out ring classification/regression");
  std::cout << "  5-NN ring test accuracy=" << correct / double(q) << "; weighted=" << weighted_correct / double(q)
            << "; radius regression MSE=" << regression_mse << "\n  all " << q * k
            << " neighbor indices agree; max squared-distance error=" << max_distance_error << '\n';
}

}  // namespace

int main() {
  try {
    std::cout << "== Nearest neighbors\n";
    Edges();
    Rings();
    std::cout << "All nearest-neighbor checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
