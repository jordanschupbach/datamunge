// CMD: just run-nn kmeans_clustering
// k-means and k-means++ with nn-op Lloyd updates.
//
// Port of tflow's kmeans_clustering (model in common/kmeans.hpp). Cluster four
// separated Gaussian clouds, compare initial and final inertia across eight
// seeded restarts, and verify monotone descent, recovered centers and
// permutation-invariant assignments. Validate each center against a host mean
// of its assigned rows. Duplicate observations exercise empty clusters and the
// all-zero-distance fallback in k-means++ initialization. Data and seeds match
// the TF version (std::mt19937(83), std::mt19937_64 initialization), so both
// runs see identical points and initial centers. No files are written.

#include <array>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "common/checks.hpp"
#include "common/kmeans.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

struct Fit {
  Tensor         centers;
  ml::KMeansStep result;
  int            iterations = 0;
};

Fit Lloyd(const ml::KMeans& model, const Tensor& x, Tensor centers) {
  double previous = std::numeric_limits<double>::infinity();
  for (int iteration = 0; iteration < 100; ++iteration) {
    auto step = model.Step(x, centers);
    check::Require(std::isfinite(step.inertia) && step.inertia <= previous + 1e-3, "Lloyd inertia must not increase");
    const double movement = nn::max(nn::abs(centers - step.centers)).item();
    centers               = step.centers;
    if (movement < 1e-6) return {centers, model.Step(x, centers), iteration + 1};
    previous = step.inertia;
  }
  throw std::runtime_error("k-means did not converge");
}

}  // namespace

int main() {
  try {
    constexpr int                             n = 600, k = 4;
    const std::array<std::array<float, 2>, k> truth{{{{-3, -3}}, {{-3, 3}}, {{3, -3}}, {{3, 3}}}};
    std::vector<float>                        values(n * 2);
    std::mt19937                              rng(83);
    std::normal_distribution<float>           noise(0, .35);
    for (int i = 0; i < n; ++i)
      for (int d = 0; d < 2; ++d) values[i * 2 + d] = truth[i % k][d] + noise(rng);
    const Tensor x = nn::from_vector(values, {n, 2});
    ml::KMeans   model(2);
    std::cout << "== K-means / k-means++\n";
    double random_initial = 0, plus_initial = 0, best_random = 1e100, best_plus = 1e100;
    Fit    best{};
    for (bool plus : {false, true}) {
      double initial_total = 0, final_total = 0, best_inertia = 1e100;
      for (int seed = 0; seed < 8; ++seed) {
        auto centers = model.Initialize(x, k, seed, plus);
        initial_total += model.Step(x, centers).inertia;
        auto fit = Lloyd(model, x, centers);
        final_total += fit.result.inertia;
        if (fit.result.inertia < best_inertia) {
          best_inertia = fit.result.inertia;
          if (plus) best = fit;
        }
      }
      if (plus) plus_initial = initial_total / 8, best_plus = best_inertia;
      else random_initial = initial_total / 8, best_random = best_inertia;
      std::cout << "  " << (plus ? "k-means++" : "random") << " eight restarts: mean initial inertia="
                << initial_total / 8 << "; mean final=" << final_total / 8 << "; best=" << best_inertia << '\n';
    }
    check::Require(plus_initial < random_initial, "k-means++ improves mean initialization on this fixture");
    check::Near(best_plus, best_random, .001, "both initializers can reach the best clustering");
    std::array<int, k>  center_class{};
    std::array<bool, k> seen{};
    const auto          labels = best.result.labels.to_vector<std::int64_t>();
    for (int c = 0; c < k; ++c) {
      double distance = 1e100;
      for (int label = 0; label < k; ++label) {
        const double error =
            std::hypot(best.centers.at({c, 0}) - truth[label][0], best.centers.at({c, 1}) - truth[label][1]);
        if (error < distance) distance = error, center_class[c] = label;
      }
      check::Require(distance < .15 && !seen[center_class[c]], "one accurate center per cloud");
      seen[center_class[c]] = true;
      std::array<double, 2> sum{};
      int                   count = 0;
      for (int i = 0; i < n; ++i)
        if (labels[i] == c) {
          ++count;
          for (int d = 0; d < 2; ++d) sum[d] += values[i * 2 + d];
        }
      check::Near(best.result.counts.at({c}), count, 0, "segment count");
      for (int d = 0; d < 2; ++d)
        check::Near(best.centers.at({c, d}), sum[d] / count, 2e-6, "centroid is assigned-row mean");
    }
    int correct = 0;
    for (int i = 0; i < n; ++i) correct += center_class[labels[i]] == i % k;
    check::Require(correct == n, "permutation-invariant clustering accuracy");
    std::cout << "  best k-means++ converged in " << best.iterations << " iterations; label agreement=" << correct / double(n)
              << '\n';

    const Tensor duplicates    = nn::full({6, 2}, 2.0);
    const auto   duplicate_fit = Lloyd(model, duplicates, model.Initialize(duplicates, 4, 3, true));
    check::Near(duplicate_fit.result.inertia, 0, 0, "duplicate observations have zero inertia");
    int empty = 0;
    for (int c = 0; c < 4; ++c) empty += duplicate_fit.result.counts.at({c}) == 0;
    check::Require(empty == 3, "empty cluster policy exercised");
    for (float v : duplicate_fit.centers.to_vector<float>()) check::Near(v, 2, 0, "empty centers remain finite and unchanged");
    std::cout << "  all-identical fixture: three empty clusters preserved; zero inertia\nAll k-means checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
