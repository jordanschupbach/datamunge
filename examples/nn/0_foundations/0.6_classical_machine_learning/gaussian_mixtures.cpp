// CMD: just run-nn gaussian_mixtures
// Full-covariance Gaussian mixtures trained by expectation-maximization.
//
// Port of tflow's gaussian_mixtures. Initialize three means with k-means++,
// then fit unequal mixture weights, means and 2x2 covariances using float64
// nn ops (batched nn::cholesky, nn::triangular_solve and nn::self_adjoint_eig
// in place of TF's Cholesky, MatrixTriangularSolve and SelfAdjointEigV2).
// Check monotone data likelihood, normalized responsibilities, independently
// computed Gaussian densities and recovery of known parameters up to
// component permutation. Evaluate a separate sample; no labels enter EM. A
// constant-data fixture exercises the covariance eigenvalue floor that
// prevents singular matrices. Artifacts go to output/nn/models/gaussian_mixtures.

#include <array>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <fstream>
#include <iostream>
#include <random>

#include "common/checks.hpp"
#include "common/gaussian_mixture.hpp"
#include "common/kmeans.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

const std::array<std::array<double, 2>, 3> means{{{{-3, -1}}, {{0, 3}}, {{3, -1}}}};
const std::array<double, 3>                mixing{{.2, .5, .3}};

Tensor Sample(int n, std::mt19937& rng, Tensor* standard = nullptr) {
  if (standard) *standard = nn::empty({n, 2}, DType::Float64);
  Tensor                           x = nn::empty({n, 2}, DType::Float64);
  double*                          p = x.data_ptr<double>();
  std::normal_distribution<double> normal;
  for (int i = 0; i < n; ++i) {
    const int    c = i % 10 < 2 ? 0 : i % 10 < 7 ? 1 : 2;
    const double a = normal(rng), b = normal(rng);
    if (standard) {
      standard->data_ptr<double>()[i * 2]     = a;
      standard->data_ptr<double>()[i * 2 + 1] = b;
    }
    p[i * 2]     = means[c][0] + (.45 + .1 * c) * a;
    p[i * 2 + 1] = means[c][1] + .2 * a + .4 * b;
  }
  return x;
}

double Density(const Tensor& x, int row, const ml::GaussianMixtureState& state, int c) {
  const double a = state.covariances.at({c, 0, 0}), b = state.covariances.at({c, 0, 1}), d = state.covariances.at({c, 1, 1});
  const double det = a * d - b * b;
  check::Require(det > 0, "positive covariance determinant");
  const double dx = x.at({row, 0}) - state.means.at({c, 0});
  const double dy = x.at({row, 1}) - state.means.at({c, 1});
  return state.weights.at({c}) * std::exp(-.5 * (d * dx * dx - 2 * b * dx * dy + a * dy * dy) / det) /
         (2 * std::acos(-1.) * std::sqrt(det));
}

Tensor Scalar(double v) { return nn::scalar(v, DType::Float64); }

}  // namespace

int main() {
  try {
    std::mt19937 rng(89);
    Tensor       fit_standard, test_standard;
    const auto   x = Sample(1000, rng, &fit_standard), test = Sample(500, rng, &test_standard);
    const auto   root = paths::OutputDir("models/gaussian_mixtures");
    nn::save_npz((root / "data.npz").string(),
                 {{"fit_points", x}, {"test_points", test}, {"fit_standard", fit_standard}, {"test_standard", test_standard}});
    const Tensor xf = nn::cast(x, DType::Float32);
    ml::KMeans   initializer(2);
    const auto   initial_centers = initializer.Initialize(xf, 3, 7, true);
    ml::GaussianMixtureState state{nn::full({3}, 1. / 3, DType::Float64), nn::cast(initial_centers, DType::Float64),
                                   nn::broadcast_to(nn::eye(2, DType::Float64), {3, 2, 2}).contiguous()};
    ml::GaussianMixture model(3, 2);
    const auto save_state = [&](const std::filesystem::path& file, const ml::GaussianMixtureState& value) {
      nn::save_npz(file.string(), {{"weights", value.weights}, {"means", value.means}, {"covariances", value.covariances}});
    };
    const auto save_step = [&](const std::filesystem::path& file, const ml::GaussianMixtureStep& value) {
      nn::save_npz(file.string(), {{"responsibilities", value.responsibilities},
                                   {"log_density", value.log_density},
                                   {"log_likelihood", Scalar(value.log_likelihood)},
                                   {"next_weights", value.next.weights},
                                   {"next_means", value.next.means},
                                   {"next_covariances", value.next.covariances}});
    };
    save_state(root / "initial_state.npz", state);
    std::ofstream history(root / "history.tsv");
    history.precision(17);
    history << "step\tfit_log_likelihood\n";
    double previous = -1e100, initial = 0;
    int    iterations = 0;
    std::cout << "== Full-covariance Gaussian mixture EM\n";
    for (; iterations < 100; ++iterations) {
      auto step = model.Step(x, state);
      save_state(root / ("step" + std::to_string(iterations) + "_before.npz"), state);
      save_step(root / ("step" + std::to_string(iterations) + ".npz"), step);
      history << iterations << '\t' << step.log_likelihood << '\n';
      check::Require(std::isfinite(step.log_likelihood) && step.log_likelihood >= previous - 1e-7,
                     "EM likelihood must not decrease");
      if (iterations == 0) initial = step.log_likelihood;
      if (iterations && step.log_likelihood - previous < 1e-7) break;
      previous = step.log_likelihood;
      state    = step.next;
    }
    check::Require(iterations < 100, "EM converges");
    const auto result = model.Step(x, state), held_out = model.Step(test, state);
    save_state(root / "model_state.npz", state);
    save_step(root / "fit_outputs.npz", result);
    save_step(root / "test_outputs.npz", held_out);
    check::Require(result.log_likelihood > initial + 100, "EM improves initialized density");
    std::array<bool, 3> matched{};
    for (int c = 0; c < 3; ++c) {
      int    label = -1;
      double error = 1e100;
      for (int j = 0; j < 3; ++j) {
        const double distance = std::hypot(state.means.at({c, 0}) - means[j][0], state.means.at({c, 1}) - means[j][1]);
        if (distance < error) error = distance, label = j;
      }
      check::Require(error < .15 && !matched[label], "recover one mean per true component");
      matched[label] = true;
      check::Near(state.weights.at({c}), mixing[label], .015, "recover unequal mixture masses");
      const double a = .45 + .1 * label;
      check::Near(state.covariances.at({c, 0, 0}), a * a, .07, "recover x variance");
      check::Near(state.covariances.at({c, 0, 1}), .2 * a, .06, "recover covariance");
      check::Near(state.covariances.at({c, 1, 1}), .2, .05, "recover y variance");
      std::cout << "  component " << c << ": weight=" << state.weights.at({c}) << "; mean=(" << state.means.at({c, 0})
                << ',' << state.means.at({c, 1}) << ")\n";
    }
    double     max_density_error = 0;
    const auto responsibilities  = held_out.responsibilities.to_vector<double>();
    const auto log_density       = held_out.log_density.to_vector<double>();
    for (int i = 0; i < 500; ++i) {
      std::array<double, 3> joint{};
      double                total = 0, responsibility_sum = 0;
      for (int c = 0; c < 3; ++c) {
        joint[c] = Density(test, i, state, c);
        total += joint[c];
      }
      for (int c = 0; c < 3; ++c) {
        const double responsibility = responsibilities[i * 3 + c];
        responsibility_sum += responsibility;
        check::Near(responsibility, joint[c] / total, 1e-11, "responsibilities match direct densities");
      }
      check::Near(responsibility_sum, 1, 1e-12, "responsibilities normalize");
      max_density_error = std::max(max_density_error, std::abs(log_density[i] - std::log(total)));
    }
    check::Require(max_density_error < 1e-11 && held_out.log_likelihood / 500 > -3, "held-out Gaussian density");
    std::cout << "  iterations=" << iterations << "; initial/final train log-likelihood=" << initial << '/'
              << result.log_likelihood << "; test mean log-density=" << held_out.log_likelihood / 500
              << "\n  direct density max log error=" << max_density_error << '\n';

    ml::GaussianMixture      degenerate(1, 2, .001);
    const Tensor             repeated = nn::zeros({8, 2}, DType::Float64);
    ml::GaussianMixtureState single{nn::ones({1}, DType::Float64), nn::zeros({1, 2}, DType::Float64),
                                    nn::reshape(nn::eye(2, DType::Float64), {1, 2, 2})};
    save_state(root / "degenerate_before.npz", single);
    const auto floored = degenerate.Step(repeated, single);
    save_step(root / "degenerate_step.npz", floored);
    save_step(root / "degenerate_after.npz", degenerate.Step(repeated, floored.next));
    const auto floored_cov = floored.next.covariances.to_vector<double>();
    for (int i = 0; i < 4; ++i) check::Near(floored_cov[i], i % 3 == 0 ? .001 : 0, 1e-12, "covariance eigenvalue floor");
    check::Require(std::isfinite(degenerate.Step(repeated, floored.next).log_likelihood),
                   "constant-data density remains finite");
    Tensor line = nn::empty({5, 2}, DType::Float64);
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 2; ++j) line.data_ptr<double>()[i * 2 + j] = i - 2;
    const auto line_step = degenerate.Step(line, single);
    nn::save_npz((root / "line_data.npz").string(), {{"points", line}});
    save_step(root / "line_step.npz", line_step);
    check::Near(line_step.next.covariances.at({0, 0, 0}), 2.0005, 1e-12, "floorOnlySmallEigenvalueNotRidge");
    check::Near(line_step.next.covariances.at({0, 0, 1}), 1.9995, 1e-12, "rotatedFloorPreservesLargeEigenvalue");
    std::ofstream meta(root / "metrics.json");
    meta.precision(17);
    meta << "{\"components\":3,\"dimensions\":2,\"accepted_em_updates\":" << iterations
         << ",\"initial_fit_log_likelihood\":" << initial << ",\"final_fit_log_likelihood\":" << result.log_likelihood
         << ",\"test_mean_log_density\":" << held_out.log_likelihood / 500
         << ",\"direct_log_density_max_error\":" << max_density_error
         << ",\"covariance_floor\":0.0001,\"source\":\"stratified synthetic correlated Gaussian;std::mt19937(89) and "
            "std::normal_distribution draws exported\"}\n";
    check::Require(bool(meta) && bool(history), "completeGMMdensityRecords");
    std::cout << "  singular covariance fixture floored to .001*I\nAll Gaussian-mixture checks passed.\nArtifacts:" << root
              << std::endl;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
