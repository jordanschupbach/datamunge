// CMD: just run-nn support_vector_machines
// Linear SVM and an approximate RBF-kernel SVM using random Fourier features.
//
// Port of tflow's support_vector_machines (RBF kernel and random Fourier
// features in common/kernels.hpp; autograd + nn::Optimizer Adam replace
// AddSymbolicGradients and the graph optimizer).
// Minimize mean(max(0,1-y*f(x))) + lambda*||w||^2/2 with Adam; leave the bias
// unpenalized. Compare a linear model on separated Gaussian clusters with
// linear and RFF models on concentric rings. Training and test points are
// independently generated. The nonlinear model uses 256 fixed random cosine
// features, so it solves a finite-feature primal problem, not the exact kernel
// dual. Compare its feature inner products with exp(-gamma*distance^2), and
// check held-out accuracy, finite objectives and the number of margin errors.
// Data and RFF draws come from the same seeded streams as the TF version.
// No files are written.

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/checks.hpp"
#include "common/kernels.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

using Dataset = std::pair<Tensor, Tensor>;
Dataset Data(int count, bool rings, std::mt19937& rng) {
  std::vector<float>                    x(count * 2), y(count);
  std::normal_distribution<float>       normal;
  std::uniform_real_distribution<float> angle(0, 2 * std::acos(-1.f));
  for (int i = 0; i < count; ++i) {
    const float label = i % 2 ? 1.f : -1.f;
    y[i]              = label;
    if (rings) {
      const float radius = (label > 0 ? 1.7f : .55f) + .09f * normal(rng), a = angle(rng);
      x[i * 2]           = radius * std::cos(a);
      x[i * 2 + 1]       = radius * std::sin(a);
    } else {
      x[i * 2]     = 1.5f * label + .45f * normal(rng);
      x[i * 2 + 1] = .5f * label + .45f * normal(rng);
    }
  }
  return {nn::from_vector(x, {count, 2}), nn::from_vector(y, {count, 1})};
}

double Fit(const Dataset& train, const Dataset& test, bool rff, const std::string& label) {
  ml::RandomFourierFeatures mapping(2, 256, 1.f, 67);
  auto                      features = [&](const Tensor& x) { return rff ? mapping.Transform(x) : x; };
  Tensor                    w        = nn::zeros({rff ? 256 : 2, 1}).requires_grad_(true);
  Tensor                    bias     = nn::zeros({}).requires_grad_(true);
  struct Outputs {
    Tensor scores, hinge, objective;
  };
  auto evaluate = [&](const Dataset& data) {
    Outputs out;
    out.scores      = nn::matmul(features(data.first), w) + bias;
    out.hinge       = nn::mean(nn::relu(1.0 - data.second * out.scores));
    out.objective   = out.hinge + .005 * nn::sum(nn::square(w));
    return out;
  };
  nn::Optimizer optimizer({w, bias}, .03, {nn::OptimizerKind::Adam});
  for (int step = 0; step < 600; ++step) {
    optimizer.set_learning_rate(step < 400 ? .03 : .003);
    optimizer.step(nn::grad(evaluate(train).objective, {w, bias}));
  }
  nn::NoGradGuard no_grad;
  const float     training_objective = evaluate(train).objective.item<float>();
  check::Require(std::isfinite(training_objective) && training_objective <= 1.001,
                 "SVM beats zero-weight training objective");
  const Outputs out    = evaluate(test);
  const auto    scores = out.scores.to_vector<float>(), labels = test.second.to_vector<float>();
  int           correct = 0, violations = 0;
  for (size_t i = 0; i < labels.size(); ++i) {
    const float margin_value = scores[i] * labels[i];
    correct += margin_value > 0;
    violations += margin_value < 1;
  }
  const double accuracy = double(correct) / labels.size();
  check::Require(std::isfinite(out.objective.item<float>()), "finite held-out SVM objective");
  std::cout << "  " << label << ": test accuracy=" << accuracy << "; hinge=" << out.hinge.item<float>()
            << "; margin violations=" << violations << '/' << labels.size()
            << "; training objective=" << training_objective << '\n';
  if (rff) {
    const Tensor probes      = nn::slice(test.first, {0, 0}, {32, 2});
    const auto   z           = features(probes);
    const auto   approximate = nn::matmul(z, z, false, true).to_vector<float>();
    const auto   exact       = ml::RbfKernel(probes, probes, 1.f).to_vector<float>();
    double       error       = 0;
    for (int i = 0; i < 1024; ++i) error += std::abs(approximate[i] - exact[i]) / 1024;
    check::Require(error < .1, "RFF approximates RBF kernel");
    std::cout << "  256-feature kernel mean absolute error=" << error << '\n';
  }
  return accuracy;
}

}  // namespace

int main() {
  try {
    std::mt19937 rng(59);
    std::cout << "== Support vector machines\n";
    const auto cluster_train = Data(500, false, rng), cluster_test = Data(400, false, rng);
    check::Require(Fit(cluster_train, cluster_test, false, "linear / clusters") > .98, "linear SVM separates clusters");
    const auto ring_train = Data(600, true, rng), ring_test = Data(400, true, rng);
    const auto linear    = Fit(ring_train, ring_test, false, "linear / rings");
    const auto nonlinear = Fit(ring_train, ring_test, true, "RFF / rings");
    check::Require(nonlinear > .98 && nonlinear - linear > .25, "RFF resolves the nonlinear boundary");
    std::cout << "All support vector machine checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
