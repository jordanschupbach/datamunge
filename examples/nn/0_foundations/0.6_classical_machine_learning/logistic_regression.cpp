// CMD: just run-nn logistic_regression
// Binary and multinomial logistic regression with stable losses.
//
// Port of tflow's logistic_regression. Fit binary labels sampled from a
// known sigmoid, using an intercept and two features with a small L2 penalty
// on the slopes. Evaluate disjoint samples and check calibration against the
// data-generating sigmoid. Check the binary loss and its derivative at zero
// and at logits of +/-1000. Then fit a three-class softmax model on
// separated Gaussian clusters and check held-out accuracy, probability
// normalisation and the confusion matrix.
//
// The data come from the same std::mt19937 streams as the TF version, so the
// two runs see identical samples. Adam is written inline here; P1 of the
// plan adds nn optimizers.

#include <array>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

/// Per-element binary cross-entropy from logits, stable for any |z|, with
/// the exact sigmoid(z) - y derivative at z = 0 (tflow ml::BinaryCrossEntropy).
Tensor BinaryCrossEntropy(const Tensor& logits, const Tensor& labels) {
  const Tensor positive           = nn::greater_equal(logits, 0.0);
  const Tensor negative_magnitude = nn::where(positive, -logits, logits);
  const Tensor linear             = nn::where(positive, logits * (1.0 - labels), -(logits * labels));
  return linear + nn::log1p(nn::exp(negative_magnitude));
}

/// Adam with bias correction (beta1 0.9, beta2 0.999, epsilon 1e-8).
class Adam {
 public:
  Adam(std::vector<Tensor> params, double rate) : params_(std::move(params)), rate_(rate) {
    for (const auto& p : params_) {
      m_.push_back(nn::zeros_like(p));
      v_.push_back(nn::zeros_like(p));
    }
  }
  void Step(const std::vector<Tensor>& grads) {
    nn::NoGradGuard guard;
    ++t_;
    const double c1 = 1 - std::pow(0.9, t_), c2 = 1 - std::pow(0.999, t_);
    for (std::size_t i = 0; i < params_.size(); ++i) {
      m_[i] = 0.9 * m_[i] + 0.1 * grads[i];
      v_[i] = 0.999 * v_[i] + 0.001 * nn::square(grads[i]);
      params_[i].sub_((m_[i] / c1) / (nn::sqrt(v_[i] / c2) + 1e-8), rate_);
    }
  }

 private:
  std::vector<Tensor> params_, m_, v_;
  double              rate_;
  int                 t_{0};
};

void Binary(int steps) {
  // Stable loss and gradient at extreme logits.
  Tensor       extreme = nn::tensor({-1000, -1000, 1000, 1000, 0, 0}).requires_grad_();
  const Tensor labels  = nn::tensor({0, 1, 0, 1, 0, 1});
  const Tensor stable  = BinaryCrossEntropy(extreme, labels);
  const auto   dloss   = nn::grad(nn::sum(stable, {0}), {extreme})[0].to_vector<double>();
  const auto   loss_v  = stable.to_vector<double>();
  const double expected_loss[]{0, 1000, 1000, 0, std::log(2.), std::log(2.)};
  const double expected_gradient[]{0, -1, 1, 0, .5, -.5};
  for (int i = 0; i < 6; ++i) {
    check::Near(loss_v[i], expected_loss[i], 1e-6, "stable binary loss");
    check::Near(dloss[i], expected_gradient[i], 1e-6, "stable binary gradient");
  }

  std::mt19937                          rng(31);
  std::normal_distribution<float>       normal;
  std::uniform_real_distribution<float> uniform;
  auto dataset = [&](int n) {
    std::vector<float> features(static_cast<std::size_t>(n) * 3), targets(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      const float a = normal(rng), b = normal(rng), chance = 1 / (1 + std::exp(-(-.4f + 1.5f * a - 2.f * b)));
      features[static_cast<std::size_t>(i) * 3 + 0] = 1;
      features[static_cast<std::size_t>(i) * 3 + 1] = a;
      features[static_cast<std::size_t>(i) * 3 + 2] = b;
      targets[static_cast<std::size_t>(i)]          = uniform(rng) < chance;
    }
    return std::pair<Tensor, Tensor>{nn::from_vector(features, {n, 3}), nn::from_vector(targets, {n, 1})};
  };
  const auto [train_x, train_y] = dataset(1200);
  const auto [test_x, test_y]   = dataset(600);

  Tensor       w    = nn::zeros({3, 1}).requires_grad_();
  const Tensor mask = nn::tensor({0, 1, 1}, {3, 1});  // no penalty on the intercept
  auto objective = [&](const Tensor& x, const Tensor& y) {
    const Tensor loss = nn::mean(BinaryCrossEntropy(nn::matmul(x, w), y), {0, 1});
    return std::pair<Tensor, Tensor>{loss, loss + .005 * nn::sum(mask * nn::square(w), {0, 1})};
  };
  Adam opt({w}, .04);
  const double initial_objective = objective(train_x, train_y).second.item();
  for (int step = 0; step < steps; ++step) opt.Step(nn::grad(objective(train_x, train_y).second, {w}));
  check::Require(objective(train_x, train_y).second.item() < initial_objective, "binary training reduces loss");
  const auto stationarity = nn::grad(objective(train_x, train_y).second, {w})[0].to_vector<double>();
  if (steps >= 400)
    for (int j = 0; j < 3; ++j) check::Near(stationarity[j], 0, 1e-5, "binary optimum gradient");

  nn::NoGradGuard guard;
  const double    test_loss   = objective(test_x, test_y).first.item();
  const auto      probability = nn::sigmoid(nn::matmul(test_x, w)).to_vector<double>();
  const auto      tx = test_x.to_vector<double>(), ty = test_y.to_vector<double>();
  int             correct            = 0;
  double          calibration_error = 0;
  int             confusion[2][2]{};
  for (int i = 0; i < 600; ++i) {
    const double chance = probability[i];
    correct += (chance >= .5) == (ty[i] == 1);
    ++confusion[ty[i] == 1 ? 1 : 0][chance >= .5 ? 1 : 0];
    const double truth = 1 / (1 + std::exp(-(-.4 + 1.5 * tx[i * 3 + 1] - 2 * tx[i * 3 + 2])));
    calibration_error += std::abs(chance - truth) / 600;
  }
  if (steps >= 400)
    check::Require(test_loss < .5 && correct > 450 && calibration_error < .06, "binary generalization and calibration");
  std::cout << "  binary test NLL=" << test_loss << "; accuracy=" << correct / 600. << "; mean |p-p_true|="
            << calibration_error << "; weights=" << w.str() << '\n';
  std::cout << "  Confusion matrix (rows=truth, columns=prediction):\n";
  for (const auto& row : confusion) std::cout << "    " << row[0] << ' ' << row[1] << '\n';
}

void Multinomial(int steps) {
  std::mt19937                    rng(47);
  std::normal_distribution<float> noise(0, .55);
  auto dataset = [&](int n) {
    std::vector<float>        x(static_cast<std::size_t>(n) * 2);
    std::vector<std::int32_t> y(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      const int    c     = i % 3;
      const double angle = 2 * std::acos(-1.) * c / 3;
      x[static_cast<std::size_t>(i) * 2 + 0] = static_cast<float>(2 * std::cos(angle) + noise(rng));
      x[static_cast<std::size_t>(i) * 2 + 1] = static_cast<float>(2 * std::sin(angle) + noise(rng));
      y[static_cast<std::size_t>(i)]         = c;
    }
    return std::pair<Tensor, Tensor>{nn::from_vector(x, {n, 2}), nn::from_vector(y)};
  };
  const auto [train_x, train_y] = dataset(450);
  const auto [test_x, test_y]   = dataset(300);

  // One dense layer (Glorot-uniform weights, zero bias), L2 0.001 on the weights.
  nn::Generator gen(19);
  const double  limit = std::sqrt(6.0 / (2 + 3));
  Tensor        W     = nn::uniform({2, 3}, -limit, limit, gen).requires_grad_();
  Tensor        b     = nn::zeros({3}).requires_grad_();
  auto          nll   = [&](const Tensor& x, const Tensor& y) {
    const Tensor log_probs = nn::log_softmax(nn::matmul(x, W) + b, 1);
    return -nn::mean(nn::sum(nn::one_hot(y, 3) * log_probs, {1}), {0});
  };
  Adam opt({W, b}, .04);
  const double initial_nll = nll(train_x, train_y).item();
  for (int step = 0; step < steps; ++step)
    opt.Step(nn::grad(nll(train_x, train_y) + .5 * .001 * nn::sum(nn::square(W)), {W, b}));
  check::Require(nll(train_x, train_y).item() < initial_nll, "multinomial training reduces loss");

  nn::NoGradGuard guard;
  const Tensor    logits        = nn::matmul(test_x, W) + b;
  const Tensor    probabilities = nn::softmax(logits, 1);
  const auto      prediction    = nn::argmax(logits, 1).to_vector<int>();
  const auto      truth         = test_y.to_vector<int>();
  const double    test_nll      = nll(test_x, test_y).item();
  const double    accuracy      = nn::mean(nn::cast(nn::equal(nn::argmax(logits, 1), test_y), DType::Float32)).item();
  std::array<std::array<int, 3>, 3> confusion{};
  const auto                        row_sums = nn::sum(probabilities, {1}).to_vector<double>();
  for (int i = 0; i < 300; ++i) {
    ++confusion[static_cast<std::size_t>(truth[static_cast<std::size_t>(i)])]
               [static_cast<std::size_t>(prediction[static_cast<std::size_t>(i)])];
    check::Near(row_sums[static_cast<std::size_t>(i)], 1, 1e-6, "softmax probabilities sum to one");
  }
  if (steps >= 300) check::Require(accuracy > .97 && test_nll < .1, "multinomial held-out classification");
  std::cout << "  multinomial test NLL=" << test_nll << "; accuracy=" << accuracy
            << "\n  Confusion matrix (rows=truth, columns=prediction):\n";
  for (const auto& row : confusion) std::cout << "    " << row[0] << ' ' << row[1] << ' ' << row[2] << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int binary_steps = 400, multinomial_steps = 300;
    for (int i = 1; i < argc; ++i) {
      const std::string argument(argv[i]);
      if (argument.rfind("epochs=", 0) == 0) binary_steps = multinomial_steps = std::stoi(argument.substr(7));
      if (argument.rfind("steps=", 0) == 0) binary_steps = multinomial_steps = std::stoi(argument.substr(6));
    }
    check::Require(binary_steps > 0 && multinomial_steps > 0, "positive training budget required");
    std::cout << "== Logistic regression\n";
    Binary(binary_steps);
    Multinomial(multinomial_steps);
    std::cout << "All logistic regression checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
