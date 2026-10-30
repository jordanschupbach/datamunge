// CMD: just run-nn dilated_convolutions
// Atrous convolutions: exact dilation, gridding and learning distant context.
//
// Port of tflow's dilated_convolutions. A 3x3 filter at rate d has effective
// width 2d+1 and nine learned taps per channel pair. Compare zero-expanded
// filters with native dilated nn::conv2d at rates 1,2,3 under SAME/VALID
// padding and check input/filter derivatives numerically. The expansion
// supplied CPU gradients where TF's native dilation backprop is unsupported;
// nn's dilated backprop is native, so this port also checks that the native
// and expanded-filter gradients agree. The reference path pays for a larger
// dense convolution.
//
// Compare three stacks: rates [1,1,1], [2,2,2], [1,2,4]. Positive-kernel input
// gradients reveal dense 7x7, sparse 13x13 and dense 15x15 dependency patterns.
// Equal rates leave gridding holes despite a larger enclosing field.
// Finally fit a nine-tap, rate-2 teacher using nine-parameter rate-1/rate-2
// students on independent random images. Only the matched receptive field can
// recover the teacher; compare held-out MSE, never select on test examples.
// Inputs come from the same std::mt19937 streams as TF's (bit-identical).
// Support arrays: output/nn/visualizations/dilated_convolutions/supports.npz.

#include <array>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <map>
#include <random>

#include "common/checks.hpp"
#include "common/dilated.hpp"
#include "common/paths.hpp"
#include "common/receptive_field.hpp"

namespace nn = datamunge::nn;
using nn::Shape;
using nn::Tensor;

namespace {

Tensor Random(const Shape& shape, std::mt19937& rng) {
  std::int64_t n = 1;
  for (auto d : shape) n *= d;
  std::vector<float>              v(static_cast<std::size_t>(n));
  std::normal_distribution<float> normal(0, .5f);
  for (auto& e : v) e = normal(rng);
  return nn::from_vector(v, shape);
}

double MaxDifference(const Tensor& a, const Tensor& b) {
  check::Require(a.shape() == b.shape(), "dilated output shape");
  const auto x = a.contiguous().to_vector<float>(), y = b.contiguous().to_vector<float>();
  double     m = 0;
  for (std::size_t i = 0; i < x.size(); ++i) m = std::max(m, double(std::abs(x[i] - y[i])));
  return m;
}

void Equivalence() {
  std::mt19937 rng(7);
  const auto   pixels = Random({2, 11, 12, 2}, rng), filter = Random({3, 3, 2, 3}, rng);
  for (int dilation : {1, 2, 3})
    for (const std::string padding : {"SAME", "VALID"}) {
      const auto pad = padding == "SAME" ? nn::Padding::Same : nn::Padding::Valid;
      struct Result {
        Tensor native, portable, objective;
        std::vector<Tensor> gradients, native_gradients;
      };
      auto run = [&](const Tensor& a, const Tensor& b, bool derivatives) {
        Tensor x = a.clone(), w = b.clone();
        x.requires_grad_();
        w.requires_grad_();
        Result r;
        r.native    = nn::conv2d(x, w, nn::conv_options(1, pad, dilation));
        r.portable  = dilated::PortableDilatedConv2D(x, w, dilation, padding);
        r.objective = nn::mean(nn::square(r.portable));
        if (derivatives) {
          r.gradients        = nn::grad(r.objective, {x, w});
          r.native_gradients = nn::grad(nn::mean(nn::square(r.native)), {x, w});
        }
        return r;
      };
      const auto result  = run(pixels, filter, true);
      const double maximum = MaxDifference(result.native, result.portable);
      check::Near(maximum, 0, 2e-6, "native and expanded-filter dilation");
      const double native_gradient_error = std::max(MaxDifference(result.gradients[0], result.native_gradients[0]),
                                                    MaxDifference(result.gradients[1], result.native_gradients[1]));
      check::Near(native_gradient_error, 0, 2e-6, "native dilated backprop equals expanded-filter gradients");
      double gradient_error = 0;
      for (int which = 0; which < 2; ++which) {
        const auto& source = which == 0 ? pixels : filter;
        const auto  base   = source.to_vector<float>();
        for (std::int64_t i : {std::int64_t(0), source.numel() / 2, source.numel() - 1}) {
          auto plus = base, minus = base;
          plus[static_cast<std::size_t>(i)] += .002f;
          minus[static_cast<std::size_t>(i)] -= .002f;
          const Tensor p = nn::from_vector(plus, source.shape()), m = nn::from_vector(minus, source.shape());
          const auto   upper  = which == 0 ? run(p, filter, false) : run(pixels, p, false);
          const auto   lower  = which == 0 ? run(m, filter, false) : run(pixels, m, false);
          const double finite = (upper.objective.item<float>() - lower.objective.item<float>()) / .004;
          const double error =
              std::abs(finite - result.gradients[which].to_vector<float>()[static_cast<std::size_t>(i)]);
          gradient_error = std::max(gradient_error, error);
          check::Near(error, 0, 8e-5, "dilated finite-difference derivative");
        }
      }
      std::cout << "  rate=" << dilation << ' ' << padding << ": forward error=" << maximum
                << "; derivative error=" << gradient_error << "; native-vs-expanded gradient error="
                << native_gradient_error << '\n';
    }
}

void Gridding() {
  std::map<std::string, Tensor>          arrays;
  const std::vector<std::vector<int>>    schedules = {{1, 1, 1}, {2, 2, 2}, {1, 2, 4}};
  const Tensor                           kernel    = nn::ones({3, 3, 1, 1});
  for (std::size_t schedule = 0; schedule < schedules.size(); ++schedule) {
    Tensor x = nn::zeros({1, 33, 33, 1});
    x.requires_grad_();
    Tensor                    h = x;
    receptive::ReceptiveField field;
    std::string               name;
    for (int dilation : schedules[schedule]) {
      h     = dilated::PortableDilatedConv2D(h, kernel, dilation);
      field = field.After(3, 1, dilation, dilation);
      name += std::to_string(dilation);
    }
    auto       center   = nn::sum(nn::slice(h, {0, 16, 16, 0}, {1, 1, 1, 1}));
    const auto gradient = nn::grad(center, {x})[0];
    const auto values   = gradient.to_vector<float>();
    int        count    = 0;
    const auto [lo, hi] = field.Bounds(16);
    for (int y = 0; y < 33; ++y)
      for (int xx = 0; xx < 33; ++xx) {
        const bool inside    = y >= lo && y < hi && xx >= lo && xx < hi;
        const bool connected = values[static_cast<std::size_t>(y * 33 + xx)] > 0;
        const bool expected  = inside && (schedule != 1 || (y % 2 == 0 && xx % 2 == 0));
        check::Require(connected == expected, "exact dense or gridded dependency pattern");
        count += connected;
      }
    arrays["rates_" + name] = gradient;
    std::cout << "  rates " << name << ": bounding width=" << field.width << "; connected pixels=" << count << '/'
              << field.width * field.width << '\n';
  }
  const auto dir = paths::OutputDir("visualizations/dilated_convolutions");
  nn::save_npz((dir / "supports.npz").string(), nn::NamedTensors(arrays.begin(), arrays.end()));
}

void LearnContext() {
  nn::ParameterStore            parameters(19);
  const std::array<float, 9>    coefficients = {.4f, -.7f, .2f, .6f, 1.f, -.3f, -.5f, .8f, .25f};
  const Tensor                  truth = nn::from_vector(std::vector<float>(coefficients.begin(), coefficients.end()),
                                                        {3, 3, 1, 1});
  for (int dilation : {1, 2}) parameters.get("rate" + std::to_string(dilation), {3, 3, 1, 1}, nn::init::zeros());
  auto losses = [&](const Tensor& x) {
    const Tensor        teacher = dilated::PortableDilatedConv2D(x, truth, 2);
    std::vector<Tensor> result;
    for (int dilation : {1, 2}) {
      auto prediction =
          dilated::PortableDilatedConv2D(x, parameters.at("rate" + std::to_string(dilation)), dilation);
      result.push_back(nn::mean(nn::squared_difference(prediction, teacher)));
    }
    return result;
  };
  nn::OptimizerConfig config;
  config.kind = nn::OptimizerKind::Adam;
  nn::Optimizer optimizer(parameters, .03, config);
  std::mt19937  rng(101);
  auto          training = Random({32, 13, 13, 1}, rng), test = Random({64, 13, 13, 1}, rng);
  for (int step = 0; step < 400; ++step) {
    const auto l = losses(training);
    optimizer.minimize(l[0] + l[1]);
  }
  nn::NoGradGuard guard;
  const auto      held_out = losses(test);
  const double    rate1 = held_out[0].item(), rate2 = held_out[1].item();
  std::cout << "  equal nine-parameter students, held-out MSE: rate1=" << rate1 << ", rate2=" << rate2 << '\n';
  check::Require(rate1 > .2f, "rate-1 student lacks independent distant context");
  check::Require(rate2 < 1e-8f, "rate-2 student recovers teacher");
  const auto learned = parameters.at("rate2").to_vector<float>();
  for (int i = 0; i < 9; ++i) check::Near(learned[i], coefficients[i], 1e-4, "learned dilated coefficient");
}

}  // namespace

int main() {
  try {
    Equivalence();
    Gridding();
    LearnContext();
    std::cout << "All dilated-convolution checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
