// CMD: just run-nn depthwise_separable
// Depthwise separable convolution: equivalence, gradients, rank and cost.
//
// Port of tflow's depthwise_separable. Compose HWIM depthwise and 1x1
// pointwise filters into a dense HWIO filter, with no intervening activation.
// Check outputs, input gradients and both factor gradients against dense
// convolution for multipliers 1/2 and strides 1/2. Check selected factor
// derivatives by central finite differences.
//
// A depth multiplier m limits each input channel's spatial-by-output kernel
// matrix to rank m. SVD gives the best rank-m approximation of arbitrary
// filters: quantify lost energy for m=1/2 and exact reconstruction at full
// rank. This explains why fewer parameters also restrict expressiveness.
// Count kernel parameters and multiply-accumulates, then time matching dense
// and separable forwards on CPU after warmup (nn with two threads, as TF's
// run). Reduced MACs do not guarantee a matching latency speedup; timing
// depends on kernels, hardware and shape.
// TF -> datamunge: DepthwiseConv2dNative is nn::depthwise_conv2d, Svd is
// nn::svd, AddSymbolicGradients is nn::grad. Inputs come from the same
// std::mt19937 streams as TF's (bit-identical).

#include <chrono>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>

#include "common/checks.hpp"
#include "common/separable.hpp"

namespace nn = datamunge::nn;
using nn::Shape;
using nn::Tensor;

namespace {

Tensor Random(const Shape& shape, std::mt19937& rng) {
  std::int64_t n = 1;
  for (auto d : shape) n *= d;
  std::vector<float>              v(static_cast<std::size_t>(n));
  std::normal_distribution<float> normal(0, .2f);
  for (auto& e : v) e = normal(rng);
  return nn::from_vector(v, shape);
}

double Difference(const Tensor& a, const Tensor& b) {
  check::Require(a.shape() == b.shape(), "matching tensor shapes");
  const auto x = a.contiguous().to_vector<float>(), y = b.contiguous().to_vector<float>();
  double     error = 0;
  for (std::size_t i = 0; i < x.size(); ++i) error = std::max(error, double(std::abs(x[i] - y[i])));
  return error;
}

void Equivalence() {
  std::mt19937 rng(41);
  for (int multiplier : {1, 2})
    for (int stride : {1, 2}) {
      const auto pixels = Random({2, 9, 10, 3}, rng), spatial = Random({3, 3, 3, multiplier}, rng),
                 mixing = Random({1, 1, 3 * multiplier, 5}, rng);
      struct Result {
        Tensor separable, dense, sep_loss, kernel;
        std::vector<Tensor> sep_grad, full_grad;
      };
      auto run = [&](const Tensor& a, const Tensor& b) {
        Tensor x = pixels.clone(), dw = a.clone(), pw = b.clone();
        x.requires_grad_();
        dw.requires_grad_();
        pw.requires_grad_();
        Result r;
        auto   depthwise = nn::depthwise_conv2d(x, dw, nn::conv_options(stride, nn::Padding::Same));
        r.separable      = nn::conv2d(depthwise, pw, nn::conv_options(1, nn::Padding::Valid));
        r.kernel         = separable::ComposeSeparableKernel(dw, pw, 3, 3, multiplier, 5);
        r.dense          = nn::conv2d(x, r.kernel, nn::conv_options(stride, nn::Padding::Same));
        r.sep_loss       = nn::mean(nn::square(r.separable));
        auto full_loss   = nn::mean(nn::square(r.dense));
        r.sep_grad       = nn::grad(r.sep_loss, {x, dw, pw});
        r.full_grad      = nn::grad(full_loss, {x, dw, pw});
        return r;
      };
      const auto result = run(spatial, mixing);
      check::Near(Difference(result.separable, result.dense), 0, 2e-6, "separable forward equals composed dense kernel");
      double gradient_error = 0;
      for (int i = 0; i < 3; ++i)
        gradient_error = std::max(gradient_error, Difference(result.sep_grad[i], result.full_grad[i]));
      check::Near(gradient_error, 0, 2e-6, "equivalent input and factor gradients");
      for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 3; ++x)
          for (int i = 0; i < 3; ++i)
            for (int o = 0; o < 5; ++o) {
              double expected = 0;
              for (int m = 0; m < multiplier; ++m)
                expected += spatial.at({y, x, i, m}) * mixing.at({0, 0, i * multiplier + m, o});
              check::Near(result.kernel.at({y, x, i, o}), expected, 1e-7, "host filter contraction");
            }
      for (int factor = 0; factor < 2; ++factor) {
        const auto& value = factor == 0 ? spatial : mixing;
        const auto  base  = value.to_vector<float>();
        for (std::int64_t i : {std::int64_t(0), value.numel() / 2, value.numel() - 1}) {
          auto plus = base, minus = base;
          plus[static_cast<std::size_t>(i)] += .001f;
          minus[static_cast<std::size_t>(i)] -= .001f;
          const Tensor p = nn::from_vector(plus, value.shape()), m = nn::from_vector(minus, value.shape());
          const auto   upper = factor == 0 ? run(p, mixing) : run(spatial, p);
          const auto   lower = factor == 0 ? run(m, mixing) : run(spatial, m);
          check::Near((upper.sep_loss.item<float>() - lower.sep_loss.item<float>()) / .002,
                      result.sep_grad[1 + factor].to_vector<float>()[static_cast<std::size_t>(i)], 2e-6,
                      "factor finite difference");
        }
      }
      std::cout << "  multiplier=" << multiplier << ", stride=" << stride
                << ": forward error=" << Difference(result.separable, result.dense)
                << "; gradient error=" << gradient_error << '\n';
    }
}

void RankRestriction() {
  std::mt19937 rng(109);
  const auto   arbitrary = Random({3, 9, 5}, rng);  // input channel, spatial tap, output channel
  const auto   svd       = nn::svd(arbitrary);
  const auto   a = arbitrary.to_vector<float>(), s = svd.s.contiguous().to_vector<float>(),
             u = svd.u.contiguous().to_vector<float>(), v = svd.v.contiguous().to_vector<float>();
  double total = 0;
  for (float e : a) total += std::pow(e, 2);
  double previous = INFINITY;
  for (int multiplier : {1, 2, 5}) {
    double error = 0, tail_energy = 0;
    for (int channel = 0; channel < 3; ++channel) {
      for (int j = multiplier; j < 5; ++j) tail_energy += std::pow(s[channel * 5 + j], 2);
      for (int tap = 0; tap < 9; ++tap)
        for (int out = 0; out < 5; ++out) {
          double approximation = 0;
          for (int m = 0; m < multiplier; ++m)
            approximation += u[(channel * 9 + tap) * 5 + m] * s[channel * 5 + m] * v[(channel * 5 + out) * 5 + m];
          error += std::pow(approximation - a[(channel * 9 + tap) * 5 + out], 2);
        }
    }
    check::Near(error, tail_energy, 2e-6, "best rank-m error equals discarded singular-value energy");
    check::Require(error < previous, "extra depth multiplier increases representational capacity");
    previous = error;
    if (multiplier == 1) check::Require(error / total > .3, "arbitrary dense filters are not generally rank one");
    if (multiplier == 5) check::Require(error / total < 1e-10, "full-rank factors reconstruct any kernel");
    std::cout << "  multiplier=" << multiplier << ": relative squared kernel error=" << error / total << '\n';
  }
}

void CostAndTiming() {
  constexpr int height = 32, width = 32, in = 32, out = 64, kernel = 3;
  constexpr int dense_parameters = kernel * kernel * in * out, separable_parameters = kernel * kernel * in + in * out;
  std::mt19937  rng(204);
  auto pixels = Random({1, height, width, in}, rng), dw = Random({kernel, kernel, in, 1}, rng),
       pw = Random({1, 1, in, out}, rng);
  nn::NoGradGuard guard;
  const Tensor    composed = separable::ComposeSeparableKernel(dw, pw, kernel, in, 1, out);
  auto full = [&] { return nn::conv2d(pixels, composed, nn::conv_options(1, nn::Padding::Same)); };
  auto sep  = [&] {
    return nn::conv2d(nn::depthwise_conv2d(pixels, dw, nn::conv_options(1, nn::Padding::Same)), pw,
                       nn::conv_options(1, nn::Padding::Valid));
  };
  check::Near(Difference(full(), sep()), 0, 1e-5, "benchmark functions match");
  const int threads = nn::get_num_threads();
  nn::set_num_threads(2);
  for (int variant = 0; variant < 2; ++variant) {
    double sink = 0;
    auto   run  = [&] { sink += nn::mean(variant ? sep() : full()).item(); };
    for (int warmup = 0; warmup < 10; ++warmup) run();
    const auto start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < 100; ++repeat) run();
    const double milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 100;
    check::Require(std::isfinite(sink), "timed forward is finite");
    const int parameters = variant ? separable_parameters : dense_parameters;
    std::cout << "  " << (variant ? "separable" : "dense") << ": kernel parameters=" << parameters
              << "; MACs=" << height * width * parameters << "; CPU forward mean=" << milliseconds
              << " ms (100 runs, two threads)\n";
  }
  nn::set_num_threads(threads);
  std::cout << "  theoretical kernel/MAC reduction=" << double(dense_parameters) / separable_parameters
            << "x; measured timings are eager nn calls\n";
}

}  // namespace

int main() {
  try {
    Equivalence();
    RankRestriction();
    CostAndTiming();
    std::cout << "All depthwise-separable checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
