// CMD: just run-nn transposed_convolutions
// Transposed convolution, its adjoint identity and checkerboard artifacts.
//
// Port of tflow's transposed_convolutions. TensorFlow implements transposed
// convolution as Conv2DBackpropInput; nn::conv2d_transpose is the same
// operation. It is the adjoint of a specified forward convolution, not
// generally its inverse. Verify SAME/VALID output shapes against an
// independent scatter-and-add loop and verify the inner-product adjoint
// identity on multi-channel inputs. Check derivatives with respect to coarse
// input and filter as well.
//
// Feed a constant 8x8 image to stride-2 transposed convolution. A 3x3 all-one
// kernel creates unequal interior overlap across the four stride phases.
// A 4x4 all-one kernel equalizes overlap; arbitrary learned 4x4 weights can
// still create phase imbalance. Nearest-neighbor resize followed by ordinary
// 3x3 convolution preserves constants away from padding boundaries. Measure
// interior phase means/CV separately from boundary effects, and export maps.
// Inputs come from the same std::mt19937(18) stream as TF's (bit-identical).
// Arrays: output/nn/visualizations/transposed_convolutions/checkerboards.npz
// (tflow's scripts/plot_convolution_geometry.py plots the TF copies).

#include <array>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <map>
#include <random>
#include <sstream>

#include "common/checks.hpp"
#include "common/im2col.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Shape;
using nn::Tensor;

namespace {

Tensor Random(const Shape& shape, std::mt19937& rng) {
  std::int64_t n = 1;
  for (auto d : shape) n *= d;
  std::vector<float>              v(static_cast<std::size_t>(n));
  std::normal_distribution<float> normal(0, .25f);
  for (auto& e : v) e = normal(rng);
  return nn::from_vector(v, shape);
}

double Difference(const Tensor& a, const Tensor& b) {
  check::Require(a.shape() == b.shape(), "transposed tensor shapes agree");
  const auto x = a.contiguous().to_vector<float>(), y = b.contiguous().to_vector<float>();
  double     error = 0;
  for (std::size_t i = 0; i < x.size(); ++i) error = std::max(error, double(std::abs(x[i] - y[i])));
  return error;
}

std::string ShapeStr(const Shape& s) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < s.size(); ++i) out << (i ? "," : "") << s[i];
  return out.str() + ']';
}

void AdjointChecks() {
  std::mt19937 rng(18);
  for (const std::string padding : {"SAME", "VALID"}) {
    const int  height = padding == "SAME" ? 8 : 9, width = padding == "SAME" ? 10 : 11;
    auto       coarse = Random({2, 4, 5, 2}, rng), kernel = Random({3, 3, 3, 2}, rng),
         fine   = Random({2, height, width, 3}, rng);
    const im2col::PatchGeometry geometry(fine.shape(), 3, 3, 2, 2, 1, 1, padding);
    const auto c = coarse.to_vector<float>(), k = kernel.to_vector<float>();
    std::vector<float> reference(static_cast<std::size_t>(fine.numel()), 0.f);
    for (int n = 0; n < 2; ++n)
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 5; ++x)
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
              const int py = 2 * y + a - int(geometry.top), px = 2 * x + b - int(geometry.left);
              if (py < 0 || py >= height || px < 0 || px >= width) continue;
              for (int out = 0; out < 3; ++out)
                for (int in = 0; in < 2; ++in)
                  reference[static_cast<std::size_t>(((n * height + py) * width + px) * 3 + out)] +=
                      c[static_cast<std::size_t>(((n * 4 + y) * 5 + x) * 2 + in)] *
                      k[static_cast<std::size_t>(((a * 3 + b) * 3 + out) * 2 + in)];
            }
    const auto options = nn::conv_options(2, padding == "SAME" ? nn::Padding::Same : nn::Padding::Valid);
    struct Result {
      Tensor transposed, forward, objective, paired;
      std::vector<Tensor> gradients;
    };
    auto run = [&](const Tensor& filter, bool derivatives) {
      Tensor x = coarse.clone(), w = filter.clone();
      x.requires_grad_();
      w.requires_grad_();
      Result r;
      r.transposed = nn::conv2d_transpose(x, w, fine.shape(), options);
      r.forward    = nn::conv2d(fine, w, options);
      r.objective  = nn::sum(r.transposed * fine);
      r.paired     = nn::sum(r.forward * x);
      if (derivatives) r.gradients = nn::grad(r.objective, {x, w});
      return r;
    };
    const auto   result        = run(kernel, true);
    const double scatter_error = Difference(result.transposed, nn::from_vector(reference, fine.shape()));
    check::Near(scatter_error, 0, 2e-7, "transposed output equals scatter-and-add reference");
    check::Near(result.objective.item(), result.paired.item(), 2e-6, "convolution inner-product adjoint identity");
    check::Near(Difference(result.forward, result.gradients[0]), 0, 1e-7,
                "transposed input gradient is forward convolution");
    const auto dw = result.gradients[1].to_vector<float>();
    for (std::int64_t index : {std::int64_t(0), kernel.numel() / 2, kernel.numel() - 1}) {
      auto plus = k, minus = k;
      plus[static_cast<std::size_t>(index)] += .002f;
      minus[static_cast<std::size_t>(index)] -= .002f;
      const double upper = run(nn::from_vector(plus, kernel.shape()), false).objective.item<float>();
      const double lower = run(nn::from_vector(minus, kernel.shape()), false).objective.item<float>();
      check::Near((upper - lower) / .004, dw[static_cast<std::size_t>(index)], 2e-4,
                  "transposed filter finite difference");
    }
    std::cout << "  " << padding << " output=" << ShapeStr(result.transposed.shape())
              << "; scatter error=" << scatter_error
              << "; adjoint error=" << std::abs(result.objective.item<float>() - result.paired.item<float>()) << '\n';
  }
}

void Checkerboards() {
  const Tensor        coarse = nn::ones({1, 8, 8, 1});
  std::vector<Tensor> result;
  for (int variant = 0; variant < 3; ++variant) {
    const int          k = variant == 0 ? 3 : 4;
    std::vector<float> filter(static_cast<std::size_t>(k * k), 1.f);
    if (variant == 2) filter[0] = 4.f;
    result.push_back(nn::conv2d_transpose(coarse, nn::from_vector(filter, {k, k, 1, 1}), Shape{1, 14 + k, 14 + k, 1},
                                          nn::conv_options(2, nn::Padding::Valid)));
  }
  auto resized = nn::resize_nearest(coarse, {16, 16});
  result.push_back(nn::conv2d(resized, nn::ones({3, 3, 1, 1}), nn::conv_options(1, nn::Padding::Same)));
  const std::array<std::string, 4> labels = {"transpose_k3", "transpose_k4", "transpose_k4_weighted", "resize_conv"};
  std::map<std::string, Tensor>    arrays;
  for (std::size_t variant = 0; variant < result.size(); ++variant) {
    std::array<double, 4> means{};
    std::array<int, 4>    counts{};
    const auto&           value = result[variant];
    // Use the same 10x10 interior window in all variants, with five samples
    // of each phase on each axis. It excludes every padding/overlap edge.
    for (int y = 4; y < 14; ++y)
      for (int x = 4; x < 14; ++x) {
        const int phase = 2 * (y % 2) + x % 2;
        means[phase] += value.at({0, y, x, 0});
        ++counts[phase];
      }
    double average = 0, variance = 0;
    for (int i = 0; i < 4; ++i) {
      means[i] /= counts[i];
      average += means[i] / 4;
    }
    for (double mean : means) variance += (mean - average) * (mean - average) / 4;
    const double cv = std::sqrt(variance) / average;
    if (variant == 0 || variant == 2)
      check::Require(cv > .1, "unequal phases produce checkerboard response");
    else
      check::Near(cv, 0, 0, "equal interior phases for uniform even kernel or resize-convolution");
    if (variant == 0) {
      const std::array<double, 4> expected = {4, 2, 2, 1};
      for (int i = 0; i < 4; ++i) check::Near(means[i], expected[i], 0, "3x3 stride-2 overlap multiplicity");
    }
    arrays[labels[variant]] = value;
    std::cout << "  " << labels[variant] << ": interior phase means";
    for (double mean : means) std::cout << ' ' << mean;
    std::cout << "; phase CV=" << cv << '\n';
  }
  const auto dir = paths::OutputDir("visualizations/transposed_convolutions");
  nn::save_npz((dir / "checkerboards.npz").string(), nn::NamedTensors(arrays.begin(), arrays.end()));
}

}  // namespace

int main() {
  try {
    AdjointChecks();
    Checkerboards();
    std::cout << "All transposed-convolution checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
