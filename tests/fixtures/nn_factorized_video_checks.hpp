#pragma once
// Independent numerical and architecture regression checks retained from the
// original executable; canonical operations are called directly. No training
// driver or example-source dependency is retained here.
#include <dm/dm.hpp>
#include <iostream>
namespace {
namespace nn = dm::nn;
namespace fv = dm::nn::models;
using nn::Padding;
using nn::Tensor;
namespace check {
inline void Require(bool value, const std::string& message) {
  if (!value)
    throw std::runtime_error(message);
}
inline void Near(double a, double b, double tolerance, const std::string& message) {
  if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a - b) > tolerance)
    throw std::runtime_error(message);
}
} // namespace check
struct HostVolume {
  int                 t, h, w, c;
  std::vector<double> x;
};

HostVolume HostFactorConv(
    const HostVolume&          input,
    const std::vector<double>& kernel,
    int                        kt,
    int                        kh,
    int                        kw,
    int                        out,
    int                        st = 1,
    int                        sh = 1,
    int                        sw = 1) {
  HostVolume y{(input.t + st - 1) / st, (input.h + sh - 1) / sh, (input.w + sw - 1) / sw, out, {}};
  y.x.resize(static_cast<std::size_t>(y.t * y.h * y.w * out));
  for (int t = 0; t < y.t; ++t)
    for (int h = 0; h < y.h; ++h)
      for (int w = 0; w < y.w; ++w)
        for (int c = 0; c < out; ++c) {
          double sum = 0;
          for (int a = 0; a < kt; ++a)
            for (int b = 0; b < kh; ++b)
              for (int d = 0; d < kw; ++d)
                for (int ci = 0; ci < input.c; ++ci) {
                  const int it = t * st + a - kt / 2, ih = h * sh + b - kh / 2,
                            iw = w * sw + d - kw / 2;
                  if (it >= 0 && it < input.t && ih >= 0 && ih < input.h && iw >= 0 && iw < input.w)
                    sum += input.x[static_cast<std::size_t>(
                               ((it * input.h + ih) * input.w + iw) * input.c + ci)] *
                           kernel[static_cast<std::size_t>(
                               (((a * kh + b) * kw + d) * input.c + ci) * out + c)];
                }
          y.x[static_cast<std::size_t>(((t * y.h + h) * y.w + w) * out + c)] = sum;
        }
  return y;
}

std::vector<float> Fill(std::size_t n, double scale, double frequency, double phase, bool cosine) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double a = frequency * static_cast<double>(i) + phase;
    v[i]           = static_cast<float>(scale * (cosine ? std::cos(a) : std::sin(a)));
  }
  return v;
}

void FactorChecks() {
  int    total         = 0;
  double worst_forward = 0, worst_derivative = 0;
  for (int variant = 0; variant < 4; ++variant) {
    const bool r2   = variant == 0;
    const int  side = r2 ? 4 : 3, mid = r2 ? fv::FactorMidplanes(2, 2) : 2, stride = r2 ? 2 : 1;
    const auto kind    = static_cast<fv::P3dKind>(r2 ? 0 : variant - 1);
    const auto input   = Fill(static_cast<std::size_t>(side * side * side * 2), .3, .37, .2, false);
    const auto spatial = Fill(static_cast<std::size_t>(9 * 2 * mid), .13, .23, .4, true);
    const auto temporal = Fill(static_cast<std::size_t>(3 * mid * 2), .17, .41, .9, false);
    std::vector<std::vector<double>> values(3);
    values[0].assign(spatial.begin(), spatial.end());
    values[1].assign(temporal.begin(), temporal.end());
    values[2].assign(input.begin(), input.end());
    Tensor x = nn::from_vector(input, {1, side, side, side, 2}),
           a = nn::from_vector(spatial, {1, 3, 3, 2, mid}),
           b = nn::from_vector(temporal, {3, 1, 1, mid, 2});
    for (Tensor* t : {&x, &a, &b})
      t->requires_grad_(true);
    const auto sf = [&](const Tensor& v) {
      return nn::relu(fv::FactorConv(v, a, 1, stride, stride));
    };
    const auto tfun = [&](const Tensor& v) {
      const Tensor z = fv::FactorConv(v, b, stride, 1, 1);
      return kind == fv::P3dKind::A ? nn::relu(z) : z;
    };
    const Tensor y       = fv::P3dMix(x, sf, tfun, kind);
    const int    outside = (side + stride - 1) / stride;
    const auto   coefficient =
        Fill(static_cast<std::size_t>(outside * outside * outside * 2), .2, .19, 0, true);
    const Tensor loss =
        nn::sum(y * nn::from_vector(coefficient, {1, outside, outside, outside, 2}));
    const auto g    = nn::grad(loss, {a, b, x});
    const auto host = [&](const std::vector<std::vector<double>>& v,
                          std::vector<double>*                    out    = nullptr,
                          bool                                    linear = false) {
      const HostVolume volume{side, side, side, 2, v[2]};
      auto             z = HostFactorConv(volume, v[0], 1, 3, 3, mid, 1, stride, stride);
      if (!linear)
        for (auto& q : z.x)
          q = std::max(0., q);
      auto t = HostFactorConv(kind == fv::P3dKind::B ? volume : z, v[1], 3, 1, 1, 2, stride, 1, 1);
      if (kind == fv::P3dKind::A) {
        if (!linear)
          for (auto& q : t.x)
            q = std::max(0., q);
      } else {
        for (std::size_t i = 0; i < t.x.size(); ++i)
          t.x[i] = std::max(0., (z.x[i] + t.x[i]) * (kind == fv::P3dKind::C ? .5 : 1));
      }
      double value = 0;
      for (std::size_t i = 0; i < t.x.size(); ++i)
        value += t.x[i] * coefficient[i];
      if (out)
        *out = t.x;
      return value;
    };
    std::vector<double> expected;
    host(values, &expected);
    const auto yv = y.to_vector<float>();
    check::Require(yv.size() == expected.size(), "factorized fixture output size");
    double forward = 0, derivative = 0;
    for (std::size_t i = 0; i < expected.size(); ++i)
      forward = std::max(forward, std::abs(expected[i] - yv[i]));
    int covered = 0;
    for (std::size_t v = 0; v < 3; ++v) {
      const auto gv = g[v].to_vector<float>();
      for (std::size_t i = 0; i < values[v].size(); ++i) {
        auto plus = values, minus = values;
        plus[v][i] += 1e-6;
        minus[v][i] -= 1e-6;
        derivative = std::max(derivative, std::abs((host(plus) - host(minus)) / 2e-6 - gv[i]));
        ++covered;
      }
    }
    check::Near(
        forward,
        0,
        1e-7,
        "every signed factorized output versus independent host convolution/topology");
    check::Near(
        derivative,
        0,
        1e-7,
        "ALL spatial/temporal/input coordinates independent host finite differences");
    if (r2) {
      std::vector<double> linear;
      host(values, &linear, true);
      double difference = 0;
      for (std::size_t i = 0; i < linear.size(); ++i)
        difference += std::abs(linear[i] - expected[i]);
      check::Require(
          difference > .001,
          "intermediate ReLU makes R(2+1)D more than a collapsed linear3D kernel");
    }
    total += covered;
    worst_forward    = std::max(worst_forward, forward);
    worst_derivative = std::max(worst_derivative, derivative);
    std::cout << (r2             ? "R(2+1)D stride2"
                  : variant == 1 ? "P3D-A"
                  : variant == 2 ? "P3D-B"
                                 : "P3D-C")
              << " signed fixture:all" << covered
              << " factor/input FD coordinates,forward=" << forward << ",gradient=" << derivative
              << std::endl;
  }
  check::Require(total == 530, "224R2+3x102P3D coordinates covered");
  check::Require(
      fv::FactorMidplanes(64, 64) == 144 && fv::FactorMidplanes(64, 128) == 230,
      "independent parameter-balance widths including floor");
  std::cout << "all530 factor/kernel/input derivatives passed;explicit symmetric stride2 borders "
               "and nonlinear routing "
               "verified;worst="
            << worst_forward << "/" << worst_derivative << std::endl;
}

/// Endpoint shapes from the padding rules alone (explicit k/2 borders; P3D's -inf ceil and temporal
/// pools).
std::vector<nn::Shape> ShapeTrace(bool p3d, int base, int image) {
  const auto conv = [](std::int64_t in, std::int64_t k, std::int64_t s) {
    return nn::conv_output_size(in, k, s, 1, Padding::Explicit, k / 2, k / 2);
  };
  std::vector<nn::Shape> shapes;
  std::int64_t           t = 16, s = conv(image, 7, 2), c = base;
  if (!p3d) {
    shapes.push_back({1, t, s, s, c});
    for (int stage = 0; stage < 4; ++stage)
      for (int block = 0; block < 2; ++block) {
        const int stride = stage && block == 0 ? 2 : 1;
        s                = conv(s, 3, stride);
        t                = conv(t, 3, stride);
        c                = std::int64_t{base} << stage;
        shapes.push_back({1, t, s, s, c});
      }
  } else {
    t = nn::conv_output_size(t, 2, 2, 1, Padding::Valid);
    s = nn::conv_output_size(s + 1, 3, 2, 1, Padding::Valid);
    shapes.push_back({1, t, s, s, c});
    const int depth[] = {3, 8, 36, 3};
    for (int stage = 0; stage < 4; ++stage) {
      if (stage)
        t = nn::conv_output_size(t, 2, 2, 1, Padding::Valid);
      for (int block = 0; block < depth[stage]; ++block) {
        if (stage && block == 0)
          s = conv(s, 1, 2);
        c = 4 * (std::int64_t{base} << stage);
        shapes.push_back({1, t, s, s, c});
      }
    }
  }
  return shapes;
}

void GraphChecks(bool run_full) {
  for (bool p3d : {false, true})
    for (bool compact : {false, true}) {
      const int base = compact ? 4 : 64, classes = compact ? 3 : 400,
                reference = compact ? 32
                            : p3d   ? 160
                                    : 112;
      const int image =
          compact || run_full ? reference : 32; // full widths execute on a 16x32x32 clip by default
      const int depths[] = {3, 8, 36, 3};
      // The published endpoint table, at the executed size and at the reference size.
      const auto expected_endpoints = [&](int side) {
        std::vector<nn::Shape> shapes;
        shapes.push_back({1, p3d ? 8 : 16, side / (p3d ? 4 : 2), side / (p3d ? 4 : 2), base});
        for (int stage = 0; stage < 4; ++stage)
          for (int block = 0; block < (p3d ? depths[stage] : 2); ++block)
            shapes.push_back(
                {1,
                 p3d ? (8 >> stage) : (16 >> stage),
                 side / (p3d ? (4 << stage) : (2 << stage)),
                 side / (p3d ? (4 << stage) : (2 << stage)),
                 base * (p3d ? 4 : 1) * (1 << stage)});
        return shapes;
      };
      check::Require(
          ShapeTrace(p3d, base, reference) == expected_endpoints(reference),
          "published reference/compact residual endpoint shapes from the padding rules");
      nn::ParameterStore p(733), state(735);
      nn::Generator      gen(5);
      const Tensor       input = nn::uniform({1, 16, image, image, 3}, 0.0, 1.0, gen);
      const auto         net   = fv::FactorVideo(p, state, input, {p3d, base, classes});
      const auto         table = expected_endpoints(image);
      check::Require(
          net.stages.size() == table.size() &&
              net.stages.size() == static_cast<std::size_t>(p3d ? 51 : 9),
          "all50P3D/eightR2 residual endpoints plusstem");
      for (std::size_t i = 0; i < table.size(); ++i)
        check::Require(net.stages[i].shape() == table[i], "executed residual endpoint shape");
      check::Require(
          net.pooled.shape() == nn::Shape({1, base * (p3d ? 32 : 8)}) &&
              net.logits.shape() == nn::Shape({1, classes}),
          "pooled/logit shapes");
      std::int64_t expected = 0;
      int          units    = 0;
      const auto   count    = [&](int in, int out, int volume) {
        expected += std::int64_t{in} * out * volume + 2 * out;
        ++units;
      };
      int incoming = base;
      if (!p3d) {
        const int middle = std::max(1, 45 * base / 64);
        count(3, middle, 49);
        count(middle, base, 3);
        for (int stage = 0; stage < 4; ++stage)
          for (int block = 0; block < 2; ++block) {
            const int out = base << stage;
            const int mid =
                static_cast<int>(std::int64_t{27} * incoming * out / (9 * incoming + 3 * out));
            count(incoming, mid, 9);
            count(mid, out, 3);
            count(out, mid, 9);
            count(mid, out, 3);
            if (stage && block == 0)
              count(incoming, out, 1);
            incoming = out;
          }
      } else {
        count(3, base, 49);
        for (int stage = 0; stage < 4; ++stage)
          for (int block = 0; block < depths[stage]; ++block) {
            const int width = base << stage, out = 4 * width;
            count(incoming, width, 1);
            count(width, width, 9);
            if (stage < 3)
              count(width, width, 3);
            count(width, out, 1);
            if (incoming != out)
              count(incoming, out, 1);
            incoming = out;
          }
      }
      expected += std::int64_t{incoming} * classes + classes;
      check::Require(
          expected == p.count() && units == (p3d ? 202 : 37) &&
              p.size() == static_cast<std::size_t>(3 * units + 2) &&
              state.size() == static_cast<std::size_t>(2 * units),
          "independent per-kernel/learnedBN/head parameter inventory");
      if (!p3d && !compact)
        check::Require(expected == 31505325, "Torchvision published R(2+1)D-18 parameter count");
      if (p3d && !compact)
        check::Require(
            expected == 66492240,
            "independently parsed authors Caffe Kinetics P3D-199 parameter count");
      const auto gradients = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
      check::Require(
          gradients.size() == p.size(), "all full/compact factorized-video learned gradients");
      for (const auto& g : gradients) {
        check::Require(g.defined(), "no missing factorized video parameter gradient");
        check::Require(
            nn::all(nn::is_finite(g)).item<double>() != 0, "finite factorized video gradient");
      }
      std::cout << (p3d ? "P3D-199" : "R(2+1)D-18") << (compact ? " compact" : " full reference")
                << " graph:" << expected << " learned parameters/" << p.size() << " learned+"
                << state.size() << " state tensors;all" << table.size()
                << " endpoint shapes/count/all gradients passed;"
                << (compact ? "executed 16x32x32"
                    : image == reference
                        ? "full width executed at reference size"
                        : "full width executed on 16x32x32 (reference-size table from padding "
                          "rules; full=1 executes it)")
                << "\n";
    }
}

/// Loads TF's trained compact arrays (DIR/{r2plus1d,p3d}/model_arrays.npz) into our
/// models and replays TF's saved test clips: an exact cross-framework forward check.

} // namespace
