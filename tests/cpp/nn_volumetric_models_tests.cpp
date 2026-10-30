#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace nn = dm::nn;
namespace m = nn::models;
using nn::Padding;
using nn::Tensor;
namespace check {
template <typename A, typename B> void Near(A a, B b, double t, const std::string &msg) {
  EXPECT_NEAR(double(a), double(b), t) << msg;
}
inline void Require(bool ok, const std::string &msg) {
  if (!ok)
    throw std::runtime_error(msg);
}
template <typename F> void Throws(F f, const std::string &msg) {
  EXPECT_ANY_THROW(f()) << msg;
}
} // namespace check
namespace {
std::vector<float> Fill(std::size_t n, const std::function<float(int)> &f) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = f(static_cast<int>(i));
  return v;
}

void ConvChecks() {
  for (const auto &spec : std::vector<std::pair<int, Padding>>{
           {1, Padding::Valid}, {2, Padding::Valid}, {2, Padding::Same}}) {
    const int stride = spec.first, d = 3, h = 4, w = 5, k = 2, ci = 2, co = 3;
    const bool same = spec.second == Padding::Same;
    const int od = same ? (d + stride - 1) / stride : (d - k) / stride + 1;
    const int oh = same ? (h + stride - 1) / stride : (h - k) / stride + 1,
              ow = same ? (w + stride - 1) / stride : (w - k) / stride + 1;
    const int pd = same ? std::max(0, (od - 1) * stride + k - d) / 2 : 0;
    const int ph = same ? std::max(0, (oh - 1) * stride + k - h) / 2 : 0,
              pw = same ? std::max(0, (ow - 1) * stride + k - w) / 2 : 0;
    const auto xv = Fill(std::size_t(d * h * w * ci), [](int i) {
      return std::sin(.13f * i);
    });
    const auto wv = Fill(std::size_t(k * k * k * ci * co), [](int i) {
      return .05f * std::sin(.17f * i);
    });
    const auto gv = Fill(std::size_t(od * oh * ow * co), [](int i) {
      return std::cos(.19f * i);
    });
    Tensor input = nn::from_vector(xv, {1, d, h, w, ci}),
           filter = nn::from_vector(wv, {k, k, k, ci, co});
    input.requires_grad_(true);
    filter.requires_grad_(true);
    const Tensor output = nn::conv3d(input, filter, nn::conv3d_options(stride, spec.second));
    const Tensor g = nn::from_vector(gv, {1, od, oh, ow, co});
    const auto gradient = nn::grad(nn::sum(output * g), {input, filter});
    check::Require(output.shape() == g.shape(), "3D convolution stride/padding shape");
    std::vector<double> y(gv.size(), 0), dx(xv.size(), 0), dw(wv.size(), 0);
    for (int z = 0; z < od; ++z)
      for (int r = 0; r < oh; ++r)
        for (int c = 0; c < ow; ++c)
          for (int o = 0; o < co; ++o) {
            const int yi = ((z * oh + r) * ow + c) * co + o;
            for (int kz = 0; kz < k; ++kz)
              for (int ky = 0; ky < k; ++ky)
                for (int kx = 0; kx < k; ++kx) {
                  const int iz = z * stride + kz - pd, iy = r * stride + ky - ph,
                            ix = c * stride + kx - pw;
                  if (iz < 0 || iy < 0 || ix < 0 || iz >= d || iy >= h || ix >= w)
                    continue;
                  for (int i = 0; i < ci; ++i) {
                    const int xi = ((iz * h + iy) * w + ix) * ci + i,
                              wi = (((kz * k + ky) * k + kx) * ci + i) * co + o;
                    y[yi] += double(xv[xi]) * wv[wi];
                    dx[xi] += double(gv[yi]) * wv[wi];
                    dw[wi] += double(gv[yi]) * xv[xi];
                  }
                }
          }
    double error = 0;
    const auto ov = output.to_vector<float>(), g0 = gradient[0].to_vector<float>(),
               g1 = gradient[1].to_vector<float>();
    for (std::size_t i = 0; i < ov.size(); ++i)
      error = std::max(error, std::abs(ov[i] - y[i]));
    for (std::size_t i = 0; i < g0.size(); ++i)
      error = std::max(error, std::abs(g0[i] - dx[i]));
    for (std::size_t i = 0; i < g1.size(); ++i)
      error = std::max(error, std::abs(g1[i] - dw[i]));
    check::Near(error, 0, 3e-6, "every 3D value/input/filter derivative matches host loops");
    std::cout << "  Conv3D stride" << stride << ' ' << (same ? "SAME" : "VALID")
              << ": host forward/every-gradient error=" << error << std::endl;
  }
}

void TransposeChecks() {
  nn::ParameterStore p(17);
  Tensor input = nn::from_vector(Fill(24,
                                      [](int i) {
                                        return .1f * i - .6f;
                                      }),
                                 {1, 2, 2, 3, 2});
  {
    nn::NoGradGuard guard;
    m::VolumeTranspose(p, "up", input, 3, true);
  }
  const auto weights = Fill(48, [](int i) {
    return .01f * i - .24f;
  });
  const auto gv = Fill(4 * 4 * 6 * 3, [](int i) {
    return .01f * (i - 100);
  });
  const std::vector<float> bias{.1f, -.2f, .3f};
  p.set("up/W", nn::from_vector(weights, {2, 2, 2, 3, 2}));
  p.set("up/b", nn::from_vector(bias, {3}));
  input.requires_grad_(true);
  const Tensor output = m::VolumeTranspose(p, "up", input, 3, true);
  const Tensor g = nn::from_vector(gv, {1, 4, 4, 6, 3});
  const auto gradient = nn::grad(nn::sum(output * g), {input, p.at("up/W"), p.at("up/b")});
  check::Require(output.shape() == g.shape(), "3D transpose doubles each spatial axis");
  const auto xv = input.to_vector<float>();
  std::vector<double> y(gv.size()), dx(xv.size(), 0), dw(weights.size(), 0), db(3, 0);
  for (std::size_t i = 0; i < gv.size(); ++i) {
    y[i] = bias[i % 3];
    db[i % 3] += gv[i];
  }
  for (int z = 0; z < 2; ++z)
    for (int r = 0; r < 2; ++r)
      for (int c = 0; c < 3; ++c)
        for (int i = 0; i < 2; ++i) {
          const int xi = ((z * 2 + r) * 3 + c) * 2 + i;
          for (int kz = 0; kz < 2; ++kz)
            for (int ky = 0; ky < 2; ++ky)
              for (int kx = 0; kx < 2; ++kx)
                for (int o = 0; o < 3; ++o) {
                  const int yi = (((2 * z + kz) * 4 + 2 * r + ky) * 6 + 2 * c + kx) * 3 + o,
                            wi = (((kz * 2 + ky) * 2 + kx) * 3 + o) * 2 + i;
                  y[yi] += double(xv[xi]) * weights[wi];
                  dx[xi] += double(gv[yi]) * weights[wi];
                  dw[wi] += double(gv[yi]) * xv[xi];
                }
        }
  double error = 0;
  const auto ov = output.to_vector<float>();
  for (std::size_t i = 0; i < ov.size(); ++i)
    error = std::max(error, std::abs(ov[i] - y[i]));
  const std::vector<const std::vector<double> *> refs{&dx, &dw, &db};
  for (std::size_t k = 0; k < 3; ++k) {
    const auto v = gradient[k].to_vector<float>();
    for (std::size_t i = 0; i < v.size(); ++i)
      error = std::max(error, std::abs(v[i] - (*refs[k])[i]));
  }
  check::Near(error, 0, 5e-6, "transpose scatter reference and every derivative");
  std::cout << "  Conv3D transpose: independent scatter/input/filter/bias derivative error="
            << error << std::endl;
}

void CropAndActivationChecks() {
  check::Require(m::Unet3DOutputSide(132) == 44 && m::Unet3DOutputSide(116) == 28 &&
                     m::Unet3DOutputSide(16, false) == 16,
                 "original anisotropic 3D U-Net geometry");
  check::Throws(
      [] {
        m::Unet3DOutputSide(64);
      },
      "too-small valid 3D input rejected");
  {
    Tensor x = nn::from_vector(Fill(630,
                                    [](int i) {
                                      return float(i - 300);
                                    }),
                               {1, 5, 7, 9, 2});
    x.requires_grad_(true);
    const Tensor crop = m::VolumeCrop(x, nn::zeros({1, 2, 4, 6, 3}));
    const Tensor gradient = nn::grad(nn::sum(crop), {x})[0];
    check::Require(crop.shape() == nn::Shape({1, 2, 4, 6, 2}),
                   "3D crop preserves source channels");
    const auto cv = crop.to_vector<float>(), xv = x.to_vector<float>(),
               gv = gradient.to_vector<float>();
    for (int z = 0; z < 2; ++z)
      for (int y = 0; y < 4; ++y)
        for (int xx = 0; xx < 6; ++xx)
          for (int c = 0; c < 2; ++c)
            check::Near(cv[(((z * 4 + y) * 6 + xx) * 2 + c)],
                        xv[((((z + 1) * 7 + y + 1) * 9 + xx + 1) * 2 + c)], 0,
                        "floor-centered signed voxel values");
    for (int z = 0; z < 5; ++z)
      for (int y = 0; y < 7; ++y)
        for (int xx = 0; xx < 9; ++xx)
          for (int c = 0; c < 2; ++c)
            check::Near(gv[(((z * 7 + y) * 9 + xx) * 2 + c)],
                        z >= 1 && z < 3 && y >= 1 && y < 5 && xx >= 1 && xx < 7 ? 1 : 0, 0,
                        "3D crop retains exactly its box Jacobian");
  }
  {
    nn::ParameterStore p(17);
    Tensor x = nn::from_vector(std::vector<float>{-2, 3, -4, -5, 0, -1}, {1, 1, 1, 3, 2});
    {
      nn::NoGradGuard guard;
      m::VolumePRelu(p, "act", x);
      m::VolumeResidual(p, "residual", x, 2, 2);
    }
    p.set("act/alpha", nn::from_vector(std::vector<float>{.25f, .5f}, {2}));
    for (const auto &name : p.names())
      if (name.rfind("residual/", 0) == 0 &&
          (name.substr(name.size() - 2) == "/W" || name.substr(name.size() - 2) == "/b"))
        p.set(name, nn::zeros(p.at(name).shape()));
    x.requires_grad_(true);
    const Tensor activation = m::VolumePRelu(p, "act", x),
                 residual = m::VolumeResidual(p, "residual", x, 2, 2);
    const auto ga = nn::grad(nn::sum(activation), {x, p.at("act/alpha")});
    const auto gr = nn::grad(nn::sum(residual), {x});
    const auto xv = x.to_vector<float>(), av = activation.to_vector<float>(),
               a0 = ga[0].to_vector<float>(), a1 = ga[1].to_vector<float>(),
               rv = residual.to_vector<float>(), r0 = gr[0].to_vector<float>();
    for (std::size_t i = 0; i < 6; ++i) {
      const double value = xv[i], a = i % 2 ? .5 : .25;
      check::Near(av[i], value >= 0 ? value : a * value, 0, "channel-specific PReLU forward");
      check::Near(a0[i], value >= 0 ? 1 : a, 0, "PReLU input slope including zero convention");
      check::Near(rv[i], value >= 0 ? value : .25 * value, 0,
                  "zero residual branch preserves shortcut before final PReLU");
      check::Near(r0[i], value >= 0 ? 1 : .25, 0, "residual shortcut derivative");
    }
    for (std::size_t i = 0; i < 2; ++i)
      check::Near(a1[i], -6, 0, "learned per-channel PReLU slope gradient");
  }
  std::cout << "  anisotropic valid geometry, signed 3D crop/Jacobian, PReLU slopes and "
               "residual shortcut fixtures passed"
            << std::endl;
}

void BatchNormChecks() {
  nn::ParameterStore p(17), state(18);
  const auto xv = Fill(48,
                       [](int i) {
                         return std::sin(.2f * i);
                       }),
             gv = Fill(48, [](int i) {
               return std::cos(.17f * i);
             });
  const std::vector<float> gamma{1.5f, .5f}, beta{-.3f, .2f};
  Tensor x = nn::from_vector(xv, {2, 2, 2, 3, 2});
  nn::BatchNormOptions o;
  o.momentum = .9, o.eps = 1e-5, o.update_stats = false;
  {
    nn::NoGradGuard guard;
    nn::batch_norm(p, state, "norm", x, true, o);
  }
  p.set("norm/gamma", nn::from_vector(gamma, {2}));
  p.set("norm/beta", nn::from_vector(beta, {2}));
  x.requires_grad_(true);
  const Tensor y = nn::batch_norm(p, state, "norm", x, true, o);
  const auto gradient = nn::grad(nn::sum(y * nn::from_vector(gv, {2, 2, 2, 3, 2})),
                                 {x, p.at("norm/gamma"), p.at("norm/beta")});
  const auto yv = y.to_vector<float>(), g0 = gradient[0].to_vector<float>(),
             g1 = gradient[1].to_vector<float>(), g2 = gradient[2].to_vector<float>();
  double error = 0;
  for (int c = 0; c < 2; ++c) {
    double mean = 0, var = 0, mg = 0, mgx = 0, dg = 0, db = 0;
    for (int i = 0; i < 24; ++i) {
      mean += xv[2 * i + c] / 24.;
      mg += gv[2 * i + c] / 24.;
    }
    for (int i = 0; i < 24; ++i) {
      const double centered = xv[2 * i + c] - mean;
      var += centered * centered / 24.;
      mgx += gv[2 * i + c] * centered / 24.;
    }
    for (int i = 0; i < 24; ++i) {
      const int j = 2 * i + c;
      const double centered = xv[j] - mean, scale = 1 / std::sqrt(var + 1e-5), gg = gv[j];
      error = std::max(error, std::abs(yv[j] - (gamma[c] * centered * scale + beta[c])));
      error = std::max(error, std::abs(g0[j] - gamma[c] * scale *
                                                   (gg - mg - centered * mgx / (var + 1e-5))));
      dg += gg * centered * scale;
      db += gg;
    }
    error = std::max(error, std::abs(g1[c] - dg));
    error = std::max(error, std::abs(g2[c] - db));
  }
  check::Near(error, 0, 2e-5, "all spatial/depth BN values/input/affine gradients");
  std::cout
      << "  volumetric BN over batch/depth/height/width: host value/every-gradient error="
      << error << std::endl;
}

void LossChecks() {
  {
    const Tensor weights = nn::from_vector(std::vector<float>{1, 2, 4}, {3});
    Tensor raw = nn::zeros({1, 1, 1, 3, 3});
    raw.requires_grad_(true);
    for (bool all_void : {false, true}) {
      const Tensor labels = nn::from_vector(all_void ? std::vector<int>{255, 255, 255}
                                                     : std::vector<int>{0, 2, 255},
                                            {1, 1, 1, 3});
      const Tensor loss = m::SparseVolumeCE(raw, labels, 3, weights);
      const auto g = nn::grad(loss, {raw})[0].to_vector<float>();
      if (!all_void) {
        check::Near(loss.item<double>(), std::log(3.), 1e-7, "weighted sparse CE reference");
        for (int i = 0; i < 3; ++i)
          for (int c = 0; c < 3; ++c)
            check::Near(g[3 * i + c],
                        i == 2 ? 0 : (1. / 3 - (c == (i ? 2 : 0))) * (i ? 4 : 1) / 5, 1e-7,
                        "all logit derivatives; unlabeled voxel exactly zero");
      } else {
        check::Near(loss.item<double>(), 0, 0, "all-unlabeled sparse loss finite zero");
        for (int i = 0; i < 9; ++i)
          check::Near(g[i], 0, 0, "all-unlabeled CE gradients zero");
      }
    }
  }
  {
    std::vector<float> pv{.2f, .5f, .9f, .7f, .1f, .4f};
    std::vector<int> tv{0, 1, 1, 1, 255, 0};
    const auto run = [&](std::vector<float> &probs, const std::vector<int> &truth,
                         double &loss, std::vector<float> &grad) {
      Tensor p = nn::from_vector(probs, {2, 1, 1, 3});
      p.requires_grad_(true);
      const Tensor l = m::SquaredVolumeDice(p, nn::from_vector(truth, {2, 1, 1, 3}));
      loss = l.item<double>();
      grad = nn::grad(l, {p})[0].to_vector<float>();
    };
    double loss;
    std::vector<float> grad;
    run(pv, tv, loss, grad);
    double expected = 0;
    for (int n = 0; n < 2; ++n) {
      double numerator = 1e-6, denominator = 1e-6;
      for (int i = 0; i < 3; ++i) {
        const int j = 3 * n + i;
        if (tv[j] == 255)
          continue;
        const double v = pv[j], g = tv[j];
        numerator += 2 * v * g;
        denominator += v * v + g * g;
      }
      expected += (1 - numerator / denominator) / 2;
      for (int i = 0; i < 3; ++i) {
        const int j = 3 * n + i;
        const double derivative = tv[j] == 255
                                      ? 0
                                      : -(2 * tv[j] * denominator - 2 * pv[j] * numerator) /
                                            (denominator * denominator) / 2;
        check::Near(grad[j], derivative, 2e-7, "every squared-Dice probability derivative");
      }
    }
    check::Near(loss, expected, 2e-7,
                "V-Net squared-denominator Dice differs from linear Dice");
    for (int i = 0; i < 6; ++i)
      pv[i] = tv[i] == 1 ? 1 : 0;
    run(pv, tv, loss, grad);
    check::Near(loss, 0, 0, "perfect Dice prediction zero loss");
    std::fill(tv.begin(), tv.end(), 255);
    run(pv, tv, loss, grad);
    check::Near(loss, 0, 0, "unlabeled Dice volumes excluded");
    for (int i = 0; i < 6; ++i)
      check::Near(grad[i], 0, 0, "unlabeled Dice gradients zero");
  }
  std::cout << "  sparse weighted CE, squared Dice every derivative/perfect/void cases "
               "and25%-slice data pairing passed"
            << std::endl;
}

void ModelChecks() {
  // Full-width models: parameters do not depend on the volume size or padding,
  // so they are executed on small SAME-padded volumes.
  for (int variant = 0; variant < 3; ++variant) {
    nn::ParameterStore p(17), state(18);
    Tensor logits;
    if (variant < 2) {
      m::Unet3DConfig cfg;
      cfg.batch_norm = variant == 1;
      cfg.valid = false;
      logits = m::Unet3D(p, state, nn::zeros({1, 8, 8, 8, 3}), cfg, true).logits;
      check::Require(p.count() == (variant == 0 ? 19069955 : 19072291),
                     "paper convolution census and explicit learned BN census");
    } else {
      logits = m::VNet(p, nn::zeros({1, 16, 16, 16, 1})).logits;
      check::Require(p.count() == 65175314,
                     "V-Net independent 5x5 residual stage/channel census");
    }
    const auto gradient = nn::grad(nn::sum(nn::square(logits)), p.parameters());
    std::size_t defined = 0;
    for (const auto &g : gradient)
      defined += g.defined();
    check::Require(gradient.size() == std::size_t(variant == 0   ? 33
                                                  : variant == 1 ? 47
                                                                 : 89) &&
                       defined == gradient.size(),
                   "every full volumetric parameter connected");
    std::cout << "  full "
              << (variant == 2   ? "V-Net"
                  : variant == 1 ? "3D U-Net+affine BN"
                                 : "3D U-Net convolution census")
              << ": " << p.count() << " parameters, " << gradient.size()
              << " derivatives (executed on a small SAME volume)" << std::endl;
  }
  for (bool vnet : {false, true}) {
    nn::ParameterStore p(17), state(18);
    const Tensor input = nn::from_vector(Fill(4096,
                                              [](int i) {
                                                return std::sin(.13f * i);
                                              }),
                                         {1, 16, 16, 16, 1});
    m::VolumeOutputs net;
    if (vnet) {
      m::VNetConfig cfg;
      cfg.base = 2;
      net = m::VNet(p, input, cfg);
    } else {
      m::Unet3DConfig cfg;
      cfg.base = 2, cfg.input_channels = 1, cfg.classes = 2, cfg.valid = false;
      net = m::Unet3D(p, state, input, cfg, true);
    }
    check::Require(p.count() == (vnet ? 1019348 : 74828),
                   "compact volumetric independent parameter census");
    check::Require(net.logits.shape() == nn::Shape({1, 16, 16, 16, 2}),
                   "dense voxel output alignment");
    for (std::size_t i = 0; i < net.encoder.size(); ++i) {
      const std::int64_t side = 16 >> i;
      const auto &s = net.encoder[i].shape();
      check::Require(s[1] == side && s[2] == side && s[3] == side,
                     "all depth/height/width scales preserved");
    }
    const auto gradient = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
    double low = INFINITY, high = 0;
    for (std::size_t i = 0; i < gradient.size(); ++i) {
      const double norm = std::sqrt(nn::sum(nn::square(gradient[i])).item<double>());
      check::Require(std::isfinite(norm) && norm > 0,
                     "finite nonzero volumetric derivative: " + p.names()[i]);
      low = std::min(low, norm);
      high = std::max(high, norm);
    }
    std::cout << "  compact " << (vnet ? "V-Net" : "3D U-Net") << ": " << p.count()
              << " parameters; every stage/derivative passed, norms [" << low << ',' << high
              << ']' << std::endl;
  }
}

} // namespace
TEST(NnVolumetric, IndependentEveryConv3DValueInputFilterGradientStridePadding) {
  ConvChecks();
}
TEST(NnVolumetric, IndependentEveryTransposeValueInputKernelBiasGradient) {
  TransposeChecks();
}
TEST(NnVolumetric, CroppedGeometryEveryInputGradientPreluSlopeAndResidualIdentity) {
  CropAndActivationChecks();
}
TEST(NnVolumetric, IndependentEveryVolumetricBatchNormValueInputAffineGradient) {
  BatchNormChecks();
}
TEST(NnVolumetric, IndependentEverySparseCeSquaredDiceGradientPerfectAndVoid) {
  LossChecks();
}
TEST(NnVolumetric, FullIndependentCensusEveryGradientCompactEveryNonzeroStageGradient) {
  nn::set_num_threads(2);
  ModelChecks();
}
TEST(NnVolumetric, RepeatableIndependentTensorVolumesSparseAxisAndRowPairing) {
  const auto data = m::MakeTensorEllipsoidVolumes(3, 16, 11),
             repeat = m::MakeTensorEllipsoidVolumes(3, 16, 11),
             other = m::MakeTensorEllipsoidVolumes(3, 16, 22);
  EXPECT_TRUE(nn::all(nn::equal(data.images, repeat.images)).item<bool>());
  EXPECT_TRUE(nn::all(nn::equal(data.labels, repeat.labels)).item<bool>());
  EXPECT_TRUE(nn::any(nn::not_equal(data.images, other.images)).item<bool>());
  const auto batch = m::TensorVolumeBatch(data, {2, 0}, true);
  const auto image = batch.first.to_vector<float>(), original = data.images.to_vector<float>();
  const auto labels = batch.second.to_vector<int>(), truth = data.labels.to_vector<int>();
  int annotated = 0;
  for (int n = 0; n < 2; ++n)
    for (int i = 0; i < 4096; ++i) {
      const int row = n ? 0 : 2;
      EXPECT_EQ(image[n * 4096 + i], original[row * 4096 + i]);
      const int expected = (i / 256) % 4 == 2 ? truth[row * 4096 + i] : 255;
      EXPECT_EQ(labels[n * 4096 + i], expected);
      annotated += expected != 255;
    }
  EXPECT_EQ(annotated, 2048);
  EXPECT_EQ(m::TensorVolumeBatch(data, {}).first.shape(), nn::Shape({0, 16, 16, 16, 1}));
}
TEST(NnVolumetric, VolumeMeanOverlapGlobalAccuracyAndOrthogonalSliceGeometry) {
  const auto truth = nn::from_vector(std::vector<std::int64_t>{0, 0, 0, 0, 1, 1, 0, 0},
                                     {2, 1, 2, 2}),
             pred = nn::from_vector(std::vector<std::int64_t>{0, 0, 0, 0, 1, 0, 1, 0},
                                    {2, 1, 2, 2});
  const auto score = m::VolumePredictionMetrics(pred, truth);
  EXPECT_DOUBLE_EQ(score.dice.item(), .75);
  EXPECT_DOUBLE_EQ(score.iou.item(), (1. + 1. / 3) / 2);
  EXPECT_DOUBLE_EQ(score.accuracy.item(), .75);
  EXPECT_EQ(score.tp.item<std::int64_t>(), 1);
  EXPECT_EQ(score.fp.item<std::int64_t>(), 1);
  EXPECT_EQ(score.fn.item<std::int64_t>(), 1);
  EXPECT_EQ(score.tn.item<std::int64_t>(), 5);
  const auto z = nn::arange(0, 16).reshape({1, 16, 1, 1, 1}),
             y = nn::arange(0, 16).reshape({1, 1, 16, 1, 1}),
             x = nn::arange(0, 16).reshape({1, 1, 1, 16, 1});
  m::TensorVolumeData data{(z * 256. + y * 16. + x) / 4096. * 2.4 - 1.2,
                           nn::zeros({1, 16, 16, 16}, nn::DType::Int64)};
  const auto preview = m::VolumeSlicePreview(data);
  EXPECT_EQ(preview.shape(), nn::Shape({3, 16, 32, 3}));
  const auto values = preview.to_vector<float>();
  for (int plane = 0; plane < 3; ++plane)
    for (int r = 0; r < 16; ++r)
      for (int c = 0; c < 16; ++c) {
        const int zz = plane == 0 ? 8 : r,
                  yy = plane == 1   ? 8
                       : plane == 0 ? r
                                    : c,
                  xx = plane == 2 ? 8 : c;
        const double intensity = std::floor((zz * 256 + yy * 16 + xx) / 4096. * 255) / 255.;
        EXPECT_NEAR(values[((plane * 16 + r) * 32 + c) * 3], intensity, 1 / 255. + 1e-7);
      }
}

TEST(NnVolumetric, FitOnlyExactAnnotatedCountsAndEmptyClassWeights) {
  const auto labels = nn::from_vector(std::vector<std::int64_t>{0, 0, 1, 255});
  const auto weight = m::SparseVolumeClassWeights(labels).to_vector<float>();
  EXPECT_NEAR(weight[0], 1 / std::sqrt(3. / 5), 1e-7);
  EXPECT_NEAR(weight[1], 1 / std::sqrt(2. / 5), 1e-7);
  const auto empty =
      m::SparseVolumeClassWeights(nn::full({4}, 255., nn::DType::Int64)).to_vector<float>();
  for (const auto v : empty)
    EXPECT_NEAR(v, std::sqrt(2.), 1e-7);
}
