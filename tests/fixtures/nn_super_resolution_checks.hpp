// Independent numeric/architecture checks; no experiment driver.
#pragma once
#include "nn_interpretation_check_support.hpp"
namespace flow {
using namespace dm::nn::models;
}
namespace pose {
using namespace dm::nn::models;
}
namespace sr {
using namespace dm::nn::models;
}
namespace vqa {
using namespace dm::nn::models;
}
namespace dream {
using namespace dm::nn::models;
}
namespace featurevis {
using namespace dm::nn::models;
}
namespace cam {
using namespace dm::nn::models;
}
namespace guided {
using namespace dm::nn::models;
}
namespace ig {
using namespace dm::nn::models;
}
namespace nn = datamunge::nn;
using nn::Tensor;
using namespace sr;

namespace {

const std::vector<SRKind> All =
    {SRKind::kSrcnn, SRKind::kEspcn, SRKind::kVdsr, SRKind::kEdsr, SRKind::kRcan};

void ShuffleChecks() {
  for (int scale : {2, 3, 4}) {
    const int          channels = 2, depth = channels * scale * scale;
    std::vector<float> input(std::size_t(2 * 2 * 3 * depth)),
        upstream(std::size_t(2 * 2 * scale * 3 * scale * channels));
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = float(int(i) - 50);
    for (std::size_t i = 0; i < upstream.size(); ++i)
      upstream[i] = float(int(i) % 17 - 7);
    Tensor     x = nn::from_vector(input, {2, 2, 3, depth}).requires_grad_();
    const auto y = SRPixelShuffle(x, channels, scale);
    const auto g =
        nn::grad(
            nn::sum(y * nn::from_vector(upstream, {2, 2 * scale, 3 * scale, channels})), {x})[0]
            .to_vector<float>();
    check::Require(
        y.shape() == nn::Shape({2, 2 * scale, 3 * scale, channels}),
        "shuffle rectangular output alignment");
    const auto out = y.contiguous().to_vector<float>();
    const int  H = 2 * scale, W = 3 * scale;
    for (int b = 0; b < 2; ++b)
      for (int h = 0; h < 2; ++h)
        for (int w = 0; w < 3; ++w)
          for (int c = 0; c < channels; ++c)
            for (int dy = 0; dy < scale; ++dy)
              for (int dx = 0; dx < scale; ++dx) {
                const int packed = c * scale * scale + dy * scale + dx;
                const int o      = ((b * H + h * scale + dy) * W + w * scale + dx) * channels + c,
                          i      = ((b * 2 + h) * 3 + w) * depth + packed;
                check::Near(out[o], input[i], 0, "every signed CRD coordinate");
                check::Near(g[i], upstream[o], 0, "every shuffle input Jacobian entry");
              }
  }
  // nn::depth_to_space is TF's DCR order: it differs from CRD when channels > 1.
  std::vector<float> v(16);
  std::iota(v.begin(), v.end(), 0.f);
  const auto crd = SRPixelShuffle(nn::from_vector(v, {1, 1, 2, 8}), 2, 2).to_vector<float>();
  const auto dcr = nn::depth_to_space(nn::from_vector(v, {1, 1, 2, 8}), 2).to_vector<float>();
  check::Require(crd != dcr, "CRD shuffle is not TF DCR DepthToSpace for multi-channel inputs");
  const auto one =
      nn::depth_to_space(
          nn::from_vector(std::vector<float>(v.begin(), v.begin() + 8), {1, 1, 2, 4}), 2)
          .to_vector<float>();
  check::Require(
      one == SRPixelShuffle(
                 nn::from_vector(std::vector<float>(v.begin(), v.begin() + 8), {1, 1, 2, 4}), 1, 2)
                 .to_vector<float>(),
      "single-channel CRD equals DepthToSpace");
  std::cout << "  x2/x3/x4 CRD shuffle: every signed multi-batch/multi-channel coordinate and "
               "derivative passed"
            << std::endl;
}

void DataMetricChecks() {
  const auto enlarged =
      SRBicubic(nn::from_vector(std::vector<float>{0, 0, 1, 0, 0}, {1, 1, 5, 1}), 1, 10)
          .to_vector<float>();
  const double expected[] = {
      0, -.0234375, -.0703125, .2265625, .8671875, .8671875, .2265625, -.0703125, -.0234375, 0};
  for (int i = 0; i < 10; ++i)
    check::Near(enlarged[i], expected[i], 0, "independent Catmull-Rom impulse/half-pixel phase");
  std::vector<float> cv(2 * 5 * 7 * 3);
  for (int b = 0; b < 2; ++b)
    for (int y = 0; y < 5; ++y)
      for (int x = 0; x < 7; ++x)
        for (int c = 0; c < 3; ++c)
          cv[std::size_t(((b * 5 + y) * 7 + x) * 3 + c)] = .1f * float(b + c + 1);
  const Tensor constant = nn::from_vector(cv, {2, 5, 7, 3});
  const auto   small    = SRBicubic(constant, 3, 4).to_vector<float>(),
             identity   = SRBicubic(constant, 5, 7).to_vector<float>();
  check::Require(identity == cv, "bicubic identity preserves every pixel exactly");
  for (int b = 0; b < 2; ++b)
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 4; ++x)
        for (int c = 0; c < 3; ++c)
          check::Near(
              small[std::size_t(((b * 3 + y) * 4 + x) * 3 + c)],
              cv[std::size_t(((b * 5) * 7) * 3 + c)],
              1e-7,
              "odd rectangular constant/DC preservation");
  std::vector<float> checker(64);
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x)
      checker[std::size_t(y * 8 + x)] = float((x + y) % 2);
  for (float v : SRBicubic(nn::from_vector(checker, {1, 8, 8, 1}), 4, 4).to_vector<float>())
    check::Require(std::abs(v - .5) < .1, "antialias kernel suppresses Nyquist checkerboard");
  check::Near(SRLuma(0, 0, 0), 16. / 255, 0, "studio Y black");
  check::Near(SRLuma(1, 1, 1), 235. / 255, 1e-15, "studio Y white");
  check::Near(SRLuma(1, 0, 0), (16 + 65.481) / 255, 1e-15, "BT.601 red coefficient");
  std::vector<float> coordinates(32 * 34);
  for (int y = 0; y < 32; ++y)
    for (int x = 0; x < 34; ++x)
      coordinates[std::size_t(y * 34 + x)] = float(100 * y + x);
  const auto crop = SRCrop(nn::from_vector(coordinates, {1, 32, 34, 1}), 6);
  check::Require(
      crop.shape() == nn::Shape({1, 20, 22, 1}), "SRCNN six-pixel rectangular target border");
  const auto cropv = crop.to_vector<float>();
  for (int y = 0; y < 20; ++y)
    for (int x = 0; x < 22; ++x)
      check::Near(
          cropv[std::size_t(y * 22 + x)],
          100 * (y + 6) + x + 6,
          0,
          "every target alignment coordinate");
  check::Near(SRPSNR(.01), 20, 1e-12, "known PSNR");
  check::Near(SRPSNR(1), 0, 0, "unit MSE PSNR");
  check::Require(std::isinf(SRPSNR(0)), "perfect reconstruction has infinite PSNR");
  std::vector<double> a(400, .2), b(400, .3);
  check::Near(SRSSIM(a, a, 20, 20), 1, 1e-12, "identical constant SSIM");
  check::Near(
      SRSSIM(a, b, 20, 20),
      (.12 + .0001) / (.13 + .0001),
      1e-12,
      "independent constant SSIM luminance term");
  nn::data::CifarDataset    source;
  std::vector<std::int32_t> pixels(4 * 3072);
  for (int i = 0; i < 4 * 3072; ++i)
    pixels[std::size_t(i)] = (i / 3072 * 37 + i % 3072) % 256;
  source.images  = nn::from_vector(pixels, {4, 32, 32, 3});
  source.labels  = nn::from_vector(std::vector<std::int32_t>(4, 0), {4});
  const auto all = SRPairs(source.images, {0, 1, 2, 3}), pair = SRPairs(source.images, {3, 1});
  check::Require(pair.lr.shape() == nn::Shape({2, 16, 16, 3}), "real x2 low-resolution grid");
  const auto ah = all.hr.to_vector<float>(), al = all.lr.to_vector<float>(),
             ph = pair.hr.to_vector<float>(), pl = pair.lr.to_vector<float>();
  for (int n = 0; n < 2; ++n) {
    const int row = n ? 1 : 3;
    for (int i = 0; i < 3072; ++i)
      check::Near(
          ph[std::size_t(n * 3072 + i)], ah[std::size_t(row * 3072 + i)], 0, "reordered HR rows");
    for (int i = 0; i < 768; ++i)
      check::Near(
          pl[std::size_t(n * 768 + i)],
          al[std::size_t(row * 768 + i)],
          0,
          "reordered LR/HR pairing");
  }
  std::vector<double> checker_a(400), checker_b(400);
  for (int y = 0; y < 20; ++y)
    for (int x = 0; x < 20; ++x) {
      checker_a[std::size_t(y * 20 + x)] = (x + y) % 2;
      checker_b[std::size_t(y * 20 + x)] = 1 - checker_a[std::size_t(y * 20 + x)];
    }
  double mass = 0, alternating = 0;
  for (int k = 0; k < 11; ++k) {
    const double weight = std::exp(-(k - 5) * (k - 5) / 4.5);
    mass += weight;
    alternating += (k % 2 ? -1 : 1) * weight;
  }
  const double q = std::pow(alternating / mass, 2), ma = (1 + q) / 2, mb = 1 - ma,
               variance = ma * mb;
  const double opposite = ((2 * ma * mb + .0001) / (ma * ma + mb * mb + .0001)) *
                          ((-2 * variance + .0009) / (2 * variance + .0009));
  check::Near(
      SRSSIM(checker_a, checker_b, 20, 20),
      opposite,
      1e-12,
      "analytic inverse-checkerboard SSIM variance/covariance, not only constant means");
  const auto perfect = SRScore(all.hr, all), perfect_crop = SRScore(SRCrop(all.hr, 6), all);
  check::Require(
      std::isinf(perfect.psnr) && std::isinf(perfect_crop.psnr),
      "full and cropped perfect reconstruction both have infinite PSNR");
  check::Near(perfect.ssim, 1, 1e-12, "full RGB perfect SSIM");
  check::Near(perfect_crop.ssim, 1, 1e-12, "common-crop perfect SSIM");
  std::cout << "  independent bicubic phase/DC/antialias, exact crop/Y, PSNR/SSIM "
               "variance/perfect/crop and reordered LR-HR fixtures passed"
            << std::endl;
}

void AttentionChecks() {
  nn::ParameterStore       p(17);
  std::vector<float>       input(48), upstream(48), wr(8), we(8);
  const std::vector<float> br = {.1f, .2f}, be = {-.1f, .2f, .05f, -.3f};
  for (int i = 0; i < 48; ++i)
    input[std::size_t(i)]    = .1f + .01f * float(i),
    upstream[std::size_t(i)] = .1f * std::sin(float(i) * .23f);
  for (int i = 0; i < 8; ++i)
    wr[std::size_t(i)] = .1f + .01f * float(i), we[std::size_t(i)] = .05f * float(i - 3);
  {
    nn::NoGradGuard guard;
    SRChannelAttention(p, "ca", nn::zeros({2, 2, 3, 4}), 4, 2);
    p.set("ca/reduce/W", nn::from_vector(wr, {1, 1, 4, 2}));
    p.set("ca/reduce/b", nn::from_vector(br, {2}));
    p.set("ca/expand/W", nn::from_vector(we, {1, 1, 2, 4}));
    p.set("ca/expand/b", nn::from_vector(be, {4}));
  }
  check::Require(p.size() == 4, "attention MLP has exactly four tensors");
  Tensor     x         = nn::from_vector(input, {2, 2, 3, 4}).requires_grad_();
  const auto attention = SRChannelAttention(p, "ca", x, 4, 2);
  const auto gradient  = nn::grad(
      nn::sum(attention.output * nn::from_vector(upstream, {2, 2, 3, 4})),
      {x, p.at("ca/reduce/W"), p.at("ca/reduce/b"), p.at("ca/expand/W"), p.at("ca/expand/b")});
  const auto outv = attention.output.to_vector<float>(), gates = attention.gates.to_vector<float>();
  std::vector<std::vector<float>> g;
  for (const auto& t : gradient)
    g.push_back(t.to_vector<float>());
  double dwr[8] = {}, dbr[2] = {}, dwe[8] = {}, dbe[4] = {}, error = 0;
  for (int b = 0; b < 2; ++b) {
    double mean[4] = {}, hidden[2] = {}, gv[4] = {}, dz[4] = {}, dh[2] = {};
    for (int c = 0; c < 4; ++c)
      for (int m = 0; m < 6; ++m)
        mean[c] += input[std::size_t((b * 6 + m) * 4 + c)] / 6.;
    for (int j = 0; j < 2; ++j) {
      hidden[j] = br[std::size_t(j)];
      for (int c = 0; c < 4; ++c)
        hidden[j] += mean[c] * wr[std::size_t(c * 2 + j)];
      check::Require(hidden[j] > 0, "active reference attention hidden units");
    }
    for (int c = 0; c < 4; ++c) {
      double z = be[std::size_t(c)], dg = 0;
      for (int j = 0; j < 2; ++j)
        z += hidden[j] * we[std::size_t(j * 4 + c)];
      gv[c] = 1 / (1 + std::exp(-z));
      check::Near(gates[std::size_t(b * 4 + c)], gv[c], 1e-7, "GAP/MLP/sigmoid gate value");
      for (int m = 0; m < 6; ++m) {
        const int index = (b * 6 + m) * 4 + c;
        check::Near(
            outv[std::size_t(index)],
            input[std::size_t(index)] * gv[c],
            1e-7,
            "gated feature value");
        dg += input[std::size_t(index)] * upstream[std::size_t(index)];
      }
      dz[c] = dg * gv[c] * (1 - gv[c]);
      dbe[c] += dz[c];
      for (int j = 0; j < 2; ++j) {
        dwe[j * 4 + c] += hidden[j] * dz[c];
        dh[j] += we[std::size_t(j * 4 + c)] * dz[c];
      }
    }
    for (int j = 0; j < 2; ++j) {
      dbr[j] += dh[j];
      for (int c = 0; c < 4; ++c)
        dwr[c * 2 + j] += mean[c] * dh[j];
    }
    for (int c = 0; c < 4; ++c)
      for (int m = 0; m < 6; ++m) {
        double dx = upstream[std::size_t((b * 6 + m) * 4 + c)] * gv[c];
        for (int j = 0; j < 2; ++j)
          dx += wr[std::size_t(c * 2 + j)] * dh[j] / 6;
        error = std::max(error, std::abs(g[0][std::size_t((b * 6 + m) * 4 + c)] - dx));
      }
  }
  for (int i = 0; i < 8; ++i) {
    error = std::max(error, std::abs(g[1][std::size_t(i)] - dwr[i]));
    error = std::max(error, std::abs(g[3][std::size_t(i)] - dwe[i]));
  }
  for (int i = 0; i < 2; ++i)
    error = std::max(error, std::abs(g[2][std::size_t(i)] - dbr[i]));
  for (int i = 0; i < 4; ++i)
    error = std::max(error, std::abs(g[4][std::size_t(i)] - dbe[i]));
  check::Near(error, 0, 1e-7, "all input and attention weight/bias derivatives");
  std::cout << "  independent RCAN attention values and every GAP/MLP/input derivative error="
            << error << std::endl;
}

void ResidualClipChecks() {
  nn::ParameterStore p(17);
  {
    nn::NoGradGuard guard;
    SRResidual(p, "block", nn::zeros({1, 3, 4, 2}), 2, .1);
    for (const auto& name : p.names())
      p.set(name, nn::full(p.at(name).shape(), name == "block/conv2/b" ? .25 : 0.));
  }
  std::vector<float> input(24);
  for (int i = 0; i < 24; ++i)
    input[std::size_t(i)] = .1f * float(i - 12);
  Tensor     x  = nn::from_vector(input, {1, 3, 4, 2}).requires_grad_();
  const auto y  = SRResidual(p, "block", x, 2, .1);
  const auto g  = nn::grad(nn::sum(y), {x, p.at("block/conv2/b")});
  const auto yv = y.to_vector<float>(), gx = g[0].to_vector<float>(), gb = g[1].to_vector<float>();
  for (int i = 0; i < 24; ++i) {
    check::Near(
        yv[std::size_t(i)],
        input[std::size_t(i)] + .025f,
        1e-7,
        "signed residual/.1 scaling, no post-add ReLU");
    check::Near(gx[std::size_t(i)], 1, 0, "residual shortcut Jacobian");
  }
  for (int i = 0; i < 2; ++i)
    check::Near(gb[std::size_t(i)], 1.2, 2e-7, "scaled residual bias derivative");
  const Tensor gradient    = nn::from_vector(std::vector<float>{-2.f, -.2f, 0.f, .2f, 2.f});
  const auto   clipped     = VdsrClippedGradient(gradient, .1).to_vector<float>(),
             lower_rate    = VdsrClippedGradient(gradient, .01).to_vector<float>();
  const double expected1[] = {-.1, -.1, 0, .1, .1}, expected2[] = {-1, -.2, 0, .2, 1};
  for (int i = 0; i < 5; ++i) {
    check::Near(clipped[std::size_t(i)], expected1[i], 1e-7, "theta/lr clipping");
    check::Near(lower_rate[std::size_t(i)], expected2[i], 1e-7, "clipping bound adjusts with rate");
    check::Require(
        std::abs(clipped[std::size_t(i)] * .1) <= .01000001 &&
            std::abs(lower_rate[std::size_t(i)] * .01) <= .01000001,
        "clipped current gradient contribution bounded at both rates");
  }
  std::cout << "  scaled signed residual/shortcut/bias Jacobians and learning-rate-adjustable "
               "gradient clipping passed"
            << std::endl;
}

std::vector<float> Fixture(std::int64_t n) {
  std::vector<float> v(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i)
    v[std::size_t(i)] = float(.4 + .2 * std::sin(float(i) * .137f));
  return v;
}

void ModelChecks() {
  const std::int64_t full_counts[]        = {8129, 21284, 665921, 40729603, 15444643},
                     compact_counts[]     = {8129, 21284, 42065, 49603, 45851};
  const std::size_t full_derivatives[]    = {6, 6, 40, 136, 1628},
                    compact_derivatives[] = {6, 6, 40, 40, 60};
  int index                               = 0;
  for (auto kind : All) {
    {
      nn::ParameterStore p(17);
      const auto         cfg  = SRConfig::Full(kind);
      const int          side = cfg.PreUpsampled() ? 33 : 5;
      const Tensor       x    = nn::from_vector(
          Fixture(side * (side + 2) * cfg.Channels()), {1, side, side + 2, cfg.Channels()});
      const auto net      = SuperResolution(p, x, cfg);
      const auto gradient = nn::grad(nn::mean(nn::square(net.image)), p.parameters());
      check::Require(
          p.count() == full_counts[index] && gradient.size() == full_derivatives[index],
          "independent full SR parameter/derivative census");
      for (const auto& name : p.names())
        check::Require(
            name.find("gamma") == std::string::npos && name.find("beta") == std::string::npos &&
                name.find("bn") == std::string::npos && name.find("moving") == std::string::npos,
            "SR architectures have no batch normalization");
      for (const auto& g : gradient)
        check::Require(g.defined(), "full SR every derivative");
      std::cout << "  full " << SRName(kind) << ": " << p.count() << " parameters/"
                << gradient.size() << " derivatives (executed)" << std::endl;
    }
    nn::ParameterStore p(17);
    const auto         cfg      = SRConfig::Compact(kind);
    const int          side     = cfg.PreUpsampled() ? 33 : 7;
    const auto         input    = Fixture(2 * side * (side + 2) * cfg.Channels());
    const Tensor       x        = nn::from_vector(input, {2, side, side + 2, cfg.Channels()});
    const auto         net      = SuperResolution(p, x, cfg);
    const auto         gradient = nn::grad(nn::mean(nn::square(net.image)), p.parameters());
    check::Require(
        p.count() == compact_counts[index] && gradient.size() == compact_derivatives[index],
        "compact independent parameter census");
    const int oh = kind == SRKind::kSrcnn ? side - 12
                   : cfg.PreUpsampled()   ? side
                                          : side * 2,
              ow = kind == SRKind::kSrcnn ? side + 2 - 12
                   : cfg.PreUpsampled()   ? side + 2
                                          : (side + 2) * 2;
    check::Require(
        net.image.shape() == nn::Shape({2, oh, ow, cfg.Channels()}),
        "rectangular model-specific HR alignment");
    for (float v : net.image.to_vector<float>())
      check::Require(std::isfinite(v), "finite unclipped raw reconstruction");
    const std::size_t expected_stages = kind == SRKind::kSrcnn   ? 2
                                        : kind == SRKind::kEspcn ? 3
                                        : kind == SRKind::kVdsr  ? 19
                                        : kind == SRKind::kEdsr  ? std::size_t(cfg.blocks + 3)
                                                                 : std::size_t(cfg.groups + 3);
    check::Require(
        net.stages.size() == expected_stages,
        "all architecturally required intermediate stages present");
    for (std::size_t i = 0; i < net.stages.size(); ++i) {
      int h = side, w = side + 2, c = cfg.width;
      if (kind == SRKind::kSrcnn)
        h -= 8, w -= 8, c = i == 0 ? 64 : 32;
      else if (kind == SRKind::kEspcn)
        c = i == 0 ? 64 : i == 1 ? 32 : cfg.scale * cfg.scale;
      else if ((kind == SRKind::kEdsr || kind == SRKind::kRcan) && i + 1 == net.stages.size())
        h *= cfg.scale, w *= cfg.scale;
      check::Require(
          net.stages[i].shape() == nn::Shape({2, h, w, c}),
          "every exact SR batch/spatial/channel intermediate shape");
    }
    double low = INFINITY, high = 0;
    int    inactive = 0;
    for (std::size_t i = 0; i < gradient.size(); ++i) {
      double norm = 0;
      for (float v : gradient[i].to_vector<float>())
        norm += double(v) * v;
      norm = double(float(std::sqrt(norm)));
      check::Require(std::isfinite(norm), "every SR parameter derivative finite");
      if (norm == 0) {
        check::Require(
            kind == SRKind::kRcan && p.names()[i].find("/ca/") != std::string::npos,
            "only inactive attention MLP branches may have zero derivatives");
        ++inactive;
      } else
        low = std::min(low, norm), high = std::max(high, norm);
    }
    {
      nn::NoGradGuard guard;
      for (const auto& name : p.names())
        p.set(name, nn::zeros(p.at(name).shape()));
      const auto out = SuperResolution(p, x, cfg).image.to_vector<float>();
      for (std::size_t i = 0; i < out.size(); ++i) {
        double expected = 0;
        if (kind == SRKind::kVdsr)
          expected = input[i];
        if (cfg.Channels() == 3)
          expected = kSRMean[i % 3];
        check::Near(
            out[i],
            expected,
            0,
            "only VDSR has an image skip; others predict directly/mean-shift only");
      }
    }
    std::cout << "  compact " << SRName(kind) << ": " << p.count() << " parameters/all"
              << gradient.size() << " evaluated gradients, norms[" << low << ',' << high
              << "], inactive attention tensors=" << inactive
              << "; stage/output geometry and true skip semantics passed" << std::endl;
    ++index;
  }
}

} // namespace
