// Independent numeric and model-geometry checks; experiment driver omitted.
#pragma once
#include "nn_interpretation_check_support.hpp"
namespace inpaint {
using namespace dm::nn::models;
}
namespace denoise {
using namespace dm::nn::models;
}
namespace flow {
using namespace dm::nn::models;
}
namespace color {
using namespace dm::nn::models;
}
namespace depth {
using namespace dm::nn::models;
}
namespace pose {
using namespace dm::nn::models;
}
namespace style {
using namespace dm::nn::models;
}
namespace crnn {
using namespace dm::nn::models;
}
namespace ocr {
using namespace dm::nn::models;
}
namespace cifar {
using namespace dm::nn::models;
}

#include <cmath>
#include <iostream>
namespace nn = datamunge::nn;
using nn::Tensor;
using namespace inpaint;

namespace {

void PartialConvChecks() {
  for (int stride : {1, 2}) {
    constexpr int      b = 2, h = 3, w = 4, in = 2, outc = 2, k = 3;
    const int          oh = (h + stride - 1) / stride, ow = (w + stride - 1) / stride;
    std::vector<float> values(b * h * w * in), valid(values.size()), filter(k * k * in * outc),
        upstream(b * oh * ow * outc);
    for (int i = 0; i < int(values.size()); ++i)
      values[i] = float((i - 17) * .07), valid[i] = float(i < h * w * in && i % 3 != 0);
    for (int i = 0; i < int(filter.size()); ++i)
      filter[i] = float((i % 13 - 6) * .03);
    for (int i = 0; i < int(upstream.size()); ++i)
      upstream[i] = float((i % 7 - 3) * .11);
    const std::vector<float> offset = {-.2f, .4f};
    Tensor                   x      = nn::from_vector(values, {b, h, w, in}).requires_grad_(),
           weights                  = nn::from_vector(filter, {k, k, in, outc}).requires_grad_(),
           bias                     = nn::from_vector(offset).requires_grad_();
    const Tensor mask               = nn::from_vector(valid, {b, h, w, in});
    const auto   p                  = PartialConv(x, mask, weights, bias, in, outc, k, stride);
    const auto   derivative         = nn::grad(
        nn::sum(p.value * nn::from_vector(upstream, {b, oh, ow, outc})), {x, weights, bias});
    const auto pv = p.value.to_vector<double>(), pm = p.mask.to_vector<double>(),
               pc = p.count.to_vector<double>();
    const auto gx = derivative[0].to_vector<double>(), gw = derivative[1].to_vector<double>(),
               gb = derivative[2].to_vector<double>();
    std::vector<double> dx(values.size()), dw(filter.size()), db(outc);
    double              value_error = 0;
    const int           py          = std::max(0, (oh - 1) * stride + k - h) / 2,
              px                    = std::max(0, (ow - 1) * stride + k - w) / 2;
    for (int n = 0; n < b; ++n)
      for (int y = 0; y < oh; ++y)
        for (int col = 0; col < ow; ++col)
          for (int o = 0; o < outc; ++o) {
            double count = 0, sum = 0;
            for (int ky = 0; ky < k; ++ky)
              for (int kx = 0; kx < k; ++kx)
                for (int c = 0; c < in; ++c) {
                  const int iy = y * stride + ky - py, ix = col * stride + kx - px;
                  if (iy < 0 || iy >= h || ix < 0 || ix >= w)
                    continue;
                  const int j = ((n * h + iy) * w + ix) * in + c,
                            z = ((ky * k + kx) * in + c) * outc + o;
                  count += valid[j];
                  sum += values[j] * valid[j] * filter[z];
                }
            const double ratio    = count ? k * k * in / count : 0,
                         expected = count ? sum * ratio + offset[o] : 0;
            const int q           = ((n * oh + y) * ow + col) * outc + o;
            value_error           = std::max(value_error, std::abs(expected - pv[q]));
            check::Near(pm[q], count > 0, 0, "binary updated mask includes empty-window rule");
            check::Near(
                pc[q], count, 0, "multi-channel window count and asymmetric SAME stride alignment");
            if (!count)
              continue;
            const double g = upstream[q];
            db[o] += g;
            for (int ky = 0; ky < k; ++ky)
              for (int kx = 0; kx < k; ++kx)
                for (int c = 0; c < in; ++c) {
                  const int iy = y * stride + ky - py, ix = col * stride + kx - px;
                  if (iy < 0 || iy >= h || ix < 0 || ix >= w)
                    continue;
                  const int j = ((n * h + iy) * w + ix) * in + c,
                            z = ((ky * k + kx) * in + c) * outc + o;
                  dx[j] += g * ratio * filter[z] * valid[j];
                  dw[z] += g * ratio * values[j] * valid[j];
                }
          }
    double gradient_error = 0;
    for (std::size_t i = 0; i < dx.size(); ++i)
      gradient_error = std::max(gradient_error, std::abs(dx[i] - gx[i]));
    for (std::size_t i = 0; i < dw.size(); ++i)
      gradient_error = std::max(gradient_error, std::abs(dw[i] - gw[i]));
    for (std::size_t i = 0; i < db.size(); ++i)
      gradient_error = std::max(gradient_error, std::abs(db[i] - gb[i]));
    check::Near(value_error, 0, 2e-6, "every independently normalized partial-convolution value");
    check::Near(gradient_error, 0, 3e-6, "every partial-convolution parameter/input derivative");
    auto filled = values;
    for (std::size_t i = 0; i < filled.size(); ++i)
      if (valid[i] == 0)
        filled[i] += 1000;
    const auto again = PartialConv(
        nn::from_vector(filled, {b, h, w, in}),
        mask,
        weights.detach(),
        bias.detach(),
        in,
        outc,
        k,
        stride);
    check::Require(
        again.value.to_vector<float>() == p.value.to_vector<float>(),
        "finite hidden values never influence partial convolution");
    std::cout << " partial convolution stride" << stride
              << ":independent signed multi-channel values/count/mask/input-kernel-bias gradients "
                 "max errors="
              << value_error << "/" << gradient_error
              << ",empty windows,bias and hidden-fill invariance passed\n";
  }
}

void DataCompositeChecks() {
  std::vector<std::int32_t> raw(4 * 3072);
  for (int i = 0; i < int(raw.size()); ++i)
    raw[i] = (i * 13 + i / 3072 * 41) % 256;
  const Tensor source = nn::from_vector(raw, {4, 32, 32, 3});
  const auto   all = ContextPairs(source, {0, 1, 2, 3}), pair = ContextPairs(source, {3, 1});
  const auto   am = all.masked.to_vector<float>(), pm = pair.masked.to_vector<float>(),
             pp = pair.patch.to_vector<float>(), pt = pair.truth.to_vector<float>();
  for (int n = 0; n < 2; ++n)
    for (int y = 0; y < 32; ++y)
      for (int x = 0; x < 32; ++x)
        for (int c = 0; c < 3; ++c) {
          const int row = n ? 1 : 3;
          check::Near(
              pm[((n * 32 + y) * 32 + x) * 3 + c],
              am[((row * 32 + y) * 32 + x) * 3 + c],
              0,
              "reordered masked RGB pairing");
          if (y >= 8 && y < 24 && x >= 8 && x < 24)
            check::Near(
                pp[((n * 16 + y - 8) * 16 + x - 8) * 3 + c],
                2 * pt[((n * 32 + y) * 32 + x) * 3 + c] - 1,
                0,
                "every central target coordinate");
        }
  auto changed = raw;
  for (int n = 0; n < 4; ++n)
    for (int y = 8; y < 24; ++y)
      for (int x = 8; x < 24; ++x)
        for (int c = 0; c < 3; ++c)
          changed[((n * 32 + y) * 32 + x) * 3 + c] ^= 255;
  const auto hidden = ContextPairs(nn::from_vector(changed, {4, 32, 32, 3}), {0, 1, 2, 3});
  check::Require(
      hidden.masked.to_vector<float>() == am &&
          hidden.patch.to_vector<float>() != all.patch.to_vector<float>(),
      "changing ground-truth holes never leaks to masked input");
  check::Require(
      ContextFillMean(hidden).to_vector<float>() == ContextFillMean(all).to_vector<float>(),
      "context-mean baseline uses visible pixels only");
  const auto perfect = InpaintScore(all, all.truth);
  check::Require(
      perfect.hole_mse == 0 && std::isinf(perfect.hole_psnr),
      "perfect hole metric is infinite PSNR");
  const auto         zero = ContextPairs(nn::zeros({4, 32, 32, 3}, nn::DType::Int32), {0, 1});
  std::vector<float> estimate(2 * 3072, 0.f);
  for (int n = 0; n < 2; ++n)
    for (int y = 8; y < 24; ++y)
      for (int x = 8; x < 24; ++x)
        for (int c = 0; c < 3; ++c)
          estimate[((n * 32 + y) * 32 + x) * 3 + c] = n ? .2f : .1f;
  const auto score = InpaintScore(zero, nn::from_vector(estimate, {2, 32, 32, 3}));
  check::Near(score.hole_mse, .025, 1e-8, "MSE restricted to hole RGB samples");
  check::Near(score.hole_psnr, (20 - 20 * std::log10(.2)) / 2, 1e-6, "mean per-image hole PSNR");
  check::Near(
      score.full_psnr - score.hole_psnr,
      10 * std::log10(4.),
      1e-12,
      "full-image PSNR inflated by75% known context");
  Tensor input = pair.masked.clone().requires_grad_(), patch = pair.patch.clone().requires_grad_();
  const Tensor output   = ContextComposite(input, patch, ContextMask());
  const auto   gradient = nn::grad(nn::sum(output), {input, patch});
  const auto   ov = output.to_vector<float>(), g0 = gradient[0].to_vector<float>(),
             g1 = gradient[1].to_vector<float>();
  for (int n = 0; n < 2; ++n)
    for (int y = 0; y < 32; ++y)
      for (int x = 0; x < 32; ++x)
        for (int c = 0; c < 3; ++c) {
          const bool hole = y >= 8 && y < 24 && x >= 8 && x < 24;
          const int  i    = ((n * 32 + y) * 32 + x) * 3 + c;
          check::Near(
              ov[i],
              hole ? pp[((n * 16 + y - 8) * 16 + x - 8) * 3 + c] : pm[i],
              0,
              "composite exact observed pixels and central patch alignment");
          check::Near(g0[i], !hole, 0, "composite input derivative at every coordinate");
        }
  for (float v : g1)
    check::Near(v, 1, 0, "composite patch derivative at every coordinate");
  std::cout << " central mask/target alignment,hidden-ground-truth independence,visible-only "
               "baseline,hole/full PSNR,exact composite and every input/patch derivative passed\n";
}

void AdversarialLossChecks() {
  const std::vector<float> value  = {-80, -3, -.2f, 0, .3f, 4, 80};
  Tensor                   logits = nn::from_vector(value).requires_grad_();
  const Tensor             real = ContextBCE(logits, true), fake = ContextBCE(logits, false);
  const auto               dr = nn::grad(real, {logits})[0].to_vector<double>(),
             df               = nn::grad(fake, {logits})[0].to_vector<double>();
  double expected_real = 0, expected_fake = 0, error = 0;
  for (int i = 0; i < 7; ++i) {
    const double z = value[i], softplus = std::max(z, 0.) + std::log1p(std::exp(-std::abs(z))),
                 sigmoid = 1 / (1 + std::exp(-z));
    expected_real += (softplus - z) / 7, expected_fake += softplus / 7;
    error = std::max(error, std::abs(dr[i] - (sigmoid - 1) / 7));
    error = std::max(error, std::abs(df[i] - sigmoid / 7));
  }
  check::Near(real.item(), expected_real, 3e-6, "stable real BCE including extreme logits");
  check::Near(fake.item(), expected_fake, 3e-6, "stable fake BCE including extreme logits");
  check::Near(error, 0, 2e-8, "every discriminator/generator BCE logit derivative");
  std::cout << " independent stable GAN losses/extreme logits/every logit derivative max error="
            << error << " passed\n";
}

void ModelChecks() {
  for (bool compact : {false, true}) {
    nn::ParameterStore g(71), gs(72), d(73), ds(74);
    const auto         config = compact ? ContextConfig::Compact() : ContextConfig{};
    std::vector<float> values(std::size_t(4) * config.side * config.side * 3);
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = float(.8 * std::sin(i * .17));
    const Tensor input          = nn::from_vector(values, {4, config.side, config.side, 3});
    const auto   net            = ContextEncoder(g, gs, input, config, true, false);
    const auto   disc           = ContextDiscriminator(d, ds, net.patch, config, true, false);
    const auto   frozen         = ContextDiscriminator(d, ds, net.patch, config, false);
    const Tensor reconstruction = nn::mean(nn::square(net.patch));
    const Tensor adversarial    = ContextBCE(frozen.logits, true),
                 dloss          = ContextBCE(disc.logits, false);
    const auto gg               = nn::grad(reconstruction, g.parameters(), {true}),
               dg               = nn::grad(dloss, d.parameters(), {true}),
               ag               = nn::grad(adversarial, g.parameters());
    check::Require(
        g.count() == (compact ? 22523 : 7221231) && d.count() == (compact ? 2745 : 2766529) &&
            gs.count() == (compact ? 160 : 4040) && ds.count() == (compact ? 32 : 1792),
        "exact full/compact context GAN/BN census");
    std::cout << " " << (compact ? "compact" : "full") << " context G/D params=" << g.count() << "/"
              << d.count() << ",G/D derivatives=" << gg.size() << "/" << dg.size()
              << ",BN values=" << gs.count() << "/" << ds.count() << " (executed)\n";
    if (!compact)
      continue;
    const std::vector<nn::Shape> shapes = {
        {4, 16, 16, 8},
        {4, 8, 8, 8},
        {4, 4, 4, 16},
        {4, 1, 1, 32},
        {4, 4, 4, 16},
        {4, 8, 8, 8},
        {4, 16, 16, 3},
        {4, 8, 8, 8},
        {4, 4, 4, 16},
        {4}};
    std::vector<Tensor> stages = net.stages;
    stages.insert(stages.end(), disc.stages.begin(), disc.stages.end());
    check::Require(stages.size() == shapes.size(), "all compact G/D stages exposed");
    for (std::size_t i = 0; i < shapes.size(); ++i)
      check::Require(
          stages[i].shape() == shapes[i], "exact context G/D batch/spatial/channel stage geometry");
    double              minimum = INFINITY, maximum = 0, adversarial_norm = 0;
    int                 zeros = 0;
    std::vector<Tensor> all   = gg;
    all.insert(all.end(), dg.begin(), dg.end());
    all.insert(all.end(), ag.begin(), ag.end());
    for (std::size_t i = 0; i < all.size(); ++i) {
      double norm = 0;
      for (float v : all[i].to_vector<float>()) {
        check::Require(std::isfinite(v), "finite every context GAN parameter derivative");
        norm += double(v) * v;
      }
      if (i >= gg.size() + dg.size())
        adversarial_norm += norm;
      norm = std::sqrt(norm);
      if (norm == 0)
        ++zeros;
      else
        minimum = std::min(minimum, norm);
      maximum = std::max(maximum, norm);
    }
    check::Require(adversarial_norm > 1e-14, "nonzero adversarial gradient reaches generator");
    std::cout
        << " all compact G24/D8 gradients plus24 adversarial-chain gradients finite,norm range["
        << minimum << "," << maximum << "],exact-zero tensors=" << zeros
        << ",adversarial norm=" << std::sqrt(adversarial_norm) << ";all10 stage shapes passed\n";
  }
}

} // namespace
