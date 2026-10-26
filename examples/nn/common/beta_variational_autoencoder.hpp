// Port of tflow's common/nn/beta_variational_autoencoder.h and
// common/ml/beta_variational_autoencoder.h: the compact 4-D beta-weighted
// Gaussian VAE, its explicit 4-D noise banks, the tiny known-factor lattice
// (circle/square x 3 sizes x 7 x x 7 y = 294 binary images, tflow's `just
// beta-vae-assets`, $DATAMUNGE_DATA_DIR/beta_vae/factors.npz) and the fit-only
// heldout factor probes. Many 2.2 ports (adversarial, flow, gumbel, hierarchical,
// importance-weighted, factor/TC VAEs) build on these pieces.
#pragma once

#include <array>
#include <fstream>
#include <iostream>

#include "common/autoencoder.hpp"

namespace ae {

// ---- model ------------------------------------------------------------------------------

struct BetaVaeOutput : VaeOutput {
  nn::Tensor unweighted_loss;
};

/// 784 -> 32 ReLU -> (mu, log variance) [4] each; z = mu + exp(lv / 2) eps;
/// 4 -> 32 ReLU -> 784 logits. loss = mean(NLL + beta KL), unweighted_loss = mean(NLL + KL).
inline BetaVaeOutput BetaVariationalAutoencoder(nn::ParameterStore& p, const nn::Tensor& image,
                                                const nn::Tensor& epsilon, float beta) {
  check::Require(std::isfinite(beta) && beta > 0, "finite positive beta");
  BetaVaeOutput out;
  auto          h      = nn::dense(p, "encoder", image, 32, nn::Activation::Relu);
  out.mu                 = nn::dense(p, "mu", h, 4);
  out.log_variance       = nn::dense(p, "log_variance", h, 4);
  out.z                  = VaeReparameterize(out.mu, out.log_variance, epsilon);
  out.logits             = nn::dense(p, "output", nn::dense(p, "decoder", out.z, 32, nn::Activation::Relu), 784);
  out.probability        = nn::sigmoid(out.logits);
  out.reconstruction_nll = VaePixelNll(out.logits, image);
  out.kl                 = VaeKl(out.mu, out.log_variance);
  out.unweighted_loss    = nn::mean(out.reconstruction_nll + out.kl, {0});
  out.loss               = nn::mean(out.reconstruction_nll + double(beta) * out.kl, {0});
  return out;
}

/// Copies named arrays into a store (every parameter must be supplied with its shape).
inline void LoadArrays(nn::ParameterStore& store, const nn::NamedTensors& arrays) {
  check::Require(arrays.size() == store.size(), "all initial model tensors supplied");
  for (const auto& name : store.names()) {
    const auto value = Find(arrays, name);
    check::Require(value.dtype() == nn::DType::Float32 && value.shape() == store.at(name).shape(),
                   "matching initial shape/dtype");
    store.set(name, value.contiguous().clone());
  }
}

class BetaVariationalAutoencoderModel {
 public:
  nn::ParameterStore             parameters{2237};
  std::unique_ptr<nn::Optimizer> optimizer;
  float                          beta;

  explicit BetaVariationalAutoencoderModel(float beta_ = 1, bool training = true) : beta(beta_) {
    {
      nn::NoGradGuard guard;
      BetaVariationalAutoencoder(parameters, nn::zeros({1, 784}), nn::zeros({1, 4}), beta);
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  static nn::Tensor Flatten(const nn::Tensor& images) { return nn::reshape(images, {images.size(0), 784}); }
  /// {mu, log_variance, z, probability, reconstruction_nll, kl, weighted loss, unweighted loss}, all finite.
  std::vector<nn::Tensor> Evaluate(const nn::Tensor& images, const nn::Tensor& noise) {
    nn::NoGradGuard         guard;
    auto                    n = BetaVariationalAutoencoder(parameters, Flatten(images), noise, beta);
    std::vector<nn::Tensor> out{n.mu, n.log_variance, n.z, n.probability, n.reconstruction_nll, n.kl, n.loss,
                                n.unweighted_loss};
    for (auto& t : out) {
      t = t.contiguous();
      for (float v : t.to_vector<float>())
        check::Require(std::isfinite(v), "ALL posterior/output/objective values finite");
    }
    return out;
  }
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    return nn::sigmoid(nn::dense(parameters, "output", nn::dense(parameters, "decoder", codes, 32, nn::Activation::Relu),
                                 784))
        .contiguous();
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& images, const nn::Tensor& noise) {
    return nn::grad(BetaVariationalAutoencoder(parameters, Flatten(images), noise, beta).loss,
                    parameters.parameters());
  }
  /// One Adam step; returns the pre-update weighted objective.
  double Update(const nn::Tensor& images, const nn::Tensor& noise) {
    auto         loss  = BetaVariationalAutoencoder(parameters, Flatten(images), noise, beta).loss;
    const double value = loss.item<float>();
    check::Require(std::isfinite(value) && value >= 0, "finite negativeELBO");
    optimizer->minimize(loss);
    return value;
  }
  void Load(const nn::NamedTensors& arrays) { LoadArrays(parameters, arrays); }
};

// ---- explicit 4-D noise ----------------------------------------------------------------

/// Banks of `samples` x [n, width] standard normals (float64 + its float32 cast).
inline VaeNoiseBank GaussianBank(std::int64_t n, std::int64_t width, std::int64_t samples, std::uint32_t seed) {
  auto standard = GaussianNoise({n, width}, samples, seed);
  return {standard, nn::cast(standard, nn::DType::Float32).contiguous()};
}
inline VaeNoiseBank BetaVaeNoise(std::int64_t n, std::int64_t samples, std::uint32_t seed) {
  return GaussianBank(n, 4, samples, seed);
}
inline nn::Tensor BetaVaeNoiseAt(const VaeNoiseBank& bank, std::int64_t k) {
  check::Require(k >= 0 && k < bank.epsilon.size(0), "valid Gaussian sample index");
  return nn::select(bank.epsilon, 0, k).contiguous();
}

/// Monte Carlo {NLL, KL, negative ELBO, NLL + beta KL} per image over every sample of `bank`.
inline std::vector<double> BetaVaeScore(BetaVariationalAutoencoderModel& model, const nn::Tensor& images,
                                        const VaeNoiseBank& bank, float beta) {
  double             reconstruction = 0, kl = 0;
  const std::int64_t n = images.size(0), samples = bank.epsilon.size(0);
  for (std::int64_t k = 0; k < samples; ++k) {
    auto       out = model.Evaluate(images, BetaVaeNoiseAt(bank, k));
    const auto nll = out[4].to_vector<float>(), kls = out[5].to_vector<float>();
    for (std::int64_t b = 0; b < n; ++b) {
      reconstruction += nll[b];
      if (k == 0) kl += kls[b];
    }
  }
  reconstruction /= double(n * samples);
  kl /= double(n);
  return {reconstruction, kl, reconstruction + kl, reconstruction + beta * kl};
}

// ---- known-factor lattice ----------------------------------------------------------------

struct BetaFactorData {
  nn::Tensor images, factors, source;  // float [n, 28, 28, 1] binary; int32 [n, 4]; int32 [n]
};

inline BetaFactorData Rows(const BetaFactorData& d, const std::vector<std::int64_t>& rows) {
  return {nn::take_rows(d.images, rows), nn::take_rows(d.factors, rows), nn::take_rows(d.source, rows)};
}

/// The 210 fit / 42 validation / 42 test pools with tflow's exhaustive regeneration checks.
inline std::vector<BetaFactorData> LoadBetaFactors(const std::filesystem::path& root) {
  const auto file = root / "factors.npz";
  check::Require(std::filesystem::exists(file),
                 file.string() + " missing; run `just beta-vae-assets` in tflow_examples or pass data=");
  const auto                  a = nn::load_npz(file.string());
  std::vector<BetaFactorData> pools;
  std::set<int>               ids;
  std::set<std::string>       pixels;
  for (const std::string split : {"fit", "validation", "test"}) {
    const int n   = split == "fit" ? 210 : 42;
    auto      raw = Find(a, split + "_images").contiguous(), factors = Find(a, split + "_factors").contiguous(),
         source = Find(a, split + "_source_ids").contiguous();
    check::Require(raw.shape() == nn::Shape({n, 28, 28, 1}) && factors.dtype() == nn::DType::Int32 &&
                       factors.shape() == nn::Shape({n, 4}) && source.dtype() == nn::DType::Int32 &&
                       source.shape() == nn::Shape({n}),
                   "exact synthetic factor dataset schema");
    const auto         r = raw.to_vector<int>(), f = factors.to_vector<int>(), s = source.to_vector<int>();
    std::vector<float> images(r.size());
    int                counts[4][7] = {};
    for (int b = 0; b < n; ++b) {
      int v[4];
      for (int j = 0; j < 4; ++j) {
        v[j]                = f[b * 4 + j];
        const int cardinality = j == 0 ? 2 : j == 1 ? 3 : 7;
        check::Require(v[j] >= 0 && v[j] < cardinality, "valid known factor index");
        ++counts[j][v[j]];
      }
      const int id = (((v[0] * 3 + v[1]) * 7 + v[2]) * 7 + v[3]), residue = (v[2] + 2 * v[3] + v[1] + 3 * v[0]) % 7;
      check::Require(s[b] == id && ids.insert(id).second, "all lattice sourceIDs unique/exact");
      check::Require(split == "test" ? residue == 0 : split == "validation" ? residue == 1 : residue >= 2,
                     "heldout compositional split");
      std::string row(784, '\0');
      for (int i = 0; i < 784; ++i) row[i] = char(r[b * 784 + i]);
      check::Require(pixels.insert(row).second, "unique binary images across splits");
      const int radius = v[1] + 2, cx = v[2] * 3 + 5, cy = v[3] * 3 + 5;
      for (int y = 0; y < 28; ++y)
        for (int x = 0; x < 28; ++x) {
          const int dx = x - cx, dy = y - cy;
          const int expected = v[0] == 0 ? dx * dx + dy * dy <= radius * radius
                                         : std::abs(dx) <= radius && std::abs(dy) <= radius;
          const int i = b * 784 + y * 28 + x;
          check::Require(r[i] == expected, "ALL294x784pixels independently regenerate circle/square factors");
          images[i] = float(expected);
        }
    }
    for (int j = 0; j < 4; ++j) {
      const int cardinality = j == 0 ? 2 : j == 1 ? 3 : 7;
      for (int k = 0; k < cardinality; ++k)
        check::Require(counts[j][k] == n / cardinality, "each known factor balanced in every split");
    }
    pools.push_back({nn::from_vector(images, {n, 28, 28, 1}), factors, source});
  }
  check::Require(ids.size() == 294 && pixels.size() == 294, "exhaustive2x3x7x7unique lattice");
  return pools;
}

/// tflow's data.npz of the factor lattice.
inline void SaveFactorData(const std::filesystem::path& root, const std::vector<BetaFactorData>& pools) {
  nn::NamedTensors arrays;
  for (int split = 0; split < 3; ++split) {
    const std::string name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    arrays.emplace_back(name + "_images", pools[split].images);
    arrays.emplace_back(name + "_factors", pools[split].factors);
    arrays.emplace_back(name + "_source_ids", pools[split].source);
  }
  nn::save_npz((root / "data.npz").string(), arrays);
}

/// tflow's Mosaic: [n, 784] probabilities as a `columns`-wide grid, nearest `scale`x,
/// 4 px gaps on gray 220.
inline void Mosaic(const std::filesystem::path& path, const nn::Tensor& probabilities, std::int64_t columns,
                   std::int64_t scale) {
  nn::save_image_grid(nn::reshape(probabilities, {probabilities.size(0), 28, 28, 1}), path.string(), columns, 0, 1,
                      4, 220.0 / 255.0, scale);
}

/// tflow's BetaFactorProbe: fit-only standardisation, ridge(.001) and best-axis
/// probes of the 4 known factors from 4 latent means; writes factor_probes.{npz,html}.
/// Returns {single-axis test R2 [4], ridge test R2 [4]}.
inline std::array<std::array<double, 4>, 2> BetaFactorProbe(const std::filesystem::path& root,
                                                            const nn::Tensor& fit_mu_t, const nn::Tensor& test_mu_t,
                                                            const nn::Tensor& fit_factors_t,
                                                            const nn::Tensor& test_factors_t,
                                                            const nn::Tensor& fit_log_variance_t) {
  const auto   fit_mu = fit_mu_t.contiguous().to_vector<float>(), test_mu = test_mu_t.contiguous().to_vector<float>(),
             fit_lv = fit_log_variance_t.contiguous().to_vector<float>();
  const auto   fit_f = fit_factors_t.contiguous().to_vector<int>(), test_f = test_factors_t.contiguous().to_vector<int>();
  const int    nf = int(fit_mu_t.size(0)), nt = int(test_mu_t.size(0));
  std::vector<double> center(4), scale(4), factor_mean(4), variance(4), unit_kl(4), x(nf * 4), y(nf * 4), xt(nt * 4);
  for (int j = 0; j < 4; ++j) {
    double m = 0, f = 0, v = 0, k = 0;
    for (int b = 0; b < nf; ++b) {
      m += fit_mu[b * 4 + j];
      f += fit_f[b * 4 + j];
    }
    m /= nf;
    f /= nf;
    for (int b = 0; b < nf; ++b) {
      const double mu = fit_mu[b * 4 + j], lv = fit_lv[b * 4 + j];
      v += (mu - m) * (mu - m);
      k += .5 * (mu * mu + std::exp(lv) - 1 - lv);
    }
    v /= nf;
    const double sd = std::max(std::sqrt(v), 1e-8);
    center[j] = m, scale[j] = sd, variance[j] = v, unit_kl[j] = k / nf, factor_mean[j] = f;
    for (int b = 0; b < nf; ++b) {
      x[b * 4 + j] = (fit_mu[b * 4 + j] - m) / sd;
      y[b * 4 + j] = fit_f[b * 4 + j] - f;
    }
    for (int b = 0; b < nt; ++b) xt[b * 4 + j] = (test_mu[b * 4 + j] - m) / sd;
  }
  auto correlation = [](const std::vector<float>& mu, const std::vector<int>& factors, int n) {
    std::vector<double> out(16);
    for (int d = 0; d < 4; ++d)
      for (int j = 0; j < 4; ++j) {
        double a = 0, b = 0, aa = 0, bb = 0, ab = 0;
        for (int i = 0; i < n; ++i) {
          a += mu[i * 4 + d];
          b += factors[i * 4 + j];
        }
        a /= n;
        b /= n;
        for (int i = 0; i < n; ++i) {
          const double u = mu[i * 4 + d] - a, v = factors[i * 4 + j] - b;
          aa += u * u;
          bb += v * v;
          ab += u * v;
        }
        out[d * 4 + j] = aa > 1e-20 && bb > 0 ? ab / std::sqrt(aa * bb) : 0;
      }
    return out;
  };
  const auto fit_corr = correlation(fit_mu, fit_f, nf), test_corr = correlation(test_mu, test_f, nt);
  // Ridge normal equations (X^T X + .001 I) C = X^T Y, solved in float64.
  auto X    = nn::from_vector(x, {nf, 4}), Y = nn::from_vector(y, {nf, 4});
  auto A    = nn::matmul(nn::transpose(X), X) + nn::mul(nn::eye(4, nn::DType::Float64), .001);
  auto coef = nn::solve(A, nn::matmul(nn::transpose(X), Y)).contiguous();
  const auto c = coef.to_vector<double>();
  std::vector<double> prediction(nt * 4), single(nt * 4), ridge_r2(4), single_r2(4);
  std::vector<int>    axis(4);
  for (int j = 0; j < 4; ++j) {
    int best = 0;
    for (int d = 1; d < 4; ++d)
      if (std::abs(fit_corr[d * 4 + j]) > std::abs(fit_corr[best * 4 + j])) best = d;
    axis[j]          = best;
    double numerator = 0, denominator = .001;
    for (int i = 0; i < nf; ++i) {
      numerator += x[i * 4 + best] * y[i * 4 + j];
      denominator += std::pow(x[i * 4 + best], 2);
    }
    const double one = numerator / denominator;
    double       truthmean = 0;
    for (int i = 0; i < nt; ++i) truthmean += test_f[i * 4 + j];
    truthmean /= nt;
    double ss = 0, err = 0, oneerr = 0;
    for (int i = 0; i < nt; ++i) {
      double p = factor_mean[j];
      for (int d = 0; d < 4; ++d) p += xt[i * 4 + d] * c[d * 4 + j];
      const double q      = factor_mean[j] + xt[i * 4 + best] * one;
      prediction[i * 4 + j] = p;
      single[i * 4 + j]     = q;
      const double target = test_f[i * 4 + j];
      ss += std::pow(target - truthmean, 2);
      err += std::pow(p - target, 2);
      oneerr += std::pow(q - target, 2);
    }
    check::Require(ss > 0, "each test factor varies");
    ridge_r2[j]  = 1 - err / ss;
    single_r2[j] = 1 - oneerr / ss;
  }
  nn::save_npz((root / "factor_probes.npz").string(),
               {{"latent_mean", nn::from_vector(center, {4})},
                {"latent_std", nn::from_vector(scale, {4})},
                {"factor_mean", nn::from_vector(factor_mean, {4})},
                {"mu_variance", nn::from_vector(variance, {4})},
                {"mean_kl_per_unit", nn::from_vector(unit_kl, {4})},
                {"fit_correlation", nn::from_vector(fit_corr, {4, 4})},
                {"test_correlation", nn::from_vector(test_corr, {4, 4})},
                {"ridge_coefficients", coef},
                {"best_axis_by_fit", nn::from_vector(axis, {4})},
                {"ridge_prediction", nn::from_vector(prediction, {nt, 4})},
                {"single_prediction", nn::from_vector(single, {nt, 4})},
                {"ridge_r2", nn::from_vector(ridge_r2, {4})},
                {"single_r2", nn::from_vector(single_r2, {4})}});
  std::ofstream html(root / "factor_probes.html");
  html << "<!doctype html><meta charset='utf-8'><title>Known-factor diagnostics</title><style>body{font:16px "
          "system-ui;margin:30px;max-width:950px}td,th{padding:8px;border:1px solid "
          "#bbb}table{border-collapse:collapse}</style><h1>Heldout factor recovery diagnostics</h1><p>Factors "
          "shape(circle0/square1),size index,x index,y index are used ONLY after representation training. Fit-only "
          "latent standardization/target means,ridge.001 and best-axis selection use210fitting records;R2 is on42heldout "
          "combinations. Correlation/linear recovery is a diagnostic,not a MIG/DCI/beta-VAE benchmark or certification "
          "of disentanglement. Test factor labels never select checkpoints.</p><table><tr><th>Factor</th><th>Best axis "
          "from fit</th><th>Single-axis test R2</th><th>All-axis ridge test R2</th></tr>";
  const char* names[] = {"shape", "size", "x", "y"};
  for (int j = 0; j < 4; ++j)
    html << "<tr><td>" << names[j] << "</td><td>" << axis[j] << "</td><td>" << single_r2[j] << "</td><td>"
         << ridge_r2[j] << "</td></tr>";
  html << "</table><h2>Test Pearson correlation:latent rows,factor columns</h2><table>";
  for (int d = 0; d < 4; ++d) {
    html << "<tr>";
    for (int j = 0; j < 4; ++j) html << "<td>" << test_corr[d * 4 + j] << "</td>";
    html << "</tr>";
  }
  html << "</table><p>Negative R2 is retained. Shared axes/off-diagonal correlations suggest mixing. Predictability "
          "from all axes does not imply one-factor-per-axis separation. Three epochs and a tiny synthetic split do "
          "not establish general disentanglement.</p>";
  check::Require(bool(html), "write factor diagnostic page");
  std::array<std::array<double, 4>, 2> r2{};
  for (int j = 0; j < 4; ++j) r2[0][j] = single_r2[j], r2[1][j] = ridge_r2[j];
  return r2;
}

/// tflow's shared VAE Checks(): known 2x2 pathwise values, analytic KL, ALL 12
/// mu/log-variance/epsilon derivatives of mean(.5|z|^2 + beta KL), zero KL and stable
/// BCE at logits +-100. Prints `label` + the summary line.
inline void VaeKnownChecks(double beta, const std::string& label) {
  namespace nn = datamunge::nn;
  const std::vector<float> m{.2f, -.4f, .1f, .3f}, l{0.f, float(std::log(4.)), float(std::log(.25)), .2f},
      e{1.f, -.5f, 2.f, -1.f};
  auto mu = nn::from_vector(m, {2, 2}).requires_grad_(), lv = nn::from_vector(l, {2, 2}).requires_grad_(),
       eps = nn::from_vector(e, {2, 2}).requires_grad_();
  auto z = VaeReparameterize(mu, lv, eps), kl = VaeKl(mu, lv);
  auto loss    = nn::mean(0.5 * nn::sum(nn::square(z), {1}) + beta * kl, {0});
  auto grad    = nn::grad(loss, {mu, lv, eps});
  auto extreme = VaePixelNll(nn::tensor({{100., -100., 100., -100.}}), nn::tensor({{1., 0., 0., 1.}}));
  auto zero_kl = VaeKl(nn::zeros({1, 2}), nn::zeros({1, 2}));
  const auto zs = z.to_vector<float>(), kls = kl.to_vector<float>(), g0 = grad[0].to_vector<float>(),
             g1 = grad[1].to_vector<float>(), g2 = grad[2].to_vector<float>();
  for (int b = 0; b < 2; ++b) {
    double k = 0;
    for (int j = 0; j < 2; ++j) {
      const int    i    = b * 2 + j;
      const double mean = m[i], logv = l[i], noise = e[i], sigma = std::exp(.5 * logv), value = mean + sigma * noise;
      k += .5 * (mean * mean + std::exp(logv) - 1 - logv);
      check::Near(zs[i], value, 2e-7, "all4reparameterized values");
      check::Near(g0[i], (value + beta * mean) / 2, 2e-7, "all4mu derivatives include pathwise PLUS KL");
      check::Near(g1[i], (.5 * value * sigma * noise + .5 * beta * (std::exp(logv) - 1)) / 2, 3e-7,
                  "all4logvariance derivatives include pathwise PLUS KL");
      check::Near(g2[i], value * sigma / 2, 3e-7, "all4epsilon derivatives follow sigma");
    }
    check::Near(kls[b], k, 2e-7, "all2analytic Gaussian KLs");
  }
  check::Near(extreme.at({0}), 200, 1e-6, "stable BCE for logits+-100");
  check::Near(zero_kl.at({0}), 0, 0, "q=N(0,I)KL exactly0");
  std::cout << label << "known4pathwise values/2KL/ALL12mu-logvariance-epsilon derivatives/zeroKL/extremeBCE passed\n";
}

/// Exact byte equality of two tensors.
inline bool SameBytes(const nn::Tensor& a, const nn::Tensor& b) { return a.shape() == b.shape() && Bytes(a) == Bytes(b); }

/// [n, 784] row of a tensor -> tflow's 7-panel "case" strip (112 px tall, 116 px pitch).
inline void CaseStrip(const std::filesystem::path& path, const std::vector<nn::Tensor>& panels, std::int64_t b) {
  PanelRow(path, [&] {
    std::vector<nn::Tensor> row;
    for (const auto& p : panels) row.push_back(Row(p, b));
    return row;
  }());
}

}  // namespace ae
