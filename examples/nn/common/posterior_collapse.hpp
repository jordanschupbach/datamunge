// Port of tflow's common/nn/posterior_collapse.h and common/ml/posterior_collapse.h: the
// sentence VAE of sequence_variational_autoencoder.hpp trained on mean(NLL + beta KL) with a
// constant or linear-warm-up KL weight, plus numerical posterior-collapse diagnostics (per-unit
// KL, active units by var(mu) > .01, empirical Gaussian-mixture information / aggregate KL).
#pragma once

#include "common/sequence_variational_autoencoder.hpp"

namespace ae {

/// tflow's AnnealedSequenceModel is the sentence VAE whose Update/Gradient take the KL weight.
using AnnealedSequenceModel = SequenceVariationalModel;

/// Linear warm-up min(1, step / (ramp_steps - 1)) over zero-based steps, or constant 1.
inline float CollapseBeta(bool warmup, int step, int ramp_steps) {
  check::Require(step >= 0 && ramp_steps >= 2, "valid zero-based warmup step and length");
  return warmup ? std::min(1.f, float(step) / float(ramp_steps - 1)) : 1.f;
}

struct CollapseSummary {
  std::vector<double> unit_kl, mu_variance;
  double              analytic_kl = 0, sampled_conditional_kl = 0, mixture_information = 0, aggregate_kl = 0;
  int                 active_units = 0;
};

/// Posterior diagnostics of mu/lv [n, 4] with aligned draws z [n, 4] (one per case per sample).
inline CollapseSummary CollapseStatistics(const nn::Tensor& mu_t, const nn::Tensor& lv_t,
                                          const std::vector<nn::Tensor>& samples) {
  const std::int64_t n = mu_t.size(0), d = mu_t.size(1);
  check::Require(n > 1 && d == 4 && mu_t.shape() == lv_t.shape() && !samples.empty(),
                 "posterior diagnostics4D and multiplecases");
  const auto      mu = mu_t.to_vector<float>(), lv = lv_t.to_vector<float>();
  CollapseSummary o;
  o.unit_kl.assign(4, 0);
  o.mu_variance.assign(4, 0);
  for (int j = 0; j < 4; ++j) {
    double mean = 0;
    for (std::int64_t b = 0; b < n; ++b) mean += mu[b * 4 + j];
    mean /= n;
    for (std::int64_t b = 0; b < n; ++b) {
      const double m = mu[b * 4 + j], v = lv[b * 4 + j];
      o.unit_kl[j] += .5 * (m * m + std::exp(v) - 1 - v) / n;
      o.mu_variance[j] += (m - mean) * (m - mean) / n;
    }
    o.analytic_kl += o.unit_kl[j];
    o.active_units += o.mu_variance[j] > .01;
  }
  for (const auto& zt : samples) {
    check::Require(zt.shape() == mu_t.shape(), "one aligned draw per posteriorcase");
    const auto          z = zt.to_vector<float>();
    std::vector<double> component(n);
    for (std::int64_t i = 0; i < n; ++i) {
      double logp = 0;
      for (std::int64_t k = 0; k < n; ++k) {
        double logq = 0;
        for (int j = 0; j < 4; ++j) {
          const double diff = double(z[i * 4 + j]) - double(mu[k * 4 + j]), variance = lv[k * 4 + j];
          logq += -.5 * (diff * diff * std::exp(-variance) + variance + std::log(2 * 3.14159265358979323846));
        }
        component[k] = logq;
      }
      const double own = component[i];
      for (int j = 0; j < 4; ++j) {
        const double v = z[i * 4 + j];
        logp += -.5 * (v * v + std::log(2 * 3.14159265358979323846));
      }
      const double maximum = *std::max_element(component.begin(), component.end());
      double       sum     = 0;
      for (double v : component) sum += std::exp(v - maximum);
      const double mixture = maximum + std::log(sum) - std::log(double(n)), scale = 1. / (n * samples.size());
      o.sampled_conditional_kl += (own - logp) * scale;
      o.mixture_information += (own - mixture) * scale;
      o.aggregate_kl += (mixture - logp) * scale;
    }
  }
  check::Near(o.sampled_conditional_kl, o.mixture_information + o.aggregate_kl, 1e-10,
              "samplewise empirical-mixture information/KL decomposition");
  return o;
}

inline void SaveCollapseStatistics(const std::filesystem::path& path, const CollapseSummary& s) {
  std::ofstream out(path);
  out.precision(17);
  out << "{\"analytic_mean_kl\":" << s.analytic_kl << ",\"sampled_conditional_kl\":" << s.sampled_conditional_kl
      << ",\"empirical_mixture_information\":" << s.mixture_information
      << ",\"empirical_aggregate_kl\":" << s.aggregate_kl
      << ",\"active_units_variance_threshold_0_01\":" << s.active_units << ",\"unit_kl\":[";
  for (int j = 0; j < 4; ++j) out << (j ? "," : "") << s.unit_kl[j];
  out << "],\"mu_variance\":[";
  for (int j = 0; j < 4; ++j) out << (j ? "," : "") << s.mu_variance[j];
  out << "]}\n";
  check::Require(bool(out), "posterior collapse diagnostic json");
}

}  // namespace ae
