#include <cmath>
#include <stdexcept>
#include <datamunge/optim/rbf_gaussian_process_surrogate.hpp>
namespace datamunge::optim {
RBFGaussianProcessSurrogate::RBFGaussianProcessSurrogate(double l, double n)
    : length_scale_(l), noise_(n) { if (l <= 0 || n < 0) throw std::invalid_argument("invalid GP hyperparameters"); }
void RBFGaussianProcessSurrogate::fit(
    const std::vector<std::vector<double>>& p,
  const std::vector<double>&              v) {
  if (p.empty() || p.size() != v.size()) throw std::invalid_argument("invalid GP training data");
  for (const auto& point : p) if (point.size() != p.front().size()) throw std::invalid_argument("inconsistent GP dimensions");
  points_       = p;
  values_       = v;
  std::size_t n = v.size();
  cholesky_.assign(n * n, 0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j <= i; ++j) {
      double s = 0;
      for (std::size_t k = 0; k < points_[i].size(); ++k) {
        double d = points_[i][k] - points_[j][k];
        s += d * d;
      }
      s = std::exp(-s / (2 * length_scale_ * length_scale_)) + (i == j ? noise_ : 0);
      for (std::size_t k = 0; k < j; ++k)
        s -= cholesky_[i * n + k] * cholesky_[j * n + k];
      cholesky_[i * n + j] = i == j ? std::sqrt(s) : s / cholesky_[j * n + j];
    }
  alpha_ = v;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < i; ++k)
      alpha_[i] -= cholesky_[i * n + k] * alpha_[k];
    alpha_[i] /= cholesky_[i * n + i];
  }
  for (std::size_t i = n; i-- > 0;) {
    for (std::size_t k = i + 1; k < n; ++k)
      alpha_[i] -= cholesky_[k * n + i] * alpha_[k];
    alpha_[i] /= cholesky_[i * n + i];
  }
}
double RBFGaussianProcessSurrogate::acquisition(const std::vector<double>& x, double best) {
  if (!points_.empty() && x.size() != points_.front().size()) throw std::invalid_argument("GP query dimension mismatch");
  std::size_t n = values_.size();
  if (!n)
    return 0;
  std::vector<double> k(n), z(n);
  double              mean = 0;
  for (std::size_t i = 0; i < n; ++i) {
    double d = 0;
    for (std::size_t q = 0; q < x.size(); ++q) {
      double e = x[q] - points_[i][q];
      d += e * e;
    }
    k[i] = std::exp(-d / (2 * length_scale_ * length_scale_));
    mean += k[i] * alpha_[i];
    z[i] = k[i];
    for (std::size_t q = 0; q < i; ++q)
      z[i] -= cholesky_[i * n + q] * z[q];
    z[i] /= cholesky_[i * n + i];
  }
  double var = 1;
  for (double q : z)
    var -= q * q;
  double sd = std::sqrt(std::max(1e-12, var)), u = (best - mean) / sd;
  return (best - mean) * .5 * (1 + std::erf(u / std::sqrt(2.0))) +
         sd * .3989422804 * std::exp(-.5 * u * u);
}
} // namespace datamunge::optim
