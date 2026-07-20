#pragma once

#include <datamunge/filter/detail/matrix_convert.hpp>
#include <datamunge/filter/kalman_filter.hpp>
#include <datamunge/filter/vector_function.hpp>
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/linalg/lu.hpp>

#include <stdexcept>
#include <vector>

namespace datamunge::filter {

/// @brief The Unscented Kalman Filter: propagates a small deterministic set of "sigma points"
///        through the true nonlinear f/h (rather than linearizing f/h itself, as
///        ExtendedKalmanFilter does), then reconstructs the predicted mean/covariance from the
///        propagated points' weighted statistics. Captures nonlinearity to (at least) second
///        order without ever needing a Jacobian -- generally more accurate than EKF for
///        strongly nonlinear models, at the cost of 2n+1 function evaluations per step instead
///        of one. Exactly reproduces KalmanFilter's result when @p f and @p h are linear.
class UnscentedKalmanFilter {
  public:
    UnscentedKalmanFilter(VectorFunction& f, VectorFunction& h, std::vector<std::vector<double>> Q, std::vector<std::vector<double>> R,
                           std::vector<double> x0, std::vector<std::vector<double>> P0, double alpha = 1e-3, double beta = 2.0,
                           double kappa = 0.0)
        : f_(&f), h_(&h), Q_(detail::to_dense(Q)), R_(detail::to_dense(R)), x_(std::move(x0)), P_(detail::to_dense(P0)), alpha_(alpha),
          beta_(beta), kappa_(kappa), n_(x_.size()) {
        if (Q_.rows() != n_ || Q_.cols() != n_) throw std::invalid_argument("UnscentedKalmanFilter: Q must be n x n where n = x0.size()");
        if (P_.rows() != n_ || P_.cols() != n_) throw std::invalid_argument("UnscentedKalmanFilter: P0 must be n x n");
        if (alpha <= 0.0) throw std::invalid_argument("UnscentedKalmanFilter: alpha must be positive");
    }

    void predict() {
        const auto sigma = sigma_points(x_, P_);
        std::vector<std::vector<double>> propagated(sigma.size());
        for (std::size_t i = 0; i < sigma.size(); ++i) propagated[i] = f_->evaluate(sigma[i]);

        x_ = weighted_mean(propagated);
        P_ = weighted_covariance(propagated, x_, propagated, x_) + Q_;
    }

    void update(const std::vector<double>& z) {
        const auto sigma = sigma_points(x_, P_);
        std::vector<std::vector<double>> observed(sigma.size());
        for (std::size_t i = 0; i < sigma.size(); ++i) observed[i] = h_->evaluate(sigma[i]);

        const std::vector<double> z_pred = weighted_mean(observed);
        if (z.size() != z_pred.size()) throw std::invalid_argument("UnscentedKalmanFilter::update: z size must match h(x)'s output size");

        const linalg::DenseMatrix<double> P_zz = weighted_covariance(observed, z_pred, observed, z_pred) + R_;
        const linalg::DenseMatrix<double> P_xz = weighted_covariance(sigma, x_, observed, z_pred);

        const linalg::DenseMatrix<double> K = P_xz * linalg::lu(P_zz).inverse();
        std::vector<double> innovation(z.size());
        for (std::size_t i = 0; i < z.size(); ++i) innovation[i] = z[i] - z_pred[i];
        const std::vector<double> correction = detail::to_vector(K * detail::to_column(innovation));
        for (std::size_t i = 0; i < x_.size(); ++i) x_[i] += correction[i];
        P_ = P_ - K * P_zz * K.transpose();
    }

    [[nodiscard]] std::vector<double> state() const { return x_; }
    [[nodiscard]] std::vector<std::vector<double>> covariance() const { return detail::to_vector2d(P_); }

    [[nodiscard]] std::vector<KalmanState> filter(const std::vector<std::vector<double>>& measurements) {
        std::vector<KalmanState> result;
        result.reserve(measurements.size());
        for (const auto& z : measurements) {
            predict();
            update(z);
            result.push_back(KalmanState{x_, detail::to_vector2d(P_)});
        }
        return result;
    }

  private:
    [[nodiscard]] double lambda() const { return alpha_ * alpha_ * (static_cast<double>(n_) + kappa_) - static_cast<double>(n_); }

    [[nodiscard]] std::vector<std::vector<double>> sigma_points(const std::vector<double>& mean, const linalg::DenseMatrix<double>& cov) const {
        const double scale = static_cast<double>(n_) + lambda();
        const linalg::DenseMatrix<double> L = linalg::cholesky(cov * scale).L;

        std::vector<std::vector<double>> points(2 * n_ + 1, mean);
        for (std::size_t i = 0; i < n_; ++i) {
            for (std::size_t r = 0; r < n_; ++r) {
                points[1 + i][r] += L(r, i);
                points[1 + n_ + i][r] -= L(r, i);
            }
        }
        return points;
    }

    [[nodiscard]] std::vector<double> weighted_mean(const std::vector<std::vector<double>>& points) const {
        const double lam = lambda();
        const double w0 = lam / (static_cast<double>(n_) + lam);
        const double wi = 1.0 / (2.0 * (static_cast<double>(n_) + lam));

        std::vector<double> mean(points[0].size(), 0.0);
        for (std::size_t i = 0; i < points.size(); ++i) {
            const double w = (i == 0) ? w0 : wi;
            for (std::size_t d = 0; d < mean.size(); ++d) mean[d] += w * points[i][d];
        }
        return mean;
    }

    /// @brief The weighted cross-covariance sum_i Wc_i (a_i - a_mean)(b_i - b_mean)^T between
    ///        two equally-sized, correspondingly-indexed (same sigma-point ordering) point
    ///        sets -- used for the predicted-state, innovation, and cross covariances alike.
    [[nodiscard]] linalg::DenseMatrix<double> weighted_covariance(const std::vector<std::vector<double>>& a_points,
                                                                    const std::vector<double>& a_mean,
                                                                    const std::vector<std::vector<double>>& b_points,
                                                                    const std::vector<double>& b_mean) const {
        const double lam = lambda();
        const double wc0 = lam / (static_cast<double>(n_) + lam) + (1.0 - alpha_ * alpha_ + beta_);
        const double wci = 1.0 / (2.0 * (static_cast<double>(n_) + lam));

        linalg::DenseMatrix<double> cov(a_mean.size(), b_mean.size(), 0.0);
        for (std::size_t i = 0; i < a_points.size(); ++i) {
            const double w = (i == 0) ? wc0 : wci;
            for (std::size_t r = 0; r < a_mean.size(); ++r) {
                const double da = a_points[i][r] - a_mean[r];
                for (std::size_t c = 0; c < b_mean.size(); ++c) {
                    cov(r, c) += w * da * (b_points[i][c] - b_mean[c]);
                }
            }
        }
        return cov;
    }

    VectorFunction* f_;
    VectorFunction* h_;
    linalg::DenseMatrix<double> Q_, R_;
    std::vector<double> x_;
    linalg::DenseMatrix<double> P_;
    double alpha_, beta_, kappa_;
    std::size_t n_;
};

} // namespace datamunge::filter
