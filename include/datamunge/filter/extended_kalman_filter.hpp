#pragma once

#include <datamunge/filter/detail/matrix_convert.hpp>
#include <datamunge/filter/kalman_filter.hpp>
#include <datamunge/filter/vector_function.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/linalg/lu.hpp>

#include <stdexcept>
#include <vector>

namespace datamunge::filter {

namespace detail {
/// @brief The Jacobian of @p fn at @p x via central finite differences (step @p h per
///        coordinate) -- avoids requiring callers to hand-derive analytic derivatives for
///        every custom nonlinear model, matching this module's uniform "just supply f/h as a
///        VectorFunction" interface across EKF/UKF/EnKF/ParticleFilter (only EKF actually
///        needs a Jacobian; it's kept private to this file rather than a public utility since
///        nothing else in the module needs it).
inline linalg::DenseMatrix<double> numerical_jacobian(VectorFunction& fn, const std::vector<double>& x, double h) {
    const std::vector<double> f0 = fn.evaluate(x);
    const std::size_t m = f0.size();
    const std::size_t n = x.size();
    linalg::DenseMatrix<double> J(m, n);
    std::vector<double> perturbed = x;
    for (std::size_t j = 0; j < n; ++j) {
        const double original = perturbed[j];
        perturbed[j] = original + h;
        const std::vector<double> f_plus = fn.evaluate(perturbed);
        perturbed[j] = original - h;
        const std::vector<double> f_minus = fn.evaluate(perturbed);
        perturbed[j] = original;
        for (std::size_t i = 0; i < m; ++i) J(i, j) = (f_plus[i] - f_minus[i]) / (2.0 * h);
    }
    return J;
}
} // namespace detail

/// @brief The Extended Kalman Filter: the standard first-order generalization of KalmanFilter
///        to a nonlinear model x_k = f(x_{k-1}) + w_k, z_k = h(x_k) + v_k -- linearizes f and h
///        (via numerical Jacobians, see detail::numerical_jacobian()) about the current state
///        estimate at every step and otherwise runs the exact same predict/update algebra as
///        the linear filter. Exactly reproduces KalmanFilter's result when @p f and @p h
///        happen to be linear (their Jacobian is then the same constant matrix everywhere).
class ExtendedKalmanFilter {
  public:
    ExtendedKalmanFilter(VectorFunction& f, VectorFunction& h, std::vector<std::vector<double>> Q, std::vector<std::vector<double>> R,
                          std::vector<double> x0, std::vector<std::vector<double>> P0, double jacobian_step = 1e-5)
        : f_(&f), h_(&h), Q_(detail::to_dense(Q)), R_(detail::to_dense(R)), x_(std::move(x0)), P_(detail::to_dense(P0)),
          jacobian_step_(jacobian_step) {
        const std::size_t n = x_.size();
        if (Q_.rows() != n || Q_.cols() != n) throw std::invalid_argument("ExtendedKalmanFilter: Q must be n x n where n = x0.size()");
        if (P_.rows() != n || P_.cols() != n) throw std::invalid_argument("ExtendedKalmanFilter: P0 must be n x n");
        if (jacobian_step <= 0.0) throw std::invalid_argument("ExtendedKalmanFilter: jacobian_step must be positive");
    }

    void predict() {
        const linalg::DenseMatrix<double> F = detail::numerical_jacobian(*f_, x_, jacobian_step_);
        x_ = f_->evaluate(x_);
        P_ = F * P_ * F.transpose() + Q_;
    }

    void update(const std::vector<double>& z) {
        const linalg::DenseMatrix<double> H = detail::numerical_jacobian(*h_, x_, jacobian_step_);
        const std::vector<double> predicted_z = h_->evaluate(x_);
        if (z.size() != predicted_z.size()) throw std::invalid_argument("ExtendedKalmanFilter::update: z size must match h(x)'s output size");
        std::vector<double> y(z.size());
        for (std::size_t i = 0; i < z.size(); ++i) y[i] = z[i] - predicted_z[i];

        const linalg::DenseMatrix<double> S = H * P_ * H.transpose() + R_;
        const linalg::DenseMatrix<double> K = P_ * H.transpose() * linalg::lu(S).inverse();
        const std::vector<double> correction = detail::to_vector(K * detail::to_column(y));
        for (std::size_t i = 0; i < x_.size(); ++i) x_[i] += correction[i];
        const linalg::DenseMatrix<double> I = linalg::DenseMatrix<double>::identity(x_.size());
        P_ = (I - K * H) * P_;
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
    VectorFunction* f_;
    VectorFunction* h_;
    linalg::DenseMatrix<double> Q_, R_;
    std::vector<double> x_;
    linalg::DenseMatrix<double> P_;
    double jacobian_step_;
};

} // namespace datamunge::filter
