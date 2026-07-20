#pragma once

#include <datamunge/filter/detail/matrix_convert.hpp>
#include <datamunge/filter/kalman_filter.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/linalg/lu.hpp>

#include <stdexcept>
#include <vector>

namespace datamunge::filter {

/// @brief The Information Filter: the algebraic dual of KalmanFilter, tracking the information
///        matrix Y = P^-1 and information vector y = Y x instead of (x, P) directly. Its
///        measurement update is purely ADDITIVE (Y += H^T R^-1 H, y += H^T R^-1 z) -- the
///        classical advantage of this form is that fusing several independent sensors, or
///        distributing a filter across nodes that each see part of the measurement stream, is
///        just summing information contributions, with no matrix inversion per sensor. The
///        time-update (predict) step has no equally simple additive form for general F, so
///        this implementation converts to covariance form, applies the ordinary
///        KalmanFilter-style predict, and converts back -- mathematically exact, and produces
///        IDENTICAL results to KalmanFilter given the same model (the two are the same
///        estimator in a different parameterization, not an approximation of each other).
class InformationFilter {
  public:
    InformationFilter(std::vector<std::vector<double>> F, std::vector<std::vector<double>> H, std::vector<std::vector<double>> Q,
                       std::vector<std::vector<double>> R, std::vector<double> x0, std::vector<std::vector<double>> P0)
        : F_(detail::to_dense(F)), H_(detail::to_dense(H)), Q_(detail::to_dense(Q)), R_(detail::to_dense(R)) {
        const std::size_t n = x0.size();
        const auto P0_dense = detail::to_dense(P0);
        if (F_.rows() != n || F_.cols() != n) throw std::invalid_argument("InformationFilter: F must be n x n where n = x0.size()");
        if (Q_.rows() != n || Q_.cols() != n) throw std::invalid_argument("InformationFilter: Q must be n x n");
        if (P0_dense.rows() != n || P0_dense.cols() != n) throw std::invalid_argument("InformationFilter: P0 must be n x n");
        if (H_.cols() != n) throw std::invalid_argument("InformationFilter: H must have n columns");

        Y_ = linalg::lu(P0_dense).inverse();
        y_ = Y_ * detail::to_column(x0);
    }

    void predict() {
        linalg::DenseMatrix<double> P = linalg::lu(Y_).inverse();
        const linalg::DenseMatrix<double> x = P * y_;
        const linalg::DenseMatrix<double> x_pred = detail::to_column(matvec(F_, detail::to_vector(x)));
        P = F_ * P * F_.transpose() + Q_;
        Y_ = linalg::lu(P).inverse();
        y_ = Y_ * x_pred;
    }

    /// @brief The additive information-form update: Y += H^T R^-1 H, y += H^T R^-1 z.
    void update(const std::vector<double>& z) {
        if (z.size() != H_.rows()) throw std::invalid_argument("InformationFilter::update: z size must match H.rows()");
        const linalg::DenseMatrix<double> R_inv = linalg::lu(R_).inverse();
        const linalg::DenseMatrix<double> HtRinv = H_.transpose() * R_inv;
        Y_ = Y_ + HtRinv * H_;
        y_ = y_ + HtRinv * detail::to_column(z);
    }

    [[nodiscard]] std::vector<double> state() const { return detail::to_vector(linalg::lu(Y_).inverse() * y_); }
    [[nodiscard]] std::vector<std::vector<double>> covariance() const { return detail::to_vector2d(linalg::lu(Y_).inverse()); }

    [[nodiscard]] std::vector<KalmanState> filter(const std::vector<std::vector<double>>& measurements) {
        std::vector<KalmanState> result;
        result.reserve(measurements.size());
        for (const auto& z : measurements) {
            predict();
            update(z);
            result.push_back(KalmanState{state(), covariance()});
        }
        return result;
    }

  private:
    static std::vector<double> matvec(const linalg::DenseMatrix<double>& m, const std::vector<double>& v) {
        return detail::to_vector(m * detail::to_column(v));
    }

    linalg::DenseMatrix<double> F_, H_, Q_, R_;
    linalg::DenseMatrix<double> Y_;
    linalg::DenseMatrix<double> y_;
};

} // namespace datamunge::filter
