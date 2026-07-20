#pragma once

#include <datamunge/filter/detail/matrix_convert.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/linalg/lu.hpp>

#include <stdexcept>
#include <vector>

namespace datamunge::filter {

/// @brief One state estimate + its covariance -- the common return shape for every "give me
///        the state at every timestep" result in this module (filtered sequences, smoothed
///        sequences, ensemble-filter output, ...).
struct KalmanState {
    std::vector<double> x;
    std::vector<std::vector<double>> P;
};

/// @brief The result of KalmanFilter::smooth(): the forward-pass (filtered, causal) estimates
///        alongside the RTS backward-pass (smoothed, uses the whole sequence) estimates.
///        smoothed.back() always exactly equals filtered.back() (the smoother has no future
///        data at the final step); every earlier smoothed covariance is <= the corresponding
///        filtered covariance (in the positive-semidefinite order) since it incorporates more
///        information.
struct SmoothResult {
    std::vector<KalmanState> filtered;
    std::vector<KalmanState> smoothed;
};

/// @brief The standard discrete-time linear Kalman filter: x_k = F x_{k-1} + w_k, z_k = H x_k
///        + v_k, w_k ~ N(0, Q), v_k ~ N(0, R) -- the minimum-mean-squared-error linear
///        estimator for exactly this model. Stateful: holds the current (x, P) estimate,
///        mutated in place by predict()/update(); F, H, Q, R are fixed for the filter's
///        lifetime (construct a new filter if they change over time).
class KalmanFilter {
  public:
    KalmanFilter(std::vector<std::vector<double>> F, std::vector<std::vector<double>> H, std::vector<std::vector<double>> Q,
                 std::vector<std::vector<double>> R, std::vector<double> x0, std::vector<std::vector<double>> P0)
        : F_(detail::to_dense(F)), H_(detail::to_dense(H)), Q_(detail::to_dense(Q)), R_(detail::to_dense(R)), x_(std::move(x0)),
          P_(detail::to_dense(P0)) {
        const std::size_t n = x_.size();
        if (F_.rows() != n || F_.cols() != n) throw std::invalid_argument("KalmanFilter: F must be n x n where n = x0.size()");
        if (Q_.rows() != n || Q_.cols() != n) throw std::invalid_argument("KalmanFilter: Q must be n x n");
        if (P_.rows() != n || P_.cols() != n) throw std::invalid_argument("KalmanFilter: P0 must be n x n");
        if (H_.cols() != n) throw std::invalid_argument("KalmanFilter: H must have n columns");
        if (R_.rows() != H_.rows() || R_.cols() != H_.rows()) throw std::invalid_argument("KalmanFilter: R must be m x m where m = H.rows()");
    }

    /// @brief The time-update (prediction) step: x = F x, P = F P F^T + Q.
    void predict() {
        x_ = matvec(F_, x_);
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    /// @brief The measurement-update (correction) step given observation @p z: the standard
    ///        innovation-covariance / Kalman-gain form (S = H P H^T + R, K = P H^T S^-1, x +=
    ///        K(z - Hx), P = (I - KH) P) -- not the numerically-hardened Joseph form, matching
    ///        this codebase's general preference for the textbook formula over a more robust
    ///        but more complex alternative.
    void update(const std::vector<double>& z) {
        if (z.size() != H_.rows()) throw std::invalid_argument("KalmanFilter::update: z size must match H.rows()");
        const std::vector<double> y = subtract(z, matvec(H_, x_));
        const linalg::DenseMatrix<double> S = H_ * P_ * H_.transpose() + R_;
        const linalg::DenseMatrix<double> K = P_ * H_.transpose() * linalg::lu(S).inverse();
        x_ = add(x_, matvec(K, y));
        const linalg::DenseMatrix<double> I = linalg::DenseMatrix<double>::identity(x_.size());
        P_ = (I - K * H_) * P_;
    }

    [[nodiscard]] std::vector<double> state() const { return x_; }
    [[nodiscard]] std::vector<std::vector<double>> covariance() const { return detail::to_vector2d(P_); }

    /// @brief Runs predict() then update(z) for every z in @p measurements, in order, mutating
    ///        this filter's running state -- the ordinary causal (online) filtered estimate at
    ///        every timestep.
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

    /// @brief filter() followed by an RTS (Rauch-Tung-Striebel) backward smoothing pass, which
    ///        uses the ENTIRE sequence (not just data up to each point) to refine every
    ///        estimate -- always at least as accurate as the filtered estimate, at the cost of
    ///        needing the full sequence in advance (not usable online). Leaves this filter's
    ///        running (x, P) at its post-filter state, same as filter() would.
    [[nodiscard]] SmoothResult smooth(const std::vector<std::vector<double>>& measurements) {
        const std::size_t n = measurements.size();
        if (n == 0) {
            return {};
        }

        std::vector<std::vector<double>> x_pred(n), x_filt(n);
        std::vector<linalg::DenseMatrix<double>> P_pred(n), P_filt(n);
        for (std::size_t k = 0; k < n; ++k) {
            predict();
            x_pred[k] = x_;
            P_pred[k] = P_;
            update(measurements[k]);
            x_filt[k] = x_;
            P_filt[k] = P_;
        }

        std::vector<std::vector<double>> x_smooth(n);
        std::vector<linalg::DenseMatrix<double>> P_smooth(n);
        x_smooth[n - 1] = x_filt[n - 1];
        P_smooth[n - 1] = P_filt[n - 1];
        for (std::size_t i = n - 1; i-- > 0;) {
            const linalg::DenseMatrix<double> C = P_filt[i] * F_.transpose() * linalg::lu(P_pred[i + 1]).inverse();
            x_smooth[i] = add(x_filt[i], matvec(C, subtract(x_smooth[i + 1], x_pred[i + 1])));
            P_smooth[i] = P_filt[i] + C * (P_smooth[i + 1] - P_pred[i + 1]) * C.transpose();
        }

        SmoothResult out;
        out.filtered.reserve(n);
        out.smoothed.reserve(n);
        for (std::size_t k = 0; k < n; ++k) {
            out.filtered.push_back(KalmanState{x_filt[k], detail::to_vector2d(P_filt[k])});
            out.smoothed.push_back(KalmanState{x_smooth[k], detail::to_vector2d(P_smooth[k])});
        }
        return out;
    }

  private:
    static std::vector<double> matvec(const linalg::DenseMatrix<double>& m, const std::vector<double>& v) {
        return detail::to_vector(m * detail::to_column(v));
    }
    static std::vector<double> add(const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> out(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
        return out;
    }
    static std::vector<double> subtract(const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> out(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] - b[i];
        return out;
    }

    linalg::DenseMatrix<double> F_, H_, Q_, R_;
    std::vector<double> x_;
    linalg::DenseMatrix<double> P_;
};

} // namespace datamunge::filter
