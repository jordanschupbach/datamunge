#pragma once

#include <datamunge/filter/detail/matrix_convert.hpp>
#include <datamunge/filter/kalman_filter.hpp>
#include <datamunge/filter/vector_function.hpp>
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/linalg/lu.hpp>

#include <random>
#include <stdexcept>
#include <vector>

namespace datamunge::filter {

/// @brief The (stochastic, "perturbed observations") Ensemble Kalman Filter: represents the
///        state distribution as a finite ensemble of sample state vectors rather than an
///        explicit mean/covariance, propagating each member through the true nonlinear f
///        (plus sampled process noise) and estimating the covariances the Kalman update needs
///        from the ensemble's own sample statistics. Like UnscentedKalmanFilter, needs no
///        Jacobian; unlike it, scales to very high-dimensional states (the classical use case
///        this filter was designed for -- e.g. geophysical data assimilation with millions of
///        state variables) since its cost scales with ensemble size, not state dimension
///        squared/cubed. Statistical, not exact: converges toward the optimal (linear-Gaussian
///        case: KalmanFilter's) estimate as @p ensemble_size grows, with Monte Carlo sampling
///        noise at any finite size.
class EnsembleKalmanFilter {
  public:
    EnsembleKalmanFilter(VectorFunction& f, VectorFunction& h, std::vector<std::vector<double>> Q, std::vector<std::vector<double>> R,
                          std::vector<double> x0, std::vector<std::vector<double>> P0, int ensemble_size = 100, unsigned seed = 42)
        : f_(&f), h_(&h), Q_(detail::to_dense(Q)), R_(detail::to_dense(R)), rng_(seed) {
        if (ensemble_size < 2) throw std::invalid_argument("EnsembleKalmanFilter: ensemble_size must be at least 2");
        const std::size_t n = x0.size();
        const auto P0_dense = detail::to_dense(P0);
        if (Q_.rows() != n || Q_.cols() != n) throw std::invalid_argument("EnsembleKalmanFilter: Q must be n x n where n = x0.size()");
        if (P0_dense.rows() != n || P0_dense.cols() != n) throw std::invalid_argument("EnsembleKalmanFilter: P0 must be n x n");

        const linalg::DenseMatrix<double> L0 = linalg::cholesky(P0_dense).L;
        std::normal_distribution<double> std_normal(0.0, 1.0);
        ensemble_.assign(static_cast<std::size_t>(ensemble_size), x0);
        for (auto& member : ensemble_) {
            std::vector<double> z(n);
            for (double& zi : z) zi = std_normal(rng_);
            const std::vector<double> noise = matvec(L0, z);
            for (std::size_t i = 0; i < n; ++i) member[i] += noise[i];
        }
    }

    /// @brief Propagates every ensemble member through the nonlinear f, each perturbed by an
    ///        independently-sampled draw from N(0, Q).
    void predict() {
        const linalg::DenseMatrix<double> L = linalg::cholesky(Q_).L;
        std::normal_distribution<double> std_normal(0.0, 1.0);
        for (auto& member : ensemble_) {
            member = f_->evaluate(member);
            std::vector<double> z(member.size());
            for (double& zi : z) zi = std_normal(rng_);
            const std::vector<double> noise = matvec(L, z);
            for (std::size_t i = 0; i < member.size(); ++i) member[i] += noise[i];
        }
    }

    /// @brief The stochastic EnKF update (Evensen/Burgers): estimates the state-observation
    ///        cross-covariance and innovation covariance from the ensemble's own sample
    ///        statistics, then updates each member with an INDEPENDENTLY perturbed copy of @p
    ///        z (adding a fresh N(0, R) draw to each member's correction) -- perturbing the
    ///        observation per member, rather than applying one shared correction to all of
    ///        them, is what keeps the ensemble's spread statistically consistent after the
    ///        update instead of collapsing it.
    void update(const std::vector<double>& z) {
        const std::size_t n_ens = ensemble_.size();
        const std::size_t n = ensemble_.front().size();
        std::vector<std::vector<double>> observed(n_ens);
        for (std::size_t i = 0; i < n_ens; ++i) observed[i] = h_->evaluate(ensemble_[i]);
        const std::size_t m = observed.front().size();
        if (z.size() != m) throw std::invalid_argument("EnsembleKalmanFilter::update: z size must match h(x)'s output size");

        std::vector<double> x_mean(n, 0.0), y_mean(m, 0.0);
        for (std::size_t i = 0; i < n_ens; ++i) {
            for (std::size_t d = 0; d < n; ++d) x_mean[d] += ensemble_[i][d];
            for (std::size_t d = 0; d < m; ++d) y_mean[d] += observed[i][d];
        }
        for (double& v : x_mean) v /= static_cast<double>(n_ens);
        for (double& v : y_mean) v /= static_cast<double>(n_ens);

        linalg::DenseMatrix<double> P_xy(n, m, 0.0), P_yy(m, m, 0.0);
        for (std::size_t i = 0; i < n_ens; ++i) {
            for (std::size_t r = 0; r < n; ++r) {
                const double dx = ensemble_[i][r] - x_mean[r];
                for (std::size_t c = 0; c < m; ++c) P_xy(r, c) += dx * (observed[i][c] - y_mean[c]);
            }
            for (std::size_t r = 0; r < m; ++r) {
                const double dyr = observed[i][r] - y_mean[r];
                for (std::size_t c = 0; c < m; ++c) P_yy(r, c) += dyr * (observed[i][c] - y_mean[c]);
            }
        }
        const double denom = static_cast<double>(n_ens - 1);
        P_xy *= 1.0 / denom;
        P_yy *= 1.0 / denom;
        P_yy = P_yy + R_;

        const linalg::DenseMatrix<double> K = P_xy * linalg::lu(P_yy).inverse();
        const linalg::DenseMatrix<double> L_r = linalg::cholesky(R_).L;
        std::normal_distribution<double> std_normal(0.0, 1.0);
        for (std::size_t i = 0; i < n_ens; ++i) {
            std::vector<double> obs_noise(m);
            for (double& v : obs_noise) v = std_normal(rng_);
            const std::vector<double> perturbation = matvec(L_r, obs_noise);
            std::vector<double> innovation(m);
            for (std::size_t d = 0; d < m; ++d) innovation[d] = (z[d] + perturbation[d]) - observed[i][d];
            const std::vector<double> correction = matvec(K, innovation);
            for (std::size_t d = 0; d < n; ++d) ensemble_[i][d] += correction[d];
        }
    }

    [[nodiscard]] std::vector<double> state() const {
        const std::size_t n = ensemble_.front().size();
        std::vector<double> mean(n, 0.0);
        for (const auto& member : ensemble_)
            for (std::size_t d = 0; d < n; ++d) mean[d] += member[d];
        for (double& v : mean) v /= static_cast<double>(ensemble_.size());
        return mean;
    }

    [[nodiscard]] std::vector<std::vector<double>> covariance() const {
        const std::vector<double> mean = state();
        const std::size_t n = mean.size();
        linalg::DenseMatrix<double> P(n, n, 0.0);
        for (const auto& member : ensemble_) {
            for (std::size_t r = 0; r < n; ++r) {
                const double dr = member[r] - mean[r];
                for (std::size_t c = 0; c < n; ++c) P(r, c) += dr * (member[c] - mean[c]);
            }
        }
        P *= 1.0 / static_cast<double>(ensemble_.size() - 1);
        return detail::to_vector2d(P);
    }

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

    VectorFunction* f_;
    VectorFunction* h_;
    linalg::DenseMatrix<double> Q_, R_;
    std::vector<std::vector<double>> ensemble_;
    std::mt19937 rng_;
};

} // namespace datamunge::filter
