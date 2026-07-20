#pragma once

#include <datamunge/filter/detail/matrix_convert.hpp>
#include <datamunge/filter/kalman_filter.hpp>
#include <datamunge/filter/vector_function.hpp>
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/linalg/lu.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace datamunge::filter {

/// @brief A bootstrap particle filter (Sequential Importance Resampling): the fully general,
///        non-Gaussian generalization of the Kalman family -- represents the state's entire
///        posterior distribution as a weighted swarm of sample points rather than assuming any
///        particular family of distribution (Gaussian, etc.), so it remains correct even for
///        strongly non-Gaussian or multimodal posteriors where every other filter in this
///        module (which all assume approximately-Gaussian beliefs) can fail. Costs scale with
///        particle count and can need many particles in high dimensions ("the curse of
///        dimensionality" for particle filters) -- EnsembleKalmanFilter is usually preferred
///        for high-dimensional problems where the Gaussian assumption is acceptable.
class ParticleFilter {
  public:
    ParticleFilter(VectorFunction& f, VectorFunction& h, std::vector<std::vector<double>> Q, std::vector<std::vector<double>> R,
                    std::vector<double> x0, std::vector<std::vector<double>> P0, int num_particles = 200, unsigned seed = 42)
        : f_(&f), h_(&h), Q_(detail::to_dense(Q)), R_(detail::to_dense(R)), rng_(seed) {
        if (num_particles < 2) throw std::invalid_argument("ParticleFilter: num_particles must be at least 2");
        const std::size_t n = x0.size();
        const auto P0_dense = detail::to_dense(P0);
        if (Q_.rows() != n || Q_.cols() != n) throw std::invalid_argument("ParticleFilter: Q must be n x n where n = x0.size()");
        if (P0_dense.rows() != n || P0_dense.cols() != n) throw std::invalid_argument("ParticleFilter: P0 must be n x n");

        const linalg::DenseMatrix<double> L0 = linalg::cholesky(P0_dense).L;
        std::normal_distribution<double> std_normal(0.0, 1.0);
        particles_.assign(static_cast<std::size_t>(num_particles), x0);
        for (auto& p : particles_) {
            std::vector<double> z(n);
            for (double& zi : z) zi = std_normal(rng_);
            const std::vector<double> noise = matvec(L0, z);
            for (std::size_t i = 0; i < n; ++i) p[i] += noise[i];
        }
        weights_.assign(particles_.size(), 1.0 / static_cast<double>(particles_.size()));
    }

    /// @brief Propagates every particle through the nonlinear f, each perturbed by an
    ///        independently-sampled draw from N(0, Q). Weights are untouched (the transition
    ///        density cancels out of the importance weight for this proposal, the standard
    ///        bootstrap-filter simplification).
    void predict() {
        const linalg::DenseMatrix<double> L = linalg::cholesky(Q_).L;
        std::normal_distribution<double> std_normal(0.0, 1.0);
        for (auto& p : particles_) {
            p = f_->evaluate(p);
            std::vector<double> z(p.size());
            for (double& zi : z) zi = std_normal(rng_);
            const std::vector<double> noise = matvec(L, z);
            for (std::size_t i = 0; i < p.size(); ++i) p[i] += noise[i];
        }
    }

    /// @brief Reweights every particle by its Gaussian (N(h(x), R)) observation likelihood
    ///        given @p z, normalizes the weights, then resamples the whole swarm from the
    ///        resulting weighted distribution (multinomial resampling, done unconditionally
    ///        every step -- simpler than the adaptive "resample only when the effective sample
    ///        size drops" scheme real-time systems often use, at the cost of extra Monte Carlo
    ///        noise from resampling more often than strictly necessary).
    void update(const std::vector<double>& z) {
        const linalg::DenseMatrix<double> R_inv = linalg::lu(R_).inverse();
        std::vector<double> log_weights(particles_.size());
        for (std::size_t i = 0; i < particles_.size(); ++i) {
            const std::vector<double> predicted = h_->evaluate(particles_[i]);
            if (predicted.size() != z.size()) throw std::invalid_argument("ParticleFilter::update: z size must match h(x)'s output size");
            std::vector<double> residual(z.size());
            for (std::size_t d = 0; d < z.size(); ++d) residual[d] = z[d] - predicted[d];
            const std::vector<double> weighted_residual = matvec(R_inv, residual);
            double quad_form = 0.0;
            for (std::size_t d = 0; d < z.size(); ++d) quad_form += residual[d] * weighted_residual[d];
            log_weights[i] = std::log(weights_[i]) - 0.5 * quad_form;
        }

        const double max_log_weight = *std::max_element(log_weights.begin(), log_weights.end());
        double sum = 0.0;
        for (std::size_t i = 0; i < weights_.size(); ++i) {
            weights_[i] = std::exp(log_weights[i] - max_log_weight);
            sum += weights_[i];
        }
        for (double& w : weights_) w /= sum;

        resample();
    }

    [[nodiscard]] std::vector<double> state() const {
        const std::size_t n = particles_.front().size();
        std::vector<double> mean(n, 0.0);
        for (std::size_t i = 0; i < particles_.size(); ++i)
            for (std::size_t d = 0; d < n; ++d) mean[d] += weights_[i] * particles_[i][d];
        return mean;
    }

    [[nodiscard]] std::vector<std::vector<double>> covariance() const {
        const std::vector<double> mean = state();
        const std::size_t n = mean.size();
        linalg::DenseMatrix<double> P(n, n, 0.0);
        for (std::size_t i = 0; i < particles_.size(); ++i) {
            for (std::size_t r = 0; r < n; ++r) {
                const double dr = particles_[i][r] - mean[r];
                for (std::size_t c = 0; c < n; ++c) P(r, c) += weights_[i] * dr * (particles_[i][c] - mean[c]);
            }
        }
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

    void resample() {
        std::discrete_distribution<std::size_t> dist(weights_.begin(), weights_.end());
        std::vector<std::vector<double>> resampled(particles_.size());
        for (std::size_t i = 0; i < particles_.size(); ++i) resampled[i] = particles_[dist(rng_)];
        particles_ = std::move(resampled);
        std::fill(weights_.begin(), weights_.end(), 1.0 / static_cast<double>(weights_.size()));
    }

    VectorFunction* f_;
    VectorFunction* h_;
    linalg::DenseMatrix<double> Q_, R_;
    std::vector<std::vector<double>> particles_;
    std::vector<double> weights_;
    std::mt19937 rng_;
};

} // namespace datamunge::filter
