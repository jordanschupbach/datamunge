#pragma once

#include <stdexcept>
#include <vector>

namespace datamunge::filter {

/// @brief The position/velocity estimate returned by AlphaBetaFilter at each step.
struct AlphaBetaState {
    double position{0.0};
    double velocity{0.0};
};

/// @brief The constant-gain alpha-beta ("g-h") filter: a constant-velocity tracker that uses
///        the SAME predict/correct structure as KalmanFilter, but with fixed gains alpha
///        (position correction) and beta (velocity correction) instead of a covariance-driven
///        Kalman gain -- it is exactly the steady-state behavior a 1D constant-velocity
///        KalmanFilter converges to once its covariance stops changing, with the covariance
///        bookkeeping stripped out entirely. Far cheaper per step (no matrices at all) at the
///        cost of not adapting its gain to changing noise conditions the way a real Kalman
///        filter does.
class AlphaBetaFilter {
  public:
    AlphaBetaFilter(double alpha, double beta, double dt, double x0 = 0.0, double v0 = 0.0) : alpha_(alpha), beta_(beta), dt_(dt), x_(x0), v_(v0) {
        if (alpha <= 0.0 || alpha >= 1.0) throw std::invalid_argument("AlphaBetaFilter: alpha must be in (0, 1)");
        if (beta <= 0.0) throw std::invalid_argument("AlphaBetaFilter: beta must be positive");
        if (dt <= 0.0) throw std::invalid_argument("AlphaBetaFilter: dt must be positive");
    }

    /// @brief Predicts one step ahead, then corrects toward measurement @p z: x_pred = x +
    ///        v*dt; residual = z - x_pred; x = x_pred + alpha*residual; v += (beta/dt)*residual.
    void update(double z) {
        const double x_pred = x_ + v_ * dt_;
        const double residual = z - x_pred;
        x_ = x_pred + alpha_ * residual;
        v_ = v_ + (beta_ / dt_) * residual;
    }

    [[nodiscard]] double position() const { return x_; }
    [[nodiscard]] double velocity() const { return v_; }

    [[nodiscard]] std::vector<AlphaBetaState> filter(const std::vector<double>& measurements) {
        std::vector<AlphaBetaState> result;
        result.reserve(measurements.size());
        for (const double z : measurements) {
            update(z);
            result.push_back(AlphaBetaState{x_, v_});
        }
        return result;
    }

  private:
    double alpha_, beta_, dt_;
    double x_, v_;
};

/// @brief The position/velocity/acceleration estimate returned by AlphaBetaGammaFilter.
struct AlphaBetaGammaState {
    double position{0.0};
    double velocity{0.0};
    double acceleration{0.0};
};

/// @brief The constant-gain alpha-beta-gamma filter: AlphaBetaFilter's constant-ACCELERATION
///        generalization (a 3-state g-h-k filter), adding a third fixed gain gamma for the
///        acceleration correction. Same fixed-gain-vs-adaptive-Kalman-gain tradeoff as
///        AlphaBetaFilter, one derivative order higher.
class AlphaBetaGammaFilter {
  public:
    AlphaBetaGammaFilter(double alpha, double beta, double gamma, double dt, double x0 = 0.0, double v0 = 0.0, double a0 = 0.0)
        : alpha_(alpha), beta_(beta), gamma_(gamma), dt_(dt), x_(x0), v_(v0), a_(a0) {
        if (alpha <= 0.0 || alpha >= 1.0) throw std::invalid_argument("AlphaBetaGammaFilter: alpha must be in (0, 1)");
        if (beta <= 0.0) throw std::invalid_argument("AlphaBetaGammaFilter: beta must be positive");
        if (gamma <= 0.0) throw std::invalid_argument("AlphaBetaGammaFilter: gamma must be positive");
        if (dt <= 0.0) throw std::invalid_argument("AlphaBetaGammaFilter: dt must be positive");
    }

    void update(double z) {
        const double x_pred = x_ + v_ * dt_ + 0.5 * a_ * dt_ * dt_;
        const double v_pred = v_ + a_ * dt_;
        const double residual = z - x_pred;
        x_ = x_pred + alpha_ * residual;
        v_ = v_pred + (beta_ / dt_) * residual;
        a_ = a_ + (2.0 * gamma_ / (dt_ * dt_)) * residual;
    }

    [[nodiscard]] double position() const { return x_; }
    [[nodiscard]] double velocity() const { return v_; }
    [[nodiscard]] double acceleration() const { return a_; }

    [[nodiscard]] std::vector<AlphaBetaGammaState> filter(const std::vector<double>& measurements) {
        std::vector<AlphaBetaGammaState> result;
        result.reserve(measurements.size());
        for (const double z : measurements) {
            update(z);
            result.push_back(AlphaBetaGammaState{x_, v_, a_});
        }
        return result;
    }

  private:
    double alpha_, beta_, gamma_, dt_;
    double x_, v_, a_;
};

} // namespace datamunge::filter
