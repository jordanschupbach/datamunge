#pragma once

#include <cmath>
#include <cstddef>

namespace datamunge::bayes::detail {

/// @brief Nesterov dual-averaging step-size adaptation (Hoffman & Gelman 2014, Algorithm 5):
///        nudges the (log) step size toward whatever value makes the observed Metropolis
///        acceptance probability match a target rate, shrinking the adjustment over time.
///        Shared by HMC and NUTS warmup.
class DualAveraging {
public:
    DualAveraging(const double initial_step_size, const double target_accept_rate)
        : mu_(std::log(10.0 * initial_step_size)), target_accept_rate_(target_accept_rate), step_size_(initial_step_size) {}

    /// @brief Call once per warmup iteration with that iteration's observed acceptance probability.
    void update(const double accept_prob) {
        ++count_;
        const double m = static_cast<double>(count_);
        h_bar_ = (1.0 - 1.0 / (m + t0_)) * h_bar_ + (1.0 / (m + t0_)) * (target_accept_rate_ - accept_prob);
        const double log_eps = mu_ - std::sqrt(m) / gamma_ * h_bar_;
        const double eta_m = std::pow(m, -kappa_);
        log_step_size_bar_ = eta_m * log_eps + (1.0 - eta_m) * log_step_size_bar_;
        step_size_ = std::exp(log_eps);
    }

    [[nodiscard]] double step_size() const { return step_size_; }
    /// @brief The Cesaro-averaged step size; use this (not step_size()) once warmup ends.
    [[nodiscard]] double finalized_step_size() const { return std::exp(log_step_size_bar_); }

private:
    static constexpr double gamma_ = 0.05;
    static constexpr double t0_ = 10.0;
    static constexpr double kappa_ = 0.75;

    double mu_;
    double target_accept_rate_;
    double step_size_;
    double h_bar_{0.0};
    double log_step_size_bar_{0.0};
    std::size_t count_{0};
};

} // namespace datamunge::bayes::detail
