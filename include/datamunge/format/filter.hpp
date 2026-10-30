#pragma once

/// @file filter.hpp
/// @brief std::format / std::print support for datamunge::filter.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// Filters print their current estimate: the state vector and the covariance diagonal
/// (per-component variances); the full covariance matrix is not dumped.
///
/// @par Example
/// @code
/// dm::filter::KalmanFilter kf({{1, 1}, {0, 1}}, {{1, 0}}, {{0.01, 0}, {0, 0.01}}, {{1}}, {0, 1}, {{1, 0}, {0, 1}});
/// std::print("{}\n", kf);
/// // KalmanFilter(n=2, state=[0, 1], variances=[1, 1])
/// @endcode

#include <datamunge/filter/filter.hpp>
#include <datamunge/format/text_formatter.hpp>

#include <exception>
#include <format>
#include <string>
#include <vector>

namespace datamunge::format_detail {

/// @brief The diagonal of a square row-major nested-vector matrix.
/// @param m The matrix rows.
/// @return The diagonal entries (stops at the shortest row).
inline std::vector<double> filter_diagonal(const std::vector<std::vector<double>>& m) {
  std::vector<double> diag;
  for (std::size_t i = 0; i < m.size() && i < m[i].size(); ++i) diag.push_back(m[i][i]);
  return diag;
}

/// @brief "Name(n=k, state=[...], variances=[...])" for any filter exposing state()/covariance().
/// @tparam Filter A filter type with const state() and covariance().
/// @param name The short type name to print.
/// @param f The filter.
/// @return The filter's current-estimate summary, or "Name(<error>)" if reading the estimate throws.
template <typename Filter>
std::string filter_estimate_text(const char* name, const Filter& f) {
  try {
    const std::vector<double> x = f.state();
    return std::format("{}(n={}, state={}, variances={})", name, x.size(), preview(x),
                       preview(filter_diagonal(f.covariance())));
  } catch (const std::exception& e) {
    return std::format("{}(<{}>)", name, e.what());
  }
}

}  // namespace datamunge::format_detail

/// @brief Prints a filter::AlphaBetaState as "AlphaBetaState(position=x, velocity=v)".
template <>
struct std::formatter<datamunge::filter::AlphaBetaState>
    : datamunge::format_detail::text_formatter<datamunge::filter::AlphaBetaState> {
  /// @brief The position/velocity estimate.
  /// @param s The state.
  /// @return "AlphaBetaState(position=x, velocity=v)".
  static std::string text(const datamunge::filter::AlphaBetaState& s) {
    return std::format("AlphaBetaState(position={}, velocity={})", s.position, s.velocity);
  }
};

/// @brief Prints a filter::AlphaBetaGammaState as "AlphaBetaGammaState(position=x, velocity=v, acceleration=a)".
template <>
struct std::formatter<datamunge::filter::AlphaBetaGammaState>
    : datamunge::format_detail::text_formatter<datamunge::filter::AlphaBetaGammaState> {
  /// @brief The position/velocity/acceleration estimate.
  /// @param s The state.
  /// @return "AlphaBetaGammaState(position=x, velocity=v, acceleration=a)".
  static std::string text(const datamunge::filter::AlphaBetaGammaState& s) {
    return std::format("AlphaBetaGammaState(position={}, velocity={}, acceleration={})", s.position, s.velocity,
                       s.acceleration);
  }
};

/// @brief Prints a filter::AlphaBetaFilter as its gains, time step and current estimate.
template <>
struct std::formatter<datamunge::filter::AlphaBetaFilter>
    : datamunge::format_detail::text_formatter<datamunge::filter::AlphaBetaFilter> {
  /// @brief The gains and time step, then the current position/velocity estimate.
  /// @param f The filter.
  /// @return "AlphaBetaFilter(alpha=a, beta=b, dt=t, position=x, velocity=v)".
  static std::string text(const datamunge::filter::AlphaBetaFilter& f) {
    return std::format("AlphaBetaFilter(alpha={}, beta={}, dt={}, position={}, velocity={})", f.alpha(), f.beta(),
                       f.dt(), f.position(), f.velocity());
  }
};

/// @brief Prints a filter::AlphaBetaGammaFilter as its gains, time step and current estimate.
template <>
struct std::formatter<datamunge::filter::AlphaBetaGammaFilter>
    : datamunge::format_detail::text_formatter<datamunge::filter::AlphaBetaGammaFilter> {
  /// @brief The gains and time step, then the current position/velocity/acceleration estimate.
  /// @param f The filter.
  /// @return "AlphaBetaGammaFilter(alpha=a, beta=b, gamma=g, dt=t, position=x, velocity=v, acceleration=a)".
  static std::string text(const datamunge::filter::AlphaBetaGammaFilter& f) {
    return std::format("AlphaBetaGammaFilter(alpha={}, beta={}, gamma={}, dt={}, position={}, velocity={}, "
                       "acceleration={})",
                       f.alpha(), f.beta(), f.gamma(), f.dt(), f.position(), f.velocity(), f.acceleration());
  }
};

/// @brief Prints a filter::KalmanState as its mean and covariance diagonal.
template <>
struct std::formatter<datamunge::filter::KalmanState>
    : datamunge::format_detail::text_formatter<datamunge::filter::KalmanState> {
  /// @brief The state mean and per-component variances.
  /// @param s The state.
  /// @return "KalmanState(x=[...], variances=[...])".
  static std::string text(const datamunge::filter::KalmanState& s) {
    return std::format("KalmanState(x={}, variances={})", datamunge::format_detail::preview(s.x),
                       datamunge::format_detail::preview(datamunge::format_detail::filter_diagonal(s.P)));
  }
};

/// @brief Prints a filter::SmoothResult as its step counts and final smoothed estimate.
template <>
struct std::formatter<datamunge::filter::SmoothResult>
    : datamunge::format_detail::text_formatter<datamunge::filter::SmoothResult> {
  /// @brief The number of filtered/smoothed steps and the final smoothed state.
  /// @param r The smoothing result.
  /// @return "SmoothResult(filtered=n, smoothed=n, final=KalmanState(...))".
  static std::string text(const datamunge::filter::SmoothResult& r) {
    std::string out = std::format("SmoothResult(filtered={}, smoothed={}", r.filtered.size(), r.smoothed.size());
    if (!r.smoothed.empty()) out += std::format(", final={}", r.smoothed.back());
    return out + ")";
  }
};

/// @brief Prints a filter::KalmanFilter as its current state and variances.
template <>
struct std::formatter<datamunge::filter::KalmanFilter>
    : datamunge::format_detail::text_formatter<datamunge::filter::KalmanFilter> {
  /// @brief The current estimate.
  /// @param f The filter.
  /// @return "KalmanFilter(n=k, state=[...], variances=[...])".
  static std::string text(const datamunge::filter::KalmanFilter& f) {
    return datamunge::format_detail::filter_estimate_text("KalmanFilter", f);
  }
};

/// @brief Prints a filter::InformationFilter as its current (moment-form) state and variances.
template <>
struct std::formatter<datamunge::filter::InformationFilter>
    : datamunge::format_detail::text_formatter<datamunge::filter::InformationFilter> {
  /// @brief The current estimate, converted from information form.
  /// @param f The filter.
  /// @return "InformationFilter(n=k, state=[...], variances=[...])".
  static std::string text(const datamunge::filter::InformationFilter& f) {
    return datamunge::format_detail::filter_estimate_text("InformationFilter", f);
  }
};

/// @brief Prints a filter::ExtendedKalmanFilter as its current state and variances.
template <>
struct std::formatter<datamunge::filter::ExtendedKalmanFilter>
    : datamunge::format_detail::text_formatter<datamunge::filter::ExtendedKalmanFilter> {
  /// @brief The current estimate.
  /// @param f The filter.
  /// @return "ExtendedKalmanFilter(n=k, state=[...], variances=[...])".
  static std::string text(const datamunge::filter::ExtendedKalmanFilter& f) {
    return datamunge::format_detail::filter_estimate_text("ExtendedKalmanFilter", f);
  }
};

/// @brief Prints a filter::UnscentedKalmanFilter as its current state and variances.
template <>
struct std::formatter<datamunge::filter::UnscentedKalmanFilter>
    : datamunge::format_detail::text_formatter<datamunge::filter::UnscentedKalmanFilter> {
  /// @brief The current estimate.
  /// @param f The filter.
  /// @return "UnscentedKalmanFilter(n=k, state=[...], variances=[...])".
  static std::string text(const datamunge::filter::UnscentedKalmanFilter& f) {
    return datamunge::format_detail::filter_estimate_text("UnscentedKalmanFilter", f);
  }
};

/// @brief Prints a filter::EnsembleKalmanFilter as its ensemble-mean state and variances.
template <>
struct std::formatter<datamunge::filter::EnsembleKalmanFilter>
    : datamunge::format_detail::text_formatter<datamunge::filter::EnsembleKalmanFilter> {
  /// @brief The current ensemble estimate.
  /// @param f The filter.
  /// @return "EnsembleKalmanFilter(n=k, state=[...], variances=[...])".
  static std::string text(const datamunge::filter::EnsembleKalmanFilter& f) {
    return datamunge::format_detail::filter_estimate_text("EnsembleKalmanFilter", f);
  }
};

/// @brief Prints a filter::ParticleFilter as its weighted-mean state and variances.
template <>
struct std::formatter<datamunge::filter::ParticleFilter>
    : datamunge::format_detail::text_formatter<datamunge::filter::ParticleFilter> {
  /// @brief The current particle estimate.
  /// @param f The filter.
  /// @return "ParticleFilter(n=k, state=[...], variances=[...])".
  static std::string text(const datamunge::filter::ParticleFilter& f) {
    return datamunge::format_detail::filter_estimate_text("ParticleFilter", f);
  }
};
