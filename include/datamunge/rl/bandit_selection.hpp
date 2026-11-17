#pragma once
/// @file bandit_selection.hpp
/// @brief Explicit-source epsilon/UCB/Gaussian/Beta arm selection, with caller-owned statistics and
/// RNG.
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
namespace datamunge::rl::foundations {
/// @brief Internal validation for explicit native RL mathematical components.
namespace selection_detail {

/// @brief Validate finite nonempty arm statistics.
///
/// @param values Caller statistics.
/// @param nonnegative Also require each statistic nonnegative.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par References
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
inline void Values(const std::vector<float>& values, bool nonnegative = false) {
  if (values.empty())
    throw std::invalid_argument("nonempty arm statistics required");
  for (float v : values)
    if (!std::isfinite(v) || (nonnegative && v < 0))
      throw std::invalid_argument("finite arm statistics required");
}

/// @brief Validate aligned, incrementable arm counts.
///
/// @param counts Caller visit counts.
/// @param size Required arm count.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par References
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
inline void Counts(const std::vector<int>& counts, size_t size) {
  if (counts.size() != size)
    throw std::invalid_argument("aligned arm counts required");
  for (int n : counts)
    if (n < 0 || n == std::numeric_limits<int>::max())
      throw std::invalid_argument("valid nonnegative arm count required");
}
} // namespace selection_detail

/// @brief Select a bandit arm using the original Float32 epsilon draw.
///
/// @param q Nonempty finite estimates; ties choose the first arm.
/// @param eps Finite exploration probability in[0,1].
/// @param rng Caller-owned engine, advanced even at epsilon0; exploration consumes a raw modulo
/// draw.
/// @return Selected index; statistics unchanged.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// std::mt19937_64 rng(1);auto arm=f::BanditEpsilonGreedy({0,1},0,rng);
/// @endcode
/// @par References
/// - Sutton and Barto, Reinforcement Learning, second edition, Chapter2
/// (http://incompleteideas.net/book/the-book-2nd.html).
/// @see UcbArm()
inline int BanditEpsilonGreedy(const std::vector<float>& q, float eps, std::mt19937_64& rng) {
  selection_detail::Values(q);
  if (!std::isfinite(eps) || eps < 0 || eps > 1)
    throw std::invalid_argument("epsilon outside[0,1]");
  if (std::uniform_real_distribution<float>(0, 1)(rng) < eps)
    return static_cast<int>(rng() % q.size());
  int best = 0;
  for (size_t a = 1; a < q.size(); ++a)
    if (q[a] > q[best])
      best = static_cast<int>(a);
  return best;
}

/// @brief Select an upper-confidence-bound arm without updating statistics.
///
/// Visited score is q+c*sqrt(log(t+1)/n). No RNG is consumed.
/// Infinity replaces the legacy finite score sentinel to handle extreme finite estimates.
///
/// @param n Aligned nonnegative visits, each below INT_MAX.
/// @param q Nonempty finite estimates.
/// @param t Zero-based total step in[0,INT_MAX).
/// @param c Finite nonnegative confidence scale.
/// @return First maximizing index; unvisited arms have positive-infinite scores.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto arm=f::UcbArm({0,0},{0,0},0,2);
/// @endcode
/// @par References
/// - Sutton and Barto, Reinforcement Learning, second edition, Chapter2
/// (http://incompleteideas.net/book/the-book-2nd.html).
/// @see BanditEpsilonGreedy()
inline int UcbArm(const std::vector<int>& n, const std::vector<float>& q, int t, float c) {
  selection_detail::Values(q);
  selection_detail::Counts(n, q.size());
  if (t < 0 || t == std::numeric_limits<int>::max() || !std::isfinite(c) || c < 0)
    throw std::invalid_argument("UCB time/scale");
  int   best       = 0;
  float best_score = -std::numeric_limits<float>::infinity();
  for (size_t a = 0; a < q.size(); ++a) {
    const float score = n[a] == 0
                            ? std::numeric_limits<float>::infinity()
                            : q[a] + c * std::sqrt(std::log(static_cast<float>(t + 1)) / n[a]);
    if (score > best_score)
      best_score = score, best = static_cast<int>(a);
  }
  return best;
}

/// @brief Sample independent unit-variance Gaussian arm posteriors.
///
/// Prior mean0 precision1; posterior mean=sum/(n+1), standard deviation=1/sqrt(n+1).
///
/// @param n Nonnegative visits below INT_MAX, aligned with sum.
/// @param sum Nonempty finite reward sums.
/// @param rng Caller engine; one fresh Float32 normal distribution per arm, in index order.
/// @return First maximizing posterior draw index; statistics unchanged.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// std::mt19937_64 rng(1);auto arm=f::ThompsonGaussianArm({0,1},{0,1},rng);
/// @endcode
/// @par References
/// - Sutton and Barto, Reinforcement Learning, second edition, Chapter2
/// (http://incompleteideas.net/book/the-book-2nd.html).
/// @see ThompsonBetaArm()
inline int ThompsonGaussianArm(
    const std::vector<int>&   n,
    const std::vector<float>& sum,
    std::mt19937_64&          rng) {
  selection_detail::Values(sum);
  selection_detail::Counts(n, sum.size());
  int   best      = 0;
  float best_draw = -std::numeric_limits<float>::infinity();
  for (size_t a = 0; a < n.size(); ++a) {
    const float mean = sum[a] / (n[a] + 1), sd = 1.f / std::sqrt(n[a] + 1.f);
    const float draw = std::normal_distribution<float>(mean, sd)(rng);
    if (draw > best_draw)
      best_draw = draw, best = static_cast<int>(a);
  }
  return best;
}

/// @brief Sample independent Bernoulli Beta posteriors through Gamma draws.
///
/// Uses a uniform Beta(1,1) prior. Fractional caller counts are accepted; no observations are
/// recorded here.
///
/// @param wins Nonempty finite nonnegative success counts.
/// @param losses Aligned finite nonnegative failure counts.
/// @param rng Caller engine; fresh Gamma(wins+1,1), then Gamma(losses+1,1) per arm.
/// @return First maximizing x/(x+y) index; statistics unchanged.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// std::mt19937_64 rng(1);auto arm=f::ThompsonBetaArm({0,1},{1,0},rng);
/// @endcode
/// @par References
/// - Sutton and Barto, Reinforcement Learning, second edition, Chapter2
/// (http://incompleteideas.net/book/the-book-2nd.html).
/// @see ThompsonGaussianArm()
inline int ThompsonBetaArm(
    const std::vector<float>& wins,
    const std::vector<float>& losses,
    std::mt19937_64&          rng) {
  selection_detail::Values(wins, true);
  selection_detail::Values(losses, true);
  if (wins.size() != losses.size())
    throw std::invalid_argument("aligned beta counts required");
  int   best      = 0;
  float best_draw = -1.f;
  for (size_t a = 0; a < wins.size(); ++a) {
    std::gamma_distribution<float> ga(wins[a] + 1, 1.f), gb(losses[a] + 1, 1.f);
    const float                    x = ga(rng), y = gb(rng);
    const float                    draw = x / (x + y); // Beta(wins+1, losses+1) sample
    if (draw > best_draw)
      best_draw = draw, best = static_cast<int>(a);
  }
  return best;
}

} // namespace datamunge::rl::foundations
