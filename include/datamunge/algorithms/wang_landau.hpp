#pragma once
/// @file wang_landau.hpp
/// @brief Checked native wang landau and actual producer state.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Raw deterministic Wang-Landau RNG helpers.
namespace detail {
/// @brief Original xorshift64 state; no cryptographic or unbiased-modulo guarantee.
struct WlRng {
  std::uint64_t s; ///< Actual raw generator state.

  /// @brief Seed the local state, replacing0 by the original fixed nonzero seed.
  explicit WlRng(std::uint64_t seed) : s(seed ? seed : 0x100000001B3ULL) {}
  /// @brief Advance and return the original xorshift64 word.
  std::uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  /// @brief Map the next word upper53 bits to [0,1).
  double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
  /// @brief Map a raw word modulo a positive validated bound.
  std::size_t below(std::size_t n) { return static_cast<std::size_t>(next() % n); }
};
} // namespace detail
/// @brief Bounded Wang-Landau toy-spin configuration.
/// @par Example
/// @code
/// dm::algorithms::WangLandauOptions o;
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
struct WangLandauOptions {
  double        flatness{.85};      ///< Required min histogram >=flatness*mean, in(0,1].
  double        log_f_min{1e-6};    ///< Positive stopping log modification factor below1.
  std::uint64_t seed{1};            ///< Original local xorshift64 seed;0 uses fixed fallback.
  std::size_t max_moves{100000000}; ///< Actual proposal ceiling;0 returns unresolved initial state.
  std::size_t check_interval{0};    ///<0 means(N+1)*1000 moves; positive custom interval<=10000000.
  std::size_t max_move_records{0};  ///< Actual first proposal records to retain, <=100000.
  std::size_t max_check_records{
      100}; ///< Actual first flatness check snapshots; histogram scalars<=1000000.
};
/// @brief One actual spin proposal, acceptance and visit update.
/// @par Example
/// @code
/// dm::algorithms::WangLandauMove r{};
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
struct WangLandauMove {
  int         before{0};    ///< Actual energy before proposal.
  int         proposed{0};  ///< Actual energy after the proposed spin flip.
  int         after{0};     ///< Actual visited energy, after acceptance/rejection.
  std::size_t spin{0};      ///< Actual proposed spin ID.
  double      log_ratio{0}; ///< Actual pre-visit logg[old]-logg[proposed].
  double      uniform{0};   ///< Actual acceptance fraction when random_used; otherwise unused0.
  double      log_f{0};     ///< Actual visit increment in force.
  bool        random_used{false}; ///< A uniform fraction was drawn only when log_ratio<0.
  bool        accepted{false};    ///< Actual flip acceptance decision.
};
/// @brief One actual histogram flatness decision.
/// @par Example
/// @code
/// dm::algorithms::WangLandauCheck r{};
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
struct WangLandauCheck {
  std::size_t              moves{0};    ///< Cumulative actual proposals at this check.
  double                   log_f{0};    ///< Actual modification factor before possible halving.
  std::vector<std::size_t> histogram;   ///< Actual visits since the current stage began.
  std::size_t              minimum{0};  ///< Actual smallest histogram count.
  double                   mean{0};     ///< Actual visits divided by N+1 bins.
  bool                     flat{false}; ///< Every bin positive and minimum>=flatness*mean.
};
/// @brief Actual normalized log density and bounded sampling status.
/// @par Example
/// @code
/// auto r=dm::algorithms::wang_landau_work(2);
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
struct WangLandauWork {
  std::vector<double>      log_density; ///< Actual logg estimate normalized by subtracting logg[0].
  std::vector<std::size_t> histogram;   ///< Actual current/last stage histogram.
  std::vector<WangLandauMove>  move_records;  ///< Selected actual proposal prefix.
  std::vector<WangLandauCheck> check_records; ///< Selected actual check prefix.
  std::size_t                  moves{0};      ///< Actual proposed spin flips.
  std::size_t                  accepted{0};   ///< Actual accepted flips.
  std::size_t                  checks{0};     ///< Actual histogram flatness checks.
  std::size_t completed_stages{0};            ///< Actual flat stages followed by log_f halving.
  double      log_f_remaining{1};             ///< Actual current log modification factor.
  bool        complete{false}; ///< Stopping log_f threshold reached before move exhaustion.
};
/// @brief Estimate toy-spin log density with bounded actual Wang-Landau sampling.
/// Preserves the original xorshift proposal/conditional acceptance stream and log_f halving. Each visit increments logg at the visited energy, including rejected proposals. Default flatness checks every(N+1)*1000 moves. Captures are bounded prefixes and never replay sampling. Flat histograms and a small modification factor do not certify estimation accuracy; compare the exact binomial density separately. O(moves+checks*N) work, O(N) base state plus bounded capture; no shared state.
/// @param N Number of independent binary spins, in[1,1000]; energy counts up-spins.
/// @param options Finite valid flatness, threshold, seed, work and capture limits.
/// @return Actual normalized estimate, move/check evidence and explicit completion.
/// @throws std::invalid_argument Invalid N, flatness, threshold or interval.
/// @throws std::length_error Work/capture budgets exceed stated ceilings.
/// @par Example
/// @code
/// dm::algorithms::WangLandauOptions o;o.max_moves=100;o.max_move_records=100;auto r=dm::algorithms::wang_landau_work(2,o);
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
WangLandauWork wang_landau_work(int N, WangLandauOptions options = {});
/// @brief Return a completed normalized log density using default work limits.
/// @param N Spin count in[1,1000].
/// @param flatness Finite histogram fraction in(0,1], default.85.
/// @param log_f_min Finite positive threshold below1, default1e-6.
/// @param seed Original local seed, default1.
/// @return Actual completed log-density values.
/// @throws std::invalid_argument Invalid parameters.
/// @throws std::runtime_error Default move budget exhausted before the stopping threshold.
/// @par Example
/// @code
/// auto logg=dm::algorithms::wang_landau_dos(2);
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
std::vector<double>
wang_landau_dos(int N, double flatness = .85, double log_f_min = 1e-6, std::uint64_t seed = 1);

} // namespace datamunge::algorithms
