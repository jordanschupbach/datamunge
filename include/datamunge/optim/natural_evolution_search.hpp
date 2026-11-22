#pragma once

/// @file
/// @brief Objective-only natural evolution and antithetic random search.

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

/// @brief Distribution or paired perturbation update rule.
/// @par Example
/// @code
/// const auto rule = dm::optim::NaturalSearchMethod::xnes;
/// @endcode
/// @see NaturalEvolutionSearch
enum class NaturalSearchMethod {
  xnes,       ///< Full covariance exponential natural evolution strategy.
  snes,       ///< Separable diagonal natural evolution strategy.
  openai_es,  ///< Antithetic Gaussian directions, centered ranks and Adam.
  ars         ///< Augmented random search with ranked directions and reward normalization.
};

/// @brief Work budget, Gaussian scale and update controls.
/// @par Example
/// @code
/// dm::optim::NaturalSearchOptions options;
/// options.method = dm::optim::NaturalSearchMethod::ars;
/// @endcode
/// @see NaturalEvolutionSearch
struct NaturalSearchOptions {
  std::size_t evaluations{1000}; ///< Positive cap, including start, final center or repeated trials.
  std::uint64_t seed{42}; ///< Seed for the local Gaussian generator.
  NaturalSearchMethod method{NaturalSearchMethod::snes}; ///< Natural or antithetic update rule.
  std::size_t population{24}; ///< Draws per NES batch or direction pairs per antithetic batch.
  double scale{0.15}; ///< Initial Gaussian deviation in units of box width, in (0,1].
  double learning_rate{0.1}; ///< Center update learning rate in (0,1].
  double covariance_rate{0.2}; ///< NES log-scale and covariance rate in (0,1].
  std::size_t retained_directions{0}; ///< ARS directions retained by ranking; zero uses all.
};

/// @brief Minimize a boxed objective with natural evolution or antithetic score differences.
/// @par Example
/// @code
/// const dm::optim::NaturalEvolutionSearch search;
/// @endcode
/// @see NaturalSearchOptions, CMASearch
class NaturalEvolutionSearch {
public:
  /// @brief Store search options by value.
  /// @param options Work cap, method and optimizer controls.
  /// @par Example
  /// @code
  /// const dm::optim::NaturalEvolutionSearch search({128, 7});
  /// @endcode
  /// @see optimize()
  explicit NaturalEvolutionSearch(NaturalSearchOptions options = {});

  /// @brief Return the earliest best value observed under the configured method.
  ///
  /// The start is clamped and scored. Search operates in normalized coordinates
  /// and clips every candidate to the box. xNES applies a symmetric
  /// matrix-exponential update to its Gaussian factor; SNES updates a diagonal
  /// scale. Both use logarithmic rank utilities. The OpenAI ES mode uses
  /// centered ranks of paired scores and updates the center with Adam. ARS
  /// ranks each pair by its best loss, then averages selected differences
  /// normalized by the standard deviation of their losses. The final partial
  /// batch or final center evaluation counts against the same budget.
  /// Replay holds within a build and callback exceptions leave coordinates unchanged.
  /// @param function Borrowed objective returning finite losses.
  /// @param coordinates Finite starting point replaced by the best observed trial.
  /// @param lower Finite inclusive lower bounds.
  /// @param upper Finite inclusive upper bounds at least as large as lower.
  /// @return Loss already observed at the output coordinates.
  /// @throws std::invalid_argument Invalid shape, budget, method or controls.
  /// @throws std::length_error Full matrix dimensions would overflow.
  /// @throws std::domain_error Objective result is nonfinite.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2}, {1});
  /// std::vector<double> x{0};
  /// const auto loss = dm::optim::NaturalEvolutionSearch({9, 42}).optimize(f, x, {1}, {1});
  /// std::print("{}\n", loss); // 0
  /// @endcode
  /// @par References
  /// - Wierstra et al. (2014). https://jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
  /// - Salimans et al. (2017). https://arxiv.org/abs/1703.03864
  /// - Mania, Guy and Recht (2018). https://arxiv.org/abs/1803.07055
  /// @see StochasticApproximation, trace_optimize()
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                  const std::vector<double>& lower, const std::vector<double>& upper) const;

private:
  NaturalSearchOptions options_; ///< Owned budget and update settings.
};

} // namespace datamunge::optim
