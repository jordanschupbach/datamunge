#pragma once

/// @file bayesian_optimization.hpp
/// @brief Bounded Bayesian optimization: a sequential design loop that fits a surrogate model to
///        every evaluation so far and evaluates the objective where the surrogate's acquisition
///        function is largest.

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::optim {

/// @brief Surrogate/acquisition extension point for BayesianOptimization.
///
/// A surrogate is refitted to all observations before every proposal. acquisition() scores a
/// candidate point; the optimizer evaluates the objective where the score is largest.
/// RBFGaussianProcessSurrogate is the built-in implementation.
class BayesianSurrogate {
public:
    /// @brief Virtual destructor for safe deletion through the base class.
    virtual ~BayesianSurrogate() = default;
    /// @brief Fits the surrogate to the observations so far.
    /// @param points  Evaluated points, one per observation.
    /// @param values  Objective values at @p points.
    virtual void fit(const std::vector<std::vector<double>>& points, const std::vector<double>& values) = 0;
    /// @brief Scores a candidate; larger means more worth evaluating.
    /// @param point      Candidate point.
    /// @param incumbent  Smallest objective value observed so far.
    /// @return The acquisition value.
    virtual double acquisition(const std::vector<double>& point, double incumbent) = 0;
};

/// @brief Settings for BayesianOptimization.
struct BayesianOptimizationOptions {
    /// @brief Uniform random points evaluated before the surrogate is first fitted (at least 1).
    std::size_t initial_samples{8};
    /// @brief Surrogate-guided evaluations after the initial design.
    std::size_t max_iterations{50};
    /// @brief Seed of the random design and the candidate draws.
    std::uint64_t seed{42};
    /// @brief Uniform random candidates scored by the acquisition function per iteration (at least 1).
    std::size_t candidates{128};
    /// @brief Compass-search step halvings that polish the best candidate on the acquisition
    ///        surface; 0 keeps the best random candidate as it is.
    std::size_t refine_iterations{20};
};

/// @brief Everything a BayesianOptimization::run() evaluated, in order.
struct BayesianOptimizationResult {
    /// @brief The evaluated point with the smallest value.
    std::vector<double> best_point;
    /// @brief The smallest observed value.
    double best_value{0.0};
    /// @brief Every evaluated point: the initial design first, then one per iteration.
    std::vector<std::vector<double>> points;
    /// @brief The objective value at each point.
    std::vector<double> values;
    /// @brief Running minimum of @ref values after each evaluation.
    std::vector<double> best_values;
    /// @brief The acquisition value of each proposed point (one per iteration, none for the design).
    std::vector<double> acquisition_values;
};

/// @brief Minimises an expensive black-box function on a box with few evaluations.
///
/// The loop is:
/// 1. evaluate `initial_samples` points drawn uniformly in `[lower, upper]`;
/// 2. repeat `max_iterations` times: fit the surrogate to all observations, score `candidates`
///    uniform random points with BayesianSurrogate::acquisition() (the incumbent being the
///    smallest value so far), polish the best one by compass search on the acquisition (steps
///    of 10% of each side, halved `refine_iterations` times, clipped to the box), and evaluate
///    the objective there.
///
/// Only point evaluations of the objective are used, so it suits expensive simulations,
/// experiments and hyperparameter tuning where each evaluation costs far more than the
/// surrogate fit.
///
/// @par Example
/// @code
/// struct Bowl : dm::optim::ArbitraryFunction {
///     double evaluate(const std::vector<double>& x) override { return (x[0] - 1.0) * (x[0] - 1.0); }
/// } bowl;
/// dm::optim::RBFGaussianProcessSurrogate gp({.fit_hyperparameters = true});
/// const auto result = dm::optim::BayesianOptimization({.initial_samples = 4, .max_iterations = 10})
///                         .run(bowl, {-3.0}, {3.0}, gp);
/// std::print("{:.3f} after {} evaluations\n", result.best_point[0], result.values.size());  // 1.002 after 14 evaluations
/// @endcode
/// @see RBFGaussianProcessSurrogate, expected_improvement()
/// @par References
/// - D. R. Jones, M. Schonlau and W. J. Welch (1998). "Efficient global optimization of expensive
///   black-box functions". Journal of Global Optimization 13, 455–492.
/// - B. Shahriari et al. (2016). "Taking the human out of the loop: a review of Bayesian
///   optimization". Proceedings of the IEEE 104(1), 148–175.
class BayesianOptimization {
public:
    /// @brief Creates the optimizer.
    /// @param options  Design size, iteration count, seed and acquisition search settings.
    explicit BayesianOptimization(BayesianOptimizationOptions options = {});

    /// @brief Runs the loop and records every evaluation.
    /// @param function   Objective to minimise.
    /// @param lower      Lower corner of the box.
    /// @param upper      Upper corner of the box (each entry larger than in @p lower).
    /// @param surrogate  Surrogate refitted before every proposal.
    /// @return The evaluations and the best point.
    /// @throws std::invalid_argument  If the bounds are empty, differ in length or are not increasing.
    [[nodiscard]] BayesianOptimizationResult run(ArbitraryFunction& function, const std::vector<double>& lower,
                                                 const std::vector<double>& upper, BayesianSurrogate& surrogate) const;

    /// @brief Runs the loop and returns only the best value.
    /// @param function     Objective to minimise.
    /// @param coordinates  Output: the best point (its length must equal the bounds').
    /// @param lower        Lower corner of the box.
    /// @param upper        Upper corner of the box.
    /// @param surrogate    Surrogate refitted before every proposal.
    /// @return The smallest observed value.
    /// @throws std::invalid_argument  If the bounds do not match @p coordinates or are not increasing.
    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates, const std::vector<double>& lower,
                    const std::vector<double>& upper, BayesianSurrogate& surrogate) const;

    /// @brief The options this BayesianOptimization was configured with.
    /// @return The stored options.
    [[nodiscard]] const BayesianOptimizationOptions& options() const noexcept { return options_; }

private:
    BayesianOptimizationOptions options_;  ///< Loop settings.
};

} // namespace datamunge::optim
