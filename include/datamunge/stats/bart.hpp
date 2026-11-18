#pragma once

/// @file bart.hpp
/// @brief Gaussian Bayesian additive regression trees and posterior predictions.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace datamunge::stats {

/// @brief Regularization, structure prior and sampler settings for Gaussian BART.
/// @par Example
/// @code
/// dm::stats::BARTOptions settings{.num_trees = 30, .warmup = 100, .draws = 200};
/// @endcode
/// @see BART
struct BARTOptions {
    std::size_t num_trees{40}; ///< Positive number of trees.
    std::size_t warmup{200}; ///< Number of initial sweeps to discard. May be zero.
    std::size_t draws{300}; ///< Positive number of consecutive posterior sweeps to retain.
    std::size_t min_leaf{3}; ///< Minimum training observations in each child of a split.
    std::size_t max_depth{6}; ///< Maximum leaf depth, from zero through 12. Zero fits stumps with no splits.
    std::size_t max_cutpoints{32}; ///< Positive cut grid capacity per predictor.
    double split_base{0.95}; ///< Root split probability, strictly between zero and one.
    double split_power{2}; ///< Finite nonnegative depth-decay exponent.
    double leaf_shrinkage{2}; ///< Positive k in the scaled leaf SD 0.5 / (k sqrt(num_trees)).
    double noise_prior_df{3}; ///< Positive scaled-inverse-chi-square prior degrees of freedom.
    double noise_prior_quantile{0.9}; ///< Prior probability that noise SD falls below the training-response SD.
    std::optional<double> fixed_noise_sd; ///< Positive known residual SD, in original response units, if supplied.
    std::uint64_t seed{42}; ///< Local MCMC engine seed.
};

/// @brief Selects which uncertainty is included in a BART prediction interval.
/// @par Example
/// @code
/// const auto target = dm::stats::BARTPredictionTarget::NewObservation;
/// @endcode
/// @see BART::predict_interval()
enum class BARTPredictionTarget {
    MeanResponse, ///< Credible interval for the regression function at each query row.
    NewObservation ///< Predictive interval including an independent Gaussian residual at each query row.
};

/// @brief Posterior means and equal-tail pointwise bounds in query-row order.
/// @par Example
/// @code
/// dm::stats::BARTPrediction prediction{{1.0}, {0.5}, {1.5}};
/// std::print("{}", prediction.mean.front()); // 1
/// @endcode
/// @see BART::predict_interval()
struct BARTPrediction {
    std::vector<double> mean; ///< Mean of latent regression draws, for either prediction target.
    std::vector<double> lower; ///< Pointwise lower quantile for the requested target.
    std::vector<double> upper; ///< Pointwise upper quantile for the requested target.
};

/// @brief Gaussian Bayesian sum-of-trees regression fitted by backfitting MCMC.
///
/// The response is centered at its midrange and divided by its range. Constant
/// responses use unit scale and a fallback noise prior calibration SD of 0.1.
/// A node splits with probability alpha/(1+depth)^beta
/// when any valid rule exists. Rules are uniform over feasible feature/cut pairs.
/// Cuts lie between distinct training values, with an evenly subsampled grid if
/// needed. Each sweep applies reversible grow/prune MH with integrated leaf
/// likelihood, samples Gaussian leaf means, and optionally samples residual variance.
/// Copies share immutable retained draws. Independent fits have independent local
/// random state. Queries follow the outermost tree branch beyond the training
/// range. Finite-chain summaries require the usual mixing checks.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4, 1, {0, 1, 2, 3});
/// const dm::stats::BART fit(x, {0, 0, 1, 1},
///                           {.num_trees = 3, .warmup = 5, .draws = 10, .min_leaf = 1});
/// std::print("{}", fit.predict(x).size()); // 4
/// @endcode
/// @par References
/// - Chipman, George and McCulloch (2010), BART: Bayesian Additive Regression Trees.
///   https://doi.org/10.1214/09-AOAS285
/// @see BARTOptions, BARTPrediction
class BART {
public:
    /// @brief Fits a Gaussian BART chain to numeric predictors and responses.
    /// @param x Finite numeric design, with at least two rows and one feature.
    /// @param y Finite continuous response, one entry per training row.
    /// @param options Tree prior and chain controls, default BARTOptions.
    /// @throws std::invalid_argument Invalid data, shape or options.
    /// @throws std::overflow_error If scaling or arithmetic exceeds finite representable values.
    /// @throws std::bad_alloc If allocation fails.
    /// @par Example
    /// See the BART class example.
    BART(const linalg::DenseMatrix<double>& x, const std::vector<double>& y, BARTOptions options = {});

    /// @brief Averages posterior regression draws at each query row.
    /// @param x Finite numeric predictors with the training feature count.
    /// @return Response-scale posterior mean, one entry per query row.
    /// @throws std::invalid_argument Invalid design values or feature count.
    /// @throws std::overflow_error If a prediction is not representable.
    /// @par Example
    /// See the BART class example.
    [[nodiscard]] std::vector<double> predict(const linalg::DenseMatrix<double>& x) const;

    /// @brief Evaluates each retained regression function on the query points.
    /// @param x Finite numeric predictors with the training feature count.
    /// @return Dense matrix with one row per MCMC draw and one column per query row.
    /// @throws std::invalid_argument Invalid design values or feature count.
    /// @throws std::overflow_error If a prediction or matrix size is not representable.
    /// @par Example
    /// @code
    /// const dm::linalg::DenseMatrix<double> x(4, 1, {0, 1, 2, 3});
    /// const dm::stats::BART fit(x, {0, 0, 1, 1}, {.num_trees = 2, .warmup = 4, .draws = 6});
    /// std::print("{}", fit.posterior_draws(x).rows()); // 6
    /// @endcode
    /// @see predict_interval()
    [[nodiscard]] linalg::DenseMatrix<double> posterior_draws(const linalg::DenseMatrix<double>& x) const;

    /// @brief Summarizes pointwise posterior credible or predictive intervals.
    /// @param x Finite numeric predictors with the training feature count.
    /// @param level Interval probability in (0,1), default 0.95.
    /// @param target MeanResponse (default) or NewObservation.
    /// @param seed Local prediction-noise seed for NewObservation, default 42.
    /// @return Latent posterior means and equal-tail type-7 interval endpoints.
    /// @throws std::invalid_argument Invalid design, level or target.
    /// @throws std::overflow_error If a prediction is not representable.
    /// @par Example
    /// @code
    /// const dm::linalg::DenseMatrix<double> x(4, 1, {0, 1, 2, 3});
    /// const dm::stats::BART fit(x, {0, 0, 1, 1}, {.num_trees = 2, .warmup = 4, .draws = 8});
    /// const auto interval = fit.predict_interval(x, 0.9);
    /// std::print("{}", interval.lower.size()); // 4
    /// @endcode
    /// @see posterior_draws(), noise_sd_draws()
    [[nodiscard]] BARTPrediction predict_interval(const linalg::DenseMatrix<double>& x, double level = 0.95,
        BARTPredictionTarget target = BARTPredictionTarget::MeanResponse, std::uint64_t seed = 42) const;

    /// @brief Borrows retained residual standard deviations, in original response units.
    /// @return One value per retained draw, in chain order.
    [[nodiscard]] const std::vector<double>& noise_sd_draws() const noexcept;

    /// @brief Returns the fraction of accepted structure moves over all sweeps.
    /// @return Fraction including rejected and infeasible grow/prune attempts in its denominator.
    [[nodiscard]] double structure_acceptance_rate() const noexcept;

private:
    struct Impl; ///< Internal owned chain state and prediction primitives.
    std::shared_ptr<const Impl> state_; ///< Shared immutable fitted chain.
};

} // namespace datamunge::stats
