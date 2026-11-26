#pragma once

/// @file detail.hpp
/// @brief Internal helpers shared by causal graph and numerical methods.

#include <datamunge/causal/aipw.hpp>
#include <datamunge/causal/matching.hpp>
#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

/// @brief Private causal fitting and search primitives.
namespace datamunge::causal::detail {

/// @brief Enumerates fixed-size subsets in lexicographic position order.
/// @tparam Visit Callable returning true to stop the search.
/// @param nodes Ordered elements available for selection.
/// @param size Desired subset size.
/// @param visit Visitor for each complete subset.
/// @return True when the visitor stops the search, false after exhaustion.
template<class Visit>
bool subsets(const std::vector<std::size_t>& nodes, std::size_t size, Visit visit) {
    if (size > nodes.size()) return false;
    std::vector<std::size_t> index(size), selected(size);
    std::iota(index.begin(), index.end(), 0);
    for (;;) {
        for (std::size_t i = 0; i < size; ++i) selected[i] = nodes[index[i]];
        if (visit(selected)) return true;
        auto j = size;
        while (j > 0 && index[j - 1] == nodes.size() - size + j - 1) --j;
        if (j == 0) return false;
        ++index[j - 1];
        for (std::size_t i = j; i < size; ++i) index[i] = index[i - 1] + 1;
    }
}

/// @brief Rejects nonfinite matrix entries.
/// @param matrix Numeric data to validate.
/// @throws std::invalid_argument If any entry is nonfinite.
void finite_matrix(const linalg::DenseMatrix<double>& matrix);

/// @brief Weighted treated-minus-control contrast with a sandwich standard error.
///
/// Fits the weighted regression of outcome on (1, treatment) over units with positive weight. The
/// variance is the cluster-robust sandwich when `clusters` is nonempty (with the vcovCL HC1 factor
/// G/(G-1)(n-1)/(n-2) if `small_sample`), otherwise the unit-level sandwich (HC1 factor n/(n-2) if
/// `small_sample`, else HC0).
/// @param outcome Outcomes for every unit.
/// @param treatment Indicators (0 or 1).
/// @param weights Nonnegative weights; zero excludes a unit.
/// @param clusters Cluster index (< n) of each weighted unit, or empty for unit-level errors.
/// @param small_sample Whether to apply the HC1-type small-sample factor.
/// @param confidence Confidence level of the t interval.
/// @return Estimate, standard error, df, p-value and interval.
/// @throws std::invalid_argument For invalid inputs.
MatchedEffect weighted_contrast(const std::vector<double>& outcome, const std::vector<int>& treatment,
                                const std::vector<double>& weights, const std::vector<std::size_t>& clusters,
                                bool small_sample, double confidence);

/// @brief Out-of-fold nuisance predictions shared by the doubly robust estimators.
struct Nuisances {
    std::vector<double> propensity;         ///< Predicted treatment probabilities (unclipped).
    std::vector<double> control_prediction; ///< Predicted outcomes under control.
    std::vector<double> treated_prediction; ///< Predicted outcomes under treatment.
    std::vector<std::size_t> fold_ids;      ///< Fold of each unit.
};

/// @brief Validates the data and fits propensity and arm-specific outcome learners out of fold.
/// @param outcome Finite outcomes.
/// @param treatment Indicators (0 or 1) with both groups represented.
/// @param covariates Finite covariates.
/// @param folds Number of folds (1: fit and predict on the full sample); ignored with explicit ids.
/// @param fold_ids Explicit 0-based fold of each unit, or empty.
/// @param seed Seed of the stratified random fold assignment.
/// @param propensity_learner Propensity learner (empty: logistic).
/// @param outcome_learner Outcome learner (empty: least squares).
/// @return Out-of-fold predictions and fold ids.
/// @throws std::invalid_argument For invalid data or folds.
/// @throws std::runtime_error When a learner fails.
Nuisances fit_nuisances(const std::vector<double>& outcome, const std::vector<int>& treatment,
                        const linalg::DenseMatrix<double>& covariates, std::size_t folds,
                        const std::vector<std::size_t>& fold_ids, std::uint64_t seed,
                        const PropensityLearner& propensity_learner, const OutcomeLearner& outcome_learner);

/// @brief Clips propensities to [bound, 1 - bound] in place.
/// @param propensity Propensities to clip.
/// @param bound Bound in [0, 0.5).
/// @return Number of values changed.
/// @throws std::invalid_argument For a bound outside [0, 0.5).
/// @throws std::runtime_error When a clipped value is still 0 or 1.
std::size_t clip_propensities(std::vector<double>& propensity, double bound);

} // namespace datamunge::causal::detail
