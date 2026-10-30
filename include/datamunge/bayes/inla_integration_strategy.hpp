#pragma once
/// @file inla_integration_strategy.hpp
/// @brief Bindable enum selecting finite-grid or mode-only hyperparameter treatment.
namespace datamunge::bayes {
/// @brief Treatment of hyperparameter uncertainty in Gaussian-strategy INLA.
/// The finite grid is a numerical approximation; EB omits hyperparameter mixing.
/// Total variance decomposes into within/between terms but need not exceed EB variance.
/// Separated from callback-based INLA so mixed-model bindings can expose this enum.
/// @par Example
/// @code
/// dm::bayes::INLAOptions o;
/// o.strategy=dm::bayes::INLAIntegrationStrategy::EmpiricalBayes;
/// @endcode
/// @par References
/// - Rue, Martino and Chopin (2009). https://doi.org/10.1111/j.1467-9868.2008.00700.x
/// @see INLA, INLAOptions
enum class INLAIntegrationStrategy {
  Grid,          ///< Integrate the bounded approximate hyperposterior on a rotated tensor grid.
  EmpiricalBayes ///< Use only the selected mode; no integrated evidence is computed.
};
} // namespace datamunge::bayes
