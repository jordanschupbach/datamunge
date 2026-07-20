#pragma once

namespace datamunge::bayes {

/// @brief Split into its own header (rather than living inline in inla.hpp) so it can be
///        exposed to SWIG bindings independently of bayes::INLA itself: INLA::fit() takes
///        std::function callback parameters that can't cross the language boundary (same
///        reasoning as AutodiffModel -- see its doc comment), but stats::INLAMixedModel
///        exposes this enum in its plain, fully bindable INLAMixedModelOptions struct.
enum class INLAIntegrationStrategy {
    /// @brief Integrate pi(theta|y) over a grid built from the Laplace-approximated curvature
    ///        at its mode (principal axes via eigendecomposition, points spaced in posterior-SD
    ///        units) -- the full "integrated nested Laplace approximation", accounting for
    ///        hyperparameter uncertainty in the latent field's marginal variances.
    Grid,
    /// @brief Empirical Bayes: treat theta as known at its posterior mode (no integration).
    ///        Cheaper, but understates latent-field uncertainty by ignoring theta's own
    ///        posterior spread -- matches INLA's own "eb" strategy.
    EmpiricalBayes,
};

} // namespace datamunge::bayes
