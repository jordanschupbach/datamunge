#pragma once
/// @file inla_diagnostics.hpp
/// @brief INLA moment accounting, Gaussian-mixture densities and scalar quadrature references.
#include <datamunge/bayes/grid_posterior.hpp>
#include <datamunge/bayes/inla.hpp>
#include <datamunge/dstruct/dataframe.hpp>
namespace datamunge::bayes {
/// @brief Conditional mode/SD and repeated convergence diagnostics for each latent coordinate.
/// Inputs are borrowed unchanged, results owned, O(p) work/storage.
/// @param fit Nonempty compatible conditional result from INLA::conditional().
/// @return component/mode/sd/iterations/gradient_inf/conditional_log_evidence.
/// @throws std::invalid_argument Incompatible dimensions or nonfinite/nonpositive variances.
/// @par Example
/// @code
/// const auto f=dm::bayes::INLA().conditional(dm::linalg::DenseMatrix<double>(3,1,1),{1,2,3},
///     dm::bayes::inla_gaussian_likelihood([](const
///     auto&){return 1.;}),dm::linalg::DenseMatrix<double>(1,1,1));
/// std::print("{:.1f}\n",dm::bayes::inla_conditional_table(f).double_at("mode",0)); // 1.5
/// @endcode
/// @see INLAConditional
dstruct::DataFrame inla_conditional_table(const INLAConditional& fit);
/// @brief Mixture moments and the exact law-of-total-variance accounting used by INLA.
/// Within/between terms are nonnegative; their sum equals sd squared to roundoff.
/// No input mutation, O(p) work/result storage; does not certify approximation accuracy.
/// @param fit Compatible result from INLA::fit().
/// @return component/mean/sd/within_variance/between_variance/total_variance.
/// @throws std::invalid_argument Incompatible/nonfinite result arrays.
/// @par Example
/// @code
/// dm::bayes::INLAResult r;r.latent_mean={1};r.latent_sd={2};
/// r.latent_within_variance={3};r.latent_between_variance={1};
/// std::print("{:.0f}\n",dm::bayes::inla_latent_summary(r).double_at("total_variance",0)); // 4
/// @endcode
/// @see INLAResult
dstruct::DataFrame inla_latent_summary(const INLAResult& fit);
/// @brief Retained theta grid, normalized weights, log integrand and conditional-mode diagnostics.
/// O(K*(d+p)) work/result storage; owns tables, leaves input unchanged. EB gives one row.
/// @param fit Complete compatible result from INLA::fit().
/// @return
/// theta_0..theta_d-1/weight/cell_volume/log_posterior/iterations/gradient_inf/mode_0..mode_p-1.
/// @throws std::invalid_argument Incompatible result shapes or invalid weights.
/// @par Example
/// @code
/// dm::bayes::INLAOptions o;o.strategy=dm::bayes::INLAIntegrationStrategy::EmpiricalBayes;
/// const auto r=dm::bayes::INLA(o).fit(dm::linalg::DenseMatrix<double>(2,1,1),{0,1},
///     dm::bayes::inla_binomial_logit_likelihood(),[](const auto&){return
///     dm::linalg::DenseMatrix<double>(1,1,1);},
///     [](const auto& t){return dm::bayes::normal_lpdf(t[0],0,1);},{-4},{4},{0});
/// std::print("{}\n",dm::bayes::inla_grid_table(r).nrows()); // 1
/// @endcode
/// @see inla_latent_summary()
dstruct::DataFrame inla_grid_table(const INLAResult& fit);
/// @brief Evaluate a latent marginal's retained Gaussian-mixture density at specified values.
/// Uses conditional modes and diagonal inverse Hessians, not a single Gaussian fitted
/// to the final mean/SD. O(K*m) work, O(m) result; no input mutation/shared state.
/// @param fit Complete result from INLA::fit(), with normalized weights.
/// @param component Valid zero-based latent coordinate.
/// @param values Finite evaluation locations; may be empty or unordered.
/// @return value/density table in input order, including both Gaussian and mixing approximations.
/// @throws std::invalid_argument Invalid dimensions/component/variances/weights or nonfinite
/// values.
/// @par Example
/// @code
/// dm::bayes::INLAOptions o;o.strategy=dm::bayes::INLAIntegrationStrategy::EmpiricalBayes;
/// const auto r=dm::bayes::INLA(o).fit(dm::linalg::DenseMatrix<double>(2,1,1),{0,1},
///     dm::bayes::inla_binomial_logit_likelihood(),[](const auto&){return
///     dm::linalg::DenseMatrix<double>(1,1,1);},
///     [](const auto& t){return dm::bayes::normal_lpdf(t[0],0,1);},{-4},{4},{0});
/// std::print("{}\n",dm::bayes::inla_latent_density(r,0,{-1,0,1}).nrows()); // 3
/// @endcode
/// @see INLAResult
dstruct::DataFrame inla_latent_density(
    const INLAResult&          fit,
    std::size_t                component,
    const std::vector<double>& values);
/// @brief Quantile of a retained latent Gaussian-mixture marginal by CDF bisection.
/// Brackets over all components and expands as needed; at most 120 bisection steps.
/// This is an interval of the approximate mixture, not an exact-posterior guarantee.
/// Work O(K*120), no input mutation/shared state.
/// @param fit Complete fitted mixture with normalized weights.
/// @param component Valid latent coordinate.
/// @param probability Finite probability strictly between zero and one.
/// @return Approximate inverse mixture CDF, in latent-coordinate units.
/// @throws std::invalid_argument Invalid mixture, component or probability.
/// @par Example
/// @code
/// dm::bayes::INLAOptions o;o.strategy=dm::bayes::INLAIntegrationStrategy::EmpiricalBayes;
/// const auto r=dm::bayes::INLA(o).fit(dm::linalg::DenseMatrix<double>(2,1,1),{0,1},
///     dm::bayes::inla_binomial_logit_likelihood(),[](const auto&){return
///     dm::linalg::DenseMatrix<double>(1,1,1);},
///     [](const auto& t){return dm::bayes::normal_lpdf(t[0],0,1);},{-4},{4},{0});
/// std::print("{:.1f}\n",dm::bayes::inla_latent_quantile(r,0,.5)); // 0.0
/// @endcode
/// @see inla_latent_density()
double inla_latent_quantile(const INLAResult& fit, std::size_t component, double probability);
/// @brief Direct finite-window quadrature of a scalar latent posterior at fixed theta.
/// Integrates the actual likelihood times a normalized N(0,1/precision) prior using
/// the midpoint grid_posterior() rule, without Laplace approximation. Truncation and
/// quadrature errors remain: widen/refine the window and inspect edge_mass.
/// O(n*points) work and O(points) result; no input mutation. This supports one scalar
/// intercept loading one on each observation, not a general multivariate reference.
/// @param y Nonempty finite likelihood-valid responses.
/// @param likelihood Observation density/derivatives; only log_density is used.
/// @param precision Positive finite scalar Gaussian prior precision.
/// @param theta Finite fixed likelihood hyperparameters, possibly empty.
/// @param lower Finite left endpoint of the latent integration window.
/// @param upper Finite right endpoint, greater than lower.
/// @param points Number of midpoint cells, default 20001, at least three.
/// @return GridPosterior with approximate actual scalar moments/density/log evidence and edge mass.
/// @throws std::invalid_argument Bad inputs or underlying grid size/window limits.
/// @par Example
/// @code
/// const auto r=dm::bayes::inla_scalar_reference({1,2,3},
///     dm::bayes::inla_gaussian_likelihood([](const auto&){return 1.;}),1,{},-6,8,2001);
/// std::print("{:.1f}\n",r.mean[0]); // 1.5
/// @endcode
/// @see INLA::conditional(), grid_posterior()
GridPosterior inla_scalar_reference(
    const std::vector<double>& y,
    const INLALikelihoodFn&    likelihood,
    double                     precision,
    const std::vector<double>& theta,
    double                     lower,
    double                     upper,
    std::size_t                points = 20001);
} // namespace datamunge::bayes
