#pragma once
/// @file adaptive_pspline.hpp
/// @brief Continuous hierarchical P-splines with recursive mesh growth and joint lambda tuning.
#include <datamunge/fda/hierarchical_bspline.hpp>
#include <string>
#include <vector>

/// @brief Functional data analysis, spline approximation and adaptive smoothers.
namespace datamunge::fda {
/// @brief Search controls for continuous HB/THB regression and hierarchical P-splines.
/// The domain is fixed before fitting. A common finest dyadic coefficient grid
/// defines the penalty, independent of the chosen hierarchical basis. Lambda zero
/// gives least squares. This API fits genuine hierarchical spaces, unlike the
/// independent residual patches of HierarchicalBSpline. No global risk optimum
/// is asserted. The fixed-lambda API uses a training-driven greedy path; the
/// joint API alternates validation tuning and recursive growth.
/// @par Example
/// @code
/// dm::fda::AdaptiveSplineOptions options;
/// options.lambda = 1e-4; options.max_splits = 3;
/// @endcode
/// @par References
/// - Giannelli, Jüttler and Speleers (2012), https://doi.org/10.1016/j.cagd.2012.03.025
/// - Eilers and Marx (1996), https://doi.org/10.1214/ss/1038425655
/// @see fit_adaptive_pspline(), fit_hierarchical_pspline()
struct AdaptiveSplineOptions {
  int degree{3};     ///< Polynomial degree, 1..5; simple interior knots give C^(degree-1).
  int root_basis{7}; ///< Root coefficient count, strictly greater than degree.
  int max_depth{4};  ///< Maximum binary cell depth, 0..9; finest basis capped at 1024 coefficients.
  int max_splits{8}; ///< Maximum accepted binary splits, 0..63, including the initial root split.
  int min_child_observations{
      12}; ///< Minimum training observations in each proposed half-open child.
  double lambda{
      0.0}; ///< Nonnegative multiplier of h_finest^-3 times squared boundary-corrected differences.
  double lower{0.0}; ///< Finite domain minimum, strictly below upper.
  double upper{
      1.0}; ///< Finite domain maximum; prediction outside the domain clamps to its endpoints.
  bool truncated{false}; ///< True selects THB coordinates; false selects classical untruncated HB.
};

/// @brief Controls for alternating mesh growth and smoothing-parameter selection.
/// Every accepted tree is retuned using the supplied validation observations.
/// Validation therefore influences subsequent growth; independent test data are
/// required to assess this complete procedure. The original fixed-lambda API
/// remains available when growth must depend on training responses alone.
/// @par Example
/// @code
/// dm::fda::JointAdaptiveSplineOptions search;
/// search.spline.max_depth=5; search.spline.max_splits=16;
/// search.curvature_priority=true; search.lambda_refinements=2;
/// @endcode
/// @see fit_joint_adaptive_pspline(), AdaptiveSplineOptions
struct JointAdaptiveSplineOptions {
  AdaptiveSplineOptions spline; ///< Domain/tree bounds; its lambda is ignored in favor of lambdas.
  std::vector<double>   lambdas{0, 1e-8, 1e-6, 1e-4, 1e-3, .01, .1, 1, 10};
  ///< Nonempty finite nonnegative seeds, at most 64; sorted/deduplicated internally.
  int  lambda_refinements{2};     ///< 0..8 rounds testing midpoints around the mesh's best lambda.
  bool curvature_priority{false}; ///< Rank positive-gain splits by width^4*mean(f''^2), requiring
                                  ///< degree >=2; false ranks local MSE gain.
};

/// @brief One smoothing-parameter evaluation on an accepted tree.
/// Entries include rejected numerical solves and all interpolation probes.
/// @par Example
/// @code
/// dm::fda::AdaptiveSplineTuning trial;
/// const bool usable=trial.fitted;
/// @endcode
struct AdaptiveSplineTuning {
  int    splits{0};         ///< Number of splits in the evaluated accepted tree.
  double lambda{0};         ///< Trial smoothing multiplier.
  double validation_mse{0}; ///< Held-out tuning MSE; infinity on numerical failure.
  bool   fitted{false};     ///< Whether fitting and scoring both succeeded.
  bool   selected{false};   ///< Whether this is the winning multiplier for this tree.
};

/// @brief One cell of a complete binary refinement tree, in creation order.
/// @par Example
/// @code
/// dm::fda::AdaptiveSplineCell cell;
/// const bool is_root = cell.parent == -1;
/// @endcode
struct AdaptiveSplineCell {
  int    id{0};        ///< Stable zero-based cell id, root 0.
  int    parent{-1};   ///< Parent id, or -1 for root.
  int    depth{0};     ///< Number of bisections from the root.
  double lower{0.0};   ///< Closed lower endpoint.
  double upper{1.0};   ///< Upper endpoint; local scoring is half-open except the domain maximum.
  bool   split{false}; ///< Whether both children have been created.
};

/// @brief Counterfactual refinement audit entry; includes unsuccessful numerical fits.
/// @par Example
/// @code
/// dm::fda::AdaptiveSplineCandidate candidate;
/// const bool was_selected = candidate.selected;
/// @endcode
struct AdaptiveSplineCandidate {
  int    step{0};         ///< One-based greedy iteration along the complete training path.
  int    cell{0};         ///< Eligible leaf being considered for bisection.
  int    observations{0}; ///< Training rows in the scored parent interval.
  double mse_before{0.0}; ///< Parent-local training MSE before the hypothetical split.
  double mse_after{0.0}; ///< Parent-local training MSE after globally refitting the proposed space.
  double gain{0.0};      ///< mse_before minus mse_after; not weighted by interval mass.
  bool   fitted{false};  ///< False means the proposed normal equations could not be solved.
  bool   selected{false}; ///< True for the winning eligible split under the requested priority.
  double lambda{0};       ///< Common multiplier for both sides of this counterfactual comparison.
  double mean_curvature_squared{0}; ///< Parent mean(f''^2) in joint searches, using exact spanwise
                                    ///< Gauss integration; zero otherwise.
  double priority{0};               ///< Local MSE gain, or width^4*mean(f''^2) in curvature mode.
};

/// @brief A fitted path prefix and its honest held-out validation score.
/// @par Example
/// @code
/// dm::fda::AdaptiveSplineStep step;
/// const int active_dimension = step.parameters;
/// @endcode
struct AdaptiveSplineStep {
  int    splits{0};     ///< Number of accepted binary bisections in this prefix.
  int    parameters{0}; ///< Active basis dimension; not an effective degrees-of-freedom estimate.
  double training_mse{0.0};   ///< Unpenalized training MSE across the entire domain.
  double validation_mse{0.0}; ///< Validation MSE; NaN for a fixed-tree fit without validation.
  double roughness{0.0};      ///< h_finest^-3 times squared boundary-corrected differences of
                              ///< finest-grid coefficients.
  double objective{0.0};      ///< training_mse + lambda * roughness.
  int    refined_cell{-1};    ///< Last split cell, or -1 at the root-only prefix.
  double lambda{0};           ///< Multiplier selected on this prefix; may vary in a joint search.
};

/// @brief Owned continuous fitted spline, selected tree, and complete refinement audit.
/// Predictions use an equivalent finest-grid B-spline representation, avoiding
/// evaluation-order dependence at cell boundaries. Const evaluation is thread-safe.
/// Path and candidates describe the entire searched path, including prefixes
/// beyond the validation-selected tree. The cells/splits fields describe only
/// the selected model. Fields are snapshots; editing them can invalidate the fit.
/// @par Example
/// @code
/// dm::fda::AdaptiveSplineOptions o; o.root_basis=4; o.max_depth=0;
/// auto fit=dm::fda::fit_hierarchical_pspline({0,.25,.5,.75,1}, {0,.25,.5,.75,1}, o);
/// const auto predictions=fit.predict({.1,.9});
/// @endcode
/// @see AdaptiveSplineOptions, fit_adaptive_pspline()
struct AdaptiveSplineFit {
  BSplineBasis        basis; ///< Owned common finest-grid basis, with repeated endpoint knots.
  std::vector<double> coefficients; ///< Finest-grid coefficients equivalent to the selected model.
  AdaptiveSplineOptions options;    ///< Exact options used by the fit.
  std::vector<int> splits; ///< Selected tree's sequential split ids; usable to replay the space.
  std::vector<AdaptiveSplineCell> cells; ///< Selected complete binary tree.
  std::vector<AdaptiveSplineStep> path;  ///< All successfully fitted training-path prefixes.
  std::vector<AdaptiveSplineCandidate>
      candidates; ///< Every sample-eligible counterfactual, in cell-id order.
  std::vector<AdaptiveSplineTuning>
      tuning; ///< Complete joint-search lambda audit, empty for fixed-lambda growth.
  int fit_evaluations{
      1}; ///< Total attempted fixed-space solves, including failed proposals/tuning fits.
  int    parameters{0};        ///< Dimension of the selected hierarchical space.
  int    failed_candidates{0}; ///< Numerically unsolved proposal/tuning solves along the search.
  double training_mse{0.0};    ///< Selected model's mean squared training error.
  double validation_mse{0.0};  ///< Selected validation score, NaN for a fixed-tree fit.
  double roughness{
      0.0}; ///< Selected finest-grid boundary-corrected difference penalty with h^-3 scaling.

  /// @brief Evaluate the fitted curve, clamping finite out-of-domain coordinates.
  /// @param x Finite query coordinates; empty is accepted.
  /// @return Owned fitted values, in query order.
  /// @throws std::invalid_argument For nonfinite queries or an invalid coefficient count.
  /// @par Example
  /// See the class example.
  std::vector<double> predict(const std::vector<double>& x) const;

  /// @brief Evaluate an analytic derivative of the fitted spline.
  /// Input coordinates are clamped before evaluation, as for predict(). These
  /// are derivatives of the interior spline evaluated at the clamped point.
  /// @param x Finite query coordinates; empty is accepted.
  /// @param order Derivative order, 0..degree, default 1; zero equals predict().
  /// @return Owned derivative values, in query order.
  /// @throws std::invalid_argument For an invalid order, query or coefficient count.
  /// @par Example
  /// @code
  /// dm::fda::AdaptiveSplineOptions o; o.root_basis=4; o.max_depth=0;
  /// auto fit=dm::fda::fit_hierarchical_pspline({0,.25,.5,.75,1}, {0,.25,.5,.75,1}, o);
  /// const auto slopes=fit.derivative({.2,.8}); // approximately {1,1}
  /// @endcode
  /// @par References
  /// - C. de Boor (1972), https://doi.org/10.1016/0021-9045(72)90080-9
  /// @see predict()
  std::vector<double> derivative(const std::vector<double>& x, int order = 1) const;

  /// @brief Save the selected interval refinement tree as a native SVG plot.
  /// @param path Output SVG filename; parent directory must exist.
  /// @param title Plot heading, default "Selected hierarchical spline mesh".
  /// @throws std::runtime_error For output/render failures.
  /// @par Example
  /// @code
  /// dm::fda::AdaptiveSplineOptions o; o.root_basis=4; o.max_depth=0;
  /// auto fit=dm::fda::fit_hierarchical_pspline({0,.25,.5,.75,1}, {0,.25,.5,.75,1}, o);
  /// fit.save_mesh_svg("mesh.svg");
  /// @endcode
  void save_mesh_svg(
      const std::string& path,
      const std::string& title = "Selected hierarchical spline mesh") const;
};

/// @brief Globally fit a fixed continuous hierarchical space with a common-grid P-spline penalty.
/// Uses active HB or THB functions on the supplied tree, maps them exactly to
/// the fixed finest dyadic grid, and minimizes MSE + lambda*h^-3*||G2*c_fine||².
/// G2 uses second divided differences at Greville sites, scaled by h². It is
/// ordinary D2 where both spacings equal h, and annihilates constants and physical
/// affine functions at clamped boundaries. For consecutive spacings a,b its
/// row is 2*h²/(a+b) times (1/a,-1/a-1/b,1/b).
/// HB/THB coordinates therefore define the same estimator on the same tree.
/// Finest-grid data assembly is O(n*p²), with dense space restriction/solution
/// thereafter. Normal-equation accumulation and Cholesky use long double;
/// no artificial identifying ridge is added.
/// Paired child creation aligns refinement-domain boundaries with parent grids.
/// @param x Finite training coordinates inside the options domain.
/// @param y Finite responses of equal nonzero length.
/// @param options Domain, degree, common penalty grid and fixed multiplier.
/// @param split_cells Sequential leaf ids to bisect, empty for the root-only space.
/// @return Owned fit; validation_mse is NaN. The tree never depends on y implicitly.
/// @throws std::invalid_argument For malformed controls/data or an invalid split sequence.
/// @throws std::runtime_error For unidentified/nonfinite normal equations.
/// @par Example
/// See AdaptiveSplineFit.
/// @par References
/// - Giannelli et al. (2012), https://doi.org/10.1016/j.cagd.2012.03.025
/// - Eilers and Marx (1996), https://doi.org/10.1214/ss/1038425655
/// @see fit_adaptive_pspline()
AdaptiveSplineFit fit_hierarchical_pspline(
    const std::vector<double>&   x,
    const std::vector<double>&   y,
    const AdaptiveSplineOptions& options     = {},
    const std::vector<int>&      split_cells = {});

/// @brief Greedily grow by the largest positive parent-local training-MSE drop; validate prefixes.
/// Each proposal bisects one eligible leaf into both children and globally
/// refits all active coefficients. Only training rows determine proposals,
/// scores and the growth order. Strictly positive gains are required; ties
/// within 1e-12*max(1,|gain|) select the lower cell id. Both children need the
/// requested minimum training count. Intervals are half-open except the domain
/// maximum. Validation chooses the lowest-MSE prefix, with the earlier/smaller
/// prefix winning ties at the same tolerance. No validation refit is performed.
/// The search is finite and greedy: it is not a global tree/risk optimum.
/// @param x Training coordinates inside the fixed domain.
/// @param y Matching training responses.
/// @param validation_x Independent finite validation coordinates, clamped for prediction.
/// @param validation_y Matching nonempty finite validation responses.
/// @param options Growth bounds, fixed lambda and coordinate representation.
/// @return Validation-selected fit plus all path scores and candidate decisions.
/// @throws std::invalid_argument For invalid data or options.
/// @throws std::runtime_error If the root fit is unidentified or nonfinite.
/// @par Example
/// @code
/// dm::fda::AdaptiveSplineOptions o; o.root_basis=4; o.max_depth=1; o.min_child_observations=2;
/// auto fit=dm::fda::fit_adaptive_pspline({0,.2,.4,.6,.8,1}, {0,.2,.4,.6,.8,1},
///                                     {.1,.5,.9}, {.1,.5,.9}, o);
/// @endcode
/// @see fit_hierarchical_pspline(), AdaptiveSplineCandidate
AdaptiveSplineFit fit_adaptive_pspline(
    const std::vector<double>&   x,
    const std::vector<double>&   y,
    const std::vector<double>&   validation_x,
    const std::vector<double>&   validation_y,
    const AdaptiveSplineOptions& options = {});
/// @brief Recursively refine a mesh while retuning lambda on every accepted tree.
/// On each tree, fit all seed lambdas plus the previous tree's chosen lambda.
/// Then test the two neighboring midpoints around the best evaluated lambda for
/// each refinement round: geometric means for positive brackets and arithmetic
/// halves beside zero. Select lowest validation MSE; ties favor the smaller
/// lambda, then the earlier/smaller prefix (tolerance 1e-12*max(1,abs(scores))).
/// Each leaf proposal is globally fitted at the CURRENT tree's tuned lambda, so
/// its local-MSE gain compares the same penalty on both sides. Require positive
/// gain and increased basis dimension. Rank by gain, or optionally by
/// width^4*mean(f''^2); curvature is integrated analytically with four-point Gauss
/// quadrature on each finest knot span. Priority ties favor lower cell ids.
/// After acceptance, retune lambda and repeat, even through worsening validation
/// prefixes, until the usual depth/sample/split bounds or no eligible gain.
/// Validation influences growth through lambda; it is a tuning set, not a test
/// estimate. Returned coefficients use training rows only. This is an alternating
/// finite search, with no global tree/lambda or test-risk optimality guarantee.
/// @param x Finite training coordinates inside the fixed domain.
/// @param y Matching finite training responses, nonempty.
/// @param validation_x Nonempty finite tuning coordinates, clamped for prediction.
/// @param validation_y Matching finite tuning responses.
/// @param options Tree bounds, lambda seeds, interpolation budget and split priority.
/// @return Lowest-validation-MSE prefix, with complete path, split and lambda audits.
/// @throws std::invalid_argument For malformed data, bounds, seeds or refinement budget.
/// @throws std::runtime_error If no root multiplier has an identified finite fit.
/// @par Example
/// @code
/// dm::fda::JointAdaptiveSplineOptions o; o.spline.min_child_observations=2;
/// o.spline.max_depth=2; o.spline.max_splits=2;
/// auto fit=dm::fda::fit_joint_adaptive_pspline({0,.2,.4,.6,.8,1}, {0,.1,.6,.9,.4,0},
///                                          {.1,.5,.9}, {.05,.75,.2}, o);
/// @endcode
/// @see fit_adaptive_pspline(), AdaptiveSplineFit::derivative()
AdaptiveSplineFit fit_joint_adaptive_pspline(
    const std::vector<double>&        x,
    const std::vector<double>&        y,
    const std::vector<double>&        validation_x,
    const std::vector<double>&        validation_y,
    const JointAdaptiveSplineOptions& options = {});
} // namespace datamunge::fda
