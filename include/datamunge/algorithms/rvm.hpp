#pragma once
/// @file rvm.hpp
/// @brief Checked sparse Bayesian kernel regression with matched final posterior uncertainty.
#include <cstddef>
#include <vector>
/// @brief Canonical checked native algorithm producers and owned outputs.
namespace datamunge::algorithms {
/// @brief Owned fitted RBF regressor with bias-first posterior means.
/// @par Example
/// @code
/// const dm::algorithms::RVMModel value{};
/// std::print("{}\n",value.beta); // 1
/// @endcode
struct RVMModel {
  std::vector<std::vector<double>> centers; ///< Owned complete training inputs.
  std::vector<double> weights;      ///< Bias followed by per-center means; inactive entries zero.
  std::vector<char>   relevant;     ///< One active marker per center; bias is always active.
  double              gamma{1};     ///< Positive finite RBF inverse squared-distance scale.
  double              beta{1};      ///< Positive finite fitted noise precision.
  std::size_t         iterations{}; ///< Actual completed hyperparameter updates.
  std::size_t         num_relevance_vectors{}; ///< Actual active nonbias center count.
};
/// @brief Complete geometry, update and capture policies; zero disables a size cap.
/// @par Example
/// @code
/// const dm::algorithms::RVMOptions value{};
/// std::print("{}\n",value.max_samples); // 256
/// @endcode
struct RVMOptions {
  double      gamma{1}; ///< RBF exp(-gamma*squared distance), positive finite.
  std::size_t max_iterations{
      200}; ///< Actual fixed-point update budget; zero gives initial-prior posterior.
  double prune_threshold{
      1e9}; ///< Active iff alpha below this threshold, required finite >1 with finite tenfold fallback.
  double      tolerance{1e-4};  ///< Nonnegative finite maximum log-alpha-change stopping tolerance.
  std::size_t max_samples{256}; ///< Complete training-row cap.
  std::size_t max_matrix_entries{1000000}; ///< Complete design or square posterior matrix cap.
  std::size_t max_updates{1000};           ///< Complete requested-update cap.
  std::size_t max_trace_entries{
      1000000}; ///< Iteration headers and five captured basis-length vectors.
};
/// @brief One actual solved posterior followed by one actual hyperparameter update.
/// @par Example
/// @code
/// const dm::algorithms::RVMStep value{};
/// std::print("{}\n",value.iteration); // 0
/// @endcode
struct RVMStep {
  std::size_t         iteration{}; ///< One-based actual update.
  std::vector<double> alpha_before; ///< Actual solve precisions, updated precisions, mean, covariance diagonal and gamma_j.
  std::vector<double> alpha_after; ///< Actual solve precisions, updated precisions, mean, covariance diagonal and gamma_j.
  std::vector<double> mean; ///< Actual solve precisions, updated precisions, mean, covariance diagonal and gamma_j.
  std::vector<double> covariance_diagonal; ///< Actual solve precisions, updated precisions, mean, covariance diagonal and gamma_j.
  std::vector<double> effective_parameters; ///< Actual solve precisions, updated precisions, mean, covariance diagonal and gamma_j.
  double beta_before{}; ///< Actual solve precision, updated precision, preupdate SSE and maximum alpha change.
  double beta_after{}; ///< Actual solve precision, updated precision, preupdate SSE and maximum alpha change.
  double squared_error{}; ///< Actual solve precision, updated precision, preupdate SSE and maximum alpha change.
  double maximum_log_change{}; ///< Actual solve precision, updated precision, preupdate SSE and maximum alpha change.
};
/// @brief Owned fitted mean and covariance matched to the final active basis and reported precisions.
/// @par Example
/// @code
/// const dm::algorithms::RVMRun value{};
/// std::print("{}\n",value.posterior_solves); // 0
/// @endcode
struct RVMRun {
  RVMModel            model; ///< Actual final active posterior mean model.
  std::vector<double> alpha; ///< Final full bias-first precisions, including inactive centers.
  std::vector<std::vector<double>>
      covariance; ///< Full expanded final covariance, inactive rows/columns zero.
  std::size_t
       posterior_solves{}; ///< Actual factorizations, including the final active-basis refit.
  bool converged{};        ///< Actual update stopping criterion met after at least three updates.
  std::vector<RVMStep> steps; ///< Optional actual update histories.
};
/// @brief Actual matched Gaussian predictive quantities at one input.
/// @par Example
/// @code
/// const dm::algorithms::RVMPrediction value{};
/// std::print("{}\n",value.mean); // 0
/// @endcode
struct RVMPrediction {
  double mean{}; ///< Mean, phi^T Sigma phi, 1/beta, and their variance sum.
  double latent_variance{}; ///< Mean, phi^T Sigma phi, 1/beta, and their variance sum.
  double noise_variance{}; ///< Mean, phi^T Sigma phi, 1/beta, and their variance sum.
  double variance{}; ///< Mean, phi^T Sigma phi, 1/beta, and their variance sum.
};
/// @brief Owned original noiseless sine fixture and its evaluation grid.
/// @par Example
/// @code
/// const dm::algorithms::RVMExample value{};
/// std::print("{}\n",value.inputs.size()); // 0
/// @endcode
struct RVMExample {
  std::vector<std::vector<double>> inputs; ///< Forty-one training and 201 evaluation one-dimensional inputs.
  std::vector<std::vector<double>> grid; ///< Forty-one training and 201 evaluation one-dimensional inputs.
  std::vector<double> targets; ///< Exact computed sine at the corresponding coordinates; no sampled noise.
  std::vector<double> truth; ///< Exact computed sine at the corresponding coordinates; no sampled noise.
};
/// @brief Fit checked sparse Bayesian RBF regression.
/// Builds a bias plus one Gaussian basis per training point. Canonical long-double Cholesky solves Sigma=(diag(alpha)+beta*Phi^T*Phi)^-1 and mu=beta*Sigma*Phi^T*t. Actual gamma_j=1-alpha_j*Sigma_jj updates alpha_j=gamma_j/mu_j^2 and beta=(n-sum gamma)/(SSE+1e-12); nonpositive/nonfinite alpha uses ten times prune_threshold and beta uses inverse(target sample variance+1e-9). Initial alpha=1 and beta=1/(0.1*sample variance+1e-9). Tiny numerical gamma_j excursions are clamped to [0,1]; substantial excursions fail. Stops on maximum log-alpha change after at least three updates or budget. After final selection alpha<threshold, ALWAYS refits the active basis plus bias using the final reported alpha and beta. This fixes historical stale means and supplies covariance matching the pruned mean; finite-budget legacy predictions may change. Zero updates gives the initial posterior. Empty/ragged/nonfinite inputs are rejected. No global evidence optimum, sparsity level, calibration or convergence guarantee. Each solve O((n+1)^3), working O((n+1)^2); trace O(updates*(n+1)). Complete geometry and capture checked before work/copies. Borrowed input unchanged; calls share no mutable state.
/// @param inputs Nonempty rectangular finite n-by-d inputs with d>0.
/// @param targets Exactly n finite regression targets.
/// @param options Kernel scale, fixed-point controls and complete geometry/history policies.
/// @return Owned actual final active posterior, full precisions, solve count and optional update histories.
/// @throws std::invalid_argument Invalid controls, shape or finite-data requirement.
/// @throws std::length_error Complete geometry/update/capture policy exceeded.
/// @throws std::domain_error Posterior is numerically non-SPD or an intermediate/output is unrepresentable.
/// @par Example
/// @code
/// dm::algorithms::RVMOptions o;o.max_iterations=0;
/// auto r=dm::algorithms::rvm_train_run({{0}},{0},o);
/// std::print("{} {}\n",r.model.weights,r.posterior_solves); // [0, 0] 1
/// @endcode
/// @par References
/// - Tipping (2001), Sparse Bayesian Learning and the Relevance Vector Machine. https://jmlr.org/papers/v1/tipping01a.html
/// @see rvm_predictive(), rvm_train()
RVMRun rvm_train_run(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<double>&              targets,
    const RVMOptions&                       options = {});
/// @brief Fit checked sparse Bayesian RBF regression retaining actual fixed-point updates.
/// Builds a bias plus one Gaussian basis per training point. Canonical long-double Cholesky solves Sigma=(diag(alpha)+beta*Phi^T*Phi)^-1 and mu=beta*Sigma*Phi^T*t. Actual gamma_j=1-alpha_j*Sigma_jj updates alpha_j=gamma_j/mu_j^2 and beta=(n-sum gamma)/(SSE+1e-12); nonpositive/nonfinite alpha uses ten times prune_threshold and beta uses inverse(target sample variance+1e-9). Initial alpha=1 and beta=1/(0.1*sample variance+1e-9). Tiny numerical gamma_j excursions are clamped to [0,1]; substantial excursions fail. Stops on maximum log-alpha change after at least three updates or budget. After final selection alpha<threshold, ALWAYS refits the active basis plus bias using the final reported alpha and beta. This fixes historical stale means and supplies covariance matching the pruned mean; finite-budget legacy predictions may change. Zero updates gives the initial posterior. Empty/ragged/nonfinite inputs are rejected. No global evidence optimum, sparsity level, calibration or convergence guarantee. Each solve O((n+1)^3), working O((n+1)^2); trace O(updates*(n+1)). Complete geometry and capture checked before work/copies. Borrowed input unchanged; calls share no mutable state.
/// @param inputs Nonempty rectangular finite n-by-d inputs with d>0.
/// @param targets Exactly n finite regression targets.
/// @param options Kernel scale, fixed-point controls and complete geometry/history policies.
/// @return Owned actual final active posterior, full precisions, solve count and optional update histories.
/// @throws std::invalid_argument Invalid controls, shape or finite-data requirement.
/// @throws std::length_error Complete geometry/update/capture policy exceeded.
/// @throws std::domain_error Posterior is numerically non-SPD or an intermediate/output is unrepresentable.
/// @par Example
/// @code
/// dm::algorithms::RVMOptions o;o.max_iterations=0;
/// auto r=dm::algorithms::rvm_train_trace({{0}},{0},o);
/// std::print("{} {}\n",r.model.weights,r.posterior_solves); // [0, 0] 1
/// @endcode
/// @par References
/// - Tipping (2001), Sparse Bayesian Learning and the Relevance Vector Machine. https://jmlr.org/papers/v1/tipping01a.html
/// @see rvm_predictive(), rvm_train()
RVMRun rvm_train_trace(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<double>&              targets,
    const RVMOptions&                       options = {});
/// @brief Fit the checked posterior through the preserved legacy signature.
/// Delegates to the same native producer with default geometry policies and final active-basis refit.
/// @param X Nonempty rectangular finite training inputs.
/// @param t Finite target for every row.
/// @param gamma Positive finite RBF inverse squared-distance scale; default 1.
/// @param max_iters Actual update budget; default 200.
/// @param prune_threshold Finite positive threshold >1; default 1e9.
/// @param tol Finite nonnegative stopping tolerance; default 1e-4.
/// @return Owned mean model with matched final precisions.
/// @throws std::invalid_argument Invalid inputs/controls.
/// @throws std::length_error Default geometry/update policy exceeded.
/// @throws std::domain_error Numerically invalid posterior or output.
/// @par Example
/// @code
/// auto m=dm::algorithms::rvm_train({{0}},{0},1,0);
/// std::print("{}\n",dm::algorithms::rvm_predict(m,{0})); // 0
/// @endcode
/// @par References
/// - Tipping (2001), Sparse Bayesian Learning and the Relevance Vector Machine. https://jmlr.org/papers/v1/tipping01a.html
/// @see rvm_train_run()
RVMModel rvm_train(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>&              t,
    double                                  gamma           = 1,
    std::size_t                             max_iters       = 200,
    double                                  prune_threshold = 1e9,
    double                                  tol             = 1e-4);
/// @brief Evaluate a checked bias-first RBF mean model.
/// Validates complete model geometry, finite weights/centers, positive gamma/beta, active markers/count and input shape before indexing. Empty/unfitted models are invalid. Borrowed storage unchanged; no shared state.
/// @param model Borrowed fitted canonical mean model.
/// @param x Finite input matching center dimension.
/// @return Finite actual predicted mean.
/// @throws std::invalid_argument Malformed model/input.
/// @throws std::domain_error Unrepresentable mean.
/// @par Example
/// @code
/// auto m=dm::algorithms::rvm_train({{0}},{0},1,0);
/// std::print("{}\n",dm::algorithms::rvm_predict(m,{0})); // 0
/// @endcode
/// @par References
/// - Tipping (2001), Sparse Bayesian Learning and the Relevance Vector Machine. https://jmlr.org/papers/v1/tipping01a.html
/// @see rvm_predictive()
double rvm_predict(const RVMModel& model, const std::vector<double>& x);
/// @brief Evaluate actual matched posterior mean and Gaussian predictive variance.
/// Uses the final active covariance stored by the producing run. Latent variance is phi^T Sigma phi; observation variance adds 1/beta. Validates covariance shape, finiteness and symmetry; arbitrary manual edits cannot be certified as a fitted posterior. Tiny negative roundoff is clamped, substantial negative variance fails. These are model-conditional quantities and do not establish empirical uncertainty calibration.
/// @param run Borrowed canonically fitted mean and matched covariance.
/// @param x Finite correctly shaped input.
/// @return Actual mean, latent variance, noise variance and total variance.
/// @throws std::invalid_argument Malformed model, input or covariance.
/// @throws std::domain_error Invalid or unrepresentable variance/mean.
/// @par Example
/// @code
/// dm::algorithms::RVMOptions o;o.max_iterations=0;
/// auto r=dm::algorithms::rvm_train_run({{0}},{0},o);
/// std::print("{}\n",dm::algorithms::rvm_predictive(r,{0}).mean); // 0
/// @endcode
/// @par References
/// - Tipping (2001), Sparse Bayesian Learning and the Relevance Vector Machine. https://jmlr.org/papers/v1/tipping01a.html
/// @see rvm_train_run()
RVMPrediction rvm_predictive(const RVMRun& run, const std::vector<double>& x);
/// @brief Return the original noiseless sine study inputs through the native library.
/// Training x=2*pi*i/40 for i=0..40 and grid x=2*pi*i/200 for i=0..200; target/truth are sin(x). Uses no random noise or shared state.
/// @return Owned inputs, targets, grid and exact computed sine truth.
/// @par Example
/// @code
/// auto e=dm::algorithms::rvm_sine_example();
/// std::print("{} {}\n",e.inputs.size(),e.grid.size()); // 41 201
/// @endcode
/// @par References
/// - Tipping (2001), Sparse Bayesian Learning and the Relevance Vector Machine. https://jmlr.org/papers/v1/tipping01a.html
/// @see rvm_train_trace()
RVMExample rvm_sine_example();
} // namespace datamunge::algorithms
