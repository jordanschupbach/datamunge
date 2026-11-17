#pragma once
/// @file self_organizing_map.hpp
/// @brief Checked online competitive maps, actual initialization/schedules and scalar reference histories.
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Owned row-major rectangular competitive codebook.
/// @par Example
/// @code
/// const dm::algorithms::SelfOrganizingMap value{};
/// std::print("{}\n",value.rows); // 0
/// @endcode
struct SelfOrganizingMap {
  std::size_t                      rows{};  ///< Positive fitted grid rows.
  std::size_t                      cols{};  ///< Positive fitted grid columns.
  std::size_t                      dim{};   ///< Positive fitted input dimension.
  std::vector<std::vector<double>> weights; ///< Actual row-major codebook vectors.

  /// @brief Resolve a checked grid address into the owned row-major codebook.
  /// @param r In-range zero-based row.
  /// @param c In-range zero-based column.
  /// @return Actual flat address.
  /// @throws std::out_of_range Grid address outside declared geometry.
  /// @throws std::length_error Declared row-major geometry overflows size_t.
  /// @par Example
  /// @code
  /// dm::algorithms::SelfOrganizingMap m{2,3,1,{}};
  /// std::print("{}\n",m.index(1,2)); // 5
  /// @endcode
  std::size_t index(std::size_t r, std::size_t c) const;
};
/// @brief Original online map controls and deterministic local initialization/sample seed.
/// @par Example
/// @code
/// const dm::algorithms::SOMParameters value{};
/// std::print("{}\n",value.iterations); // 1000
/// @endcode
struct SOMParameters {
  std::size_t rows{5};           ///< Positive grid rows.
  std::size_t cols{5};           ///< Positive grid columns.
  std::size_t iterations{1000};  ///< Complete single-sample update budget, including zero.
  double      learning_rate{.5}; ///< Finite initial alpha in (0,1], exponentially decaying.
  double radius{0}; ///< Finite positive initial sigma, or nonpositive selects max(rows,cols)/2.
  std::uint64_t seed{0}; ///< Local mt19937_64 initialization/sample stream.
};
/// @brief Complete geometry, potential scalar-update and optional history bounds; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::SOMOptions value{};
/// std::print("{}\n",value.max_units); // 100000
/// @endcode
struct SOMOptions {
  std::size_t max_samples{1000000};    ///< Complete input sample count.
  std::size_t max_units{100000};       ///< Complete row*column codebook units.
  std::size_t max_iterations{1000000}; ///< Complete requested single-sample updates.
  std::size_t max_elements{1000000};   ///< Complete input or codebook scalar entries.
  std::size_t max_scalar_updates{
      20000000}; ///< Complete iterations*units*dimensions potential coordinate updates.
  std::size_t max_trace_entries{
      1000000}; ///< Actual step headers, two full codebook copies and influence scalars.
};
/// @brief Owned actual initialization and complete preselected sample/decay schedule, with no fitted updates.
/// @par Example
/// @code
/// const dm::algorithms::SOMPlan value{};
/// std::print("{}\n",value.sample_indices.size()); // 0
/// @endcode
struct SOMPlan {
  SelfOrganizingMap initial; ///< Actual sample-initialized codebook before any fitting.
  std::vector<std::size_t>
      initialization_indices; ///< Actual data row selected for every initial codebook unit.
  std::vector<std::size_t>
                      sample_indices; ///< Actual subsequent data-row draw for each planned update.
  std::vector<double> learning_rates; ///< Actual alpha0*exp(-t/T) schedule.
  std::vector<double> radii;          ///< Actual sigma0*exp(-(t/T)*log1p(sigma0)) schedule.
};
/// @brief One actual scalar reference update, suitable for independent tensor-training checks.
/// @par Example
/// @code
/// const dm::algorithms::SOMStep value{};
/// std::print("{}\n",value.iteration); // 0
/// @endcode
struct SOMStep {
  std::size_t iteration{};     ///< Actual zero-based update.
  std::size_t sample{};        ///< Actual selected input row.
  std::size_t winner{};        ///< Actual lowest-index minimum squared-distance codebook unit.
  double      learning_rate{}; ///< Actual decayed alpha.
  double      radius{};        ///< Actual decayed sigma.
  std::vector<std::vector<double>> before; ///< Actual complete codebook before this update.
  std::vector<std::vector<double>> after;  ///< Actual complete codebook after this update.
  std::vector<double> influence; ///< Actual grid Gaussian influences; below1e-6 set to zero.
};
/// @brief Owned actual scalar reference codebook and selected competitive-learning work/history.
/// @par Example
/// @code
/// const dm::algorithms::SOMRun value{};
/// std::print("{}\n",value.updates); // 0
/// @endcode
struct SOMRun {
  SelfOrganizingMap model;          ///< Actual final codebook.
  std::size_t       updates{};      ///< Actual completed single-sample updates.
  std::size_t distance_terms{};     ///< Actual squared-coordinate terms tested in winner searches.
  std::size_t coordinate_updates{}; ///< Actual coordinates changed by active neighborhood updates.
  std::vector<SOMStep> steps;       ///< Actual optional scalar reference histories.
};
/// @brief Owned original three-blob fixture; labels are presentation-only, never supplied to training.
/// @par Example
/// @code
/// const dm::algorithms::SOMExample value{};
/// std::print("{}\n",value.inputs.size()); // 0
/// @endcode
struct SOMExample {
  std::vector<std::vector<double>> inputs;  ///< Actual240 two-dimensional samples,80 per center.
  std::vector<int>                 labels;  ///< Original generating-blob identities0,1,2.
  std::vector<std::vector<double>> centers; ///< Original fixed centers(0,0),(10,1),(4,9).
};
/// @brief Return checked squared Euclidean distance for two complete finite vectors.
/// Equal empty vectors give zero. Uses the historical direct sum when finite and a widened fallback before rejecting a squared result outside finite double range. Inputs unchanged, O(dim) work and constant auxiliary storage.
/// @param a Borrowed finite first vector.
/// @param b Borrowed finite equal-length second vector.
/// @return Finite squared distance.
/// @throws std::invalid_argument Unequal dimensions or nonfinite values.
/// @throws std::overflow_error Squared distance outside finite double range.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::som_sq_dist({0,0},{3,4})); // 25
/// @endcode
/// @par References
/// - Kohonen (1982), Self-organized formation of topologically correct feature maps, Biological Cybernetics43,59-69. doi:10.1007/BF00337288.
/// @see som_bmu()
double som_sq_dist(const std::vector<double>& a, const std::vector<double>& b);
/// @brief Select the checked lowest-index best-matching map unit.
/// Validates positive complete model geometry and all finite weights/input before searching. Strict distance comparisons keep the lowest row-major unit on ties. No mutation/shared state. O(units*dimensions) time including validation.
/// @param som Borrowed complete finite fitted codebook.
/// @param input Borrowed finite vector matching positive model dimension.
/// @return Actual winning row and column.
/// @throws std::invalid_argument Malformed model/input geometry or nonfinite values.
/// @throws std::length_error Complete geometry exceeds capacity/default policies.
/// @throws std::overflow_error A required squared distance is unrepresentable.
/// @par Example
/// @code
/// dm::algorithms::SelfOrganizingMap m{1,2,1,{{0},{2}}};auto b=dm::algorithms::som_bmu(m,{1});
/// std::print("{} {}\n",b.first,b.second); // 0 0
/// @endcode
/// @par References
/// - Kohonen (1982), Self-organized formation of topologically correct feature maps, Biological Cybernetics43,59-69. doi:10.1007/BF00337288.
/// @see som_training_plan()
std::pair<std::size_t, std::size_t> som_bmu(
    const SelfOrganizingMap&   som,
    const std::vector<double>& input);
/// @brief Prepare actual original initialization draws and complete sample/decay controls without fitting.
/// Validates positive finite rectangular data, positive grid, alpha in(0,1], finite radius and complete geometry/update bounds. Nonpositive radius selects max(rows,cols)/2. Uses one local mt19937_64 and uniform_int_distribution<size_t> for all initial-unit draws followed by all update-sample draws, preserving historical draw order. Zero updates still initializes every unit and returns empty schedules. Stable log1p evaluates the same exponential radius schedule near zero. Complete potential update geometry is a policy bound, not an observed-work counter. Inputs borrowed unchanged; output owns initial codebook, ids and schedules; no model updates hidden here.
/// @param data Nonempty finite rectangular input rows with positive dimension.
/// @param params Complete map/decay/seed controls.
/// @param options Complete data/codebook/schedule/potential-work bounds.
/// @return Owned actual initialization and planned sample/decay controls, without fitting.
/// @throws std::invalid_argument Malformed finite data or controls.
/// @throws std::length_error Complete geometry, index or policy/capacity exceeded.
/// @throws std::overflow_error A positive scheduled rate/radius becomes unrepresentable.
/// @par Example
/// @code
/// dm::algorithms::SOMParameters p;p.rows=1;p.cols=2;p.iterations=0;
/// auto r=dm::algorithms::som_training_plan({{0},{2}},p);
/// std::print("{} {}\n",r.initial.weights.size(),r.sample_indices.size()); // 2 0
/// @endcode
/// @par References
/// - Kohonen (1982), Self-organized formation of topologically correct feature maps, Biological Cybernetics43,59-69. doi:10.1007/BF00337288.
/// @see som_train_run(), dm::nn::som_tensor_plan()
SOMPlan som_training_plan(
    const std::vector<std::vector<double>>& data,
    const SOMParameters&                    params,
    const SOMOptions&                       options = {});
/// @brief Execute checked original scalar online SOM reference updates.
/// Uses the actual native initialization/sample plan and lowest-index squared-distance winner. Applies alpha*h*(x-w) to every unit with h=exp(-grid_distance^2/(2*sigma^2)) at least1e-6. Rates/radii follow the explicit plan; no objective monotonicity, global minimum or guaranteed topology preservation is asserted. Empty/ragged/nonfinite data and malformed controls reject before fitting. Zero updates returns the actual initialized model. Complete potential work and optional history geometry checked before updates. Time O(iterations*units*dimension); intrinsic codebook/plans own their values; optional histories own full actual before/after weights and influences. This scalar producer is retained for compatibility/reference checks; neural-network reports demonstrate native tensor leaves/objectives and an explicit optimizer loop instead.
/// @param data Borrowed complete finite training rows.
/// @param params Complete map/update controls.
/// @param options Complete intrinsic/potential-work/history bounds.
/// @return Owned actual final scalar codebook, observed selected work and optional actual updates.
/// @throws std::invalid_argument Malformed finite data or controls.
/// @throws std::length_error Complete geometry/work/history policy or capacity exceeded.
/// @throws std::overflow_error Required distance, schedule or updated weight is unrepresentable.
/// @par Example
/// @code
/// dm::algorithms::SOMParameters p;p.rows=p.cols=1;p.iterations=1;
/// auto r=dm::algorithms::som_train_run({{2}},p);
/// std::print("{} {}\n",r.model.weights[0][0],r.updates); // 2 1
/// @endcode
/// @par References
/// - Kohonen (1982), Self-organized formation of topologically correct feature maps, Biological Cybernetics43,59-69. doi:10.1007/BF00337288.
/// @see som_training_plan(), dm::nn::som_neighborhood_objective()
SOMRun som_train_run(
    const std::vector<std::vector<double>>& data,
    const SOMParameters&                    params,
    const SOMOptions&                       options = {});
/// @brief Execute checked original scalar online SOM reference updates with actual producing histories.
/// Uses the actual native initialization/sample plan and lowest-index squared-distance winner. Applies alpha*h*(x-w) to every unit with h=exp(-grid_distance^2/(2*sigma^2)) at least1e-6. Rates/radii follow the explicit plan; no objective monotonicity, global minimum or guaranteed topology preservation is asserted. Empty/ragged/nonfinite data and malformed controls reject before fitting. Zero updates returns the actual initialized model. Complete potential work and optional history geometry checked before updates. Time O(iterations*units*dimension); intrinsic codebook/plans own their values; optional histories own full actual before/after weights and influences. This scalar producer is retained for compatibility/reference checks; neural-network reports demonstrate native tensor leaves/objectives and an explicit optimizer loop instead.
/// @param data Borrowed complete finite training rows.
/// @param params Complete map/update controls.
/// @param options Complete intrinsic/potential-work/history bounds.
/// @return Owned actual final scalar codebook, observed selected work and optional actual updates.
/// @throws std::invalid_argument Malformed finite data or controls.
/// @throws std::length_error Complete geometry/work/history policy or capacity exceeded.
/// @throws std::overflow_error Required distance, schedule or updated weight is unrepresentable.
/// @par Example
/// @code
/// dm::algorithms::SOMParameters p;p.rows=p.cols=1;p.iterations=1;
/// auto r=dm::algorithms::som_train_trace({{2}},p);
/// std::print("{} {}\n",r.model.weights[0][0],r.updates); // 2 1
/// @endcode
/// @par References
/// - Kohonen (1982), Self-organized formation of topologically correct feature maps, Biological Cybernetics43,59-69. doi:10.1007/BF00337288.
/// @see som_training_plan(), dm::nn::som_neighborhood_objective()
SOMRun som_train_trace(
    const std::vector<std::vector<double>>& data,
    const SOMParameters&                    params,
    const SOMOptions&                       options = {});
/// @brief Return the checked scalar reference map through the preserved signature.
/// Same actual producer/default bounds without optional retained updates.
/// @param data Complete finite nonempty rectangular input.
/// @param params Complete validated controls.
/// @return Owned actual fitted map.
/// @throws std::invalid_argument Malformed data/controls.
/// @throws std::length_error Default complete policy/capacity exceeded.
/// @throws std::overflow_error Unrepresentable actual computation.
/// @par Example
/// @code
/// dm::algorithms::SOMParameters p;p.rows=p.cols=1;p.iterations=0;
/// std::print("{}\n",dm::algorithms::som_train({{2}},p).weights[0][0]); // 2
/// @endcode
/// @par References
/// - Kohonen (1982), Self-organized formation of topologically correct feature maps, Biological Cybernetics43,59-69. doi:10.1007/BF00337288.
/// @see som_train_run()
SelfOrganizingMap som_train(
    const std::vector<std::vector<double>>& data,
    const SOMParameters&                    params);
/// @brief Measure mean Euclidean distance to actual checked best-matching units.
/// Validates the entire positive finite model and complete finite input geometry; empty data gives zero after model validation. Uses widened accumulation of actual Euclidean distances; selected winner uses squared distance. O(samples*units*dimensions) work, constant auxiliary storage apart from validation. This training quantization summary is not a held-out generalization or topology certificate.
/// @param som Borrowed complete finite codebook.
/// @param data Borrowed finite rows matching positive model dimension; empty allowed.
/// @return Finite actual mean distance, zero for no observations.
/// @throws std::invalid_argument Malformed model/data.
/// @throws std::length_error Default complete input/model/potential-distance policy exceeded.
/// @throws std::overflow_error A required distance or summary is unrepresentable.
/// @par Example
/// @code
/// dm::algorithms::SelfOrganizingMap m{1,1,1,{{0}}};
/// std::print("{}\n",dm::algorithms::som_quantization_error(m,{{2},{4}})); // 3
/// @endcode
/// @par References
/// - Kohonen (1982), Self-organized formation of topologically correct feature maps, Biological Cybernetics43,59-69. doi:10.1007/BF00337288.
/// @see som_bmu()
double som_quantization_error(
    const SelfOrganizingMap&                som,
    const std::vector<std::vector<double>>& data);
/// @brief Return the complete original three-blob SOM study fixture.
/// Uses local mt19937_64 seed7, normal_distribution<double>(0,.4), and eighty successive x/y perturbations at each of (0,0),(10,1),(4,9). Labels are generating identities for plots only. No shared random state or training occurs.
/// @return Owned original inputs, generating labels and exact fixed centers.
/// @par Example
/// @code
/// auto e=dm::algorithms::som_three_blob_example();
/// std::print("{} {}\n",e.inputs.size(),e.labels.size()); // 240 240
/// @endcode
/// @par References
/// - Kohonen (1982), Self-organized formation of topologically correct feature maps, Biological Cybernetics43,59-69. doi:10.1007/BF00337288.
/// @see som_training_plan()
SOMExample som_three_blob_example();
} // namespace datamunge::algorithms
