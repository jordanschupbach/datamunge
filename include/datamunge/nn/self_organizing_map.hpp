#pragma once
/// @file self_organizing_map.hpp
/// @brief Native tensor competitive-map objectives for explicit optimizer training.
#include <datamunge/algorithms/self_organizing_map.hpp>
#include <datamunge/nn/tensor.hpp>
#include <datamunge/plot/plot.hpp>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::nn {
/// @brief Owned data/initialization tensors and explicit actual sampling/decay controls, without fitting.
/// @par Example
/// @code
/// const dm::nn::SOMTensorPlan value{};
/// std::print("{}\n",value.inputs.defined()); // false
/// @endcode
struct SOMTensorPlan {
  Tensor inputs; ///< Owned Float64 CPU input[N,D].
  Tensor
      initial_weights; ///< Owned nonlearnable Float64 CPU initial codebook[K,D]; caller explicitly enables gradients.
  Tensor grid; ///< Owned fixed Float64 CPU row/column topology[K,2].
  std::vector<std::size_t>
      initialization_indices; ///< Actual original data rows selected for initial units.
  std::vector<std::size_t> sample_indices; ///< Actual original subsequent sampled row ids.
  std::vector<double>      learning_rates; ///< Actual alpha per explicit optimizer step.
  std::vector<double>      radii;          ///< Actual fixed sigma per explicit objective.
};
/// @brief Actual native tensor winner, neighborhood and differentiable frozen-winner objective.
/// @par Example
/// @code
/// const dm::nn::SOMTensorObjective value{};
/// std::print("{}\n",value.objective.defined()); // false
/// @endcode
struct SOMTensorObjective {
  std::size_t winner{};          ///< Actual lowest-index argmin of current squared distances.
  Tensor      squared_distances; ///< Actual Float64 unit-to-sample squared distances[K].
  Tensor      influence; ///< Actual fixed-grid Gaussian weights[K], values below1e-6 zeroed.
  Tensor objective; ///< Actual scalar half-sum h_k*||w_k-x||^2; preserves the codebook gradient.
};
/// @brief Prepare native tensor data/codebook/topology and actual original sampling controls without fitting.
/// Delegates validation and random draw order to the canonical som_training_plan, then creates owned Float64 CPU tensors. Initial weights are ordinary untracked values; the report/user explicitly calls requires_grad_ and constructs an optimizer. All subsequent objective, winner and inference computations use native tensor operations. No training loop is hidden in this helper. O(input entries+codebook entries+iterations) work/storage. Source borrowed unchanged; independent calls share no mutable state.
/// @param data Nonempty finite rectangular native input rows.
/// @param params Complete original grid/update/decay/seed controls.
/// @param options Complete input/codebook/schedule geometry bounds.
/// @return Owned input/initialization/topology tensors and actual sampling/rate/radius plan.
/// @throws std::invalid_argument Malformed finite data/controls.
/// @throws std::length_error Complete geometry/policy or tensor index capacity exceeded.
/// @throws std::overflow_error A positive schedule is unrepresentable.
/// @par Example
/// @code
/// dm::algorithms::SOMParameters p;p.rows=p.cols=1;p.iterations=0;
/// auto plan=dm::nn::som_tensor_plan({{2}},p);
/// std::print("{}\n",plan.initial_weights.item()); // 2
/// @endcode
/// @par References
/// - Kohonen (1982), Self-organized formation of topologically correct feature maps, Biological Cybernetics43,59-69. doi:10.1007/BF00337288; frozen-winner weighted quadratic and explicit SGD formulation here.
/// @see som_neighborhood_objective()
SOMTensorPlan som_tensor_plan(
    const std::vector<std::vector<double>>& data,
    const algorithms::SOMParameters&        params,
    const algorithms::SOMOptions&           options = {});
/// @brief Build the actual native tensor frozen-winner neighborhood objective for one explicit optimizer step.
/// Requires finite CPU Float64 weights[K,D], sample[D] and fixed untracked grid[K,2], with K,D positive and finite positive radius. Native squared_difference/sum/argmin compute the lowest-index best unit; gather selects its fixed grid position. Native Gaussian influences below1e-6 become zero, matching the scalar reference cutoff. Half-sum h_k*||weights_k-sample||^2 has gradient h_k*(weights_k-sample); a plain SGD step at alpha reproduces the competitive update. Winner selection is discrete and held fixed during this objective derivative; this is not a smooth global objective or monotone-descent guarantee across winner/radius/sample changes. Grid must not require gradients. Sample/weights may retain gradients; inputs are not mutated. Complete K*D geometry is checked before tensor intermediates; actual distance/objective overflow rejects. O(K*D) tensor work/storage, owned outputs retain their autograd references. This function computes an objective and does not fit the model.
/// @param weights Finite CPU Float64 current codebook[K,D], usually an explicit learnable leaf.
/// @param sample Finite CPU Float64 input[D].
/// @param grid Fixed finite untracked Float64 CPU coordinates[K,2].
/// @param radius Finite positive fixed grid-neighborhood sigma.
/// @param max_elements Complete intermediate K*D element cap; default1000000, zero disables.
/// @return Actual winner, unit distances/influences and differentiable weighted scalar objective.
/// @throws std::invalid_argument Invalid shape/device/dtype/finite inputs, tracked grid or radius.
/// @throws std::length_error Complete intermediate element policy exceeded.
/// @throws std::overflow_error Shape arithmetic or actual distance/objective is unrepresentable.
/// @par Example
/// @code
/// auto w=dm::nn::from_vector(std::vector<double>{0,2},{2,1}).requires_grad_();
/// auto x=dm::nn::from_vector(std::vector<double>{1});
/// auto grid=dm::nn::from_vector(std::vector<double>{0,0,0,1},{2,2});
/// auto s=dm::nn::som_neighborhood_objective(w,x,grid,1);
/// std::print("{}\n",s.winner); // 0
/// @endcode
/// @par References
/// - Kohonen (1982), Self-organized formation of topologically correct feature maps, Biological Cybernetics43,59-69. doi:10.1007/BF00337288; frozen-winner weighted quadratic and explicit SGD formulation here.
/// @see som_assignments(), Optimizer
SOMTensorObjective som_neighborhood_objective(
    const Tensor& weights,
    const Tensor& sample,
    const Tensor& grid,
    double        radius,
    std::size_t   max_elements = 1000000);
/// @brief Compute actual native tensor winner assignments.
/// Requires finite CPU Float64 inputs[N,D] and weights[K,D], positive K,D, matching dimensions; empty N is supported. Uses native broadcast squared distances, reduction and argmin with lowest-index ties. Complete N*K*D geometry checked before allocation. Inputs unchanged; output owns storage and retained tensor references. O(N*K*D) work/intermediate storage, with independently configurable element cap.
/// @param weights Finite CPU Float64 codebook[K,D].
/// @param inputs Finite CPU Float64 observations[N,D].
/// @param max_elements Complete intermediate N*K*D cap; default1000000, zero disables.
/// @return Owned Int64 tensor[N] of actual lowest-index winner ids.
/// @throws std::invalid_argument Invalid shape/dtype/device/finite values.
/// @throws std::length_error Complete intermediate policy exceeded.
/// @throws std::overflow_error Shape arithmetic or actual squared-distance range exceeded.
/// @par Example
/// @code
/// auto w=dm::nn::from_vector(std::vector<double>{0,2},{2,1});
/// auto x=dm::nn::from_vector(std::vector<double>{1},{1,1});
/// std::print("{}\n",dm::nn::som_assignments(w,x).to_vector<std::int64_t>()); // [0]
/// @endcode
/// @par References
/// - Kohonen (1982), Self-organized formation of topologically correct feature maps, Biological Cybernetics43,59-69. doi:10.1007/BF00337288; frozen-winner weighted quadratic and explicit SGD formulation here.
/// @see som_neighborhood_objective()
Tensor
som_assignments(const Tensor& weights, const Tensor& inputs, std::size_t max_elements = 1000000);
/// @brief Compute actual native tensor mean quantization error.
/// Requires finite CPU Float64 inputs[N,D] and weights[K,D], positive K,D, matching dimensions; empty N is supported. Uses native broadcast squared distances, reduction and minimum-distance square roots and mean. This training reconstruction summary is not held-out accuracy or topology preservation. Complete N*K*D geometry checked before allocation. Inputs unchanged; output owns storage and retained tensor references. O(N*K*D) work/intermediate storage, with independently configurable element cap.
/// @param weights Finite CPU Float64 codebook[K,D].
/// @param inputs Finite CPU Float64 observations[N,D].
/// @param max_elements Complete intermediate N*K*D cap; default1000000, zero disables.
/// @return Owned scalar tensor of mean Euclidean winner distance, zero for empty input.
/// @throws std::invalid_argument Invalid shape/dtype/device/finite values.
/// @throws std::length_error Complete intermediate policy exceeded.
/// @throws std::overflow_error Shape arithmetic or actual squared-distance range exceeded.
/// @par Example
/// @code
/// auto w=dm::nn::from_vector(std::vector<double>{0,2},{2,1});
/// auto x=dm::nn::from_vector(std::vector<double>{1},{1,1});
/// std::print("{}\n",dm::nn::som_tensor_quantization_error(w,x).item()); // 1
/// @endcode
/// @par References
/// - Kohonen (1982), Self-organized formation of topologically correct feature maps, Biological Cybernetics43,59-69. doi:10.1007/BF00337288; frozen-winner weighted quadratic and explicit SGD formulation here.
/// @see som_neighborhood_objective()
Tensor som_tensor_quantization_error(
    const Tensor& weights,
    const Tensor& inputs,
    std::size_t   max_elements = 1000000);
/// @brief Plot actual two-dimensional input points and the fitted row/column lattice with equal axis units.
/// Presentation helper only: validates finite Float64 CPU tensors, positive rows/cols with weights shape[rows*cols,2], inputs[N,2] and optional label ids0..31 matching N. Copies actual values for native lines/points and connects only grid neighbors. Labels color input presentation and do not enter any model computation. Equal plotting spans and square plot rectangle preserve input-space geometry. Empty observations allowed; codebook remains nonempty. Does not fit or evaluate a model; O(N+units) presentation work/storage. Borrowed tensors unchanged; plot owns copied coordinates.
/// @param weights Finite actual fitted codebook[rows*cols,2].
/// @param inputs Finite actual observations[N,2].
/// @param rows Positive grid rows.
/// @param cols Positive grid columns.
/// @param labels Optional complete generating labels0..31; empty uses one color.
/// @return Owned native plot ready to title/save.
/// @throws std::invalid_argument Invalid geometry/dtype/device/finite values or labels.
/// @throws std::length_error Default complete plotting geometry exceeded.
/// @throws std::overflow_error Equal-range plot bounds cannot be represented as finite doubles.
/// @par Example
/// @code
/// auto w=dm::nn::from_vector(std::vector<double>{0,0},{1,2});
/// auto x=dm::nn::zeros({0,2},dm::nn::DType::Float64);
/// auto plot=dm::nn::som_lattice_plot(w,x,1,1);
/// @endcode
/// @par References
/// - Kohonen (1982), Self-organized formation of topologically correct feature maps, Biological Cybernetics43,59-69. doi:10.1007/BF00337288; frozen-winner weighted quadratic and explicit SGD formulation here.
/// @see som_assignments()
graphics::RPlot som_lattice_plot(
    const Tensor&           weights,
    const Tensor&           inputs,
    std::size_t             rows,
    std::size_t             cols,
    const std::vector<int>& labels = {});
} // namespace datamunge::nn
