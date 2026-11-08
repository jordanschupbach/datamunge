#pragma once
/// @file zplus_relevance.hpp
/// @brief Stabilized positive-connection redistribution of signed relevance, including small CNN explanations.
#include <datamunge/nn/small_image_models.hpp>
namespace datamunge::nn {
/// @brief Redistribute dense-output relevance through nonnegative activations and positive weights.
/// Computes a * ((R/(a W+ + epsilon)) W+^T), W+=max(W,0), excluding bias.
/// a is finite/nonnegative; W and R are finite, and R may be signed. A zero
/// positive-contribution denominator passes no relevance; positive epsilon also
/// loses some mass. This is a stabilized signed extension of z-plus redistribution,
/// not an exact conservation, probability, ordinary gradient or causal certificate.
/// Runs under NoGradGuard, retaining no history and changing no input/leaf gradients.
/// Dense matrix work/storage follows the two matrix products; results own storage.
/// @param activation Finite nonnegative floating [N,I], N/I positive.
/// @param weight Same-dtype finite [I,O], O positive; negative connections are discarded.
/// @param relevance Same-dtype finite signed [N,O] incoming scores.
/// @param epsilon Finite positive stabilizer representable positive in tensor dtype, default 1e-9.
/// @return Detached finite [N,I] redistributed relevance.
/// @throws std::invalid_argument If shapes/dtypes/data/nonnegative domain or epsilon are invalid.
/// @throws std::overflow_error If a computed product, denominator or result is nonfinite.
/// @throws std::length_error If tensor storage capacity is exceeded.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto r=dm::nn::zplus_dense(dm::nn::tensor({1,3},{1,2}),dm::nn::tensor({1,1},{2,1}),dm::nn::tensor({4},{1,1}));
/// std::print("{:.1f} {:.1f}\n",r.to_vector<float>()[0],r.to_vector<float>()[1]); // 1.0 3.0
/// @endcode
/// @par References
/// - Montavon et al. (2017), Explaining nonlinear classification decisions with deep
///   Taylor decomposition, https://doi.org/10.1016/j.patcog.2016.11.008.
///   This API adds epsilon and permits signed seeds, so positivity/conservation
///   conclusions for the original positive-relevance rule do not follow unchanged.
/// @see zplus_conv2d(), models::SmallImageCnnZPlus()
Tensor zplus_dense(const Tensor& activation,const Tensor& weight,const Tensor& relevance,double epsilon=1e-9);
/// @brief Redistribute convolution-output relevance through positive cross-correlation connections.
/// Computes x * conv_backprop_input(W+, R/(conv(x,W+)+epsilon)), with no bias.
/// Input activations must be nonnegative; signed relevance is allowed. Group,
/// stride, padding and dilation follow the supplied ordinary convolution options.
/// Zero positive paths and epsilon can lose relevance mass. Detached tensor-only
/// computation changes no learned/input values or gradients. Convolution/adjoint
/// work and temporary storage follow the requested geometry. No causal guarantee.
/// @param activation Finite nonnegative floating positive NHWC input.
/// @param kernel Same-dtype finite HWIO kernel, interpreted with options.
/// @param relevance Same-dtype finite signed output relevance matching the actual convolution shape.
/// @param options Shared forward/adjoint geometry; default ordinary SAME convolution.
/// @param epsilon Finite positive dtype-representable stabilizer, default 1e-9.
/// @return Detached finite input-shape relevance, owning its computed storage.
/// @throws std::invalid_argument If data, shape/dtype, convolution options or epsilon are invalid.
/// @throws std::overflow_error If computed convolution, denominator or result is nonfinite.
/// @throws std::length_error If storage capacity is exceeded.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto r=dm::nn::zplus_conv2d(dm::nn::ones({1,2,2,1}),dm::nn::ones({1,1,1,1}),dm::nn::ones({1,2,2,1}));
/// std::print("{:.1f}\n",dm::nn::sum(r).item()); // 4.0
/// @endcode
/// @par References
/// - Montavon et al. (2017), https://doi.org/10.1016/j.patcog.2016.11.008;
///   stabilized signed-seed adaptation as described by zplus_dense().
/// @see zplus_dense(), models::SmallImageCnnZPlus()
Tensor zplus_conv2d(const Tensor& activation,const Tensor& kernel,const Tensor& relevance,const Conv2DOptions& options={},double epsilon=1e-9);
namespace models {
/// @brief Actual forward logits and each detached signed-relevance redistribution endpoint.
/// Target scores include output bias, while redistribution denominators exclude
/// all biases and negative weights. Seeds can be negative; the maps then need not
/// be nonnegative. Actual max-pool winners route pooled relevance back to the map.
/// Epsilon and zero positive paths can lose mass; compare layer sums explicitly.
/// This is a model-inspection heuristic, not proof that pixels caused a prediction.
/// No borrowed input references or learnable-gradient history are retained.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(1);
/// (void)dm::nn::models::SmallImageCnn(p,dm::nn::ones({1,12,12,1}));
/// const auto r=dm::nn::models::SmallImageCnnZPlus(p,dm::nn::ones({1,12,12,1}),{0});
/// std::print("{}\n",r.input_relevance.size(1)); // 12
/// @endcode
/// @see SmallImageCnnZPlus(), zplus_dense(), zplus_conv2d()
struct CnnZPlusExplanation {
  Tensor logits; ///< Actual finite [N,K] forward scores, detached.
  Tensor target_ids; ///< Actual Int64 [N] requested targets or first-argmax predicted targets.
  Tensor output_relevance; ///< [N,K] target raw score at the target column, zero elsewhere.
  Tensor hidden_relevance; ///< [N,D] signed relevance after output-to-hidden redistribution.
  Tensor pooled_relevance; ///< NHWC signed relevance after hidden-to-flat redistribution and reshape.
  Tensor conv_relevance; ///< NHWC signed relevance scattered through actual max-pool winners.
  Tensor input_relevance; ///< [N,H,W,C] final signed pixel relevance, detached and finite.
};
/// @brief Explain selected signed logits of a populated small tensor CNN with stabilized z-plus rules.
/// Runs one actual forward without fitting or replay, retains its actual max-pool
/// argmax, then redistributes through positive output/hidden/convolution weights.
/// All six expected parameter leaves must already exist: no random explanation
/// is silently initialized. Nonnegative finite image data and finite learned
/// leaves are required. Forward includes biases; denominators omit them. Missing
/// positive paths and epsilon can lose mass. Negative selected logits produce
/// signed maps. No parameter/input or existing gradient mutation; all operations
/// run under NoGradGuard. Work/storage add ordinary forward/adjoint passes.
/// @param p Populated matching Float32/Float64 learnable store; no new leaves are created.
/// @param image Positive NHWC batch of finite nonnegative pixels.
/// @param targets One valid class id per image, or empty to use actual first-argmax predictions.
/// @param config Exact shape/pooling/head configuration used to populate p.
/// @param prefix Exact populated model prefix, default cnn.
/// @param epsilon Finite positive dtype-representable stabilizer, default 1e-9.
/// @return Owned actual logits, target seeds and each detached redistribution endpoint.
/// @throws std::invalid_argument If image/targets/configuration/dtypes/leaves/domain or epsilon are invalid.
/// @throws std::overflow_error If tensor dimensions or computed scores/redistributions cannot fit finite range.
/// @throws std::length_error If storage capacity is exceeded.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(1);
/// const auto x=dm::nn::ones({1,12,12,1});
/// (void)dm::nn::models::SmallImageCnn(p,x);
/// const auto r=dm::nn::models::SmallImageCnnZPlus(p,x);
/// std::print("{}\n",r.target_ids.size(0)); // 1
/// @endcode
/// @par References
/// - Bach et al. (2015), On Pixel-Wise Explanations for Non-Linear Classifier Decisions
///   by Layer-Wise Relevance Propagation, https://doi.org/10.1371/journal.pone.0130140.
/// - Montavon et al. (2017), https://doi.org/10.1016/j.patcog.2016.11.008.
///   This is a custom stabilized signed-logit adaptation for SmallImageCnn, not
///   a proof of positivity, exact conservation, faithfulness or causal explanation.
/// @see SmallImageCnn(), CnnZPlusExplanation, zplus_dense(), zplus_conv2d()
CnnZPlusExplanation SmallImageCnnZPlus(ParameterStore& p,const Tensor& image,const std::vector<std::int64_t>& targets={},SmallImageCnnConfig config={},const std::string& prefix="cnn",double epsilon=1e-9);
}
}
