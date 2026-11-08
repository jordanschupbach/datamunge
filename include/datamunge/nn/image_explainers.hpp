#pragma once
/// @file image_explainers.hpp
/// @brief Batched tensor image coalitions, weighted local surrogates and exact fixed-baseline Shapley attribution.
#include <datamunge/nn/tensor.hpp>
#include <datamunge/plot/diagram.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
namespace datamunge::nn {

/// @brief Black-box batched image scores for a fixed target and score definition.
/// Receives finite CPU floating [B,H,W,C] masked images and returns finite floating
/// [B] CPU scores. A score may be signed or a probability; use one fixed target for all
/// coalitions. The callable must be deterministic and independent of batch grouping
/// for comparable explanations. Called under NoGradGuard, in ascending coalition or
/// sample order; callbacks may retain their own state, so exceptions cannot undo
/// external side effects. Returned scores are copied as detached Float64 values.
/// Do not mutate borrowed arguments or learned leaves during explanation.
///
///
/// @par Example
/// @code
/// dm::nn::ImageScoreBatch score=[](const dm::nn::Tensor& x){return dm::nn::sum(x,{1,2,3});};
/// std::print("{:.0f}\n",score(dm::nn::ones({2,1,1,1})).to_vector<float>()[0]); // 1
/// @endcode
///
/// @see exact_image_shapley(), lime_image()
using ImageScoreBatch=std::function<Tensor(const Tensor&)>;

/// @brief A fixed rectangular patch game, bounded callback batch and evaluation quota.
/// Patch p=floor(row*patch_rows/H)*patch_cols+floor(col*patch_cols/W), in row-major
/// order. Every patch must contain pixels. Off patches use one scalar baseline,
/// rounded to image dtype; all channels of a spatial patch switch together.
/// max_evaluations limits actual scored images, not callback calls or peak RAM.
/// Zero disables only this quota; geometry and exact-Shapley P<=20 remain enforced.
///
///
/// @par Example
/// @code
/// dm::nn::ImagePatchOptions o;o.patch_rows=1;o.patch_cols=2;
/// std::print("{} {}\n",o.patch_rows,o.patch_cols); // 1 2
/// @endcode
///
/// @see mask_image_patches(), exact_image_shapley()
struct ImagePatchOptions {
    std::int64_t patch_rows=4; ///< Positive grid rows, no greater than H.
    std::int64_t patch_cols=4; ///< Positive grid columns, no greater than W.
    double baseline=0; ///< Finite off-patch value, finite after dtype rounding.
    std::size_t batch_size=256; ///< Positive maximum images per prediction callback.
    std::size_t max_evaluations=1000000; ///< Maximum scored images; zero disables only this quota.
};

/// @brief Seeded full/empty-anchored weighted ridge surrogate controls.
/// The first sample keeps all patches, the second removes all; remaining bits are
/// independent low bits of a local MT19937-64 stream. This is not the legacy scalar
/// wrapper sampling sequence. samples>=P+1 and at least two; kernel_width and ridge
/// must be finite positive. Kernel exp(-(fraction_off/kernel_width)^2); intercept
/// is unpenalized. A Float64 tensor Cholesky solve fits all coefficients at once,
/// without an optimizer or explicit inverse. Very ill-conditioned fits may fail.
///
///
/// @par Example
/// @code
/// dm::nn::ImageLimeOptions o;o.samples=800;
/// std::print("{} {:.3f}\n",o.samples,o.ridge); // 800 0.001
/// @endcode
///
/// @par References
/// - Ribeiro, Singh and Guestrin (2016), "Why Should I Trust You?": Explaining the Predictions of Any Classifier. https://arxiv.org/abs/1602.04938. Custom fixed-grid fraction-off kernel and ridge variant; no sparse feature selection.
///
/// @see lime_image()
struct ImageLimeOptions {
    ImagePatchOptions patches; ///< Common geometry, baseline, batch and evaluation quota.
    std::size_t samples=800; ///< Number of actual perturbations, including full/empty anchors.
    double kernel_width=.25; ///< Positive fraction-off locality scale.
    double ridge=1e-3; ///< Positive coefficient penalty, excluding intercept.
    std::uint64_t seed=1; ///< Local MT19937-64 bit stream seed.
};

/// @brief Counts from actual prediction dispatch and Shapley marginal reduction.
/// evaluations counts images scored, predict_calls counts dispatched batches and
/// marginal_terms counts actual weighted differences. Excludes mask construction,
/// validation, allocation, tensor arithmetic instructions and callback internal work.
/// LIME has zero Shapley marginal terms. No mutable global counters are used.
///
///
/// @par Example
/// @code
/// dm::nn::ImageExplanationWork w;
/// std::print("{} {} {}\n",w.evaluations,w.predict_calls,w.marginal_terms); // 0 0 0
/// @endcode
///
/// @see ExactImageShapley, TensorImageLime
struct ImageExplanationWork {
    std::size_t evaluations=0; ///< Actual scored images.
    std::size_t predict_calls=0; ///< Actual batch callback invocations.
    std::size_t marginal_terms=0; ///< Actual Shapley weighted marginal differences.
};

/// @brief Owned exact-enumeration fixed-baseline Shapley values and actual coalition scores.
/// patch_values is Float64 [R,C]; coalition_scores is Float64 [2^P], indexed by
/// ascending integer masks with patch zero the least significant bit. All retained
/// tensors are detached independent storage. Base/full are actual endpoints, and
/// efficiency_residual=sum(patch_values)-(full-base), including rounding error.
/// Exact means no coalition sampling, not exact real arithmetic or causal evidence.
/// Attributions may be signed; changing baseline, partition or score changes the game.
///
///
/// @par Example
/// @code
/// dm::nn::ImagePatchOptions o;o.patch_rows=1;o.patch_cols=1;
/// const auto e=dm::nn::exact_image_shapley(dm::nn::ones({1,1,1,1}),[](const dm::nn::Tensor& x){return dm::nn::sum(x,{1,2,3});},o);
/// std::print("{:.0f} {}\n",e.patch_values.item(),e.work.evaluations); // 1 2
/// @endcode
///
/// @par References
/// - Lundberg and Lee (2017), A Unified Approach to Interpreting Model Predictions. https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf. Exact fixed-baseline coalition game, not conditional-expectation or KernelSHAP estimation.
///
/// @see exact_image_shapley(), expand_image_attributions()
struct ExactImageShapley {
    Tensor patch_values; ///< Detached Float64 signed [R,C] Shapley values.
    Tensor coalition_scores; ///< Owned actual Float64 [2^P] scores, including empty/full.
    double base_value=0; ///< Actual all-baseline score.
    double full_value=0; ///< Actual original-image score.
    double efficiency_residual=0; ///< Measured sum(phi)-(full-base), not forcibly corrected.
    ImageExplanationWork work; ///< Actual scored images, batches and weighted marginals.
};

/// @brief Owned actual perturbations, scores, kernel weights and weighted-ridge fit.
/// patch_values is detached Float64 [R,C]. masks is Bool [N,P], scores and
/// proximity_weights Float64 [N]. These are the actual samples used in the fit,
/// not a replay. Outputs share no input/callback storage. r_squared is weighted
/// training R2, zero by convention when weighted score variance is zero; it is not
/// held-out fidelity. fitted_full is intercept+sum(coefficients), while full_value
/// is the actual anchored full-image score. Neither efficiency nor causality follows.
///
///
/// @par Example
/// @code
/// dm::nn::ImageLimeOptions o;o.patches.patch_rows=1;o.patches.patch_cols=1;o.samples=8;
/// const auto e=dm::nn::lime_image(dm::nn::ones({1,1,1,1}),[](const dm::nn::Tensor& x){return dm::nn::sum(x,{1,2,3});},o);
/// std::print("{} {}\n",e.masks.size(0),e.work.evaluations); // 8 8
/// @endcode
///
/// @par References
/// - Ribeiro, Singh and Guestrin (2016), "Why Should I Trust You?": Explaining the Predictions of Any Classifier. https://arxiv.org/abs/1602.04938. Custom fixed-grid fraction-off kernel and ridge variant; no sparse feature selection.
///
/// @see lime_image(), ExactImageShapley
struct TensorImageLime {
    Tensor patch_values; ///< Detached Float64 signed [R,C] surrogate coefficients.
    Tensor masks; ///< Owned actual Bool [N,P] full/empty-anchored masks.
    Tensor scores; ///< Owned actual detached Float64 [N] model scores.
    Tensor proximity_weights; ///< Owned actual Float64 [N] kernel weights.
    double intercept=0; ///< Unpenalized fitted intercept.
    double r_squared=0; ///< Weighted in-sample fidelity, zero when weighted SST is zero.
    double fitted_full=0; ///< Surrogate score at all patches on.
    double full_value=0; ///< Actual first anchored full-image score.
    ImageExplanationWork work; ///< Actual scored images/batches; zero marginal terms.
};

/// @brief Mask one tensor image into a batch using explicit whole-patch coalitions.
/// image is a finite CPU Float32/Float64 [1,H,W,C] with positive H/W/C; masks is Bool
/// [B,P] CPU data, B positive, P=R*C. All channels switch together. Uses tensor gather/where;
/// input pixel values stay in tensors. Returns independent same-dtype [B,H,W,C].
/// Options batch/evaluation controls do not split or limit this direct masking call.
/// Time/storage O(BHWC); host metadata holds only patch indices. Empty inputs fail.
///
/// @param image Borrowed finite single NHWC floating image; any signed finite values allowed.
/// @param masks Borrowed positive Bool [B,P] on/off flags.
/// @param options Common grid/baseline; defaults to4x4 and zero baseline.
/// @return Owned detached masked-image batch, preserving image dtype.
/// @throws std::invalid_argument On invalid image/grid/baseline or malformed Bool masks.
/// @throws std::length_error On checked geometry/container limits.
/// @throws std::bad_alloc If required tensor/metadata storage cannot be allocated.
///
/// @par Example
/// @code
/// dm::nn::ImagePatchOptions o;o.patch_rows=1;o.patch_cols=2;
/// const auto x=dm::nn::tensor({2,3},{1,1,2,1});
/// const auto z=dm::nn::cast(dm::nn::tensor({1,0},{1,2}),dm::nn::DType::Bool);
/// std::print("{:.0f}\n",dm::nn::sum(dm::nn::mask_image_patches(x,z,o)).item()); // 2
/// @endcode
///
/// @see ImagePatchOptions, exact_image_shapley()
Tensor mask_image_patches(const Tensor& image,const Tensor& masks,ImagePatchOptions options={});

/// @brief Enumerate every patch coalition once and reduce its exact Shapley marginals.
/// Uses P=R*C<=20, M=2^P actual model evaluations, in ascending mask order.
/// Retains actual Float64 M scores and P values; batch images use at most
/// min(batch_size,M)*H*W*C elements. Marginal index/weight scratch is O(M).
/// Work is O(M*(mask pixels+model cost)+P*M); exact enumeration is exponential.
/// Each coefficient sums weighted score differences 1/(P*choose(P-1,|S|)).
/// CPU tensors are required; no device transfer is implicit.
/// All input data/geometry and known evaluation quota are checked before callbacks.
/// Runs under NoGradGuard, without fitting/mutating any learned model or existing
/// input gradients. Model callbacks must satisfy ImageScoreBatch; exceptions propagate
/// and prior external callback effects cannot be undone. No partial result escapes.
/// Independent calls share no mutable RNG/state; caller mutation during a call is unsafe.
///
/// @param image Borrowed finite single positive-shape NHWC floating image.
/// @param predict Nonempty deterministic batched fixed-target score callable.
/// @param options Patch/baseline/batch/quota controls; default4x4/zero/256/1m.
/// @return Owned signed patch values, actual coalition scores/endpoints, measured efficiency residual and work.
/// @throws std::invalid_argument On invalid image/grid/baseline/batch/callback or malformed/nonfinite scores.
/// @throws std::length_error On checked geometry/container limits or enabled evaluation quota.
/// @throws std::overflow_error On nonfinite computed attribution/fit arithmetic.
/// @throws std::bad_alloc If required tensor/metadata storage cannot be allocated.
///
/// @par Example
/// @code
/// dm::nn::ImagePatchOptions o;o.patch_rows=1;o.patch_cols=2;
/// const auto e=dm::nn::exact_image_shapley(dm::nn::tensor({2,3},{1,1,2,1}),[](const dm::nn::Tensor& x){return dm::nn::sum(x,{1,2,3});},o);
/// std::print("{:.0f} {:.0f} {}\n",e.base_value,e.full_value,e.work.marginal_terms); // 0 5 4
/// @endcode
///
/// @par References
/// - Lundberg and Lee (2017), A Unified Approach to Interpreting Model Predictions. https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf. Exact fixed-baseline coalition game, not conditional-expectation or KernelSHAP estimation.
///
/// @see ExactImageShapley, lime_image()
ExactImageShapley exact_image_shapley(const Tensor& image,const ImageScoreBatch& predict,ImagePatchOptions options={});

/// @brief Fit a tensor Float64 local weighted ridge surrogate to actual image perturbations.
/// Uses the full/empty anchors and seeded masks in ImageLimeOptions. Fits
/// (X^T W X+ridge*diag(0,1,...))beta=X^T W scores by Cholesky, with tensor arithmetic
/// and no explicit inverse. This fixed-grid fraction-off/ridge variant does not
/// perform sparse LIME feature selection or KernelSHAP. Nonzero ridge plus full
/// anchor gives a positive-definite system mathematically; rounding may still fail.
/// Time adds O(N*P^2+P^3) fit work to N mask/model evaluations. Retains O(NP+P^2)
/// fit data/scratch plus one bounded image batch; quota limits model images only.
/// CPU tensors are required; no device transfer is implicit.
/// All input data/geometry and known evaluation quota are checked before callbacks.
/// Runs under NoGradGuard, without fitting/mutating any learned model or existing
/// input gradients. Model callbacks must satisfy ImageScoreBatch; exceptions propagate
/// and prior external callback effects cannot be undone. No partial result escapes.
/// Independent calls share no mutable RNG/state; caller mutation during a call is unsafe.
///
/// @param image Borrowed finite single positive-shape NHWC floating image.
/// @param predict Nonempty deterministic batched fixed-target score callable.
/// @param options Common controls plus N800, width.25, ridge.001, seed1 defaults.
/// @return Owned actual samples/scores/weights, signed patch coefficients, intercept and fidelity diagnostics.
/// @throws std::invalid_argument On invalid image/grid/baseline/batch/callback or malformed/nonfinite scores.
/// @throws std::length_error On checked geometry/container limits or enabled evaluation quota.
/// @throws std::overflow_error On nonfinite computed attribution/fit arithmetic.
/// @throws std::bad_alloc If required tensor/metadata storage cannot be allocated.
/// @throws std::runtime_error If numerical Cholesky factorization fails.
///
/// @par Example
/// @code
/// dm::nn::ImageLimeOptions o;o.patches.patch_rows=1;o.patches.patch_cols=1;o.samples=8;
/// const auto e=dm::nn::lime_image(dm::nn::ones({1,1,1,1}),[](const dm::nn::Tensor& x){return dm::nn::sum(x,{1,2,3});},o);
/// std::print("{} {}\n",e.masks.size(1),e.work.marginal_terms); // 1 0
/// @endcode
///
/// @par References
/// - Ribeiro, Singh and Guestrin (2016), "Why Should I Trust You?": Explaining the Predictions of Any Classifier. https://arxiv.org/abs/1602.04938. Custom fixed-grid fraction-off kernel and ridge variant; no sparse feature selection.
///
/// @see TensorImageLime, exact_image_shapley()
TensorImageLime lime_image(const Tensor& image,const ImageScoreBatch& predict,ImageLimeOptions options={});

/// @brief Expand signed patch scores to a spatial tensor map with explicit area conservation.
/// CPU tensors are required, without implicit device transfer.
/// Patch membership follows ImagePatchOptions for possibly uneven image/grid shapes.
/// With distribute=true each score is divided among its patch pixels, so the map sum
/// matches the patch sum up to rounding. false repeats a patch score for display,
/// whose pixel sum instead weights by patch area. This is a spatial explanation map
/// shared across channels, not independently inferred pixel evidence. Returns detached
/// same-dtype [1,H,W,1] with independent storage; O(HW+RC) work/storage.
///
/// @param patch_values Borrowed finite floating nonempty [R,C] signed scores; R<=height,C<=width.
/// @param height Positive spatial map height.
/// @param width Positive spatial map width.
/// @param distribute Default true splits score over patch area; false repeats it.
/// @return Owned detached signed spatial attribution map.
/// @throws std::invalid_argument On invalid image/grid/baseline/batch/callback or malformed/nonfinite scores.
/// @throws std::length_error On checked geometry/container limits or enabled evaluation quota.
/// @throws std::overflow_error On nonfinite computed attribution/fit arithmetic.
/// @throws std::bad_alloc If required tensor/metadata storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto p=dm::nn::tensor({2,3},{1,2},dm::nn::DType::Float64);
/// std::print("{:.0f} {:.0f}\n",dm::nn::sum(dm::nn::expand_image_attributions(p,2,4)).item(),dm::nn::sum(dm::nn::expand_image_attributions(p,2,4,false)).item()); // 5 20
/// @endcode
///
/// @see ExactImageShapley, TensorImageLime
Tensor expand_image_attributions(const Tensor& patch_values,std::int64_t height,std::int64_t width,bool distribute=true);

/// @brief Measure signed Pearson agreement of finite equal-shape attribution tensors.
/// Requires CPU tensors and computes centered Float64 tensor reductions; inputs may have different floating
/// dtypes but must have identical shape and at least two elements. positive_only
/// explicitly clamps each map below zero before centering; default keeps signs.
/// Returns NaN when either centered norm is zero (undefined correlation), never
/// silently zero. Scaling by max absolute value before centering reduces overflow.
/// O(numel) work/storage, detached; no input/gradient mutation or causal interpretation.
///
/// @param first Borrowed finite floating tensor with at least two elements.
/// @param second Borrowed same-shape finite floating tensor.
/// @param positive_only Default false retains signed evidence; true compares positive support only.
/// @return Signed Pearson coefficient, or NaN for a constant input after optional clamping.
/// @throws std::invalid_argument On invalid image/grid/baseline/batch/callback or malformed/nonfinite scores.
/// @throws std::overflow_error On nonfinite computed attribution/fit arithmetic.
/// @throws std::bad_alloc If required tensor/metadata storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto a=dm::nn::tensor({-1,0,1},{3});
/// std::print("{:.1f}\n",dm::nn::attribution_correlation(a,a)); // 1.0
/// @endcode
///
/// @see expand_image_attributions()
double attribution_correlation(const Tensor& first,const Tensor& second,bool positive_only=false);
/// @brief Draw equal-scale tensor image/map tiles with explicit numeric grayscale or signed colour keys.
/// Presentation copies finite CPU floating [N,H,W,1] values to native square SVG
/// cells, first row at top. Empty signed_maps means every tile is signed; otherwise
/// N flags choose blue-white-red symmetric per-map max-absolute scales (true) or
/// black-white [0,1] intensity scales (false, clipped). All-zero signed tiles use
/// [-1,1]. Each tile has its own numeric low/zero/high key and owned short label.
/// No raster payload, gradients or model state is retained. O(NHW) work/storage;
/// a fixed one-million-pixel geometry bound precedes copies/rendering. Width is
/// fitted uniformly; long labels may widen bounds. All returned content owns data.
/// @param maps Borrowed finite CPU Float32/Float64 positive [N,H,W,1] image/maps.
/// @param columns Positive tiles per row, default two.
/// @param labels Empty for numeric ids, or exactly N short unwrapped labels.
/// @param signed_maps Empty for all signed, or N flags; false chooses intensity [0,1].
/// @param max_width Positive finite maximum diagram width, default1100 screen pixels.
/// @return Owned native editable diagram with square cells and numerical colour keys.
/// @throws std::invalid_argument On invalid tensor/label/flag geometry or width.
/// @throws std::length_error On more than one million pixels or container limits.
/// @throws std::bad_alloc If presentation storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto figure=dm::nn::diagram_image_maps(dm::nn::tensor({-1,0,1},{1,1,3,1}),1,{"Signed example"});
/// std::print("{}\n",figure.to_svg().empty()); // false
/// @endcode
///
/// @see expand_image_attributions(), graphics::diverging_colors(), diagram_tensor_images()
graphics::Diagram diagram_image_maps(const Tensor& maps,std::int64_t columns=2,const std::vector<std::string>& labels={},const std::vector<bool>& signed_maps={},double max_width=1100);
} // namespace datamunge::nn
