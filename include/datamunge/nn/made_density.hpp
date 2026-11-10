#pragma once
/// @file made_density.hpp
/// @brief Degree-masked Bernoulli densities and stable equal-probability joint-density mixtures.
#include "datamunge/nn/autoregressive_pixels.hpp"

namespace datamunge::nn::models {
/// @brief Private validation of discrete degree metadata before allocating a mask.
namespace made_detail {
/// @brief Require a nonempty nonnegative Int32/64 CPU degree vector.
/// @param degree Discrete CPU[F],F>0, arbitrary strides; zero degree is allowed.
/// @throws std::invalid_argument For undefined/wrong geometry/dtype or negative degree.
/// @throws std::logic_error For unsupported storage.
inline void Degrees(const Tensor& degree) {
  if(!degree.defined() || degree.dim()!=1 || degree.size(0)<=0 ||
     (degree.dtype()!=DType::Int32 && degree.dtype()!=DType::Int64))
    throw std::invalid_argument("Autoregressive degrees require nonempty integer vectors");
  for(auto value:degree.contiguous().to_vector<std::int64_t>())
    if(value<0)throw std::invalid_argument("Autoregressive degrees must be nonnegative");
}
} // namespace made_detail

/// @brief Compare incoming and outgoing degrees to construct a stopped structural dense mask.
///
/// Nonempty nonnegative Int32/64 CPU input_degrees[In],output_degrees[Out],
/// arbitrary strides and mixed integer widths allowed. Entry(i,j) is one when
/// input_degree[i]<=output_degree[j], or strict< when strict is true. Return
/// owned detached Float32(default)/64[In,Out] exact zeros/ones. Zero degrees
/// are allowed for constant features. This comparison alone does not validate
/// a complete architecture/order; strict output masks must prevent own/future
/// input paths. No store, RNG, mutation, fitting or retained input graph.
/// Work/storage O(In*Out), representable product required; local independent calls.
/// @param input_degrees Nonempty nonnegative integer CPU[In] incoming degrees.
/// @param output_degrees Nonempty nonnegative integer CPU[Out] outgoing degrees.
/// @param strict False(default) for<= hidden masks; true for< output masks.
/// @param dtype Float32(default) or Float64 structural mask dtype.
/// @return Owning detached floating CPU[In,Out] exact comparison mask.
/// @throws std::invalid_argument For invalid degree vectors/dtype or overflowing mask product.
/// @throws std::logic_error For unsupported storage or primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto d=n::from_vector(std::vector<std::int32_t>{1,2,3},{3});
/// auto m=n::models::AutoregressiveDegreeMask(d,d,true);
/// // m[3,3] is strictly upper triangular, in incoming-row/outgoing-column order.
/// @endcode
/// @par References
/// - Germain, Gregor, Murray and Larochelle(2015), MADE: Masked Autoencoder
/// for Distribution Estimation, https://arxiv.org/abs/1502.03509 .
/// @see MadeMasks(), MaskedDensityDense(), MadeDensity()
inline Tensor AutoregressiveDegreeMask(
    const Tensor& input_degrees,const Tensor& output_degrees,bool strict=false,
    DType dtype=DType::Float32) {
  made_detail::Degrees(input_degrees);made_detail::Degrees(output_degrees);
  if(input_degrees.numel()>std::numeric_limits<std::int64_t>::max()/output_degrees.numel() ||
     (dtype!=DType::Float32 && dtype!=DType::Float64))
    throw std::invalid_argument("Degree mask needs a representable positive floating matrix");
  NoGradGuard guard;
  auto a=reshape(input_degrees,{-1,1}),b=reshape(output_degrees,{1,-1});
  return cast(strict?less(a,b):less_equal(a,b),dtype);
}

/// @brief Own the fixed64-pixel teaching degrees and their three structural dense masks.
///
/// Three owning detached Int32 degree vectors and three owning detached floating
/// masks survive parent collection. No learned leaves, order state, optimizer or
/// sampling stream is retained. Degree values and mask axes follow MadeMasks.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto masks=dm::nn::models::MadeMasks(false);
/// // input degrees[64], hidden degrees[16] each; masks[64,16],[16,16],[16,64].
/// @endcode
/// @par References
/// - Germain et al.(2015), MADE: Masked Autoencoder for Distribution Estimation,
/// https://arxiv.org/abs/1502.03509 . Source uses two fixed deterministic orders.
/// @see MadeMasks(), MadeDensity(), AutoregressiveDegreeMask()
struct MadeMaskInventory {
  Tensor input_degrees;   ///< Owning detached Int32[64] degrees1..64, or64..1 in reverse mode.
  Tensor hidden1_degrees; ///< Owning detached Int32[16] values1+floor(i*62/15),i=0..15.
  Tensor hidden2_degrees; ///< Owning detached Int32[16] hidden1[(5*i)mod16].
  Tensor input_mask;      ///< Owning detached floating[64,16], input degree<=hidden1 degree.
  Tensor hidden_mask;     ///< Owning detached floating[16,16], hidden1 degree<=hidden2 degree.
  Tensor output_mask;     ///< Owning detached floating[16,64], strict hidden2 degree<input degree.
};

/// @brief Build exact raster/reverse degree metadata for the source two-layer MADE.
///
/// Input/output degrees are1..64 for raster or64..1 for reverse. Hidden1[i]=
/// 1+floor(i*62/15), hidden2[i]=hidden1[(5*i)mod16],i=0..15. The input/hidden
/// masks use<=; output uses strict<, excluding each output's current and later
/// pixels in that order. Int32 degree vectors; masks Float32(default)/64. Six
/// owning detached fields, no store/RNG/global order state/mutation. 2400 stored
/// model values include1144 forbidden kernel coordinates and96 bias values.
/// Work/storage O(64*16+16*16+16*64), fixed small geometry; local independent calls.
/// @param reverse False for raster top-left first; true for reverse bottom-right first.
/// @param dtype Float32(default) or Float64 mask dtype, independent of integer degrees.
/// @return Owning six-field MadeMaskInventory with exact source degrees/masks.
/// @throws std::invalid_argument For nonfloating mask dtype.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto r=dm::nn::models::MadeMasks(true);
/// // r.input_degrees starts64,63,...,1; the first generated raw pixel is index63.
/// @endcode
/// @par References
/// - Germain et al.(2015), MADE: Masked Autoencoder for Distribution Estimation,
/// https://arxiv.org/abs/1502.03509 . Compact source uses deterministic degrees,
/// two16-unit sigmoid layers and separately fitted raster/reverse models.
/// @see MadeMaskInventory, AutoregressiveDegreeMask(), MadeDensity()
inline MadeMaskInventory MadeMasks(bool reverse,DType dtype=DType::Float32) {
  if(dtype!=DType::Float32 && dtype!=DType::Float64)
    throw std::invalid_argument("MadeMasks requires a floating mask dtype");
  std::vector<std::int32_t> input(64),h1(16),h2(16);
  for(int i=0;i<64;++i)input[i]=reverse?64-i:i+1;
  for(int i=0;i<16;++i)h1[i]=1+i*62/15;
  for(int i=0;i<16;++i)h2[i]=h1[(i*5)%16];
  MadeMaskInventory out;
  out.input_degrees=from_vector(input,{64});out.hidden1_degrees=from_vector(h1,{16});
  out.hidden2_degrees=from_vector(h2,{16});
  out.input_mask=AutoregressiveDegreeMask(out.input_degrees,out.hidden1_degrees,false,dtype);
  out.hidden_mask=AutoregressiveDegreeMask(out.hidden1_degrees,out.hidden2_degrees,false,dtype);
  out.output_mask=AutoregressiveDegreeMask(out.hidden2_degrees,out.input_degrees,true,dtype);
  return out;
}

/// @brief Apply a live biased dense projection with an explicit stopped structural mask.
///
/// Finite matching Float32/64 CPU input[N,In] and mask[In,Out], positive axes,
/// arbitrary strides; mask must contain exact zeros/ones. Create missing
/// name/W normal(0,.1) and name/b zeros, then bias_add(input@(W*mask),b).
/// Mask is structural and detached; applicable input/W/b first/higher paths
/// stay live. Masked coordinates have exact zero partials of every order.
/// This primitive alone guarantees no full-network causal order. Validate
/// supplied arguments before leaves; stored-leaf/math errors may follow creation.
/// Only missing leaves initialize; no existing-leaf writes, activation, optimizer,
/// fitting, sampling, RNG after initialization or hidden no-grad. Owning live
/// output survives store collection. Work O(N*In*Out), storage O(N*Out+In*Out);
/// synchronize shared mutable stores. Nonfinite output is rejected.
/// @param parameters Caller-owned matching floating parameter store.
/// @param name Nonempty leaf prefix, followed by /W and /b.
/// @param input Finite signed matching floating CPU[N,In] input features.
/// @param mask Matching finite floating CPU[In,Out] exact zero/one structural mask.
/// @return Owning live floating CPU[N,Out] masked affine features.
/// @throws std::invalid_argument For invalid name/inputs/products/values/leaves or nonfinite output.
/// @throws std::logic_error For unsupported storage or primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(8803);
/// auto y=n::models::MaskedDensityDense(p,"toy",n::ones({2,3}),n::ones({3,4}));
/// // y[2,4]; p has12 kernel+4 bias values in two leaves.
/// @endcode
/// @par References
/// - Germain et al.(2015), MADE: Masked Autoencoder for Distribution Estimation,
/// https://arxiv.org/abs/1502.03509 . Mask multiplication remains inside the graph.
/// @see AutoregressiveDegreeMask(), MadeDensity(), MaskedPixelConv()
inline Tensor MaskedDensityDense(ParameterStore& parameters,const std::string& name,
                                  const Tensor& input,const Tensor& mask) {
  gan_detail::Finite(mask);
  if(name.empty() || mask.dim()!=2 || mask.dtype()!=parameters.dtype())
    throw std::invalid_argument("MaskedDensityDense requires a named matching floating matrix mask");
  gan_detail::Matrix(input,mask.size(0),parameters.dtype());
  autoregressive_pixel_detail::Range(mask,true);
  if(input.size(0)>std::numeric_limits<std::int64_t>::max()/mask.size(1))
    throw std::invalid_argument("MaskedDensityDense output product overflows");
  auto w=parameters.get(name+"/W",mask.shape(),init::normal(.1));
  auto b=parameters.get(name+"/b",{mask.size(1)},init::zeros());
  auto out=bias_add(matmul(input,w*mask.detach()),b);
  gan_detail::Finite(out);
  return out;
}

/// @brief Own two live sigmoid feature maps and a fixed-order MADE image-logit map.
///
/// Three flat Tensor fields retain applicable graphs/storage after parent/store
/// collection. No mask inventory, order state, optimizer or model object is
/// retained separately; the graph includes its stopped structural masks.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(8803);
/// auto o=n::models::MadeDensity(p,n::zeros({2,8,8,1}),false);
/// // hidden1/hidden2[2,16], logits[2,8,8,1];2400 values in6 leaves.
/// @endcode
/// @par References
/// - Germain et al.(2015), MADE: Masked Autoencoder for Distribution Estimation,
/// https://arxiv.org/abs/1502.03509 . Compact source teaching model.
/// @see MadeDensity(), MadeMasks(), BernoulliPixels()
struct MadeDensityOutput {
  Tensor hidden1; ///< Owning live first16-unit sigmoid map[N,16].
  Tensor hidden2; ///< Owning live second16-unit sigmoid map[N,16].
  Tensor logits;  ///< Owning live raw fixed-order Bernoulli logits[N,8,8,1].
};

/// @brief Evaluate the source64-pixel degree-masked Bernoulli joint density.
///
/// Finite signed matching Float32/64 CPU image[N,8,8,1],N>0, arbitrary strides.
/// Flatten64 pixels; two16-unit sigmoid layers and a64-logit head use MadeMasks'
/// <=,<=,strict< degree masks. Source seed8803 creates2400 values in6 named
/// leaves: input/hidden/output W,b. Missing W normal(0,.1), biases zero;1144
/// forbidden kernel coordinates,1256 active values. First output in the chosen
/// order has no input paths, only its bias. Strict own/future exclusion is in
/// raster order or reverse order; raw output layout is always8x8 raster.
///
/// Reverse is explicit per call, not retained. The source fits separate stores
/// for each order; reusing one store with changed masks is a different protocol.
/// All supplied inputs validate before leaves; stored-leaf/math errors may follow
/// creation. Only missing leaves initialize; no existing-leaf writes, fitting,
/// optimizer, sampling, normalization, batch coupling, dropout or hidden no-grad.
/// Three owning fields retain applicable first/higher image/weight paths; masks
/// are stopped structural data. Fixed-width work/storage O(N*64*16); synchronize
/// shared mutable stores. Caller supplies binary targets to BernoulliPixels.
/// @param parameters Caller-owned matching floating store, source seed8803.
/// @param image Finite signed matching CPU[N,8,8,1] image/canvas.
/// @param reverse False(default) for top-left first; true for bottom-right first.
/// @return Owning three-field MadeDensityOutput, including logits[N,8,8,1].
/// @throws std::invalid_argument For incompatible geometry/dtype/values/products or stored leaves.
/// @throws std::logic_error For unsupported storage or primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(8803);
/// auto o=n::models::MadeDensity(p,n::zeros({2,8,8,1}),true);
/// // Reverse-order predictions; sampling ordinal j writes raw site63-j.
/// @endcode
/// @par References
/// - Germain et al.(2015), MADE: Masked Autoencoder for Distribution Estimation,
/// https://arxiv.org/abs/1502.03509 . Source fits two independent fixed orders,
/// not one shared-weight order-agnostic model.
/// @see MadeDensityOutput, MadeMasks(), MaskedDensityDense(), EqualDensityMixture()
inline MadeDensityOutput MadeDensity(ParameterStore& parameters,const Tensor& image,bool reverse=false) {
  autoregressive_pixel_detail::Image(image,1,parameters.dtype());
  if(image.size(1)!=8 || image.size(2)!=8)
    throw std::invalid_argument("MadeDensity requires fixed8x8 gray images");
  auto masks=MadeMasks(reverse,parameters.dtype());
  MadeDensityOutput out;
  out.hidden1=sigmoid(MaskedDensityDense(parameters,"input",reshape(image,{-1,64}),masks.input_mask));
  out.hidden2=sigmoid(MaskedDensityDense(parameters,"hidden",out.hidden1,masks.hidden_mask));
  out.logits=reshape(MaskedDensityDense(parameters,"output",out.hidden2,masks.output_mask),{-1,8,8,1});
  return out;
}

/// @brief Own live Float64 equal-mixture joint NLLs and component responsibilities.
///
/// Three flat owning fields retain applicable input graphs/storage after parent
/// collection. These operate on per-observation joint NLLs, not per-coordinate
/// average losses. No model, optimizer, fitted weights or sampling is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto o=n::models::EqualDensityMixture(n::tensor({0.,2.,2.,0.},{2,2},n::DType::Float64));
/// // Two equally weighted components; per_image_nll[2], responsibilities[2,2].
/// @endcode
/// @par References
/// - Finite equal-probability mixture identity p(x)=sum_k p_k(x)/K and its
/// stable logsumexp evaluation. MADE source mixes two separately normalized orders.
/// @see EqualDensityMixture(), MadeDensity(), BernoulliPixels()
struct DensityMixtureOutput {
  Tensor per_image_nll;   ///< Owning live Float64[N] joint NLL of the equally weighted mixture.
  Tensor mean_nll;        ///< Owning live Float64 scalar mean joint NLL over observations.
  Tensor responsibilities; ///< Owning live Float64[N,K] component probability divided by component sum.
};

/// @brief Mix normalized joint densities equally by stable logmeanexp of negative NLLs.
///
/// Finite Float32/64 CPU per_model_nll[N,K],N>0,K>0, arbitrary strides; signed
/// NLLs allowed for continuous densities. Cast to Float64 and compute
/// -log(sum_k exp(-nll_k)/K) for each observation, with a stopped row maximum
/// for stability. Return joint NLLs, their mean and softmax(-nll) component
/// responsibilities. Applicable first/higher paths remain live, including to
/// Float32 inputs through the cast; the stabilizing maximum is stopped.
/// K=1 preserves the supplied NLL. No clipping, model evaluation, fitting,
/// sampling, mutation or hidden no-grad. Three owning fields survive input/
/// parent collection. Work/storage O(NK); local independent calls.
///
/// A normalized mixture interpretation requires every supplied column to be a
/// normalized joint density of the same observations/base measure. The function
/// cannot establish that fact from numbers. Use per-image sums for image joints,
/// not mean pixel losses. This is not the arithmetic mean of component NLLs:
/// posterior responsibility is its derivative with respect to each component NLL.
/// Source report evaluates fitted orders; no shared-weight training is implied.
/// @param per_model_nll Finite floating CPU[N,K] per-observation joint NLLs, one component per column.
/// @return Owning live Float64 DensityMixtureOutput with[N],scalar,[N,K] fields.
/// @throws std::invalid_argument For invalid inputs or nonfinite returned NLL/mean/weights.
/// @throws std::logic_error For unsupported storage or primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;auto t=n::DType::Float64;
/// auto o=n::models::EqualDensityMixture(-n::log(n::tensor({.8,.2},{1,2},t)));
/// // Mixture probability=.5, NLL=log(2), responsibilities(.8,.2).
/// @endcode
/// @par References
/// - Equal-probability finite-mixture identity p(x)=sum_k p_k(x)/K, evaluated
/// with logsumexp; source MADE teaching experiment mixes two fixed-order joints.
/// @see DensityMixtureOutput, MadeDensity(), logsumexp()
inline DensityMixtureOutput EqualDensityMixture(const Tensor& per_model_nll) {
  gan_detail::Finite(per_model_nll);
  if(per_model_nll.dim()!=2)
    throw std::invalid_argument("EqualDensityMixture requires a positive floating[N,K] joint-NLL matrix");
  auto logmass=-cast(per_model_nll,DType::Float64);
  auto maximum=stop_gradient(max(logmass,{1},true));
  auto mass=exp(logmass-maximum),denominator=sum(mass,{1},true);
  DensityMixtureOutput out;
  out.per_image_nll=-squeeze(maximum+log(denominator)-std::log(double(per_model_nll.size(1))),{1});
  out.mean_nll=mean(out.per_image_nll);out.responsibilities=mass/denominator;
  gan_detail::Finite(out.per_image_nll);gan_detail::Finite(out.mean_nll);gan_detail::Finite(out.responsibilities);
  return out;
}
} // namespace datamunge::nn::models
