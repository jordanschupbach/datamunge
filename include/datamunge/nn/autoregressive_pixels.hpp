#pragma once
/// @file autoregressive_pixels.hpp
/// @brief Explicit binary-image preparation, masked PixelCNN and teacher-forced Bernoulli loss.
#include <random>
#include "datamunge/nn/gan_models.hpp"
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/conv.hpp"
#include "datamunge/nn/image_ops.hpp"

namespace datamunge::nn::models {
/// @brief Private positive NHWC geometry and range validation for binary-image primitives.
namespace autoregressive_pixel_detail {
/// @brief Require a finite floating CPU NHWC batch with the requested channel count.
/// @param x Finite positive-axis Float32/64 CPU batch, arbitrary strides.
/// @param channels Required positive channel count.
/// @param dtype Required computation dtype.
/// @throws std::invalid_argument For incompatible geometry, dtype or nonfinite values.
/// @throws std::logic_error For unsupported storage.
inline void Image(const Tensor& x, std::int64_t channels, DType dtype) {
  gan_detail::Finite(x);
  if (x.dim()!=4 || x.size(3)!=channels || x.dtype()!=dtype)
    throw std::invalid_argument("Binary pixel operation requires matching finite NHWC images");
}
/// @brief Require values in the closed unit interval, optionally restricted to its endpoints.
/// @param x Already geometry-validated finite floating tensor.
/// @param binary True for exact zero/one values; false for any value in[0,1].
/// @throws std::invalid_argument For a value outside the selected range.
inline void Range(const Tensor& x, bool binary) {
  for (double v:x.contiguous().to_vector<double>())
    if(v<0 || v>1 || (binary && v!=0 && v!=1))
      throw std::invalid_argument("Binary pixel values violate the requested unit range");
}
} // namespace autoregressive_pixel_detail

/// @brief Own resized gray values and their thresholded binary image batch.
///
/// Two detached Float32[N,8,8,1] tensors own independent storage beyond input
/// and parent lifetimes. gray uses Double interpolation and one final cast;
/// images uses gray>=.5, including exact ties. No fitting or model is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <datamunge/nn/autoregressive_pixels.hpp>
/// auto out=dm::nn::models::BinaryPixels(dm::nn::full({2,28,28,1},.5));
/// // out.gray[2,8,8,1] is .5; out.images is all ones.
/// @endcode
/// @par References
/// - Preserves the repository binary-pixel MNIST teaching protocol: half-pixel
/// bilinear28->8, Double accumulation and Float32 threshold at .5.
/// @see BinaryPixels(), BernoulliPixels()
struct BinaryPixelData {
  Tensor gray;   ///< Owning detached Float32[N,8,8,1] resized gray values in[0,1].
  Tensor images; ///< Owning detached Float32[N,8,8,1] exact zero/one thresholded values.
};

/// @brief Resize normalized 28x28 gray images and threshold the rounded Float32 pixels.
///
/// Finite Float32 CPU[N,28,28,1],N>0, values in[0,1], arbitrary strides.
/// Half-pixel centers map each output coordinate j to(j+.5)*3.5-.5. Promote
/// detached input to Double for bilinear arithmetic, cast the completed8x8
/// image once to Float32, then threshold gray>=.5. No antialias filter, random
/// binarization, fitting, split selection or source mutation. Both outputs own
/// stopped storage. Work/storage O(N*28*28); independent calls share no state.
/// @param images Normalized finite Float32 CPU[N,28,28,1] source images.
/// @return Owning detached BinaryPixelData gray and binary[N,8,8,1] tensors.
/// @throws std::invalid_argument For wrong geometry/dtype or out-of-range/nonfinite values.
/// @throws std::logic_error For unsupported storage or primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <datamunge/nn/autoregressive_pixels.hpp>
/// namespace n=dm::nn;
/// auto out=n::models::BinaryPixels(n::full({1,28,28,1},.5));
/// // Gray pixels equal .5; threshold ties become binary one.
/// @endcode
/// @par References
/// - Preserves the repository binary-pixel MNIST teaching protocol: deterministic
/// half-pixel28->8 interpolation with Float32 threshold, distinct from random MNIST binarization.
/// @see BinaryPixelData, PixelCnn(), resize_bilinear()
inline BinaryPixelData BinaryPixels(const Tensor& images) {
  autoregressive_pixel_detail::Image(images,1,DType::Float32);
  if(images.size(1)!=28 || images.size(2)!=28)
    throw std::invalid_argument("BinaryPixels requires source28x28 gray images");
  autoregressive_pixel_detail::Range(images,false);
  NoGradGuard guard;
  BinaryPixelData out;
  out.gray=cast(resize_bilinear(cast(images.detach(),DType::Float64),{8,8},false,true),DType::Float32);
  out.images=cast(greater_equal(out.gray,.5),DType::Float32);
  return out;
}

/// @brief Construct a structural3x3 raster mask with or without the center site.
///
/// Shape[3,3,input_channels,output_channels], exact zeros/ones. Both types allow
/// all channels in the top row and the left cell of the middle row. Type A
/// excludes the middle cell; type B includes it. Entire bottom row and middle
/// right cell are zero. Intended for one-channel raster densities: A on raw
/// pixels, B only on already causal features. No RGB channel ordering is
/// implemented. Owning detached floating CPU data, no store/global RNG writes.
/// Work/storage O(input_channels*output_channels); calls share no mutable state.
/// @param input_channels Positive input width, with representable kernel product.
/// @param output_channels Positive output width.
/// @param type_b False for A(no center); true for B(center allowed).
/// @param dtype Float32(default) or Float64 result type.
/// @return Owning detached floating[3,3,input_channels,output_channels] mask.
/// @throws std::invalid_argument For invalid widths, product overflow or nonfloating dtype.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <datamunge/nn/autoregressive_pixels.hpp>
/// auto a=dm::nn::models::CausalPixelMask(1,8,false);
/// // 32 ones,40 zeros in a[3,3,1,8]; the current input pixel is excluded.
/// @endcode
/// @par References
/// - van den Oord, Kalchbrenner and Kavukcuoglu(2016), Pixel Recurrent Neural
/// Networks, https://arxiv.org/abs/1601.06759 . Spatial A/B masks; this compact
/// binary teaching variant does not implement multichannel output ordering.
/// @see MaskedPixelConv(), PixelCnn()
inline Tensor CausalPixelMask(std::int64_t input_channels,std::int64_t output_channels,
                             bool type_b,DType dtype=DType::Float32) {
  const auto limit=std::numeric_limits<std::int64_t>::max();
  if(input_channels<=0 || output_channels<=0 || input_channels>limit/9 ||
     output_channels>limit/(9*input_channels) || (dtype!=DType::Float32 && dtype!=DType::Float64))
    throw std::invalid_argument("CausalPixelMask requires representable positive floating widths");
  std::vector<float> value(static_cast<std::size_t>(9*input_channels*output_channels),0.f);
  const auto width=input_channels*output_channels;
  for(std::int64_t cell=0;cell<(type_b?5:4);++cell)
    std::fill(value.begin()+cell*width,value.begin()+(cell+1)*width,1.f);
  return cast(from_vector(value,{3,3,input_channels,output_channels}),dtype);
}

/// @brief Apply a biased SAME convolution with an explicit stopped structural kernel mask.
///
/// Finite matching Float32/64 CPU input[N,H,W,Cin] and mask[Kh,Kw,Cin,Cout],
/// positive odd kernel axes and arbitrary strides. Mask must contain exact zeros/ones;
/// it is detached as structural data. Create missing name/W normal(0,.1) and
/// name/b zeros; compute conv2d(input,W*mask,SAME,stride1)+b. Input/kernel/bias
/// first/higher paths stay live; masked kernel coordinates have exact zero
/// partials of every order. Causality depends on the supplied mask and input
/// features. Validate input/mask before leaf creation. No activation, optimizer,
/// fitting, sampling, existing-leaf write or hidden no-grad. Owning result
/// retains its graph beyond store lifetime. Work O(NHWCinCoutKhKw);
/// synchronize shared mutable stores. Stored-leaf/math errors may follow creation.
/// @param parameters Caller-owned matching floating parameter store.
/// @param name Nonempty leaf prefix, followed by /W and /b.
/// @param input Finite signed matching CPU[N,H,W,Cin] features.
/// @param mask Matching finite structural CPU[Kh,Kw,Cin,Cout], positive odd Kh/Kw, exact zeros/ones.
/// @return Owning live CPU[N,H,W,Cout] biased masked-convolution features.
/// @throws std::invalid_argument For empty name, incompatible inputs/products/values or leaves.
/// @throws std::logic_error For unsupported storage or primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <datamunge/nn/autoregressive_pixels.hpp>
/// namespace n=dm::nn;n::ParameterStore p(8401);
/// auto y=n::models::MaskedPixelConv(p,"head",n::zeros({1,8,8,1}),n::models::CausalPixelMask(1,8,false));
/// // y[1,8,8,8]; p has80 stored values in two leaves.
/// @endcode
/// @par References
/// - van den Oord, Kalchbrenner and Kavukcuoglu(2016), Pixel Recurrent Neural
/// Networks, https://arxiv.org/abs/1601.06759 . Mask multiplication remains in
/// the graph; compact source variant uses biased3x3 layers and normal(.1) kernels.
/// @see CausalPixelMask(), PixelCnn(), conv2d()
inline Tensor MaskedPixelConv(ParameterStore& parameters,const std::string& name,
                              const Tensor& input,const Tensor& mask) {
  gan_detail::Finite(mask);
  if(name.empty() || mask.dim()!=4 || mask.size(0)%2!=1 || mask.size(1)%2!=1 ||
     mask.dtype()!=parameters.dtype())
    throw std::invalid_argument("MaskedPixelConv requires a named matching odd rectangular kernel mask");
  autoregressive_pixel_detail::Image(input,mask.size(2),parameters.dtype());
  autoregressive_pixel_detail::Range(mask,true);
  if(input.numel()/input.size(3)>INT64_MAX/mask.size(3))
    throw std::invalid_argument("MaskedPixelConv output product overflows");
  auto w=parameters.get(name+"/W",mask.shape(),init::normal(.1));
  auto b=parameters.get(name+"/b",{mask.size(3)},init::zeros());
  return bias_add(conv2d(input,w*mask.detach()),b);
}

/// @brief Own the six live boundaries of the compact four-layer masked PixelCNN.
///
/// Flat Tensor handles retain graph/storage beyond parent/store lifetime. No
/// optimizer, fitted state, sample stream or class labels are retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <datamunge/nn/autoregressive_pixels.hpp>
/// namespace n=dm::nn;n::ParameterStore p(8401);
/// auto out=n::models::PixelCnn(p,n::zeros({2,8,8,1}));
/// // logits[2,8,8,1]; all four activated feature maps have width8.
/// @endcode
/// @par References
/// - van den Oord, Kalchbrenner and Kavukcuoglu(2016), Pixel Recurrent Neural
/// Networks, https://arxiv.org/abs/1601.06759 . Compact teaching architecture.
/// @see PixelCnn(), BernoulliPixels()
struct PixelCnnOutput {
  Tensor input_features; ///< Owning live mask-A head preactivation[N,H,W,8].
  Tensor head;           ///< Owning live ReLU(input_features)[N,H,W,8].
  Tensor hidden0;        ///< Owning live first mask-B ReLU[N,H,W,8].
  Tensor hidden1;        ///< Owning live second mask-B ReLU[N,H,W,8].
  Tensor hidden;         ///< Owning live third mask-B ReLU[N,H,W,8].
  Tensor logits;         ///< Owning live unmasked1x1 Bernoulli logits[N,H,W,1].
};

/// @brief Evaluate a compact raster-causal binary PixelCNN with four width-eight ReLU layers.
///
/// Finite signed matching Float32/64 CPU input[N,H,W,1], positive rectangular
/// axes and arbitrary strides. Four biased3x3 SAME stride1 layers: input mask A
/// 1->8, hidden0/1/2 mask B8->8, each followed by ReLU. An unmasked1x1 biased
/// 8->1 head returns Bernoulli logits. Missing W use normal(0,.1), b zeros;
/// source store seed8401 creates1841 values in10 leaves, of which808 masked
/// kernel coordinates receive exact zero partials. No current/future input
/// dependence, normalization, pooling, dropout or batch coupling. Nominal9x9
/// receptive field retains the ordinary masked-CNN blind spot. Inputs validate
/// before leaf creation; stored-leaf/math errors may follow creation. All live
/// input/weight first/higher paths and six real boundaries are returned. No
/// training loss, fitting, sampling, optimizer, existing-leaf writes or hidden
/// no-grad. Work/storage O(NHW), fixed width8; synchronize shared mutable stores.
/// @param parameters Caller-owned matching floating store, source seed8401.
/// @param input Finite signed CPU[N,H,W,1] image canvas; training binary targets are separate.
/// @return Owning six-field PixelCnnOutput, including logits[N,H,W,1].
/// @throws std::invalid_argument For incompatible input geometry/dtype/products/values or leaves.
/// @throws std::logic_error For unsupported storage or primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <datamunge/nn/autoregressive_pixels.hpp>
/// namespace n=dm::nn;n::ParameterStore p(8401);
/// auto out=n::models::PixelCnn(p,n::zeros({2,8,8,1}));
/// // p.count()==1841,p.size()==10; logits are teacher-forced or sampling-canvas logits.
/// @endcode
/// @par References
/// - van den Oord, Kalchbrenner and Kavukcuoglu(2016), Pixel Recurrent Neural
/// Networks, https://arxiv.org/abs/1601.06759 . Source teaching variant uses
/// four width8 ReLU layers and a binary head, preserving the standard blind spot.
/// @see PixelCnnOutput, CausalPixelMask(), MaskedPixelConv(), BernoulliPixels()
inline PixelCnnOutput PixelCnn(ParameterStore& parameters,const Tensor& input) {
  autoregressive_pixel_detail::Image(input,1,parameters.dtype());
  if(input.numel()>INT64_MAX/8)
    throw std::invalid_argument("PixelCnn feature product overflows");
  const Tensor a=CausalPixelMask(1,8,false,parameters.dtype());
  const Tensor b=CausalPixelMask(8,8,true,parameters.dtype());
  PixelCnnOutput out;
  out.input_features=MaskedPixelConv(parameters,"input",input,a);
  out.head=relu(out.input_features);
  out.hidden0=relu(MaskedPixelConv(parameters,"hidden0",out.head,b));
  out.hidden1=relu(MaskedPixelConv(parameters,"hidden1",out.hidden0,b));
  out.hidden=relu(MaskedPixelConv(parameters,"hidden2",out.hidden1,b));
  const auto w=parameters.get("output/W",{1,1,8,1},init::normal(.1));
  out.logits=bias_add(conv2d(out.hidden,w),parameters.get("output/b",{1},init::zeros()));
  gan_detail::Finite(out.logits);
  return out;
}

/// @brief Own live Bernoulli probabilities and teacher-forced image-density losses.
///
/// Four flat tensors retain their input graphs/storage beyond parent lifetime.
/// loss averages all pixels and rows; per_image_nll sums each image in nats.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <datamunge/nn/autoregressive_pixels.hpp>
/// namespace n=dm::nn;
/// auto out=n::models::BernoulliPixels(n::zeros({2,8,8,1}),n::ones({2,8,8,1}));
/// // loss=log(2) nats/pixel; per_image_nll has two values, each64*log(2).
/// @endcode
/// @par References
/// - Bernoulli log likelihood in natural units, using stable softplus(z)-x*z;
/// raster image factorization follows https://arxiv.org/abs/1601.06759 .
/// @see BernoulliPixels(), PixelCnn()
struct BernoulliPixelOutput {
  Tensor probability;   ///< Owning live sigmoid(logits)[N,H,W,1] probabilities.
  Tensor pixel_nll;     ///< Owning live softplus(logits)-binary*logits[N,H,W,1] in nats.
  Tensor per_image_nll; ///< Owning live sum over H,W,C[N], nats per image.
  Tensor loss;          ///< Owning live scalar mean over N,H,W,C, nats per binary pixel.
};

/// @brief Compute a stable teacher-forced Bernoulli density for exact binary targets.
///
/// Finite matching Float32/64 CPU logits and binary[N,H,W,1], positive axes,
/// arbitrary strides; targets must be exact zero/one. softplus(logits)-binary*
/// logits yields per-pixel nats. Return sigmoid probabilities, that pixel loss,
/// each image's HWC sum and the overall mean. The unit is a binary8x8 pixel
/// when used by the source example, not an original28x28 gray pixel. No loss
/// clipping, bit conversion, input mutation, fitting or hidden no-grad. Both
/// applicable logit/target first/higher paths remain live. Four owning results
/// retain storage/graphs after inputs/parent collection. Work/storage O(NHW);
/// independent calls share no mutable state. Extreme finite inputs can still
/// overflow reductions, which are rejected when the scalar loss is nonfinite.
/// @param logits Finite signed floating CPU[N,H,W,1] Bernoulli logits.
/// @param binary Matching finite exact zero/one CPU[N,H,W,1] teacher-forcing targets.
/// @return Owning four-field BernoulliPixelOutput, losses in natural-log units.
/// @throws std::invalid_argument For incompatible inputs, nonbinary targets or nonfinite loss.
/// @throws std::logic_error For unsupported storage or primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <datamunge/nn/autoregressive_pixels.hpp>
/// namespace n=dm::nn;
/// auto out=n::models::BernoulliPixels(n::zeros({1,8,8,1}),n::ones({1,8,8,1}));
/// // out.loss is approximately .693147 nats per binary pixel.
/// @endcode
/// @par References
/// - van den Oord, Kalchbrenner and Kavukcuoglu(2016), Pixel Recurrent Neural
/// Networks, https://arxiv.org/abs/1601.06759 . Binary teaching likelihood uses
/// a Bernoulli head and teacher forcing, rather than a256-way color head.
/// @see BernoulliPixelOutput, PixelCnn(), softplus()
inline BernoulliPixelOutput BernoulliPixels(const Tensor& logits,const Tensor& binary) {
  autoregressive_pixel_detail::Image(logits,1,logits.dtype());
  autoregressive_pixel_detail::Image(binary,1,logits.dtype());
  if(logits.shape()!=binary.shape())
    throw std::invalid_argument("BernoulliPixels requires aligned logits and binary targets");
  autoregressive_pixel_detail::Range(binary,true);
  BernoulliPixelOutput out;
  out.probability=sigmoid(logits);
  out.pixel_nll=softplus(logits)-binary*logits;
  out.per_image_nll=sum(out.pixel_nll,{1,2,3});
  out.loss=mean(out.pixel_nll);
  gan_detail::Finite(out.loss);
  return out;
}

/// @brief Draw a complete explicit64-site bank of midpoint MT19937 sampling uniforms.
///
/// Positive rows N, result Double[N,64] in row-then-raster-ordinal order.
/// Each value is(raw_uint32+.5)/2^32, strictly inside(0,1). Every row draws all
/// 64 values even when a caller preserves a prefix. Owning detached CPU data;
/// no global generator, parameter store, model call or hidden sampling loop.
/// Work/storage O(64N); independent calls use independent local RNGs.
/// @param rows Positive requested sample count, with representable64*rows.
/// @param seed Local uint32 MT19937 seed, source8423 for sampling and8443 for completion.
/// @return Owning detached Float64[N,64] uniform bank, reproducible by seed.
/// @throws std::invalid_argument For nonpositive or overflowing row count.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <datamunge/nn/autoregressive_pixels.hpp>
/// auto u=dm::nn::models::BinaryPixelUniforms(32,8423);
/// // u[32,64]; a prefix-completion caller still draws this complete bank.
/// @endcode
/// @par References
/// - Preserves the original binary-pixel teaching sampler's raw MT19937
/// midpoint stream; this is not std::uniform_real_distribution's mapping.
/// @see PixelCnn(), BernoulliPixels()
inline Tensor BinaryPixelUniforms(std::int64_t rows,std::uint32_t seed) {
  if(rows<=0 || rows>INT64_MAX/64)
    throw std::invalid_argument("BinaryPixelUniforms requires positive representable rows");
  std::mt19937 rng(seed);
  std::vector<double> value(static_cast<std::size_t>(rows*64));
  for(auto& u:value)u=(double(rng())+.5)/4294967296.;
  return from_vector(value,{rows,64});
}
} // namespace datamunge::nn::models
