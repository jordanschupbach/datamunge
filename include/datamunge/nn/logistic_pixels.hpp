#pragma once
/// @file logistic_pixels.hpp
/// @brief Quantized RGB area preparation and stable channel-coupled logistic pixel densities.
#include <limits>
#include "datamunge/nn/gated_pixels.hpp"

namespace datamunge::nn::models {
/// @brief Own the integer area codes and their signed Float32 image representation.
///
/// Two flat independent detached Tensor handles survive input/parent collection.
/// No source row IDs, labels, fitting or dataset selection are retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto raw=dm::nn::full({1,4,4,3},255,dm::nn::DType::Int32);
/// auto out=dm::nn::models::RgbAreaPixels(raw,4);
/// // out.codes[1,1,1,3]=255; out.images[1,1,1,3]=1.
/// @endcode
/// @par References
/// - Original compact PixelCNN++ teaching data protocol: integer area means,
/// nearest half-up8-bit codes, then code/127.5-1 with one Float32 cast.
/// @see RgbAreaPixels(), PixelCnnPP()
struct QuantizedRgbData {
  Tensor codes;  ///< Owning detached Int32[N,H/block,W/block,3] codes in0..255.
  Tensor images; ///< Owning detached Float32 of the same shape, double(code)/127.5-1 cast once.
};

/// @brief Average disjoint integer RGB blocks and round to nearest half-up byte codes.
///
/// Int32/64 CPU original[N,H,W,3], all axes positive, values0..255, arbitrary
/// strides. Positive block_size(default4) must divide H/W. Sum each disjoint
/// square in integer arithmetic, round its mean half-up to an Int32 code, then
/// compute double(code)/127.5-1 and cast once to Float32. For32x32 and block4
/// this is the original compact PixelCNN++8x8 preparation, preserving exact
/// quantized source values. This is not bilinear resizing. No source mutation,
/// split selection, normalization fitting, RNG or input gradient. Independent
/// owning outputs, work O(NHW), storage O(NHW/block_size^2); calls share no state.
/// @param original Integer CPU NHWC byte-valued RGB source batch.
/// @param block_size Positive square block extent dividing H/W, default4.
/// @return Owning stopped QuantizedRgbData with codes and signed images.
/// @throws std::invalid_argument For invalid geometry/type/range or unsafe integer block sum.
/// @throws std::logic_error For unsupported storage or primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto raw=dm::nn::from_vector(std::vector<std::int32_t>{0,0,0,1,1,1,0,0,0,1,1,1},{1,2,2,3});
/// auto out=dm::nn::models::RgbAreaPixels(raw,2);
/// // A mean of.5 rounds up: all three codes=1, images=float(1/127.5-1).
/// @endcode
/// @par References
/// - Preserves the source compact RGB density protocol:4x4 integer area means,
/// nearest half-up codes and signed8-bit-code representation; no learned transform.
/// @see QuantizedRgbData, DiscretizedLogisticRGB(), PixelCnnPP()
inline QuantizedRgbData RgbAreaPixels(const Tensor& original,std::int64_t block_size=4) {
  if(!original.defined() || original.device().backend!=compute::Backend::CPU ||
     (original.dtype()!=DType::Int32 && original.dtype()!=DType::Int64) ||
     original.dim()!=4 || original.size(0)<=0 || original.size(1)<=0 || original.size(2)<=0 ||
     original.size(3)!=3 || block_size<=0 || original.size(1)%block_size || original.size(2)%block_size ||
     block_size>std::numeric_limits<std::int64_t>::max()/block_size ||
     block_size*block_size>std::numeric_limits<std::int64_t>::max()/256)
    throw std::invalid_argument("RgbAreaPixels requires positive divisible integer CPU RGB byte images and safe blocks");
  auto raw=cast(original,DType::Int64).contiguous().to_vector<std::int64_t>();
  for(auto v:raw)if(v<0 || v>255)throw std::invalid_argument("RgbAreaPixels source codes must be in0..255");
  const auto n=original.size(0),h=original.size(1),w=original.size(2),oh=h/block_size,ow=w/block_size,area=block_size*block_size;
  std::vector<std::int32_t> codes(static_cast<std::size_t>(n*oh*ow*3));
  std::vector<float> images(codes.size());
  for(std::int64_t b=0;b<n;++b)for(std::int64_t y=0;y<oh;++y)for(std::int64_t x=0;x<ow;++x)for(int c=0;c<3;++c) {
    std::int64_t sum=0;
    for(std::int64_t dy=0;dy<block_size;++dy)for(std::int64_t dx=0;dx<block_size;++dx)
      sum+=raw[static_cast<std::size_t>(((b*h+y*block_size+dy)*w+x*block_size+dx)*3+c)];
    const auto index=static_cast<std::size_t>(((b*oh+y)*ow+x)*3+c);
    codes[index]=static_cast<std::int32_t>((sum+area/2)/area);
    images[index]=static_cast<float>(double(codes[index])/127.5-1);
  }
  return {from_vector(codes,{n,oh,ow,3}),from_vector(images,{n,oh,ow,3})};
}

/// @brief Own live RGB logistic-mixture probabilities and their actual intermediate fields.
///
/// Eight flat handles keep applicable image/parameter first and higher paths
/// alive after parent collection. P is the product of image axes before its
/// final RGB axis, K is the positive component count. All fields match the
/// input floating dtype; pixel ordering follows contiguous flattened NHWC.
/// No model/store, fitting, sampling or mutation is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto o=dm::nn::models::DiscretizedLogisticRGB(dm::nn::zeros({2,3}),dm::nn::zeros({2,30}));
/// // pixel_log_probability[2], channel_log_mass[2,3,3], scalar mean_channel_nll.
/// @endcode
/// @par References
/// - Salimans, Karpathy, Chen and Kingma(2017), PixelCNN++: Improving the PixelCNN
/// with Discretized Logistic Mixture Likelihood and Other Modifications,
/// https://arxiv.org/abs/1701.05517 . Stable exact-bin compact teaching variant.
/// @see DiscretizedLogisticRGB(), PixelCnnPPOutput, PixelCnnPP()
struct LogisticRgbOutput {
  Tensor pixel_log_probability;     ///< Owning live[P] joint log probability of each RGB pixel.
  Tensor component_log_probability; ///< Owning live[P,K] sum of three conditional channel log masses.
  Tensor mixture_log_probability;   ///< Owning live[P,K] normalized log weights, log_softmax(raw logits).
  Tensor means;                     ///< Owning live[P,K,3] means after teacher-forced R/G coupling.
  Tensor log_scales;                ///< Owning live[P,K,3] raw log scales lower-clipped at-7.
  Tensor coefficients;              ///< Owning live[P,K,3] tanh coefficients RG,BR,BG in that order.
  Tensor channel_log_mass;          ///< Owning live[P,K,3] stable8-bit bin log masses including infinite tails.
  Tensor mean_channel_nll;          ///< Owning live scalar -mean(pixel_log_probability)/3, nats per RGB channel.
};

/// @brief Evaluate a discretized RGB logistic mixture with triangular within-pixel coupling.
///
/// Matching finite Float32/64 CPU image[...,3] and parameters[...,10*K], same
/// leading axes, each positive, arbitrary strides, K=mixtures>0(default3).
/// Image values must be in[-1,1]; exact8-bit observations use code/127.5-1.
/// Interior continuous values are permitted for differentiation but do not
/// identify a separate normalized categorical grid. Component-major parameter
/// layout is logit, meanR/G/B, logscaleR/G/B, coefficientRG/BR/BG. Clip logscales
/// from below at-7(ties retain the raw-parameter derivative), tanh coefficients,
/// then add coefficientRG*observedR to meanG and coefficientBR*observedR plus
/// coefficientBG*observedG to meanB. Means are not clipped.
///
/// For each channel use hi=(x-mean+step)*exp(-s),lo=(x-mean-step)*exp(-s),
/// delta=width*exp(-s), middle=hi-softplus(hi)-softplus(lo)+log(-expm1(-delta)).
/// Low x<=float(-.999) uses -softplus(-hi); high x>=float(.999) uses
/// -softplus(lo), extending the endpoint bins to infinity. Source-compatible
/// step=float(1/255),width=float(2/255) are promoted even for Float64. No small-mass
/// fallback or CDF subtraction. Sum channel log masses, add log_softmax mixture
/// weights, then stable logsumexp with a stopped maximum. All applicable first/
/// higher paths remain live; returned mean loss is nats per RGB channel.
///
/// Outputs flatten leading axes to P; owning handles outlive the inputs. No
/// mutation, fitting, RNG, sampling, store or hidden no-grad; O(PK) work/storage,
/// local independent calls. Invalid input rejects before math. Extremely large
/// finite parameters may overflow/underflow intermediate masses; nonfinite
/// returned fields reject rather than silently clipping more of the density.
/// @param image Finite floating CPU[...,3] signed RGB observations in[-1,1].
/// @param parameters Matching finite CPU[...,10*K] component-major parameters.
/// @param mixtures Positive K(default3) with representable10*K.
/// @return Owning live LogisticRgbOutput, preserving input floating dtype.
/// @throws std::invalid_argument For incompatible geometry/dtype/range, unsafe K or nonfinite outputs.
/// @throws std::logic_error For unsupported storage or primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto p=n::zeros({1,10},n::DType::Float64).requires_grad_();
/// auto o=n::models::DiscretizedLogisticRGB(n::zeros({1,3},n::DType::Float64),p,1);
/// auto g=n::grad(o.mean_channel_nll,{p})[0];
/// // Three center-bin logistic masses; finite live gradient[1,10].
/// @endcode
/// @par References
/// - Salimans, Karpathy, Chen and Kingma(2017), PixelCNN++: Improving the PixelCNN
/// with Discretized Logistic Mixture Likelihood and Other Modifications,
/// https://arxiv.org/abs/1701.05517 . Uses the teaching source's stable exact-bin
/// identity and minimum log scale; no published multiscale architecture claim.
/// @see LogisticRgbOutput, PixelCnnPP(), log_softmax(), expm1()
inline LogisticRgbOutput DiscretizedLogisticRGB(const Tensor& image,const Tensor& parameters,int mixtures=3) {
  gan_detail::Finite(image);gan_detail::Finite(parameters);
  if(mixtures<=0 || mixtures>std::numeric_limits<int>::max()/10 || image.dim()<1 ||
     parameters.dim()!=image.dim() || image.dtype()!=parameters.dtype() ||
     image.size(image.dim()-1)!=3 || parameters.size(parameters.dim()-1)!=10*mixtures)
    throw std::invalid_argument("DiscretizedLogisticRGB requires matching floating[...,3]/[...,10*K]");
  for(std::int64_t i=0;i<image.dim()-1;++i)if(image.size(i)!=parameters.size(i))
    throw std::invalid_argument("DiscretizedLogisticRGB leading axes must match");
  for(double x:image.contiguous().to_vector<double>())if(x< -1 || x>1)
    throw std::invalid_argument("DiscretizedLogisticRGB observations must lie in[-1,1]");
  const Tensor raw=reshape(parameters,{-1,mixtures,10}),rgb=reshape(image,{-1,3});
  /// @brief Select and flatten one validated component parameter slot.
  /// @param i Slot0..9 in the validated component-major parameter tensor.
  /// @return Live[P,K] strided parameter selection.
  auto part=[&](int i){return reshape(slice_axis(raw,2,i,i+1),{-1,mixtures});};
  /// @brief Select one observed RGB channel while retaining broadcasting width1.
  /// @param i Channel0..2 of the validated flattened RGB image.
  /// @return Live[P,1] observed channel slice.
  auto channel=[&](int i){return slice_axis(rgb,1,i,i+1);};
  const Tensor logits=part(0);
  std::vector<Tensor> means,scales,coefficients,log_mass;
  for(int c=0;c<3;++c){means.push_back(part(c+1));scales.push_back(maximum(part(c+4),scalar_like(-7.,part(c+4))));coefficients.push_back(tanh(part(c+7)));}
  means[1]=add(means[1],mul(coefficients[0],channel(0)));
  means[2]=add(means[2],add(mul(coefficients[1],channel(0)),mul(coefficients[2],channel(1))));
  /// @brief Source-compatible Float32 half-bin width, promoted for Double arithmetic.
  const float step=1.f/255;
  /// @brief Source-compatible Float32 full-bin width, promoted for Double arithmetic.
  const float width=2.f/255;
  for(int c=0;c<3;++c){
    const Tensor inv=exp(neg(scales[c])),centered=sub(channel(c),means[c]);
    const Tensor hi=mul(add(centered,double(step)),inv),lo=mul(sub(centered,double(step)),inv),delta=mul(inv,double(width));
    const Tensor middle=add(sub(sub(hi,softplus(hi)),softplus(lo)),log(neg(expm1(neg(delta)))));
    const Tensor lower=neg(softplus(neg(hi))),upper=neg(softplus(lo));
    log_mass.push_back(where(broadcast_to(less_equal(channel(c),-.999f),middle.shape()),lower,
      where(broadcast_to(greater_equal(channel(c),.999f),middle.shape()),upper,middle)));
  }
  const Tensor components=add(add(log_mass[0],log_mass[1]),log_mass[2]);
  const Tensor mixture=log_softmax(logits,1),weighted=add(components,mixture);
  const Tensor maximum=stop_gradient(max(weighted,{1},true));
  const Tensor pixel=add(reshape(maximum,{-1}),log(sum(exp(sub(weighted,maximum)),{1})));
  LogisticRgbOutput out{pixel,components,mixture,stack(means,2),stack(scales,2),stack(coefficients,2),stack(log_mass,2),div(neg(mean(pixel)),3.)};
  for(const auto& x:{out.pixel_log_probability,out.component_log_probability,out.mixture_log_probability,
      out.means,out.log_scales,out.coefficients,out.channel_log_mass,out.mean_channel_nll})gan_detail::Finite(x);
  return out;
}

/// @brief Own the compact gated RGB context, logistic likelihood and image joint losses.
///
/// Twelve flat owning fields retain applicable image/parameter first/higher
/// paths after parent/store collection. Context parameters remain in the caller's
/// ParameterStore; no fitting, optimizer or sampling is hidden in this record.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore p(8609);
/// auto out=dm::nn::models::PixelCnnPP(p,dm::nn::zeros({2,8,8,3}));
/// // out.parameters[2,8,8,30], per_image_nll[2], scalar loss in nats/channel.
/// @endcode
/// @par References
/// - Salimans et al.(2017), https://arxiv.org/abs/1701.05517 . Compact three
/// width4 gated blocks and three logistic components; no multiscale/dropout.
/// @see PixelCnnPP(), LogisticRgbOutput, GatedPixelConvolutional()
struct PixelCnnPPOutput {
  Tensor parameters;                ///< Owning live[N,H,W,30] whole-pixel-causal component-major context head.
  Tensor vertical;                  ///< Owning live[N,H,W,4] final vertical gate features.
  Tensor horizontal;                ///< Owning live[N,H,W,4] final horizontal residual features.
  Tensor pixel_log_probability;     ///< Owning live[N*H*W] joint RGB pixel log probabilities.
  Tensor component_log_probability; ///< Owning live[N*H*W,3] conditional RGB component log probabilities.
  Tensor mixture_log_probability;   ///< Owning live[N*H*W,3] normalized context mixture log weights.
  Tensor means;                     ///< Owning live[N*H*W,3,3] means after observed R/G coupling.
  Tensor log_scales;                ///< Owning live[N*H*W,3,3] lower-clipped log scales.
  Tensor coefficients;              ///< Owning live[N*H*W,3,3] tanh coupling coefficients RG,BR,BG.
  Tensor channel_log_mass;          ///< Owning live[N*H*W,3,3] exact-bin channel log masses with endpoint tails.
  Tensor per_image_nll;             ///< Owning live[N] sum of H*W RGB joint pixel NLLs in nats.
  Tensor loss;                      ///< Owning live scalar mean NLL in nats per RGB channel.
};

/// @brief Evaluate the compact three-block whole-pixel-causal RGB logistic mixture density.
///
/// Finite matching Float32/64 CPU image[N,H,W,3], positive axes, values[-1,1],
/// arbitrary strides, caller-owned matching floating ParameterStore. Three
/// dual-stream width4 gated blocks feed a30-channel head for three logistic
/// components. Whole current/future raw pixels are excluded from every head
/// parameter; within-pixel R/G dependence belongs to the separate likelihood.
/// Source8x8 geometry and seed8609 yield1410 stored values in23 ordered leaves,
///962 active and448 structurally masked. Other positive rectangles are supported.
/// No labels, dropout, multiscale stack, learned initial state or batch coupling.
///
/// Missing leaves initialize normal(.1) kernels/zero biases in source order;
/// existing leaves are reused. Input validates before leaves. Applicable first/
/// higher image/store paths stay live and all twelve outputs own storage/graphs.
/// No fitting, optimizer, sampling, RNG bank, global mutation or hidden no-grad;
/// O(NHW) fixed-width work/storage. Independent stores can run concurrently;
/// callers synchronize writes to a shared store. Finite input may still produce
/// overflowing arithmetic: nonfinite context/likelihood output rejects.
/// @param parameters Caller-owned matching floating store, initialized/reused in gated source order.
/// @param image Finite matching CPU[N,H,W,3] signed RGB observations in[-1,1].
/// @return Owning live PixelCnnPPOutput with scalar nats/channel and summed image joint NLLs.
/// @throws std::invalid_argument For invalid input, incompatible existing leaves or nonfinite output fields.
/// @throws std::logic_error For unsupported storage or primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(8609);
/// auto o=n::models::PixelCnnPP(p,n::zeros({1,8,8,3}));
/// auto g=n::grad(o.loss,p.parameters());
/// // p.count()==1410, p.size()==23; optimizer and complete training loop remain explicit.
/// @endcode
/// @par References
/// - Salimans, Karpathy, Chen and Kingma(2017), PixelCNN++: Improving the PixelCNN
/// with Discretized Logistic Mixture Likelihood and Other Modifications,
/// https://arxiv.org/abs/1701.05517 . Compact teaching context, not the published
/// multiscale network; retains exact RGB mixture and whole-pixel causal semantics.
/// @see PixelCnnPPOutput, DiscretizedLogisticRGB(), GatedPixelConvolutional(), RgbAreaPixels()
inline PixelCnnPPOutput PixelCnnPP(ParameterStore& parameters,const Tensor& image) {
  autoregressive_pixel_detail::Image(image,3,parameters.dtype());
  for(double v:image.contiguous().to_vector<double>())if(v< -1 || v>1)
    throw std::invalid_argument("PixelCnnPP observations must lie in[-1,1]");
  auto c=GatedPixelConvolutional(parameters,image,Tensor(),false,3,30);
  auto l=DiscretizedLogisticRGB(image,c.logits);
  PixelCnnPPOutput out{c.logits,c.vertical,c.horizontal,l.pixel_log_probability,l.component_log_probability,
      l.mixture_log_probability,l.means,l.log_scales,l.coefficients,l.channel_log_mass,
      neg(sum(reshape(l.pixel_log_probability,{image.size(0),-1}),{1})),l.mean_channel_nll};
  gan_detail::Finite(out.per_image_nll);return out;
}
} // namespace datamunge::nn::models
