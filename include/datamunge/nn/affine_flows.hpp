#pragma once
/// @file affine_flows.hpp
/// @brief Live diagonal flow scaling and compact bounded affine RealNVP maps.
#include "datamunge/nn/nice_density.hpp"

namespace datamunge::nn::models {
/// @brief Own a live transformed point batch and accumulated scalar or per-row log determinant.
///
/// Two flat Tensor handles retain applicable first/higher input, incoming
/// determinant and parameter graphs/storage after parent/store collection.
/// No optimizer, sampling stream or model state is retained separately.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9403);
/// auto out=n::models::FlowDiagonalScale(p,n::ones({3,2}),n::zeros({3}));
/// // out.value[3,2] is unchanged initially; out.logdet[3] contains zeros.
/// @endcode
/// @par References
/// - Dinh, Sohl-Dickstein and Bengio(2016), Density estimation using Real NVP,
/// https://arxiv.org/abs/1605.08803 .
/// @see FlowDiagonalScale(), RealNvpInverse()
struct FlowScaleOutput {
  Tensor value;  ///< Owning live matching Float32/64 CPU[N,D] transformed points.
  Tensor logdet; ///< Owning live matching scalar or[N] accumulated log absolute determinant, nats.
};

/// @brief Apply a learned global positive diagonal scale and accumulate its signed log determinant.
///
/// Finite matching Float32/64 CPU input[N,D] and incoming_logdet scalar or[N],
/// positive axes and arbitrary strides. Create missing log_scale[D] as zeros.
/// Forward returns input*exp(log_scale), incoming_logdet+sum(log_scale).
/// Inverse returns input*exp(-log_scale), incoming_logdet-sum(log_scale).
/// Both directions of the scale must remain positive finite; underflow to zero,
/// overflow and nonfinite results reject without clipping. Applicable first/
/// higher paths stay live, including incoming_logdet. Existing leaves never
/// change; only missing leaves initialize. No fitting, optimizer, sampling, RNG
/// after initialization or hidden no-grad. Owns outputs/graphs after parent/store
/// collection. O(ND) work/storage; synchronize shared mutable stores. Supplied
/// inputs validate before leaf creation; stored-leaf/math failures can follow it.
/// This function does not derive an incoming Jacobian or prove invertibility of
/// earlier stages. A scalar logdet remains scalar; a per-row logdet remains[N].
/// @param parameters Caller-owned floating store matching input/determinant dtype.
/// @param input Finite matching floating CPU[N,D], N,D>0, arbitrary strides.
/// @param incoming_logdet Finite matching scalar or[N] accumulated log absolute determinant.
/// @param inverse False(default) for multiplication; true for division and negative logdet.
/// @return Owning live FlowScaleOutput with transformed points and accumulated logdet.
/// @throws std::invalid_argument For invalid inputs/store/scale or nonfinite stored/computed values.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9403,n::DType::Float64);
/// p.set("log_scale",n::from_vector(std::vector<double>{std::log(2.),-std::log(2.)},{2}));
/// auto out=n::models::FlowDiagonalScale(p,n::ones({1,2},n::DType::Float64),n::full({},.3,n::DType::Float64));
/// // out.value is approximately[[2,.5]], scalar logdet remains.3.
/// @endcode
/// @par References
/// - Dinh, Sohl-Dickstein and Bengio(2016), Density estimation using Real NVP,
/// https://arxiv.org/abs/1605.08803 . Source compact flows use one final global scale.
/// @see FlowScaleOutput, RealNvpDensity(), RealNvpInverse()
inline FlowScaleOutput FlowDiagonalScale(ParameterStore& parameters,const Tensor& input,
                                        const Tensor& incoming_logdet,bool inverse=false) {
  flow_detail::Matrix(input);gan_detail::Finite(incoming_logdet);
  if(input.dtype()!=parameters.dtype() || incoming_logdet.dtype()!=input.dtype() ||
     incoming_logdet.device().backend!=compute::Backend::CPU ||
     !(incoming_logdet.dim()==0 || (incoming_logdet.dim()==1 && incoming_logdet.size(0)==input.size(0))))
    throw std::invalid_argument("FlowDiagonalScale needs matching floating points and scalar/per-row logdet");
  const auto scale=parameters.get("log_scale",{input.size(1)},init::zeros());gan_detail::Finite(scale);
  const auto positive=exp(scale),negative=exp(-scale);
  gan_detail::Finite(positive);gan_detail::Finite(negative);
  for(auto v:{positive,negative})for(double x:v.contiguous().to_vector<double>())
    if(x<=0)throw std::invalid_argument("Flow diagonal scale and inverse must be positive finite");
  auto value=input*(inverse?negative:positive);
  auto total=sum(scale,{0});auto logdet=inverse?incoming_logdet-total:incoming_logdet+total;
  gan_detail::Finite(value);gan_detail::Finite(logdet);return {value,logdet};
}

/// @brief Own a live affine coupling value, signed determinant, bounded log scale and shift.
///
/// Four flat Tensor handles retain applicable input/parameter first/higher
/// paths/storage after parent/store collection. scale means log scale, not
/// exp(scale); its shape[N,1] is distinct from logdet[N]. No fitting or RNG state.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9403);
/// auto out=n::models::RealNvpCoupling(p,n::zeros({2,2}),0);
/// // value[2,2], logdet[2], scale/shift[2,1]; scale=.8*tanh(raw head).
/// @endcode
/// @par References
/// - Dinh, Sohl-Dickstein and Bengio(2016), Real NVP, https://arxiv.org/abs/1605.08803 .
/// @see RealNvpCoupling(), RealNvpDensity()
struct RealNvpCouplingOutput {
  Tensor value;  ///< Owning live matching floating CPU[N,2] coupled coordinates in original raw order.
  Tensor logdet; ///< Owning live matching floating[N] scale in forward, negative scale in inverse.
  Tensor scale;  ///< Owning live matching floating[N,1] bounded log scale .8*tanh(raw), not its exponential.
  Tensor shift;  ///< Owning live matching floating[N,1] unconstrained finite affine shift.
};

/// @brief Apply a source bounded affine coupling or its exact-form inverse on two coordinates.
///
/// Finite matching Float32/64 CPU input[N,2], positive N, arbitrary strides and
/// block0..3. Condition on coordinate block%2; preserve it in raw axis order.
/// Prefix coupling<block>/hidden is1->16 tanh, missing W normal(.3), b=.05;
/// /heads is16->2 linear, W normal(.01), b=0. Head0 gives s=.8*tanh(raw), head1
/// shift t. Forward changes the other coordinate to b*exp(s)+t with logdet=s;
/// inverse changes it to(b-t)*exp(-s) with logdet=-s. Evaluate both heads on
/// the supplied unchanged conditioning coordinate in either direction. Bounds
/// use .8 rounded to computation dtype, including Float32's nearest value.
/// Four leaves/66 values per block. Only missing leaves initialize, hidden W/b
/// then heads W/b; existing leaves never change. No fitting, optimizer, sampling,
/// RNG after warmup or hidden no-grad. Owns live results/graphs; applicable
/// first/higher paths survive parent/store collection. O(16N) work/storage;
/// synchronize shared stores. Inputs validate before leaves; stored-leaf/math
/// failures can follow partial creation. Nonfinite intermediate/results reject.
/// @param parameters Caller-owned matching floating store; full source seed9403.
/// @param input Finite matching Float32/64 CPU[N,2], N>0, arbitrary strides.
/// @param block Source block index0..3, conditioning coordinate block%2.
/// @param inverse False(default) for affine forward; true for inverse and negated logdet.
/// @return Owning live RealNvpCouplingOutput with value/logdet/scale/shift fields.
/// @throws std::invalid_argument For invalid inputs/block/store or nonfinite stored/computed values.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9403);
/// auto x=n::from_vector(std::vector<float>{1,2},{1,2});
/// auto forward=n::models::RealNvpCoupling(p,x,0);
/// auto rec=n::models::RealNvpCoupling(p,forward.value,0,true);
/// // forward.value[0,0]=1; rec.value approximates x and the two logdets cancel.
/// @endcode
/// @par References
/// - Dinh, Sohl-Dickstein and Bengio(2016), Density estimation using Real NVP,
/// https://arxiv.org/abs/1605.08803 . The .8*tanh bound and tiny conditioner are
/// the migrated CPU teaching adaptation, not the full image architecture.
/// @see RealNvpCouplingOutput, RealNvpDensity(), RealNvpInverse()
inline RealNvpCouplingOutput RealNvpCoupling(
    ParameterStore& parameters,const Tensor& input,int block,bool inverse=false) {
  flow_detail::Matrix(input,2);
  if(block<0 || block>3 || input.dtype()!=parameters.dtype() ||
     input.size(0)>std::numeric_limits<std::int64_t>::max()/16)
    throw std::invalid_argument("RealNvpCoupling needs block0..3 and matching finite floating matrices");
  const auto name="coupling"+std::to_string(block);const int condition=block%2;
  const auto a=slice_axis(input,1,condition,condition+1),b=slice_axis(input,1,1-condition,2-condition);
  const auto hidden=nice_detail::Dense(parameters,name+"/hidden",a,1,16,.3,.05,true);
  const auto heads=nice_detail::Dense(parameters,name+"/heads",hidden,16,2,.01,0);
  const auto scale=.8*tanh(slice_axis(heads,1,0,1)),shift=slice_axis(heads,1,1,2);
  const auto changed=inverse?(b-shift)*exp(-scale):b*exp(scale)+shift;
  auto value=condition?concat({changed,a},1):concat({a,changed},1);
  auto logdet=reshape(scale,{-1});if(inverse)logdet=-logdet;
  gan_detail::Finite(value);gan_detail::Finite(logdet);return {value,logdet,scale,shift};
}

/// @brief Own all four affine stages and their live per-row density intermediates.
///
/// Nineteen flat Tensor handles retain applicable first/higher graphs and
/// storage after parent/store collection. No nested record/vector ownership,
/// optimizer, fitting workflow or sampler is retained. scale fields mean log
/// scales .8*tanh(raw), not multiplicative scale factors.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9403);
/// auto out=n::models::RealNvpDensity(p,n::zeros({3,2}));
/// // coupling0..3/unscaled/latent[3,2], logdet/coupling_logdet0..3[3],
/// // coupling_scale0..3/coupling_shift0..3[3,1]. Total266 values/17 leaves.
/// @endcode
/// @par References
/// - Dinh, Sohl-Dickstein and Bengio(2016), Real NVP, https://arxiv.org/abs/1605.08803 .
/// @see RealNvpDensity(), RealNvpCoupling(), NormalFlowDensity()
struct RealNvpDensityOutput {
  Tensor coupling0;       ///< Owning live floating[N,2] first affine stage, condition coordinate0.
  Tensor coupling_logdet0;///< Owning live floating[N] first stage forward log absolute determinant.
  Tensor coupling_scale0; ///< Owning live floating[N,1] first stage bounded log scale.
  Tensor coupling_shift0; ///< Owning live floating[N,1] first stage affine shift.
  Tensor coupling1;       ///< Owning live floating[N,2] second stage, condition coordinate1.
  Tensor coupling_logdet1;///< Owning live floating[N] second stage forward log absolute determinant.
  Tensor coupling_scale1; ///< Owning live floating[N,1] second stage bounded log scale.
  Tensor coupling_shift1; ///< Owning live floating[N,1] second stage affine shift.
  Tensor coupling2;       ///< Owning live floating[N,2] third stage, condition coordinate0.
  Tensor coupling_logdet2;///< Owning live floating[N] third stage forward log absolute determinant.
  Tensor coupling_scale2; ///< Owning live floating[N,1] third stage bounded log scale.
  Tensor coupling_shift2; ///< Owning live floating[N,1] third stage affine shift.
  Tensor coupling3;       ///< Owning live floating[N,2] fourth stage, condition coordinate1.
  Tensor coupling_logdet3;///< Owning live floating[N] fourth stage forward log absolute determinant.
  Tensor coupling_scale3; ///< Owning live floating[N,1] fourth stage bounded log scale.
  Tensor coupling_shift3; ///< Owning live floating[N,1] fourth stage affine shift.
  Tensor unscaled;        ///< Live floating[N,2] alias of coupling3 before final global scaling.
  Tensor latent;          ///< Owning live matching floating[N,2] globally scaled Gaussian coordinates.
  Tensor logdet;          ///< Owning live matching floating[N] accumulated forward log determinant.
};

/// @brief Map two-dimensional observations through four bounded affine RealNVP stages and global scale.
///
/// Finite matching Float32/64 CPU input[N,2], positive rows and arbitrary
/// strides. Apply RealNvpCoupling blocks0..3, accumulate each per-row logdet in
/// order from input-linked zeros, then FlowDiagonalScale. Expose every source
/// stage, scale/shift and determinant, plus unscaled/latent/total logdet. Four
///1->16 tanh->2 conditioners and log_scale[2] give266 values/17 leaves; warm
/// forward with source seed9403 to preserve initialization order. Only missing
/// leaves initialize; warmed inference is immutable and consumes no RNG.
/// No fitting, optimizer, sampling, hidden no-grad, normalization or multiscale
/// image architecture. Owns live graphs/fields after parent/store collection;
/// first/higher paths remain applicable. O(64N) work/storage; synchronize shared
/// stores. Inputs validate before creation; stored-leaf/math failures can follow
/// partial creation. Both global scale directions must stay positive finite.
/// Nonfinite intermediates/results reject; finite roundtrip errors remain.
/// @param parameters Caller-owned matching floating store, source initialization seed9403.
/// @param input Finite matching Float32/64 CPU[N,2] observed coordinates.
/// @return Owning live RealNvpDensityOutput with19 flat source diagnostic fields.
/// @throws std::invalid_argument For invalid inputs/store/scale or nonfinite stored/computed values.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9403);
/// auto path=n::models::RealNvpDensity(p,n::zeros({2,2}));
/// auto density=n::models::NormalFlowDensity(path.latent,path.logdet,true);
/// // p.count()==266,p.size()==17; loss is mean source-compatible NLL in nats/row.
/// @endcode
/// @par References
/// - Dinh, Sohl-Dickstein and Bengio(2016), Density estimation using Real NVP,
/// https://arxiv.org/abs/1605.08803 . This four-stage2-D CPU teaching model uses
/// bounded .8*tanh log scales; the full paper's image/multiscale model differs.
/// @see RealNvpDensityOutput, RealNvpInverse(), FlowDiagonalScale()
inline RealNvpDensityOutput RealNvpDensity(ParameterStore& parameters,const Tensor& input) {
  flow_detail::Matrix(input,2);
  if(input.dtype()!=parameters.dtype())throw std::invalid_argument("RealNVP input/store dtypes must match");
  auto ld=reshape(slice_axis(input,1,0,1),{-1})*0.;
  auto a=RealNvpCoupling(parameters,input,0);ld=ld+a.logdet;
  auto b=RealNvpCoupling(parameters,a.value,1);ld=ld+b.logdet;
  auto c=RealNvpCoupling(parameters,b.value,2);ld=ld+c.logdet;
  auto d=RealNvpCoupling(parameters,c.value,3);ld=ld+d.logdet;
  auto scaled=FlowDiagonalScale(parameters,d.value,ld);
  return {a.value,a.logdet,a.scale,a.shift,b.value,b.logdet,b.scale,b.shift,
          c.value,c.logdet,c.scale,c.shift,d.value,d.logdet,d.scale,d.shift,
          d.value,scaled.value,scaled.logdet};
}

/// @brief Invert compact RealNVP and accumulate per-row log determinants along the inverse path.
///
/// Matching finite Float32/64 CPU latent[N,2], positive rows and arbitrary
/// strides. Start with input-linked zero determinants, unscale globally, then
/// reverse affine blocks3,2,1,0, evaluating each bounded log scale/shift on that
/// inverse path's unchanged coordinate. The returned inverse logdet depends
/// on the supplied latent, not on a cached forward pass or unrelated point.
/// A forward-warmed store reuses all266 values without mutation/RNG; missing
/// leaves initialize global scale first then reversed conditioners, so warm
/// RealNvpDensity to reproduce source seed9403 ordering. No fitting, sampling,
/// optimizer or hidden no-grad. Owns live result/graph after parent/store
/// collection; applicable first/higher paths remain live. O(64N) work/storage;
/// synchronize shared stores. Validate supplied inputs before creation; leaf/
/// math failures can follow partial creation. Both scale directions must stay
/// positive finite, nonfinite results reject; floating roundtrip errors remain.
/// @param parameters Caller-owned matching floating store, preferably forward-warmed.
/// @param latent Finite matching Float32/64 CPU[N,2] caller-supplied latent coordinates.
/// @return Owning live FlowScaleOutput with inverse value[N,2] and per-row logdet[N].
/// @throws std::invalid_argument For invalid inputs/store/scale or nonfinite stored/computed values.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9403);
/// auto x=n::ones({2,2});auto path=n::models::RealNvpDensity(p,x);
/// auto rec=n::models::RealNvpInverse(p,path.latent);
/// // rec.value approximates x; rec.logdet+path.logdet approximates zero per row.
/// @endcode
/// @par References
/// - Dinh, Sohl-Dickstein and Bengio(2016), Density estimation using Real NVP,
/// https://arxiv.org/abs/1605.08803 . Source compact bounded CPU adaptation.
/// @see RealNvpDensity(), RealNvpCoupling(), FlowScaleOutput
inline FlowScaleOutput RealNvpInverse(ParameterStore& parameters,const Tensor& latent) {
  flow_detail::Matrix(latent,2);
  if(latent.dtype()!=parameters.dtype() || latent.size(0)>std::numeric_limits<std::int64_t>::max()/16)
    throw std::invalid_argument("RealNVP inverse needs matching finite floating points and safe activations");
  auto zero=reshape(slice_axis(latent,1,0,1),{-1})*0.;
  auto out=FlowDiagonalScale(parameters,latent,zero,true);
  for(int i=3;i>=0;--i){auto block=RealNvpCoupling(parameters,out.value,i,true);
    out.value=block.value;out.logdet=out.logdet+block.logdet;}
  gan_detail::Finite(out.logdet);return out;
}
} // namespace datamunge::nn::models
