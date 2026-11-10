#pragma once
/// @file nice_density.hpp
/// @brief Live compact additive NICE couplings, exact inverse and global diagonal scale.
#include "datamunge/nn/normalizing_flows.hpp"

namespace datamunge::nn::models {
/// @brief Internal source dense conditioner construction without fitting or optimizer state.
namespace nice_detail {
/// @brief Apply one source affine layer with optional tanh, creating only missing leaves.
/// @param p Matching floating parameter store, synchronized by the caller.
/// @param name Leaf prefix followed by /W and /b.
/// @param x Live matching floating[N,in] features, already checked.
/// @param in Input width, one or16 in this compact model.
/// @param out Positive output width selected by the calling compact conditioner.
/// @param stddev Missing-weight normal deviation, .3 or .01.
/// @param bias Missing-bias constant, .05 or zero.
/// @param activate True for hidden tanh, false(default) for the final shift.
/// @return Owning live matching floating[N,out] conditioner features.
/// @throws std::invalid_argument For incompatible/nonfinite stored leaves or nonfinite result.
/// @throws std::logic_error For unsupported primitive execution.
inline Tensor Dense(ParameterStore& p,const std::string& name,const Tensor& x,
                    std::int64_t in,std::int64_t out,double stddev,double bias,bool activate=false) {
  const auto w=p.get(name+"/W",{in,out},init::normal(stddev));
  const auto b=p.get(name+"/b",{out},init::constant(bias));
  gan_detail::Finite(w);gan_detail::Finite(b);
  auto y=bias_add(matmul(x,w),b);
  if(activate)y=tanh(y);
  gan_detail::Finite(y);return y;
}
} // namespace nice_detail

/// @brief Apply one alternating two-coordinate additive NICE coupling or its exact inverse.
///
/// Finite Float32/64 CPU input[N,2] matching parameters; block0..3 conditions on
/// coordinate block%2. Prefix coupling<block>/hidden is1->16 tanh with missing
/// W~normal(.3), b=.05; /shift is16->1 linear with W~normal(.01), b=0. Preserve
/// the conditioning coordinate in raw axis order and add the shift to the other
/// coordinate; inverse=true subtracts it. Each coupling determinant is exactly
/// one analytically. One block owns49 stored values in four leaves. Missing
/// leaves initialize in hidden W/b then shift W/b order; existing leaves never
/// change. Input/options validate before creation; stored-leaf/math failures may
/// occur after partial creation. No fitting/optimizer/sampling/no-grad. Live
/// input/weight first/higher paths survive parent/store collection. Arbitrary
/// strides, O(16N) work/storage. Synchronize shared mutable stores; nonfinite
/// intermediate/output values are rejected without clipping.
/// @param parameters Caller-owned matching Float32/64 store; source full-model seed9301.
/// @param input Finite matching floating CPU[N,2], N>0, arbitrary strides.
/// @param block Source block index0..3; conditions on coordinate block%2.
/// @param inverse False(default) adds shift; true subtracts with the same parameters.
/// @return Owning live matching floating CPU[N,2] coupled coordinates.
/// @throws std::invalid_argument For invalid input/block/store or nonfinite stored/computed values.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9301);
/// auto x=n::from_vector(std::vector<float>{1,2},{1,2});
/// auto y=n::models::NiceCoupling(p,x,0);
/// auto rec=n::models::NiceCoupling(p,y,0,true);
/// // y[0,0]=1 exactly; rec approximates x within floating rounding.
/// @endcode
/// @par References
/// - Dinh, Krueger and Bengio(2014), NICE: Non-linear Independent Components
/// Estimation, https://arxiv.org/abs/1410.8516 . This source uses four small tanh conditioners.
/// @see NiceDensity(), NiceInverse()
inline Tensor NiceCoupling(ParameterStore& parameters,const Tensor& input,int block,bool inverse=false) {
  flow_detail::Matrix(input,2);
  if(block<0 || block>3 || input.dtype()!=parameters.dtype() ||
     input.size(0)>std::numeric_limits<std::int64_t>::max()/16)
    throw std::invalid_argument("NiceCoupling needs block0..3 and matching finite floating matrices");
  const std::string name="coupling"+std::to_string(block);
  const int condition=block%2;
  const auto a=slice_axis(input,1,condition,condition+1),b=slice_axis(input,1,1-condition,2-condition);
  const auto hidden=nice_detail::Dense(parameters,name+"/hidden",a,1,16,.3,.05,true);
  const auto shift=nice_detail::Dense(parameters,name+"/shift",hidden,16,1,.01,0);
  const auto changed=b+(inverse?-1.:1.)*shift;
  auto out=condition?concat({changed,a},1):concat({a,changed},1);
  gan_detail::Finite(out);return out;
}

/// @brief Own the compact NICE stages, pre-scale map, latent map and scalar log determinant.
///
/// Seven flat live Tensor handles survive parent/store collection and retain
/// applicable first/higher input/parameter paths. No training model, optimizer,
/// row order, sampler or RNG state is retained separately.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore p(9301);
/// auto out=dm::nn::models::NiceDensity(p,dm::nn::zeros({3,2}));
/// // Four coupling stages/unscaled/latent[3,2], scalar logdet=0 initially.
/// @endcode
/// @par References
/// - Dinh, Krueger and Bengio(2014), NICE, https://arxiv.org/abs/1410.8516 .
/// @see NiceDensity(), NiceInverse(), NormalFlowDensity()
struct NiceDensityOutput {
  Tensor coupling0; ///< Owning live Float32/64[N,2] stage conditioning on raw coordinate0.
  Tensor coupling1; ///< Owning live Float32/64[N,2] stage conditioning on raw coordinate1.
  Tensor coupling2; ///< Owning live Float32/64[N,2] stage conditioning on raw coordinate0.
  Tensor coupling3; ///< Owning live Float32/64[N,2] stage conditioning on raw coordinate1.
  Tensor unscaled;  ///< Live pre-scale[N,2] alias of coupling3 with no additional arithmetic.
  Tensor latent;    ///< Owning live Float32/64[N,2] unscaled*exp(log_scale).
  Tensor logdet;    ///< Owning live floating scalar sum(log_scale), constant across rows.
};

/// @brief Map two-dimensional observations through four additive NICE stages and a positive scale.
///
/// Matching finite Float32/64 CPU input[N,2], arbitrary strides, N>0. Apply blocks
///0..3 in order, then multiply by exp(log_scale[2]); missing log_scale is zero.
/// Exposes every stage, pre-scale map, latent and global logdet=sum(log_scale).
/// Four1->16 tanh->1 conditioners plus scale give198 values/17 leaves. Only
/// missing leaves initialize, in forward stage order followed by scale. No
/// existing-leaf writes, optimizer, fitting, sampling, implicit RNG after warmup
/// or no-grad. Owns live outputs/graphs; applicable first/higher paths survive
/// parent/store collection. O(64N) work/storage; synchronize shared stores.
/// Positive finite scales are required: underflow to zero, overflow or nonfinite
/// stored/computed fields reject. Supplied input validates before any creation;
/// stored-leaf/math failures can follow partial leaf creation. This compact
/// architecture uses a spatially constant determinant, not constant Jacobian entries.
/// @param parameters Caller-owned matching floating store; source initialization seed9301.
/// @param input Finite matching floating CPU[N,2] observations.
/// @return Owning live NiceDensityOutput with four stages, unscaled/latent and scalar logdet.
/// @throws std::invalid_argument For invalid inputs/store/scale or nonfinite stored/computed values.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9301);
/// auto out=n::models::NiceDensity(p,n::zeros({2,2}));
/// auto density=n::models::NormalFlowDensity(out.latent,out.logdet,true);
/// // p.count()==198, p.size()==17; density.loss is source-compatible mean NLL.
/// @endcode
/// @par References
/// - Dinh, Krueger and Bengio(2014), NICE, https://arxiv.org/abs/1410.8516 .
/// @see NiceDensityOutput, NiceCoupling(), NiceInverse(), NormalFlowDensity()
inline NiceDensityOutput NiceDensity(ParameterStore& parameters,const Tensor& input) {
  flow_detail::Matrix(input,2);
  if(input.dtype()!=parameters.dtype())throw std::invalid_argument("NICE input/store dtypes must match");
  NiceDensityOutput out;
  out.coupling0=NiceCoupling(parameters,input,0);
  out.coupling1=NiceCoupling(parameters,out.coupling0,1);
  out.coupling2=NiceCoupling(parameters,out.coupling1,2);
  out.coupling3=NiceCoupling(parameters,out.coupling2,3);
  out.unscaled=out.coupling3;
  const auto scale=parameters.get("log_scale",{2},init::zeros());gan_detail::Finite(scale);
  const auto positive=exp(scale);gan_detail::Finite(positive);
  for(double x:positive.contiguous().to_vector<double>())
    if(x<=0)throw std::invalid_argument("NICE scale must be strictly positive and finite");
  out.logdet=sum(scale,{0});out.latent=out.unscaled*positive;
  gan_detail::Finite(out.logdet);gan_detail::Finite(out.latent);return out;
}

/// @brief Own a live inverse map and its analytic log absolute determinant.
///
/// Two flat Tensor handles retain applicable input/parameter first/higher paths
/// and storage after parent/store collection. No optimizer, RNG or model state.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore p(9301);
/// dm::nn::models::NiceDensity(p,dm::nn::zeros({1,2}));
/// auto out=dm::nn::models::NiceInverse(p,dm::nn::zeros({4,2}));
/// // value[4,2] gives four inverse points; logdet is scalar zero initially.
/// @endcode
/// @par References
/// - Dinh, Krueger and Bengio(2014), NICE, https://arxiv.org/abs/1410.8516 .
/// @see NiceInverse(), NiceDensity()
struct FlowInverseOutput {
  Tensor value;  ///< Owning live matching floating CPU[N,2] inverse points.
  Tensor logdet; ///< Owning live matching floating scalar inverse log absolute determinant.
};

/// @brief Invert the compact NICE map by unscaling and reversing its four additive shifts.
///
/// Matching finite Float32/64 CPU latent[N,2], arbitrary strides, N>0. Multiply
/// by exp(-log_scale), then apply inverse blocks3,2,1,0. Return points and
/// -sum(log_scale), a scalar global inverse log determinant. A previously warmed
/// store reuses all198 values without mutation/RNG. Missing leaves may initialize
/// in inverse order (scale first, then reversed conditioners), so warm with
/// NiceDensity for source seed9301 ordering. No optimizer, fitting, sampling or
/// no-grad; input/parameter first/higher paths remain live. Owns results after
/// parent/store collection; O(64N) work/storage, synchronize shared stores.
/// Both forward/inverse scale must remain strictly positive and finite; finite
/// precision roundtrip errors remain. Input validates before leaf creation;
/// stored-leaf/math failures can follow partial creation.
/// @param parameters Caller-owned matching floating store, preferably forward-warmed.
/// @param latent Finite matching floating CPU[N,2] latent coordinates supplied by caller.
/// @return Owning live FlowInverseOutput with inverse value[N,2] and scalar logdet.
/// @throws std::invalid_argument For invalid inputs/store/scale or nonfinite stored/computed values.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9301);
/// auto x=n::zeros({2,2});auto z=n::models::NiceDensity(p,x);
/// auto rec=n::models::NiceInverse(p,z.latent);
/// // rec.value approximates x; rec.logdet+z.logdet is exactly zero.
/// @endcode
/// @par References
/// - Dinh, Krueger and Bengio(2014), NICE, https://arxiv.org/abs/1410.8516 .
/// @see NiceDensity(), FlowInverseOutput, NiceCoupling()
inline FlowInverseOutput NiceInverse(ParameterStore& parameters,const Tensor& latent) {
  flow_detail::Matrix(latent,2);
  if(latent.dtype()!=parameters.dtype() || latent.size(0)>std::numeric_limits<std::int64_t>::max()/16)
    throw std::invalid_argument("NICE inverse input/store dtypes and activation sizes must match");
  const auto scale=parameters.get("log_scale",{2},init::zeros());gan_detail::Finite(scale);
  const auto positive=exp(scale),negative=exp(-scale);
  gan_detail::Finite(positive);gan_detail::Finite(negative);
  for(auto v:{positive,negative})for(double x:v.contiguous().to_vector<double>())
    if(x<=0)throw std::invalid_argument("NICE forward/inverse scales must be positive and finite");
  auto h=latent*negative;
  for(int i=3;i>=0;--i)h=NiceCoupling(parameters,h,i,true);
  auto logdet=-sum(scale,{0});gan_detail::Finite(logdet);return {h,logdet};
}
} // namespace datamunge::nn::models
