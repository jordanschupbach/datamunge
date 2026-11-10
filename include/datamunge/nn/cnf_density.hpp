#pragma once
/// @file cnf_density.hpp
/// @brief Live paired-trace velocity fields and fixed-step differentiable RK4 density paths.
#include "datamunge/nn/affine_flows.hpp"
#include <array>

namespace datamunge::nn::models {
/// @brief Validation shared by the compact continuous-flow field and RK4 paths.
namespace cnf_detail {
/// @brief Require matching finite two-coordinate points and one of the four source trace modes.
/// @param parameters Floating caller-owned store matching input dtype.
/// @param input Finite floating CPU[N,2],positive N, arbitrary strides; safe for eight hidden units.
/// @param trace_mode Integer0 exact,1 plus probe,2 minus probe,3 deterministic pair mean.
/// @throws std::invalid_argument For invalid points, dtype, unsafe geometry or trace mode.
inline void Inputs(const ParameterStore& parameters,const Tensor& input,int trace_mode) {
  flow_detail::Matrix(input,2);
  if(input.dtype()!=parameters.dtype() || input.size(0)>std::numeric_limits<std::int64_t>::max()/8 || trace_mode<0 || trace_mode>3)
    throw std::invalid_argument("CNF needs matching floating points and trace mode0..3");
}
/// @brief Require a positive fixed RK4 step count with exactly four quarter checkpoints.
/// @param steps At least4 and divisible by4; default public source count8.
/// @throws std::invalid_argument For fewer than4 steps or non-quarter-divisible counts.
inline void Steps(int steps) {
  if(steps<4 || steps%4)throw std::invalid_argument("CNF RK4 steps must be at least4 and divisible by4");
}
} // namespace cnf_detail

/// @brief Own a live velocity field, selected trace and all exact/paired trace diagnostics.
///
/// Five flat Tensor handles retain storage and applicable first/higher input/
/// parameter graphs after parent/store collection. velocity is[N,2]; the four
/// trace fields are[N] in inverse-time units for the unit-time trajectory. Probe
/// vectors are fixed(1,1) and(1,-1), not RNG draws. Their mean equals the exact
/// 2-D field trace in exact arithmetic, with floating roundoff. No fitted model,
/// optimizer, sampling stream or trajectory state is retained separately.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(10003);
/// auto out=n::models::CnfField(p,n::zeros({2,2}),.5f);
/// // velocity[2,2], divergence/exact_trace/hutchinson_plus/hutchinson_minus[2].
/// @endcode
/// @par References
/// - Chen, Rubanova, Bettencourt and Duvenaud(2018), Neural Ordinary Differential
/// Equations, https://arxiv.org/abs/1806.07366 . This source differentiates
/// directly through fixed RK4 steps, not a constant-memory continuous adjoint.
/// - Grathwohl et al.(2018), FFJORD: Free-form Continuous Dynamics for Scalable
/// Reversible Generative Models, https://arxiv.org/abs/1810.01367 . This2-D
/// teaching adaptation uses a deterministic pair of probes, not a noisy single probe.
/// @see CnfField(), CnfIntegrate()
struct CnfFieldOutput {
  Tensor velocity;         ///< Owning live matching floating CPU[N,2] source-scaled velocity .3*MLP([input,time]).
  Tensor divergence;       ///< Owning live matching floating[N] selected trace contraction, true divergence for modes0/3 only.
  Tensor exact_trace;      ///< Owning live matching floating[N] analytic sum of the two diagonal field-Jacobian entries.
  Tensor hutchinson_plus;  ///< Owning live matching floating[N] fixed(1,1) probe contraction e^T J_field e.
  Tensor hutchinson_minus; ///< Owning live matching floating[N] fixed(1,-1) probe contraction e^T J_field e.
};

/// @brief Evaluate the source time-conditioned velocity field and exact or paired-probe divergence.
///
/// Finite matching Float32/64 CPU input[N,2],N>0, arbitrary strides, finite
/// Float32 scalar time. Time is a structural float argument, promoted after source
/// rounding even for Float64 tensors; no derivative with respect to that scalar.
/// Append time via input_col0*0+Double(time). Hidden3->8 tanh has W normal(.3),
/// b=.05; output8->2 linear has W normal(.1),b=0; velocity=.3*output. Prefixes
/// field/hidden and field/output. Create missing hidden W then output W BEFORE
/// biases, preserving source kernel-first initialization with seed10003.50 values
/// in4 leaves; no global scale here. Only missing leaves initialize; existing
/// leaves unchanged and warmed calls consume no RNG.
///
/// Let a,b be the two spatial rows of hidden W, c,d the output W columns, and
/// s=1-hidden^2. exact_trace=.3*sum(s*(a*c+b*d)); plus=.3*sum(s*(a+b)*(c+d));
/// minus=.3*sum(s*(a-b)*(c-d)). trace_mode0 selects exact,1 plus,2 minus,
/// 3(default) .5*(plus+minus), always returning all diagnostics. The deterministic
/// pair equals the exact2-D trace in exact arithmetic; a single probe generally
/// differs and is not the source fitting likelihood. These are field traces,
/// not finite RK4-map log determinants.
///
/// All applicable first/higher input/parameter paths remain live, including trace
/// partials. Owning flat fields survive collection. No autograd differentiation
/// inside the operation, fitting/optimizer/sampler/hidden no-grad or random probes.
/// O(8N) work/storage; synchronize shared stores. Validate input/time/mode before
/// leaves; stored-leaf/math failure can follow partial creation. Reject nonfinite
/// leaves, preactivations, velocity or traces without clipping. Extreme finite
/// inputs/parameters can reject; finite tanh saturation remains natural.
/// @param parameters Caller-owned matching floating store; full source seed10003.
/// @param input Finite matching floating CPU[N,2],positive N, arbitrary strides.
/// @param time Finite Float32 structural time, any value; source diagnostics use.5f.
/// @param trace_mode 0 exact,1 plus,2 minus,3(default) paired mean; only0..3 accepted.
/// @return Owning live CnfFieldOutput with velocity, selected divergence and all trace diagnostics.
/// @throws std::invalid_argument For invalid inputs/options/leaves or unrepresentable/nonfinite math.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(10003,n::DType::Float64);
/// auto x=n::ones({3,2},n::DType::Float64);
/// auto field=n::models::CnfField(p,x,.5f,3);
/// //50 values/4 leaves; .5*(plus+minus) approximates exact_trace to roundoff.
/// @endcode
/// @par References
/// - Chen, Rubanova, Bettencourt and Duvenaud(2018), Neural Ordinary Differential
/// Equations, https://arxiv.org/abs/1806.07366 . This source differentiates
/// directly through fixed RK4 steps, not a constant-memory continuous adjoint.
/// - Grathwohl et al.(2018), FFJORD: Free-form Continuous Dynamics for Scalable
/// Reversible Generative Models, https://arxiv.org/abs/1810.01367 . This2-D
/// teaching adaptation uses a deterministic pair of probes, not a noisy single probe.
/// @see CnfFieldOutput, CnfIntegrate(), CnfDensity()
inline CnfFieldOutput CnfField(ParameterStore& parameters,const Tensor& input,float time,int trace_mode=3) {
  cnf_detail::Inputs(parameters,input,trace_mode);
  if(!std::isfinite(time))throw std::invalid_argument("CNF field time must be finite Float32");
  const auto wi=parameters.get("field/hidden/W",{3,8},init::normal(.3));
  const auto wo=parameters.get("field/output/W",{8,2},init::normal(.1));
  const auto bi=parameters.get("field/hidden/b",{8},init::constant(.05));
  gan_detail::Finite(wi);gan_detail::Finite(wo);gan_detail::Finite(bi);
  const auto t=slice_axis(input,1,0,1)*0.+double(time),encoded=concat({input,t},1);
  const auto preactivation=bias_add(matmul(encoded,wi),bi);gan_detail::Finite(preactivation);
  const auto hidden=tanh(preactivation),bo=parameters.get("field/output/b",{2},init::zeros());gan_detail::Finite(bo);
  const auto output=bias_add(matmul(hidden,wo),bo);gan_detail::Finite(output);const auto velocity=.3*output;
  const auto a=slice(wi,{0,0},{1,8}),b=slice(wi,{1,0},{1,8});
  const auto c=transpose(slice(wo,{0,0},{8,1}),{1,0}),d=transpose(slice(wo,{0,1},{8,1}),{1,0}),slope=1.-square(hidden);
  const auto exact=.3*sum(slope*(a*c+b*d),{1}),plus=.3*sum(slope*((a+b)*(c+d)),{1}),minus=.3*sum(slope*((a-b)*(c-d)),{1});
  auto divergence=trace_mode==0?exact:trace_mode==1?plus:trace_mode==2?minus:.5*(plus+minus);
  for(const auto& value:{velocity,exact,plus,minus,divergence})gan_detail::Finite(value);
  return {velocity,divergence,exact,plus,minus};
}

/// @brief Own all four quarter-trajectory checkpoints and the final continuous-flow density path.
///
/// Nineteen flat Tensor handles own storage and applicable first/higher graphs
/// after parent/store collection. Each stage has its state, CUMULATIVE integrated
/// trace from the supplied starting point, endpoint divergence and endpoint
/// velocity. Source archive coupling_scale/shift fields mean divergence/velocity,
/// not affine scales or shifts. CnfIntegrate leaves latent and unscaled aliased
/// to its final state; CnfDensity adds global scaling only to latent/full logdet.
/// For modes0/3 integrated trace numerically approximates continuous-flow logdet,
/// not exact discrete RK4-map logdet. Modes1/2 integrate fixed probe contractions,
/// generally not full logdet approximations. No fit/optimizer/sampler state.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(10003);
/// auto out=n::models::CnfIntegrate(p,n::zeros({2,2}));
/// // stage0..3 and stage_velocity0..3[2,2]; cumulative stage_logdet/divergence[2].
/// @endcode
/// @par References
/// - Chen, Rubanova, Bettencourt and Duvenaud(2018), Neural Ordinary Differential
/// Equations, https://arxiv.org/abs/1806.07366 . This source differentiates
/// directly through fixed RK4 steps, not a constant-memory continuous adjoint.
/// - Grathwohl et al.(2018), FFJORD: Free-form Continuous Dynamics for Scalable
/// Reversible Generative Models, https://arxiv.org/abs/1810.01367 . This2-D
/// teaching adaptation uses a deterministic pair of probes, not a noisy single probe.
/// @see CnfIntegrate(), CnfDensity(), CnfInverse()
struct CnfDensityOutput {
  Tensor stage0; ///< Owning live matching floating[N,2] state after quarter1 of the supplied trajectory.
  Tensor stage_logdet0; ///< Owning live matching floating[N] CUMULATIVE integrated trace through quarter1, before global scale.
  Tensor stage_divergence0; ///< Owning live matching floating[N] selected field trace contraction at quarter1 endpoint.
  Tensor stage_velocity0; ///< Owning live matching floating[N,2] field velocity at quarter1 endpoint.
  Tensor stage1; ///< Owning live matching floating[N,2] state after quarter2 of the supplied trajectory.
  Tensor stage_logdet1; ///< Owning live matching floating[N] CUMULATIVE integrated trace through quarter2, before global scale.
  Tensor stage_divergence1; ///< Owning live matching floating[N] selected field trace contraction at quarter2 endpoint.
  Tensor stage_velocity1; ///< Owning live matching floating[N,2] field velocity at quarter2 endpoint.
  Tensor stage2; ///< Owning live matching floating[N,2] state after quarter3 of the supplied trajectory.
  Tensor stage_logdet2; ///< Owning live matching floating[N] CUMULATIVE integrated trace through quarter3, before global scale.
  Tensor stage_divergence2; ///< Owning live matching floating[N] selected field trace contraction at quarter3 endpoint.
  Tensor stage_velocity2; ///< Owning live matching floating[N,2] field velocity at quarter3 endpoint.
  Tensor stage3; ///< Owning live matching floating[N,2] state after quarter4 of the supplied trajectory.
  Tensor stage_logdet3; ///< Owning live matching floating[N] CUMULATIVE integrated trace through quarter4, before global scale.
  Tensor stage_divergence3; ///< Owning live matching floating[N] selected field trace contraction at quarter4 endpoint.
  Tensor stage_velocity3; ///< Owning live matching floating[N,2] field velocity at quarter4 endpoint.
  Tensor unscaled; ///< Live matching floating[N,2] alias of final stage3, before any global scaling.
  Tensor latent;   ///< Owning live matching floating[N,2] final state; globally scaled only in CnfDensity.
  Tensor logdet;   ///< Owning live matching floating[N] integrated trace, plus global scale only in CnfDensity.
};

/// @brief Integrate the source augmented velocity/selected-trace ODE through fixed differentiable RK4 steps.
///
/// Finite matching Float32/64 CPU input[N,2],N>0, arbitrary strides. steps>=4
/// and divisible by4(default8); trace_mode0..3(default3 paired mean). Integrate
/// state derivative f and accumulated selected field trace from t0=0
/// to1, or1 to0 if inverse=true. Initial cumulative logdet is input_col0*0.
/// Source arithmetic computes dt=(inverse?-1.f:1.f)/steps, half=.5f*dt,
/// sixth=dt/6.f and ALL stage times in Float32, then promotes coefficients/time
/// for tensor arithmetic, even in Float64. Preserve RK4 grouping(a+d)+2*(b+c)
/// for both velocity and divergence. Four field stages per step and one additional
/// endpoint field at each quarter. Expose four states, cumulative logdets and
/// endpoint fields, in chronological order along the supplied direction. Reverse
/// quarter endpoints run approximately.75,.5,.25,0; they are not forward checkpoints.
///
/// No global scale here:50 values/4 leaves, kernel-first source seed10003. Only
/// missing leaves initialize; warmed calls change no leaves or RNG. All applicable
/// first/higher state/trace/parameter paths remain live through the ENTIRE unrolled
/// RK4 graph. Ordinary reverse mode, no continuous adjoint, adaptive solver or
/// constant-memory claim. Owns flat fields after collection. No fitting, optimizer,
/// sampling stream or hidden no-grad. O(steps*8N) work and retained graph storage;
/// synchronize shared stores. Inputs/options validate before leaves; stored-leaf/
/// math failures may follow partial creation. Nonfinite fields/states reject.
///
/// The ODE flow is mathematically invertible under its smooth field, but fixed
/// finite RK4 reverse integration is a NUMERICAL inverse. Integrated divergence
/// approximates continuous-flow logdet for modes0/3; modes1/2 integrate fixed
/// probe contractions and generally do not approximate the full logdet. No mode
/// computes the exact discrete RK4-map logdet.
/// Roundtrip, actual-Jacobian and change-of-variables residuals remain. Float32
/// coefficients/times can limit refinement even with Float64 tensor arithmetic.
/// @param parameters Caller-owned matching floating store, source seed10003.
/// @param input Finite matching floating CPU[N,2] supplied initial state, positive rows.
/// @param inverse False(default) integrates0->1; true integrates1->0 with negative dt.
/// @param steps Fixed integer step count, default8, at least4 and divisible by4.
/// @param trace_mode 0 exact,1 plus,2 minus,3(default) deterministic paired field trace.
/// @return Owning live CnfDensityOutput with four quarter checkpoints and aliased final unscaled/latent state.
/// @throws std::invalid_argument For invalid inputs/options/leaves or unrepresentable/nonfinite math.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(10003,n::DType::Float64);
/// auto x=n::ones({2,2},n::DType::Float64);auto path=n::models::CnfIntegrate(p,x);
/// auto reverse=n::models::CnfIntegrate(p,path.latent,true);
/// // Reverse approximately recovers x; path.logdet is integrated trace, not exact RK4 logdet.
/// @endcode
/// @par References
/// - Chen, Rubanova, Bettencourt and Duvenaud(2018), Neural Ordinary Differential
/// Equations, https://arxiv.org/abs/1806.07366 . This source differentiates
/// directly through fixed RK4 steps, not a constant-memory continuous adjoint.
/// - Grathwohl et al.(2018), FFJORD: Free-form Continuous Dynamics for Scalable
/// Reversible Generative Models, https://arxiv.org/abs/1810.01367 . This2-D
/// teaching adaptation uses a deterministic pair of probes, not a noisy single probe.
/// @see CnfField(), CnfDensityOutput, CnfDensity(), CnfInverse()
inline CnfDensityOutput CnfIntegrate(ParameterStore& parameters,const Tensor& input,bool inverse=false,int steps=8,int trace_mode=3) {
  cnf_detail::Inputs(parameters,input,trace_mode);cnf_detail::Steps(steps);
  Tensor h=input,ld=reshape(slice_axis(input,1,0,1),{-1})*0.;
  const float dt=(inverse?-1.f:1.f)/steps,half=.5f*dt,sixth=dt/6.f;
  std::array<Tensor,4> states,determinants,divergences,velocities;int stage=0;
  for(int i=0;i<steps;++i){
    const float t=inverse?1.f+dt*i:dt*i;
    const auto a=CnfField(parameters,h,t,trace_mode);
    const auto b=CnfField(parameters,h+double(half)*a.velocity,t+half,trace_mode);
    const auto c=CnfField(parameters,h+double(half)*b.velocity,t+half,trace_mode);
    const auto d=CnfField(parameters,h+double(dt)*c.velocity,t+dt,trace_mode);
    h=h+double(sixth)*((a.velocity+d.velocity)+2.*(b.velocity+c.velocity));
    ld=ld+double(sixth)*((a.divergence+d.divergence)+2.*(b.divergence+c.divergence));
    gan_detail::Finite(h);gan_detail::Finite(ld);
    if((i+1)%(steps/4)==0){
      const auto endpoint=CnfField(parameters,h,t+dt,trace_mode);
      states[stage]=h;determinants[stage]=ld;divergences[stage]=endpoint.divergence;velocities[stage]=endpoint.velocity;++stage;
    }
  }
  return {states[0],determinants[0],divergences[0],velocities[0],
          states[1],determinants[1],divergences[1],velocities[1],
          states[2],determinants[2],divergences[2],velocities[2],
          states[3],determinants[3],divergences[3],velocities[3],
          h,h,ld};
}

/// @brief Map observations through the source numerical continuous flow and positive global scale.
///
/// Finite matching Float32/64 CPU input[N,2],N>0, arbitrary strides. CnfIntegrate
/// forward uses steps(default8,>=4,divisible4) and trace_mode(default3,0..3), then
/// FlowDiagonalScale. Preserve source Float32 RK4 coefficients/times, ordinary
/// reverse-mode differentiation through every step and all quarter diagnostics.
/// 52 values/5 leaves: field hidden3->8 tanh->2 output,50 values/4 leaves, then
/// log_scale[2]. Source seed10003 creates hidden W then output W before biases,
/// and global scale last. Only missing leaves initialize; warmed inference changes
/// no leaves/RNG. Quarter logdets remain CUMULATIVE field integrals before scale.
/// This is one shared field over time, not four independently parameterized maps;
/// the original generic metadata hidden16 does not describe its actual width8.
///
/// All applicable first/higher input/parameter paths stay live through the unrolled
/// solver; flat owning fields survive collection. No fitting, optimizer, sampling
/// stream, adaptive solve, continuous adjoint or hidden no-grad. O(steps*8N) work/
/// retained graph storage; synchronize stores. Inputs/options validate before leaves;
/// stored-leaf/math failures can follow partial creation. Reject nonfinite results
/// or unsafe global scale. Modes0/3 numerically approximate the flow likelihood;
/// modes1/2 are fixed-probe diagnostic integrals, generally not a density. No
/// mode computes exact discrete RK4-map logdet; finite numerical residuals remain.
/// @param parameters Caller-owned matching floating store; source seed10003.
/// @param input Finite matching floating CPU[N,2] observed coordinates, positive rows.
/// @param steps Fixed RK4 count, default8, at least4 and divisible by4.
/// @param trace_mode 0 exact,1 plus,2 minus,3(default) deterministic paired field trace.
/// @return Owning live CnfDensityOutput with19 stage/density fields; latent and full logdet include global scale.
/// @throws std::invalid_argument For invalid inputs/options/leaves or unrepresentable/nonfinite math.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(10003);
/// auto path=n::models::CnfDensity(p,n::zeros({3,2}));
/// auto density=n::models::NormalFlowDensity(path.latent,path.logdet,true);
/// //52 values/5 leaves; numerical source-compatible mean NLL, nats/row.
/// @endcode
/// @par References
/// - Chen, Rubanova, Bettencourt and Duvenaud(2018), Neural Ordinary Differential
/// Equations, https://arxiv.org/abs/1806.07366 . This source differentiates
/// directly through fixed RK4 steps, not a constant-memory continuous adjoint.
/// - Grathwohl et al.(2018), FFJORD: Free-form Continuous Dynamics for Scalable
/// Reversible Generative Models, https://arxiv.org/abs/1810.01367 . This2-D
/// teaching adaptation uses a deterministic pair of probes, not a noisy single probe.
/// @see CnfDensityOutput, CnfIntegrate(), CnfInverse(), NormalFlowDensity()
inline CnfDensityOutput CnfDensity(ParameterStore& parameters,const Tensor& input,int steps=8,int trace_mode=3) {
  auto out=CnfIntegrate(parameters,input,false,steps,trace_mode);
  const auto scaled=FlowDiagonalScale(parameters,out.latent,out.logdet);out.latent=scaled.value;out.logdet=scaled.logdet;return out;
}

/// @brief Numerically reverse the compact continuous-flow density and accumulate its own inverse-path trace.
///
/// Finite matching Float32/64 CPU latent[N,2],N>0, arbitrary strides. Validate
/// steps(default8,>=4,divisible4)/trace_mode(default3,0..3) before leaves. Unscale
/// supplied latent by exp(-log_scale), integrate1->0 with CnfIntegrate and subtract
/// sum(log_scale) from its reverse accumulated trace. Recompute every RK4 field
/// on that supplied inverse path; no cached forward points. Source Float32 times/
/// coefficients apply even for Float64 tensors. Owns live value/logdet after
/// collection; applicable first/higher input/parameter paths differentiate through
/// the full unrolled reverse solver. This is a numerical reverse ODE path, not an
/// exact inverse of the finite forward RK4 map. Its integrated trace likewise
/// approximates continuous inverse logdet for modes0/3 only; modes1/2 integrate
/// fixed-probe diagnostic contractions, generally not full inverse logdet.
/// Numerical residuals remain.
///
/// A forward-warmed store reuses52 values/5 leaves without writes or RNG; missing
/// inverse leaves initialize global scale before field kernels/biases, so warm
/// CnfDensity first with source seed10003 for source ordering. No fitting, optimizer,
/// sampling stream, adaptive solve, continuous adjoint, hidden no-grad or inverse
/// refit. O(steps*8N) work/retained graph storage; synchronize stores. Input/options
/// validate before leaves; stored-leaf/math failures may follow partial creation.
/// Reject nonfinite fields/results or unsafe global scale without clipping.
/// @param parameters Caller-owned matching floating store, preferably forward-warmed with source seed10003.
/// @param latent Finite matching floating CPU[N,2] explicit Gaussian coordinates, positive rows.
/// @param steps Fixed RK4 count, default8, at least4 and divisible by4.
/// @param trace_mode 0 exact,1 plus,2 minus,3(default) deterministic paired field trace.
/// @return Owning live FlowScaleOutput with observation value[N,2] and reverse-path integrated logdet[N].
/// @throws std::invalid_argument For invalid inputs/options/leaves or unrepresentable/nonfinite math.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(10003);
/// auto x=n::ones({2,2});auto path=n::models::CnfDensity(p,x);
/// auto rec=n::models::CnfInverse(p,path.latent);
/// // rec.value approximately recovers x; signed integrated traces approximately cancel.
/// @endcode
/// @par References
/// - Chen, Rubanova, Bettencourt and Duvenaud(2018), Neural Ordinary Differential
/// Equations, https://arxiv.org/abs/1806.07366 . This source differentiates
/// directly through fixed RK4 steps, not a constant-memory continuous adjoint.
/// - Grathwohl et al.(2018), FFJORD: Free-form Continuous Dynamics for Scalable
/// Reversible Generative Models, https://arxiv.org/abs/1810.01367 . This2-D
/// teaching adaptation uses a deterministic pair of probes, not a noisy single probe.
/// @see CnfDensity(), CnfIntegrate(), CnfField(), FlowScaleOutput
inline FlowScaleOutput CnfInverse(ParameterStore& parameters,const Tensor& latent,int steps=8,int trace_mode=3) {
  cnf_detail::Inputs(parameters,latent,trace_mode);cnf_detail::Steps(steps);
  const auto zero=reshape(slice_axis(latent,1,0,1),{-1})*0.;const auto unscaled=FlowDiagonalScale(parameters,latent,zero,true);
  const auto path=CnfIntegrate(parameters,unscaled.value,true,steps,trace_mode);
  auto ld=path.logdet-sum(parameters.at("log_scale"),{0});gan_detail::Finite(ld);return {path.latent,ld};
}
} // namespace datamunge::nn::models
