#pragma once
/// @file diffusion.hpp
/// @brief Explicit cosine diffusion coefficients, forward noising and live compact DDPM operations.
#include "datamunge/nn/gan_models.hpp"
#include "datamunge/nn/autograd.hpp"
#include <array>

namespace datamunge::nn::models {
/// @brief Internal validation for explicit diffusion states, coefficients and structural indices.
namespace diffusion_detail {
/// @brief Require finite floating CPU storage with positive axes and at least the requested rank.
/// @param x Nonempty Float32/64 CPU tensor with arbitrary strides.
/// @param rank Minimum number of axes, default two for batched samples.
/// @throws std::invalid_argument For invalid storage, dtype, geometry or nonfinite values.
inline void Floating(const Tensor& x,int rank=2) {
  gan_detail::Finite(x);
  if(x.device().backend!=compute::Backend::CPU || x.dim()<rank)
    throw std::invalid_argument("Diffusion requires finite floating CPU samples");
}
/// @brief Require an explicit nonnegative integer timestep vector, optionally bounded and sized.
/// @param time Int32/64 CPU[N], N>0, arbitrary strides.
/// @param rows Required N, or zero(default) for any positive count.
/// @param limit Exclusive upper bound, or zero(default) for no upper bound.
/// @throws std::invalid_argument For invalid storage/type/count or an out-of-range index.
inline void Time(const Tensor& time,std::int64_t rows=0,std::int64_t limit=0) {
  if(!time.defined() || time.device().backend!=compute::Backend::CPU || time.dim()!=1 ||
     time.size(0)<=0 || (rows && time.size(0)!=rows) ||
     (time.dtype()!=DType::Int32 && time.dtype()!=DType::Int64))
    throw std::invalid_argument("Diffusion timesteps require an Int32/64 CPU vector matching N");
  for(auto t:cast(time,DType::Int64).to_vector<std::int64_t>())
    if(t<0 || (limit && t>=limit))throw std::invalid_argument("Diffusion timestep out of range");
}
/// @brief Require matching finite floating sample geometry without broadcasting input samples.
/// @param a First batched floating CPU tensor of rank at least two.
/// @param b Second floating CPU tensor of exactly matching shape and dtype.
/// @throws std::invalid_argument For mismatched or invalid inputs.
inline void Same(const Tensor& a,const Tensor& b) {
  Floating(a);Floating(b);
  if(a.shape()!=b.shape() || a.dtype()!=b.dtype())
    throw std::invalid_argument("Diffusion sample tensors must have identical shapes and dtypes");
}
/// @brief Reject negative coefficients, and optionally reject zero as well.
/// @param x Finite floating coefficient tensor already validated by extraction.
/// @param strict True requires positive values; false(default) permits zero.
/// @throws std::invalid_argument For a negative coefficient, or zero when strict.
inline void Positive(const Tensor& x,bool strict=false) {
  for(double value:x.to_vector<double>())
    if(strict?value<=0:value<0)throw std::invalid_argument("Invalid diffusion coefficient sign");
}
} // namespace diffusion_detail

/// @brief Own all nine detached Double tables of a cosine diffusion schedule.
///
/// Flat owning Float64 CPU[T] fields survive parent collection, with no RNG,
/// model, static cache or optimizer retained. Field names follow the source
/// coefficient archive. Callers may construct explicit tables; consuming
/// operations validate only the fields they use and do not prove consistency
/// of an arbitrary user-supplied schedule.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto s=dm::nn::models::CosineDiffusionSchedule();
/// // Nine independent owning Double[32] fields; posterior_variance[0]=0.
/// @endcode
/// @see CosineDiffusionSchedule(), DiffusionNoised(), DdpmReverse()
struct DiffusionScheduleOutput {
  Tensor beta; ///< Detached Float64[T] forward conditional variances in (0,1).
  Tensor alpha; ///< Detached Float64[T] one minus beta.
  Tensor alpha_bar; ///< Detached Float64[T] serial cumulative products of alpha.
  Tensor alpha_bar_previous; ///< Detached Float64[T] previous products, beginning with one.
  Tensor sqrt_alpha_bar; ///< Detached Float64[T] positive signal coefficients.
  Tensor sqrt_one_minus_alpha_bar; ///< Detached Float64[T] nonnegative forward-noise coefficients.
  Tensor posterior_variance; ///< Detached Float64[T] fixed posterior variances; first is exactly zero.
  Tensor posterior_coefficient1; ///< Detached Float64[T] coefficients multiplying predicted clean samples.
  Tensor posterior_coefficient2; ///< Detached Float64[T] coefficients multiplying current noisy states.
};

/// @brief Construct the original cosine alpha-bar schedule and fixed-posterior coefficients.
///
/// f(t)=cos^2((t+offset)/(1+offset)*pi/2). For i=0..T-1 use
/// beta_i=min(beta_cap,1-f((i+1)/T)/f(i/T)), alpha_i=1-beta_i, then
/// bar_i=bar_previous*alpha_i with bar_previous initially one. Posterior
/// variance=beta*(1-bar_previous)/(1-bar); coefficients are
/// beta*sqrt(bar_previous)/(1-bar) and (1-bar_previous)*sqrt(alpha)/(1-bar).
/// Scalar Double evaluation and serial products preserve the source's T=32,
/// offset=.008, cap=.999 protocol. No fitted/learned variance, RNG or hidden
/// timestep sampler. O(T) work/storage, no shared mutable state; each result
/// owns its tables. Unsafe/degenerate floating calculations are rejected.
/// @param timesteps Number of transitions T>=2, default32.
/// @param offset Finite nonnegative cosine offset, default .008.
/// @param beta_cap Finite strict (0,1) upper beta cap, default .999.
/// @return Owning detached DiffusionScheduleOutput with nine Float64 CPU[T] tables.
/// @throws std::invalid_argument For invalid options or degenerate/nonfinite coefficients.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto s=dm::nn::models::CosineDiffusionSchedule();
/// // alpha_bar[31] approximately2.369970271e-6, posterior_variance[0]=0.
/// @endcode
/// @par References
/// - Nichol and Dhariwal(2021), Improved Denoising Diffusion Probabilistic Models,
/// https://arxiv.org/abs/2102.09672 . This compact source uses32 levels and fixed
/// posterior variances, not the paper's learned-variance training objective.
/// @see DiffusionScheduleOutput, DiffusionExtract(), DdpmReverse()
inline DiffusionScheduleOutput CosineDiffusionSchedule(
    std::int64_t timesteps=32,double offset=.008,double beta_cap=.999) {
  if(timesteps<2 || std::uint64_t(timesteps)>std::vector<double>().max_size() || !std::isfinite(offset) || offset<0 || !std::isfinite(beta_cap) || beta_cap<=0 || beta_cap>=1)
    throw std::invalid_argument("Invalid cosine diffusion schedule options");
  std::array<std::vector<double>,9> tables;
  /// @brief Evaluate the cosine-square interpolation before schedule normalization.
  /// @param t Continuous normalized level in[0,1].
  /// @return Double squared cosine at the captured offset.
  const auto f=[offset](double t){return std::pow(std::cos((t+offset)/(1+offset)*std::acos(-1.)/2),2);};
  double bar=1;
  for(std::int64_t i=0;i<timesteps;++i) {
    const double beta=std::min(beta_cap,1-f(double(i+1)/double(timesteps))/f(double(i)/double(timesteps)));
    const double alpha=1-beta,previous=bar;bar*=alpha;
    if(!std::isfinite(beta) || beta<=0 || beta>=1 || !std::isfinite(bar) || bar<=0 || bar>=1)
      throw std::invalid_argument("Degenerate cosine diffusion schedule");
    std::array<double,9> row{beta,alpha,bar,previous,std::sqrt(bar),std::sqrt(1-bar),
      beta*(1-previous)/(1-bar),beta*std::sqrt(previous)/(1-bar),(1-previous)*std::sqrt(alpha)/(1-bar)};
    for(int k=0;k<9;++k) {
      if(!std::isfinite(row[k]))throw std::invalid_argument("Nonfinite cosine diffusion coefficient");
      tables[k].push_back(row[k]);
    }
  }
  return {from_vector(tables[0],{timesteps}),from_vector(tables[1],{timesteps}),from_vector(tables[2],{timesteps}),
    from_vector(tables[3],{timesteps}),from_vector(tables[4],{timesteps}),from_vector(tables[5],{timesteps}),
    from_vector(tables[6],{timesteps}),from_vector(tables[7],{timesteps}),from_vector(tables[8],{timesteps})};
}

/// @brief Select explicit timestep coefficients and reshape them for batched sample broadcasting.
///
/// Finite Float32/64 CPU coefficients[T], T>0, and Int32/64 CPU time[N], N>0.
/// Select coefficients[time] then cast to dtype(default Float32) and reshape to
/// [N,1,...,1] with sample_rank axes (default4). Arbitrary strides accepted.
/// Dtype conversion stays live for applicable coefficient first/higher paths;
/// integer indices are structural. No mutation, RNG, sampling or fitting.
/// Owns storage/graph after input collection. O(N+T) validation and O(N) result.
/// Final cast overflow is rejected. Passing dtype Float64 does not first round
/// to Float32; DiffusionNoised/DdpmReverse explicitly do that source rounding.
/// @param coefficients Finite floating CPU[T] explicit coefficient table.
/// @param time Integer CPU[N] indices in [0,T).
/// @param sample_rank Positive result rank, default4; rank1 returns[N].
/// @param dtype Float32(default) or Float64 result dtype.
/// @return Owning live matching floating CPU[N,1,...,1] selected coefficients.
/// @throws std::invalid_argument For invalid geometry/options/type/indices or cast overflow.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;auto s=n::models::CosineDiffusionSchedule();
/// auto t=n::from_vector(std::vector<std::int32_t>{0,31},{2});
/// auto v=n::models::DiffusionExtract(s.posterior_variance,t);
/// // v has Float32 shape[2,1,1,1]; first entry is zero.
/// @endcode
/// @par References
/// - Ho, Jain and Abbeel(2020), Denoising Diffusion Probabilistic Models,
/// https://arxiv.org/abs/2006.11239 ; timestep-indexed scalar coefficients.
/// @see CosineDiffusionSchedule(), DiffusionNoised(), DdpmReverse()
inline Tensor DiffusionExtract(const Tensor& coefficients,const Tensor& time,
    int sample_rank=4,DType dtype=DType::Float32) {
  diffusion_detail::Floating(coefficients,1);
  if(coefficients.dim()!=1 || sample_rank<1 || (dtype!=DType::Float32 && dtype!=DType::Float64))
    throw std::invalid_argument("Invalid diffusion table, result rank or dtype");
  diffusion_detail::Time(time,0,coefficients.size(0));
  Shape shape(std::size_t(sample_rank),1);shape[0]=time.size(0);
  auto out=reshape(cast(gather(coefficients,cast(time,DType::Int64),0),dtype),shape);
  gan_detail::Finite(out);return out;
}

/// @brief Encode structural integer times as the source's eight rounded sine/cosine features.
///
/// Int32/64 CPU time[N],N>0, nonnegative indices and arbitrary strides.
/// Normalize in Float32 by positive finite Float32 maximum_time(default31),
/// multiply by Float32(pi*[1,2,4,8]), concatenate four sines then four cosines,
/// and finally cast to dtype(default Float32). Source Float32 phase/feature
/// rounding is retained even for Double output. This is structural integer-time
/// conditioning with no derivative with respect to timestep values. No RNG,
/// fitting or mutation; owning detached[N,8] survives collection. O(N) work and
/// storage. Nonfinite normalized phases or outputs are rejected.
/// @param time Nonnegative structural Int32/64 CPU[N] timestep IDs.
/// @param maximum_time Positive finite Float32 normalizer, default31.f.
/// @param dtype Float32(default) or Float64 final feature dtype.
/// @return Owning detached floating CPU[N,8] features, sine block then cosine block.
/// @throws std::invalid_argument For invalid inputs/options or unsafe intermediate math.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto t=dm::nn::from_vector(std::vector<std::int32_t>{0},{1});
/// auto h=dm::nn::models::DiffusionTimeEmbedding(t);
/// // h is [[0,0,0,0,1,1,1,1]].
/// @endcode
/// @par References
/// - Original compact DDPM teaching protocol's fixed four-frequency sin/cos time
/// features; the source rounds normalization, frequencies and features to Float32.
/// @see DdpmEpsilon(), DiffusionExtract()
inline Tensor DiffusionTimeEmbedding(const Tensor& time,float maximum_time=31.f,DType dtype=DType::Float32) {
  diffusion_detail::Time(time);
  if(!std::isfinite(maximum_time) || maximum_time<=0 || (dtype!=DType::Float32 && dtype!=DType::Float64))
    throw std::invalid_argument("Invalid diffusion time embedding options");
  NoGradGuard guard;
  auto normalized=reshape(cast(time,DType::Float32),{-1,1})/double(maximum_time);
  const double pi=std::acos(-1.);
  auto phase=normalized*from_vector(std::vector<float>{float(pi),float(2*pi),float(4*pi),float(8*pi)},{1,4});
  gan_detail::Finite(phase);auto out=cast(concat({sin(phase),cos(phase)},1),dtype);gan_detail::Finite(out);return out;
}

/// @brief Form a live forward-diffused batch from explicit clean samples, noise and signal tables.
///
/// Matching finite Float32/64 CPU clean/noise[N,...], positive axes, rank>=2,
/// and Int32/64 CPU time[N]. Select schedule.sqrt_alpha_bar and
/// sqrt_one_minus_alpha_bar as Float32 first, then promote to sample dtype,
/// even for Double samples. Return signal*clean+sigma*noise, with source binary
/// grouping. Consumed tables must have matching positive lengths; selected
/// signal is positive and selected sigma nonnegative. Does not prove that
/// arbitrary user tables define a consistent Markov schedule. All applicable
/// clean/noise/table first/higher paths remain live; no hidden no-grad, sampler,
/// fit, RNG or mutation. Owning graph/storage survives collection; arbitrary
/// strides accepted. O(input elements+table length) work/transient storage.
/// @param clean Finite floating CPU[N,...] explicit clean samples, rank>=2.
/// @param noise Matching explicit standard-normal noise; no distribution test is performed.
/// @param time Int32/64 CPU[N] timestep IDs within the consumed tables.
/// @param schedule Explicit schedule; only its two signal/noise tables are consumed.
/// @return Owning live matching floating CPU[N,...] forward-diffused samples.
/// @throws std::invalid_argument For invalid inputs/tables or nonfinite intermediate/results.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;auto s=n::models::CosineDiffusionSchedule();
/// auto t=n::from_vector(std::vector<std::int32_t>{0},{1});
/// auto x=n::models::DiffusionNoised(n::ones({1,2}),n::zeros({1,2}),t,s);
/// // Both values are the first source-rounded signal coefficient, about 0.998208.
/// @endcode
/// @par References
/// - Ho, Jain and Abbeel(2020), Denoising Diffusion Probabilistic Models,
/// https://arxiv.org/abs/2006.11239 , closed-form forward conditional.
/// @see CosineDiffusionSchedule(), DiffusionExtract(), DdpmReverse()
inline Tensor DiffusionNoised(const Tensor& clean,const Tensor& noise,const Tensor& time,
    const DiffusionScheduleOutput& schedule) {
  diffusion_detail::Same(clean,noise);diffusion_detail::Time(time,clean.size(0));
  auto signal=cast(DiffusionExtract(schedule.sqrt_alpha_bar,time,int(clean.dim())),clean.dtype());
  auto sigma=cast(DiffusionExtract(schedule.sqrt_one_minus_alpha_bar,time,int(clean.dim())),clean.dtype());
  if(schedule.sqrt_alpha_bar.shape()!=schedule.sqrt_one_minus_alpha_bar.shape())
    throw std::invalid_argument("Diffusion signal/noise table lengths differ");
  diffusion_detail::Positive(signal,true);diffusion_detail::Positive(sigma);
  auto a=signal*clean,b=sigma*noise;gan_detail::Finite(a);gan_detail::Finite(b);
  auto out=a+b;gan_detail::Finite(out);return out;
}

/// @brief Own the compact DDPM epsilon predictor and its live hidden endpoints.
///
/// Four flat owning matching floating CPU tensors: epsilon[N,8,8,1],
/// time_embedding[N,8], hidden1/hidden2[N,32]. Epsilon and hidden fields retain
/// applicable state/parameter first/higher graphs after parent/store collection;
/// the time features are structural detached constants. No fitted model,
/// optimizer, RNG or sampling trajectory is retained separately.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(11003);
/// auto t=n::from_vector(std::vector<std::int32_t>{0},{1});
/// auto out=n::models::DdpmEpsilon(p,n::zeros({1,8,8,1}),t);
/// // p has5504 values/six leaves; out.hidden1 has shape[1,32].
/// @endcode
/// @see DdpmEpsilon(), DiffusionTimeEmbedding()
struct DdpmEpsilonOutput {
  Tensor epsilon; ///< Owning live matching floating CPU[N,8,8,1] predicted standard noise.
  Tensor time_embedding; ///< Owning detached matching floating CPU[N,8] source-rounded structural features.
  Tensor hidden1; ///< Owning live matching floating CPU[N,32] first x*sigmoid(x) hidden endpoint.
  Tensor hidden2; ///< Owning live matching floating CPU[N,32] second x*sigmoid(x) hidden endpoint.
};

/// @brief Evaluate the original 5504-value, two-SiLU-layer MNIST epsilon predictor without fitting.
///
/// Matching finite Float32/64 CPU state[N,8,8,1],N>0, caller-owned floating
/// parameter store and Int32/64 CPU time[N] in0..31. Flatten64 pixels and append
/// the eight source-rounded time features; use72->32 SiLU->32 SiLU->64 with
/// SiLU spelled x*sigmoid(x). Missing hidden1/W,b, hidden2/W,b, output/W,b are
/// created in that order BEFORE evaluation, normal(.1) kernels and zero biases.
/// Source seed11003 yields5504 values/six leaves. Existing leaves are not reset;
/// warmed calls draw no RNG and do not mutate the store. Names have no added
/// prefix; use separate stores for independent predictors. Inputs validate before
/// leaf creation; stored
/// leaf/math failures may follow partial missing-leaf creation.
///
/// First/higher state and parameter graphs remain live. Flat fields own their
/// graphs/storage after parent/store collection. Arbitrary strides accepted.
/// No timestep/noise sampling, fitting loop, optimizer or hidden no-grad. Work
/// and transient storage O(N*5504); synchronize shared mutable stores. The compact
/// MLP is not the paper's image U-Net. Nonfinite leaves/preactivations/results
/// are rejected; finite sigmoid saturation follows the primitive naturally.
/// @param parameters Caller-owned matching floating store; source seed11003.
/// @param state Finite matching floating CPU[N,8,8,1] explicit noisy states.
/// @param time Structural Int32/64 CPU[N] timestep IDs in[0,32).
/// @return Owning live DdpmEpsilonOutput with epsilon, time features and two hidden endpoints.
/// @throws std::invalid_argument For invalid inputs/leaves or unsafe computed math.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(11003);
/// auto t=n::from_vector(std::vector<std::int32_t>{0,31},{2});
/// auto out=n::models::DdpmEpsilon(p,n::zeros({2,8,8,1}),t);
/// // out.epsilon is[2,8,8,1]; caller computes MSE and owns optimizer.step().
/// @endcode
/// @par References
/// - Ho, Jain and Abbeel(2020), Denoising Diffusion Probabilistic Models,
/// https://arxiv.org/abs/2006.11239 . The source uses a compact two-SiLU MLP,
/// fixed posterior variance and32 cosine steps rather than the full image model.
/// @see DdpmEpsilonOutput, DiffusionNoised(), DdpmReverse()
inline DdpmEpsilonOutput DdpmEpsilon(ParameterStore& parameters,const Tensor& state,const Tensor& time) {
  diffusion_detail::Floating(state);diffusion_detail::Time(time,state.size(0),32);
  if(state.shape()!=Shape{state.size(0),8,8,1} || state.dtype()!=parameters.dtype())
    throw std::invalid_argument("DDPM epsilon needs matching floating CPU[N,8,8,1]");
  for(const std::string name:{"hidden1","hidden2","output"}) {
    const std::int64_t in=name=="hidden1"?72:32,out=name=="output"?64:32;
    parameters.get(name+"/W",{in,out},init::normal(.1));parameters.get(name+"/b",{out},init::zeros());
  }
  auto embedding=DiffusionTimeEmbedding(time,31.f,state.dtype());
  auto encoded=concat({reshape(state,{-1,64}),embedding},1);
  std::array<Tensor,3> endpoints;int k=0;auto h=encoded;
  for(const std::string name:{"hidden1","hidden2","output"}) {
    auto w=parameters.at(name+"/W"),b=parameters.at(name+"/b");gan_detail::Finite(w);gan_detail::Finite(b);
    auto z=bias_add(matmul(h,w),b);gan_detail::Finite(z);
    h=name=="output"?z:z*sigmoid(z);gan_detail::Finite(h);endpoints[k++]=h;
  }
  return {reshape(endpoints[2],state.shape()),embedding,endpoints[0],endpoints[1]};
}

/// @brief Own one DDPM reverse transition and its live posterior diagnostics.
///
/// Flat fields value/mean/x0_raw/x0_clipped match the sample shape[N,...];
/// variance is broadcastable[N,1,...,1]. Applicable state, epsilon, innovation
/// and consumed coefficient graphs/storage survive parent collection. No model,
/// optimizer or complete sampling loop is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;auto s=n::models::CosineDiffusionSchedule();
/// auto t=n::from_vector(std::vector<std::int32_t>{0},{1});
/// auto o=n::models::DdpmReverse(n::zeros({1,2}),t,n::zeros({1,2}),n::ones({1,2}),s);
/// // First posterior variance is zero; o.value equals o.mean despite innovation.
/// @endcode
/// @see DdpmReverse(), DdpmEpsilon()
struct DdpmReverseOutput {
  Tensor value; ///< Owning live matching floating CPU[N,...] mean+sqrt(variance)*innovation.
  Tensor mean; ///< Owning live matching floating CPU[N,...] fixed-posterior mean.
  Tensor variance; ///< Owning live matching floating CPU[N,1,...,1] selected posterior variance.
  Tensor x0_raw; ///< Owning live matching floating CPU[N,...] inferred clean sample before clipping.
  Tensor x0_clipped; ///< Owning live matching floating CPU[N,...] inferred clean sample clipped to[-1,1].
};

/// @brief Apply one fixed-variance DDPM posterior step to explicit state, predicted noise and innovation.
///
/// Finite matching Float32/64 CPU state/epsilon/innovation[N,...], positive
/// axes/rank>=2, and Int32/64 CPU time[N]. Extract signal,sigma,posterior c1/c2
/// and variance as Float32 first, then promote even for Double samples. Compute
/// raw=(state-sigma*epsilon)/signal, clipped=min(max(raw,-1),1),
/// mean=c1*clipped+c2*state, value=mean+sqrt(variance)*innovation. Consumed
/// tables must be finite, one-dimensional and equally long; selected signal
/// positive, sigma/variance nonnegative. Caller owns their consistency. With
/// the generated schedule, t=0 has zero variance: innovation has no effect.
/// Innovation must still be finite and correctly shaped at that last step.
///
/// All applicable first/higher paths remain live; clipping uses core min/max
/// tie subgradients and is nonsmooth at the bounds; sqrt(variance) is also
/// nonsmooth at zero for trainable variance tables. No gradients with respect
/// to integer time, hidden no-grad, RNG, full trajectory, rejection, fit or
/// optimizer. Fields own graph/storage after collection; arbitrary strides.
/// O(sample elements+table length) work/transient storage; no mutable globals.
/// Nonfinite intermediate raw values are rejected before clipping.
/// @param state Finite floating CPU[N,...] explicit current noisy state.
/// @param time Structural Int32/64 CPU[N] indices within all consumed tables.
/// @param epsilon Matching predicted standard noise, potentially carrying model graphs.
/// @param innovation Matching explicit standard-normal innovations; no distribution test.
/// @param schedule Explicit schedule; signal,sigma,c1,c2,variance fields are consumed.
/// @return Owning live DdpmReverseOutput with value,mean,variance and raw/clipped clean estimates.
/// @throws std::invalid_argument For invalid inputs/tables/signs or unsafe computed math.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;auto s=n::models::CosineDiffusionSchedule();
/// auto t=n::from_vector(std::vector<std::int32_t>{31},{1});
/// auto o=n::models::DdpmReverse(n::ones({1,2}),t,n::zeros({1,2}),n::zeros({1,2}),s);
/// // x0_raw exceeds one, x0_clipped is one; caller owns all32 reverse steps.
/// @endcode
/// @par References
/// - Ho, Jain and Abbeel(2020), Denoising Diffusion Probabilistic Models,
/// https://arxiv.org/abs/2006.11239 , epsilon parameterization/fixed-posterior sampling.
/// - Nichol and Dhariwal(2021), Improved Denoising Diffusion Probabilistic Models,
/// https://arxiv.org/abs/2102.09672 , cosine schedule; no learned variance here.
/// @see DdpmReverseOutput, DdpmEpsilon(), DiffusionNoised()
inline DdpmReverseOutput DdpmReverse(const Tensor& state,const Tensor& time,const Tensor& epsilon,
    const Tensor& innovation,const DiffusionScheduleOutput& schedule) {
  diffusion_detail::Same(state,epsilon);diffusion_detail::Same(state,innovation);
  diffusion_detail::Time(time,state.size(0));
  std::array<Tensor,5> tables{schedule.sqrt_alpha_bar,schedule.sqrt_one_minus_alpha_bar,
    schedule.posterior_coefficient1,schedule.posterior_coefficient2,schedule.posterior_variance};
  for(const auto& table:tables)diffusion_detail::Floating(table,1);
  std::array<Tensor,5> c;
  for(int i=0;i<5;++i) {
    if(tables[i].shape()!=tables[0].shape())throw std::invalid_argument("DDPM posterior table lengths differ");
    c[i]=cast(DiffusionExtract(tables[i],time,int(state.dim())),state.dtype());
  }
  diffusion_detail::Positive(c[0],true);diffusion_detail::Positive(c[1]);diffusion_detail::Positive(c[4]);
  auto product=c[1]*epsilon;gan_detail::Finite(product);auto raw=(state-product)/c[0];gan_detail::Finite(raw);
  auto clipped=minimum(maximum(raw,scalar_like(-1,raw)),scalar_like(1,raw));
  auto a=c[2]*clipped,b=c[3]*state;gan_detail::Finite(a);gan_detail::Finite(b);auto mean=a+b;gan_detail::Finite(mean);
  auto perturbation=sqrt(c[4])*innovation;gan_detail::Finite(perturbation);auto value=mean+perturbation;gan_detail::Finite(value);
  return {value,mean,c[4],raw,clipped};
}
} // namespace datamunge::nn::models
