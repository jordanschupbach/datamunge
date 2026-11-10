#pragma once
/// @file glow_density.hpp
/// @brief Explicit ActNorm calibration, signed two-coordinate mixing and compact Glow maps.
#include "datamunge/nn/affine_flows.hpp"

namespace datamunge::nn::models {
/// @brief Own detached ActNorm calibration moments and proposed affine parameters.
///
/// Four flat owning CPU[D] fields survive input/parent collection. Mean and
/// population variance are Float64; proposed bias/log_scale match the input
/// Float32/64 dtype. No parameter store, writes, calibration flag or RNG retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto x=dm::nn::from_vector(std::vector<float>{1,2,3,6},{2,2});
/// auto out=dm::nn::models::FlowActNormCalibration(x);
/// // mean=[2,4], population variance=[1,4], proposed bias=[-2,-4].
/// @endcode
/// @par References
/// - Kingma and Dhariwal(2018), Glow: Generative Flow with Invertible 1x1
/// Convolutions, https://arxiv.org/abs/1807.03039 . Compact source uses variance+epsilon.
/// @see FlowActNormCalibration(), FlowActNorm()
struct ActNormCalibration {
  Tensor mean;      ///< Owning detached Float64[D] serial Double calibration row mean.
  Tensor variance;  ///< Owning detached Float64[D] population variance, divisor N.
  Tensor bias;      ///< Owning detached matching floating[D] final-cast negative mean.
  Tensor log_scale; ///< Owning detached matching floating[D] final-cast -.5*log(variance+epsilon).
};

/// @brief Derive ActNorm parameters from an explicit calibration batch without changing a store.
///
/// Finite Float32/64 CPU input[N,D], positive axes/strides and finite positive
/// epsilon(default1e-6). Promote to Double, compute mean by summing each row/N
/// serially, then population variance by summing squared centered rows/N in
/// that same order. Return Double moments and bias=-mean,
/// log_scale=-.5*log(variance+epsilon), cast once to input dtype. This preserves
/// the compact source's Double calibration arithmetic. Constant columns and
/// N=1 are supported via epsilon. Stop all input graphs; independently owned
/// outputs, no store writes, hidden row selection, RNG, fitting loop or persistent
/// initialized flag. Caller selects fitting-only rows and explicitly installs
/// parameters once at each desired stage. O(ND) work/transient storage; separate
/// calls share no state. Nonfinite moments/parameters reject; a later transform
/// can still reject unrepresentable exp(+/-log_scale) for extreme epsilon/data.
/// @param input Finite floating CPU[N,D] explicit fitting-only calibration rows.
/// @param epsilon Finite strictly positive population-variance ridge, default1e-6.
/// @return Owning detached ActNormCalibration with mean/variance/bias/log_scale[D].
/// @throws std::invalid_argument For invalid inputs/epsilon or nonfinite computed fields.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto x=n::from_vector(std::vector<double>{1,2,3,6},{2,2});
/// auto fit=n::models::FlowActNormCalibration(x);
/// n::ParameterStore p(9503,n::DType::Float64);
/// p.set("stage/bias",fit.bias);p.set("stage/log_scale",fit.log_scale);
/// // A later FlowActNorm(p,"stage",x) uses these explicit installed values.
/// @endcode
/// @par References
/// - Kingma and Dhariwal(2018), Glow, https://arxiv.org/abs/1807.03039 .
/// Source CPU adaptation uses first32 fitting rows, population variance and epsilon1e-6.
/// @see ActNormCalibration, FlowActNorm(), GlowDensity()
inline ActNormCalibration FlowActNormCalibration(const Tensor& input,double epsilon=1e-6) {
  flow_detail::Matrix(input);
  if(!std::isfinite(epsilon) || epsilon<=0)
    throw std::invalid_argument("ActNorm calibration epsilon must be finite and positive");
  NoGradGuard guard;
  const auto x=cast(input,DType::Float64);
  const auto rows=input.size(0);
  const auto mean=select(cumsum(x/double(rows),0),0,rows-1);
  const auto variance=select(cumsum(square(x-mean)/double(rows),0),0,rows-1);
  auto bias=cast(-mean,input.dtype()),scale=cast(-.5*log(variance+epsilon),input.dtype());
  gan_detail::Finite(mean);gan_detail::Finite(variance);gan_detail::Finite(bias);gan_detail::Finite(scale);
  return {mean,variance,bias,scale};
}

/// @brief Apply named ActNorm or its exact-form inverse without automatic data initialization.
///
/// Finite matching Float32/64 CPU input[N,D], positive axes/strides and nonempty
/// name. Missing name/bias[D] and name/log_scale[D] initialize to zero. Forward
/// returns(input+bias)*exp(log_scale), scalar logdet=sum(log_scale); inverse
/// returns input*exp(-log_scale)-bias, scalar logdet=-sum(log_scale). Both scale
/// directions must remain strictly positive finite. No batch statistics, moving
/// averages, calibration, initialized flag, fitting, optimizer, RNG after warmup
/// or hidden no-grad. Caller explicitly installs FlowActNormCalibration fields.
/// Existing leaves never change; own live results/graphs after parent/store
/// collection. Applicable first/higher paths remain live; O(ND) work/storage,
/// synchronize shared stores. Validate supplied inputs before leaf creation;
/// stored-leaf/math failures can follow partial creation. Nonfinite fields reject.
/// @param parameters Caller-owned floating store matching input dtype.
/// @param name Nonempty leaf prefix followed by /bias and /log_scale.
/// @param input Finite matching floating CPU[N,D], N,D>0, arbitrary strides.
/// @param inverse False(default) for ActNorm forward; true for inverse and negative logdet.
/// @return Owning live FlowScaleOutput with value[N,D] and scalar logdet.
/// @throws std::invalid_argument For invalid inputs/name/store/scale or nonfinite fields.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9503);
/// auto x=n::ones({2,3});auto out=n::models::FlowActNorm(p,"stage",x);
/// // Identity initially, scalar logdet0; six values/two leaves, no calibration.
/// @endcode
/// @par References
/// - Kingma and Dhariwal(2018), Glow, https://arxiv.org/abs/1807.03039 .
/// @see FlowActNormCalibration(), FlowScaleOutput, GlowDensity(), GlowInverse()
inline FlowScaleOutput FlowActNorm(ParameterStore& parameters,const std::string& name,
                                   const Tensor& input,bool inverse=false) {
  flow_detail::Matrix(input);
  if(name.empty() || input.dtype()!=parameters.dtype())
    throw std::invalid_argument("FlowActNorm requires a nonempty name and matching floating input");
  const auto bias=parameters.get(name+"/bias",{input.size(1)},init::zeros());
  const auto scale=parameters.get(name+"/log_scale",{input.size(1)},init::zeros());
  gan_detail::Finite(bias);gan_detail::Finite(scale);
  const auto positive=exp(scale),negative=exp(-scale);
  gan_detail::Finite(positive);gan_detail::Finite(negative);
  for(auto v:{positive,negative})for(double x:v.contiguous().to_vector<double>())
    if(x<=0)throw std::invalid_argument("ActNorm scales in both directions must stay positive finite");
  auto value=inverse?input*negative-bias:(input+bias)*positive;
  auto ld=sum(scale,{0});if(inverse)ld=-ld;
  gan_detail::Finite(value);gan_detail::Finite(ld);return {value,ld};
}

/// @brief Mix two row-vector coordinates through a named invertible matrix, including negative determinants.
///
/// Finite matching Float32/64 CPU input[N,2], positive N/strides, nonempty name
/// and finite angle(default.35 radians). Missing name/W[2,2] is the rotation
/// [[cos(angle),-sin(angle)],[sin(angle),cos(angle)]], computed in Double, rounded
/// once to Float32 and promoted if needed. No RNG is consumed by this initializer.
/// Forward returns input@W, logdet=.5*log((ad-bc)^2). Inverse uses the explicit
/// adjugate/determinant and negative logdet. Row-vector convention gives input
/// Jacobian W transpose; determinant sign is unchanged. Negative determinants
/// are valid orientation reversals; singular matrices reject. The explicit
/// squared-determinant arithmetic preserves the source: det^2 must be positive
/// finite in computation dtype; extreme invertible matrices can therefore reject.
/// No conditioning, calibration, fitting, optimizer, existing-leaf writes or
/// hidden no-grad. Owns live input/weight first/higher paths after parent/store
/// collection. O(N) work/storage; synchronize shared stores. Inputs validate
/// before creation; stored-leaf/math errors can follow it. Nonfinite fields reject.
/// @param parameters Caller-owned floating store matching input dtype.
/// @param name Nonempty prefix followed by /W, one trainable2x2 matrix.
/// @param input Finite matching floating CPU[N,2], N>0, arbitrary strides.
/// @param angle Finite missing-matrix rotation angle in radians, default.35; ignored for existing W.
/// @param inverse False(default) for input@W; true for input@inverse(W).
/// @return Owning live FlowScaleOutput with value[N,2] and scalar signed-direction log|det W|.
/// @throws std::invalid_argument For invalid inputs/name/angle/store, singular/unrepresentable determinant or nonfinite fields.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(1,n::DType::Float64);
/// p.set("mix/W",n::from_vector(std::vector<double>{1,2,3,4},{2,2}));
/// auto out=n::models::FlowMix2D(p,"mix",n::ones({1,2},n::DType::Float64));
/// // value=[[4,6]], logdet=log(2), although det(W)=-2.
/// @endcode
/// @par References
/// - Kingma and Dhariwal(2018), Glow, https://arxiv.org/abs/1807.03039 .
/// Source two-channel CPU adaptation uses a direct matrix/adjugate, not LU parameterization.
/// @see FlowScaleOutput, GlowDensity(), GlowInverse()
inline FlowScaleOutput FlowMix2D(ParameterStore& parameters,const std::string& name,
                                const Tensor& input,double angle=.35,bool inverse=false) {
  flow_detail::Matrix(input,2);
  if(name.empty() || !std::isfinite(angle) || input.dtype()!=parameters.dtype())
    throw std::invalid_argument("FlowMix2D needs matching floating points, a name and finite angle");
  const auto w=parameters.get(name+"/W",{2,2},[angle](const Shape&,DType dtype,Generator&){
    return cast(from_vector(std::vector<float>{float(std::cos(angle)),float(-std::sin(angle)),
      float(std::sin(angle)),float(std::cos(angle))},{2,2}),dtype);
  });
  gan_detail::Finite(w);
  const auto a=reshape(slice(w,{0,0},{1,1}),{}),b=reshape(slice(w,{0,1},{1,1}),{});
  const auto c=reshape(slice(w,{1,0},{1,1}),{}),d=reshape(slice(w,{1,1},{1,1}),{});
  const auto det=a*d-b*c,squared=square(det);gan_detail::Finite(squared);
  if(squared.item()<=0)throw std::invalid_argument("FlowMix2D needs a nonzero representable squared determinant");
  auto ld=.5*log(squared);auto matrix=w;
  if(inverse)matrix=stack({stack({d,-b}),stack({-c,a})})/det;
  auto value=matmul(input,matrix);if(inverse)ld=-ld;
  gan_detail::Finite(value);gan_detail::Finite(ld);return {value,ld};
}

/// @brief Own all four compact Glow stages and their live density intermediates.
///
/// Thirty-one flat Tensor handles retain applicable first/higher graphs and
/// storage after parent/store collection. Each stage exposes its actual ActNorm
/// input/output, mixed coordinates, affine output/logdet/bounded log scale/shift.
/// No model object, optimizer, calibration state or sampler is retained separately.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9503);
/// auto out=n::models::GlowDensity(p,n::zeros({3,2}));
/// // ActNorm/mix/coupling/unscaled/latent[3,2], total/stage logdet[3],
/// // coupling log scales/shifts[3,1];296 values/28 leaves before calibration.
/// @endcode
/// @par References
/// - Kingma and Dhariwal(2018), Glow, https://arxiv.org/abs/1807.03039 .
/// @see GlowDensity(), GlowInverse(), FlowActNormCalibration()
struct GlowDensityOutput {
  Tensor actnorm_input0;  ///< Live floating[N,2] input to stage0 ActNorm.
  Tensor actnorm_output0; ///< Owning live floating[N,2] stage0 normalized coordinates.
  Tensor mixed0;          ///< Owning live floating[N,2] stage0 row-matrix product.
  Tensor coupling0;       ///< Owning live floating[N,2] stage0 affine coupling output.
  Tensor coupling_logdet0;///< Owning live floating[N] stage0 affine-only logdet.
  Tensor coupling_scale0; ///< Owning live floating[N,1] stage0 bounded log scale.
  Tensor coupling_shift0; ///< Owning live floating[N,1] stage0 affine shift.
  Tensor actnorm_input1;  ///< Live floating[N,2] input to stage1 ActNorm, alias of coupling0.
  Tensor actnorm_output1; ///< Owning live floating[N,2] stage1 normalized coordinates.
  Tensor mixed1;          ///< Owning live floating[N,2] stage1 row-matrix product.
  Tensor coupling1;       ///< Owning live floating[N,2] stage1 affine coupling output.
  Tensor coupling_logdet1;///< Owning live floating[N] stage1 affine-only logdet.
  Tensor coupling_scale1; ///< Owning live floating[N,1] stage1 bounded log scale.
  Tensor coupling_shift1; ///< Owning live floating[N,1] stage1 affine shift.
  Tensor actnorm_input2;  ///< Live floating[N,2] input to stage2 ActNorm, alias of coupling1.
  Tensor actnorm_output2; ///< Owning live floating[N,2] stage2 normalized coordinates.
  Tensor mixed2;          ///< Owning live floating[N,2] stage2 row-matrix product.
  Tensor coupling2;       ///< Owning live floating[N,2] stage2 affine coupling output.
  Tensor coupling_logdet2;///< Owning live floating[N] stage2 affine-only logdet.
  Tensor coupling_scale2; ///< Owning live floating[N,1] stage2 bounded log scale.
  Tensor coupling_shift2; ///< Owning live floating[N,1] stage2 affine shift.
  Tensor actnorm_input3;  ///< Live floating[N,2] input to stage3 ActNorm, alias of coupling2.
  Tensor actnorm_output3; ///< Owning live floating[N,2] stage3 normalized coordinates.
  Tensor mixed3;          ///< Owning live floating[N,2] stage3 row-matrix product.
  Tensor coupling3;       ///< Owning live floating[N,2] stage3 affine coupling output.
  Tensor coupling_logdet3;///< Owning live floating[N] stage3 affine-only logdet.
  Tensor coupling_scale3; ///< Owning live floating[N,1] stage3 bounded log scale.
  Tensor coupling_shift3; ///< Owning live floating[N,1] stage3 affine shift.
  Tensor unscaled;        ///< Live floating[N,2] alias of final coupling3; no extra global scale.
  Tensor latent;          ///< Live floating[N,2] alias of final coupling3 Gaussian coordinates.
  Tensor logdet;          ///< Owning live floating[N] full per-point ActNorm+mix+coupling logdet.
};

/// @brief Apply four source two-channel Glow steps with explicit ActNorm, mixing and affine coupling.
///
/// Finite matching Float32/64 CPU input[N,2], positive rows/strides. Stage i=0..3
/// applies FlowActNorm named actnorm<i>, FlowMix2D named mix<i> initialized at
/// angle .35+.2*i, then bounded RealNvpCoupling block i. Accumulate
/// ld=ld+(sum(actnorm_log_scale)+(mix_logdet+coupling_logdet)) in source order,
/// starting at input-linked zeros. No extra final global scale. Exposes all31
/// flat source fields;296 values in28 leaves, source seed9503. Missing ActNorm
/// bias/scale are zero; this call never calibrates on its input. The caller
/// explicitly selects calibration rows and installs derived parameters once,
/// sequentially between calls, before fitting. Only missing leaves initialize;
/// existing leaves never change. No fitting, optimizer, sampling, batch/moving
/// statistics, multiscale image model or hidden no-grad. Owns applicable first/
/// higher graphs after parent/store collection. O(64N) work/storage; synchronize
/// shared stores. Inputs validate before creation; leaf/math failures can follow
/// partial creation. Both scale directions and squared matrix determinants must
/// remain positive finite; nonfinite fields reject, negative matrix signs allowed.
/// @param parameters Caller-owned matching floating store, source initialization seed9503.
/// @param input Finite matching Float32/64 CPU[N,2] explicit observations.
/// @return Owning live GlowDensityOutput with31 flat stage/density fields.
/// @throws std::invalid_argument For invalid inputs/store/scales/matrices or nonfinite fields.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9503);
/// auto path=n::models::GlowDensity(p,n::zeros({2,2}));
/// auto density=n::models::NormalFlowDensity(path.latent,path.logdet,true);
/// // p.count()==296,p.size()==28; explicit mean source-compatible NLL, no calibration.
/// @endcode
/// @par References
/// - Kingma and Dhariwal(2018), Glow: Generative Flow with Invertible 1x1
/// Convolutions, https://arxiv.org/abs/1807.03039 . Four two-channel CPU teaching
/// stages, direct matrix arithmetic and .8*tanh bounded affine log scales.
/// @see GlowDensityOutput, GlowInverse(), FlowActNormCalibration(), RealNvpCoupling()
inline GlowDensityOutput GlowDensity(ParameterStore& parameters,const Tensor& input) {
  flow_detail::Matrix(input,2);
  if(input.dtype()!=parameters.dtype() || input.size(0)>std::numeric_limits<std::int64_t>::max()/16)
    throw std::invalid_argument("Glow input/store dtypes and activation sizes must match");
  GlowDensityOutput out;auto h=input,ld=reshape(slice_axis(input,1,0,1),{-1})*0.;
  for(int i=0;i<4;++i){
    auto before=h;auto norm=FlowActNorm(parameters,"actnorm"+std::to_string(i),h);
    auto mix=FlowMix2D(parameters,"mix"+std::to_string(i),norm.value,.35+.2*i);
    auto coupling=RealNvpCoupling(parameters,mix.value,i);h=coupling.value;
    ld=ld+(norm.logdet+(mix.logdet+coupling.logdet));
    if(i==0){out.actnorm_input0=before;out.actnorm_output0=norm.value;out.mixed0=mix.value;
      out.coupling0=h;out.coupling_logdet0=coupling.logdet;out.coupling_scale0=coupling.scale;out.coupling_shift0=coupling.shift;}
    if(i==1){out.actnorm_input1=before;out.actnorm_output1=norm.value;out.mixed1=mix.value;
      out.coupling1=h;out.coupling_logdet1=coupling.logdet;out.coupling_scale1=coupling.scale;out.coupling_shift1=coupling.shift;}
    if(i==2){out.actnorm_input2=before;out.actnorm_output2=norm.value;out.mixed2=mix.value;
      out.coupling2=h;out.coupling_logdet2=coupling.logdet;out.coupling_scale2=coupling.scale;out.coupling_shift2=coupling.shift;}
    if(i==3){out.actnorm_input3=before;out.actnorm_output3=norm.value;out.mixed3=mix.value;
      out.coupling3=h;out.coupling_logdet3=coupling.logdet;out.coupling_scale3=coupling.scale;out.coupling_shift3=coupling.shift;}
  }
  out.unscaled=out.latent=h;out.logdet=ld;gan_detail::Finite(ld);return out;
}

/// @brief Invert compact Glow in reverse affine, matrix and ActNorm order, using its own path logdet.
///
/// Matching finite Float32/64 CPU latent[N,2], positive rows/strides. Stages3..0
/// invert bounded affine coupling, named mixing matrix, then named ActNorm.
/// Accumulate ld=ld+((inverse_coupling_logdet+inverse_mix_logdet)-sum(actnorm_scale))
/// on that inverse path, not a cached/unrelated forward point. Forward-warmed
/// source seed9503 preserves initialization order; missing inverse leaves can
/// initialize in reverse stage order. This function never calibrates, fits,
/// samples, changes existing leaves or consumes RNG after warmup. Owns live
/// value/logdet graphs after parent/store collection; applicable first/higher
/// paths remain live. O(64N) work/storage; synchronize shared stores. Inputs
/// validate before creation; leaf/math failures can follow partial creation.
/// Nonfinite scales/matrices/results reject; finite roundtrip errors remain.
/// Negative matrix determinants are valid and use log absolute determinants.
/// @param parameters Caller-owned matching floating store, preferably forward-warmed/calibrated.
/// @param latent Finite matching Float32/64 CPU[N,2] caller-supplied latent coordinates.
/// @return Owning live FlowScaleOutput with inverse value[N,2] and logdet[N].
/// @throws std::invalid_argument For invalid inputs/store/scales/matrices or nonfinite fields.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9503);
/// auto x=n::ones({2,2});auto path=n::models::GlowDensity(p,x);
/// auto rec=n::models::GlowInverse(p,path.latent);
/// // rec.value approximates x; rec.logdet+path.logdet approximates zero per row.
/// @endcode
/// @par References
/// - Kingma and Dhariwal(2018), Glow, https://arxiv.org/abs/1807.03039 .
/// @see GlowDensity(), FlowActNorm(), FlowMix2D(), RealNvpCoupling()
inline FlowScaleOutput GlowInverse(ParameterStore& parameters,const Tensor& latent) {
  flow_detail::Matrix(latent,2);
  if(latent.dtype()!=parameters.dtype() || latent.size(0)>std::numeric_limits<std::int64_t>::max()/16)
    throw std::invalid_argument("Glow inverse needs matching floating points and safe activation sizes");
  auto h=latent,ld=reshape(slice_axis(latent,1,0,1),{-1})*0.;
  for(int i=3;i>=0;--i){
    auto coupling=RealNvpCoupling(parameters,h,i,true);
    auto mix=FlowMix2D(parameters,"mix"+std::to_string(i),coupling.value,.35+.2*i,true);
    auto norm=FlowActNorm(parameters,"actnorm"+std::to_string(i),mix.value,true);
    h=norm.value;ld=ld+((coupling.logdet+mix.logdet)+norm.logdet);
  }
  gan_detail::Finite(ld);return {h,ld};
}
} // namespace datamunge::nn::models
