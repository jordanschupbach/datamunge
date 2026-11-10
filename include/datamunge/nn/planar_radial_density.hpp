#pragma once
/// @file planar_radial_density.hpp
/// @brief Constrained planar/radial transforms, source inverse derivatives and compact density maps.
#include "datamunge/nn/affine_flows.hpp"
#include <array>

namespace datamunge::nn::models {
/// @brief Validation and finite-radius arithmetic for elementary two-coordinate flow maps.
namespace elementary_detail {
/// @brief Require a finite matching floating CPU scalar without changing its graph.
/// @param value Candidate scalar leaf or expression, arbitrary strides.
/// @param input Validated reference floating CPU point matrix.
/// @throws std::invalid_argument For nonfinite values, wrong rank, dtype or device.
inline void Scalar(const Tensor& value,const Tensor& input) {
  gan_detail::Finite(value);
  if(value.dim()!=0 || value.dtype()!=input.dtype() || value.device().backend!=compute::Backend::CPU)
    throw std::invalid_argument("Elementary flow parameters must be matching floating CPU scalars");
}
/// @brief Reject nonpositive or nonfinite represented flow factors without clipping.
/// @param value Floating scalar or matrix whose every entry must be positive finite.
/// @throws std::invalid_argument For nonfinite or nonpositive entries.
inline void Positive(const Tensor& value) {
  gan_detail::Finite(value);
  for(double x:value.contiguous().to_vector<double>())if(x<=0)
    throw std::invalid_argument("Elementary flow factors must remain positive finite");
}
/// @brief Compute source row radii with explicitly finite zero-radius derivatives.
///
/// For finite floating CPU offset[N,2], select sqrt(max(sum(offset^2),1e-30))
/// at positive squared radius and square*0 at exactly zero. The source tiny
/// positive-radius stabilization is preserved; this is not an arbitrary norm
/// subgradient. Owning live[N,1] result, O(N) work/storage, no mutable state.
/// @param offset Matching finite floating[N,2] centered coordinate batch.
/// @return Owning live floating[N,1] nonnegative radius with source zero derivative at center.
/// @throws std::invalid_argument For nonfinite squared distances.
inline Tensor Radius(const Tensor& offset) {
  const auto square_distance=sum(square(offset),{1},true);gan_detail::Finite(square_distance);
  return where(greater(square_distance,0.),sqrt(maximum(square_distance,full({},1e-30,offset.dtype()))),square_distance*0.);
}
} // namespace elementary_detail

/// @brief Own a live elementary planar/radial map and its per-row determinant diagnostics.
///
/// Four flat Tensor handles retain storage and applicable graph paths after
/// parent/parameter-store collection. factor is the planar determinant factor or
/// radial tangential factor, not a log scale; coordinate is the planar solved
/// scalar t or radial source radius. For radial inverse this is the recovered
/// input radius. No fitting loop, optimizer or sampling stream is retained.
/// Planar inverse graphs follow the stopped-root/denominator correction, so their
/// higher derivatives are source graph derivatives, not exact implicit derivatives.
/// @return Owning fields with the shapes and semantics documented below.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9903);
/// auto out=n::models::ElementaryFlow(p,n::zeros({2,2}),0);
/// // value[2,2], logdet[2], factor/coordinate[2,1].
/// @endcode
/// @par References
/// - Rezende and Mohamed(2015), Variational Inference with Normalizing Flows,
/// https://arxiv.org/abs/1505.05770 . This compact standalone density adaptation
/// uses four2-D maps and explicit inverses; it is not a VAE/ELBO implementation.
/// @see ElementaryFlow(), PlanarFlow(), RadialFlow()
struct ElementaryFlowOutput {
  Tensor value;      ///< Owning live matching floating CPU[N,2] transformed coordinates.
  Tensor logdet;     ///< Owning live matching floating[N] signed log absolute input determinant, nats.
  Tensor factor;     ///< Owning live matching floating[N,1] positive planar determinant or radial tangential factor.
  Tensor coordinate; ///< Owning live matching floating[N,1] planar t or radial source radius, recovered for inverse.
};

/// @brief Apply a constrained two-dimensional planar map or its numerical implicit-gradient inverse.
///
/// Finite Float32/64 CPU input[N,2],N>0, arbitrary strides; all five parameters
/// are finite matching CPU scalar Tensors. Set e=exp(log_frequency), direction=
/// (cos(angle),sin(angle)), tangent=(-sin(angle),cos(angle)), projection=
/// softplus(raw_projection)-.9, w=e*direction, u=(projection/e)*direction+
/// perpendicular*tangent. Forward t=sum(input*w)+bias, value=input+u*tanh(t),
/// factor=1+projection*(1-tanh(t)^2), logdet=log(factor) per row.
///
/// Inverse solves t+projection*tanh(t)=c=sum(input*w)+bias using32 bisections
/// bracketed by c+/- (abs(projection)+1), then ONE Newton correction: stop root
/// and denominator, t=root+((c-root)-projection*tanh(root))/denominator. Return
/// input-u*tanh(t) and negative log(factor) evaluated on corrected t. This is a
/// numerical inverse, not a closed form. First gradients use the source implicit
/// correction, avoiding discrete bisection branches. Higher graphs remain live
/// through the correction but are NOT exact higher implicit derivatives because
/// the denominator/root are stopped. Floating residuals remain.
///
/// Owns all fields after collection; forward applicable first/higher paths stay
/// live. No store/RNG/fitting/optimizer/sampler/hidden no-grad. O(32N) inverse or
/// O(N) forward work/storage. Calls share no mutable state. Validate inputs before
/// arithmetic; reject zero/overflowed frequency/reciprocal, nonfinite intermediates
/// and nonpositive represented determinants without clipping or fallback.
/// @param input Finite floating CPU[N,2], positive rows and arbitrary strides.
/// @param angle Finite matching scalar orientation in radians.
/// @param log_frequency Finite matching scalar log positive frequency; both frequency directions representable.
/// @param raw_projection Finite matching scalar, projection=softplus(raw)-.9 > -1 in exact arithmetic.
/// @param perpendicular Finite matching scalar tangential displacement coefficient.
/// @param bias Finite matching scalar offset in the planar tanh argument.
/// @param inverse False(default) for forward; true for32 bisections and implicit Newton correction.
/// @return Owning live ElementaryFlowOutput with value/logdet and factor/t diagnostics.
/// @throws std::invalid_argument For invalid inputs/leaves or unrepresentable/nonfinite math.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;auto type=n::DType::Float64;
/// auto x=n::ones({2,2},type),zero=n::full({},0.,type);
/// auto y=n::models::PlanarFlow(x,zero,zero,n::full({},.4,type),zero,zero);
/// auto rec=n::models::PlanarFlow(y.value,zero,zero,n::full({},.4,type),zero,zero,true);
/// // rec.value approximates x; forward/inverse logdets approximately cancel.
/// @endcode
/// @par References
/// - Rezende and Mohamed(2015), Variational Inference with Normalizing Flows,
/// https://arxiv.org/abs/1505.05770 . This compact standalone density adaptation
/// uses four2-D maps and explicit inverses; it is not a VAE/ELBO implementation.
/// @see ElementaryFlowOutput, ElementaryFlow(), RadialFlow()
inline ElementaryFlowOutput PlanarFlow(const Tensor& input,const Tensor& angle,
    const Tensor& log_frequency,const Tensor& raw_projection,const Tensor& perpendicular,
    const Tensor& bias,bool inverse=false) {
  flow_detail::Matrix(input,2);
  for(const auto& parameter:{angle,log_frequency,raw_projection,perpendicular,bias})elementary_detail::Scalar(parameter,input);
  const auto frequency=exp(log_frequency);elementary_detail::Positive(frequency);elementary_detail::Positive(exp(-log_frequency));
  const auto projection=softplus(raw_projection)-.9;
  const auto direction=stack({cos(angle),sin(angle)}),tangent=stack({-sin(angle),cos(angle)});
  const auto w=frequency*direction,u=(projection/frequency)*direction+perpendicular*tangent;
  gan_detail::Finite(w);gan_detail::Finite(u);
  const auto c=sum(input*w,{1},true)+bias;gan_detail::Finite(c);Tensor t=c;
  if(inverse){
    const auto margin=sqrt(square(projection))+1.;gan_detail::Finite(margin);
    Tensor low=c-margin,high=c+margin;gan_detail::Finite(low);gan_detail::Finite(high);
    for(int iteration=0;iteration<32;++iteration){
      const auto mid=.5*(low+high),residual=(mid+projection*tanh(mid))-c;
      gan_detail::Finite(mid);gan_detail::Finite(residual);const auto below=less(residual,0.);
      low=where(below,mid,low);high=where(below,high,mid);
    }
    const auto root=stop_gradient(.5*(low+high)),ht=tanh(root);
    const auto denominator=stop_gradient(1.+projection*(1.-square(ht)));elementary_detail::Positive(denominator);
    const auto residual=(c-root)-projection*ht;t=root+residual/denominator;
  }
  gan_detail::Finite(t);const auto activation=tanh(t),factor=1.+projection*(1.-square(activation));elementary_detail::Positive(factor);
  const auto shift=activation*u;auto value=inverse?input-shift:input+shift;
  auto ld=log(factor);if(inverse)ld=-ld;ld=reshape(ld,{-1});
  gan_detail::Finite(value);gan_detail::Finite(ld);return {value,ld,factor,t};
}

/// @brief Apply a constrained radial map or its stable quadratic-radius inverse.
///
/// Finite Float32/64 CPU input[N,2],N>0, center[2], raw_alpha/raw_beta scalars,
/// all matching dtype and arbitrary strides. alpha=.1+softplus(raw_alpha)>0,
/// beta=softplus(raw_beta)-alpha > -alpha. Offset=input-center, distance is the
/// source stabilized radius: sqrt(max(sum(offset^2),1e-30)) for positive squared
/// radius, explicitly zero with zero radius derivative at center. Forward uses
/// r=distance, factor=1+beta/(alpha+r), radial_factor=1+beta*alpha/(alpha+r)^2;
/// value=center+offset*factor and logdet=log(factor)+log(radial_factor).
///
/// Inverse recovers r from r^2+(alpha+beta-distance)*r-alpha*distance=0.
/// With a=alpha+beta-distance and disc=sqrt(a*a+4*alpha*distance), use
/// 2*alpha*distance/max(a+disc,1e-20) for a>=0, otherwise .5*(disc-a).
/// Return center+offset/factor and negated logdet at that recovered radius.
/// Both quadratic branches are evaluated; inactive denominator is stabilized.
/// At center the value derivative is factor*I forward and I/factor inverse;
/// source radius derivative is zero, including logdet's chosen center derivative.
/// Do not infer a smooth mathematical Hessian of radius/logdet at center.
///
/// Owning live flat fields survive collection; applicable first/higher input and
/// parameter graphs remain within structural branches and the source radius rule.
/// No store/RNG/fitting/optimizer/sampler/hidden no-grad. O(N) work/storage, no
/// shared mutable state. Inputs validate before arithmetic; reject nonpositive
/// represented alpha, alpha+beta, discriminant, denominator or factors and any
/// nonfinite intermediate/result. Extreme finite parameters may reject; floating
/// inverse residuals and the source tiny-radius stabilization remain.
/// @param input Finite matching floating CPU[N,2], positive rows, arbitrary strides.
/// @param center Finite matching floating CPU[2] center.
/// @param raw_alpha Finite matching scalar; alpha=.1+softplus(raw_alpha).
/// @param raw_beta Finite matching scalar; beta=softplus(raw_beta)-alpha.
/// @param inverse False(default) for forward; true for the stable positive quadratic radius.
/// @return Owning live ElementaryFlowOutput with value/logdet and tangential factor/source radius.
/// @throws std::invalid_argument For invalid inputs/leaves or unrepresentable/nonfinite math.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;auto type=n::DType::Float64;
/// auto x=n::ones({2,2},type),center=n::zeros({2},type);
/// auto a=n::full({},.4,type),b=n::full({},.56,type);
/// auto y=n::models::RadialFlow(x,center,a,b);
/// auto rec=n::models::RadialFlow(y.value,center,a,b,true);
/// // rec.value approximates x; positive tangential factors are[N,1].
/// @endcode
/// @par References
/// - Rezende and Mohamed(2015), Variational Inference with Normalizing Flows,
/// https://arxiv.org/abs/1505.05770 . This compact standalone density adaptation
/// uses four2-D maps and explicit inverses; it is not a VAE/ELBO implementation.
/// @see ElementaryFlowOutput, ElementaryFlow(), PlanarFlow()
inline ElementaryFlowOutput RadialFlow(const Tensor& input,const Tensor& center,
    const Tensor& raw_alpha,const Tensor& raw_beta,bool inverse=false) {
  flow_detail::Matrix(input,2);gan_detail::Finite(center);
  if(center.dim()!=1 || center.size(0)!=2 || center.dtype()!=input.dtype() || center.device().backend!=compute::Backend::CPU)
    throw std::invalid_argument("RadialFlow center must be a matching floating CPU[2] tensor");
  elementary_detail::Scalar(raw_alpha,input);elementary_detail::Scalar(raw_beta,input);
  const auto alpha=.1+softplus(raw_alpha),beta=softplus(raw_beta)-alpha;
  elementary_detail::Positive(alpha);elementary_detail::Positive(alpha+beta);
  const auto offset=input-center,distance=elementary_detail::Radius(offset);Tensor radius=distance;
  if(inverse){
    const auto a=(alpha+beta)-distance,discriminant=square(a)+4.*(alpha*distance);elementary_detail::Positive(discriminant);
    const auto disc=sqrt(discriminant),positive=(2.*(alpha*distance))/maximum(a+disc,full({},1e-20,input.dtype())),negative=.5*(disc-a);
    gan_detail::Finite(positive);gan_detail::Finite(negative);radius=where(greater_equal(a,0.),positive,negative);
  }
  const auto denom=alpha+radius;gan_detail::Finite(radius);elementary_detail::Positive(denom);
  const auto factor=1.+beta/denom,radial_factor=1.+(beta*alpha)/square(denom);
  elementary_detail::Positive(factor);elementary_detail::Positive(radial_factor);
  auto ld=log(factor)+log(radial_factor);if(inverse)ld=-ld;ld=reshape(ld,{-1});
  const auto value=center+(inverse?offset/factor:offset*factor);gan_detail::Finite(value);gan_detail::Finite(ld);
  return {value,ld,factor,radius};
}

/// @brief Apply one named source planar or radial flow with caller-owned parameters.
///
/// Finite matching Float32/64 CPU input[N,2],N>0, arbitrary strides, block0..3.
/// radial=false(default) creates flow<block>/angle,log_frequency,raw_projection,
/// perpendicular,bias scalar leaves in that order: angle=Double(.4f+.65f*block),
/// log_frequency=0, raw_projection=.4, perpendicular=Double(.02f*(odd?-1:1)),
/// bias=.05. Five values/five leaves. radial=true creates center[2] normal(.2),
/// raw_alpha=.4, raw_beta=.56: four values/three leaves. Calls PlanarFlow or
/// RadialFlow with those leaves; diagnostics retain family-specific semantics.
/// Use a separate store per family: their shared prefix does not convert or clear
/// other-family leaves. Only missing leaves initialize; existing leaves unchanged.
/// Warm source blocks0..3 with seed9903 before inverses to preserve source order.
///
/// Owns outputs/graphs after collection. Planar inverse first/higher graphs follow
/// its stopped-root/denominator correction; higher paths are not exact implicit
/// higher derivatives. Radial center/tiny-radius rules are source conventions.
/// No fitting/optimizer/sampler/hidden no-grad or warmed RNG. O(N) forward/radial
/// or O(32N) planar inverse work/storage. Synchronize shared stores. Inputs validate
/// before leaves; stored-leaf/math errors can follow partial creation.
/// @param parameters Caller-owned matching floating store, source full-model seed9903.
/// @param input Finite matching floating CPU[N,2], positive rows and arbitrary strides.
/// @param block Source block0..3 used for names and planar orientation initialization.
/// @param radial False(default) for planar; true for radial, preferably separate stores.
/// @param inverse False(default) forward; true uses the family inverse and signed logdet.
/// @return Owning live ElementaryFlowOutput with four flat value/determinant/diagnostic fields.
/// @throws std::invalid_argument For invalid inputs/leaves or unrepresentable/nonfinite math.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9903);
/// auto x=n::ones({2,2});auto y=n::models::ElementaryFlow(p,x,0);
/// auto rec=n::models::ElementaryFlow(p,y.value,0,false,true);
/// // Planar block:5 values/5 leaves; rec.value approximately recovers x.
/// @endcode
/// @par References
/// - Rezende and Mohamed(2015), Variational Inference with Normalizing Flows,
/// https://arxiv.org/abs/1505.05770 . This compact standalone density adaptation
/// uses four2-D maps and explicit inverses; it is not a VAE/ELBO implementation.
/// @see PlanarFlow(), RadialFlow(), PlanarRadialDensity()
inline ElementaryFlowOutput ElementaryFlow(ParameterStore& parameters,const Tensor& input,
    int block,bool radial=false,bool inverse=false) {
  flow_detail::Matrix(input,2);
  if(input.dtype()!=parameters.dtype() || block<0 || block>3)
    throw std::invalid_argument("ElementaryFlow needs matching floating store and block0..3");
  const auto name="flow"+std::to_string(block)+"/";
  if(radial){
    const auto center=parameters.get(name+"center",{2},init::normal(.2));
    const auto alpha=parameters.get(name+"raw_alpha",{},init::constant(.4));
    const auto beta=parameters.get(name+"raw_beta",{},init::constant(.56));
    return RadialFlow(input,center,alpha,beta,inverse);
  }
  const auto angle=parameters.get(name+"angle",{},init::constant(double(.4f+.65f*block)));
  const auto frequency=parameters.get(name+"log_frequency",{},init::constant(0));
  const auto projection=parameters.get(name+"raw_projection",{},init::constant(.4));
  const auto perpendicular=parameters.get(name+"perpendicular",{},init::constant(double(.02f*(block%2?-1:1))));
  const auto bias=parameters.get(name+"bias",{},init::constant(.05));
  return PlanarFlow(input,angle,frequency,projection,perpendicular,bias,inverse);
}

/// @brief Own four elementary-flow stages and compact Gaussian-density intermediates.
///
/// Nineteen flat Tensor handles own storage and applicable live graphs after
/// parent/store collection. Stage factor/coordinate fields have explicit names:
/// planar determinant factor/t, or radial tangential factor/source radius. They
/// correspond to legacy archive scale/shift diagnostics, not affine log scales or
/// shifts. No nested records, fitting, optimizer or sampling stream is retained.
/// @return Owning live stage and full density fields documented below.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9903);
/// auto out=n::models::PlanarRadialDensity(p,n::zeros({2,2}),true);
/// // Radial model:18 values/13 leaves; latent[2,2] and logdet[2].
/// @endcode
/// @par References
/// - Rezende and Mohamed(2015), Variational Inference with Normalizing Flows,
/// https://arxiv.org/abs/1505.05770 . This compact standalone density adaptation
/// uses four2-D maps and explicit inverses; it is not a VAE/ELBO implementation.
/// @see PlanarRadialDensity(), PlanarRadialInverse(), ElementaryFlowOutput
struct PlanarRadialDensityOutput {
  Tensor flow0; ///< Owning live matching floating[N,2] source stage0 transformed coordinates.
  Tensor flow_logdet0; ///< Owning live matching floating[N] stage0 forward log absolute determinant, nats.
  Tensor flow_factor0; ///< Owning live matching floating[N,1] stage0 positive planar determinant or radial tangential factor.
  Tensor flow_coordinate0; ///< Owning live matching floating[N,1] stage0 planar t or radial source radius.
  Tensor flow1; ///< Owning live matching floating[N,2] source stage1 transformed coordinates.
  Tensor flow_logdet1; ///< Owning live matching floating[N] stage1 forward log absolute determinant, nats.
  Tensor flow_factor1; ///< Owning live matching floating[N,1] stage1 positive planar determinant or radial tangential factor.
  Tensor flow_coordinate1; ///< Owning live matching floating[N,1] stage1 planar t or radial source radius.
  Tensor flow2; ///< Owning live matching floating[N,2] source stage2 transformed coordinates.
  Tensor flow_logdet2; ///< Owning live matching floating[N] stage2 forward log absolute determinant, nats.
  Tensor flow_factor2; ///< Owning live matching floating[N,1] stage2 positive planar determinant or radial tangential factor.
  Tensor flow_coordinate2; ///< Owning live matching floating[N,1] stage2 planar t or radial source radius.
  Tensor flow3; ///< Owning live matching floating[N,2] source stage3 transformed coordinates.
  Tensor flow_logdet3; ///< Owning live matching floating[N] stage3 forward log absolute determinant, nats.
  Tensor flow_factor3; ///< Owning live matching floating[N,1] stage3 positive planar determinant or radial tangential factor.
  Tensor flow_coordinate3; ///< Owning live matching floating[N,1] stage3 planar t or radial source radius.
  Tensor unscaled; ///< Live matching floating[N,2] alias of flow3 before global scaling.
  Tensor latent;   ///< Owning live matching floating[N,2] globally scaled Gaussian coordinates.
  Tensor logdet;   ///< Owning live matching floating[N] total forward log absolute determinant, nats.
};

/// @brief Map observations through four source planar or radial stages and global scale.
///
/// Finite matching Float32/64 CPU input[N,2],N>0, arbitrary strides. Apply
/// ElementaryFlow blocks0..3 in order, accumulating each per-row logdet, then
/// FlowDiagonalScale. Use one family per store: radial=false(default) planar has
/// 22 values/21 leaves; radial=true has18 values/13 leaves. Source seed9903 warms
/// each block in order then log_scale[2]. Only missing leaves initialize; warmed
/// inference changes no leaves and consumes no RNG. Owns all live flat stages,
/// factors, coordinates and density intermediates after collection. Applicable
/// first/higher forward paths remain live under the source radius conventions.
/// No fitting/optimizer/sampler/hidden no-grad or VAE/ELBO workflow. O(N) work/
/// storage for four fixed stages. Synchronize shared stores. Inputs validate before
/// leaves; stored-leaf/math failure may follow partial creation. Reject nonfinite
/// or unrepresentable flow/scale math; floating inverse residuals remain.
/// @param parameters Caller-owned matching floating store, source seed9903, one family per store.
/// @param input Finite matching floating CPU[N,2] observations, positive rows, arbitrary strides.
/// @param radial False(default) for four planar maps; true for four radial maps.
/// @return Owning live PlanarRadialDensityOutput with19 flat stage and full density fields.
/// @throws std::invalid_argument For invalid inputs/leaves or unrepresentable/nonfinite math.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9903);
/// auto path=n::models::PlanarRadialDensity(p,n::ones({3,2}));
/// auto density=n::models::NormalFlowDensity(path.latent,path.logdet,true);
/// // Planar22 values/21 leaves; source-compatible mean NLL in nats/row.
/// @endcode
/// @par References
/// - Rezende and Mohamed(2015), Variational Inference with Normalizing Flows,
/// https://arxiv.org/abs/1505.05770 . This compact standalone density adaptation
/// uses four2-D maps and explicit inverses; it is not a VAE/ELBO implementation.
/// @see PlanarRadialDensityOutput, PlanarRadialInverse(), ElementaryFlow(), NormalFlowDensity()
inline PlanarRadialDensityOutput PlanarRadialDensity(ParameterStore& parameters,const Tensor& input,bool radial=false) {
  flow_detail::Matrix(input,2);
  if(input.dtype()!=parameters.dtype())throw std::invalid_argument("Elementary density input/store dtypes must match");
  auto h=input,ld=reshape(slice_axis(input,1,0,1),{-1})*0.;std::array<ElementaryFlowOutput,4> stages;
  for(int i=0;i<4;++i){stages[i]=ElementaryFlow(parameters,h,i,radial);h=stages[i].value;ld=ld+stages[i].logdet;}
  const auto scaled=FlowDiagonalScale(parameters,h,ld);
  return {stages[0].value,stages[0].logdet,stages[0].factor,stages[0].coordinate,
          stages[1].value,stages[1].logdet,stages[1].factor,stages[1].coordinate,
          stages[2].value,stages[2].logdet,stages[2].factor,stages[2].coordinate,
          stages[3].value,stages[3].logdet,stages[3].factor,stages[3].coordinate,
          h,scaled.value,scaled.logdet};
}

/// @brief Invert the compact planar/radial density and accumulate its own path determinant.
///
/// Finite matching Float32/64 CPU latent[N,2],N>0, arbitrary strides. Unscale
/// globally, then ElementaryFlow blocks3..0 with inverse=true. Recompute each
/// stage on its supplied inverse path, independent of cached forward points.
/// Planar uses32 bisections plus the stopped-root/denominator Newton correction:
/// source implicit first gradients remain live, higher graphs follow that stopped
/// correction and are not exact implicit higher derivatives. Radial uses stable
/// quadratic radius inversion and explicit finite center derivative. Missing
/// leaves initialize global scale then reversed blocks; warm PlanarRadialDensity
/// first with seed9903 for source ordering. Use a separate store per family.
///
/// Owns live value/logdet after collection; applicable input/parameter graph paths
/// remain. No fitting/optimizer/sampler/hidden no-grad or warmed writes/RNG.
/// O(128N) planar or O(N) radial work/storage. Synchronize shared stores. Inputs
/// validate before leaves; stored-leaf/math failures may follow partial creation.
/// Reject nonfinite or unrepresentable math without clipping/fallback. Numerical
/// planar and floating radial residuals remain; neither path refits parameters.
/// @param parameters Caller-owned matching floating store, preferably forward-warmed for selected family.
/// @param latent Finite matching floating CPU[N,2] supplied Gaussian coordinates, positive rows.
/// @param radial False(default) planar numerical inverses; true stable radial quadratic inverses.
/// @return Owning live FlowScaleOutput with value[N,2] and inverse-path logdet[N].
/// @throws std::invalid_argument For invalid inputs/leaves or unrepresentable/nonfinite math.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9903);
/// auto x=n::ones({2,2});auto path=n::models::PlanarRadialDensity(p,x);
/// auto rec=n::models::PlanarRadialInverse(p,path.latent);
/// // rec.value approximates x; signed path logdets approximately cancel.
/// @endcode
/// @par References
/// - Rezende and Mohamed(2015), Variational Inference with Normalizing Flows,
/// https://arxiv.org/abs/1505.05770 . This compact standalone density adaptation
/// uses four2-D maps and explicit inverses; it is not a VAE/ELBO implementation.
/// @see PlanarRadialDensity(), ElementaryFlow(), PlanarFlow(), RadialFlow(), FlowScaleOutput
inline FlowScaleOutput PlanarRadialInverse(ParameterStore& parameters,const Tensor& latent,bool radial=false) {
  flow_detail::Matrix(latent,2);
  if(latent.dtype()!=parameters.dtype())throw std::invalid_argument("Elementary inverse input/store dtypes must match");
  auto zero=reshape(slice_axis(latent,1,0,1),{-1})*0.;auto out=FlowDiagonalScale(parameters,latent,zero,true);
  for(int i=3;i>=0;--i){auto stage=ElementaryFlow(parameters,out.value,i,radial,true);out.value=stage.value;out.logdet=out.logdet+stage.logdet;}
  gan_detail::Finite(out.logdet);return out;
}
} // namespace datamunge::nn::models
