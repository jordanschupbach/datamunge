#pragma once
/// @file spline_density.hpp
/// @brief Live monotone rational-quadratic splines, analytic inverses and compact spline density flows.
#include "datamunge/nn/affine_flows.hpp"
#include <cmath>

namespace datamunge::nn::models {
/// @brief Own live spline values, log determinants, actual bin sizes and positive knot derivatives.
///
/// Five flat Tensor handles retain applicable first/higher input/head/parameter
/// graphs and storage after parent/store collection. The pure spline's value
/// is[N,1]; SplineCoupling returns[N,2] coordinates with the same record type.
/// Widths/heights are actual adjacent cumulative-knot differences, not logits
/// or log scales. Derivatives include the two boundary slopes of one. No
/// parameter store, optimizer, fitting workflow or sampler is retained separately.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;auto x=n::zeros({2,1});auto heads=n::zeros({2,11});
/// auto out=n::models::RationalQuadraticSpline(x,heads);
/// // value[2,1], logdet[2], widths/heights[2,4], derivatives[2,5].
/// @endcode
/// @par References
/// - Durkan, Bekasov, Murray and Papamakarios(2019), Neural Spline Flows,
/// https://arxiv.org/abs/1906.04032 .
/// @see RationalQuadraticSpline(), SplineCoupling(), SplineDensity()
struct RationalQuadraticSplineOutput {
  Tensor value;       ///< Owning live matching floating[N,1] spline values, or[N,2] coupled coordinates.
  Tensor logdet;      ///< Owning live matching floating[N] signed log absolute input derivative, nats.
  Tensor widths;      ///< Owning live matching floating[N,K] positive actual x-knot differences, summing to2*bound.
  Tensor heights;     ///< Owning live matching floating[N,K] positive actual y-knot differences, summing to2*bound.
  Tensor derivatives; ///< Owning live matching floating[N,K+1] positive knot slopes, boundary entries one.
};

/// @brief Transform scalar rows with a monotone rational-quadratic spline or its closed-form inverse.
///
/// Finite matching Float32/64 CPU input[N,1] and heads[N,3*K-1],N>0,K>=2,
/// arbitrary strides. Heads firstK width logits, nextK height logits, finalK-1
/// interior derivative logits. With span=2*bound, form declared bin sizes as
/// span*(min_bin_fraction+(1-K*min_bin_fraction)*softmax(logits)). Accumulate
/// interior knots from -bound, fix endpoints at+/-bound, and return actual
/// adjacent differences. Interior slopes are min_derivative+softplus(raw+offset),
/// offset=log(expm1(1-min_derivative)); boundary slopes one. rounded_offset=true
/// rounds offset once to Float32 before promotion, preserving the source even
/// for Float64; false(default) retains Double. Zero heads are near identity,
/// with floating rounding; rounded mode can leave small Double deviations.
///
/// Clamp only the internal bin lookup to[-bound,bound]. Assign exact knots to
/// their right bin by counting>=right edges, capped atK-1; select x edges in
/// forward and y edges in inverse. Evaluate the rational-quadratic value and
/// analytic derivative, or solve the bin quadratic with2*c/(-b-sqrt(b*b-4*a*c))
/// for the inverse. Inclusive endpoints use the spline; strict outside tails
/// return the original input and logdet zero. Boundary slopes one give C1
/// identity tails in exact arithmetic. Log-derivative partials can have distinct
/// one-sided values at knots; branch selection is structural and stopped.
/// No root search, numerical determinant or automatic clipping of results.
/// All applicable first/higher input/head paths remain live within the selected
/// branch, including returned sizes/slopes. Owning fields survive collection.
/// No store, RNG, fitting, optimizer, sampler or hidden no-grad. O(N*K*K) work
/// for source-order prefix reductions and O(N*K) field storage; independent calls share no mutable state. Options validate before
/// arithmetic. Reject nonpositive/nonfinite represented sizes, slopes, quadratic
/// discriminant, derivative numerator/denominator or nonfinite computed fields.
/// Floating inverse errors remain; extreme finite parameters may be rejected.
/// @param input Finite floating CPU[N,1] scalar rows, positive N, arbitrary strides.
/// @param heads Matching finite floating CPU[N,3*K-1] logits, integer K>=2.
/// @param inverse False(default) for forward; true for the analytic inverse and negated logdet.
/// @param bound Finite positive symmetric tail bound, default4;2*bound must be representable.
/// @param min_bin_fraction Finite positive lower span fraction per bin, default.01;K*fraction<1.
/// @param min_derivative Finite slope floor strictly between0 and1, default.001.
/// @param rounded_offset False(default) for Double offset; true for source Float32-rounded offset.
/// @return Owning live RationalQuadraticSplineOutput with values, per-row logdet and knot metadata.
/// @throws std::invalid_argument For invalid inputs/options, unsafe geometry or unrepresentable math.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;auto type=n::DType::Float64;
/// auto x=n::from_vector(std::vector<double>{-5,-1,0,2,5},{5,1});
/// auto heads=n::zeros({5,11},type);auto y=n::models::RationalQuadraticSpline(x,heads);
/// auto rec=n::models::RationalQuadraticSpline(y.value,heads,true);
/// // All rows approximately recover x; strict tails +/-5 remain exactly unchanged.
/// @endcode
/// @par References
/// - Durkan et al.(2019), Neural Spline Flows, https://arxiv.org/abs/1906.04032 .
/// The migrated source usesK=4,bound4,fraction.01,floor.001,rounded_offset=true.
/// @see RationalQuadraticSplineOutput, SplineCoupling(), SplineInverse()
inline RationalQuadraticSplineOutput RationalQuadraticSpline(
    const Tensor& input,const Tensor& heads,bool inverse=false,double bound=4.,
    double min_bin_fraction=.01,double min_derivative=.001,bool rounded_offset=false) {
  flow_detail::Matrix(input,1);flow_detail::Matrix(heads);
  if(heads.dtype()!=input.dtype() || heads.size(0)!=input.size(0) || heads.size(1)<5 || heads.size(1)%3!=2)
    throw std::invalid_argument("RationalQuadraticSpline needs matching points[N,1] and heads[N,3K-1],K>=2");
  const auto bins=heads.size(1)/3+1;
  if(bins>std::numeric_limits<std::int32_t>::max())throw std::invalid_argument("Spline bin count exceeds Int32 lookup capacity");
  if(!std::isfinite(bound) || bound<=0 || !std::isfinite(2*bound) ||
     !std::isfinite(min_bin_fraction) || min_bin_fraction<=0 || min_bin_fraction*double(bins)>=1 ||
     !std::isfinite(min_derivative) || min_derivative<=0 || min_derivative>=1)
    throw std::invalid_argument("Invalid rational-quadratic spline bounds or positive bin/slope floors");
  if(input.dtype()==DType::Float32 && (!std::isfinite(float(2*bound)) || float(2*bound)<=0 ||
     float(min_bin_fraction)<=0 || float(min_derivative)<=0))
    throw std::invalid_argument("Spline constants must be representable in the input dtype");
  auto widths=(2*bound)*(min_bin_fraction+(1-double(bins)*min_bin_fraction)*softmax(slice_axis(heads,1,0,bins),1));
  auto heights=(2*bound)*(min_bin_fraction+(1-double(bins)*min_bin_fraction)*softmax(slice_axis(heads,1,bins,2*bins),1));
  auto zeros=slice_axis(heads,1,0,1)*0.,left=zeros-bound,right=zeros+bound,ones=zeros+1.;
  double offset=std::log(std::expm1(1-min_derivative));if(rounded_offset)offset=double(float(offset));
  auto derivatives=concat({ones,min_derivative+softplus(slice_axis(heads,1,2*bins,3*bins-1)+offset),ones},1);
  std::vector<Tensor> xk{left},yk{left};
  for(std::int64_t j=1;j<bins;++j){xk.push_back(-bound+sum(slice_axis(widths,1,0,j),{1},true));yk.push_back(-bound+sum(slice_axis(heights,1,0,j),{1},true));}
  xk.push_back(right);yk.push_back(right);
  const auto x0all=concat(std::vector<Tensor>(xk.begin(),xk.end()-1),1),x1all=concat(std::vector<Tensor>(xk.begin()+1,xk.end()),1);
  const auto y0all=concat(std::vector<Tensor>(yk.begin(),yk.end()-1),1),y1all=concat(std::vector<Tensor>(yk.begin()+1,yk.end()),1);
  widths=x1all-x0all;heights=y1all-y0all;
  for(const auto& field:{widths,heights,derivatives}){
    gan_detail::Finite(field);for(double value:field.contiguous().to_vector<double>())if(value<=0)throw std::invalid_argument("Spline bin sizes and knot slopes must stay positive");}
  const auto edges=inverse?y1all:x1all;
  const auto bounded=minimum(maximum(input,full({},-bound,input.dtype())),full({},bound,input.dtype()));
  auto index=sum(cast(greater_equal(bounded,edges),DType::Int32),{1});index=minimum(index,full({},double(bins-1),DType::Int32));
  const auto mask=cast(one_hot(index,bins,1.,0.),input.dtype());
  const auto x0=sum(x0all*mask,{1},true),y0=sum(y0all*mask,{1},true),w=sum(widths*mask,{1},true),h=sum(heights*mask,{1},true);
  const auto d0=sum(slice_axis(derivatives,1,0,bins)*mask,{1},true),d1=sum(slice_axis(derivatives,1,1,bins+1)*mask,{1},true);
  const auto delta=h/w,curvature=(d0+d1)-2.*delta;
  Tensor theta,value;
  if(inverse){
    const auto distance=bounded-y0,a=distance*curvature+h*(delta-d0),b=h*d0-distance*curvature,c=-(delta*distance);
    const auto discriminant=square(b)-4.*(a*c);gan_detail::Finite(discriminant);
    for(double v:discriminant.contiguous().to_vector<double>())if(v<=0)throw std::invalid_argument("Spline inverse discriminant must stay positive finite");
    theta=(2.*c)/(-b-sqrt(discriminant));value=x0+theta*w;
  }else theta=(bounded-x0)/w;
  const auto complement=1.-theta,blend=theta*complement,denominator=delta+curvature*blend;
  if(!inverse)value=y0+h*(delta*square(theta)+d0*blend)/denominator;
  const auto numerator=square(delta)*((d1*square(theta)+2.*(delta*blend))+d0*square(complement));
  for(const auto& field:{denominator,numerator}){
    gan_detail::Finite(field);for(double v:field.contiguous().to_vector<double>())if(v<=0)throw std::invalid_argument("Spline derivative numerator and denominator must stay positive finite");}
  const auto ld=log(numerator)-2.*log(denominator);
  const auto inside=logical_and(greater_equal(input,-bound),less_equal(input,bound));
  auto result=where(inside,value,input),logdet=reshape(where(inside,inverse?-ld:ld,zeros),{-1});
  gan_detail::Finite(value);gan_detail::Finite(result);gan_detail::Finite(logdet);
  return {result,logdet,widths,heights,derivatives};
}

/// @brief Apply one source four-bin rational-quadratic coupling or its analytic inverse.
///
/// Finite matching Float32/64 CPU input[N,2],N>0, arbitrary strides, block0..3.
/// Preserve coordinate block%2 and condition a1->16 tanh->11 head network on
/// it. Prefix coupling<block>/hidden missing W normal(.3),b=.05, then /heads
/// W normal(.01),b=0;219 stored values in four leaves per block. Transform the
/// other coordinate with RationalQuadraticSpline, K4,bound4,fraction.01,
/// floor.001,rounded_offset=true. Keep raw coordinate order. The determinant
/// is the scalar spline derivative per row; tails preserve the changed input.
/// Returned widths/heights are actual bin sizes[N,4], not log scales/shifts;
/// knot derivatives[N,5]. Only missing leaves initialize, existing leaves never
/// change. No fitting, optimizer, sampler, warmed RNG or hidden no-grad. Owning
/// flat fields retain applicable first/higher input/parameter paths after
/// collection within the selected structural branch. O(16N) conditioner work
/// plus fixed four-bin spline/activations; synchronize shared stores. Inputs
/// validate before leaves; stored-leaf/math failure may follow partial creation.
/// Reject nonfinite stored/computed fields and unrepresentable spline math.
/// @param parameters Caller-owned matching floating store, full source seed9803.
/// @param input Finite matching Float32/64 CPU[N,2] coordinates, positive rows.
/// @param block Source index0..3; conditions on raw coordinate block%2.
/// @param inverse False(default) for forward; true for inverse on the same unchanged coordinate.
/// @return Owning live RationalQuadraticSplineOutput with value[N,2],logdet[N],bin metadata.
/// @throws std::invalid_argument For invalid inputs/block/store or unrepresentable spline math.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9803);
/// auto x=n::ones({3,2});auto y=n::models::SplineCoupling(p,x,0);
/// auto rec=n::models::SplineCoupling(p,y.value,0,true);
/// // rec.value approximates x; untouched coordinate0 is preserved exactly.
/// @endcode
/// @par References
/// - Durkan et al.(2019), Neural Spline Flows, https://arxiv.org/abs/1906.04032 .
/// Source compact teaching adaptation uses four bins and small tanh conditioners.
/// @see RationalQuadraticSpline(), SplineDensity(), SplineInverse()
inline RationalQuadraticSplineOutput SplineCoupling(ParameterStore& parameters,const Tensor& input,int block,bool inverse=false) {
  flow_detail::Matrix(input,2);
  if(block<0 || block>3 || input.dtype()!=parameters.dtype() || input.size(0)>std::numeric_limits<std::int64_t>::max()/16)
    throw std::invalid_argument("SplineCoupling needs block0..3 and matching safe floating matrices");
  const auto name="coupling"+std::to_string(block);const int condition=block%2;
  const auto a=slice_axis(input,1,condition,condition+1),b=slice_axis(input,1,1-condition,2-condition);
  const auto hidden=nice_detail::Dense(parameters,name+"/hidden",a,1,16,.3,.05,true);
  const auto heads=nice_detail::Dense(parameters,name+"/heads",hidden,16,11,.01,0);
  auto out=RationalQuadraticSpline(b,heads,inverse,4.,.01,.001,true);
  out.value=condition?concat({out.value,a},1):concat({a,out.value},1);gan_detail::Finite(out.value);return out;
}

/// @brief Own all compact spline-flow stages, actual bin geometry and density intermediates.
///
/// Twenty-three flat Tensor handles retain applicable first/higher input/
/// parameter graphs and storage after parent/store collection. Each coupling
/// transforms one raw coordinate; widths/heights are actual positive bin sizes,
/// derivatives include boundary slopes one. No nested record/vector ownership,
/// fitting workflow, optimizer or sampler. Structural bin choices are stopped.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9803);
/// auto out=n::models::SplineDensity(p,n::zeros({2,2}));
/// // coupling0..3/latent[2,2], widths/heights[2,4], derivatives[2,5], logdets[2].
/// @endcode
/// @par References
/// - Durkan et al.(2019), Neural Spline Flows, https://arxiv.org/abs/1906.04032 .
/// @see SplineDensity(), SplineCoupling(), RationalQuadraticSpline()
struct SplineDensityOutput {
  Tensor coupling0; ///< Owning live matching floating[N,2] stage0 transformed coordinates.
  Tensor coupling_logdet0; ///< Owning live matching floating[N] stage0 forward log absolute input derivative, nats.
  Tensor coupling_widths0; ///< Owning live matching floating[N,4] stage0 positive actual x-bin widths, not log scales.
  Tensor coupling_heights0; ///< Owning live matching floating[N,4] stage0 positive actual y-bin heights, not affine shifts.
  Tensor coupling_derivatives0; ///< Owning live matching floating[N,5] stage0 positive knot derivatives, boundary entries one.
  Tensor coupling1; ///< Owning live matching floating[N,2] stage1 transformed coordinates.
  Tensor coupling_logdet1; ///< Owning live matching floating[N] stage1 forward log absolute input derivative, nats.
  Tensor coupling_widths1; ///< Owning live matching floating[N,4] stage1 positive actual x-bin widths, not log scales.
  Tensor coupling_heights1; ///< Owning live matching floating[N,4] stage1 positive actual y-bin heights, not affine shifts.
  Tensor coupling_derivatives1; ///< Owning live matching floating[N,5] stage1 positive knot derivatives, boundary entries one.
  Tensor coupling2; ///< Owning live matching floating[N,2] stage2 transformed coordinates.
  Tensor coupling_logdet2; ///< Owning live matching floating[N] stage2 forward log absolute input derivative, nats.
  Tensor coupling_widths2; ///< Owning live matching floating[N,4] stage2 positive actual x-bin widths, not log scales.
  Tensor coupling_heights2; ///< Owning live matching floating[N,4] stage2 positive actual y-bin heights, not affine shifts.
  Tensor coupling_derivatives2; ///< Owning live matching floating[N,5] stage2 positive knot derivatives, boundary entries one.
  Tensor coupling3; ///< Owning live matching floating[N,2] stage3 transformed coordinates.
  Tensor coupling_logdet3; ///< Owning live matching floating[N] stage3 forward log absolute input derivative, nats.
  Tensor coupling_widths3; ///< Owning live matching floating[N,4] stage3 positive actual x-bin widths, not log scales.
  Tensor coupling_heights3; ///< Owning live matching floating[N,4] stage3 positive actual y-bin heights, not affine shifts.
  Tensor coupling_derivatives3; ///< Owning live matching floating[N,5] stage3 positive knot derivatives, boundary entries one.
  Tensor unscaled; ///< Live floating[N,2] alias of coupling3 before global scaling.
  Tensor latent;   ///< Owning live matching floating[N,2] globally scaled Gaussian coordinates.
  Tensor logdet;   ///< Owning live matching floating[N] full forward log absolute determinant, nats.
};

/// @brief Map observations through four source rational-quadratic couplings and global scale.
///
/// Finite matching Float32/64 CPU input[N,2],N>0, arbitrary strides. Apply
/// SplineCoupling blocks0..3, accumulate per-row logdet from input-linked zeros,
/// then FlowDiagonalScale. Each block is1->16 tanh->11 heads for four bins on
///[-4,4] with source Float32-rounded derivative offset and C1 identity tails.
/// Four219-value conditioners plus log_scale[2] give878 stored values/17 leaves.
/// Source seed9803 warms hidden W/b then heads W/b in forward block order,
/// then global scale. Only missing leaves initialize; warmed inference changes
/// no leaves and consumes no RNG. Owns all live stage/bin/density fields after
/// collection; applicable first/higher paths remain within selected structural
/// branches. No fitting, optimizer, sampler, root search or hidden no-grad.
/// O(64N) conditioner work plus four fixed spline stages and activations;
/// synchronize shared stores. Inputs validate before leaves; stored-leaf/math
/// failure may follow partial creation. Reject nonfinite/unrepresentable math;
/// both global scale directions must stay positive finite. Floating roundtrip
/// errors remain. Source archival scale/shift diagnostics correspond to the
/// public coupling_widths/coupling_heights fields, not log scales or shifts.
/// @param parameters Caller-owned matching floating store, source seed9803.
/// @param input Finite matching Float32/64 CPU[N,2] observed coordinates, positive rows.
/// @return Owning live SplineDensityOutput with23 flat stage/bin/density fields.
/// @throws std::invalid_argument For invalid inputs/store or unrepresentable spline/global-scale math.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9803);
/// auto path=n::models::SplineDensity(p,n::zeros({3,2}));
/// auto density=n::models::NormalFlowDensity(path.latent,path.logdet,true);
/// //878 stored values/17 leaves; density.loss is mean source-compatible NLL, nats/row.
/// @endcode
/// @par References
/// - Durkan et al.(2019), Neural Spline Flows, https://arxiv.org/abs/1906.04032 .
/// Four-bin2-D teaching adaptation; model fitting and Gaussian draws belong to callers.
/// @see SplineDensityOutput, SplineInverse(), SplineCoupling(), NormalFlowDensity()
inline SplineDensityOutput SplineDensity(ParameterStore& parameters,const Tensor& input) {
  flow_detail::Matrix(input,2);
  if(input.dtype()!=parameters.dtype())throw std::invalid_argument("Spline density input/store dtypes must match");
  auto ld=reshape(slice_axis(input,1,0,1),{-1})*0.;
  auto a=SplineCoupling(parameters,input,0);ld=ld+a.logdet;
  auto b=SplineCoupling(parameters,a.value,1);ld=ld+b.logdet;
  auto c=SplineCoupling(parameters,b.value,2);ld=ld+c.logdet;
  auto d=SplineCoupling(parameters,c.value,3);ld=ld+d.logdet;
  auto scaled=FlowDiagonalScale(parameters,d.value,ld);
  return {a.value,a.logdet,a.widths,a.heights,a.derivatives,
          b.value,b.logdet,b.widths,b.heights,b.derivatives,
          c.value,c.logdet,c.widths,c.heights,c.derivatives,
          d.value,d.logdet,d.widths,d.heights,d.derivatives,
          d.value,scaled.value,scaled.logdet};
}

/// @brief Invert compact spline density with closed quadratic roots and its own path determinant.
///
/// Finite matching Float32/64 CPU latent[N,2],N>0, arbitrary strides. Unscale
/// globally, then reverse SplineCoupling blocks3..0, recomputing heads on each
/// unchanged conditioning coordinate. Select inverse bins with y edges and use
/// the analytic quadratic root; accumulate each inverse logdet along its own
/// supplied path. No cached forward points or root search. A forward-warmed
/// store reuses878 values/17 leaves without writes/RNG; missing leaves initialize
/// global scale then reversed blocks, so warm SplineDensity for source seed9803
/// ordering. Owns live value/logdet after collection; applicable first/higher
/// paths survive within structural branches. No fitting, optimizer, sampler or
/// hidden no-grad. O(64N) conditioner work plus four fixed spline inverses and
/// activations; synchronize shared stores. Validate inputs before leaves;
/// stored-leaf/math failure may follow partial creation. Both global scale
/// directions must remain positive finite. Nonfinite/unrepresentable math
/// rejects without a numerical-inverse fallback; floating roundtrip error remains.
/// @param parameters Caller-owned matching floating store, preferably forward-warmed.
/// @param latent Finite matching Float32/64 CPU[N,2] caller-supplied Gaussian coordinates.
/// @return Owning live FlowScaleOutput with observation value[N,2] and inverse-path logdet[N].
/// @throws std::invalid_argument For invalid inputs/store or unrepresentable spline/global-scale math.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9803);
/// auto x=n::ones({2,2});auto path=n::models::SplineDensity(p,x);
/// auto rec=n::models::SplineInverse(p,path.latent);
/// // rec.value approximately recovers x; rec.logdet+path.logdet approximately cancels per row.
/// @endcode
/// @par References
/// - Durkan et al.(2019), Neural Spline Flows, https://arxiv.org/abs/1906.04032 .
/// @see SplineDensity(), SplineCoupling(), RationalQuadraticSpline(), FlowScaleOutput
inline FlowScaleOutput SplineInverse(ParameterStore& parameters,const Tensor& latent) {
  flow_detail::Matrix(latent,2);
  if(latent.dtype()!=parameters.dtype() || latent.size(0)>std::numeric_limits<std::int64_t>::max()/16)
    throw std::invalid_argument("Spline inverse needs matching safe floating matrices");
  auto zero=reshape(slice_axis(latent,1,0,1),{-1})*0.;auto out=FlowDiagonalScale(parameters,latent,zero,true);
  for(int i=3;i>=0;--i){auto stage=SplineCoupling(parameters,out.value,i,true);out.value=stage.value;out.logdet=out.logdet+stage.logdet;}
  gan_detail::Finite(out.logdet);return out;
}
} // namespace datamunge::nn::models
