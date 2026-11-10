#pragma once
/// @file flow_study.hpp
/// @brief Explicit three-family flow-study observations, reference densities and sample diagnostics.
#include "datamunge/nn/normalizing_flows.hpp"
#include <array>
#include <limits>

namespace datamunge::nn::models {
/// @brief Internal scalar rules of the original three-family teaching protocol.
namespace flow_study_detail {
/// @brief Validate the explicit family selector without accepting an implicit fallback.
/// @param kind Zero(banana), one(eight Gaussians) or two(checkerboard).
/// @throws std::invalid_argument For any other selector.
inline void Kind(int kind) {
  if(kind<0 || kind>2)throw std::invalid_argument("Flow study kind must be 0, 1 or 2");
}
/// @brief Evaluate the standard-normal CDF using the source erfc formula.
/// @param z Finite Double standard-normal coordinate.
/// @return .5*erfc(-z/sqrt(2)), including rounded zero/one in extreme tails.
inline double Phi(double z) {return .5*std::erfc(-z/std::sqrt(2.));}
/// @brief Test the eight allowed half-open unit cells in [-2,2) squared.
/// @param x Finite horizontal coordinate.
/// @param y Finite vertical coordinate.
/// @return True for even floor(x+2)+floor(y+2) parity inside the bounding square.
inline bool Allowed(double x,double y) {
  if(x < -2 || x >= 2 || y < -2 || y >= 2)return false;
  return (int(std::floor(x+2))+int(std::floor(y+2)))%2==0;
}
} // namespace flow_study_detail

/// @brief Own detached two-dimensional observations and their audited component labels.
///
/// Flat Tensor values own storage after parent/input collection. Components are
/// diagnostics, never extra model input features. No RNG, fit or input graph is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto out=dm::nn::models::FlowStudyPoints(dm::nn::zeros({1,3}),1);
/// // out.points is Float32 [[-2,approximately 0]]; out.components is Int32 [4].
/// @endcode
/// @see FlowStudyPoints(), FlowStudyLogProbability()
struct FlowStudyPointsOutput {
  Tensor points; ///< Owning detached contiguous Float32/64 CPU[N,2] observations.
  Tensor components; ///< Owning detached contiguous Int32 CPU[N]; -1 for banana, otherwise 0..7.
};

/// @brief Transform explicit Gaussian triples into banana, ring-mixture or checkerboard observations.
///
/// Accept finite Float32/64 CPU standard[N,3], N>0 and arbitrary strides.
/// Promote each value to Double before scalar math, then cast observations once
/// to dtype(default Float32). kind=0: (u,v+.5*(u*u-1)), component=-1.
/// Otherwise c=min(7,int(8*Phi(w))), with Phi(z)=.5*erfc(-z/sqrt(2)).
/// kind=1: (2*cos(2*pi*c/8)+.15*u,2*sin(2*pi*c/8)+.15*v).
/// kind=2: row=c/2, col=2*(c%2)+row%2, then (col-2+Phi(u),row-2+Phi(v)).
/// Extreme CDF tails can round to cell boundaries; results are not clipped or
/// forced back into support. In particular, final Float32 rounding can create
/// boundary points even though the ideal continuous law assigns them zero mass.
/// Caller owns Gaussian generation, the full flat draw stream, splits and IDs;
/// three-dimensional row grouping differs from a two-dimensional banana bank.
/// No sampling, fitting, mutation or gradients. O(N) work/storage; independent
/// calls share no mutable state. Computed nonfinite coordinates are rejected.
/// @param standard Finite floating CPU[N,3] explicit standard-normal triples.
/// @param kind Required family: 0 banana, 1 eight Gaussians, 2 checkerboard.
/// @param dtype Float32(default) or Float64 final observation dtype.
/// @return Owning detached FlowStudyPointsOutput, with points[N,2] and components[N].
/// @throws std::invalid_argument For invalid geometry/storage/type/options or nonfinite results.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto u=dm::nn::from_vector(std::vector<double>{0,0,0,2,1,-1},{2,3});
/// auto out=dm::nn::models::FlowStudyPoints(u,0);
/// // points=[[0,-.5],[2,2.5]], components=[-1,-1]; no graph is retained.
/// @endcode
/// @par References
/// - Original flow_density_2d teaching protocol: explicit three-coordinate
/// Gaussian bank, eight equal ring components and eight alternating unit cells.
/// @see FlowStudyPointsOutput, BananaPoints(), GaussianNoise(), FlowStudyLogProbability()
inline FlowStudyPointsOutput FlowStudyPoints(
    const Tensor& standard,int kind,DType dtype=DType::Float32) {
  flow_detail::Matrix(standard,3);flow_study_detail::Kind(kind);
  if(dtype!=DType::Float32 && dtype!=DType::Float64)
    throw std::invalid_argument("Flow study observations require Float32 or Float64 output");
  NoGradGuard guard;
  const auto base=cast(standard,DType::Float64).to_vector<double>();
  const auto n=standard.size(0);std::vector<double> points(std::size_t(n)*2);
  std::vector<std::int32_t> components(std::size_t(n),-1);
  const double pi=std::acos(-1.);
  for(std::int64_t i=0;i<n;++i) {
    const double u=base[i*3],v=base[i*3+1];
    if(kind==0) {points[i*2]=u;points[i*2+1]=v+.5*(u*u-1);}
    else {
      const int c=std::min(7,int(8*flow_study_detail::Phi(base[i*3+2])));components[i]=c;
      if(kind==1) {points[i*2]=2*std::cos(2*pi*c/8)+.15*u;points[i*2+1]=2*std::sin(2*pi*c/8)+.15*v;}
      else {const int row=c/2,col=2*(c%2)+row%2;points[i*2]=col-2+flow_study_detail::Phi(u);points[i*2+1]=row-2+flow_study_detail::Phi(v);}
    }
  }
  auto out=cast(from_vector(points,{n,2}),dtype);gan_detail::Finite(out);
  return {out,from_vector(components,{n})};
}

/// @brief Evaluate detached Double reference log densities for the three flow-study laws.
///
/// For finite floating CPU points[N,2], N>0 and arbitrary strides, kind=0 uses
/// the unit-determinant banana inverse; kind=1 the equal eight-component mixture
/// of isotropic Gaussians with radius-two centers and standard deviation .15,
/// evaluated by Double log-sum-exp. kind=2 is -log(8) in allowed checkerboard
/// cells and negative infinity elsewhere. Cells occupy [-2,2) squared, with even
/// floor(x+2)+floor(y+2) parity; boundaries use those exact half-open rules.
/// Negative infinity for off-support kind=2 is a legitimate zero PDF, not an
/// invalid result. Nonfinite intermediate/results for kinds0/1 are rejected.
/// This is an analytic reference-law evaluator: no autograd, fit, sampling,
/// mutation or RNG; output owns stopped storage after collection. O(N) work and
/// storage, eight terms per row for kind1; independent calls are thread-safe.
/// @param points Finite Float32/64 CPU[N,2] observation coordinates.
/// @param kind Required family: 0 banana, 1 eight Gaussians, 2 checkerboard.
/// @return Owning detached contiguous Float64 CPU[N] log probabilities in nats.
/// @throws std::invalid_argument For invalid inputs/kind or unsafe kind0/1 calculations.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto x=dm::nn::from_vector(std::vector<double>{-1.5,-1.5,-.5,-1.5},{2,2});
/// auto logp=dm::nn::models::FlowStudyLogProbability(x,2);
/// // [-log(8),-infinity]; corresponding PDFs are [1/8,0].
/// @endcode
/// @par References
/// - Original flow_density_2d analytic benchmark laws; mixture weights are 1/8,
/// not fitted from component counts in a finite sample.
/// @see FlowStudyPoints(), BananaLogProbability(), FlowStudySampleDiagnostics()
inline Tensor FlowStudyLogProbability(const Tensor& points,int kind) {
  flow_detail::Matrix(points,2);flow_study_detail::Kind(kind);NoGradGuard guard;
  const auto x=cast(points,DType::Float64).to_vector<double>();const auto n=points.size(0);
  std::vector<double> out(std::size_t(n),0.);const double pi=std::acos(-1.);
  for(std::int64_t i=0;i<n;++i) {
    const double a=x[i*2],b=x[i*2+1];
    if(kind==0) {const double v=b-.5*(a*a-1);out[i]=-std::log(2*pi)-.5*(a*a+v*v);}
    else if(kind==2)out[i]=flow_study_detail::Allowed(a,b)?-std::log(8.):-std::numeric_limits<double>::infinity();
    else {
      std::array<double,8> terms;double largest=-std::numeric_limits<double>::infinity();
      for(int j=0;j<8;++j) {
        const double u=a-2*std::cos(2*pi*j/8),v=b-2*std::sin(2*pi*j/8);
        terms[j]=-std::log(8*2*pi*.15*.15)-.5*(u*u+v*v)/(.15*.15);largest=std::max(largest,terms[j]);
      }
      double total=0;for(double v:terms)total+=std::exp(v-largest);
      out[i]=largest+std::log(total);
    }
    if(kind!=2 && !std::isfinite(out[i]))throw std::invalid_argument("Flow study reference density overflow");
  }
  return from_vector(out,{n});
}

/// @brief Own all three flow-study sample diagnostics, independent of fitted family.
///
/// Scalar fields and the owning count Tensor survive parent/input collection.
/// No sample graph, fitted model, RNG or reference observations are retained.
/// Eight covered modes only means each count is positive; it does not establish
/// concentration near the modes. Support fraction and curvature are separate checks.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto x=dm::nn::from_vector(std::vector<double>{2,0,-2,0},{2,2});
/// auto out=dm::nn::models::FlowStudySampleDiagnostics(x);
/// // fraction_near_modes=1, modes_covered=2, banana_curvature_moment=0.
/// @endcode
/// @see FlowStudySampleDiagnostics()
struct FlowStudyDiagnosticsOutput {
  Tensor mode_counts; ///< Owning detached contiguous Int32 CPU[8] nearest-mode counts at distance <=.45.
  double checkerboard_support_fraction=0; ///< Fraction of rows in allowed half-open checkerboard cells.
  double fraction_near_modes=0; ///< Fraction within .45 of a nearest radius-two ring center, including ties.
  int modes_covered=0; ///< Number of positive counts, in [0,8]; no concentration guarantee.
  double banana_curvature_moment=0; ///< Double mean y*(x*x-1); true banana expectation is one.
};

/// @brief Measure checker support, eight-ring mode concentration and banana curvature for explicit samples.
///
/// Accept finite Float32/64 CPU samples[N,2], 0<N<=INT32_MAX, arbitrary strides.
/// Compute all diagnostics for every input family. Each ring center is
/// (2*cos(2*pi*j/8),2*sin(2*pi*j/8)); nearest uses Double hypot with strict '<'
/// updates, so exact ties choose the lowest index. Count if distance<=.45
/// (three component standard deviations). Checker support uses the same
/// half-open parity rule as FlowStudyLogProbability(kind=2). Curvature is
/// mean(y*(x*x-1)), with theoretical banana expectation one. Source-compatible
/// serial Double accumulation divides each contribution by N before adding.
/// No gradients, fitting, resampling, mutation or RNG. O(8N) work, O(N) copied
/// coordinates plus eight counts; unsafe curvature arithmetic is rejected.
/// @param samples Finite floating CPU[N,2] explicit observations or generated samples.
/// @return Owning detached FlowStudyDiagnosticsOutput with counts and four scalar diagnostics.
/// @throws std::invalid_argument For invalid inputs, too many rows or nonfinite curvature calculations.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto out=dm::nn::models::FlowStudySampleDiagnostics(dm::nn::zeros({1,2}));
/// // checkerboard_support_fraction=1, fraction_near_modes=0, modes_covered=0.
/// @endcode
/// @par References
/// - Original flow_density_2d protocol's all-family sample diagnostics, with
/// three-sigma nearest-center threshold, support parity and banana moment.
/// @see FlowStudyDiagnosticsOutput, FlowStudyLogProbability(), FlowStudyPoints()
inline FlowStudyDiagnosticsOutput FlowStudySampleDiagnostics(const Tensor& samples) {
  flow_detail::Matrix(samples,2);
  const auto n=samples.size(0);
  if(n>std::numeric_limits<std::int32_t>::max())throw std::invalid_argument("Flow study mode counts require N<=INT32_MAX");
  NoGradGuard guard;const auto x=cast(samples,DType::Float64).to_vector<double>();
  std::vector<std::int32_t> counts(8,0);FlowStudyDiagnosticsOutput out;const double pi=std::acos(-1.);
  for(std::int64_t i=0;i<n;++i) {
    const double a=x[i*2],b=x[i*2+1],moment=b*(a*a-1)/double(n);
    if(!std::isfinite(moment))throw std::invalid_argument("Flow study curvature moment overflow");
    out.checkerboard_support_fraction+=flow_study_detail::Allowed(a,b)/double(n);
    out.banana_curvature_moment+=moment;
    double nearest=std::numeric_limits<double>::infinity();int mode=0;
    for(int j=0;j<8;++j) {
      const double distance=std::hypot(a-2*std::cos(2*pi*j/8),b-2*std::sin(2*pi*j/8));
      if(distance<nearest) {nearest=distance;mode=j;}
    }
    if(nearest<=.45)++counts[mode];
  }
  if(!std::isfinite(out.banana_curvature_moment))throw std::invalid_argument("Flow study curvature sum overflow");
  std::int64_t total=0;for(auto count:counts) {out.modes_covered+=count>0;total+=count;}
  out.fraction_near_modes=double(total)/double(n);out.mode_counts=from_vector(counts,{8});return out;
}
} // namespace datamunge::nn::models
