#pragma once
/// @file generative_metrics.hpp
/// @brief Detached Float64 low-rank Gaussian Frechet and probability-space Inception diagnostics.
#include "datamunge/nn/gan_models.hpp"
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/linalg.hpp"

namespace datamunge::nn::models {
/// @brief Private validation for supplied features and normalized class probabilities.
namespace generative_metric_detail {
/// @brief Validate a nonempty finite floating CPU matrix with a minimum pool size.
/// @param x Supplied Float32/64 CPU[N,F], positive feature width.
/// @param minimum Minimum allowed positive rows.
/// @throws std::invalid_argument For invalid geometry, dtype or values.
/// @throws std::logic_error For non-CPU storage.
inline void Matrix(const Tensor& x,std::int64_t minimum) {
  gan_detail::Finite(x);
  if(x.dim()!=2 || x.size(0)<minimum || x.size(1)<1)
    throw std::invalid_argument("Generative metric requires a finite nonempty floating matrix");
}
/// @brief Require probabilities in[0,1] with at least two classes and row sums near one.
/// @param p Supplied finite floating CPU[N,C], N>0,C>1.
/// @throws std::invalid_argument For invalid geometry, range or row sum outside3e-6.
/// @throws std::logic_error For non-CPU storage.
inline void Probability(const Tensor& p) {
  Matrix(p,1);
  if(p.size(1)<2)throw std::invalid_argument("Inception score requires at least two classes");
  for(double v:p.contiguous().to_vector<double>())
    if(v<0 || v>1)throw std::invalid_argument("Inception score requires probabilities, not logits");
  NoGradGuard guard;
  auto sums=sum(cast(p.detach(),DType::Float64),{1});
  for(double v:sums.to_vector<double>())
    if(std::abs(v-1)>3e-6)throw std::invalid_argument("Inception score requires row-normalized probabilities");
}
} // namespace generative_metric_detail

/// @brief Own seven detached Gaussian-distance diagnostics, including the low-rank singular spectrum.
///
/// Float64 scalar fields and cross_singular_values[min(Nreal,Nfake)] retain
/// independent storage beyond parent/input lifetimes. No fitting, sampling,
/// extractor or gradient path is retained. This is a moment comparison, not a
/// classifier accuracy or a test of equality of the full feature distributions.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <datamunge/nn/generative_metrics.hpp>
/// namespace n=dm::nn;
/// auto x=n::tensor({-1.,0.,1.,0.},{2,2});
/// auto out=n::models::GaussianFrechet(x,x); // out.fid is zero up to round-off.
/// @endcode
/// @par References
/// - Heusel et al.(2017), GANs Trained by a Two Time-Scale Update Rule Converge
/// to a Local Nash Equilibrium, https://arxiv.org/abs/1706.08500 .
/// @see GaussianFrechet(), InceptionScore()
struct GaussianFrechetOutput {
  Tensor fid;                   ///< Owning detached Float64 scalar max(raw_fid,0).
  Tensor raw_fid;               ///< Owning detached scalar mean_term+trace_a+trace_b-2*cross_trace.
  Tensor mean_term;             ///< Owning detached scalar squared Euclidean feature-mean distance.
  Tensor trace_a;               ///< Owning detached scalar real unbiased covariance trace.
  Tensor trace_b;               ///< Owning detached scalar fake unbiased covariance trace.
  Tensor cross_trace;           ///< Owning detached scalar sum of cross singular values.
  Tensor cross_singular_values; ///< Owning detached Float64[min(Nreal,Nfake)] spectrum.
};

/// @brief Compare unbiased feature moments by the exact low-rank Gaussian Frechet formula.
///
/// Finite Float32/64 CPU real[N,F],fake[M,F], N,M>=2,F>0; unequal pool sizes
/// and dtypes/strides allowed. Detach and promote to Float64, center each pool
/// and divide by sqrt(its row count-1). Compute thin SVD of the[N,M] cross
/// product, rather than forming two[F,F] covariances. The singular-value sum
/// equals the Gaussian covariance cross trace, including rank-deficient pools.
/// Raw distance is squared mean difference+both covariance traces-2*cross trace.
/// Clamp only nonnegative round-off: raw<-1e-7 throws; a tiny negative result
/// becomes zero. This fixed source tolerance is absolute, not scale-adaptive.
/// Work O(NMF+NM*min(N,M)); cross storage O(NM), plus centered inputs/SVD.
/// No data/model mutation, fitting, sampling, hidden extractor or autograd path.
/// Independent calls share no mutable state. FID terminology applies only when
/// features actually use the specified pretrained Inception protocol; arbitrary
/// features still define a Gaussian Frechet moment distance.
/// @param real Finite floating CPU[N,F] reference feature pool, at least two rows.
/// @param fake Finite floating CPU[M,F] candidate pool with matching positive width.
/// @return Owning detached seven-field Float64 GaussianFrechetOutput.
/// @throws std::invalid_argument For invalid inputs, incompatible widths/products or raw<-1e-7.
/// @throws std::logic_error For non-CPU storage or unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <datamunge/nn/generative_metrics.hpp>
/// namespace n=dm::nn;auto t=n::DType::Float64;
/// auto a=n::tensor({-1.,-1.,-1.,1.,1.,-1.,1.,1.},{4,2},t);
/// auto b=n::tensor({-2.,-1.,0.,2.,1.,1.},{3,2},t);
/// auto out=n::models::GaussianFrechet(a,b); // Unequal pools; no extractor or fit.
/// @endcode
/// @par References
/// - Heusel et al.(2017), GANs Trained by a Two Time-Scale Update Rule Converge
/// to a Local Nash Equilibrium, https://arxiv.org/abs/1706.08500 . Low-rank form
/// uses the nuclear norm of the two centered data factors' cross product.
/// @see GaussianFrechetOutput, InceptionScore(), datamunge::nn::svd()
inline GaussianFrechetOutput GaussianFrechet(const Tensor& real,const Tensor& fake) {
  generative_metric_detail::Matrix(real,2);
  generative_metric_detail::Matrix(fake,2);
  if(real.size(1)!=fake.size(1) || real.size(0)>INT64_MAX/fake.size(0))
    throw std::invalid_argument("Gaussian Frechet needs aligned widths and representable cross product");
  NoGradGuard guard;
  auto a=cast(real.detach(),DType::Float64),b=cast(fake.detach(),DType::Float64);
  auto am=mean(a,{0}),bm=mean(b,{0});
  auto av=(a-am)/std::sqrt(double(a.size(0)-1)),bv=(b-bm)/std::sqrt(double(b.size(0)-1));
  auto decomposition=svd(matmul(av,bv,false,true),false);
  GaussianFrechetOutput o;
  o.mean_term=sum(square(am-bm));o.trace_a=sum(square(av));o.trace_b=sum(square(bv));
  o.cross_singular_values=decomposition.s;o.cross_trace=sum(o.cross_singular_values);
  o.raw_fid=o.mean_term+(o.trace_a+o.trace_b)-2.0*o.cross_trace;
  gan_detail::Finite(o.raw_fid);
  if(o.raw_fid.item() < -1e-7)throw std::invalid_argument("Gaussian Frechet has more than round-off negative distance");
  o.fid=maximum(o.raw_fid,scalar_like(0.0,o.raw_fid));
  return o;
}

/// @brief Measure exp(mean conditional-to-marginal class KL) from supplied probabilities.
///
/// Finite floating CPU[N,C], N>0,C>1; signed logits rejected, values in[0,1],
/// row sums within3e-6 of one. Arbitrary strides allowed. Stop inputs and cast
/// to Float64; marginal=mean over rows. Evaluate exp(mean(sum(p*(logp-logm))))
/// with both log arguments floored at1e-300, so zero classes contribute zero.
/// Does not renormalize rows. No extractor, sampling, fitting or state mutation;
/// output is an owning detached Float64 scalar. Work/storage O(NC), independent
/// calls share no mutable state. Uniform uncertain and fully class-collapsed
/// distributions both score one; a balanced confident C-class pool scores C.
/// This score is not accuracy, recall, image uniqueness or proof against copying.
/// @param probability Finite normalized floating CPU[N,C] class probabilities.
/// @return Owning detached Float64 scalar Inception score.
/// @throws std::invalid_argument For invalid geometry, range, normalization or nonfinite result.
/// @throws std::logic_error For non-CPU storage or unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <datamunge/nn/generative_metrics.hpp>
/// namespace n=dm::nn;
/// auto p=n::tensor({1.,0.,0.,1.},{2,2});
/// auto score=n::models::InceptionScore(p); // Approximately2; no network is run.
/// @endcode
/// @par References
/// - Salimans et al.(2016), Improved Techniques for Training GANs,
/// https://arxiv.org/abs/1606.03498 .
/// @see InceptionScoreSplits(), GaussianFrechet()
inline Tensor InceptionScore(const Tensor& probability) {
  generative_metric_detail::Probability(probability);
  NoGradGuard guard;
  auto p=cast(probability.detach(),DType::Float64),marginal=mean(p,{0});
  auto logp=log(maximum(p,scalar_like(1e-300,p))),logm=log(maximum(marginal,scalar_like(1e-300,marginal)));
  auto score=exp(mean(sum(p*(logp-logm),{1})));
  gan_detail::Finite(score);return score;
}

/// @brief Own contiguous-split Inception scores, their mean and population standard deviation.
///
/// Three owning detached Float64 fields; population_std uses divisor splits,
/// not splits-1. These splits partition one supplied pool, so their spread is
/// not a confidence interval over independently fitted generators.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <datamunge/nn/generative_metrics.hpp>
/// namespace n=dm::nn;
/// auto out=n::models::InceptionScoreSplits(n::full({8,4},.25),4);
/// // scores[4] are1, mean1, population_std0.
/// @endcode
/// @see InceptionScoreSplits(), InceptionScore()
struct InceptionSplitScores {
  Tensor scores;         ///< Owning detached Float64[splits] scores in source row order.
  Tensor mean;           ///< Owning detached Float64 scalar arithmetic split-score mean.
  Tensor population_std; ///< Owning detached Float64 scalar population spread, divisor splits.
};

/// @brief Score consecutive source-order partitions without shuffling or dropping any row.
///
/// Uses InceptionScore's finite normalized floating CPU[N,C] contract. For S
/// positive splits<=N, boundaries are floor(i*N/S), i=0..S, so sizes may differ
/// by one. All rows occur exactly once. Boundaries use overflow-free quotient/
/// remainder accumulation. No RNG, fitting, extractor, hidden permutation,
/// input mutation or gradient path. Mean and population spread summarize scores
/// from this one ordered pool; they depend on row ordering. Work/storage O(NC+S).
/// @param probability Supplied finite normalized floating CPU[N,C], positive rows,C>1.
/// @param splits Positive partition count S<=N, default4 as the compact source diagnostics.
/// @return Owning detached scores[S], scalar mean and population_std.
/// @throws std::invalid_argument For invalid probabilities, split count or nonfinite result.
/// @throws std::logic_error For non-CPU storage or unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <datamunge/nn/generative_metrics.hpp>
/// namespace n=dm::nn;
/// auto p=n::tensor({1.,0.,0.,1.,1.,0.,0.,1.},{4,2});
/// auto out=n::models::InceptionScoreSplits(p,2); // scores(2,2), mean2, spread0.
/// @endcode
/// @par References
/// - Salimans et al.(2016), Improved Techniques for Training GANs,
/// https://arxiv.org/abs/1606.03498; original contiguous-partition recipe:
/// https://github.com/openai/improved-gan/blob/master/inception_score/model.py .
/// @see InceptionSplitScores, InceptionScore()
inline InceptionSplitScores InceptionScoreSplits(const Tensor& probability,std::int64_t splits=4) {
  generative_metric_detail::Probability(probability);
  auto rows=probability.size(0);
  if(splits<1 || splits>rows)throw std::invalid_argument("Inception split count must lie in[1,N]");
  NoGradGuard guard;
  std::vector<Tensor> values;std::int64_t start=0,remainder=0;
  for(std::int64_t i=0;i<splits;++i) {
    auto count=rows/splits;remainder+=rows%splits;
    if(remainder>=splits){++count;remainder-=splits;}
    values.push_back(InceptionScore(slice_axis(probability,0,start,start+count)));start+=count;
  }
  InceptionSplitScores o;o.scores=stack(values,0);o.mean=mean(o.scores);
  o.population_std=sqrt(mean(square(o.scores-o.mean)));return o;
}
} // namespace datamunge::nn::models
