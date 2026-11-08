/// @file bootstrap.cpp
/// @brief Shared finite bootstrap summaries and resampling with default or caller-owned random streams.
#include <algorithm>
#include <cmath>
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/bootstrap.hpp>
#include <limits>
#include <random>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Private finite-value checks and stable bootstrap summary primitives.
namespace {
/// @brief Require a finite statistic before sorting or accumulating it.
/// @param value Original, resampled or jackknife statistic.
/// @return The unchanged finite value.
/// @throws std::domain_error If value is nonfinite.
double finite_statistic(double value) {
  if (!std::isfinite(value)) throw std::domain_error("Bootstrap: statistic must be finite");
  return value;
}
/// @brief Narrow a summary or interval endpoint only if finite double can represent it.
/// @param value Widened computed value.
/// @return Finite double, with ordinary rounding/underflow.
/// @throws std::overflow_error If value is nonfinite or outside double range.
double checked_double(long double value) {
  if (!std::isfinite(value) || std::fabs(value)>std::numeric_limits<double>::max())
    throw std::overflow_error("Bootstrap: summary exceeds finite double range");
  return static_cast<double>(value);
}
/// @brief Stable incremental mean of a nonempty finite sequence.
/// @param values Validated observations.
/// @return Long-double mean without summing large uncentred values.
long double centred_mean(const std::vector<double>& values) {
  long double mean=0;std::size_t count=0;
  for (double value:values) mean+=(static_cast<long double>(value)-mean)/++count;
  return mean;
}
/// @brief R type-7 quantile of a nonempty sorted finite sequence.
/// @param values Sorted finite statistics.
/// @param p Probability in [0,1].
/// @return Overflow-safe linear interpolation at (B-1)*p.
double quantile_sorted(const std::vector<double>& values,double p) {
  const double h=static_cast<double>(values.size()-1)*p;
  const auto lo=static_cast<std::size_t>(std::floor(h));
  if (lo+1>=values.size()) return values.back();
  return std::lerp(values[lo],values[lo+1],h-static_cast<double>(lo));
}
} // namespace

BootstrapResult bootstrap_summary(double estimate,std::vector<double> replicates,
    const std::vector<double>& jackknife_values,double confidence_level) {
  if (replicates.empty()) throw std::invalid_argument("Bootstrap: replicates must not be empty");
  if (!(confidence_level>0 && confidence_level<1)) throw std::invalid_argument("Bootstrap: confidence level must be in (0,1)");
  finite_statistic(estimate);
  for (double value:replicates) finite_statistic(value);
  for (double value:jackknife_values) finite_statistic(value);
  BootstrapResult r;r.estimate=estimate;r.confidence_level=confidence_level;r.replicates=std::move(replicates);
  const auto count=r.replicates.size();const long double mean=centred_mean(r.replicates);
  long double squared=0;
  for (double value:r.replicates) {const long double d=static_cast<long double>(value)-mean;squared+=d*d;}
  r.bias=checked_double(mean-static_cast<long double>(estimate));
  r.standard_error=count>1?checked_double(std::sqrt(squared/static_cast<long double>(count-1))):0;
  // Symmetry avoids rounding 1-alpha/2 to exactly one at the largest valid confidence level.
  const double alpha=1-confidence_level,z_lo=random::normal_quantile(alpha/2),z_hi=-z_lo;
  auto sorted=r.replicates;std::sort(sorted.begin(),sorted.end());
  const double q_lo=quantile_sorted(sorted,alpha/2),q_hi=quantile_sorted(sorted,1-alpha/2);
  r.percentile_interval={q_lo,q_hi};
  const long double theta=estimate,centre=theta-r.bias;
  r.basic_interval={checked_double(theta+(theta-q_hi)),checked_double(theta+(theta-q_lo))};
  r.normal_interval={checked_double(centre+static_cast<long double>(z_lo)*r.standard_error),
                     checked_double(centre+static_cast<long double>(z_hi)*r.standard_error)};
  r.bca_interval=r.percentile_interval;
  const auto less=static_cast<std::size_t>(std::count_if(r.replicates.begin(),r.replicates.end(),
      [&](double value){return value<estimate;}));
  if (less>0 && less<count && jackknife_values.size()>=2) {
    const double z0=random::normal_quantile(static_cast<double>(less)/static_cast<double>(count));
    const long double jack_mean=centred_mean(jackknife_values);
    long double scale=0;
    for (double value:jackknife_values) scale=std::max(scale,std::fabs(jack_mean-value));
    long double numerator=0,denominator=0;
    if (scale>0) for (double value:jackknife_values) {
      const long double d=(jack_mean-value)/scale;numerator+=d*d*d;denominator+=d*d;
    }
    const double acceleration=denominator>0?static_cast<double>(numerator/(6*std::pow(denominator,1.5L))):0;
    /// @brief Compute a BCa probability only on the nonsingular monotone branch.
    /// @param z Nominal normal quantile.
    /// @return Adjusted probability, or NaN to request the documented percentile fallback.
    const auto adjust=[&](double z) {
      const double denom=1-acceleration*(z0+z);
      if (!(denom>1e-12)) return std::numeric_limits<double>::quiet_NaN();
      return random::normal_cdf(z0+(z0+z)/denom);
    };
    const double p1=adjust(z_lo),p2=adjust(z_hi);
    if (std::isfinite(p1) && std::isfinite(p2) && p1<=p2)
      r.bca_interval={quantile_sorted(sorted,p1),quantile_sorted(sorted,p2)};
  }
  return r;
}

Bootstrap::Bootstrap(BootstrapOptions options):options_(options) {
  if (!(options_.confidence_level>0 && options_.confidence_level<1))
    throw std::invalid_argument("Bootstrap: confidence_level must be in (0,1)");
  if (options_.num_resamples==0) throw std::invalid_argument("Bootstrap: num_resamples must be >=1");
}
BootstrapResult Bootstrap::run(const std::vector<double>& sample,const SampleStatistic& statistic) const {
  if (sample.empty()) throw std::invalid_argument("Bootstrap::run: sample must not be empty");
  std::mt19937_64 rng(options_.seed);
  std::uniform_int_distribution<std::size_t> distribution(0,sample.size()-1);
  /// @brief Pick a valid index from the existing standard-library engine sequence.
  /// @return Index in [0,n), preserving the default resampling stream.
  const auto pick=[&]{return distribution(rng);};
  return run_with_picker(sample,statistic,pick);
}
BootstrapResult Bootstrap::run(const std::vector<double>& sample,const SampleStatistic& statistic,
    random::RandomGenerator& generator) const {
  if (sample.empty()) throw std::invalid_argument("Bootstrap::run: sample must not be empty");
  /// @brief Pick one unbiased index from the caller-owned stream.
  /// @return Index in [0,sample.size()).
  const auto pick=[&]{return static_cast<std::size_t>(generator.uniform_u64(0,sample.size()-1));};
  return run_with_picker(sample,statistic,pick);
}
BootstrapResult Bootstrap::run_with_picker(const std::vector<double>& sample,const SampleStatistic& statistic,
    const std::function<std::size_t()>& pick) const {
  const auto n=sample.size();const double estimate=finite_statistic(statistic(sample));
  std::vector<double> replicates(options_.num_resamples),resample(n);
  for (auto& replicate:replicates) {
    for (auto& value:resample) value=sample[pick()];
    replicate=finite_statistic(statistic(resample));
  }
  std::vector<double> loo;
  const auto less=std::count_if(replicates.begin(),replicates.end(),[&](double value){return value<estimate;});
  if (n>1 && less>0 && static_cast<std::size_t>(less)<replicates.size()) {
    loo.resize(n);std::vector<double> minus(n-1);
    for (std::size_t i=0;i<n;++i) {
      std::size_t at=0;for (std::size_t j=0;j<n;++j) if (j!=i) minus[at++]=sample[j];
      loo[i]=finite_statistic(statistic(minus));
    }
  }
  return bootstrap_summary(estimate,std::move(replicates),loo,options_.confidence_level);
}
} // namespace datamunge::stats
