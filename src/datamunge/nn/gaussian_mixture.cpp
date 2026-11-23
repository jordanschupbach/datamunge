/// @file gaussian_mixture.cpp
/// @brief Seeded multistart fitting and covariance-structure comparison by BIC.

#include <datamunge/nn/gaussian_mixture.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <random>
#include <set>
#include <span>
#include <utility>

namespace datamunge::nn::ml {
/// @brief Initialization and comparison helpers for Gaussian mixtures.
namespace {

/// @brief Data used for deterministic initialization of each candidate.
struct SeedData {
  std::span<const double> values; ///< Borrowed observations in row-major order.
  std::size_t rows{};             ///< Number of rows.
  std::size_t cols{};             ///< Number of features.
  std::vector<double> mean;       ///< Empirical feature means.
  std::vector<double> variance;   ///< Empirical feature population variances.
};

/// @brief Validates observations and computes diagonal starting moments.
SeedData seed_data(const std::vector<double>& values, std::size_t n, std::size_t d) {
  SeedData data{values,n,d,std::vector<double>(d),std::vector<double>(d)};
  for (std::size_t j = 0; j < d; ++j) {
    long double mean = 0, scatter = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const auto x = values[i*d+j];
      if (!std::isfinite(x))
        throw std::invalid_argument("Gaussian mixture selection requires finite observations");
      const auto difference = x-mean;
      mean += difference/(i+1);
      scatter += difference*(x-mean);
    }
    data.mean[j] = static_cast<double>(mean);
    data.variance[j] = static_cast<double>(std::max(0.0L,scatter/n));
    if (!std::isfinite(data.variance[j]))
      throw std::invalid_argument("Gaussian mixture selection: variance is outside finite range");
  }
  return data;
}

/// @brief Picks means by distance-weighted sampling, with a uniform start for identical data.
std::vector<double> seed_means(const SeedData& data, int k, std::mt19937_64& random) {
  if (k == 1)
    return data.mean;
  const auto [n,d] = std::pair{data.rows,data.cols};
  std::vector<double> means;
  means.reserve(static_cast<std::size_t>(k)*d);
  std::vector<long double> squared_distance(n,std::numeric_limits<long double>::infinity());
  std::vector<bool> chosen(n,false);
  const double distance_scale = std::max(1.0,std::sqrt(*std::max_element(
      data.variance.begin(),data.variance.end())));
  auto row = std::uniform_int_distribution<std::size_t>(0,n-1)(random);
  for (int component = 0; component < k; ++component) {
    const auto first = data.values.begin()+row*d;
    means.insert(means.end(),first,first+d);
    if (component+1 == k)
      break;
    chosen[row] = true;
    long double largest = 0;
    for (std::size_t i = 0; i < n; ++i) {
      long double distance = 0;
      for (std::size_t j = 0; j < d; ++j) {
        const auto difference = (static_cast<long double>(data.values[i*d+j])-first[j])/distance_scale;
        distance += difference*difference;
      }
      squared_distance[i] = std::min(squared_distance[i],distance);
      largest = std::max(largest,squared_distance[i]);
    }
    std::vector<double> weights(n);
    for (std::size_t i = 0; i < n; ++i)
      if (!chosen[i])
        weights[i] = largest > 0 ? static_cast<double>(squared_distance[i]/largest) : 1;
    row = std::discrete_distribution<std::size_t>(weights.begin(),weights.end())(random);
  }
  return means;
}

/// @brief Builds a positive diagonal starting covariance that satisfies all three constraints.
GaussianMixtureState initialize(const SeedData& data, int k, double floor, std::mt19937_64& random) {
  const auto d = static_cast<std::int64_t>(data.cols);
  auto diagonal = data.variance;
  for (double& v : diagonal)
    v = std::max(floor,v/k);
  const auto covariance = nn::matrix_diag(nn::from_vector(diagonal,{1,d}));
  return {nn::full({k},1.0/k,nn::DType::Float64),
          nn::from_vector(seed_means(data,k,random),{k,d}),nn::repeat(covariance,k,0)};
}

/// @brief Current accepted mixture, likelihood and posterior from one optimization attempt.
struct Attempt {
  GaussianMixtureState state;          ///< Accepted parameters.
  nn::Tensor posterior;                ///< Responsibilities at those parameters.
  double likelihood{-std::numeric_limits<double>::infinity()}; ///< Log likelihood at those parameters.
  bool converged{};                    ///< True if the improvement met the tolerance.
};

/// @brief Iterates the shared EM step and retains the last finite improving likelihood.
Attempt fit_start(const nn::Tensor& data, const GaussianMixture& em, GaussianMixtureState initial,
    const GaussianMixtureSelectionOptions& options) {
  Attempt result;
  auto state = std::move(initial);
  for (std::size_t update = 0; ; ++update) {
    GaussianMixtureStep step;
    try {
      step = em.Step(data,state);
    } catch (const std::invalid_argument&) {
      break; // A candidate covariance could not be factored.
    } catch (const std::runtime_error&) {
      break; // A component may have collapsed.
    }
    const auto likelihood = step.log_likelihood;
    if (!std::isfinite(likelihood))
      break;
    if (update == 0) {
      result = {std::move(state),std::move(step.responsibilities),likelihood,false};
    } else {
      const auto improvement = likelihood-result.likelihood;
      const auto scale = 1+std::abs(result.likelihood);
      if (improvement < 0) {
        result.converged = -improvement <= std::min(options.relative_tolerance,
            64*std::numeric_limits<double>::epsilon())*scale;
        break;
      }
      result = {std::move(state),std::move(step.responsibilities),likelihood,
          improvement <= options.relative_tolerance*scale};
    }
    if (result.converged || update == options.max_iterations)
      break;
    state = std::move(step.next);
  }
  return result;
}

/// @brief Counts independent covariance, weight and mean parameters.
std::size_t parameter_count(std::size_t k, std::size_t d, GaussianMixtureCovariance kind) {
  const auto symmetric = d*(d+1)/2;
  const auto covariance = kind == GaussianMixtureCovariance::Tied ? symmetric
      : kind == GaussianMixtureCovariance::Diagonal ? k*d : k*symmetric;
  return covariance + k*d + k-1;
}

/// @brief Checks that the largest temporary can represent every requested model shape.
void check_workspace(std::size_t n, std::size_t d, std::size_t k) {
  const auto limit = std::vector<double>{}.max_size();
  if (d > limit/d || k > limit/(d*d) || n > limit/(k*d*d))
    throw std::length_error("Gaussian mixture EM workspace exceeds storage capacity");
}

} // namespace

GaussianMixtureSelection select_gaussian_mixture(const nn::Tensor& x,
    const std::vector<int>& component_counts, const GaussianMixtureSelectionOptions& options) {
  if (!x.defined() || x.device().backend != compute::Backend::CPU ||
      x.dtype() != nn::DType::Float64 || x.dim() != 2 || x.size(0) < 1 || x.size(1) < 1 ||
      x.size(1) > std::numeric_limits<int>::max() || component_counts.empty() ||
      options.starts == 0 || options.max_iterations == 0 ||
      !(options.relative_tolerance > 0 && options.relative_tolerance < 1) ||
      !(options.covariance_floor > 0) || !std::isfinite(options.covariance_floor))
    throw std::invalid_argument("Gaussian mixture selection: invalid data geometry or controls");
  const auto n = static_cast<std::size_t>(x.size(0)), d = static_cast<std::size_t>(x.size(1));
  std::set<int> seen;
  for (const int k : component_counts) {
    if (k < 1 || static_cast<std::size_t>(k) > n || !seen.insert(k).second)
      throw std::invalid_argument("Gaussian mixture selection requires distinct component counts in [1, N]");
    check_workspace(n,d,k);
  }

  nn::NoGradGuard guard;
  const auto values = x.to_vector<double>();
  const auto data = seed_data(values,n,d);
  GaussianMixtureSelection result;
  double lowest = std::numeric_limits<double>::infinity();
  const std::array constraints{GaussianMixtureCovariance::Tied, GaussianMixtureCovariance::Diagonal,
      GaussianMixtureCovariance::Full};
  for (const int k : component_counts)
    for (const auto kind : constraints) {
      GaussianMixture em(k,static_cast<int>(d),options.covariance_floor,kind);
      std::mt19937_64 random(options.seed);
      Attempt best;
      for (std::size_t start = 0; start < (k == 1 ? 1 : options.starts); ++start) {
        auto attempt = fit_start(x,em,initialize(data,k,options.covariance_floor,random),options);
        if (attempt.likelihood > best.likelihood ||
            (attempt.likelihood == best.likelihood && attempt.converged && !best.converged))
          best = std::move(attempt);
      }
      const auto parameters = parameter_count(k,d,kind);
      const double bic = -2*best.likelihood + parameters*std::log(static_cast<double>(n));
      if (bic < lowest) {
        lowest = bic;
        result.best_index = result.candidates.size();
        result.model = std::move(best.state);
        result.responsibilities = std::move(best.posterior);
      }
      result.candidates.push_back({k,kind,parameters,best.likelihood,bic,best.converged});
    }
  if (!std::isfinite(lowest))
    throw std::runtime_error("Gaussian mixture selection: no finite candidate");
  return result;
}

} // namespace datamunge::nn::ml
