/// @file
/// @brief Batched feedback rollouts and Gaussian policy-search updates.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/linalg.hpp"

namespace datamunge::rl {

/// @brief Evaluate bounded proportional-derivative feedback on a damped point mass.
///
/// The action is tanh(kp*position + kd*velocity + bias). Each step
/// updates velocity to 0.96*v + 0.2*action and advances position by
/// 0.2 times that velocity. The return is the negative accumulated
/// position, 0.1 times velocity and 0.025 times action squared costs, averaged across starts.
/// Optionally retain the position and velocity before each transition.
/// @param policies Floating CPU [policies,3] controller coefficients.
/// @param starts Floating CPU [rollouts,2] initial states with matching dtype.
/// @param steps Positive number of transitions, default 24.
/// @param trace Whether to return observations, default false.
/// @return Returns [policies] and observations [policies,rollouts,steps,2]
/// when requested, otherwise an undefined second tensor.
/// @throws std::invalid_argument Incompatible shapes, devices, dtypes or steps.
/// @par Example
/// @code
/// const auto result=dm::rl::feedback_returns(dm::nn::zeros({1,3}),dm::nn::zeros({1,2}));
/// std::print("{:.0f}\n", std::abs(result.first.item())); // 0
/// @endcode
/// @see gaussian_elite_refit()
inline std::pair<nn::Tensor,nn::Tensor> feedback_returns(
    const nn::Tensor& policies,const nn::Tensor& starts,std::int64_t steps=24,bool trace=false) {
  namespace n=nn;
  if (!policies.defined() || !starts.defined() || policies.dim()!=2 || starts.dim()!=2 ||
      policies.size(1)!=3 || starts.size(1)!=2 || policies.size(0)<=0 || starts.size(0)<=0 ||
      !n::is_floating(policies.dtype()) || policies.dtype()!=starts.dtype() || steps<=0 ||
      policies.device().backend!=compute::Backend::CPU || starts.device().backend!=compute::Backend::CPU) {
    throw std::invalid_argument("feedback_returns: incompatible policies or starts");
  }
  const auto count=policies.size(0), rollouts=starts.size(0);
  const auto kp=n::slice(policies,{0,0},{count,1}), kd=n::slice(policies,{0,1},{count,1});
  const auto bias=n::slice(policies,{0,2},{count,1});
  auto position=n::broadcast_to(n::slice(starts,{0,0},{rollouts,1}).reshape({1,rollouts}),{count,rollouts});
  auto velocity=n::broadcast_to(n::slice(starts,{0,1},{rollouts,1}).reshape({1,rollouts}),{count,rollouts});
  auto cost=n::zeros_like(position);
  std::vector<n::Tensor> observations;
  for (std::int64_t i=0;i<steps;++i) {
    if (trace) observations.push_back(n::stack({position,velocity},2));
    const auto action=n::tanh(kp*position+kd*velocity+bias);
    velocity=.96*velocity+.2*action;
    position=position+.2*velocity;
    cost=cost+(n::square(position)+.1*n::square(velocity)+.025*n::square(action));
  }
  return {-n::mean(cost,{1}),trace?n::stack(observations,2):n::Tensor{}};
}

/// @brief Refit a diagonal Gaussian to the highest-fitness candidates.
///
/// Fit the maximum-likelihood mean and variance to the elite subset.
/// Smooth those statistics against the old parameters, then apply a
/// variance floor. Fitness ties use ascending candidate row ID.
/// @param candidates Floating CPU [population,width], nonempty and finite.
/// @param fitness Finite scores [population], with larger values preferred.
/// @param center Old mean [1,width].
/// @param variance Old nonnegative variance [1,width].
/// @param elites Number of candidates kept, in [1,population].
/// @param update Weight on new elite statistics, in [0,1].
/// @param floor Nonnegative finite floor for updated variance.
/// @return Elite indices, elite mean and variance, then updated mean and variance.
/// @throws std::invalid_argument Incompatible tensors, numbers, or elite count.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto x=n::from_vector<float>({1,3},{2,1});
/// const auto out=dm::rl::gaussian_elite_refit(x,x.reshape({2}),n::zeros({1,1}),n::ones({1,1}),1,1,0);
/// std::print("{:.0f}\n",out[3].item()); // 3
/// @endcode
/// @par References
/// - Rubinstein and Kroese (2004), The Cross-Entropy Method. https://doi.org/10.1007/978-1-4757-4321-0
/// @see feedback_returns(), datamunge::optim::CrossEntropyMethod
inline std::array<nn::Tensor,5> gaussian_elite_refit(
    const nn::Tensor& candidates,const nn::Tensor& fitness,const nn::Tensor& center,
    const nn::Tensor& variance,std::int64_t elites,double update,double floor) {
  namespace n=nn;
  for (const auto* x:{&candidates,&fitness,&center,&variance}) {
    if (!x->defined() || x->device().backend!=compute::Backend::CPU || x->dtype()!=candidates.dtype()) {
      throw std::invalid_argument("gaussian_elite_refit: expected compatible CPU tensors");
    }
  }
  if (!n::is_floating(candidates.dtype()) || candidates.dim()!=2 || candidates.numel()==0 ||
      fitness.shape()!=n::Shape{candidates.size(0)} || center.shape()!=n::Shape{1,candidates.size(1)} ||
      variance.shape()!=center.shape() || elites<1 || elites>candidates.size(0) ||
      !(update>=0 && update<=1) || !(floor>=0 && std::isfinite(floor))) {
    throw std::invalid_argument("gaussian_elite_refit: invalid shapes or options");
  }
  for (const auto* x:{&candidates,&fitness,&center,&variance}) {
    const auto values=x->to_vector<double>();
    if (std::any_of(values.begin(),values.end(),[&](double z) {
          return !std::isfinite(z) || (x==&variance && z<0);
        })) throw std::invalid_argument("gaussian_elite_refit: invalid numeric input");
  }
  const auto indices=n::top_k(fitness,elites).second;
  const auto selected=n::gather(candidates,indices);
  const auto mean=n::mean(selected,{0},true);
  const auto var=n::mean(n::square(selected-mean),{0},true);
  const auto next=n::maximum((1-update)*variance+update*var,n::from_scalar(floor).to(variance.dtype()));
  return {indices,mean,var,(1-update)*center+update*mean,next};
}

/// @brief Update a linear policy from top-ranked mirrored search directions.
///
/// Rank directions by the better of their two returns. Divide the summed
/// return-difference weighted directions by the selected count and by the
/// population standard deviation of their returns, floored at 1e-8.
/// @param center Floating CPU policy coefficients [1,width].
/// @param directions Finite perturbations [count,width] with matching dtype.
/// @param returns Finite returns [2*count], positive directions first, then negative.
/// @param selected Number of directions kept, in [1,count].
/// @param rate Finite nonnegative step size.
/// @return Updated center, selected row IDs, and reward standard deviation (Float64 scalar).
/// @throws std::invalid_argument Invalid tensor geometry, devices, dtypes or options.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto [next, ids, sd]=dm::rl::ars_update(n::zeros({1,1}),n::ones({1,1}),
///     n::from_vector<float>({1,-1}),1,.5);
/// std::print("{:.0f}\n",next.item()); // 1
/// @endcode
/// @par References
/// - Mania, Guy and Recht (2018), Simple Random Search Provides a Competitive Approach to Reinforcement Learning.
///   https://arxiv.org/abs/1803.07055
/// @see feedback_returns()
inline std::array<nn::Tensor,3> ars_update(
    const nn::Tensor& center,const nn::Tensor& directions,const nn::Tensor& returns,
    std::int64_t selected,double rate) {
  namespace n=nn;
  for (const auto* x:{&center,&directions,&returns}) {
    if (!x->defined() || x->device().backend!=compute::Backend::CPU || x->dtype()!=center.dtype()) {
      throw std::invalid_argument("ars_update: expected matching CPU tensors");
    }
  }
  if (!n::is_floating(center.dtype()) || directions.dim()!=2 || directions.numel()==0 ||
      center.shape()!=n::Shape{1,directions.size(1)} || returns.dim()!=1 ||
      returns.size(0)%2!=0 || returns.size(0)/2!=directions.size(0) ||
      selected<1 || selected>directions.size(0) || !(rate>=0 && std::isfinite(rate))) {
    throw std::invalid_argument("ars_update: invalid shapes or options");
  }
  const auto reward=returns.to_vector<double>();
  const auto count=directions.size(0);
  if (std::any_of(reward.begin(),reward.end(),[](double x){return !std::isfinite(x);})) {
    throw std::invalid_argument("ars_update: returns must be finite");
  }
  std::vector<std::int64_t> order(count);
  std::iota(order.begin(),order.end(),0);
  const auto best=[&](std::int64_t i){return std::max(reward[i],reward[i+count]);};
  std::stable_sort(order.begin(),order.end(),[&](auto x,auto y){return best(x)>best(y);});
  order.resize(selected);
  double mean=0, second=0;
  for (auto i:order) {
    for (double value:{reward[i],reward[i+count]}) {
      mean+=value/(2.*selected);
      second+=value*value/(2.*selected);
    }
  }
  const auto sd=std::sqrt(std::max(0.,second-mean*mean));
  const auto widths=directions.size(1);
  const auto data=directions.to_vector<double>();
  auto next=center.to_vector<double>();
  for (std::int64_t j=0;j<widths;++j) {
    double delta=0;
    for (auto i:order) delta+=(reward[i]-reward[i+count])*data[i*widths+j];
    next[j]+=rate*delta/(selected*std::max(1e-8,sd));
  }
  return {n::from_vector(next,center.shape()).to(center.dtype()),n::from_vector(order),n::from_scalar(sd)};
}

/// @brief Estimate a mirrored evolution-strategy ascent direction from centered ranks.
///
/// Ranks range from -0.5 to 0.5, with ties assigned the average rank.
/// Multiply the positive-minus-negative rank difference by each sampled
/// direction and divide their sum by the population size times sigma.
/// @param directions Finite CPU Float32 or Float64 perturbations [count,width].
/// @param returns Finite returns [2*count], same dtype and device, positive then negative.
/// @param sigma Finite positive sampling standard deviation.
/// @return Centered ranks [2*count] and gradient estimate [width], matching input dtype.
/// @throws std::invalid_argument Invalid geometry, dtype, device, returns or sigma.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto out=dm::rl::mirrored_rank_gradient(n::ones({1,1}),n::from_vector<float>({2,1}),0.5);
/// std::print("{:.0f}\n",out.second.item()); // 1
/// @endcode
/// @par References
/// - Salimans et al. (2017), Evolution Strategies as a Scalable Alternative to Reinforcement Learning.
///   https://arxiv.org/abs/1703.03864
/// @see ars_update(), feedback_returns()
inline std::pair<nn::Tensor,nn::Tensor> mirrored_rank_gradient(
    const nn::Tensor& directions,const nn::Tensor& returns,double sigma) {
  namespace n=nn;
  if (!directions.defined() || !returns.defined() || directions.dim()!=2 || directions.numel()==0 ||
      returns.dim()!=1 || returns.size(0)%2!=0 || returns.size(0)/2!=directions.size(0) ||
      !n::is_floating(directions.dtype()) || returns.dtype()!=directions.dtype() ||
      directions.device().backend!=compute::Backend::CPU || returns.device().backend!=compute::Backend::CPU ||
      !(sigma>0 && std::isfinite(sigma))) {
    throw std::invalid_argument("mirrored_rank_gradient: incompatible shapes or sampling scale");
  }
  const auto calculate=[&]<typename T>() {
    const auto values=returns.to_vector<T>();
    if (std::any_of(values.begin(),values.end(),[](T x){return !std::isfinite(x);})) {
      throw std::invalid_argument("mirrored_rank_gradient: returns must be finite");
    }
    std::vector<std::size_t> order(values.size());
    std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return values[a]<values[b];});
    std::vector<T> ranks(values.size());
    for (std::size_t first=0;first<order.size();) {
      std::size_t end=first+1;
      while (end<order.size() && values[order[first]]==values[order[end]]) ++end;
      const T rank=T(.5)*(T(first)+T(end-1))/T(order.size()-1)-T(.5);
      for (auto i=first;i<end;++i) ranks[order[i]]=rank;
      first=end;
    }
    const auto draws=directions.to_vector<T>();
    const auto count=directions.size(0), width=directions.size(1);
    std::vector<T> gradient(width);
    for (std::int64_t j=0;j<width;++j) {
      T sum=0;
      for (std::int64_t i=0;i<count;++i) sum+=(ranks[i]-ranks[i+count])*draws[i*width+j];
      gradient[j]=sum/(T(values.size())*T(sigma));
    }
    return std::pair{n::from_vector(ranks),n::from_vector(gradient)};
  };
  return directions.dtype()==n::DType::Float32 ? calculate.template operator()<float>() :
                                                calculate.template operator()<double>();
}

/// @brief Update a diagonal Gaussian search distribution using separable NES.
///
/// Log-rank utilities are normalized and centered. Equal returns share the
/// mean utility of the ranks they occupy. The Gaussian Fisher inverse scales
/// the log-standard-deviation direction by one half.
/// @param mean Current finite mean [1,width].
/// @param scale Current positive standard deviations [1,width].
/// @param noise Finite standard-normal samples [population,width].
/// @param returns Finite fitness values [population] in sample order, with at least two samples.
/// @param mean_rate Nonnegative mean learning rate, default 1.
/// @param scale_rate Nonnegative log-scale learning rate, default (3+log(width))/(5*sqrt(width)).
///
/// @return Updated mean [1,width], standard deviations [1,width], and utilities [population,1].
///   Inputs and outputs share the same CPU floating-point dtype.
/// @throws std::invalid_argument Incompatible inputs or invalid learning rates.
/// @throws std::overflow_error Updated means or scales fall outside their valid range.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto [mean,scale,utility]=dm::rl::snes_update(n::zeros({1,1}),n::ones({1,1}),
///     n::from_vector<float>({-1,1},{2,1}),n::from_vector<float>({0,1}));
/// std::print("{:.0f} {:.0f}\n",mean.item(),scale.item()); // 1 1
/// @endcode
/// @par References
/// - Wierstra et al. (2014), Natural Evolution Strategies.
///   https://www.jmlr.org/papers/v15/wierstra14a.html
/// @see mirrored_rank_gradient()
inline std::array<nn::Tensor,3> snes_update(
    const nn::Tensor& mean,const nn::Tensor& scale,const nn::Tensor& noise,
    const nn::Tensor& returns,double mean_rate=1.,std::optional<double> scale_rate=std::nullopt) {
  namespace n=nn;
  for (const auto* tensor:{&mean,&scale,&noise,&returns}) {
    if (!tensor->defined() || tensor->device().backend!=compute::Backend::CPU ||
        tensor->dtype()!=mean.dtype()) {
      throw std::invalid_argument("snes_update: expected matching CPU tensors");
    }
  }
  if (!n::is_floating(mean.dtype()) || noise.dim()!=2 || noise.size(0)<2 || noise.size(1)<1 ||
      mean.shape()!=n::Shape{1,noise.size(1)} || scale.shape()!=mean.shape() ||
      returns.shape()!=n::Shape{noise.size(0)} || !(mean_rate>=0 && std::isfinite(mean_rate)) ||
      (scale_rate && !(*scale_rate>=0 && std::isfinite(*scale_rate)))) {
    throw std::invalid_argument("snes_update: invalid shapes or learning rates");
  }
  const auto finite=[](double value){return std::isfinite(value);};
  const auto positive=[](double value){return value>0 && std::isfinite(value);};
  const auto means=mean.to_vector<double>();
  const auto draws=noise.to_vector<double>();
  const auto scales=scale.to_vector<double>();
  const auto fitness=returns.to_vector<double>();
  if (!std::all_of(scales.begin(),scales.end(),positive) ||
      !std::all_of(fitness.begin(),fitness.end(),finite) ||
      !std::all_of(means.begin(),means.end(),finite) ||
      !std::all_of(draws.begin(),draws.end(),finite)) {
    throw std::invalid_argument("snes_update: expected finite inputs and positive scales");
  }
  const auto count=fitness.size();
  std::vector<std::size_t> order(count);
  std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return fitness[a]>fitness[b];});
  std::vector<double> ranked(count), weights(count);
  for (std::size_t i=0;i<count;++i) {
    ranked[i]=std::max(0.,std::log(count/2.+1.)-std::log(i+1.));
  }
  const double mass=std::accumulate(ranked.begin(),ranked.end(),0.);
  for (std::size_t first=0;first<count;) {
    std::size_t end=first+1;
    while (end<count && fitness[order[first]]==fitness[order[end]]) ++end;
    const double value=end-first==count ? 0. :
        std::accumulate(ranked.begin()+first,ranked.begin()+end,0.)/
        (mass*(end-first))-1./count;
    for (auto i=first;i<end;++i) weights[order[i]]=value;
    first=end;
  }
  const auto utility=n::from_vector(weights,{noise.size(0),1}).to(mean.dtype());
  const double width=noise.size(1);
  const double rate=scale_rate.value_or((3.+std::log(width))/(5.*std::sqrt(width)));
  const auto next_mean=mean+mean_rate*scale*(utility*noise).sum({0},true);
  const auto next_scale=scale*n::exp((.5*rate)*(utility*(noise*noise-1.)).sum({0},true));
  const auto updated_scales=next_scale.to_vector<double>();
  const auto updated_means=next_mean.to_vector<double>();
  if (!std::all_of(updated_scales.begin(),updated_scales.end(),positive) ||
      !std::all_of(updated_means.begin(),updated_means.end(),finite)) {
    throw std::overflow_error("snes_update: update is out of range");
  }
  return {next_mean,next_scale,utility};
}

/// @brief Full covariance matrix adaptation evolution strategy for maximization.
///
/// Maintains a Float64 CPU Gaussian search distribution with separate
/// evolution paths for the covariance matrix and global step size.
/// Uses positive log-weight recombination over the best half of the
/// population. Equal returns preserve candidate order.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// dm::rl::CmaEs search(n::zeros({1,1},n::DType::Float64),0.35);
/// const auto candidates=search.sample(n::from_vector<double>({-1,1},{2,1}));
/// search.update(candidates,n::from_vector<double>({0,1}));
/// @endcode
/// @par References
/// - Hansen (2016), The CMA Evolution Strategy: A Tutorial.
///   https://arxiv.org/abs/1604.00772
/// @see snes_update()
class CmaEs {
 public:
  /// @brief Start with identity covariance and zero evolution paths.
  /// @param mean Finite initial mean [1,width], Float64 CPU.
  /// @param sigma Finite positive global step size.
  /// @throws std::invalid_argument Invalid mean or sigma.
  /// @par Example
  /// @code
  /// dm::rl::CmaEs search(dm::nn::zeros({1,2},dm::nn::DType::Float64),0.35);
  /// @endcode
  explicit CmaEs(const nn::Tensor& mean,double sigma):sigma_(sigma) {
    if (!mean.defined() || mean.dim()!=2 || mean.size(0)!=1 || mean.size(1)<1 ||
        !valid_tensor(mean) || !(sigma>0 && std::isfinite(sigma))) {
      throw std::invalid_argument("CmaEs: expected finite Float64 CPU row and positive sigma");
    }
    mean_=mean.detach().clone();
    covariance_=nn::eye(mean.size(1),nn::DType::Float64);
    step_path_=nn::zeros_like(mean_);
    covariance_path_=nn::zeros_like(mean_);
  }

  /// @brief Map independent standard-normal draws into candidate rows.
  /// @param noise Finite Float64 CPU draws [population,width], population at least two.
  /// @return Candidates of the same shape and dtype, ready for evaluation.
  /// @throws std::invalid_argument Invalid draws.
  /// @throws std::overflow_error Nonfinite candidate or invalid covariance spectrum.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;
  /// const dm::rl::CmaEs search(n::zeros({1,1},n::DType::Float64),0.35);
  /// const auto population=search.sample(n::from_vector<double>({-1,1},{2,1}));
  /// @endcode
  [[nodiscard]] nn::Tensor sample(const nn::Tensor& noise) const {
    check_population(noise);
    const nn::NoGradGuard no_grad;
    const auto [values,basis]=spectrum(covariance_);
    const auto root=nn::matmul(basis*nn::sqrt(values),basis,false,true);
    const auto candidates=mean_+sigma_*nn::matmul(noise,root);
    if (!valid_tensor(candidates)) throw std::overflow_error("CmaEs: nonfinite candidate");
    return candidates;
  }

  /// @brief Adapt the distribution after evaluating one generation.
  /// @param candidates Float64 CPU [population,width] sampled from the current state.
  /// @param returns Finite Float64 CPU fitness vector [population]. Larger is better.
  /// @return Selected parent indices in descending fitness order.
  /// @throws std::invalid_argument Invalid candidates, fitness, or changed population size.
  /// @throws std::overflow_error The proposed distribution exceeds numerical range.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;
  /// dm::rl::CmaEs search(n::zeros({1,1},n::DType::Float64),0.35);
  /// const auto rows=search.sample(n::from_vector<double>({-1,1},{2,1}));
  /// const auto parents=search.update(rows,n::from_vector<double>({0,1}));
  /// std::print("{}\n",parents.to_vector<std::int64_t>().front()); // 1
  /// @endcode
  nn::Tensor update(const nn::Tensor& candidates,const nn::Tensor& returns) {
    namespace n=nn;
    check_population(candidates);
    const auto count=candidates.size(0);
    if (!valid_tensor(returns) || returns.shape()!=n::Shape{count} ||
        (population_!=0 && population_!=count)) {
      throw std::invalid_argument("CmaEs: incompatible fitness or population size");
    }
    const n::NoGradGuard no_grad;
    const auto reward=returns.to_vector<double>();
    std::vector<std::int64_t> order(count);
    std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return reward[a]>reward[b];});
    order.resize(count/2);
    std::vector<double> weights(order.size());
    for (std::size_t i=0;i<weights.size();++i) {
      weights[i]=std::log(weights.size()+.5)-std::log(i+1.);
    }
    const double mass=std::accumulate(weights.begin(),weights.end(),0.);
    for (auto& weight:weights) weight/=mass;
    const double mu=1./std::inner_product(weights.begin(),weights.end(),weights.begin(),0.);
    const double width=mean_.size(1);
    const double cs=(mu+2)/(width+mu+5);
    const double ds=1+2*std::max(0.,std::sqrt((mu-1)/(width+1))-1)+cs;
    const double cc=(4+mu/width)/(width+4+2*mu/width);
    const double c1=2/(std::pow(width+1.3,2)+mu);
    const double cmu=std::min(1-c1,2*(mu-2+1/mu)/(std::pow(width+2,2)+mu));
    const double expected_norm=std::sqrt(width)*(1-1/(4*width)+1/(21*width*width));
    const auto selected=n::from_vector(order);
    const auto offsets=(n::gather(candidates,selected)-mean_)/sigma_;
    const auto weighted=n::from_vector(weights,{count/2,1})*offsets;
    const auto shift=weighted.sum({0},true);
    const auto [values,basis]=spectrum(covariance_);
    const auto whiten=n::matmul(basis/n::sqrt(values),basis,false,true);
    const auto step_path=(1-cs)*step_path_+std::sqrt(cs*(2-cs)*mu)*n::matmul(shift,whiten);
    const double norm=std::sqrt(n::square(step_path).sum().item());
    const bool adapt_path=norm/std::sqrt(1-std::pow(1-cs,2.*(generation_+1)))/expected_norm <
                          1.4+2/(width+1);
    const auto covariance_path=(1-cc)*covariance_path_+
        (adapt_path ? std::sqrt(cc*(2-cc)*mu) : 0.)*shift;
    const auto next_covariance=(1-c1-cmu+(adapt_path ? 0. : c1*cc*(2-cc)))*covariance_+
        c1*n::matmul(covariance_path,covariance_path,true,false)+cmu*n::matmul(offsets,weighted,true,false);
    const auto next_mean=mean_+sigma_*shift;
    const double next_sigma=sigma_*std::exp(cs/ds*(norm/expected_norm-1));
    if (!valid_tensor(next_mean) || !(next_sigma>0 && std::isfinite(next_sigma))) {
      throw std::overflow_error("CmaEs: update exceeds numerical range");
    }
    const auto symmetric=.5*(next_covariance+n::transpose(next_covariance,{1,0}));
    (void)spectrum(symmetric);
    mean_=next_mean;
    covariance_=symmetric;
    step_path_=step_path;
    covariance_path_=covariance_path;
    sigma_=next_sigma;
    population_=count;
    ++generation_;
    return selected;
  }

  /// @brief Copy the current search distribution and adaptation paths.
  /// @return Mean [1,width], covariance [width,width], step-size path [1,width],
  /// covariance path [1,width], and global step size, in that order.
  /// @par Example
  /// @code
  /// dm::rl::CmaEs search(dm::nn::zeros({1,1},dm::nn::DType::Float64),0.35);
  /// const auto [mean,covariance,step_path,covariance_path,sigma]=search.state();
  /// @endcode
  [[nodiscard]] auto state() const {
    return std::tuple{mean_.clone(),covariance_.clone(),step_path_.clone(),
                      covariance_path_.clone(),sigma_};
  }

 private:
  /// @brief Check that every entry is finite and stored as Float64 on CPU.
  /// @param value Tensor to validate.
  /// @return Whether the tensor satisfies the storage and finiteness requirements.
  static bool valid_tensor(const nn::Tensor& value) {
    if (!value.defined() || value.dtype()!=nn::DType::Float64 ||
        value.device().backend!=compute::Backend::CPU) return false;
    const auto entries=value.to_vector<double>();
    return std::all_of(entries.begin(),entries.end(),[](double x){return std::isfinite(x);});
  }

  /// @brief Validate finite candidate or sample rows.
  /// @param value Tensor of rows.
  /// @throws std::invalid_argument Wrong dimensions, storage, or nonfinite values.
  void check_population(const nn::Tensor& value) const {
    if (!valid_tensor(value) || value.dim()!=2 || value.size(0)<2 || value.size(1)!=mean_.size(1)) {
      throw std::invalid_argument("CmaEs: expected finite Float64 CPU population rows");
    }
  }

  /// @brief Diagonalize a finite positive-definite covariance.
  /// @param value Symmetric covariance.
  /// @return Positive eigenvalues and the corresponding orthonormal basis.
  /// @throws std::overflow_error Covariance or eigenvalues outside the valid range.
  static nn::EighResult spectrum(const nn::Tensor& value) {
    if (!valid_tensor(value)) throw std::overflow_error("CmaEs: nonfinite covariance");
    auto result=nn::eigh(value);
    const auto eigenvalues=result.e.to_vector<double>();
    if (!std::all_of(eigenvalues.begin(),eigenvalues.end(),[](double x){return x>0 && std::isfinite(x);})) {
      throw std::overflow_error("CmaEs: covariance must be positive definite");
    }
    return result;
  }

  nn::Tensor mean_;             ///< Current mean row.
  nn::Tensor covariance_;       ///< Current symmetric covariance.
  nn::Tensor step_path_;        ///< Accumulated whitened displacement.
  nn::Tensor covariance_path_;  ///< Accumulated displacement for rank-one covariance adaptation.
  double sigma_;               ///< Global standard deviation multiplier.
  std::int64_t generation_=0;   ///< Number of completed updates.
  std::int64_t population_=0;   ///< Fixed population size after the first update.
};

} // namespace datamunge::rl
