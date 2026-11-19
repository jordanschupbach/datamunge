#pragma once

/// @file linear_counterfactual.hpp
/// @brief Minimum-cost recourse for a linear logit under box constraints.

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Find the closest box-constrained input reaching a minimum linear score.
///
/// Minimizes half the sum of weighted squared coordinate changes. Equal
/// lower and upper bounds fix a coordinate. Returns the input unchanged
/// when it already reaches the threshold. Uses scalar dual bisection.
/// @param input Finite original input inside the box.
/// @param weight Finite linear score coefficients.
/// @param cost Finite positive squared-distance weights.
/// @param lower Finite lower bounds.
/// @param upper Finite upper bounds.
/// @param bias Finite linear score intercept.
/// @param target Finite minimum desired score.
/// @return Approximate optimum, or nullopt when the maximum valid score is below target.
/// @throws std::invalid_argument For nonfinite inputs, incompatible sizes, invalid bounds or costs.
/// @par Example
/// @code
/// auto x=dm::nn::linear_counterfactual({0},{1},{1},{0},{1},0,.5);
/// @endcode
inline std::optional<std::vector<double>> linear_counterfactual(
    const std::vector<double>& input,const std::vector<double>& weight,
    const std::vector<double>& cost,const std::vector<double>& lower,
    const std::vector<double>& upper,double bias,double target) {
  const auto size=input.size();
  if (!size || weight.size()!=size || cost.size()!=size || lower.size()!=size ||
      upper.size()!=size || !std::isfinite(bias) || !std::isfinite(target))
    throw std::invalid_argument("Invalid linear counterfactual dimensions or score");
  auto maximizer=input;
  long double high=0;
  long double initial=bias, maximum=bias;
  for (std::size_t i=0;i<size;++i) {
    if (!std::isfinite(input[i]) || !std::isfinite(weight[i]) || !std::isfinite(cost[i]) ||
        !std::isfinite(lower[i]) || !std::isfinite(upper[i]) ||
        !(lower[i]<=input[i] && input[i]<=upper[i]) || !(cost[i]>0))
      throw std::invalid_argument("Invalid linear counterfactual coordinate");
    const long double w=weight[i];
    if (w!=0) {
      maximizer[i]=w>0 ? upper[i] : lower[i];
      high=std::max(high,(static_cast<long double>(maximizer[i])-input[i])*cost[i]/w);
    }
    initial+=w*input[i];
    maximum+=w*maximizer[i];
  }
  if (initial>=target) return input;
  if (maximum<target) return std::nullopt;
  auto result=maximizer, candidate=input;
  long double low=0;
  for (int step=0;step<96;++step) {
    const auto mid=std::midpoint(low,high);
    if (mid==low || mid==high) break;
    long double score=bias;
    for (std::size_t i=0;i<size;++i) {
      const auto moved=input[i]+mid*weight[i]/cost[i];
      candidate[i]=static_cast<double>(std::clamp(moved,static_cast<long double>(lower[i]),
                                                        static_cast<long double>(upper[i])));
      score+=static_cast<long double>(weight[i])*candidate[i];
    }
    if (score>=target) {
      high=mid;
      result.swap(candidate);
    } else {
      low=mid;
    }
  }
  return result;
}

}  // namespace datamunge::nn
