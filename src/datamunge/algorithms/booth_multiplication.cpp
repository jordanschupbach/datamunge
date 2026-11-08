/// @file booth_multiplication.cpp
/// @brief Shared widened Booth recoding traversal for ordinary multiplication and actual traces.
#include <datamunge/algorithms/booth_multiplication.hpp>
namespace datamunge::algorithms {
/// @brief Private allocation-free signed-digit accumulation.
namespace {
/// @brief Visit all 32 bit pairs, accumulating exact representable int64 terms.
/// @tparam Observe Callable receiving each completed BoothStep; exceptions propagate.
/// @param multiplicand Signed int32 value, widened before scaling.
/// @param multiplier Signed int32 value viewed as uint32 bits.
/// @param observe Consumer of actual decisions, including no-ops.
/// @return Exact product and numbers of positive/negative recoding digits.
template<class Observe>
BoothResult recode(std::int32_t multiplicand,std::int32_t multiplier,Observe observe) {
  BoothResult result;unsigned previous=0;
  const auto bits=static_cast<std::uint32_t>(multiplier);
  for (unsigned i=0;i<32;++i) {
    const unsigned current=(bits>>i)&1u;
    const int digit=static_cast<int>(previous)-static_cast<int>(current);
    const auto scaled=static_cast<std::int64_t>(multiplicand)*(std::int64_t{1}<<i);
    const auto contribution=scaled*digit;
    result.product+=contribution;
    result.additions+=digit==1;result.subtractions+=digit==-1;
    observe(BoothStep{i,current,previous,digit,scaled,contribution,result.product});
    previous=current;
  }
  return result;
}
} // namespace
BoothResult booth_multiply(std::int32_t multiplicand,std::int32_t multiplier) {
  return recode(multiplicand,multiplier,[](const BoothStep&){});
}
BoothTrace booth_multiply_trace(std::int32_t multiplicand,std::int32_t multiplier) {
  BoothTrace trace;trace.multiplicand=multiplicand;trace.multiplier=multiplier;trace.steps.reserve(32);
  trace.result=recode(multiplicand,multiplier,[&](const BoothStep& step){trace.steps.push_back(step);});
  return trace;
}
} // namespace datamunge::algorithms
