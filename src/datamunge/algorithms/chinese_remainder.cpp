/// @file chinese_remainder.cpp
/// @brief Checked sequential generalized CRT with actual widened Bezout/candidate certificates.
#include <datamunge/algorithms/chinese_remainder.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private validated arithmetic and shared ordinary/traced CRT execution.
namespace crt_detail {
/// @brief Signed compiler-supported intermediate, sufficient for products of positive int64 moduli.
using Wide=__int128;
/// @brief Largest period representable by the legacy result type.
constexpr Wide max_period=std::numeric_limits<std::int64_t>::max();
/// @brief Increment an actual work counter without wrapping.
/// @param value Mutable size_t counter.
/// @throws std::overflow_error If value is already SIZE_MAX.
void count(std::size_t& value) {
  if(value==std::numeric_limits<std::size_t>::max())throw std::overflow_error("CRT: work counter overflow");
  ++value;
}
/// @brief Validate shape and every positive modulus before any arithmetic or work cap.
/// @param residues Borrowed signed residues.
/// @param moduli Borrowed signed moduli, required positive.
/// @return Actual number of validated moduli.
/// @throws std::invalid_argument If shapes differ or a modulus is nonpositive.
/// @throws std::overflow_error If the visit counter cannot fit size_t.
std::size_t validate(const std::vector<std::int64_t>& residues,const std::vector<std::int64_t>& moduli) {
  if(residues.size()!=moduli.size())throw std::invalid_argument("chinese_remainder: remainders and moduli must have equal length");
  std::size_t checked=0;
  for(const auto modulus:moduli) {
    if(modulus<=0)throw std::invalid_argument("chinese_remainder: every modulus must be strictly positive");
    count(checked);
  }
  return checked;
}
/// @brief Normalize a widened value without ever adding a modulus to an unreduced operand.
/// @param value Arbitrary signed working value.
/// @param modulus Validated positive int64 modulus.
/// @return int64 representative in [0,modulus).
std::int64_t normalize(Wide value,std::int64_t modulus) noexcept {
  value%=modulus;if(value<0)value+=modulus;return static_cast<std::int64_t>(value);
}
/// @brief Per-call complete-work limits and actual counters, with no shared state.
struct Context {
  CrtOptions options; ///< Caller-supplied merge/division limits.
  CrtWork work; ///< Actual successful validation and executed arithmetic visits.

  /// @brief Admit the next attempted merge before its arithmetic.
  /// @throws std::length_error If the positive merge cap is full.
  /// @throws std::overflow_error If the attempt counter cannot be incremented.
  void begin_merge() {
    if(options.max_merges && work.attempted_merges==options.max_merges)
      throw std::length_error("CRT: attempted merge budget exhausted");
    count(work.attempted_merges);
  }
  /// @brief Admit the next actual extended-gcd quotient/remainder iteration.
  /// @throws std::length_error If the positive complete-call division cap is full.
  /// @throws std::overflow_error If the division counter cannot be incremented.
  void division() {
    if(options.max_euclidean_steps && work.euclidean_steps==options.max_euclidean_steps)
      throw std::length_error("CRT: Euclidean division budget exhausted");
    count(work.euclidean_steps);
  }
};
/// @brief Actual positive gcd and widened Bezout coefficients from one iterative call.
struct Bezout {
  std::int64_t gcd{0}; ///< Positive gcd, at most either validated input modulus.
  Wide first{0}; ///< Actual coefficient of the first input.
  Wide second{0}; ///< Actual coefficient of the second input.
};
/// @brief Execute the original iterative Euclidean ordering with widened coefficient updates.
/// Positive int64 inputs keep coefficients bounded by the input magnitudes;
/// their unreduced quotient products therefore fit signed 128-bit intermediates.
/// @param a Positive first modulus.
/// @param b Positive second modulus.
/// @param context Actual complete-call division counter and cap.
/// @return Actual gcd and coefficients satisfying a*first+b*second=gcd.
/// @throws std::length_error If the complete-call division budget is reached.
/// @throws std::overflow_error If the division counter overflows size_t.
Bezout extended_gcd(std::int64_t a,std::int64_t b,Context& context) {
  Wide old_r=a,r=b,old_s=1,s=0,old_t=0,t=1;
  while(r) {
    context.division();const Wide quotient=old_r/r;
    auto next=old_r-quotient*r;old_r=r;r=next;
    next=old_s-quotient*s;old_s=s;s=next;
    next=old_t-quotient*t;old_t=t;t=next;
  }
  return {static_cast<std::int64_t>(old_r),old_s,old_t};
}
/// @brief Perform one actual compatible/inconsistent/wide-period merge without narrowing its candidate.
/// @param before Representable compatible prefix, with positive period and canonical residue.
/// @param residue Original signed next residue.
/// @param modulus Positive next modulus.
/// @param index Original input index.
/// @param context Actual counters and complete-call caps.
/// @return Actual certificate, outcome and wide candidate when compatible.
/// @throws std::length_error If an attempted-merge or division cap is exhausted.
/// @throws std::overflow_error If a work counter overflows size_t.
CrtMerge merge(const CrtSolution& before,std::int64_t residue,std::int64_t modulus,
               std::size_t index,Context& context) {
  context.begin_merge();CrtMerge step;step.index=index;step.before=before;
  step.input_remainder=residue;step.input_modulus=modulus;step.normalized_remainder=normalize(residue,modulus);
  step.difference=step.normalized_remainder-before.remainder;
  const auto prior_divisions=context.work.euclidean_steps;
  const auto bezout=extended_gcd(before.modulus,modulus,context);
  step.euclidean_steps=context.work.euclidean_steps-prior_divisions;
  step.gcd=bezout.gcd;step.bezout_first=bezout.first;step.bezout_second=bezout.second;
  step.reduced_modulus=modulus/step.gcd;
  if(step.difference%step.gcd) {step.status=CrtStatus::Inconsistent;return step;}
  const auto reduced=step.reduced_modulus;
  step.correction_product=(static_cast<Wide>(step.difference/step.gcd)%reduced)*(bezout.first%reduced);
  count(context.work.correction_products);
  step.multiplier=normalize(step.correction_product,reduced);
  step.candidate_modulus=static_cast<Wide>(before.modulus)*reduced;count(context.work.period_products);
  const Wide displaced=static_cast<Wide>(before.modulus)*step.multiplier;count(context.work.residue_products);
  step.candidate_remainder=(static_cast<Wide>(before.remainder)+displaced)%step.candidate_modulus;
  step.status=step.candidate_modulus>max_period?CrtStatus::Overflow:CrtStatus::Complete;
  return step;
}
/// @brief Shared actual sequential solver, retaining merge records only on request.
/// @param residues Borrowed signed target residues.
/// @param moduli Borrowed positive moduli, validated in full before arithmetic.
/// @param options Complete-call attempted-merge/division limits.
/// @param record Whether to retain actual merge certificates.
/// @return Complete verdict or first arithmetic failure, with actual prefix and counters.
/// @throws std::invalid_argument If lengths differ or any modulus is nonpositive.
/// @throws std::length_error If a work cap or record vector capacity is exceeded.
/// @throws std::overflow_error If an actual work counter overflows.
/// @throws std::bad_alloc If requested record storage allocation fails.
CrtTrace run(const std::vector<std::int64_t>& residues,const std::vector<std::int64_t>& moduli,
             CrtOptions options,bool record) {
  Context context{options,{}};context.work.validated_moduli=validate(residues,moduli);
  CrtTrace out;CrtSolution accumulator{true,0,1};
  for(std::size_t i=0;i<moduli.size();++i) {
    const auto step=merge(accumulator,residues[i],moduli[i],i,context);
    if(record)out.steps.push_back(step);
    if(step.status!=CrtStatus::Complete) {
      out.status=step.status;out.prefix=accumulator;out.work=context.work;
      if(step.status==CrtStatus::Inconsistent)out.solution=CrtSolution{false,0,0};
      return out;
    }
    accumulator={true,static_cast<std::int64_t>(step.candidate_remainder),static_cast<std::int64_t>(step.candidate_modulus)};
    count(context.work.completed_merges);
  }
  out.status=CrtStatus::Complete;out.solution=accumulator;out.prefix=accumulator;out.work=context.work;return out;
}
} // namespace crt_detail
std::int64_t crt_normalize(std::int64_t value,std::int64_t modulus) {
  if(modulus<=0)throw std::invalid_argument("CRT: normalization modulus must be positive");
  return crt_detail::normalize(value,modulus);
}
bool crt_residue_satisfies(std::int64_t candidate,const std::vector<std::int64_t>& remainders,
                          const std::vector<std::int64_t>& moduli) {
  (void)crt_detail::validate(remainders,moduli);
  for(std::size_t i=0;i<moduli.size();++i)
    if(crt_detail::normalize(candidate,moduli[i])!=crt_detail::normalize(remainders[i],moduli[i]))return false;
  return true;
}
CrtSolution chinese_remainder(const std::vector<std::int64_t>& remainders,const std::vector<std::int64_t>& moduli) {
  return chinese_remainder_with_options(remainders,moduli,{});
}
CrtSolution chinese_remainder_with_options(const std::vector<std::int64_t>& remainders,
    const std::vector<std::int64_t>& moduli,CrtOptions options) {
  const auto out=crt_detail::run(remainders,moduli,options,false);
  if(out.status==CrtStatus::Overflow)throw std::overflow_error("chinese_remainder: compatible prefix period exceeds int64");
  return *out.solution;
}
CrtTrace chinese_remainder_trace(const std::vector<std::int64_t>& remainders,
    const std::vector<std::int64_t>& moduli,CrtOptions options) {
  return crt_detail::run(remainders,moduli,options,true);
}
} // namespace datamunge::algorithms
