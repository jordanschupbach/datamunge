/// @file chudnovsky.cpp
/// @brief Shared stable floating recurrence and log-domain exact-series diagnostics.
#include <datamunge/algorithms/chudnovsky.hpp>
#include <cmath>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private exact-series constants, bounded recurrence and optional actual recording.
namespace chudnovsky_detail {
/// @brief Constant linear-series coefficient, exactly representable in double.
constexpr double A=13591409;
/// @brief Slope of the linear-series coefficient, exactly representable in double.
constexpr double B=545140134;
/// @brief Exact integer C^3/24, used after cancelling consecutive factorial factors.
constexpr double cube_over_24=10939058860032000.0;
/// @brief Positive exact-series limiting magnitude of consecutive terms, rounded to long double.
constexpr long double rho=72.0L/10939058860032000.0L;
/// @brief Compute a ratio for a previously validated nonnegative int index.
/// @param index Current series index.
/// @return Finite negative double recurrence factor with bounded rational factors.
double ratio(int index) {
  const double k=index,n=k+1;
  return -(6-5/n)*(2-1/n)*(6-1/n)/cube_over_24*(1+B/(A+B*k));
}
/// @brief Execute the same accumulator path with optional retention of actual prefix values.
/// @tparam Record Whether to append actual snapshots and evaluate their displayed estimates.
/// @param terms Positive requested count, validated before any arithmetic.
/// @return Final estimate, actual counts and optional owned term records.
/// @throws std::invalid_argument If terms is nonpositive.
/// @throws std::length_error If actual record capacity is exceeded.
/// @throws std::bad_alloc If record allocation fails.
template<bool Record>
ChudnovskyTrace run(int terms) {
  if(terms<1)throw std::invalid_argument("chudnovsky_pi: terms must be positive");
  ChudnovskyTrace out;out.terms_requested=terms;out.stop=ChudnovskyStop::RequestedTerms;
  const double normalizer=426880.0*std::sqrt(10005.0);double sum=0,term=A;
  for(int k=0;k<terms;++k) {
    double factor=0;if(k){factor=ratio(k-1);term*=factor;++out.ratio_evaluations;}
    const auto before=sum;sum+=term;++out.sum_additions;++out.terms_evaluated;
    const bool changed=sum!=before;
    if(!changed && out.first_unchanged_index<0)out.first_unchanged_index=k;
    if constexpr(Record)out.steps.push_back({k,factor,term,before,sum,normalizer/sum,changed});
    if(term==0){out.stop=ChudnovskyStop::TermUnderflow;break;}
  }
  out.value=normalizer/sum;return out;
}
} // namespace chudnovsky_detail

double chudnovsky_term_ratio(int index) {
  if(index<0)throw std::invalid_argument("chudnovsky: ratio index must be nonnegative");
  return chudnovsky_detail::ratio(index);
}
double chudnovsky_digits_per_term() {return static_cast<double>(-std::log10(chudnovsky_detail::rho));}
double chudnovsky_log10_truncation_bound(int terms) {
  if(terms<1)throw std::invalid_argument("chudnovsky: truncation term count must be positive");
  const long double first=5.0L*(chudnovsky_detail::A+chudnovsky_detail::B)/chudnovsky_detail::cube_over_24;
  const long double lower=chudnovsky_detail::A-first;
  return static_cast<double>(std::log10(426880.0L)+0.5L*std::log10(10005.0L)+std::log10(first)
      +(static_cast<long double>(terms)-1)*std::log10(chudnovsky_detail::rho)-2*std::log10(lower));
}
double chudnovsky_pi(int terms) {return chudnovsky_detail::run<false>(terms).value;}
ChudnovskyTrace chudnovsky_pi_trace(int terms) {return chudnovsky_detail::run<true>(terms);}
} // namespace datamunge::algorithms
