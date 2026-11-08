/// @file cipolla.cpp
/// @brief Shared full-range Cipolla search/powering with reduced sums, safe signed return and actual records.
#include <datamunge/algorithms/cipolla.hpp>
#include <datamunge/algorithms/miller_rabin.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private validation, counted scalar products and one ordinary/traced extension kernel.
namespace cipolla_detail {
/// @brief Unsigned working residue/modulus type.
using U64=std::uint64_t;
/// @brief Exact product type for two uint64 operands; sums are reduced before combination.
using Wide=unsigned __int128;
/// @brief Increment one actual named operation counter without wrapping.
/// @param value Mutable size_t count.
/// @throws std::overflow_error If no next count is representable.
void count(std::size_t& value) {
  if(value==std::numeric_limits<std::size_t>::max())throw std::overflow_error("Cipolla: work counter overflow");++value;
}
/// @brief Validate the complete uint64 prime domain before any remainder/shortcut.
/// @param p Caller-supplied modulus.
/// @throws std::invalid_argument If p is zero, one or composite.
void validate(U64 p) {
  if(!is_probable_prime(p))throw std::invalid_argument("Cipolla: modulus must be prime");
}
/// @brief Compute one actual modular product in unsigned 128-bit arithmetic.
/// @param a Canonical first residue.
/// @param b Canonical second residue.
/// @param p Validated prime modulus.
/// @param counter Named category incremented for this executed operation.
/// @return Canonical a*b modulo p.
/// @throws std::overflow_error If counter cannot be incremented.
U64 product(U64 a,U64 b,U64 p,std::size_t& counter) {
  count(counter);return static_cast<U64>(static_cast<Wide>(a)*b%p);
}
/// @brief Add already reduced components with a wide sum, never a sum of unreduced products.
/// @param a First residue below p.
/// @param b Second residue below p.
/// @param p Positive validated modulus.
/// @return Canonical sum modulo p.
U64 add(U64 a,U64 b,U64 p) {return static_cast<U64>((static_cast<Wide>(a)+b)%p);}
/// @brief Subtract canonical residues without forming a+p-b.
/// @param a Canonical minuend.
/// @param b Canonical subtrahend.
/// @param p Positive validated modulus.
/// @return Canonical difference modulo p.
U64 subtract(U64 a,U64 b,U64 p) {return a>=b?a-b:p-(b-a);}
/// @brief Execute original scalar powering order, including the closing unused square.
/// @param base Canonical scalar base.
/// @param exponent Nonnegative exponent.
/// @param p Validated prime modulus.
/// @param counter Actual modular-product category for every multiply and square.
/// @return Canonical base^exponent modulo p.
/// @throws std::overflow_error If the category counter cannot fit.
U64 power(U64 base,U64 exponent,U64 p,std::size_t& counter) {
  U64 result=1;
  while(exponent){if(exponent&1)result=product(result,base,p,counter);base=product(base,base,p,counter);exponent>>=1;}
  return result;
}
/// @brief Per-call work and admission caps, with no shared state.
struct Context {
  CipollaOptions options; ///< Positive whole-call limits; zero is unlimited.
  CipollaWork work; ///< Actual named operations outside primality-test internals.
  std::size_t extension_products{0}; ///< Actual combined result/base pair operations admitted.

  /// @brief Admit one actual auxiliary candidate before square/Euler arithmetic.
  /// @throws std::length_error If the positive candidate cap is exhausted.
  /// @throws std::overflow_error If the candidate count cannot fit size_t.
  void candidate() {
    if(options.max_candidates && work.candidate_tests==options.max_candidates)throw std::length_error("Cipolla: candidate cap exhausted");
    count(work.candidate_tests);
  }
  /// @brief Admit one actual result multiplication or base square before its scalar products.
  /// @param square Whether the actual operation squares the current base.
  /// @throws std::length_error If the positive extension-product cap is exhausted.
  /// @throws std::overflow_error If an operation counter cannot fit.
  void extension(bool square) {
    if(options.max_extension_products && extension_products==options.max_extension_products)throw std::length_error("Cipolla: extension-product cap exhausted");
    count(extension_products);count(square?work.extension_base_squares:work.extension_result_products);
  }
};
/// @brief Multiply two canonical x+y*s pairs using five actual reduced scalar products.
/// @param a First pair in the call's extension.
/// @param b Second pair in the same extension.
/// @param w Validated quadratic nonresidue s^2.
/// @param p Validated odd prime modulus.
/// @param context Actual scalar-product category.
/// @return Canonical pair product.
/// @throws std::overflow_error If its scalar-product counter cannot fit.
CipollaElement multiply(CipollaElement a,CipollaElement b,U64 w,U64 p,Context& context) {
  auto& counter=context.work.extension_scalar_products;
  const auto xx=product(a.x,b.x,p,counter),yy=product(a.y,b.y,p,counter);
  const auto wyy=product(yy,w,p,counter),xy=product(a.x,b.y,p,counter),yx=product(b.x,a.y,p,counter);
  return {add(xx,wyy,p),add(xy,yx,p)};
}
/// @brief Build complete unsigned/legacy root outputs without an out-of-range signed cast.
/// @param root Actual canonical root from a validated path.
/// @param p Validated prime modulus.
/// @return Full root pair and a representable nonnegative legacy root.
CipollaResult roots(U64 root,U64 p) {
  const auto other=root?p-root:0;
  const auto limit=static_cast<U64>(std::numeric_limits<long long>::max());
  const auto legacy=root<=limit?root:other;
  return {true,root,other,static_cast<long long>(legacy)};
}
/// @brief Execute checked validation, shortcuts, deterministic search and extension powering once.
/// @tparam Record Whether to retain actual candidate and exponent-bit states.
/// @param input_n Original arbitrary uint64 residue.
/// @param p Caller modulus, validated before reducing input_n.
/// @param options Whole-call actual candidate/pair-product caps.
/// @return Complete result, path, counters and optionally actual records.
/// @throws std::invalid_argument If p is not prime.
/// @throws std::length_error If a work cap or actual vector capacity is exceeded.
/// @throws std::overflow_error If an actual work counter cannot fit size_t.
/// @throws std::logic_error If a validated-field invariant or final root check fails.
/// @throws std::bad_alloc If requested record allocation fails.
template<bool Record>
CipollaTrace run(U64 input_n,U64 p,CipollaOptions options) {
  Context context{options,{},{}};count(context.work.primality_checks);validate(p);
  CipollaTrace out;out.input_n=input_n;out.prime=p;out.normalized_n=input_n%p;const auto n=out.normalized_n;
  if(p==2){out.path=CipollaPath::BinaryField;out.result=roots(n,p);out.work=context.work;return out;}
  if(!n){out.path=CipollaPath::Zero;out.result=roots(0,p);out.work=context.work;return out;}
  out.residue_euler=power(n,(p-1)/2,p,context.work.residue_power_products);
  if(out.residue_euler!=1){out.path=CipollaPath::NonResidue;out.work=context.work;return out;}
  out.path=CipollaPath::Extension;U64 a=1,w=0;
  for(;;) {
    context.candidate();w=subtract(product(a,a,p,context.work.candidate_squares),n,p);
    const auto euler=power(w,(p-1)/2,p,context.work.candidate_power_products);
    if constexpr(Record)out.candidates.push_back({a,w,euler});
    if(euler==p-1)break;
    a=a==p-1?0:a+1;
    if(a==1)throw std::logic_error("Cipolla: no nonresidue auxiliary in a validated field");
  }
  out.a=a;out.w=w;CipollaElement result{1,0},base{a,1};U64 exponent=p/2+1;
  while(exponent) {
    CipollaPowerStep step;step.remaining_exponent=exponent;step.result_before=result;step.base_before=base;step.multiplied=exponent&1;
    if(step.multiplied){context.extension(false);result=multiply(result,base,w,p,context);}
    context.extension(true);base=multiply(base,base,w,p,context);
    step.result_after=result;step.base_after=base;
    if constexpr(Record)out.steps.push_back(step);
    exponent>>=1;
  }
  const auto check=product(result.x,result.x,p,context.work.verification_products);
  if(result.y!=0 || check!=n)throw std::logic_error("Cipolla: final extension root invariant failed");
  out.final_element=result;out.result=roots(result.x,p);out.work=context.work;return out;
}
} // namespace cipolla_detail
long long cipolla_sqrt(std::uint64_t n,std::uint64_t p) {return cipolla_detail::run<false>(n,p,{}).result.legacy_root;}
CipollaResult cipolla_sqrt_result(std::uint64_t n,std::uint64_t p,CipollaOptions options) {return cipolla_detail::run<false>(n,p,options).result;}
CipollaTrace cipolla_sqrt_trace(std::uint64_t n,std::uint64_t p,CipollaOptions options) {return cipolla_detail::run<true>(n,p,options);}
bool cipolla_root_satisfies(std::uint64_t n,std::uint64_t p,std::uint64_t root) {
  cipolla_detail::validate(p);root%=p;return static_cast<cipolla_detail::Wide>(root)*root%p==n%p;
}
} // namespace datamunge::algorithms
