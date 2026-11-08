/// @file chakravala.cpp
/// @brief Checked cyclic composition using quotient/remainder arithmetic and actual provenance.
#include <datamunge/algorithms/chakravala.hpp>
#include <array>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private fixed-word arithmetic and shared ordinary/traced cyclic kernel.
namespace chakravala_detail {
/// @brief Unsigned working type; products of two uint64 values fit exactly.
using Wide=unsigned __int128;
/// @brief Largest permitted positive coordinate in the legacy signed result type.
constexpr Wide coordinate_max=(Wide{1}<<127)-1;
/// @brief Compute a nonnegative Euclidean gcd without allocation.
/// @param a First unsigned working value.
/// @param b Second unsigned working value.
/// @return gcd(a,b), including gcd(0,0)=0.
Wide gcd(Wide a,Wide b) noexcept {
  while(b) {const auto remainder=a%b;a=b;b=remainder;}return a;
}
/// @brief Invert a coprime residue modulo a positive uint64 modulus.
/// Extended-Euclidean coefficients fit signed 128-bit for uint64 inputs.
/// @param value Residue in [0,modulus).
/// @param modulus Positive modulus greater than one.
/// @return Least nonnegative inverse.
/// @throws std::logic_error If the solver's coprimality invariant fails.
std::uint64_t inverse(std::uint64_t value,std::uint64_t modulus) {
  std::uint64_t left=modulus,right=value;
  __int128 first=0,second=1;
  while(right) {
    const auto quotient=left/right,remainder=left%right;
    const auto next=first-static_cast<__int128>(quotient)*second;
    left=right;right=remainder;first=second;second=next;
  }
  if(left!=1)throw std::logic_error("chakravala: nonprimitive congruence");
  first%=modulus;if(first<0)first+=modulus;return static_cast<std::uint64_t>(first);
}
/// @brief Owned quotient/remainder of an unmaterialized 128-by-64 product.
struct DividedProduct {
  Wide quotient{0}; ///< Checked quotient, at most the positive coordinate limit.
  std::uint64_t remainder{0}; ///< Exact remainder, less than the divisor.
};
/// @brief Divide x*multiplier by divisor without forming the full product.
/// @param x Nonnegative coordinate, at most coordinate_max.
/// @param multiplier Any uint64 coefficient.
/// @param divisor Positive uint64 divisor.
/// @return Exact quotient and remainder, or no result on overflow.
/// @throws std::overflow_error If the quotient exceeds coordinate_max.
DividedProduct divided_product(Wide x,std::uint64_t multiplier,std::uint64_t divisor) {
  const auto whole=x/divisor,part=x%divisor;
  if(multiplier && whole>coordinate_max/multiplier)
    throw std::overflow_error("chakravala: reduced coordinate exceeds signed 128-bit range");
  const auto large=whole*multiplier,small=part*multiplier;
  const auto extra=small/divisor;
  if(extra>coordinate_max-large)
    throw std::overflow_error("chakravala: reduced coordinate exceeds signed 128-bit range");
  return {large+extra,static_cast<std::uint64_t>(small%divisor)};
}
/// @brief Evaluate the exact reduced nonnegative sum (a*u+b*v)/divisor.
/// Combines individually divided products and a remainder carry; every term
/// is nonnegative, so overflow of one quotient cannot hide a representable sum.
/// @param a First nonnegative coordinate.
/// @param u First uint64 coefficient.
/// @param b Second nonnegative coordinate.
/// @param v Second uint64 coefficient.
/// @param divisor Positive shared divisor.
/// @return Exact signed-128 reduced coordinate.
/// @throws std::overflow_error If the reduced result exceeds coordinate_max.
/// @throws std::logic_error If exact divisibility, an internal invariant, fails.
__int128 scaled_sum(__int128 a,std::uint64_t u,__int128 b,std::uint64_t v,std::uint64_t divisor) {
  const auto first=divided_product(static_cast<Wide>(a),u,divisor);
  const auto second=divided_product(static_cast<Wide>(b),v,divisor);
  const Wide remainders=static_cast<Wide>(first.remainder)+second.remainder;
  if(remainders%divisor)throw std::logic_error("chakravala: inexact composition");
  if(second.quotient>coordinate_max-first.quotient)
    throw std::overflow_error("chakravala: reduced coordinate exceeds signed 128-bit range");
  auto result=first.quotient+second.quotient;
  const auto carry=remainders/divisor;
  if(carry>coordinate_max-result)
    throw std::overflow_error("chakravala: reduced coordinate exceeds signed 128-bit range");
  return static_cast<__int128>(result+carry);
}
/// @brief Select the actual minimum-norm positive member of the congruence class.
/// Only the two progression elements bracketing sqrt(N) can minimise the norm.
/// @param N Nonsquare coefficient.
/// @param root Exact floor square root.
/// @param state Current primitive positive-coordinate, nonzero-norm triple.
/// @return Actual class, candidates, objective values and smaller-on-ties choice.
/// @throws std::overflow_error If an auxiliary representation exceeds uint64.
/// @throws std::logic_error If the primitive congruence invariant fails.
ChakravalaChoice choose(std::uint64_t N,std::uint64_t root,const ChakravalaState& state) {
  const Wide magnitude=state.k<0?static_cast<Wide>(-(state.k+1))+1:static_cast<Wide>(state.k);
  if(!magnitude || magnitude>std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("chakravala: auxiliary modulus outside uint64");
  ChakravalaChoice result;result.modulus=static_cast<std::uint64_t>(magnitude);
  const auto h=result.modulus;
  if(h>1) {
    const auto a=static_cast<std::uint64_t>(static_cast<Wide>(state.a)%h);
    const auto b=static_cast<std::uint64_t>(static_cast<Wide>(state.b)%h);
    result.residue=static_cast<std::uint64_t>((static_cast<Wide>((h-a)%h)*inverse(b,h))%h);
  }
  if(result.residue<=root) {
    result.below=result.residue+((root-result.residue)/h)*h;
    const Wide upper=static_cast<Wide>(result.below)+h;
    if(upper>std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("chakravala: auxiliary candidate outside uint64");
    result.above=static_cast<std::uint64_t>(upper);
  } else result.above=result.residue;
  if(result.below)result.below_error=static_cast<Wide>(N)-static_cast<Wide>(result.below)*result.below;
  result.above_error=static_cast<Wide>(result.above)*result.above-N;
  result.chosen=result.below && result.below_error<=result.above_error?result.below:result.above;
  return result;
}
/// @brief Shared actual setup and update kernel, optionally retaining owned records.
/// @param N Nonnegative uint64 coefficient.
/// @param max_steps Strict maximum complete cyclic updates, including zero.
/// @param trace Optional owned output record, populated during execution.
/// @return Completed Pell pair or square-input sentinel.
/// @throws std::overflow_error If a reduced coordinate or auxiliary representation exceeds its range.
/// @throws std::length_error If max_steps or vector capacity is exhausted.
/// @throws std::bad_alloc If trace allocation fails.
/// @throws std::logic_error If an internal arithmetic invariant fails.
PellSolution solve(std::uint64_t N,std::size_t max_steps,ChakravalaTrace* trace) {
  const auto root=integer_sqrt_u64(N);
  ChakravalaState state{static_cast<__int128>(root),1,static_cast<__int128>(static_cast<Wide>(root)*root)-N};
  if(trace) {trace->N=N;trace->initial=state;}
  if(!state.k)return {};
  std::size_t count=0;
  while(state.k!=1) {
    if(count==max_steps)throw std::length_error("chakravala: cyclic update budget exhausted");
    const auto choice=choose(N,root,state);
    const auto a=scaled_sum(state.a,choice.chosen,state.b,N,choice.modulus);
    const auto b=scaled_sum(state.a,1,state.b,choice.chosen,choice.modulus);
    const Wide square=static_cast<Wide>(choice.chosen)*choice.chosen;
    const auto difference=square<N?static_cast<Wide>(N)-square:square-N;
    if(difference%choice.modulus)throw std::logic_error("chakravala: inexact norm reduction");
    const auto norm=difference/choice.modulus;
    if(norm>std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("chakravala: auxiliary norm outside uint64");
    const bool negative=(square<N)!=(state.k<0);
    const ChakravalaState next{a,b,negative?-static_cast<__int128>(norm):static_cast<__int128>(norm)};
    if(trace)trace->steps.push_back({state,choice,next});
    state=next;++count;
  }
  return {state.a,state.b,true};
}
} // namespace chakravala_detail
PellSolution chakravala(std::uint64_t N) {return chakravala_detail::solve(N,100000,nullptr);}
ChakravalaTrace chakravala_trace(std::uint64_t N,std::size_t max_steps) {
  ChakravalaTrace result;result.solution=chakravala_detail::solve(N,max_steps,&result);return result;
}
bool pell_is_solution(std::uint64_t N,const PellSolution& solution) noexcept {
  if(!solution.found || N<2 || solution.x<=1 || solution.y<=0)return false;
  using chakravala_detail::Wide;
  const Wide x=static_cast<Wide>(solution.x),y=static_cast<Wide>(solution.y);
  std::array<Wide,2> left{x-1,x+1};std::array<Wide,3> right{N,y,y};
  for(auto& a:left)for(auto& b:right) {const auto factor=chakravala_detail::gcd(a,b);a/=factor;b/=factor;}
  for(const auto a:left)if(a!=1)return false;
  for(const auto b:right)if(b!=1)return false;
  return true;
}
} // namespace datamunge::algorithms
