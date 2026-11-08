/// @file modular_exponentiation.cpp
/// @brief Shared exact full-product arithmetic and actual right-to-left powering states.
#include <datamunge/algorithms/modular_exponentiation.hpp>
#include <bit>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Internal arithmetic after validating a nonzero modulus; no shared state.
namespace {
/// @brief Reduce the complete two-word product after nonzero-modulus validation.
/// @param a First arbitrary unsigned factor.
/// @param b Second arbitrary unsigned factor.
/// @param m Previously validated nonzero modulus.
/// @return Exact product residue, without an intermediate 64-bit wrap.
std::uint64_t product(std::uint64_t a,std::uint64_t b,std::uint64_t m) {
 return static_cast<std::uint64_t>((static_cast<unsigned __int128>(a)*b)%m);
}
/// @brief Execute the existing fold/square/shift order, optionally retaining actual states.
/// @param base Original base, reduced before powering.
/// @param exponent Exponent consumed low bit first.
/// @param modulus Previously validated nonzero modulus.
/// @param trace Null for scalar calls; otherwise receives actual transition records/counts.
/// @return Exact residue, with modulus one returning zero without any products.
/// @throws std::bad_alloc If a history append allocates and fails.
std::uint64_t power(std::uint64_t base,std::uint64_t exponent,std::uint64_t modulus,ModPowTrace* trace) {
 if(modulus==1)return 0;
 std::uint64_t accumulator=1,running=base%modulus;std::size_t bit=0;
 while(exponent>0) {
  ModPowStep step;step.bit_index=bit++;step.remaining_before=exponent;step.running_before=running;step.accumulator_before=accumulator;step.multiply=(exponent&1)!=0;
  if(step.multiply)accumulator=product(accumulator,running,modulus);
  step.accumulator_after=accumulator;running=product(running,running,modulus);step.squared_base=running;exponent>>=1;
  if(trace){trace->steps.push_back(step);++trace->work.bits;++trace->work.squarings;++trace->work.products;if(step.multiply){++trace->work.accumulator_products;++trace->work.products;}}
 }
 return accumulator;
}
} // namespace
std::uint64_t mul_mod(std::uint64_t a,std::uint64_t b,std::uint64_t m) {
 if(m==0)throw std::invalid_argument("mul_mod: modulus must be nonzero");return product(a,b,m);
}
std::uint64_t mod_pow(std::uint64_t base,std::uint64_t exponent,std::uint64_t modulus) {
 if(modulus==0)throw std::invalid_argument("mod_pow: modulus must be nonzero");return power(base,exponent,modulus,nullptr);
}
ModPowTrace mod_pow_trace(std::uint64_t base,std::uint64_t exponent,std::uint64_t modulus,std::size_t max_steps) {
 if(modulus==0)throw std::invalid_argument("mod_pow_trace: modulus must be nonzero");
 const std::size_t size=modulus==1?0:std::bit_width(exponent);
 if(max_steps && size>max_steps)throw std::length_error("mod_pow_trace: complete bit count exceeds cap");
 ModPowTrace trace;trace.base=base;trace.exponent=exponent;trace.modulus=modulus;trace.initial_base=base%modulus;trace.steps.reserve(size);
 trace.result=power(base,exponent,modulus,&trace);return trace;
}
} // namespace datamunge::algorithms
