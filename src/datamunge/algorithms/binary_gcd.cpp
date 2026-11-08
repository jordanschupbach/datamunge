/// @file binary_gcd.cpp
/// @brief Shared checked-width unsigned binary-GCD loop, with optional recording.
#include <datamunge/algorithms/binary_gcd.hpp>
#include <algorithm>
#include <bit>

namespace datamunge::algorithms {
/// @brief Private canonical binary-GCD implementation.
namespace {
/// @brief Reduce odd pairs with optional owned snapshots; never count trailing zeros of zero.
/// @param a First operand, passed by value.
/// @param b Second operand, passed by value.
/// @param trace Nullable borrowed result record, initially empty.
/// @return Exact unsigned greatest common divisor.
std::uint64_t gcd_loop(std::uint64_t a,std::uint64_t b,BinaryGcdTrace* trace) {
  if (a==0) return b;
  if (b==0) return a;
  const unsigned common=std::countr_zero(a|b);
  a >>= std::countr_zero(a);
  if (trace) {
    trace->common_twos=common;
    trace->initial_odd_a=a;
  }
  do {
    const unsigned removed=std::countr_zero(b);
    b >>= removed;
    if (a>b) std::swap(a,b);
    if (trace) trace->steps.push_back({a,b,b-a,removed});
    b-=a;
  } while (b!=0);
  return a<<common;
}
} // namespace

std::uint64_t binary_gcd(std::uint64_t a,std::uint64_t b) {
  return gcd_loop(a,b,nullptr);
}

BinaryGcdTrace binary_gcd_trace(std::uint64_t a,std::uint64_t b) {
  BinaryGcdTrace trace;
  trace.gcd=gcd_loop(a,b,&trace);
  return trace;
}
} // namespace datamunge::algorithms
