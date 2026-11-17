#pragma once
/// @file long_division.hpp
/// @brief Checked fixed-width binary long division and actual bit-step capture.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Exact unsigned quotient, remainder and actual bit count.
/// @code
/// auto r=dm::algorithms::long_division(100,7);
/// std::print("{} {}\n",r.quotient,r.remainder); // 14 2
/// @endcode
struct DivResult {
  std::uint64_t quotient{0};   ///< Floor of dividend divided by divisor.
  std::uint64_t remainder{0};  ///< Nonnegative remainder strictly below divisor.
  int           iterations{0}; ///< Actual processed bits, always 64 on success.
};
/// @brief Actual state of one producing shift/compare/subtract bit step.
/// @code
/// auto t=dm::algorithms::long_division_trace(220,13);
/// std::print("{}\n",t.steps.back().remainder); // 12
/// @endcode
struct LongDivisionStep {
  int           bit;        ///< Original dividend bit position, descending 63 through 0.
  std::uint64_t before;     ///< Remainder before bringing down the next bit.
  std::uint64_t trial;      ///< Exact remainder after bringing down the next bit.
  bool          subtracted; ///< Whether trial was at least the divisor.
  std::uint64_t quotient;   ///< Actual quotient bits accumulated in original positions.
  std::uint64_t remainder;  ///< Actual remainder after optional subtraction.
};
/// @brief Owned complete actual bit execution and returned quotient/remainder.
/// @code
/// auto t=dm::algorithms::long_division_trace(220,13);
/// std::print("{}\n",t.steps.size()); // 64
/// @endcode
struct LongDivisionTrace {
  std::uint64_t                 dividend; ///< Original dividend.
  std::uint64_t                 divisor;  ///< Original nonzero divisor.
  DivResult                     result;   ///< Actual final producing result.
  std::vector<LongDivisionStep> steps;    ///< All 64 real steps, including leading zero bits.
};
/// @brief Divide in 64 actual shift/compare/subtract steps, using constant auxiliary storage.
/// @param dividend Full-range unsigned input.
/// @param divisor Nonzero unsigned divisor.
/// @return Exact quotient, remainder and 64 actual iterations.
/// @throws std::invalid_argument If divisor is zero.
/// @code{.cpp}
/// auto d=datamunge::algorithms::long_division(220,13);
/// @endcode
DivResult long_division(std::uint64_t dividend, std::uint64_t divisor);
/// @brief Capture the same producing division, without replay or significant-bit reconstruction.
/// @param dividend Full-range unsigned input.
/// @param divisor Nonzero unsigned divisor.
/// @param max_entries Complete logical capture quota: two inputs plus 64 records; zero disables it.
/// @return Owned original inputs, result and all actual bit steps.
/// @throws std::invalid_argument If divisor is zero.
/// @throws std::length_error If a nonzero quota is below 66, before producing/copying steps.
/// @code{.cpp}
/// auto t=datamunge::algorithms::long_division_trace(100,7);
/// @endcode
LongDivisionTrace long_division_trace(
    std::uint64_t dividend,
    std::uint64_t divisor,
    std::size_t   max_entries = 1000000);
} // namespace datamunge::algorithms
