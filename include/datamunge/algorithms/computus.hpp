#pragma once
/// @file computus.hpp
/// @brief Checked positive-year Gregorian Easter arithmetic, actual integer steps and bounded range summaries.
#include <array>
#include <cstddef>
#include <cstdint>

namespace datamunge::algorithms {

/// @brief Owned Gregorian Easter month/day with the historical field order and types.
/// Computed dates lie from March 22 through April 25. Value-initialized zeros are a
/// placeholder, not a computed date. No year, timezone, input reference or
/// historical calendar-adoption information is retained.
/// @par Example
/// @code
/// const auto e=dm::algorithms::computus_gregorian(2024);
/// std::print("{} {}\n",e.month,e.day); // 3 31
/// @endcode
/// @see computus_gregorian(), ComputusGregorianTrace
struct EasterDate {
  int month; ///< Computed month, three or four; value initialization gives a zero placeholder.
  int day; ///< Computed day of month; value initialization gives a zero placeholder.
};

/// @brief Owned actual Meeus/Jones/Butcher integer steps and the resulting date.
/// All divisions are floors of nonnegative integers for supported positive years.
/// a is a zero-based Metonic index; the golden number is a+1. h is a nominal
/// lunar offset, not an unconditional observed or corrected full-moon date.
/// l+1 is the nominal following-Sunday offset. m subtracts one week in the two
/// exceptional Sunday cases. No iteration, replay, allocation or shared state.
/// @par Example
/// @code
/// const auto t=dm::algorithms::computus_gregorian_trace(1954);
/// std::print("{} {} {} {} {}\n",t.a,t.h,t.l,t.m,t.date.day); // 16 28 6 1 18
/// @endcode
/// @see computus_gregorian_trace()
struct ComputusGregorianTrace {
  int year=0; ///< Validated positive Gregorian rule year.
  int a=0; ///< year modulo 19, in [0,18]; golden number is a+1.
  int b=0; ///< Complete centuries, floor(year/100).
  int c=0; ///< Year within century, year modulo 100.
  int d=0; ///< Complete groups of four centuries, floor(b/4).
  int e=0; ///< Century modulo four.
  int f=0; ///< Auxiliary lunar correction, floor((b+8)/25).
  int g=0; ///< Auxiliary Gregorian correction, floor((b-f+1)/3).
  int h=0; ///< Nominal lunar offset (19a+b-d-g+15) modulo 30, in [0,29].
  int i=0; ///< Complete groups of four years within century, floor(c/4).
  int k=0; ///< Year-within-century modulo four.
  int l=0; ///< (32+2e+2i-h-k) modulo seven, in [0,6].
  int m=0; ///< floor((a+11h+22l)/451), zero or one.
  int encoding=0; ///< h+l-7m+114, used to encode month and day.
  int uncorrected_march_day=0; ///< One-based March coordinate 22+h+l, before subtracting 7m.
  int march_day=0; ///< One-based March coordinate of final Easter; March 22=22, April 25=56.
  EasterDate date{}; ///< Computed final month/day, preserving the legacy layout.
};

/// @brief Owned first-tie extrema and 35 date counts for an inclusive year range.
/// Bins zero through 34 mean March 22 through April 25 on a fixed March axis.
/// Counts describe deterministic Gregorian ecclesiastical rules over the selected
/// years, not an astronomical forecast or a random-sampling probability model.
/// Earliest/latest compare month/day, not chronology; equal dates keep the first
/// ascending year. Storage is constant, with no per-year records or allocation.
/// Default storage alone is not a valid summary.
/// @par Example
/// @code
/// const auto s=dm::algorithms::computus_gregorian_summary(2000,2040);
/// std::print("{} {} {}\n",s.earliest_year,s.latest_year,s.spread_days); // 2008 2038 33
/// @endcode
/// @see computus_gregorian_summary()
struct EasterRangeSummary {
  int first_year=0; ///< Inclusive validated first rule year.
  int last_year=0; ///< Inclusive validated last rule year.
  std::uint64_t year_count=0; ///< Complete inclusive count; equals the sum of date_counts.
  std::array<std::uint64_t,35> date_counts{}; ///< Owned fixed-axis counts, index=March coordinate minus 22.
  std::uint64_t corrected_years=0; ///< Number of actual m=1 week corrections.
  int earliest_year=0; ///< First year attaining the smallest month/day coordinate.
  int latest_year=0; ///< First year attaining the largest month/day coordinate.
  EasterDate earliest{}; ///< Earliest month/day among these years.
  EasterDate latest{}; ///< Latest month/day among these years.
  int spread_days=0; ///< Latest minus earliest fixed March coordinate, from zero through 34.
};

/// @brief Calculate Gregorian ecclesiastical Easter for a positive int year.
/// Uses the Anonymous Gregorian / Meeus-Jones-Butcher integer rule. Year one
/// through INT_MAX are supported; zero/BCE numbering is rejected. Before local
/// Gregorian adoption the answer is a proleptic rule projection, not a historical
/// observance claim. The moon/equinox are ecclesiastical tables and fixed March
/// 21, not measured astronomical events. Results range March 22 through April 25.
/// All intermediate int expressions are bounded for positive int input.
/// Constant fixed-width work/storage, no allocation, mutation or shared state.
/// @param year Positive Gregorian rule year, from one through INT_MAX.
/// @return Owned Easter month/day from the same kernel as the traced call.
/// @throws std::invalid_argument If year is zero or negative.
/// @par Example
/// @code
/// const auto e=dm::algorithms::computus_gregorian(2038);
/// std::print("{} {}\n",e.month,e.day); // 4 25
/// @endcode
/// @par References
/// - Anonymous (1876), To Find Easter, Nature 13, 487, doi:10.1038/013487a0
///   (historical arithmetic rule; the modern m-form is written explicitly here).
/// - U.S. Naval Observatory, The Date of Easter,
///   https://aa.usno.navy.mil/faq/easter (ecclesiastical definitions).
/// @see computus_gregorian_trace(), computus_gregorian_summary()
EasterDate computus_gregorian(int year);

/// @brief Return actual nonnegative integer steps without replaying the Easter calculation.
/// Uses exactly the same positive-year validation and Gregorian rule as
/// computus_gregorian(). Retains standard symbols a,b,c,d,e,f,g,h,i,k,l,m so the
/// result can be checked against the published equations. In particular, m=1
/// exactly when l=6 and either h=29 or h=28 with a>=11. The uncorrected March
/// coordinate may be 57 (April 26); the final date always lies in [22,56].
/// Constant work/storage; owned scalar data, no heap or shared state.
/// @param year Positive rule year in [1,INT_MAX]; earlier historical years are proleptic.
/// @return Actual intermediates, nominal March coordinate, final date and March coordinate.
/// @throws std::invalid_argument If year is zero or negative.
/// @par Example
/// @code
/// const auto t=dm::algorithms::computus_gregorian_trace(1981);
/// std::print("{} {} {}\n",t.h,t.uncorrected_march_day,t.march_day); // 29 57 50
/// @endcode
/// @par References
/// - Anonymous (1876), To Find Easter, Nature 13, 487, doi:10.1038/013487a0.
/// @see ComputusGregorianTrace, computus_gregorian()
ComputusGregorianTrace computus_gregorian_trace(int year);

/// @brief Summarize an inclusive positive-year interval using the canonical Easter kernel.
/// Fully validates both positive endpoints and their order, then preflights the
/// complete inclusive count against the positive cap before evaluating any year.
/// Scans ascending years once, accumulates all 35 bins and retains first ties for
/// date extrema. Loop termination is safe even when last_year=INT_MAX.
/// Time O(last-first+1), constant 35-bin storage, no allocation or shared state.
/// One complete 5,700,000-year rule period needs an explicit sufficient cap.
/// @param first_year Positive inclusive first year.
/// @param last_year Positive inclusive last year, at least first_year.
/// @param max_years Positive complete-work cap, default 100000; not an online record cap.
/// @return Owned exact date counts, correction count, first-tie extrema and spread.
/// @throws std::invalid_argument If either year is nonpositive, bounds are reversed or cap is zero.
/// @throws std::length_error If the complete inclusive count exceeds max_years.
/// @par Example
/// @code
/// const auto s=dm::algorithms::computus_gregorian_summary(2024,2024,1);
/// std::print("{} {} {}\n",s.year_count,s.date_counts[9],s.spread_days); // 1 1 0
/// @endcode
/// @see EasterRangeSummary, computus_gregorian_trace()
EasterRangeSummary computus_gregorian_summary(int first_year,int last_year,std::size_t max_years=100000);

} // namespace datamunge::algorithms
