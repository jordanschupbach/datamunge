#pragma once
/// @file doomsday.hpp
/// @brief Checked proleptic Gregorian Doomsday weekdays, actual arithmetic and monthly anchors.
#include <array>
#include <string>

namespace datamunge::algorithms {

/// @brief Owned actual year terms of Conway's Doomsday rule.
/// Uses astronomical year numbering, including zero and negative years, and
/// floor division for centuries. All weekday codes are Sunday=0 through
/// Saturday=6. The dozen decomposition is an equivalent mental-arithmetic
/// identity evaluated alongside the direct offset, not a second weekday solver.
/// Default fields are an uncomputed placeholder, not the terms of year zero.
/// O(1), no allocation, input references or shared mutable state.
/// @par Example
/// @code
/// const auto y=dm::algorithms::doomsday_year_trace(2025);
/// std::print("{} {} {}\n",y.century_anchor,y.year_in_century,y.doomsday); // 2 25 5
/// @endcode
/// @par References
/// - S. W. Graham (1995), The Doomsday Rule (notes on Conway's method),
///   https://people.se.cmich.edu/graha1sw/pub/doomsday/Doomsday.pdf
/// @see doomsday_year_trace(), DoomsdayTrace
struct DoomsdayYear {
  int year=0; ///< Original astronomical int year; zero means 1 BCE.
  int century=0; ///< floor(year/100), including negative centuries.
  int century_mod4=0; ///< Nonnegative century residue in [0,3].
  int century_anchor=0; ///< (5*century_mod4+2) modulo seven.
  int year_in_century=0; ///< year-100*century in [0,99].
  int leap_quarters=0; ///< floor(year_in_century/4), the direct within-century leap offset.
  int dozens=0; ///< floor(year_in_century/12), for the mental-arithmetic identity.
  int dozen_remainder=0; ///< year_in_century modulo twelve.
  int dozen_quarters=0; ///< floor(dozen_remainder/4).
  int doomsday=0; ///< Weekday of the last day of February, in [0,6].
  bool leap=false; ///< Gregorian leap status, including century exceptions.
};

/// @brief Owned actual year terms and signed monthly offset for a validated date.
/// No replay: doomsday_weekday() calls this same executing kernel. The monthly
/// reference is January3/4, February28/29, March14, April4, May9, June6,
/// July11, August8, September5, October10, November7 or December12.
/// All fields are values; no allocation or references into caller storage.
/// @par Example
/// @code
/// const auto t=dm::algorithms::doomsday_date_trace(2025,12,25);
/// std::print("{} {} {}\n",t.reference_day,t.offset,t.weekday); // 12 13 4
/// @endcode
/// @par References
/// - S. W. Graham (1995), The Doomsday Rule (notes on Conway's method),
///   https://people.se.cmich.edu/graha1sw/pub/doomsday/Doomsday.pdf
/// @see doomsday_date_trace(), DoomsdayYear
struct DoomsdayTrace {
  DoomsdayYear year; ///< Actual century, within-century and leap terms.
  int month=0; ///< Valid month in [1,12]; zero in an uncomputed placeholder.
  int day=0; ///< Valid one-based day of month; zero in a placeholder.
  int reference_day=0; ///< This month's easy-to-remember date sharing the year's doomsday.
  int offset=0; ///< Signed day-reference_day, before modular reduction.
  int weekday=0; ///< (year.doomsday+offset) modulo seven, normalized to [0,6].
};

/// @brief Owned monthly anchor with its one-based day of year and computed weekday.
/// All twelve anchors for one year share its doomsday; differences between their
/// ordinal days are multiples of seven. All fields are values, without references.
/// @par Example
/// @code
/// const auto a=dm::algorithms::doomsday_month_anchors(2024);
/// std::print("{} {}\n",a[1].day,a[1].day_of_year); // 29 60
/// @endcode
/// @see doomsday_month_anchors()
struct DoomsdayAnchor {
  int month=0; ///< Month in [1,12], or zero in an uncomputed placeholder.
  int day=0; ///< Canonical anchor day in this month.
  int day_of_year=0; ///< One-based ordinal day, accounting for the year's leap status.
  int weekday=0; ///< Computed Sunday-first weekday in [0,6].
};

/// @brief Test whether an astronomical year is a proleptic Gregorian leap year.
/// Year zero and negative years use the same divisibility rule. No historical
/// adoption cutover, allocation, mutation or timezone; O(1).
/// @param year Any int year.
/// @return True if divisible by400, or by4 but not100.
/// @par Example
/// @code
/// std::print("{} {}\n",dm::algorithms::doomsday_is_leap(1900),dm::algorithms::doomsday_is_leap(2000)); // false true
/// @endcode
/// @par References
/// - US Naval Observatory, Gregorian leap-year rule, https://aa.usno.navy.mil/faq/leap_years
/// @see doomsday_of_year()
bool doomsday_is_leap(int year);

/// @brief Compute a year's Doomsday weekday with actual integer terms.
/// Supports every int year with floor-based century decomposition. Applies the
/// Gregorian rules proleptically, with astronomical year zero and no historical
/// Julian/Gregorian cutover. No allocation, replay or shared mutation; O(1).
/// @param year Any int year.
/// @return Owned century/within-century terms, leap status and normalized doomsday.
/// @par Example
/// @code
/// const auto y=dm::algorithms::doomsday_year_trace(-1);
/// std::print("{} {} {}\n",y.century,y.year_in_century,y.doomsday); // -1 99 0
/// @endcode
/// @par References
/// - S. W. Graham (1995), The Doomsday Rule (notes on Conway's method),
///   https://people.se.cmich.edu/graha1sw/pub/doomsday/Doomsday.pdf
/// @see doomsday_of_year(), doomsday_date_trace()
DoomsdayYear doomsday_year_trace(int year);

/// @brief Weekday of the last day of February, using the shared Doomsday year kernel.
/// Supports every astronomical int year, with zero denoting1 BCE. All dates use
/// the proleptic Gregorian calendar, without historical adoption cutovers. O(1)
/// time/storage, no allocation or shared mutable state.
/// @param year Any int year.
/// @return Sunday=0, Monday=1, ..., Saturday=6.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::doomsday_of_year(2025)); // 5
/// @endcode
/// @par References
/// - S. W. Graham (1995), The Doomsday Rule (notes on Conway's method),
///   https://people.se.cmich.edu/graha1sw/pub/doomsday/Doomsday.pdf
/// @see doomsday_year_trace(), doomsday_weekday()
int doomsday_of_year(int year);

/// @brief Evaluate a validated date through actual Doomsday year and month terms.
/// Reuses gregorian_month_days() for complete date validation before reference
/// lookup or offset arithmetic. Negative offsets are normalized mathematically.
/// Supports every int year, with the same proleptic/astronomical conventions as
/// doomsday_year_trace(). O(1), no allocation or shared mutation.
/// @param year Any int year.
/// @param month Month in [1,12].
/// @param day Valid one-based day of this month.
/// @return Owned actual terms, canonical month reference, signed offset and weekday.
/// @throws std::invalid_argument If the month/day is invalid, including a nonexistent leap day.
/// @par Example
/// @code
/// const auto t=dm::algorithms::doomsday_date_trace(2025,1,1);
/// std::print("{} {}\n",t.offset,t.weekday); // -2 3
/// @endcode
/// @par References
/// - S. W. Graham (1995), The Doomsday Rule (notes on Conway's method),
///   https://people.se.cmich.edu/graha1sw/pub/doomsday/Doomsday.pdf
/// @see doomsday_weekday(), doomsday_month_anchors()
DoomsdayTrace doomsday_date_trace(int year,int month,int day);

/// @brief Weekday of a valid proleptic Gregorian date by Conway's Doomsday rule.
/// The original three-argument function now validates complete dates and uses
/// floor-based negative-century terms through doomsday_date_trace(). No timezone,
/// locale or historical cutover is inferred. O(1), no shared mutation/allocation.
/// @param year Any astronomical int year, including zero and negative values.
/// @param month Month in [1,12].
/// @param day Valid one-based day of month.
/// @return Normalized code Sunday=0, Monday=1, ..., Saturday=6.
/// @throws std::invalid_argument If the date is invalid.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::doomsday_weekday(1969,7,20)); // 0
/// @endcode
/// @par References
/// - S. W. Graham (1995), The Doomsday Rule (notes on Conway's method),
///   https://people.se.cmich.edu/graha1sw/pub/doomsday/Doomsday.pdf
/// @see doomsday_date_trace(), doomsday_day_name()
int doomsday_weekday(int year,int month,int day);

/// @brief Name a Sunday-first weekday code, normalizing arbitrary signed integers.
/// No locale/timezone is consulted; the returned English name owns its string.
/// @param w Any int; Sunday=0 through Saturday=6, reduced modulo seven.
/// @return Owned English weekday name, including for negative/out-of-range codes.
/// @throws std::bad_alloc If the returned string cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::doomsday_day_name(-1)); // Saturday
/// @endcode
/// @see doomsday_weekday()
std::string doomsday_day_name(int w);

/// @brief List all twelve canonical monthly Doomsday anchors for an astronomical year.
/// Retains the original March14 convention; January3/4 and February28/29 adjust
/// for leap years. Weekdays come from the shared date kernel, ordinal days from
/// existing Gregorian month lengths. O(12) fixed work/storage, no allocation.
/// @param year Any int year, using proleptic Gregorian rules.
/// @return Owned fixed array in January-through-December order.
/// @par Example
/// @code
/// const auto a=dm::algorithms::doomsday_month_anchors(2025);
/// std::print("{} {} {}\n",a[11].day,a[11].day_of_year,a[11].weekday); // 12 346 5
/// @endcode
/// @par References
/// - S. W. Graham (1995), The Doomsday Rule (notes on Conway's method),
///   https://people.se.cmich.edu/graha1sw/pub/doomsday/Doomsday.pdf
/// @see doomsday_year_trace(), DoomsdayAnchor
std::array<DoomsdayAnchor,12> doomsday_month_anchors(int year);
} // namespace datamunge::algorithms
