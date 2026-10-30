#pragma once

/// @file zeller.hpp
/// @brief Checked proleptic Gregorian weekdays, monthly features and 400-year cycle diagnostics.
#include <datamunge/dstruct/dataframe.hpp>
#include <string>

namespace datamunge::algorithms {

/// @brief Number of days in a proleptic Gregorian calendar month.
///
/// Uses astronomical year numbering (year 0 is 1 BCE); the Gregorian rule is
/// applied even before its historical adoption. O(1), with no mutable state.
/// @param year Any int year, including zero and negative years.
/// @param month Month in [1,12].
/// @return 28, 29, 30 or 31 days, including Gregorian century exceptions.
/// @throws std::invalid_argument If month is outside [1,12].
/// @par Example
/// @code
/// const int days = dm::algorithms::gregorian_month_days(1900, 2); // 28
/// @endcode
/// @par References
/// - Ch. Zeller (1883), Problema duplex calendarii fundamentale, BSMF 11, 59-61,
///   https://doi.org/10.24033/bsmf.254
/// @see zeller_gregorian(), gregorian_month_features()
int gregorian_month_days(int year, int month);

/// @brief Weekday of a valid proleptic Gregorian date by Zeller's congruence.
///
/// Astronomical year numbering includes year zero. Floor division and 64-bit
/// intermediates make January/February and negative years well-defined for
/// every int year. There is no historical Julian/Gregorian cutover. O(1) time
/// and storage, with no mutable state; concurrent calls are safe.
/// @param year Any int year.
/// @param month Month in [1,12].
/// @param day Valid day of that month, starting at one.
/// @return Code in [0,6]: Saturday=0, Sunday=1, Monday=2, ..., Friday=6.
/// @throws std::invalid_argument If the date is invalid, including a nonexistent leap day.
/// @par Example
/// @code
/// const int h = dm::algorithms::zeller_gregorian(2000, 1, 1); // 0 (Saturday)
/// @endcode
/// @par References
/// - Ch. Zeller (1883), Problema duplex calendarii fundamentale, BSMF 11, 59-61,
///   https://doi.org/10.24033/bsmf.254
/// @see zeller_day_name(), zeller_terms(), gregorian_month_days()
int zeller_gregorian(int year, int month, int day);

/// @brief Name a Zeller weekday code, normalising arbitrary integer codes modulo seven.
/// @param h Any int; 0=Saturday, 1=Sunday, ..., 6=Friday.
/// @return Owned English weekday name, including for negative or out-of-range codes.
/// @par Example
/// @code
/// const auto day = dm::algorithms::zeller_day_name(-1); // "Friday"
/// @endcode
/// @see zeller_gregorian()
std::string zeller_day_name(int h);

/// @brief Inspect the signed terms of the full-year Zeller congruence for one date.
///
/// Jan/Feb become months 13/14 of the preceding year. Rows give day,
/// floor(13*(month+1)/5), shifted year Y, floor(Y/4), -floor(Y/100),
/// floor(Y/400), their sum and its non-negative remainder modulo seven.
/// @param year Any int year.
/// @param month Month in [1,12].
/// @param day Valid day of the supplied month.
/// @return Owned columns term and value; eight rows, exact for all int years.
/// @throws std::invalid_argument If the date is invalid.
/// @par Example
/// @code
/// const auto terms = dm::algorithms::zeller_terms(1949, 1, 1);
/// const double weekday = terms.double_at("value", 7); // 0 (Saturday)
/// @endcode
/// @par References
/// - Ch. Zeller (1883), https://doi.org/10.24033/bsmf.254
/// @see zeller_gregorian()
dstruct::DataFrame zeller_terms(int year, int month, int day);

/// @brief Add calendar lengths, weekend counts and month-start weekdays to a monthly dataset.
///
/// Preserves every input column and row, appending calendar_days, weekdays,
/// weekend_days, start_weekday_code and start_weekday. A weekend means Saturday
/// or Sunday; holidays and local working-week conventions are not modelled.
/// Numeric year/month values must be finite integers in the supported ranges.
/// O(rows) time and output storage in addition to the copied input; no mutation.
/// @param frame Borrowed monthly records; empty input is allowed with valid schema.
/// @param year_column Numeric year column, default "year"; values must fit int.
/// @param month_column Numeric month column, default "month"; values in [1,12].
/// @return An owned augmented frame. Repeated or unordered months remain as supplied.
/// @throws std::invalid_argument If types/nulls/values are invalid or an output column exists.
/// @throws std::out_of_range If either named input column is absent.
/// @par Example
/// @code
/// const auto air = dm::algorithms::gregorian_month_features(dm::datasets::air_passengers());
/// const double days = air.double_at("calendar_days", 0); // 31
/// @endcode
/// @see gregorian_month_days(), zeller_gregorian()
dstruct::DataFrame gregorian_month_features(
    const dstruct::DataFrame& frame,
    const std::string&        year_column  = "year",
    const std::string&        month_column = "month");

/// @brief Count the weekdays of a month-day over a complete 400-year Gregorian cycle.
///
/// Visits 4800 months, omitting those without the requested day (e.g. February
/// for day 30). Every complete cycle has the same counts regardless of start.
/// O(4800) time and O(1) counting storage, with owned tabular output.
/// @param day Day of month in [1,31], default 13.
/// @param start_year First year, default 2000; must leave room for 399 more int years.
/// @return Seven rows in Zeller order: weekday_code, weekday, count, proportion.
/// @throws std::invalid_argument If day is invalid or start_year+399 exceeds INT_MAX.
/// @par Example
/// @code
/// const auto cycle = dm::algorithms::zeller_cycle_counts();
/// const double friday_count = cycle.double_at("count", 6); // 688
/// @endcode
/// @par References
/// - Ch. Zeller (1883), https://doi.org/10.24033/bsmf.254 (Gregorian calendar arithmetic).
/// @see zeller_gregorian(), gregorian_month_features()
dstruct::DataFrame zeller_cycle_counts(int day = 13, int start_year = 2000);
} // namespace datamunge::algorithms
