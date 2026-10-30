/// @file zeller.cpp
/// @brief Floor-based Gregorian arithmetic and DataFrame calendar diagnostics.
#include <array>
#include <cmath>
#include <cstdint>
#include <datamunge/algorithms/zeller.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {
/// @brief Internal exact integer arithmetic and checked DataFrame conversion.
namespace {
/// @brief Divide by a positive integer, rounding down rather than towards zero.
/// @param a Signed numerator.
/// @param b Positive divisor.
/// @return Mathematical floor of a/b.
std::int64_t floor_div(std::int64_t a, std::int64_t b) {
  return a / b - (a % b < 0 ? 1 : 0);
}
/// @brief Validate a date before evaluating arithmetic or producing a trace.
/// @param year Astronomical int year.
/// @param month Month in [1,12].
/// @param day Valid day of month.
/// @throws std::invalid_argument If the date is invalid.
void validate_date(int year, int month, int day) {
  if (day < 1 || day > gregorian_month_days(year, month))
    throw std::invalid_argument("zeller_gregorian: invalid day of month");
}
/// @brief The six signed summands of the congruence after moving Jan/Feb to the previous year.
/// @param year Any int year; widened before adjustment.
/// @param month Valid original month.
/// @param day Valid original day.
/// @return Exact signed summands, before reduction modulo seven.
std::array<std::int64_t, 6> congruence_terms(int year, int month, int day) {
  std::int64_t y = year;
  if (month < 3) {
    month += 12;
    --y;
  }
  return {day, 13 * (month + 1) / 5, y, floor_div(y, 4), -floor_div(y, 100), floor_div(y, 400)};
}
/// @brief Check and convert a numeric table cell to an int without truncation.
/// @param value Numeric year or month cell.
/// @return Exact integer value.
/// @throws std::invalid_argument For non-finite, fractional or out-of-range values.
int calendar_integer(double value) {
  if (!std::isfinite(value) || value != std::floor(value) ||
      value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    throw std::invalid_argument(
        "gregorian_month_features: expected integer year/month in int range");
  return static_cast<int>(value);
}
} // namespace
int gregorian_month_days(int year, int month) {
  if (month < 1 || month > 12)
    throw std::invalid_argument("gregorian_month_days: month outside [1,12]");
  /// @brief Ordinary Gregorian month lengths, indexed by month minus one.
  static constexpr int lengths[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  const bool           leap      = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
  return lengths[month - 1] + (month == 2 && leap ? 1 : 0);
}
int zeller_gregorian(int year, int month, int day) {
  validate_date(year, month, day);
  const auto terms = congruence_terms(year, month, day);
  const auto sum   = std::accumulate(terms.begin(), terms.end(), std::int64_t{0});
  return static_cast<int>((sum % 7 + 7) % 7);
}
std::string zeller_day_name(int h) {
  /// @brief Weekday names in Zeller's Saturday-first convention.
  static const char* names[] = {
      "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"};
  return names[(h % 7 + 7) % 7];
}
dstruct::DataFrame zeller_terms(int year, int month, int day) {
  validate_date(year, month, day);
  const auto          terms = congruence_terms(year, month, day);
  const auto          sum   = std::accumulate(terms.begin(), terms.end(), std::int64_t{0});
  std::vector<double> values(terms.begin(), terms.end());
  values.push_back(static_cast<double>(sum));
  values.push_back(static_cast<double>((sum % 7 + 7) % 7));
  dstruct::DataFrame out;
  out.add_column(
      "term",
      std::vector<std::string>{
          "day",
          "month term",
          "shifted year",
          "leap quarters",
          "century correction",
          "four centuries",
          "sum",
          "weekday code"});
  out.add_column("value", values);
  return out;
}
dstruct::DataFrame gregorian_month_features(
    const dstruct::DataFrame& frame,
    const std::string&        year_column,
    const std::string&        month_column) {
  for (const char* name :
       {"calendar_days", "weekdays", "weekend_days", "start_weekday_code", "start_weekday"})
    if (frame.has_column(name))
      throw std::invalid_argument("gregorian_month_features: output column already exists");
  const auto&              years  = frame.get_column(year_column);
  const auto&              months = frame.get_column(month_column);
  std::vector<double>      lengths, weekdays, weekends, starts;
  std::vector<std::string> names;
  for (std::size_t r = 0; r < frame.nrows(); ++r) {
    const int year = calendar_integer(years.at(r)), month = calendar_integer(months.at(r));
    const int days = gregorian_month_days(year, month), first = zeller_gregorian(year, month, 1);
    int       weekend = 2 * (days / 7);
    for (int i = 0; i < days % 7; ++i)
      weekend += (first + i) % 7 <= 1;
    lengths.push_back(days);
    weekdays.push_back(days - weekend);
    weekends.push_back(weekend);
    starts.push_back(first);
    names.push_back(zeller_day_name(first));
  }
  auto out = frame;
  out.add_column("calendar_days", lengths);
  out.add_column("weekdays", weekdays);
  out.add_column("weekend_days", weekends);
  out.add_column("start_weekday_code", starts);
  out.add_column("start_weekday", names);
  return out;
}
dstruct::DataFrame zeller_cycle_counts(int day, int start_year) {
  if (day < 1 || day > 31 || start_year > std::numeric_limits<int>::max() - 399)
    throw std::invalid_argument("zeller_cycle_counts: invalid day or overflowing year range");
  std::array<int, 7> counts{};
  for (int offset = 0; offset < 400; ++offset)
    for (int month = 1; month <= 12; ++month)
      if (day <= gregorian_month_days(start_year + offset, month))
        ++counts[zeller_gregorian(start_year + offset, month, day)];
  const double             total = std::accumulate(counts.begin(), counts.end(), 0.0);
  std::vector<double>      codes, n, p;
  std::vector<std::string> names;
  for (int h = 0; h < 7; ++h) {
    codes.push_back(h);
    names.push_back(zeller_day_name(h));
    n.push_back(counts[h]);
    p.push_back(counts[h] / total);
  }
  dstruct::DataFrame out;
  out.add_column("weekday_code", codes);
  out.add_column("weekday", names);
  out.add_column("count", n);
  out.add_column("proportion", p);
  return out;
}
} // namespace datamunge::algorithms
