/// @file doomsday.cpp
/// @brief Exact floor-based Doomsday arithmetic and Gregorian-validated month references.
#include <datamunge/algorithms/doomsday.hpp>
#include <datamunge/algorithms/zeller.hpp>
#include <cstdint>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Pure signed remainder and fixed anchor lookup shared by public operations.
namespace {
/// @brief Normalize a signed integer modulo a positive small divisor.
/// @param value Signed value, widened before calendar arithmetic.
/// @param modulus Positive divisor.
/// @return Remainder in [0,modulus-1].
int residue(std::int64_t value,int modulus) {
  return static_cast<int>((value%modulus+modulus)%modulus);
}
/// @brief Divide a signed year by a positive divisor, rounding toward negative infinity.
/// @param value Signed astronomical year.
/// @param divisor Positive divisor.
/// @return Mathematical floor, without subtracting from the original int year.
std::int64_t floor_div(std::int64_t value,int divisor) {
  return value/divisor-(value%divisor<0?1:0);
}
/// @brief Obtain the canonical anchor day for an already validated month.
/// @param month Valid month in [1,12].
/// @param leap Gregorian leap status.
/// @return Valid day of this month, with leap-adjusted January/February.
int reference(int month,bool leap) {
  /// @brief Common-year anchors; March14 is congruent to March7 and March0.
  static constexpr std::array<int,12> days{3,28,14,4,9,6,11,8,5,10,7,12};
  return days[month-1]+(leap && month<=2?1:0);
}
} // namespace
bool doomsday_is_leap(int year) {
  return (year%4==0 && year%100!=0)||year%400==0;
}
DoomsdayYear doomsday_year_trace(int year) {
  DoomsdayYear t;t.year=year;t.leap=doomsday_is_leap(year);
  const auto century=floor_div(year,100);
  t.century=static_cast<int>(century);t.century_mod4=residue(century,4);
  t.century_anchor=(5*t.century_mod4+2)%7;
  t.year_in_century=static_cast<int>(static_cast<std::int64_t>(year)-100*century);
  t.leap_quarters=t.year_in_century/4;
  t.dozens=t.year_in_century/12;t.dozen_remainder=t.year_in_century%12;
  t.dozen_quarters=t.dozen_remainder/4;
  t.doomsday=(t.century_anchor+t.year_in_century+t.leap_quarters)%7;
  return t;
}
int doomsday_of_year(int year) {return doomsday_year_trace(year).doomsday;}
DoomsdayTrace doomsday_date_trace(int year,int month,int day) {
  const int last=gregorian_month_days(year,month);
  if(day<1 || day>last)throw std::invalid_argument("doomsday_weekday: invalid day of month");
  DoomsdayTrace t;t.year=doomsday_year_trace(year);t.month=month;t.day=day;
  t.reference_day=reference(month,t.year.leap);t.offset=day-t.reference_day;
  t.weekday=residue(t.year.doomsday+t.offset,7);
  return t;
}
int doomsday_weekday(int year,int month,int day) {return doomsday_date_trace(year,month,day).weekday;}
std::string doomsday_day_name(int w) {
  /// @brief English weekday names in the original Sunday-first order.
  static constexpr std::array<const char*,7> names{"Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"};
  return names[residue(w,7)];
}
std::array<DoomsdayAnchor,12> doomsday_month_anchors(int year) {
  std::array<DoomsdayAnchor,12> result{};
  const auto y=doomsday_year_trace(year);int preceding=0;
  for(int month=1;month<=12;++month) {
    const int day=reference(month,y.leap);
    result[month-1]={month,day,preceding+day,doomsday_weekday(year,month,day)};
    preceding+=gregorian_month_days(year,month);
  }
  return result;
}
} // namespace datamunge::algorithms
