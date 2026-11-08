/// @file computus.cpp
/// @brief One validated Gregorian Easter evaluation and bounded constant-storage range accumulation.
#include <datamunge/algorithms/computus.hpp>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private positive-year validation and single executing arithmetic rule.
namespace computus_detail {
/// @brief Reject unsupported year-zero/BCE numbering before arithmetic.
/// @param year Candidate rule year.
/// @throws std::invalid_argument If year is not positive.
void validate(int year) {
  if(year<1)throw std::invalid_argument("Gregorian computus requires a positive year");
}

/// @brief Evaluate the actual standard integer expressions for an already validated year.
/// @param year Positive int year; century values are at most INT_MAX/100.
/// @return Owned actual symbols and encoded/decoded Easter date.
ComputusGregorianTrace evaluate(int year) {
  ComputusGregorianTrace t;t.year=year;
  t.a=year%19;t.b=year/100;t.c=year%100;
  t.d=t.b/4;t.e=t.b%4;t.f=(t.b+8)/25;t.g=(t.b-t.f+1)/3;
  t.h=(19*t.a+t.b-t.d-t.g+15)%30;
  t.i=t.c/4;t.k=t.c%4;
  t.l=(32+2*t.e+2*t.i-t.h-t.k)%7;
  t.m=(t.a+11*t.h+22*t.l)/451;
  t.encoding=t.h+t.l-7*t.m+114;
  t.date={t.encoding/31,t.encoding%31+1};
  t.uncorrected_march_day=22+t.h+t.l;
  t.march_day=t.uncorrected_march_day-7*t.m;
  return t;
}
} // namespace computus_detail

EasterDate computus_gregorian(int year) {
  computus_detail::validate(year);return computus_detail::evaluate(year).date;
}

ComputusGregorianTrace computus_gregorian_trace(int year) {
  computus_detail::validate(year);return computus_detail::evaluate(year);
}

EasterRangeSummary computus_gregorian_summary(int first_year,int last_year,std::size_t max_years) {
  computus_detail::validate(first_year);computus_detail::validate(last_year);
  if(last_year<first_year || !max_years)throw std::invalid_argument("Gregorian computus requires ordered years and a positive cap");
  const auto count=static_cast<std::uint64_t>(last_year)-static_cast<std::uint64_t>(first_year)+1;
  if(count>max_years)throw std::length_error("Gregorian computus complete year limit");
  EasterRangeSummary result;result.first_year=first_year;result.last_year=last_year;result.year_count=count;
  int earliest=57,latest=21;
  for(int year=first_year;;++year) {
    const auto t=computus_detail::evaluate(year);
    ++result.date_counts[static_cast<std::size_t>(t.march_day-22)];
    if(t.m)++result.corrected_years;
    if(t.march_day<earliest){earliest=t.march_day;result.earliest_year=year;result.earliest=t.date;}
    if(t.march_day>latest){latest=t.march_day;result.latest_year=year;result.latest=t.date;}
    if(year==last_year)break;
  }
  result.spread_days=latest-earliest;return result;
}
} // namespace datamunge::algorithms
