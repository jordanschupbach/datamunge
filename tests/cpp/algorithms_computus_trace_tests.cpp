#include <gtest/gtest.h>
#include <datamunge/algorithms/computus.hpp>
#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using namespace datamunge::algorithms;
static_assert(std::is_trivial_v<EasterDate> && std::is_standard_layout_v<EasterDate>);
// Independent Oudin (1940) rule published by the US Naval Observatory:
// https://aa.usno.navy.mil/faq/easter . Test calls use years >=1700, so
// the c-17 division is nonnegative, exactly as in the published arithmetic.
EasterDate oudin(std::int64_t year) {
  const auto c=year/100,n=year-19*(year/19),k=(c-17)/25;
  auto i=(c-c/4-(c-k)/3+19*n+15)%30;
  i-= (i/28)*(1-(i/28)*(29/(i+1))*((21-n)/11));
  const auto j=(year+year/4+i+2-c+c/4)%7,l=i-j;
  const auto month=3+(l+40)/44,day=l+28-31*(month/4);
  return {static_cast<int>(month),static_cast<int>(day)};
}
int ordinal(EasterDate e){return e.month==3?e.day:31+e.day;}
bool sunday(int year,int march_day) {
  const std::int64_t before=static_cast<std::int64_t>(year)-1;
  const int leap=year%4==0 && (year%100!=0 || year%400==0);
  return (before+before/4-before/100+before/400+31+28+leap+march_day)%7==0;
}
void check_summary(int first,int last) {
  const auto s=computus_gregorian_summary(first,last,1000);
  std::array<std::uint64_t,35> counts{};
  std::vector<std::pair<int,int>> days;
  for(int y=first;;++y) {
    const auto e=oudin(y);const int day=ordinal(e);++counts[day-22];days.emplace_back(day,y);
    if(y==last)break;
  }
  const auto minimum=*std::min_element(days.begin(),days.end());
  const auto maximum=*std::max_element(days.begin(),days.end(),[](auto a,auto b){return a.first<b.first || (a.first==b.first && a.second>b.second);});
  ASSERT_EQ(s.first_year,first);ASSERT_EQ(s.last_year,last);ASSERT_EQ(s.year_count,days.size());
  ASSERT_EQ(s.date_counts,counts);ASSERT_EQ(s.earliest_year,minimum.second);ASSERT_EQ(s.latest_year,maximum.second);
  ASSERT_EQ(ordinal(s.earliest),minimum.first);ASSERT_EQ(ordinal(s.latest),maximum.first);
  ASSERT_EQ(s.spread_days,maximum.first-minimum.first);
}
}

TEST(ComputusTrace, OriginalAndExceptionalDatesMatchPrimaryPublishedTables) {
  // Census Bureau 1600-2099 table, plus the USNO's discussed 1954 exception.
  const std::array<std::array<int,3>,11> cases{{{1818,3,22},{1943,4,25},{1954,4,18},
    {1981,4,19},{2000,4,23},{2008,3,23},{2010,4,4},{2024,3,31},{2025,4,20},{2026,4,5},{2038,4,25}}};
  for(const auto& row:cases) {
    const auto e=computus_gregorian(row[0]);EXPECT_EQ(e.month,row[1]);EXPECT_EQ(e.day,row[2]);
  }
}

TEST(ComputusTrace, ActualStepsRetainBothExceptionalWeekCorrections) {
  const auto a=computus_gregorian_trace(1954),b=computus_gregorian_trace(1981);
  EXPECT_EQ(a.a,16);EXPECT_EQ(a.h,28);EXPECT_EQ(a.l,6);EXPECT_EQ(a.m,1);
  EXPECT_EQ(a.uncorrected_march_day,56);EXPECT_EQ(a.march_day,49);
  EXPECT_EQ(b.a,5);EXPECT_EQ(b.h,29);EXPECT_EQ(b.l,6);EXPECT_EQ(b.m,1);
  EXPECT_EQ(b.uncorrected_march_day,57);EXPECT_EQ(b.march_day,50);
  for(int year=1;year<=10000;++year) {
    const auto t=computus_gregorian_trace(year);const auto plain=computus_gregorian(year);
    EXPECT_EQ(t.year,year);EXPECT_EQ(t.date.month,plain.month);EXPECT_EQ(t.date.day,plain.day);
    EXPECT_EQ(t.m,static_cast<int>(t.l==6 && (t.h==29 || (t.h==28 && t.a>=11))));
    EXPECT_EQ(t.encoding,t.h+t.l-7*t.m+114);EXPECT_EQ(t.march_day,ordinal(t.date));
  }
}

TEST(ComputusTrace, EntireRulePeriodMatchesIndependentOudinAndGregorianWeekday) {
  std::array<std::uint64_t,35> counts{};
  for(int year=1700;year<5701700;++year) {
    const auto expected=oudin(year),e=computus_gregorian(year);
    const int day=ordinal(e);
    if(e.month!=expected.month || e.day!=expected.day || day<22 || day>56 || !sunday(year,day)) {
      FAIL()<<"Independent date/weekday mismatch in year "<<year;return;
    }
    ++counts[day-22];
  }
  const auto s=computus_gregorian_summary(1,5700000,5700000);
  EXPECT_EQ(s.date_counts,counts);EXPECT_EQ(s.year_count,5700000);
  EXPECT_EQ(std::accumulate(counts.begin(),counts.end(),std::uint64_t{0}),5700000);
  EXPECT_EQ(s.spread_days,34);
  for(const auto count:counts)EXPECT_GT(count,0);
}

TEST(ComputusTrace, ProlepticEarlyYearsAndExtremePositiveIntsStayBoundedAndSunday) {
  for(int year=1;year<1700;++year) {
    const auto t=computus_gregorian_trace(year);EXPECT_GE(t.march_day,22);EXPECT_LE(t.march_day,56);
    EXPECT_TRUE(sunday(year,t.march_day));
  }
  std::mt19937 engine(624);
  for(int trial=0;trial<10000;++trial) {
    const int year=static_cast<int>(1700+engine()%static_cast<unsigned>(INT_MAX-1699));
    const auto t=computus_gregorian_trace(year);
    const auto e=oudin(year);
    ASSERT_EQ(t.date.month,e.month);ASSERT_EQ(t.date.day,e.day);ASSERT_TRUE(sunday(year,t.march_day));
  }
  const auto e=computus_gregorian(INT_MAX),expected=oudin(INT_MAX);
  EXPECT_EQ(e.month,expected.month);EXPECT_EQ(e.day,expected.day);
}

TEST(ComputusTrace, FiveMillionSevenHundredThousandShiftRetainsAllReducedSteps) {
  constexpr int period=5700000;std::mt19937 engine(14250);
  for(int trial=0;trial<10000;++trial) {
    const int year=1+static_cast<int>(engine()%static_cast<unsigned>(INT_MAX-period));
    const auto a=computus_gregorian_trace(year),b=computus_gregorian_trace(year+period);
    ASSERT_EQ(b.a,a.a);ASSERT_EQ(b.c,a.c);ASSERT_EQ(b.e,a.e);ASSERT_EQ(b.i,a.i);ASSERT_EQ(b.k,a.k);
    ASSERT_EQ(b.b-a.b,57000);ASSERT_EQ(b.d-a.d,14250);ASSERT_EQ(b.f-a.f,2280);ASSERT_EQ(b.g-a.g,18240);
    ASSERT_EQ(b.h,a.h);ASSERT_EQ(b.l,a.l);ASSERT_EQ(b.m,a.m);ASSERT_EQ(b.march_day,a.march_day);
  }
}

TEST(ComputusTrace, OriginalRangeSummaryUsesFixedMarchAxisAndFirstTies) {
  const auto s=computus_gregorian_summary(2000,2040);
  EXPECT_EQ(s.year_count,41);EXPECT_EQ(s.earliest_year,2008);EXPECT_EQ(s.earliest.month,3);EXPECT_EQ(s.earliest.day,23);
  EXPECT_EQ(s.latest_year,2038);EXPECT_EQ(s.latest.month,4);EXPECT_EQ(s.latest.day,25);EXPECT_EQ(s.spread_days,33);
  check_summary(2000,2040);check_summary(2024,2024);
}

TEST(ComputusTrace, SummaryMatchesIndependentExtremaAndAddsAcrossPartitions) {
  const auto full=computus_gregorian_summary(1700,2099),a=computus_gregorian_summary(1700,1899),b=computus_gregorian_summary(1900,2099);
  for(std::size_t i=0;i<35;++i)EXPECT_EQ(full.date_counts[i],a.date_counts[i]+b.date_counts[i]);
  EXPECT_EQ(full.corrected_years,a.corrected_years+b.corrected_years);
  std::mt19937 engine(901);
  for(int trial=0;trial<1000;++trial) {
    const int first=1700+static_cast<int>(engine()%2000),last=first+static_cast<int>(engine()%100);
    check_summary(first,last);if(::testing::Test::HasFatalFailure())return;
  }
}

TEST(ComputusTrace, ValidatesPositiveYearsAndCompleteWorkCapsBeforeScan) {
  for(const int year:{INT_MIN,-1,0}) {
    EXPECT_THROW(computus_gregorian(year),std::invalid_argument);
    EXPECT_THROW(computus_gregorian_trace(year),std::invalid_argument);
    EXPECT_THROW(computus_gregorian_summary(year,INT_MAX,1),std::invalid_argument);
    EXPECT_THROW(computus_gregorian_summary(1,year,1),std::invalid_argument);
  }
  EXPECT_THROW(computus_gregorian_summary(2025,2024),std::invalid_argument);
  EXPECT_THROW(computus_gregorian_summary(2024,2024,0),std::invalid_argument);
  EXPECT_NO_THROW(computus_gregorian_summary(2000,2040,41));
  EXPECT_THROW(computus_gregorian_summary(2000,2040,40),std::length_error);
  EXPECT_THROW(computus_gregorian_summary(1,INT_MAX),std::length_error);
}

TEST(ComputusTrace, IntMaxRangeTerminationAndReturnedCountsAreOwned) {
  check_summary(INT_MAX-2,INT_MAX);
  auto a=computus_gregorian_summary(2024,2024,1);auto b=a;a.date_counts[9]=99;a.earliest.day=1;
  EXPECT_EQ(b.date_counts[9],1);EXPECT_EQ(b.earliest.day,31);EXPECT_EQ(b.earliest_year,2024);EXPECT_EQ(b.latest_year,2024);
}

TEST(ComputusTrace, MetonicIndexDoesNotMakeNineteenYearsAnExactDatePeriod) {
  const auto a=computus_gregorian_trace(2000),b=computus_gregorian_trace(2019);
  EXPECT_EQ(a.a,b.a);EXPECT_EQ(a.date.month,4);EXPECT_EQ(b.date.month,4);
  EXPECT_EQ(a.date.day,23);EXPECT_EQ(b.date.day,21);
}
