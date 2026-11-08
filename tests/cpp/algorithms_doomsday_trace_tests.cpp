#include <gtest/gtest.h>
#include <datamunge/algorithms/doomsday.hpp>
#include <datamunge/algorithms/zeller.hpp>
#include <chrono>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
namespace {
bool leap(int y){return y%400==0||(y%4==0&&y%100!=0);}
int length(int y,int m){const int d[]{31,28,31,30,31,30,31,31,30,31,30,31};return d[m-1]+(m==2&&leap(y));}
void check_terms(const DoomsdayTrace& t){auto&y=t.year;EXPECT_EQ(static_cast<std::int64_t>(y.century)*100+y.year_in_century,y.year);EXPECT_GE(y.year_in_century,0);EXPECT_LT(y.year_in_century,100);EXPECT_EQ(y.century_mod4,(y.century%4+4)%4);EXPECT_EQ(y.leap,leap(y.year));EXPECT_EQ(y.year_in_century,12*y.dozens+y.dozen_remainder);EXPECT_EQ((y.century_anchor+y.dozens+y.dozen_remainder+y.dozen_quarters)%7,y.doomsday);EXPECT_EQ(t.offset,t.day-t.reference_day);EXPECT_EQ((y.doomsday+t.offset%7+7)%7,t.weekday);EXPECT_EQ(t.weekday,doomsday_weekday(y.year,t.month,t.day));EXPECT_EQ(y.doomsday,doomsday_of_year(y.year));}
}
TEST(DoomsdayTrace, OriginalDatesYearAndAllMonthFirsts){EXPECT_EQ(doomsday_weekday(1969,7,20),0);EXPECT_EQ(doomsday_weekday(1776,7,4),4);EXPECT_EQ(doomsday_weekday(2025,12,25),4);auto t=doomsday_date_trace(2025,12,25);check_terms(t);EXPECT_EQ(t.year.century_anchor,2);EXPECT_EQ(t.year.leap_quarters,6);EXPECT_EQ(t.year.doomsday,5);EXPECT_EQ(t.reference_day,12);EXPECT_EQ(t.offset,13);for(int m=1;m<=12;++m)EXPECT_EQ(doomsday_weekday(2025,m,1),(zeller_gregorian(2025,m,1)+6)%7);}
TEST(DoomsdayTrace, FiveCompleteCyclesMatchIndependentSequentialDayOracle){for(int first:{-2000,-400,0,1600,2000}){int expected=6,days=0,leaps=0;for(int y=first;y<first+400;++y){leaps+=leap(y);for(int m=1;m<=12;++m)for(int d=1;d<=length(y,m);++d){auto t=doomsday_date_trace(y,m,d);ASSERT_EQ(t.weekday,expected)<<y<<'-'<<m<<'-'<<d;ASSERT_EQ(t.weekday,doomsday_weekday(y,m,d));ASSERT_EQ(t.weekday,(zeller_gregorian(y,m,d)+6)%7);expected=(expected+1)%7;++days;}}EXPECT_EQ(expected,6);EXPECT_EQ(days,146097);EXPECT_EQ(leaps,97);}}
TEST(DoomsdayTrace, ChronoAndExtremeYearsMatchFourHundredYearReduction){using namespace std::chrono;for(int y:{-32767,-1900,-100,-1,0,1,1582,1700,1900,2000,2024,2025,2100,32767})for(int m=1;m<=12;++m)for(int d=1;d<=length(y,m);++d){year_month_day date{year{y},month{static_cast<unsigned>(m)},day{static_cast<unsigned>(d)}};ASSERT_TRUE(date.ok());EXPECT_EQ(doomsday_weekday(y,m,d),weekday{sys_days{date}}.c_encoding());}
std::mt19937 rng(82);for(int i=0;i<10000;++i){const int y=i==0?std::numeric_limits<int>::min():i==1?std::numeric_limits<int>::max():static_cast<int>(rng());const int reduced=(y%400+400)%400;for(int m:{1,2,3,12})for(int d:{1,length(y,m)}){auto t=doomsday_date_trace(y,m,d);check_terms(t);year_month_day date{year{2000+reduced},month{static_cast<unsigned>(m)},day{static_cast<unsigned>(d)}};EXPECT_EQ(t.weekday,weekday{sys_days{date}}.c_encoding());}}}
TEST(DoomsdayTrace, EveryMonthlyAnchorAndLeapBoundaryIsExact){for(int y=-400;y<400;++y){auto a=doomsday_month_anchors(y);int prefix=0;for(int m=1;m<=12;++m){EXPECT_EQ(a[m-1].month,m);EXPECT_EQ(a[m-1].day_of_year,prefix+a[m-1].day);EXPECT_EQ((a[m-1].day_of_year-a[1].day_of_year)%7,0);EXPECT_EQ(a[m-1].weekday,doomsday_of_year(y));EXPECT_EQ(a[m-1].weekday,doomsday_weekday(y,m,a[m-1].day));prefix+=length(y,m);}EXPECT_EQ(a[0].day,leap(y)?4:3);EXPECT_EQ(a[1].day,leap(y)?29:28);EXPECT_EQ(a[2].day,14);EXPECT_EQ(prefix,leap(y)?366:365);EXPECT_EQ((doomsday_of_year(y+1)-doomsday_of_year(y)+7)%7,1+leap(y+1));}EXPECT_FALSE(doomsday_is_leap(1900));EXPECT_TRUE(doomsday_is_leap(2000));EXPECT_FALSE(doomsday_is_leap(2100));}
TEST(DoomsdayTrace, CompleteValidationPrecedesReferenceAndOffsetArithmetic){for(int y:{std::numeric_limits<int>::min(),-100,0,1900,2000,std::numeric_limits<int>::max()}){for(int m:{std::numeric_limits<int>::min(),-1,0,13,std::numeric_limits<int>::max()}){EXPECT_THROW(doomsday_weekday(y,m,1),std::invalid_argument);
EXPECT_THROW(doomsday_date_trace(y,m,1),std::invalid_argument);}
for(int m=1;m<=12;++m)for(int d:{std::numeric_limits<int>::min(),-1,0,length(y,m)+1,std::numeric_limits<int>::max()}){EXPECT_THROW(doomsday_weekday(y,m,d),std::invalid_argument);
EXPECT_THROW(doomsday_date_trace(y,m,d),std::invalid_argument);}}
EXPECT_THROW(doomsday_weekday(1900,2,29),std::invalid_argument);
EXPECT_NO_THROW(doomsday_weekday(2000,2,29));}
TEST(DoomsdayTrace, NamesNormalizeAllSignedCodesAndRecordsOwnTheirValues){for(int w:{std::numeric_limits<int>::min(),-100,-7,-1,0,6,7,100,std::numeric_limits<int>::max()}){const int reduced=(w%7+7)%7;EXPECT_EQ(doomsday_day_name(w),doomsday_day_name(reduced));}EXPECT_EQ(doomsday_day_name(-1),"Saturday");auto t=doomsday_date_trace(2025,12,25);auto copy=t;t.year.doomsday=0;t.weekday=0;EXPECT_EQ(copy.year.doomsday,5);EXPECT_EQ(copy.weekday,4);}
