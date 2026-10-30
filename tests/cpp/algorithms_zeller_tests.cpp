#include <datamunge/algorithms/zeller.hpp>
#include <gtest/gtest.h>
#include <limits>

namespace alg = datamunge::algorithms;
TEST(Zeller, ValidatesMonthLengthsAndLeapDays) {
    EXPECT_EQ(alg::gregorian_month_days(1900,2),28);
    EXPECT_EQ(alg::gregorian_month_days(2000,2),29);
    EXPECT_EQ(alg::gregorian_month_days(0,2),29);
    EXPECT_EQ(alg::gregorian_month_days(-100,2),28);
    EXPECT_EQ(alg::gregorian_month_days(-400,2),29);
    EXPECT_THROW(alg::zeller_gregorian(1900,2,29),std::invalid_argument);
    EXPECT_THROW(alg::zeller_gregorian(2000,4,31),std::invalid_argument);
    EXPECT_THROW(alg::zeller_gregorian(2000,0,1),std::invalid_argument);
    EXPECT_THROW(alg::zeller_gregorian(2000,13,1),std::invalid_argument);
    EXPECT_THROW(alg::zeller_gregorian(2000,1,0),std::invalid_argument);
}
TEST(Zeller, FullCycleAdvancesEveryDayAndRepeats) {
    // Independent sequential-day oracle anchored on known Saturday 2000-01-01.
    int weekday = 0, days = 0;
    for (int year=2000;year<2400;++year) for (int month=1;month<=12;++month)
        for (int day=1;day<=alg::gregorian_month_days(year,month);++day) {
            ASSERT_EQ(alg::zeller_gregorian(year,month,day),weekday);
            ASSERT_EQ(alg::zeller_gregorian(year-400,month,day),weekday);
            weekday=(weekday+1)%7; ++days;
        }
    EXPECT_EQ(days,146097); EXPECT_EQ(weekday,0);
}
TEST(Zeller, FloorDivisionSupportsNegativeAndExtremeYears) {
    EXPECT_EQ(alg::zeller_gregorian(0,1,1),0);
    EXPECT_EQ(alg::zeller_gregorian(-1,1,1),alg::zeller_gregorian(399,1,1));
    for (int year : {std::numeric_limits<int>::min(),std::numeric_limits<int>::max(),-100,-400}) {
        int reduced=(year%400+400)%400;
        for (int month : {1,2,3,12})
            EXPECT_EQ(alg::zeller_gregorian(year,month,1),alg::zeller_gregorian(reduced,month,1));
    }
    EXPECT_EQ(alg::zeller_day_name(-1),"Friday");
    EXPECT_EQ(alg::zeller_day_name(std::numeric_limits<int>::min()),"Thursday");
    EXPECT_EQ(alg::zeller_terms(1949,1,1).double_at("value",7),0);
}
TEST(Zeller, FridayThirteenthCycleAndMissingMonthDays) {
    const auto cycle=alg::zeller_cycle_counts();
    const int expected[]={684,687,685,685,687,684,688};
    const auto other=alg::zeller_cycle_counts(13,-399);
    for (std::size_t i=0;i<7;++i) {
        EXPECT_EQ(cycle.double_at("count",i),expected[i]);
        EXPECT_EQ(other.double_at("count",i),expected[i]);
    }
    const auto last=alg::zeller_cycle_counts(31);
    double total=0;for(std::size_t i=0;i<7;++i)total+=last.double_at("count",i);
    EXPECT_EQ(total,2800);
    EXPECT_THROW(alg::zeller_cycle_counts(0),std::invalid_argument);
    EXPECT_THROW(alg::zeller_cycle_counts(13,std::numeric_limits<int>::max()),std::invalid_argument);
}
TEST(Zeller, MonthlyFeaturesPreserveRowsAndCheckSchema) {
    datamunge::dstruct::DataFrame frame;
    frame.add_column("year",std::vector<double>{1949,1952,2000});
    frame.add_column("month",std::vector<double>{1,2,2});
    const auto out=alg::gregorian_month_features(frame);
    EXPECT_EQ(frame.ncols(),2);EXPECT_EQ(out.ncols(),7);
    EXPECT_EQ(out.string_at("start_weekday",0),"Saturday");
    EXPECT_EQ(out.double_at("weekend_days",0),10);
    EXPECT_EQ(out.double_at("calendar_days",1),29);
    EXPECT_EQ(out.double_at("weekend_days",1),8);
    EXPECT_EQ(out.double_at("weekdays",2),21);
    EXPECT_THROW(alg::gregorian_month_features(out),std::invalid_argument);
    EXPECT_THROW(alg::gregorian_month_features(frame,"absent"),std::out_of_range);
    frame.set_null("year",0);
    EXPECT_THROW(alg::gregorian_month_features(frame),std::invalid_argument);
    datamunge::dstruct::DataFrame fractional;
    fractional.add_column("year",std::vector<double>{2000.5});fractional.add_column("month",std::vector<double>{1});
    EXPECT_THROW(alg::gregorian_month_features(fractional),std::invalid_argument);
}
