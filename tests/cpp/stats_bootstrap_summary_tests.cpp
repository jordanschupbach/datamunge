#include <dm/dm.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <random>

using dm::stats::bootstrap_summary;
namespace {
constexpr double infinity=std::numeric_limits<double>::infinity();
constexpr double missing_value=std::numeric_limits<double>::quiet_NaN();
constexpr double maximum=std::numeric_limits<double>::max();
}

TEST(NativeMedian, SelectionKeepsInputAndHandlesOddEvenAndTies) {
  const auto input=dm::c(9,1,4,2);
  EXPECT_EQ(dm::median(input),3);
  EXPECT_EQ(input[0],9); EXPECT_EQ(input[1],1);
  EXPECT_EQ(dm::median(dm::c(9,1,4)),4);
  EXPECT_EQ(dm::median(dm::c(3,3,3,1,7,3)),3);
  EXPECT_EQ(dm::median(dm::c(7)),7);
}
TEST(NativeMedian, NonfiniteAndExtremeMidpointsAreDefined) {
  EXPECT_TRUE(std::isnan(dm::median(dm::Vector<>{})));
  EXPECT_TRUE(std::isnan(dm::median(dm::c(1,missing_value,2))));
  EXPECT_EQ(dm::median(dm::c(-maximum,maximum)),0);
  EXPECT_EQ(dm::median(dm::c(maximum,maximum)),maximum);
  EXPECT_EQ(dm::median(dm::c(-infinity,1,infinity)),1);
  EXPECT_EQ(dm::median(dm::c(infinity,infinity)),infinity);
  EXPECT_TRUE(std::isnan(dm::median(dm::c(-infinity,infinity))));
}
TEST(BootstrapSummary, ExactTypeSevenQuantilesAndAllIntervals) {
  const auto r=bootstrap_summary(2.5,{4,1,3,2},{},.5);
  EXPECT_EQ(r.replicates,(std::vector<double>{4,1,3,2}));
  EXPECT_EQ(r.bias,0); EXPECT_NEAR(r.standard_error,std::sqrt(5.0/3),1e-14);
  EXPECT_EQ(r.percentile_interval,(std::pair{1.75,3.25}));
  EXPECT_EQ(r.basic_interval,r.percentile_interval);
  EXPECT_EQ(r.bca_interval,r.percentile_interval);
  const double half=dm::random::normal_quantile(.75)*r.standard_error;
  EXPECT_NEAR(r.normal_interval.first,2.5-half,1e-14);
  EXPECT_NEAR(r.normal_interval.second,2.5+half,1e-14);
}
TEST(BootstrapSummary, DegenerateAndLargestFiniteConstantAreStable) {
  const auto single=bootstrap_summary(2,{2});
  EXPECT_EQ(single.standard_error,0); EXPECT_EQ(single.bias,0);
  EXPECT_EQ(single.bca_interval,(std::pair{2.0,2.0}));
  const auto r=bootstrap_summary(maximum,{maximum,maximum,maximum},{maximum,maximum});
  EXPECT_EQ(r.standard_error,0); EXPECT_EQ(r.bias,0);
  for (auto interval:{r.basic_interval,r.percentile_interval,r.bca_interval,r.normal_interval}) {
    EXPECT_EQ(interval.first,maximum); EXPECT_EQ(interval.second,maximum);
  }
}
TEST(BootstrapSummary, ConfidenceLevelNearestOneHasFiniteNormalBounds) {
  const auto r=bootstrap_summary(2.5,{1,2,3,4},{},std::nextafter(1.0,0.0));
  EXPECT_TRUE(std::isfinite(r.normal_interval.first));
  EXPECT_TRUE(std::isfinite(r.normal_interval.second));
  EXPECT_LT(r.normal_interval.first,r.normal_interval.second);
}
TEST(BootstrapSummary, RejectsNonfiniteValuesAndInvalidOptions) {
  EXPECT_THROW(bootstrap_summary(1,{}),std::invalid_argument);
  EXPECT_THROW(bootstrap_summary(1,{1},{},missing_value),std::invalid_argument);
  EXPECT_THROW(bootstrap_summary(1,{1},{},0),std::invalid_argument);
  EXPECT_THROW(bootstrap_summary(1,{1},{},1),std::invalid_argument);
  EXPECT_THROW(bootstrap_summary(infinity,{1}),std::domain_error);
  EXPECT_THROW(bootstrap_summary(1,{1,missing_value}),std::domain_error);
  EXPECT_THROW(bootstrap_summary(1,{1},{infinity,1}),std::domain_error);
  EXPECT_THROW(bootstrap_summary(maximum,{-maximum}),std::overflow_error);
}
TEST(BootstrapSummary, AccelerationAndVarianceSurviveChangesOfUnits) {
  const auto reference=bootstrap_summary(2.5,{1,2,3,4,5},{1,2,3,6});
  EXPECT_NE(reference.bca_interval,reference.percentile_interval);
  for (const double scale:{1e-200,1e200}) {
    const auto scaled=bootstrap_summary(2.5*scale,{scale,2*scale,3*scale,4*scale,5*scale},
        {scale,2*scale,3*scale,6*scale});
    EXPECT_NEAR(scaled.standard_error/scale,reference.standard_error,1e-13);
    EXPECT_NEAR(scaled.bias/scale,reference.bias,1e-13);
    EXPECT_NEAR(scaled.bca_interval.first/scale,reference.bca_interval.first,1e-13);
    EXPECT_NEAR(scaled.bca_interval.second/scale,reference.bca_interval.second,1e-13);
  }
}
TEST(BootstrapSummary, SingularBcaAdjustmentFallsBackToOrderedPercentiles) {
  std::vector<double> replicates(100001,0),jackknife(100,1);
  replicates.back()=2;jackknife[0]=-99;
  const auto r=bootstrap_summary(1,replicates,jackknife);
  EXPECT_EQ(r.bca_interval,r.percentile_interval);
  EXPECT_LE(r.bca_interval.first,r.bca_interval.second);
}
TEST(BootstrapFinite, DefaultStreamPreservedAndNativeMedianCallable) {
  const std::vector<double> sample{1,2,4};
  const auto r=dm::stats::Bootstrap({200,.95,7}).run(sample,dm::median);
  std::mt19937_64 engine(7);std::uniform_int_distribution<std::size_t> index(0,2);
  for (double actual:r.replicates) {
    std::vector<double> values(3);for (auto& value:values) value=sample[index(engine)];
    EXPECT_EQ(actual,dm::median(values));
  }
  const auto again=dm::stats::Bootstrap({200,.95,7}).run(sample,dm::median);
  EXPECT_EQ(r.replicates,again.replicates);
}
TEST(BootstrapFinite, CallerOwnedStreamIsReproducibleAndAdvanced) {
  dm::random::Xoshiro256StarStar first(123),second(123),untouched(123);
  const dm::stats::Bootstrap boot({100,.95,7});
  EXPECT_EQ(boot.run({1,2,4},[](const auto& v){return dm::mean(v);},first).replicates,boot.run({1,2,4},[](const auto& v){return dm::mean(v);},second).replicates);
  EXPECT_EQ(first.next_u64(),second.next_u64());
  EXPECT_NE(first.next_u64(),untouched.next_u64());
}
TEST(BootstrapFinite, RejectsNonfiniteOriginalResampleAndJackknifeStatistics) {
  const dm::stats::Bootstrap boot({200,.95,7});
  EXPECT_THROW(boot.run({1,2,4},[](const auto&){return missing_value;}),std::domain_error);
  std::size_t calls=0;
  EXPECT_THROW(boot.run({1,2,4},[&](const auto&){return calls++==0?1:infinity;}),std::domain_error);
  EXPECT_THROW(boot.run({1,2,4},[](const auto& v){return v.size()<3?missing_value:dm::mean(v);}),std::domain_error);
  EXPECT_THROW(boot.run({1},dm::stats::SampleStatistic{}),std::bad_function_call);
}
TEST(BootstrapFinite, StatisticControlsInputFiltering) {
  const auto r=dm::stats::Bootstrap({20,.95,1}).run({missing_value,infinity},[](const auto&){return 4;});
  EXPECT_EQ(r.estimate,4); EXPECT_EQ(r.standard_error,0);
  EXPECT_EQ(r.bca_interval,(std::pair{4.0,4.0}));
}
