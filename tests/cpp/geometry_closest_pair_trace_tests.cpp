#include <datamunge/geometry/closest_pair.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
namespace {
namespace g=datamunge::geometry;using Points=std::vector<g::Point2D>;
void integer_oracle(const Points& p) {
  const auto t=g::closest_pair_trace(p);const auto r=g::closest_pair_indexed(p);const auto old=g::closest_pair(p);
  long long best=std::numeric_limits<long long>::max();std::size_t a=0,b=0,ordinal=0,ties=0,zeros=0;std::vector<g::ClosestPairImprovement> records;
  for(std::size_t i=0;i<p.size();++i)for(std::size_t j=i+1;j<p.size();++j) {
    const auto dx=static_cast<long long>(p[i].x)-static_cast<long long>(p[j].x),dy=static_cast<long long>(p[i].y)-static_cast<long long>(p[j].y),sq=dx*dx+dy*dy;++ordinal;zeros+=sq==0;
    if(sq<best){best=sq;a=i;b=j;records.push_back({ordinal,i,j,std::sqrt(static_cast<long double>(sq))});}else if(sq==best)++ties;
  }
  ASSERT_EQ(t.improvements.size(),records.size());
  for(std::size_t k=0;k<records.size();++k){const auto& actual=t.improvements[k];const auto& expected=records[k];EXPECT_EQ(actual.first,expected.first);EXPECT_EQ(actual.second,expected.second);EXPECT_EQ(actual.comparison,expected.comparison);EXPECT_NEAR(actual.distance,expected.distance,1e-12);if(k)EXPECT_LT(actual.distance,t.improvements[k-1].distance);}
  EXPECT_EQ(t.result.first,a);EXPECT_EQ(t.result.second,b);EXPECT_EQ(r.first,a);EXPECT_EQ(r.second,b);EXPECT_EQ(t.result.a,p[a]);EXPECT_EQ(t.result.b,p[b]);EXPECT_EQ(old.a,p[a]);EXPECT_EQ(old.b,p[b]);EXPECT_EQ(old.distance,static_cast<double>(r.distance));EXPECT_EQ(r.distance,t.result.distance);EXPECT_NEAR(r.distance,std::sqrt(static_cast<long double>(best)),1e-12);
  EXPECT_EQ(t.work.validated_points,p.size());EXPECT_EQ(t.work.pair_visits,ordinal);EXPECT_EQ(t.work.distance_evaluations,ordinal);EXPECT_EQ(t.work.improvements,records.size());EXPECT_EQ(t.work.ties,ties);EXPECT_EQ(t.work.zero_pairs,zeros);
}
}
TEST(ClosestPairTrace, HandExampleRetainsEveryActualStrictImprovementAndOriginalIds) {
  const Points p{{0,0},{3,4},{0,1},{0.25,1}};const auto t=g::closest_pair_trace(p);
  ASSERT_EQ(t.improvements.size(),3u);EXPECT_EQ(t.improvements[0].comparison,1u);EXPECT_EQ(t.improvements[1].comparison,2u);EXPECT_EQ(t.improvements[2].comparison,6u);EXPECT_EQ(t.result.first,2u);EXPECT_EQ(t.result.second,3u);EXPECT_EQ(t.result.distance,0.25);EXPECT_EQ(t.work.pair_visits,6u);
}
TEST(ClosestPairTrace, AllFourPointTuplesOnThreeByThreeGridMatchExactIntegerSquaredDistances) {
  for(std::size_t code=0;code<6561;++code){auto value=code;Points p;for(int i=0;i<4;++i){const auto id=value%9;value/=9;p.push_back({double(id%3),double(id/3)});}integer_oracle(p);}
}
TEST(ClosestPairTrace, EveryNontrivialSubsetOfThreeByThreeGridMatchesExactSquaredOracle) {
  std::size_t sets=0;for(unsigned mask=0;mask<512;++mask){Points p;for(int i=0;i<9;++i)if(mask&(1u<<i))p.push_back({double(i%3),double(i/3)});if(p.size()>=2){integer_oracle(p);++sets;}}EXPECT_EQ(sets,502u);
}
TEST(ClosestPairTrace, ThousandRandomIntegerCloudsCheckEveryTraceRecordAndNamedCounter) {
  std::mt19937_64 random(424242);for(int trial=0;trial<1000;++trial){Points p(2+random()%24);for(auto& v:p)v={double(int(random()%2001)-1000),double(int(random()%2001)-1000)};integer_oracle(p);}
}
TEST(ClosestPairTrace, AllInputPermutationsAndExactSimilarityTransformsKeepIndexedTieContract) {
  const Points base{{0,0},{2,0},{0,3},{4,4},{6,4},{-5,7}};std::vector<std::size_t> order{0,1,2,3,4,5};std::size_t permutations=0;
  do{Points p;for(auto i:order)p.push_back(base[i]);integer_oracle(p);Points transformed;for(auto v:p)transformed.push_back({-8*v.y+1024,8*v.x-2048});integer_oracle(transformed);++permutations;}while(std::next_permutation(order.begin(),order.end()));EXPECT_EQ(permutations,720u);
}
TEST(ClosestPairTrace, DuplicateIdentityAndZeroDistanceNeverStopRemainingPairWork) {
  const Points p(100,{2,3});g::ClosestPairOptions o;o.max_improvements=1;const auto t=g::closest_pair_trace(p,o);EXPECT_EQ(t.result.first,0u);EXPECT_EQ(t.result.second,1u);EXPECT_EQ(t.result.distance,0);EXPECT_EQ(t.improvements.size(),1u);EXPECT_EQ(t.work.pair_visits,4950u);EXPECT_EQ(t.work.zero_pairs,4950u);EXPECT_EQ(t.work.ties,4949u);
}
TEST(ClosestPairTrace, TinyDistancesDoNotCollapseToSquaredDoubleZero) {
  const auto t=g::closest_pair_trace({{0,0},{3e-200,4e-200},{0,1e-200}});EXPECT_EQ(t.result.first,0u);EXPECT_EQ(t.result.second,2u);EXPECT_GT(t.result.distance,0);EXPECT_EQ(g::closest_pair({{0,0},{0,1e-200}}).distance,1e-200);
  const double small=std::numeric_limits<double>::denorm_min();const auto r=g::closest_pair_indexed({{0,0},{0,small},{0,4*small}});EXPECT_EQ(r.first,0u);EXPECT_EQ(r.second,1u);EXPECT_EQ(r.distance,small);
}
TEST(ClosestPairTrace, HugeDistancesAndFarIrrelevantPairsDoNotLeaveDefaultCoordinates) {
  const Points p{{0,0},{3e200,4e200},{0,1e200}};const auto r=g::closest_pair_indexed(p);EXPECT_EQ(r.first,0u);EXPECT_EQ(r.second,2u);EXPECT_EQ(g::closest_pair(p).distance,1e200);
  const double big=std::numeric_limits<double>::max();const auto wide=g::closest_pair_indexed({{-big,0},{big,0},{0,0}});EXPECT_EQ(wide.first,0u);EXPECT_EQ(wide.second,2u);EXPECT_EQ(wide.distance,big);EXPECT_EQ(g::closest_pair({{-big,0},{big,0},{0,0}}).distance,big);
}
TEST(ClosestPairTrace, WideSelectedDistanceIsAvailableWhileLegacyDoubleOverflowThrows) {
  const double big=std::numeric_limits<double>::max();const Points p{{-big,0},{big,0}};const auto r=g::closest_pair_indexed(p);EXPECT_GT(r.distance,big);EXPECT_TRUE(std::isfinite(r.distance));
  EXPECT_THROW(g::closest_pair(p),std::overflow_error);
}
TEST(ClosestPairTrace, RoundedHypotTiesAreDocumentedRatherThanClaimedExactPredicates) {
  const double epsilon=std::ldexp(1.0,-std::numeric_limits<long double>::digits);const auto t=g::closest_pair_trace({{0,0},{1,epsilon},{10,0},{11,0}});EXPECT_GT(epsilon,0);EXPECT_EQ(t.result.first,0u);EXPECT_EQ(t.result.second,1u);EXPECT_EQ(t.result.distance,1);EXPECT_EQ(t.work.ties,1u);
}
TEST(ClosestPairTrace, FullFiniteInputValidationPrecedesPositivePairCapEvenAfterDuplicate) {
  const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();g::ClosestPairOptions o;o.max_pair_visits=1;
  EXPECT_THROW(g::closest_pair_trace({{0,0},{0,0},{nan,0}},o),std::invalid_argument);
  EXPECT_THROW(g::closest_pair_indexed({{0,0},{inf,0}}),std::invalid_argument);
  EXPECT_THROW(g::closest_pair({{0,0},{0,-inf}}),std::invalid_argument);
  EXPECT_THROW(g::closest_pair({}),std::invalid_argument);
  EXPECT_THROW(g::closest_pair_trace({{0,0}}),std::invalid_argument);
}
TEST(ClosestPairTrace, CompleteWorkCapAndOnlineActualImprovementCapHaveDifferentSemantics) {
  const Points p{{0,0},{3,4},{0,1},{0.25,1}};g::ClosestPairOptions o;o.max_pair_visits=6;o.max_improvements=3;EXPECT_EQ(g::closest_pair_trace(p,o).improvements.size(),3u);o.max_pair_visits=5;
  EXPECT_THROW(g::closest_pair_trace(p,o),std::length_error);
  EXPECT_THROW(g::closest_pair_indexed(p,o),std::length_error);
  o.max_pair_visits=6;o.max_improvements=2;
  EXPECT_THROW(g::closest_pair_trace(p,o),std::length_error);
  EXPECT_EQ(g::closest_pair_indexed(p,o).distance,0.25);
  const Points tied(4,{0,0});o.max_improvements=1;EXPECT_EQ(g::closest_pair_trace(tied,o).improvements.size(),1u);
}
TEST(ClosestPairTrace, PairForecastRejectsHugeInputBeforeQuadraticExecution) {
  const Points p(100000,{0,0});g::ClosestPairOptions o;o.max_pair_visits=1;
  EXPECT_THROW(g::closest_pair_trace(p,o),std::length_error);
}
TEST(ClosestPairTrace, MeasuredPairCountsFollowClosedFormAcrossDoublingSizes) {
  for(std::size_t n=2;n<=1024;n*=2){Points p;for(std::size_t i=0;i<n;++i)p.push_back({double(i),0});const auto t=g::closest_pair_trace(p);EXPECT_EQ(t.work.pair_visits,n*(n-1)/2);EXPECT_EQ(t.work.improvements,1u);EXPECT_EQ(t.work.ties,n-2);EXPECT_EQ(t.result.first,0u);EXPECT_EQ(t.result.second,1u);}
}
TEST(ClosestPairTrace, OriginalMt19937CloudAndPlantedPairArePreservedByReusableGenerator) {
  auto p=g::uniform_rectangle_points(60,424242);std::mt19937 engine(424242);std::uniform_real_distribution<double> draw(0,10);for(const auto& v:p){EXPECT_EQ(v.x,draw(engine));EXPECT_EQ(v.y,draw(engine));}
  p.push_back({5,5});p.push_back({5.03,4.98});const auto r=g::closest_pair_indexed(p);EXPECT_EQ(r.first,60u);EXPECT_EQ(r.second,61u);EXPECT_NEAR(r.distance,std::sqrt(0.0013L),1e-15);EXPECT_EQ(g::closest_pair_trace(p).work.pair_visits,1891u);
}
TEST(ClosestPairTrace, RectangleSamplingValidatesBoundsAndPreservesHalfOpenIntervals) {
  const auto p=g::uniform_rectangle_points(1000,123,-4,-2,1,8);EXPECT_EQ(p,g::uniform_rectangle_points(1000,123,-4,-2,1,8));for(const auto v:p){EXPECT_GE(v.x,-4);EXPECT_LT(v.x,-2);EXPECT_GE(v.y,1);EXPECT_LT(v.y,8);}EXPECT_TRUE(g::uniform_rectangle_points(0).empty());
  const double inf=std::numeric_limits<double>::infinity(),big=std::numeric_limits<double>::max();
  EXPECT_THROW(g::uniform_rectangle_points(0,0,1,1),std::invalid_argument);
  EXPECT_THROW(g::uniform_rectangle_points(0,0,0,1,inf,2),std::invalid_argument);
  EXPECT_THROW(g::uniform_rectangle_points(0,0,-big,big),std::invalid_argument);
  const double high=std::nextafter(1.0,2.0);for(const auto v:g::uniform_rectangle_points(1000,0,1,high,1,high)){EXPECT_EQ(v.x,1);EXPECT_EQ(v.y,1);}
}
TEST(ClosestPairTrace, OwnedResultAndImprovementRecordsSurviveInputAndRecordMutation) {
  Points p{{0,0},{3,4},{0,1},{0.25,1}};auto t=g::closest_pair_trace(p);const auto last=t.improvements.back();p[2]={99,99};t.improvements[0].distance=999;EXPECT_EQ(t.result.a,(g::Point2D{0,1}));EXPECT_EQ(t.result.distance,last.distance);EXPECT_EQ(t.improvements.back().distance,last.distance);
}
