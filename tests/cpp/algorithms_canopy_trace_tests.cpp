#include <gtest/gtest.h>
#include <datamunge/algorithms/canopy_clustering.hpp>
#include <datamunge/datasets/translated_grid_2d.hpp>
#include <random>
#include <numeric>
namespace {
namespace a=datamunge::algorithms;
using Points=std::vector<std::vector<double>>;
// Independent live-list reference uses squared distances only on small exact integers.
std::vector<a::Canopy> reference(const Points& points,double t1,double t2) {
  std::vector<std::size_t> eligible(points.size());std::iota(eligible.begin(),eligible.end(),std::size_t{0});
  std::vector<a::Canopy> result;
  while(!eligible.empty()) {
    const auto seed=eligible.front();a::Canopy canopy;canopy.center=seed;
    std::vector<std::size_t> next;
    for(std::size_t j=0;j<points.size();++j) {
      double squared=0;for(std::size_t d=0;d<points[j].size();++d) {const double difference=points[seed][d]-points[j][d];squared+=difference*difference;}
      if(squared<t1*t1) canopy.members.push_back(j);
    }
    for(auto point:eligible) {
      double squared=0;for(std::size_t d=0;d<points[point].size();++d) {const double diff=points[seed][d]-points[point][d];squared+=diff*diff;}
      if(point!=seed && !(squared<t2*t2)) next.push_back(point);
    }
    result.push_back(std::move(canopy));eligible=std::move(next);
  }
  return result;
}
void compare(const std::vector<a::Canopy>& actual,const std::vector<a::Canopy>& expected) {
  ASSERT_EQ(actual.size(),expected.size());
  for(std::size_t i=0;i<actual.size();++i) {EXPECT_EQ(actual[i].center,expected[i].center);EXPECT_EQ(actual[i].members,expected[i].members);}
}
void verify(const Points& points,double t1,double t2) {
  const auto trace=a::canopy_clustering_trace(points,t1,t2);const auto want=reference(points,t1,t2);
  compare(trace.canopies,want);compare(a::canopy_clustering(points,t1,t2),want);
  EXPECT_EQ(trace.visits.size(),points.size()*want.size());EXPECT_EQ(trace.distance_evaluations,trace.visits.size());
  EXPECT_EQ(trace.coordinate_differences,trace.visits.size()*(points.empty()?0:points[0].size()));
  ASSERT_EQ(trace.steps.size(),want.size());
  std::vector<bool> removed(points.size(),false);std::size_t remaining=points.size(),memberships=0;
  for(std::size_t id=0;id<want.size();++id) {
    const auto& step=trace.steps[id];const auto seed=want[id].center;
    EXPECT_FALSE(removed[seed]);EXPECT_EQ(step.remaining_before,remaining);EXPECT_EQ(step.visit_begin,id*points.size());EXPECT_EQ(step.visit_count,points.size());
    std::vector<std::size_t> new_removals;
    for(std::size_t j=0;j<points.size();++j) {
      const auto& visit=trace.visits[id*points.size()+j];EXPECT_EQ(visit.canopy,id);EXPECT_EQ(visit.point,j);EXPECT_EQ(visit.already_removed,removed[j]);
      double squared=0;for(std::size_t d=0;d<points[j].size();++d) {const double delta=points[seed][d]-points[j][d];squared+=delta*delta;}
      EXPECT_NEAR(visit.distance,std::sqrt(squared),1e-13);EXPECT_EQ(visit.member,squared<t1*t1);EXPECT_EQ(visit.within_tight,squared<t2*t2);
      const bool newly=!removed[j] && (j==seed || squared<t2*t2);EXPECT_EQ(visit.newly_removed,newly);
      if(newly) {new_removals.push_back(j);removed[j]=true;--remaining;}
      memberships+=visit.member;
    }
    EXPECT_EQ(step.newly_removed,new_removals);EXPECT_EQ(step.remaining_after,remaining);
  }
  EXPECT_EQ(remaining,0u);EXPECT_EQ(trace.total_memberships,memberships);
  const auto coverage=a::canopy_coverage(points.size(),want);
  EXPECT_EQ(coverage.total_memberships,memberships);EXPECT_EQ(coverage.uncovered_points,0u);
  std::vector<std::pair<std::size_t,std::size_t>> pairs;std::size_t pair_visits=0,overlap=0;
  for(std::size_t i=0;i<points.size();++i) {
    std::vector<std::size_t> memberships_for_point;
    for(std::size_t c=0;c<want.size();++c) if(std::find(want[c].members.begin(),want[c].members.end(),i)!=want[c].members.end()) memberships_for_point.push_back(c);
    EXPECT_EQ(coverage.point_canopies[i],memberships_for_point);overlap+=memberships_for_point.size()>1;
    for(std::size_t j=i+1;j<points.size();++j) {
      bool shared=false;
      for(const auto& canopy:want) {
        const bool both=std::find(canopy.members.begin(),canopy.members.end(),i)!=canopy.members.end() && std::find(canopy.members.begin(),canopy.members.end(),j)!=canopy.members.end();
        pair_visits+=both;shared|=both;
      }
      if(shared) pairs.emplace_back(i,j);
    }
  }
  EXPECT_EQ(coverage.shared_pairs,pairs);EXPECT_EQ(coverage.pair_visits,pair_visits);EXPECT_EQ(coverage.overlapping_points,overlap);
}
}
TEST(CanopyTrace, TinyActualDistancesAndConsumedPointsRemainMembers) {
  const auto t=a::canopy_clustering_trace({{0},{1},{2},{3}},2.5,1.5);
  ASSERT_EQ(t.canopies.size(),2u);EXPECT_EQ(t.canopies[0].members,(std::vector<std::size_t>{0,1,2}));
  EXPECT_EQ(t.canopies[1].members,(std::vector<std::size_t>{0,1,2,3}));EXPECT_TRUE(t.visits[4].already_removed);EXPECT_TRUE(t.visits[4].member);
  EXPECT_EQ(t.steps[0].newly_removed,(std::vector<std::size_t>{0,1}));EXPECT_EQ(t.steps[1].newly_removed,(std::vector<std::size_t>{2,3}));
  EXPECT_EQ(t.distance_evaluations,8u);const auto s=a::canopy_coverage(4,t.canopies);
  EXPECT_EQ(s.total_memberships,7u);EXPECT_EQ(s.overlapping_points,3u);EXPECT_EQ(s.pair_visits,9u);EXPECT_EQ(s.shared_pairs.size(),6u);
}
TEST(CanopyTrace, ExhaustiveOrderedOneDimensionalArrays) {
  std::size_t configurations=0;
  for(std::size_t n=0,power=1;n<=5;++n,power*=4) for(std::size_t code=0;code<power;++code) {
    Points points(n,std::vector<double>(1));auto digits=code;
    for(auto& point:points) {point[0]=digits%4;digits/=4;}
    for(auto [loose,tight]:std::vector<std::pair<double,double>>{{.5,0},{1,1},{1.5,.5},{2.5,1.5},{4,4}}) {verify(points,loose,tight);++configurations;}
  }
  EXPECT_EQ(configurations,6825u);
}
TEST(CanopyTrace, RandomMultidimensionalLiveListAndPairOracle) {
  std::mt19937 rng(2000);
  for(int trial=0;trial<1000;++trial) {
    const auto n=rng()%31,d=rng()%5;Points points(n,std::vector<double>(d));
    for(auto& row:points) for(auto& value:row) value=static_cast<int>(rng()%11)-5;
    // Half-integer radii avoid square-distance equality and rounding-sensitive cases.
    const double tight=(rng()%5)+.5,loose=tight+(rng()%5);
    verify(points,loose,tight);
  }
}
TEST(CanopyTrace, StrictEqualityZeroTightAndZeroDimension) {
  const auto boundary=a::canopy_clustering_trace({{0},{1}},1,1);
  EXPECT_EQ(boundary.canopies.size(),2u);EXPECT_EQ(boundary.canopies[0].members,(std::vector<std::size_t>{0}));EXPECT_FALSE(boundary.visits[1].within_tight);
  const auto zero=a::canopy_clustering_trace({{0},{0}},1,0);EXPECT_EQ(zero.canopies.size(),2u);EXPECT_TRUE(zero.visits[0].newly_removed);EXPECT_FALSE(zero.visits[0].within_tight);
  EXPECT_EQ(zero.steps[0].remaining_after,1u);EXPECT_EQ(zero.steps[1].remaining_after,0u);
  EXPECT_EQ(a::canopy_clustering({{},{}},1,1).size(),1u);EXPECT_EQ(a::canopy_clustering({{},{}},1,0).size(),2u);
  EXPECT_TRUE(a::canopy_clustering({},1,0).empty());verify({{},{}},1,0);
}
TEST(CanopyTrace, ClosePairCounterexampleAndSufficientRadius) {
  const Points points{{0},{2},{.875},{1.125}};
  const auto small=a::canopy_coverage(4,a::canopy_clustering(points,1.125,1));
  EXPECT_EQ(small.shared_pairs,(std::vector<std::pair<std::size_t,std::size_t>>{{0,2},{1,3}}));
  EXPECT_FALSE(std::binary_search(small.shared_pairs.begin(),small.shared_pairs.end(),std::pair<std::size_t,std::size_t>{2,3}));
  const auto safe=a::canopy_coverage(4,a::canopy_clustering(points,2,1));
  EXPECT_EQ(safe.shared_pairs.size(),5u);EXPECT_EQ(safe.pair_visits,6u);
  EXPECT_TRUE(std::binary_search(safe.shared_pairs.begin(),safe.shared_pairs.end(),std::pair<std::size_t,std::size_t>{2,3}));
}
TEST(CanopyTrace, SufficientMetricConditionAcrossRandomInputsAndOrders) {
  std::mt19937 rng(42);
  for(int trial=0;trial<500;++trial) {
    Points points(20,std::vector<double>(2));for(auto& p:points) for(auto& v:p)v=static_cast<int>(rng()%21)-10;
    std::shuffle(points.begin(),points.end(),rng);
    const double tight=2.5,radius=3.5;
    const auto summary=a::canopy_coverage(points.size(),a::canopy_clustering(points,tight+radius,tight));
    for(std::size_t i=0;i<points.size();++i) for(std::size_t j=i+1;j<points.size();++j) {
      const auto dx=points[i][0]-points[j][0],dy=points[i][1]-points[j][1];
      if(dx*dx+dy*dy<radius*radius) EXPECT_TRUE(std::binary_search(summary.shared_pairs.begin(),summary.shared_pairs.end(),std::pair{i,j}));
    }
  }
}
TEST(CanopyTrace, LooseRadiusKeepsSeedsAndIncreasesMembership) {
  const Points p{{0},{1},{2},{4},{6},{7}};
  std::vector<std::size_t> seeds;std::set<std::pair<std::size_t,std::size_t>> previous;
  for(double loose:{1.5,2.5,4.5,8.0}) {
    const auto c=a::canopy_clustering(p,loose,1.5);std::vector<std::size_t> current;for(const auto& canopy:c)current.push_back(canopy.center);
    if(seeds.empty())seeds=current;EXPECT_EQ(seeds,current);
    const auto s=a::canopy_coverage(p.size(),c);const std::set<std::pair<std::size_t,std::size_t>> pairs(s.shared_pairs.begin(),s.shared_pairs.end());
    EXPECT_TRUE(std::includes(pairs.begin(),pairs.end(),previous.begin(),previous.end()));previous=pairs;
  }
}
TEST(CanopyTrace, SeedChoiceDependsOnInputOrder) {
  auto left=a::canopy_clustering({{0},{1},{2}},2,1.5),middle=a::canopy_clustering({{1},{0},{2}},2,1.5);
  EXPECT_EQ(left.size(),2u);EXPECT_EQ(middle.size(),1u);
}
TEST(CanopyTrace, HypotAvoidsSquareOverflowAndUnderflow) {
  EXPECT_EQ(a::canopy_clustering({{0},{1e200}},1.1e200,1.1e200).size(),1u);
  EXPECT_EQ(a::canopy_clustering({{0},{1e-200}},.5e-200,.5e-200).size(),2u);
  const auto huge=a::canopy_clustering_trace({{std::numeric_limits<double>::max()},{-std::numeric_limits<double>::max()}},1,1);
  EXPECT_EQ(huge.canopies.size(),2u);EXPECT_TRUE(std::isinf(huge.visits[1].distance));
}
TEST(CanopyTrace, DistanceAndPairBudgetsAreWholeCall) {
  const Points points{{0},{1},{2},{3}};
  for(std::size_t cap=1;cap<8;++cap) EXPECT_THROW(a::canopy_clustering_trace(points,2.5,1.5,cap),std::length_error);
  const auto t=a::canopy_clustering_trace(points,2.5,1.5,8);EXPECT_EQ(t.distance_evaluations,8u);
  for(std::size_t cap=1;cap<9;++cap) EXPECT_THROW(a::canopy_coverage(4,t.canopies,cap),std::length_error);
  EXPECT_EQ(a::canopy_coverage(4,t.canopies,9).pair_visits,9u);
}
TEST(CanopyTrace, RejectsBadInputsEvenWhenEmptyOrPure) {
  EXPECT_THROW(a::canopy_clustering({},0,0),std::invalid_argument);
  EXPECT_THROW(a::canopy_clustering({},1,-1),std::invalid_argument);
  EXPECT_THROW(a::canopy_clustering({},1,2),std::invalid_argument);
  EXPECT_THROW(a::canopy_clustering({{0},{0,1}},1,1),std::invalid_argument);
  for(double bad:{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(a::canopy_clustering({},bad,0),std::invalid_argument);
    EXPECT_THROW(a::canopy_clustering({},1,bad),std::invalid_argument);
    EXPECT_THROW(a::canopy_clustering({{bad}},1,0),std::invalid_argument);
  }
}
TEST(CanopyTrace, CoverageValidatesManualRecordsWithoutChangingThem) {
  const std::vector<a::Canopy> records{{0,{2,0,1}},{0,{1,2}}};const auto s=a::canopy_coverage(4,records);
  EXPECT_EQ(s.total_memberships,5u);EXPECT_EQ(s.overlapping_points,2u);EXPECT_EQ(s.uncovered_points,1u);EXPECT_EQ(s.pair_visits,4u);EXPECT_EQ(s.shared_pairs.size(),3u);
  EXPECT_EQ(records[0].members,(std::vector<std::size_t>{2,0,1}));
  EXPECT_THROW(a::canopy_coverage(3,{{3,{}}}),std::invalid_argument);
  EXPECT_THROW(a::canopy_coverage(3,{{0,{3}}}),std::invalid_argument);
  EXPECT_THROW(a::canopy_coverage(3,{{0,{1,1}}}),std::invalid_argument);
  EXPECT_EQ(a::canopy_coverage(3,{}).uncovered_points,3u);EXPECT_TRUE(a::canopy_coverage(0,{}).shared_pairs.empty());
}
TEST(CanopyTrace, CentresAndTraceAreIndependentOwnedCopies) {
  const Points points{{0,1},{2,3}};const auto c=a::canopy_clustering(points,1,1);
  auto centres=a::canopy_centers(points,c);centres[0][0]=99;EXPECT_EQ(points[0][0],0);
  EXPECT_THROW(a::canopy_centers(points,{{2,{}}}),std::out_of_range);
  EXPECT_EQ(a::canopy_centers({{1},{2,3}},{{1,{999}}}),(Points{{2,3}}));
  auto trace=a::canopy_clustering_trace(points,1,1),copy=trace;copy.canopies[0].members.clear();copy.visits[0].distance=99;copy.steps[0].newly_removed.clear();
  EXPECT_EQ(trace.canopies[0].members.size(),1u);EXPECT_EQ(trace.visits[0].distance,0);EXPECT_EQ(trace.steps[0].newly_removed.size(),1u);
}
TEST(CanopyTrace, OriginalLatticeAndPairDeduplication) {
  const Points centres{{0,0},{10,0},{5,8}};
  const auto points=datamunge::datasets::translated_grid_2d(centres,6,5,.05,.04,-.15,-.1);ASSERT_EQ(points.size(),90u);
  Points old;for(const auto& c:centres)for(int i=0;i<30;++i)old.push_back({c[0]+.05*(i%6)-.15,c[1]+.04*(i/6)-.1});EXPECT_EQ(points,old);
  for(double loose:{3.0,6.0,9.0,12.0}) {
    const auto t=a::canopy_clustering_trace(points,loose,2.5);ASSERT_EQ(t.canopies.size(),3u);
    EXPECT_EQ(t.canopies[0].center,0u);EXPECT_EQ(t.canopies[1].center,30u);EXPECT_EQ(t.canopies[2].center,60u);EXPECT_EQ(t.distance_evaluations,270u);EXPECT_EQ(t.coordinate_differences,540u);
    const auto s=a::canopy_coverage(90,t.canopies);EXPECT_EQ(s.total_memberships,loose==12?270u:90u);
    EXPECT_EQ(s.shared_pairs.size(),loose==12?4005u:1305u);EXPECT_EQ(s.pair_visits,loose==12?12015u:1305u);
  }
}
TEST(CanopyTrace, TranslatedGridEmptyShapeStepsAndOverflow) {
  using datamunge::datasets::translated_grid_2d;
  EXPECT_EQ(translated_grid_2d({{0,0}},2,2,1,2),(Points{{0,0},{1,0},{0,2},{1,2}}));
  EXPECT_TRUE(translated_grid_2d({},2,2,1,1).empty());EXPECT_TRUE(translated_grid_2d({{0,0}},0,2,1,1).empty());
  EXPECT_EQ(translated_grid_2d({{1,2}},2,1,-1,0),(Points{{1,2},{0,2}}));
  EXPECT_THROW(translated_grid_2d({{1}},0,0,1,1),std::invalid_argument);
  EXPECT_THROW(translated_grid_2d({},0,0,std::numeric_limits<double>::infinity(),1),std::invalid_argument);
  EXPECT_THROW(translated_grid_2d({{0,0}},std::numeric_limits<std::size_t>::max(),2,1,1),std::length_error);
  EXPECT_THROW(translated_grid_2d({{std::numeric_limits<double>::max(),0}},2,1,std::numeric_limits<double>::max(),1),std::overflow_error);
}
TEST(CanopyTrace, LargerTightRadiusCanIncreaseGreedySeedCount) {
  const Points p{{0,0},{2,0},{2.5,1.5},{2.5,-1.5}};
  const auto small=a::canopy_clustering(p,4,1.625),large=a::canopy_clustering(p,4,2.25);
  ASSERT_EQ(small.size(),2u);EXPECT_EQ(small[1].center,1u);
  ASSERT_EQ(large.size(),3u);EXPECT_EQ(large[1].center,2u);EXPECT_EQ(large[2].center,3u);
}
