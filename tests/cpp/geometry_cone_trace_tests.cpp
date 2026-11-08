#include <gtest/gtest.h>
#include <datamunge/geometry/cone_algorithm.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {
using namespace datamunge::geometry;
constexpr long double pi=std::numbers::pi_v<long double>,tau=2*pi;
// Exact supporting-line oracle for small integer clouds, including edge points
// and coincident records. It uses no atan2, angle sorting or floating predicates.
bool supporting_line(const std::vector<Point2D>& p,std::size_t query) {
  bool any=false;
  for(std::size_t j=0;j<p.size();++j) {
    const auto x=static_cast<long long>(p[j].x-p[query].x),y=static_cast<long long>(p[j].y-p[query].y);
    if(x==0 && y==0)continue;
    any=true;bool nonnegative=true,nonpositive=true;
    for(const auto& q:p) {
      const auto cross=x*static_cast<long long>(q.y-p[query].y)-y*static_cast<long long>(q.x-p[query].x);
      nonnegative=nonnegative && cross>=0;nonpositive=nonpositive && cross<=0;
    }
    if(nonnegative || nonpositive)return true;
  }
  return !any;
}
void check_hull(const std::vector<Point2D>& p) {
  const auto a=cone_surface_analyze(p,100,std::numbers::pi_v<double>);
  ASSERT_EQ(a.surface.size(),p.size());
  for(std::size_t i=0;i<p.size();++i)ASSERT_EQ(bool(a.surface[i]),supporting_line(p,i));
  ASSERT_EQ(a.work.pair_visits,p.size()*(p.empty()?0:p.size()-1));
}
long double pairwise_gap(const std::vector<std::pair<long long,long long>>& v) {
  if(v.size()<2)return tau;
  long double largest=0;
  for(const auto& a:v) {
    long double nearest=tau;
    for(const auto& b:v) {
      const auto cross=a.first*b.second-a.second*b.first,dot=a.first*b.first+a.second*b.second;
      auto angle=std::atan2(static_cast<long double>(cross),static_cast<long double>(dot));
      if(angle<=0)angle+=tau;
      nearest=std::min(nearest,angle);
    }
    largest=std::max(largest,nearest);
  }
  return largest;
}
std::vector<char> former_mask(const std::vector<Point2D>& p,double radius,double threshold) {
  std::vector<char> result(p.size());
  for(std::size_t i=0;i<p.size();++i) {
    std::vector<double> a;
    for(std::size_t j=0;j<p.size();++j)if(i!=j) {
      const double x=p[j].x-p[i].x,y=p[j].y-p[i].y;
      if(x*x+y*y<=radius*radius)a.push_back(std::atan2(y,x));
    }
    if(a.size()<2){result[i]=1;continue;}
    std::sort(a.begin(),a.end());double gap=0;
    for(std::size_t k=1;k<a.size();++k)gap=std::max(gap,a[k]-a[k-1]);
    gap=std::max(gap,a.front()+2*std::numbers::pi_v<double>-a.back());result[i]=gap>=threshold;
  }
  return result;
}
}

TEST(ConeTrace, CardinalCrossHasHandCheckableInclusiveRadiusAndDirections) {
  const auto t=cone_surface_trace({{0,0},{1,0},{-1,0},{0,1},{0,-1}},1,pi);
  const auto& q=t.analysis.points[0];ASSERT_EQ(q.neighbors,4);EXPECT_FALSE(q.surface);
  EXPECT_NEAR(q.largest_gap,pi/2,1e-18L);
  ASSERT_EQ(t.directions[0].size(),4);
  EXPECT_EQ(t.directions[0][0].index,4);EXPECT_EQ(t.directions[0][1].index,1);
  EXPECT_EQ(t.directions[0][2].index,3);EXPECT_EQ(t.directions[0][3].index,2);
}

TEST(ConeTrace, WrapGapAndActualWitnessIdsAreRetained) {
  const auto t=cone_surface_trace({{0,0},{1,0},{0,1}},2,3);
  const auto& q=t.analysis.points[0];EXPECT_NEAR(q.largest_gap,3*pi/2,1e-18L);
  EXPECT_EQ(q.gap_from,2);EXPECT_EQ(q.gap_to,1);EXPECT_TRUE(q.surface);
  EXPECT_EQ(t.analysis.work.pair_visits,6);EXPECT_EQ(t.analysis.work.distance_evaluations,6);
  EXPECT_EQ(t.analysis.work.angle_evaluations,6);EXPECT_EQ(t.analysis.work.sort_calls,3);EXPECT_EQ(t.analysis.work.gap_tests,6);
}

TEST(ConeTrace, EmptySparseAndCoincidentRecordsHaveExplicitFullTurnConvention) {
  EXPECT_TRUE(cone_surface_analyze({},1,pi).surface.empty());
  const auto a=cone_surface_trace({{0,0}},1,tau);
  EXPECT_TRUE(a.analysis.points[0].insufficient);EXPECT_EQ(a.analysis.points[0].gap_from,1);
  EXPECT_EQ(a.analysis.points[0].largest_gap,tau);EXPECT_TRUE(a.analysis.points[0].surface);
  const auto b=cone_surface_trace({{2,3},{2,3},{2,3}},1,tau);
  EXPECT_EQ(b.analysis.work.pair_visits,6);EXPECT_EQ(b.analysis.work.coincident_visits,6);
  EXPECT_EQ(b.analysis.work.distance_evaluations,0);EXPECT_EQ(b.analysis.work.angle_evaluations,0);
  for(const auto& q:b.analysis.points){EXPECT_EQ(q.neighbors,0);EXPECT_EQ(q.coincident,2);EXPECT_TRUE(q.surface);}
}

TEST(ConeTrace, CoincidentQueriesCannotInventAnEastwardDirection) {
  const std::vector<Point2D> p{{0,0},{0,0},{0,1},{-1,0},{0,-1}};
  const auto a=cone_surface_analyze(p,1,.75L*pi);
  EXPECT_TRUE(a.points[0].surface);EXPECT_TRUE(a.points[1].surface);
  EXPECT_EQ(a.points[0].neighbors,3);EXPECT_EQ(a.points[0].coincident,1);
  EXPECT_NEAR(a.points[0].largest_gap,pi,1e-18L);
  EXPECT_FALSE(bool(former_mask(p,1,.75*std::numbers::pi_v<double>)[0]));
}

TEST(ConeTrace, EverySmallGridTupleAndSubsetMatchesExactSupportingLines) {
  const std::vector<Point2D> grid{{-1,-1},{0,-1},{1,-1},{-1,0},{0,0},{1,0},{-1,1},{0,1},{1,1}};
  for(std::size_t code=0;code<6561;++code) {
    auto value=code;std::vector<Point2D> p;
    for(int j=0;j<4;++j){p.push_back(grid[value%9]);value/=9;}
    check_hull(p);if(::testing::Test::HasFatalFailure())return;
  }
  for(unsigned mask=0;mask<512;++mask) {
    std::vector<Point2D> p;for(unsigned j=0;j<9;++j)if(mask&(1U<<j))p.push_back(grid[j]);
    check_hull(p);if(::testing::Test::HasFatalFailure())return;
  }
  std::mt19937 engine(111);
  for(int trial=0;trial<1000;++trial) {
    std::vector<Point2D> p(engine()%20);
    for(auto& q:p){q.x=static_cast<int>(engine()%21)-10;q.y=static_cast<int>(engine()%21)-10;}
    check_hull(p);if(::testing::Test::HasFatalFailure())return;
  }
}

TEST(ConeTrace, RandomLocalGapsMatchIndependentPairwiseCrossDotOracle) {
  std::mt19937 engine(898);
  for(int trial=0;trial<500;++trial) {
    std::vector<Point2D> p(1+engine()%15);const int radius=1+engine()%10;
    for(auto& q:p){q.x=static_cast<int>(engine()%11)-5;q.y=static_cast<int>(engine()%11)-5;}
    const auto t=cone_surface_trace(p,radius,.55L*pi);
    for(std::size_t i=0;i<p.size();++i) {
      std::vector<std::pair<long long,long long>> v;
      for(std::size_t j=0;j<p.size();++j)if(i!=j) {
        const auto x=static_cast<long long>(p[j].x-p[i].x),y=static_cast<long long>(p[j].y-p[i].y);
        if((x!=0 || y!=0) && x*x+y*y<=radius*radius)v.emplace_back(x,y);
      }
      ASSERT_EQ(t.analysis.points[i].neighbors,v.size());
      ASSERT_NEAR(t.analysis.points[i].largest_gap,pairwise_gap(v),1e-15L);
      const auto& d=t.directions[i];
      for(std::size_t k=0;k<d.size();++k) {
        ASSERT_EQ(d[k].point,p[d[k].index]);ASSERT_LE(d[k].distance,radius);ASSERT_GT(d[k].distance,0);
        if(k)ASSERT_LE(d[k-1].angle,d[k].angle);
      }
    }
  }
}

TEST(ConeTrace, InclusiveThresholdCanReuseActualGapAndDistanceHasNoTolerance) {
  const std::vector<Point2D> triangle{{0,0},{1,0},{0,1}};
  const auto gap=cone_surface_analyze(triangle,2,1).points[0].largest_gap;
  EXPECT_TRUE(cone_surface_analyze(triangle,2,gap).points[0].surface);
  EXPECT_FALSE(cone_surface_analyze(triangle,2,std::nextafter(gap,tau)).points[0].surface);
  const std::vector<Point2D> line{{0,0},{3,4},{-3,-4}};
  EXPECT_EQ(cone_surface_analyze(line,5,3.5).points[0].neighbors,2);
  EXPECT_EQ(cone_surface_analyze(line,std::nextafter(5.,0.),3.5).points[0].neighbors,0);
}

TEST(ConeTrace, LocalGapCanFlagAnInteriorPointOfTheGlobalConvexHull) {
  const std::vector<Point2D> p{{0,0},{1,0},{-1,0},{0,-1},{0,2}};
  EXPECT_FALSE(supporting_line(p,0));
  EXPECT_TRUE(cone_surface_analyze(p,1,std::numbers::pi_v<double>).points[0].surface);
  EXPECT_FALSE(cone_surface_analyze(p,2.1,std::numbers::pi_v<double>).points[0].surface);
}

TEST(ConeTrace, RadiusAndThresholdMonotonicityAndExactSimilarities) {
  const std::vector<Point2D> p{{0,0},{1,0},{-1,0},{0,-1},{0,2},{3,3},{-3,2}};
  std::vector<long double> previous(p.size(),tau);
  for(double radius:{.5,1.,2.,4.,8.,100.}) {
    const auto a=cone_surface_analyze(p,radius,.55L*pi);
    for(std::size_t i=0;i<p.size();++i){EXPECT_LE(a.points[i].largest_gap,previous[i]);previous[i]=a.points[i].largest_gap;}
    EXPECT_EQ(a.work.pair_visits,p.size()*(p.size()-1));
  }
  auto previous_mask=cone_surface_analyze(p,2.5,1).surface;
  for(long double threshold:{2.L,3.L,4.L,5.L,6.L}) {
    const auto a=cone_surface_analyze(p,2.5,threshold);
    for(std::size_t i=0;i<p.size();++i)EXPECT_LE(a.surface[i],previous_mask[i]);previous_mask=a.surface;
  }
  const auto a=cone_surface_analyze(p,2.5,.55L*pi);
  std::vector<Point2D> moved;for(const auto& q:p)moved.push_back({8-4*q.y,-4+4*q.x});
  const auto b=cone_surface_analyze(moved,10,.55L*pi);EXPECT_EQ(a.surface,b.surface);
  for(std::size_t i=0;i<p.size();++i)EXPECT_NEAR(a.points[i].largest_gap,b.points[i].largest_gap,1e-18L);
  std::vector<Point2D> reversed(p.rbegin(),p.rend());const auto c=cone_surface_analyze(reversed,2.5,.55L*pi);
  for(std::size_t i=0;i<p.size();++i)EXPECT_EQ(a.surface[i],c.surface[p.size()-1-i]);
}

TEST(ConeTrace, WidenedDistancesHandleHugeAndSubnormalCutoffs) {
  const auto maximum=std::numeric_limits<double>::max();
  const std::vector<Point2D> huge{{-maximum,0},{maximum,0},{0,0}};
  if(std::numeric_limits<long double>::max_exponent>std::numeric_limits<double>::max_exponent) {
    EXPECT_EQ(cone_surface_points(huge,maximum,3.5),(std::vector<char>{1,1,0}));
  } else {
    EXPECT_THROW(cone_surface_points(huge,maximum,3.5),std::overflow_error);
  }
  const double tiny=std::numeric_limits<double>::denorm_min();
  const auto t=cone_surface_trace({{0,0},{tiny,0},{2*tiny,0},{0,tiny}},tiny,pi);
  ASSERT_EQ(t.directions[0].size(),2);EXPECT_EQ(t.analysis.points[0].neighbors,2);
  for(const auto& d:t.directions[0])EXPECT_NE(d.index,2);
}

TEST(ConeTrace, CompletePairAndActualDirectionCapsFollowFullValidation) {
  const std::vector<Point2D> p{{0,0},{1,0},{0,1}};
  ConeSurfaceLimits limits;limits.max_pair_visits=6;limits.max_directions=6;
  EXPECT_NO_THROW(cone_surface_trace(p,2,3,limits));
  limits.max_directions=5;EXPECT_THROW(cone_surface_trace(p,2,3,limits),std::length_error);
  EXPECT_NO_THROW(cone_surface_analyze(p,2,3,limits));
  limits.max_directions=1;EXPECT_NO_THROW(cone_surface_trace(p,.1,3,limits));
  limits.max_pair_visits=5;EXPECT_THROW(cone_surface_analyze(p,2,3,limits),std::length_error);
  auto invalid=p;invalid.back().x=std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(cone_surface_analyze(invalid,2,3,limits),std::invalid_argument);
  for(double bad:{0.,-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(cone_surface_analyze({},bad,3),std::invalid_argument);
    EXPECT_THROW(cone_surface_analyze({},1,bad),std::invalid_argument);
  }
  EXPECT_THROW(cone_surface_analyze({},1,std::nextafter(tau,10.L)),std::invalid_argument);
  EXPECT_NO_THROW(cone_surface_analyze({},1,tau));
}

TEST(ConeTrace, OriginalCandidateStreamAndFormerMaskArePreservedOnOrdinaryDisk) {
  const auto p=sample_disk_candidates(2500,10,1);std::vector<Point2D> old;
  std::mt19937 engine(1);std::uniform_real_distribution<double> u(-10,10);
  for(int i=0;i<2500;++i){const double x=u(engine),y=u(engine);if(x*x+y*y<=100)old.push_back({x,y});}
  ASSERT_EQ(p,old);
  const double threshold=std::numbers::pi_v<double>*.55;
  const auto a=cone_surface_analyze(p,2.5,threshold);
  EXPECT_EQ(a.surface,former_mask(old,2.5,threshold));EXPECT_EQ(a.surface,cone_surface_points(p,2.5,threshold));
  const auto radial=radial_flag_summary(p,a.surface,7,9.4);
  EXPECT_GT(radial.flagged_points,0);EXPECT_EQ(radial.inner_flagged,0);EXPECT_GT(radial.min_flagged_radius,8);
  EXPECT_EQ(a.work.pair_visits,p.size()*(p.size()-1));
}

TEST(ConeTrace, DiskSamplerValidatesWidthAndCompleteTrialBudget) {
  EXPECT_TRUE(sample_disk_candidates(0,1).empty());
  EXPECT_THROW(sample_disk_candidates(100,1,1,99),std::length_error);
  EXPECT_THROW(sample_disk_candidates(0,1,1,0),std::invalid_argument);
  EXPECT_THROW(sample_disk_candidates(1,std::numeric_limits<double>::max()),std::invalid_argument);
  for(double radius:{0.,-1.,std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(sample_disk_candidates(1,radius),std::invalid_argument);
  const auto a=sample_disk_candidates(100,1,4),b=sample_disk_candidates(100,1,4);EXPECT_EQ(a,b);
  for(const auto& q:a)EXPECT_LE(std::hypot(static_cast<long double>(q.x),static_cast<long double>(q.y)),1);
}

TEST(ConeTrace, RadialBandsAndAbsentFlagsHaveExplicitSemantics) {
  const std::vector<Point2D> p{{1,0},{2,0},{3,0}};
  const auto a=radial_flag_summary(p,{1,1,0},2,2);
  EXPECT_EQ(a.inner_points,1);EXPECT_EQ(a.inner_flagged,1);EXPECT_EQ(a.outer_points,1);EXPECT_EQ(a.outer_flagged,0);
  EXPECT_EQ(a.min_flagged_radius,1);EXPECT_EQ(a.max_flagged_radius,2);EXPECT_TRUE(a.has_flagged);
  const auto b=radial_flag_summary(p,{0,0,0},0,4);EXPECT_FALSE(b.has_flagged);EXPECT_EQ(b.min_flagged_radius,0);
  EXPECT_THROW(radial_flag_summary(p,{0},1,2),std::invalid_argument);
  EXPECT_THROW(radial_flag_summary(p,{0,2,0},1,2),std::invalid_argument);
  EXPECT_THROW(radial_flag_summary(p,{0,0,0},2,1),std::invalid_argument);
  EXPECT_THROW(radial_flag_summary(p,{0,0,0},1,2,{0,std::numeric_limits<double>::quiet_NaN()}),std::invalid_argument);
}

TEST(ConeTrace, InputAndActualRecordsRetainIndependentOwnedCopies) {
  std::vector<Point2D> p{{0,0},{1,0},{0,1}};auto t=cone_surface_trace(p,2,3);
  p[1].x=99;EXPECT_EQ(t.directions[0][0].point.x,1);
  t.directions[0][0].point.x=88;EXPECT_EQ(t.analysis.points[1].point.x,1);
  EXPECT_EQ(t.analysis.points[0].neighbors,2);EXPECT_TRUE(t.analysis.points[0].surface);
}
