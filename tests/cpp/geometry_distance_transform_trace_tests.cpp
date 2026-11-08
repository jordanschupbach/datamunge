#include <gtest/gtest.h>
#include <datamunge/geometry/distance_transform.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
using namespace datamunge::geometry;
namespace {
const double inf=std::numeric_limits<double>::infinity();
void scalar_oracle(const std::vector<double>& f,const SquaredDistanceTrace& t){
 const auto n=f.size();EXPECT_EQ(t.costs,f);EXPECT_EQ(t.values,squared_distance_transform_1d(f));EXPECT_EQ(t.values,detail::dt_1d(f));
 std::size_t finite=0;for(auto x:f)finite+=std::isfinite(x);EXPECT_EQ(t.work.finite_candidates,finite);EXPECT_EQ(t.work.envelope_pushes,finite);
 EXPECT_EQ(t.work.intersections,(finite?finite-1:0)+t.work.envelope_pops);EXPECT_EQ(t.work.evaluations,finite?n:0);EXPECT_LE(t.work.selection_advances,t.envelope_sites.empty()?0:t.envelope_sites.size()-1);
 EXPECT_EQ(t.work.input_values,n);EXPECT_EQ(t.work.lines,1u);EXPECT_EQ(t.comparisons.size(),t.work.intersections);
 for(const auto& c:t.comparisons){EXPECT_LT(c.previous,c.candidate);const long double p=c.previous,q=c.candidate;
  const auto cross=(f[c.candidate]+q*q-f[c.previous]-p*p)/(2*(q-p));EXPECT_NEAR(c.intersection,cross,1e-15);EXPECT_EQ(c.removed,c.intersection<=c.previous_begin);}
 for(std::size_t q=0;q<n;++q){long double best=std::numeric_limits<long double>::infinity();std::size_t arg=n;
  for(std::size_t p=0;p<n;++p)if(std::isfinite(f[p])){const long double delta=static_cast<long double>(q)-p,value=f[p]+delta*delta;if(value<best){best=value;arg=p;}}
  if(arg==n)EXPECT_TRUE(std::isinf(t.values[q]));else EXPECT_DOUBLE_EQ(t.values[q],static_cast<double>(best));EXPECT_EQ(t.nearest_index[q],arg);
 }
}
}
TEST(DistanceEnvelope, WorkedActualPopBoundariesAndTieAreHandCheckable){
 const auto t=squared_distance_transform_1d_trace({0,inf,9,inf,0});scalar_oracle({0,inf,9,inf,0},t);
 EXPECT_EQ(t.values,(std::vector<double>{0,1,4,1,0}));EXPECT_EQ(t.nearest_index,(std::vector<std::size_t>{0,0,0,4,4}));EXPECT_EQ(t.envelope_sites,(std::vector<std::size_t>{0,4}));
 ASSERT_EQ(t.comparisons.size(),3u);EXPECT_DOUBLE_EQ(t.comparisons[0].intersection,3.25);EXPECT_DOUBLE_EQ(t.comparisons[1].intersection,.75);EXPECT_TRUE(t.comparisons[1].removed);EXPECT_DOUBLE_EQ(t.comparisons[2].intersection,2);EXPECT_EQ(t.work.envelope_pops,1u);EXPECT_EQ(t.work.selection_advances,1u);
 ASSERT_EQ(t.breakpoints.size(),3u);EXPECT_EQ(t.breakpoints[0],-std::numeric_limits<long double>::infinity());EXPECT_EQ(t.breakpoints[1],2);EXPECT_EQ(t.breakpoints[2],std::numeric_limits<long double>::infinity());
}
TEST(DistanceEnvelope, EverySmallWeightedVectorMatchesIndependentQuadraticMinimum){
 const double alphabet[]{-2,0,3,inf};std::size_t total=1;
 for(std::size_t n=0;n<=8;++n){if(n)total*=4;for(std::size_t code=0;code<total;++code){std::vector<double> f(n);auto digits=code;
  for(auto& x:f){x=alphabet[digits%4];digits/=4;}scalar_oracle(f,squared_distance_transform_1d_trace(f));}}
}
TEST(DistanceField, ExhaustiveSmallMasksMatchBruteForceDistancesWitnessesAndColumnPass){
 for(std::size_t rows=1;rows<=4;++rows)for(std::size_t cols=1;cols<=4;++cols){const auto total=std::size_t{1}<<(rows*cols);
  for(std::size_t mask=0;mask<total;++mask){std::vector<std::vector<char>> grid(rows,std::vector<char>(cols));std::size_t sites=0,active_columns=0;
   for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<cols;++c){grid[r][c]=(mask>>(r*cols+c))&1;sites+=grid[r][c];}
   for(std::size_t c=0;c<cols;++c){bool any=false;for(std::size_t r=0;r<rows;++r)any|=grid[r][c]!=0;active_columns+=any;}
   const auto t=euclidean_distance_transform_trace(grid);const auto& f=t.result;EXPECT_EQ(euclidean_distance_transform(grid),f.distance);EXPECT_EQ(euclidean_distance_transform_features(grid).nearest_row,f.nearest_row);
   EXPECT_EQ(t.work.lines,rows+cols);EXPECT_EQ(t.work.input_values,2*rows*cols);EXPECT_EQ(t.work.finite_candidates,sites+rows*active_columns);
   const auto nonempty=sites?active_columns+rows:0;EXPECT_EQ(t.work.intersections,t.work.finite_candidates-nonempty+t.work.envelope_pops);
   for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<cols;++c){double best=inf,column_best=inf;std::size_t br=rows,bc=cols,cr=rows;
    for(std::size_t sc=0;sc<cols;++sc)for(std::size_t sr=0;sr<rows;++sr)if(grid[sr][sc]){const double dr=static_cast<double>(r)-sr,dc=static_cast<double>(c)-sc,value=dr*dr+dc*dc;
     if(value<best){best=value;br=sr;bc=sc;}if(sc==c && dr*dr<column_best){column_best=dr*dr;cr=sr;}}
    EXPECT_DOUBLE_EQ(f.squared_distance[r][c],best);EXPECT_DOUBLE_EQ(f.distance[r][c],std::sqrt(best));EXPECT_EQ(f.nearest_row[r][c],br);EXPECT_EQ(f.nearest_column[r][c],bc);EXPECT_DOUBLE_EQ(t.column_squared[r][c],column_best);EXPECT_EQ(t.column_nearest_row[r][c],cr);
   }
  }
 }
}
TEST(DistanceField, OriginalThreeSitesProfileAndColumnFirstTieConvention){
 const auto grid=distance_site_grid(9,15,{{2,3},{6,11},{4,7}});const auto f=euclidean_distance_transform_features(grid);
 EXPECT_DOUBLE_EQ(f.distance[4][7],0);EXPECT_DOUBLE_EQ(f.squared_distance[0][0],13);EXPECT_DOUBLE_EQ(f.squared_distance[8][14],13);EXPECT_DOUBLE_EQ(f.distance[4][3],2);EXPECT_DOUBLE_EQ(f.distance[4][11],2);
 const auto tie=euclidean_distance_transform_features(distance_site_grid(3,3,{{0,2},{2,0}}));EXPECT_DOUBLE_EQ(tie.squared_distance[1][1],2);EXPECT_EQ(tie.nearest_row[1][1],2u);EXPECT_EQ(tie.nearest_column[1][1],0u);
 EXPECT_DOUBLE_EQ(euclidean_distance_transform({{static_cast<char>(-7),0},{0,2}})[1][1],0);
}
TEST(DistanceField, EmptyZeroWidthNoSiteAndRaggedShapesAreExplicit){
 EXPECT_TRUE(euclidean_distance_transform({}).empty());const auto empty=euclidean_distance_transform_trace({{}, {}});EXPECT_EQ(empty.result.distance.size(),2u);EXPECT_TRUE(empty.result.distance[0].empty());EXPECT_EQ(empty.result.nearest_row.size(),2u);EXPECT_EQ(empty.work.lines,0u);
 const auto f=euclidean_distance_transform_features({{0,0,0},{0,0,0}});EXPECT_EQ(f.nearest_row[1][1],2u);EXPECT_EQ(f.nearest_column[1][1],3u);EXPECT_TRUE(std::isinf(f.distance[1][1]));
 EXPECT_THROW((void)euclidean_distance_transform({{1,0},{}}),std::invalid_argument);
 EXPECT_THROW((void)euclidean_distance_transform({{},{1}}),std::invalid_argument);
 EXPECT_TRUE(squared_distance_transform_1d({}).empty());const auto t=squared_distance_transform_1d_trace({inf,inf});EXPECT_TRUE(t.envelope_sites.empty());EXPECT_EQ(t.nearest_index,(std::vector<std::size_t>{2,2}));
 EXPECT_EQ(squared_distance_transform_1d({-4,inf,0}),(std::vector<double>{-4,-3,0}));
}
TEST(DistanceEnvelope, LongIndicesAvoidSignedProductsAndHistoryCapsPrecedeAllocation){
 const auto dense=squared_distance_transform_1d_trace(std::vector<double>(50001,0));
 for(std::size_t q=0;q<dense.values.size();++q){EXPECT_DOUBLE_EQ(dense.values[q],0);EXPECT_EQ(dense.nearest_index[q],q);}
 std::vector<double> f(100001,inf);f[0]=0;const auto t=squared_distance_transform_1d_trace(f);EXPECT_DOUBLE_EQ(t.values.back(),10000000000.);EXPECT_EQ(t.nearest_index.back(),0u);EXPECT_EQ(t.work.intersections,0u);
 const auto grid=distance_site_grid(1,100001,{{0,0}});const auto field=euclidean_distance_transform_features(grid);EXPECT_DOUBLE_EQ(field.distance[0].back(),100000);EXPECT_DOUBLE_EQ(field.squared_distance[0].back(),10000000000.);
 EXPECT_THROW((void)squared_distance_transform_1d_trace({0,1,2},2),std::length_error);
 EXPECT_NO_THROW((void)squared_distance_transform_1d_trace({0,1,2},3));
 EXPECT_NO_THROW((void)squared_distance_transform_1d_trace({0,1,2},0));
 EXPECT_THROW((void)euclidean_distance_transform_trace({{1,0},{0,0}},3),std::length_error);
 EXPECT_NO_THROW((void)euclidean_distance_transform_trace({{1,0},{0,0}},4));
 EXPECT_NO_THROW((void)euclidean_distance_transform_trace({{1,0},{0,0}},0));
}
TEST(DistanceField, DenseAndSingleSiteSweepCountsMatchClosedForms){
 for(std::size_t r:{1u,3u,16u,64u})for(std::size_t c:{1u,5u,32u}){
  auto g=distance_site_grid(r,c,{{r/2,c/2}});const auto a=euclidean_distance_transform_trace(g);const auto n=r*c;
  EXPECT_EQ(a.work.input_values,2*n);EXPECT_EQ(a.work.finite_candidates,r+1);EXPECT_EQ(a.work.intersections,0u);EXPECT_EQ(a.work.evaluations,n+r);
  for(auto& row:g)std::fill(row.begin(),row.end(),1);const auto b=euclidean_distance_transform_trace(g);
  EXPECT_EQ(b.work.finite_candidates,2*n);EXPECT_EQ(b.work.intersections,2*n-r-c);EXPECT_EQ(b.work.envelope_pops,0u);EXPECT_EQ(b.work.selection_advances,2*n-r-c);EXPECT_EQ(b.work.evaluations,2*n);
 }
}
TEST(DistanceField, RejectsInvalidCostsSitesAndCountsAndOwnsAllResults){
 for(double bad:{std::numeric_limits<double>::quiet_NaN(),-inf}){
  EXPECT_THROW((void)squared_distance_transform_1d({0,bad}),std::invalid_argument);
 }
 EXPECT_THROW((void)distance_site_grid(2,2,{{2,0}}),std::invalid_argument);
 EXPECT_THROW((void)distance_site_grid(0,0,{{0,0}}),std::invalid_argument);
 EXPECT_THROW((void)distance_site_grid(2,2,{},3),std::length_error);
 EXPECT_THROW((void)distance_site_grid(std::numeric_limits<std::size_t>::max(),2,{},0),std::length_error);
 const auto duplicate=distance_site_grid(2,2,{{0,0},{0,0}});EXPECT_EQ(duplicate[0][0],1);
 auto f=std::vector<double>{0,inf,4};const auto t=squared_distance_transform_1d_trace(f);f[0]=99;EXPECT_DOUBLE_EQ(t.costs[0],0);auto copy=t;copy.values[0]=99;EXPECT_DOUBLE_EQ(t.values[0],0);
}
TEST(DistanceFieldPlot, SupportsFiniteAndInfiniteCellsAndRejectsMalformedOrExcessiveFields){
 auto f=euclidean_distance_transform_features({{1,0},{0,0}});EXPECT_GT(plot_distance_field(f,true).series().size(),0u);
 EuclideanDistanceField grey;grey.distance={{inf}};EXPECT_GT(plot_distance_field(grey).series().size(),0u);
 EXPECT_THROW((void)plot_distance_field(f,false,3),std::length_error);
 EXPECT_NO_THROW((void)plot_distance_field(f,false,4));
 EXPECT_THROW((void)plot_distance_field(EuclideanDistanceField{}),std::invalid_argument);
 f.distance={{-1}};
 EXPECT_THROW((void)plot_distance_field(f),std::invalid_argument);
 f.distance={{std::numeric_limits<double>::quiet_NaN()}};
 EXPECT_THROW((void)plot_distance_field(f),std::invalid_argument);
 f.distance={{1},{}};
 EXPECT_THROW((void)plot_distance_field(f),std::invalid_argument);
}
