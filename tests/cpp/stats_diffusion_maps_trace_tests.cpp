#include <gtest/gtest.h>
#include <datamunge/stats/diffusion_maps.hpp>
#include <datamunge/stats/manifold_diagnostics.hpp>
#include <datamunge/datasets/generators.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
using namespace datamunge;
namespace {
dstruct::DataFrame frame(const std::vector<double>& values){dstruct::DataFrame d;d.add_column("x",values);return d;}
using Matrix=linalg::DenseMatrix<double>;
Matrix product(const Matrix& a,const Matrix& b){Matrix c(a.rows(),b.cols(),0);for(std::size_t i=0;i<a.rows();++i)for(std::size_t j=0;j<b.cols();++j)for(std::size_t k=0;k<a.cols();++k)c(i,j)+=a(i,k)*b(k,j);return c;}
}
TEST(DiffusionTrace, ThreePointsHaveHandDerivedKernelTransitionsAndSpectrum){
 const auto t=stats::diffusion_maps_trace(frame({0,1,2}),{"x"},{1,1/std::log(2.),0,1});
 EXPECT_NEAR(t.kernel(0,1),.5,1e-15);EXPECT_NEAR(t.kernel(0,2),1./16,1e-15);
 EXPECT_NEAR(t.degrees[0],25./16,1e-15);EXPECT_NEAR(t.degrees[1],2,1e-15);
 EXPECT_NEAR(t.transition(0,0),16./25,1e-15);EXPECT_NEAR(t.transition(0,1),8./25,1e-15);
 EXPECT_NEAR(t.transition(0,2),1./25,1e-15);EXPECT_NEAR(t.transition(1,0),.25,1e-15);
 EXPECT_NEAR(t.eigenvalues[0],1,1e-14);EXPECT_NEAR(t.eigenvalues[1],.6,1e-14);EXPECT_NEAR(t.eigenvalues[2],.18,1e-14);
 EXPECT_NEAR(t.stationary[0],25./82,1e-15);EXPECT_NEAR(t.stationary[1],32./82,1e-15);
 const double delta=t.embedding(0,0)-t.embedding(2,0);EXPECT_NEAR(delta*delta*82./16,2.3616,1e-13);
}
TEST(DiffusionTrace, IndependentMatrixPowersMatchCompleteSpectralDiffusionDistances){
 std::mt19937 rng(91);std::uniform_real_distribution<double> u(-2,2);
 for(std::size_t n:{3u,5u,8u})for(int trial=0;trial<12;++trial)for(double alpha:{0.,.5,1.}){
  std::vector<double>x(n);for(auto& v:x)v=u(rng);
  const auto t=stats::diffusion_maps_trace(frame(x),{"x"},{1,.75,alpha,1});
  Matrix power(n,n,0);for(std::size_t i=0;i<n;++i)power(i,i)=1;
  double total=0;for(auto d:t.degrees)total+=d;
  for(unsigned time=0;time<4;++time){
   for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j){
    double direct=0,spectral=0;
    for(std::size_t k=0;k<n;++k){double d=power(i,k)-power(j,k);direct+=d*d/t.stationary[k];
     d=t.right_eigenvectors(i,k)-t.right_eigenvectors(j,k);spectral+=std::pow(t.eigenvalues[k],2*time)*d*d*total;}
    EXPECT_NEAR(direct,spectral,3e-9);
   }
   power=product(power,t.transition);
  }
 }
}
TEST(DiffusionTrace, NormalizationReversibilityEigenResidualsAndTimeFiltering){
 for(double alpha:{0.,.5,1.}){
  auto d=frame({0,.1,.5,1.2,1.8,2.4});
  const auto t=stats::diffusion_maps_trace(d,{"x"},{3,1.3,alpha,0});
  const auto u=stats::diffusion_maps_trace(d,{"x"},{3,1.3,alpha,2.5});
  const stats::DiffusionMaps m(d,{"x"},{3,1.3,alpha,2.5});
  for(std::size_t i=0;i<6;++i){double row=0;
   for(std::size_t j=0;j<6;++j){row+=t.transition(i,j);
    EXPECT_NEAR(t.kernel(i,j),std::exp(-std::pow(d.double_at("x",i)-d.double_at("x",j),2)/1.3),1e-15);
    EXPECT_NEAR(t.normalized_kernel(i,j),t.kernel(i,j)/std::pow(t.kernel_degrees[i]*t.kernel_degrees[j],alpha),1e-14);
    EXPECT_NEAR(t.stationary[i]*t.transition(i,j),t.stationary[j]*t.transition(j,i),1e-15);
   }EXPECT_NEAR(row,1,1e-14);
   for(std::size_t c=0;c<6;++c){double ppsi=0;for(std::size_t j=0;j<6;++j)ppsi+=t.transition(i,j)*t.right_eigenvectors(j,c);
    EXPECT_NEAR(ppsi,t.eigenvalues[c]*t.right_eigenvectors(i,c),2e-13);}
   for(std::size_t c=0;c<3;++c){EXPECT_NEAR(u.embedding(i,c),t.embedding(i,c)*std::pow(t.eigenvalues[c+1],2.5),1e-14);EXPECT_DOUBLE_EQ(u.embedding(i,c),m.embedding()(i,c));}
  }
  for(std::size_t a=0;a<6;++a)for(std::size_t b=0;b<6;++b){double norm=0;for(std::size_t i=0;i<6;++i)norm+=t.degrees[i]*t.right_eigenvectors(i,a)*t.right_eigenvectors(i,b);EXPECT_NEAR(norm,a==b?1:0,1e-14);}
 }
}
TEST(DiffusionTrace, ValidationCapsDisconnectedKernelsAndFiniteDegenerateData){
 const auto d=frame({0,1,2,3});const auto nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
 for(double bad:{nan,inf,-inf,0.,-1.}){
  EXPECT_THROW(stats::diffusion_maps_trace(d,{"x"},{2,bad,.5,1}),std::invalid_argument);
 }
 for(double bad:{nan,inf,-.1,1.1}){
  EXPECT_THROW(stats::diffusion_maps_trace(d,{"x"},{2,1,bad,1}),std::invalid_argument);
 }
 for(double bad:{nan,inf,-1.}){
  EXPECT_THROW(stats::diffusion_maps_trace(d,{"x"},{2,1,.5,bad}),std::invalid_argument);
 }
 EXPECT_THROW(stats::diffusion_maps_trace(frame({0,inf,1}),{"x"},{1,1,0,1}),std::invalid_argument);
 EXPECT_THROW(stats::diffusion_maps_trace(d,{"x"},{2,1,.5,1},3),std::length_error);
 EXPECT_NO_THROW(stats::diffusion_maps_trace(d,{"x"},{2,1,.5,1},4));
 EXPECT_NO_THROW(stats::diffusion_maps_trace(d,{"x"},{2,1,.5,1},0));
 EXPECT_THROW(stats::diffusion_maps_trace(d,{"x"},{2,1e-300,0,1}),std::runtime_error);
 EXPECT_THROW(stats::diffusion_maps_trace(frame({-1e308,0,1e308}),{"x"},{1,1,0,1}),std::overflow_error);
 const auto t=stats::diffusion_maps_trace(frame({1,1,1,1}),{"x"},{2,1,.5,0});
 for(double v:t.eigenvalues)EXPECT_GE(v,0);
 for(std::size_t i=0;i<4;++i)for(std::size_t c=0;c<2;++c)EXPECT_TRUE(std::isfinite(t.embedding(i,c)));
}
TEST(DiffusionTrace, NullMappingAndOwnedResults){
 auto d=frame({0,1,2,3,4});d.set_null("x",1);auto t=stats::diffusion_maps_trace(d,{"x"});
 EXPECT_EQ(t.kept_row_indices,(std::vector<std::size_t>{0,2,3,4}));
 const auto saved=t.kernel(0,1);d=frame({90,91,92});EXPECT_DOUBLE_EQ(t.kernel(0,1),saved);
 auto copy=t;copy.kernel(0,1)=99;EXPECT_DOUBLE_EQ(t.kernel(0,1),saved);
}
TEST(ManifoldDiagnostics, DistanceScalesMatchSortedIntegerOracle){
 for(std::size_t n=2;n<=19;++n){Matrix x(n,2);std::vector<double>all;
  for(std::size_t i=0;i<n;++i){x(i,0)=i%5;x(i,1)=(i*i)%7;}
  for(std::size_t i=0;i<n;++i)for(std::size_t j=i+1;j<n;++j)all.push_back(std::pow(x(i,0)-x(j,0),2)+std::pow(x(i,1)-x(j,1),2));
  std::sort(all.begin(),all.end());double sum=0;for(auto v:all)sum+=v;
  for(std::size_t k=1;k<n;++k){std::vector<double>local;
   for(std::size_t i=0;i<n;++i){std::vector<double>row;for(std::size_t j=0;j<n;++j)if(i!=j)row.push_back(std::pow(x(i,0)-x(j,0),2)+std::pow(x(i,1)-x(j,1),2));std::sort(row.begin(),row.end());local.push_back(row[k-1]);}
   std::sort(local.begin(),local.end());const auto s=stats::manifold_distance_scales(x,k);
   EXPECT_DOUBLE_EQ(s.upper_median_squared,all[all.size()/2]);EXPECT_NEAR(s.mean_squared,sum/all.size(),1e-14);EXPECT_DOUBLE_EQ(s.upper_median_neighbor_squared,local[n/2]);
  }
 }
}
TEST(ManifoldDiagnostics, SeparationHasScatterIdentityAndSimilarityInvariance){
 Matrix x(4,2,{0,0,2,0,4,0,6,0}),y(4,2);std::vector<std::string>labels{"a","a","b","b"};
 const auto s=stats::embedding_separation(x,labels);EXPECT_DOUBLE_EQ(s.within,4);EXPECT_DOUBLE_EQ(s.between,16);EXPECT_DOUBLE_EQ(s.ratio,4);EXPECT_DOUBLE_EQ(s.within+s.between,20);
 for(std::size_t i=0;i<4;++i){y(i,0)=7-3*x(i,1);y(i,1)=-2+3*x(i,0);}
 const auto r=stats::embedding_separation(y,labels);EXPECT_DOUBLE_EQ(r.ratio,s.ratio);EXPECT_DOUBLE_EQ(r.within,9*s.within);EXPECT_DOUBLE_EQ(r.between,9*s.between);
 EXPECT_TRUE(std::isinf(stats::embedding_separation(Matrix(2,1,{0,1}),{"a","b"}).ratio));
 EXPECT_TRUE(std::isnan(stats::embedding_separation(Matrix(2,1,{0,0}),{"a","b"}).ratio));
 EXPECT_DOUBLE_EQ(stats::embedding_separation(x,{"a","a","a","a"}).between,0);
}
TEST(ManifoldDiagnostics, RejectsInvalidShapesRanksCoordinatesAndOverflow){
 const Matrix x(3,1,{0,1,2});
 EXPECT_THROW(stats::manifold_distance_scales(x,0),std::invalid_argument);
 EXPECT_THROW(stats::manifold_distance_scales(x,3),std::invalid_argument);
 EXPECT_THROW(stats::embedding_separation(x,{"a"}),std::invalid_argument);
 EXPECT_THROW(stats::embedding_separation(Matrix(0,1),{}),std::invalid_argument);
 EXPECT_THROW(stats::manifold_distance_scales(Matrix(2,1,{0,std::numeric_limits<double>::infinity()}),1),std::invalid_argument);
 EXPECT_THROW(stats::manifold_distance_scales(Matrix(2,1,{-1e308,1e308}),1),std::overflow_error);
 EXPECT_THROW(stats::embedding_separation(Matrix(2,1,{-1e308,1e308}),{"a","a"}),std::overflow_error);
}
TEST(SwissRollEngine, RetainsOriginalMtDrawsAndExistingXoshiroOutput){
 const auto roll=datasets::swiss_roll(400,0,42,datasets::SwissRollEngine::Mt19937);
 std::mt19937_64 rng(42);std::uniform_real_distribution<double> u(0,1);
 for(std::size_t i=0;i<400;++i){const auto t=1.5*std::numbers::pi*(1+2*u(rng)),h=21*u(rng);
  EXPECT_DOUBLE_EQ(roll.double_at("t",i),t);EXPECT_DOUBLE_EQ(roll.double_at("height",i),h);
  EXPECT_DOUBLE_EQ(roll.double_at("x",i),t*std::cos(t));EXPECT_DOUBLE_EQ(roll.double_at("y",i),h);EXPECT_DOUBLE_EQ(roll.double_at("z",i),t*std::sin(t));
  EXPECT_NEAR(roll.double_at("arc_length",i),.5*(t*std::sqrt(1+t*t)+std::asinh(t)),2e-14);
 }
 const auto old=datasets::swiss_roll(40,.1,42),same=datasets::swiss_roll(40,.1,42,datasets::SwissRollEngine::Xoshiro);
 for(std::size_t i=0;i<40;++i)for(auto col:{"x","y","z","t","height","arc_length"})EXPECT_DOUBLE_EQ(old.double_at(col,i),same.double_at(col,i));
 EXPECT_EQ(datasets::swiss_roll(0,0,42,datasets::SwissRollEngine::Mt19937).nrows(),0u);
 EXPECT_THROW(datasets::swiss_roll(2,0,0,static_cast<datasets::SwissRollEngine>(9)),std::invalid_argument);
 EXPECT_THROW(datasets::swiss_roll(2,std::numeric_limits<double>::infinity(),0,datasets::SwissRollEngine::Mt19937),std::invalid_argument);
}
