#include <gtest/gtest.h>
#include <dm/dm.hpp>
namespace n=dm::nn;namespace m=n::models;

TEST(NnNiceDensity, BananaKnownCoordinatesRoundOnceAndDetachStrides) {
  auto base=n::from_vector(std::vector<double>{0,0,2,1,-1,3},{3,2});base.requires_grad_();
  auto x=m::BananaPoints(base);
  EXPECT_EQ(x.to_vector<float>(),(std::vector<float>{0,-.5,2,2.5,-1,3}));
  EXPECT_FALSE(x.requires_grad());EXPECT_EQ(x.dtype(),n::DType::Float32);
  auto strided=n::transpose(n::transpose(base,{1,0}).contiguous(),{1,0});
  EXPECT_TRUE(n::equal_bits(x,m::BananaPoints(strided)));
  auto y=m::BananaPoints(base,-.5,n::DType::Float64);
  EXPECT_EQ(y.to_vector<double>(),(std::vector<double>{0,.5,2,-.5,-1,3}));
  auto logp=m::BananaLogProbability(base);
  auto g=n::grad(n::sum(logp),{base})[0];
  EXPECT_NEAR(g.to_vector<double>()[1],-.5,1e-14);
  EXPECT_NEAR(logp.to_vector<double>()[0],-std::log(2*std::acos(-1.))-.125,1e-14);
}
TEST(NnNiceDensity, GaussianConstantsScalarAndPerPointDeterminantsAndHigherPaths) {
  auto z=n::from_vector(std::vector<double>{1,2,-1,.5},{2,2});z.requires_grad_();
  auto ld=n::from_vector(std::vector<double>{.2,-.3},{2});ld.requires_grad_();
  auto o=m::NormalFlowDensity(z,ld);
  EXPECT_EQ(o.log_probability.shape(),(n::Shape{2}));EXPECT_EQ(o.loss.dim(),0);
  EXPECT_NEAR(o.loss.item(),std::log(2*std::acos(-1.))+.25*6.25+.05,1e-14);
  auto g=n::grad(o.loss,{z,ld},n::GradOptions{true,true,false});
  EXPECT_EQ(g[0].to_vector<double>(),(std::vector<double>{.5,1,-.5,.25}));
  EXPECT_EQ(g[1].to_vector<double>(),(std::vector<double>{-.5,-.5}));
  auto gg=n::grad(n::sum(g[0]),{z})[0];EXPECT_EQ(gg.to_vector<double>(),(std::vector<double>(4,.5)));
  auto legacy=m::StandardNormalLogProbability(n::zeros({1,2},n::DType::Float64),true);
  EXPECT_EQ(legacy.item(),-double(float(std::log(2*std::acos(-1.)))));
  EXPECT_NE(legacy.item(),m::StandardNormalLogProbability(n::zeros({1,2},n::DType::Float64)).item());
  auto scalar=m::NormalFlowDensity(z,n::full({},.2,n::DType::Float64));
  EXPECT_NEAR(scalar.loss.item(),std::log(2*std::acos(-1.))+.25*6.25-.2,1e-14);
}
TEST(NnNiceDensity, ExplicitCouplingTriangularJacobianInverseAndRawAxisOrder) {
  n::ParameterStore p(9301,n::DType::Float64);
  auto x=n::from_vector(std::vector<double>{.4,-.7,1.2,.3},{2,2});x.requires_grad_();
  for(int block=0;block<4;++block){
    auto y=m::NiceCoupling(p,x,block);int c=block%2;
    EXPECT_TRUE(n::equal_bits(n::slice_axis(x,1,c,c+1),n::slice_axis(y,1,c,c+1)));
    auto rec=m::NiceCoupling(p,y,block,true);
    EXPECT_LT(n::max(n::abs(rec-x)).item(),3e-16);
    auto j0=n::grad(n::sum(n::slice_axis(y,1,0,1)),{x},n::GradOptions{true,false,false})[0];
    auto j1=n::grad(n::sum(n::slice_axis(y,1,1,2)),{x},n::GradOptions{true,false,false})[0];
    auto a=j0.to_vector<double>(),b=j1.to_vector<double>();
    for(int row=0;row<2;++row)EXPECT_NEAR(a[row*2]*b[row*2+1]-a[row*2+1]*b[row*2],1,1e-14);
  }
  EXPECT_EQ(p.count(),196);EXPECT_EQ(p.size(),16);
}
TEST(NnNiceDensity, Full198CensusGlobalVolumeActualJacobianAndBothRoundtrips) {
  n::ParameterStore p(9301,n::DType::Float64);
  auto x=n::from_vector(std::vector<double>{.4,-.7,1.2,.3,-1,2},{3,2});x.requires_grad_();
  m::NiceDensity(p,x);p.set("log_scale",n::from_vector(std::vector<double>{.3,-.1},{2}));
  auto before=p.flatten();auto out=m::NiceDensity(p,x);
  EXPECT_EQ(p.count(),198);EXPECT_EQ(p.size(),17);
  EXPECT_TRUE(n::equal_bits(out.unscaled,out.coupling3));EXPECT_NEAR(out.logdet.item(),.2,1e-15);
  auto rec=m::NiceInverse(p,out.latent);EXPECT_LT(n::max(n::abs(rec.value-x)).item(),6e-16);
  EXPECT_EQ((rec.logdet+out.logdet).item(),0);
  auto z=n::from_vector(std::vector<double>{-.3,1.1,.2,-.8},{2,2});
  auto samples=m::NiceInverse(p,z);auto round=m::NiceDensity(p,samples.value);
  EXPECT_LT(n::max(n::abs(round.latent-z)).item(),6e-16);
  auto j0=n::grad(n::sum(n::slice_axis(out.latent,1,0,1)),{x},n::GradOptions{true,false,false})[0];
  auto j1=n::grad(n::sum(n::slice_axis(out.latent,1,1,2)),{x})[0];
  auto a=j0.to_vector<double>(),b=j1.to_vector<double>();
  for(int row=0;row<3;++row)EXPECT_NEAR(std::log(a[row*2]*b[row*2+1]-a[row*2+1]*b[row*2]),out.logdet.item(),2e-15);
  auto density=m::NormalFlowDensity(round.latent,round.logdet);
  auto identity=density.log_probability+samples.logdet-m::StandardNormalLogProbability(z);
  EXPECT_LT(n::max(n::abs(identity)).item(),1e-15);EXPECT_EQ(p.flatten(),before);
}
TEST(NnNiceDensity, MeanNllLeafGradientFiniteDifferencesAndLiveSecondDerivatives) {
  n::ParameterStore p(9301,n::DType::Float64);
  auto x=n::from_vector(std::vector<double>{.4,-.7,1.2,.3,-1,2},{3,2});x.requires_grad_();
  auto path=m::NiceDensity(p,x);auto loss=m::NormalFlowDensity(path.latent,path.logdet).loss;
  auto grads=n::grad(loss,p.parameters(),n::GradOptions{true,true,false});
  // Differentiate the retained graph before deliberate finite-difference writes
  // bump leaf versions; autograd correctly rejects use after those writes.
  auto higher=n::grad(n::sum(grads[0]),{x})[0];EXPECT_GT(n::max(n::abs(higher)).item(),1e-8);
  auto names=p.names();auto params=p.parameters();ASSERT_EQ(grads.size(),17);
  for(std::size_t leaf=0;leaf<names.size();++leaf){
    auto original=params[leaf].detach().clone();auto v=original.to_vector<double>();
    auto hi=v,lo=v;hi[0]+=1e-5;lo[0]-=1e-5;
    p.set(names[leaf],n::from_vector(hi,original.shape()));auto a=m::NiceDensity(p,x);
    double high=m::NormalFlowDensity(a.latent,a.logdet).loss.item();
    p.set(names[leaf],n::from_vector(lo,original.shape()));auto b=m::NiceDensity(p,x);
    double low=m::NormalFlowDensity(b.latent,b.logdet).loss.item();p.set(names[leaf],original);
    EXPECT_NEAR((high-low)/2e-5,grads[leaf].to_vector<double>()[0],2e-9)<<names[leaf];
  }
}
TEST(NnNiceDensity, StridedInputsOwningStageGraphsAndStoreImmutability) {
  n::Tensor saved,x=n::from_vector(std::vector<double>{.1,.2,.3,.4,.5,.6},{3,2});x.requires_grad_();
  {
    n::ParameterStore p(9301,n::DType::Float64);auto first=m::NiceDensity(p,x);auto before=p.flatten();
    auto strided=n::transpose(n::transpose(x,{1,0}).contiguous(),{1,0});
    auto next=m::NiceDensity(p,strided);EXPECT_TRUE(n::equal_bits(first.latent,next.latent));
    saved=next.coupling2;EXPECT_EQ(p.flatten(),before);
  }
  EXPECT_EQ(saved.shape(),(n::Shape{3,2}));
  auto dx=n::grad(n::sum(saved),{x})[0];EXPECT_GT(n::max(n::abs(dx)).item(),.9);
}
TEST(NnNiceDensity, RejectInvalidGeometryDtypesCurvatureAndNonfiniteBeforeLeaves) {
  n::ParameterStore p(9301);
  for(auto x:{n::Tensor(),n::zeros({2}),n::zeros({1,3}),n::zeros({0,2}),n::zeros({1,2},n::DType::Int32),n::full({1,2},NAN),n::zeros({1,2},n::DType::Float64)}){
    EXPECT_THROW(m::NiceDensity(p,x),std::invalid_argument);
    EXPECT_THROW(m::NiceInverse(p,x),std::invalid_argument);EXPECT_EQ(p.size(),0);
  }
  EXPECT_THROW(m::NiceCoupling(p,n::zeros({1,2}),-1),std::invalid_argument);
  EXPECT_THROW(m::NiceCoupling(p,n::zeros({1,2}),4),std::invalid_argument);EXPECT_EQ(p.size(),0);
  EXPECT_THROW(m::BananaPoints(n::zeros({1,2}),INFINITY),std::invalid_argument);
  EXPECT_THROW(m::BananaPoints(n::zeros({1,2}),.5,n::DType::Int32),std::invalid_argument);
  EXPECT_THROW(m::BananaLogProbability(n::zeros({1,2}),NAN),std::invalid_argument);
  EXPECT_THROW(m::NormalFlowDensity(n::zeros({2,2}),n::zeros({1})),std::invalid_argument);
  EXPECT_THROW(m::NormalFlowDensity(n::zeros({2,2}),n::zeros({2,1})),std::invalid_argument);
  EXPECT_THROW(m::NormalFlowDensity(n::zeros({2,2}),n::full({},NAN)),std::invalid_argument);
}
TEST(NnNiceDensity, StoredNonfiniteAndScaleOverflowOrUnderflowRejectWithoutWrites) {
  n::ParameterStore p(9301);auto x=n::zeros({2,2});m::NiceDensity(p,x);
  auto original=p.at("log_scale").detach().clone();
  for(double scale:{double(INFINITY),double(NAN),1000.,-1000.}){
    p.set("log_scale",n::full({2},scale));
    EXPECT_THROW(m::NiceDensity(p,x),std::invalid_argument);
    EXPECT_THROW(m::NiceInverse(p,x),std::invalid_argument);
    // NaNs need bit comparison, not vector floating equality.
    EXPECT_TRUE(n::equal_bits(p.at("log_scale"),n::full({2},scale)));
  }
  p.set("log_scale",original);p.set("coupling0/hidden/W",n::full({1,16},INFINITY));
  EXPECT_THROW(m::NiceDensity(p,x),std::invalid_argument);
  EXPECT_THROW(m::StandardNormalLogProbability(n::full({1,2},1e30)),std::invalid_argument);
  EXPECT_THROW(m::BananaPoints(n::full({1,2},1e30)),std::invalid_argument);
}
