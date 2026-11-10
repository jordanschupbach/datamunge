#include <gtest/gtest.h>
#include <dm/dm.hpp>
namespace n=dm::nn;namespace m=n::models;
TEST(NnAffineFlows, ExplicitTriangularCouplingsMatchAnalyticValuesSlopesAndDeterminants) {
 auto type=n::DType::Float64;n::ParameterStore p(9403,type);
 auto x=n::from_vector(std::vector<double>{-.4,1.7,.8,-.5},{2,2});x.requires_grad_();
 for(int block=0;block<4;++block){
  m::RealNvpCoupling(p,x,block);auto prefix="coupling"+std::to_string(block);
  std::vector<double> w(16),b(16),heads(32);w[0]=.7;b[0]=.1;heads[0]=.2;heads[1]=.3;
  p.set(prefix+"/hidden/W",n::from_vector(w,{1,16}));p.set(prefix+"/hidden/b",n::from_vector(b,{16}));
  p.set(prefix+"/heads/W",n::from_vector(heads,{16,2}));p.set(prefix+"/heads/b",n::from_vector(std::vector<double>{.05,-.1},{2}));
  auto out=m::RealNvpCoupling(p,x,block);int c=block%2;auto input=x.to_vector<double>(),value=out.value.to_vector<double>();
  auto j0=n::grad(n::sum(n::slice_axis(out.value,1,0,1)),{x},n::GradOptions{true,false,false})[0].to_vector<double>();
  auto j1=n::grad(n::sum(n::slice_axis(out.value,1,1,2)),{x},n::GradOptions{true,false,false})[0].to_vector<double>();
  auto dld=n::grad(n::sum(out.logdet),{x})[0].to_vector<double>();
  for(int row=0;row<2;++row){
   const double a=input[row*2+c],other=input[row*2+1-c],h=std::tanh(.7*a+.1),raw=.2*h+.05,s=.8*std::tanh(raw),t=.3*h-.1;
   const double hs=.7*(1-h*h),ds=.8*(1-std::pow(std::tanh(raw),2))*.2*hs,dt=.3*hs;
   EXPECT_EQ(value[row*2+c],a);EXPECT_NEAR(value[row*2+1-c],other*std::exp(s)+t,2e-15);
   EXPECT_NEAR(out.logdet.to_vector<double>()[row],s,2e-15);
   EXPECT_NEAR(dld[row*2+c],ds,2e-15);EXPECT_EQ(dld[row*2+1-c],0);
   const auto& changed=c?j0:j1;EXPECT_NEAR(changed[row*2+c],other*std::exp(s)*ds+dt,2e-15);
   EXPECT_NEAR(std::log(j0[row*2]*j1[row*2+1]-j0[row*2+1]*j1[row*2]),s,2e-15);
  }
  auto inverse=m::RealNvpCoupling(p,out.value,block,true);
  EXPECT_LT(n::max(n::abs(inverse.value-x)).item(),1e-15);
  EXPECT_EQ(n::max(n::abs(inverse.logdet+out.logdet)).item(),0);
 }
 EXPECT_EQ(p.count(),264);EXPECT_EQ(p.size(),16);
}
TEST(NnAffineFlows, GlobalDiagonalScalePreservesScalarAndPerRowDeterminantsAndHigherPaths) {
 auto type=n::DType::Float64;n::ParameterStore p(1,type);
 p.set("log_scale",n::from_vector(std::vector<double>{std::log(2.),std::log(.5),.2},{3}));
 auto x=n::ones({2,3},type).requires_grad_(),ld=n::full({2},.3,type).requires_grad_();
 auto out=m::FlowDiagonalScale(p,x,ld);auto before=p.flatten();
 EXPECT_EQ(out.logdet.shape(),(n::Shape{2}));EXPECT_NEAR(out.logdet.to_vector<double>()[0],.5,1e-15);
 EXPECT_NEAR(out.value.to_vector<double>()[0],2.,1e-15);EXPECT_NEAR(out.value.to_vector<double>()[1],.5,1e-15);
 auto inv=m::FlowDiagonalScale(p,out.value,out.logdet,true);
 EXPECT_LT(n::max(n::abs(inv.value-x)).item(),3e-16);EXPECT_LT(n::max(n::abs(inv.logdet-ld)).item(),1e-16);
 auto scalar=m::FlowDiagonalScale(p,x,n::full({},.3,type));EXPECT_EQ(scalar.logdet.dim(),0);
 auto g=n::grad(n::sum(out.value)+n::sum(out.logdet),p.parameters(),n::GradOptions{true,true,false})[0];
 auto higher=n::grad(n::sum(g),{x})[0];EXPECT_GT(n::max(n::abs(higher)).item(),1.9);
 EXPECT_EQ(p.flatten(),before);EXPECT_EQ(p.count(),3);
}
TEST(NnAffineFlows, Full266CensusAllStageFieldsAndBothInversePathsUseTheirOwnPoints) {
 n::ParameterStore p(9403,n::DType::Float64);auto x=n::from_vector(std::vector<double>{.4,-.7,1.2,.3,-1,2},{3,2});x.requires_grad_();
 auto out=m::RealNvpDensity(p,x);auto before=p.flatten();EXPECT_EQ(p.count(),266);EXPECT_EQ(p.size(),17);
 EXPECT_TRUE(n::equal_bits(out.unscaled,out.coupling3));EXPECT_EQ(out.logdet.shape(),(n::Shape{3}));
 for(auto stage:{out.coupling_logdet0,out.coupling_logdet1,out.coupling_logdet2,out.coupling_logdet3})EXPECT_EQ(stage.shape(),(n::Shape{3}));
 for(auto stage:{out.coupling_scale0,out.coupling_scale1,out.coupling_scale2,out.coupling_scale3})EXPECT_LE(n::max(n::abs(stage)).item(),.8);
 auto rec=m::RealNvpInverse(p,out.latent);EXPECT_LT(n::max(n::abs(rec.value-x)).item(),9e-16);
 EXPECT_LT(n::max(n::abs(rec.logdet+out.logdet)).item(),3e-17);
 auto z=n::from_vector(std::vector<double>{-.3,1.1,.2,-.8},{2,2});auto samples=m::RealNvpInverse(p,z);auto round=m::RealNvpDensity(p,samples.value);
 EXPECT_LT(n::max(n::abs(round.latent-z)).item(),9e-16);EXPECT_LT(n::max(n::abs(round.logdet+samples.logdet)).item(),5e-17);
 EXPECT_GT((n::max(out.logdet)-n::min(out.logdet)).item(),1e-5);
 auto identity=m::NormalFlowDensity(round.latent,round.logdet).log_probability+samples.logdet-m::StandardNormalLogProbability(z);
 EXPECT_LT(n::max(n::abs(identity)).item(),1e-15);EXPECT_EQ(p.flatten(),before);
}
TEST(NnAffineFlows, ActualForwardAndInverseJacobianDeterminantsMatchAllPerPointLogdets) {
 n::ParameterStore p(9403,n::DType::Float64);auto x=n::from_vector(std::vector<double>{.4,-.7,1.2,.3,-1,2},{3,2}).requires_grad_();
 auto out=m::RealNvpDensity(p,x);auto z=n::from_vector(std::vector<double>{.8,-1.2,-2,.9},{2,2}).requires_grad_();auto inv=m::RealNvpInverse(p,z);
 for(auto entry:{std::make_tuple(out.latent,out.logdet,x),std::make_tuple(inv.value,inv.logdet,z)}){
  auto value=std::get<0>(entry),ld=std::get<1>(entry),input=std::get<2>(entry);
  auto a=n::grad(n::sum(n::slice_axis(value,1,0,1)),{input},n::GradOptions{true,false,false})[0].to_vector<double>();
  auto b=n::grad(n::sum(n::slice_axis(value,1,1,2)),{input})[0].to_vector<double>();
  for(int row=0;row<input.size(0);++row){auto det=a[row*2]*b[row*2+1]-a[row*2+1]*b[row*2];EXPECT_GT(det,0);EXPECT_NEAR(std::log(det),ld.to_vector<double>()[row],7e-16);}
 }
}
TEST(NnAffineFlows, MeanNllPartialsMatchSeventeenLeafFiniteDifferencesAndRetainHigherPaths) {
 n::ParameterStore p(9403,n::DType::Float64);auto x=n::from_vector(std::vector<double>{.4,-.7,1.2,.3,-1,2},{3,2}).requires_grad_();
 auto path=m::RealNvpDensity(p,x);auto loss=m::NormalFlowDensity(path.latent,path.logdet).loss;
 auto gradients=n::grad(loss,p.parameters(),n::GradOptions{true,true,false});
 auto higher=n::grad(n::sum(gradients[0]),{x})[0];EXPECT_GT(n::max(n::abs(higher)).item(),1e-8);
 auto names=p.names();auto parameters=p.parameters();ASSERT_EQ(names.size(),17);
 for(std::size_t i=0;i<names.size();++i){
  auto original=parameters[i].detach().clone();auto hi=original.to_vector<double>(),lo=hi;hi[0]+=1e-5;lo[0]-=1e-5;
  p.set(names[i],n::from_vector(hi,original.shape()));auto a=m::RealNvpDensity(p,x);double high=m::NormalFlowDensity(a.latent,a.logdet).loss.item();
  p.set(names[i],n::from_vector(lo,original.shape()));auto b=m::RealNvpDensity(p,x);double low=m::NormalFlowDensity(b.latent,b.logdet).loss.item();p.set(names[i],original);
  EXPECT_NEAR((high-low)/2e-5,gradients[i].to_vector<double>()[0],2e-9)<<names[i];
 }
}
TEST(NnAffineFlows, StridedInputsAndOwningStageGraphsOutliveParentsAndStore) {
 auto x=n::from_vector(std::vector<double>{.1,.2,.3,.4,.5,.6},{3,2}).requires_grad_();n::Tensor stage,ld;
 {n::ParameterStore p(9403,n::DType::Float64);auto first=m::RealNvpDensity(p,x);auto before=p.flatten();
  auto strided=n::transpose(n::transpose(x,{1,0}).contiguous(),{1,0});auto second=m::RealNvpDensity(p,strided);
  EXPECT_TRUE(n::equal_bits(first.latent,second.latent));EXPECT_TRUE(n::equal_bits(first.logdet,second.logdet));
  stage=second.coupling2;ld=second.coupling_logdet1;EXPECT_EQ(p.flatten(),before);}
 EXPECT_GT(n::max(n::abs(n::grad(n::sum(stage),{x},n::GradOptions{true,false,false})[0])).item(),.9);
 EXPECT_GT(n::max(n::abs(n::grad(n::sum(ld),{x})[0])).item(),1e-5);
}
TEST(NnAffineFlows, InvalidMatricesBlocksAndIncomingDeterminantsRejectBeforeLeafCreation) {
 n::ParameterStore p(9403);
 for(auto x:{n::Tensor(),n::zeros({2}),n::zeros({1,3}),n::zeros({0,2}),n::zeros({1,2},n::DType::Int32),n::full({1,2},NAN),n::zeros({1,2},n::DType::Float64)}){
  EXPECT_THROW(m::RealNvpDensity(p,x),std::invalid_argument);
  EXPECT_THROW(m::RealNvpInverse(p,x),std::invalid_argument);EXPECT_EQ(p.size(),0);}
 for(int block:{-1,4})EXPECT_THROW(m::RealNvpCoupling(p,n::zeros({1,2}),block),std::invalid_argument);
 for(auto ld:{n::Tensor(),n::zeros({1}),n::zeros({2,1}),n::full({},NAN),n::zeros({2},n::DType::Float64)}){
  EXPECT_THROW(m::FlowDiagonalScale(p,n::zeros({2,2}),ld),std::invalid_argument);EXPECT_EQ(p.size(),0);}
}
TEST(NnAffineFlows, NonfiniteStoredValuesScalesAndComputedAffineOverflowReject) {
 n::ParameterStore p(9403);auto x=n::zeros({2,2});m::RealNvpDensity(p,x);
 for(double scale:{double(INFINITY),double(NAN),1000.,-1000.}){
  p.set("log_scale",n::full({2},scale));EXPECT_THROW(m::RealNvpDensity(p,x),std::invalid_argument);
  EXPECT_THROW(m::RealNvpInverse(p,x),std::invalid_argument);
  EXPECT_TRUE(n::equal_bits(p.at("log_scale"),n::full({2},scale)));}
 p.set("log_scale",n::zeros({2}));p.set("coupling0/hidden/W",n::full({1,16},INFINITY));
 EXPECT_THROW(m::RealNvpDensity(p,x),std::invalid_argument);
 n::ParameterStore q(9403);m::RealNvpCoupling(q,x,0);
 q.set("coupling0/hidden/W",n::zeros({1,16}));q.set("coupling0/hidden/b",n::zeros({16}));
 q.set("coupling0/heads/W",n::zeros({16,2}));q.set("coupling0/heads/b",n::from_vector(std::vector<float>{10,0},{2}));
 auto huge=n::from_vector(std::vector<float>{0,std::numeric_limits<float>::max()},{1,2});
 EXPECT_THROW(m::RealNvpCoupling(q,huge,0),std::invalid_argument);
}
TEST(NnAffineFlows, SourceFloat32BoundsAndRepeatInferenceRemainImmutable) {
 n::NoGradGuard guard;n::ParameterStore p(9403);auto x=n::models::BananaPoints(n::models::GaussianNoise({2},128,9221));
 auto a=m::RealNvpDensity(p,x);auto before=p.flatten();auto b=m::RealNvpDensity(p,x);
 EXPECT_TRUE(n::equal_bits(a.latent,b.latent));EXPECT_TRUE(n::equal_bits(a.logdet,b.logdet));
 EXPECT_EQ(a.logdet.dtype(),n::DType::Float32);
 auto rec=m::RealNvpInverse(p,a.latent);EXPECT_LT(n::max(n::abs(rec.value-x)).item(),5e-6);
 EXPECT_LT(n::max(n::abs(rec.logdet+a.logdet)).item(),5e-6);EXPECT_EQ(p.flatten(),before);
}
