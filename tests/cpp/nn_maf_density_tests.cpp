#include <gtest/gtest.h>
#include <dm/dm.hpp>
namespace n=dm::nn;namespace m=n::models;
TEST(NnMafDensity, AlternatingDegreesAndStrictMasksMatchAllSourceMetadata) {
 for(int block=0;block<4;++block)for(auto dtype:{n::DType::Float32,n::DType::Float64}){
  auto masks=m::MafMasks(block,dtype);auto input=masks.input_degrees.to_vector<int>();
  auto hidden=masks.hidden_degrees.to_vector<int>(),heads=masks.output_degrees.to_vector<int>();
  EXPECT_EQ(input,(block%2?std::vector<int>{2,1}:std::vector<int>{1,2}));
  auto im=masks.input_mask.to_vector<double>(),om=masks.output_mask.to_vector<double>();int zero=0;
  for(int j=0;j<16;++j){EXPECT_EQ(hidden[j],j/8);
   for(int c=0;c<2;++c){EXPECT_EQ(im[c*16+j],double(input[c]<=j/8));zero+=im[c*16+j]==0;}
   for(int k=0;k<4;++k){EXPECT_EQ(heads[k],input[k%2]);EXPECT_EQ(om[j*4+k],double(j/8<input[k%2]));zero+=om[j*4+k]==0;}}
  EXPECT_EQ(zero,40);EXPECT_EQ(masks.input_mask.dtype(),dtype);
  for(auto field:{masks.input_degrees,masks.hidden_degrees,masks.output_degrees,masks.input_mask,masks.output_mask})EXPECT_FALSE(field.requires_grad());
 }
}
TEST(NnMafDensity, ConditionerHeadsHaveStrictZeroOwnAndFutureInputJacobians) {
 for(int i=0;i<4;++i){n::ParameterStore p(9603,n::DType::Float64);
  auto x=n::from_vector(std::vector<double>{.2,-.7,1.1,.4,-.3,2},{3,2}).requires_grad_();auto c=m::MafConditioner(p,x,i);
  EXPECT_EQ(p.count(),116);EXPECT_EQ(p.size(),4);auto degrees=m::MafMasks(i).input_degrees.to_vector<int>();
  for(int k=0;k<4;++k){auto head=n::slice_axis(k<2?c.mean:c.scale,1,k%2,k%2+1);
   auto derivative=n::grad(n::sum(head),{x},n::GradOptions{true,true,false})[0].to_vector<double>();
   double allowed=0;for(int r=0;r<3;++r)for(int j=0;j<2;++j){if(degrees[j]>=degrees[k%2])EXPECT_EQ(derivative[r*2+j],0);else allowed=std::max(allowed,std::abs(derivative[r*2+j]));}
   if(degrees[k%2]==2)EXPECT_GT(allowed,1e-6);
  }
 }
}
TEST(NnMafDensity, SourceCensusAndSequentialInverseRecoverBothAlternatingOrders) {
 for(auto dtype:{n::DType::Float32,n::DType::Float64}){
  n::ParameterStore p(9603,dtype);auto x=n::cast(n::from_vector(std::vector<double>{.2,-.7,1.1,.4,-.3,2},{3,2}),dtype);
  auto path=m::MafDensity(p,x);auto inverse=m::MafInverse(p,path.latent);EXPECT_EQ(p.count(),466);EXPECT_EQ(p.size(),17);
  double tolerance=dtype==n::DType::Float32?2e-6:2e-15;
  EXPECT_LT(n::max(n::abs(inverse.value-x)).item(),tolerance);
  EXPECT_LT(n::max(n::abs(inverse.logdet+path.logdet)).item(),tolerance);
  EXPECT_TRUE(n::equal_bits(path.unscaled,path.coupling3));
  for(auto pair:{std::make_pair(path.coupling_logdet0,path.coupling_scale0),std::make_pair(path.coupling_logdet1,path.coupling_scale1),std::make_pair(path.coupling_logdet2,path.coupling_scale2),std::make_pair(path.coupling_logdet3,path.coupling_scale3)})
   EXPECT_TRUE(n::equal_bits(pair.first,-n::sum(pair.second,{1})));
 }
}
TEST(NnMafDensity, BothActualJacobiansAndSpatialDeterminantsMatchTheirOwnPaths) {
 n::ParameterStore p(9603,n::DType::Float64);
 auto x=n::from_vector(std::vector<double>{.2,-.7,1.1,.4,-.3,2},{3,2}).requires_grad_();auto path=m::MafDensity(p,x);
 p.set("log_scale",n::from_vector(std::vector<double>{.2,-.3},{2}));path=m::MafDensity(p,x);
 auto z=n::from_vector(std::vector<double>{-.8,1.2,.4,-1.3},{2,2}).requires_grad_();auto inverse=m::MafInverse(p,z);
 for(auto entry:{std::make_tuple(path.latent,path.logdet,x),std::make_tuple(inverse.value,inverse.logdet,z)}){
  auto value=std::get<0>(entry),ld=std::get<1>(entry),input=std::get<2>(entry);
  auto a=n::grad(n::sum(n::slice_axis(value,1,0,1)),{input},n::GradOptions{true,false,false})[0].to_vector<double>();
  auto b=n::grad(n::sum(n::slice_axis(value,1,1,2)),{input})[0].to_vector<double>();
  for(int r=0;r<input.size(0);++r){double det=a[r*2]*b[r*2+1]-a[r*2+1]*b[r*2];EXPECT_GT(det,0);EXPECT_NEAR(std::log(det),ld.to_vector<double>()[r],1e-15);}
 }
 EXPECT_GT((n::max(path.logdet)-n::min(path.logdet)).item(),1e-5);
 auto density=m::MafDensity(p,inverse.value);EXPECT_LT(n::max(n::abs(density.logdet+inverse.logdet)).item(),2e-15);
}
TEST(NnMafDensity, MeanNllMatchesAll17LeafFiniteDifferencesAndRetainsHigherPaths) {
 n::ParameterStore p(9603,n::DType::Float64);auto x=n::from_vector(std::vector<double>{.4,-.7,1.2,.3,-1,2},{3,2}).requires_grad_();
 auto path=m::MafDensity(p,x);auto loss=m::NormalFlowDensity(path.latent,path.logdet).loss;
 auto gradients=n::grad(loss,p.parameters(),n::GradOptions{true,true,false});
 auto higher=n::grad(n::sum(gradients[0]),{x})[0];EXPECT_GT(n::max(n::abs(higher)).item(),1e-8);
 auto names=p.names();auto parameters=p.parameters();ASSERT_EQ(names.size(),17);
 for(std::size_t i=0;i<names.size();++i){auto original=parameters[i].detach().clone();auto hi=original.to_vector<double>(),lo=hi;auto g=gradients[i].to_vector<double>();
  auto at=std::max_element(g.begin(),g.end(),[](double a,double b){return std::abs(a)<std::abs(b);})-g.begin();hi[at]+=1e-5;lo[at]-=1e-5;
  p.set(names[i],n::from_vector(hi,original.shape()));auto a=m::MafDensity(p,x);double high=m::NormalFlowDensity(a.latent,a.logdet).loss.item();
  p.set(names[i],n::from_vector(lo,original.shape()));auto b=m::MafDensity(p,x);double low=m::NormalFlowDensity(b.latent,b.logdet).loss.item();p.set(names[i],original);
  EXPECT_NEAR((high-low)/2e-5,g[at],3e-9)<<names[i];
 }
}
TEST(NnMafDensity, All160MaskedKernelPartialsStayZeroAndAdamPreservesStoredWeights) {
 n::ParameterStore p(9603);auto x=n::from_vector(std::vector<float>{.4f,-.7f,1.2f,.3f,-1,2},{3,2});m::MafDensity(p,x);
 n::OptimizerConfig options;options.kind=n::OptimizerKind::Adam;n::Optimizer optimizer(p,.003,options);int checked=0;
 for(int step=0;step<3;++step){auto path=m::MafDensity(p,x);auto gradients=n::grad(m::NormalFlowDensity(path.latent,path.logdet).loss,p.parameters());
  auto before=p.flatten();auto names=p.names();auto parameters=p.parameters();optimizer.step(gradients);
  for(int i=0;i<4;++i){auto masks=m::MafMasks(i);
   for(auto pair:{std::make_pair(std::string("hidden/W"),masks.input_mask),std::make_pair(std::string("heads/W"),masks.output_mask)}){
    const auto key="autoregressive"+std::to_string(i)+"/"+pair.first;auto at=std::find(names.begin(),names.end(),key)-names.begin();
    auto mask=pair.second.to_vector<float>(),g=gradients[at].to_vector<float>();auto after=p.at(key).to_vector<float>();
    // Parameter handles are live after step; compare the separately saved flattened store below.
    std::size_t offset=0;for(decltype(at) j=0;j<at;++j)offset+=parameters[j].numel();
    for(std::size_t j=0;j<mask.size();++j)if(mask[j]==0){EXPECT_EQ(g[j],0);EXPECT_EQ(after[j],before[offset+j]);++checked;}
   }
  }
 }EXPECT_EQ(checked,480);
}
TEST(NnMafDensity, StridesWarmImmutabilityAndNoAdditionalRngDraws) {
 n::ParameterStore p(9603,n::DType::Float64),other(9603,n::DType::Float64);
 auto x=n::from_vector(std::vector<double>{.1,.2,.3,.4,.5,.6},{3,2});auto first=m::MafDensity(p,x);m::MafDensity(other,x);auto before=p.flatten();
 auto strided=n::transpose(n::transpose(x,{1,0}).contiguous(),{1,0});auto second=m::MafDensity(p,strided);
 EXPECT_TRUE(n::equal_bits(first.latent,second.latent));EXPECT_TRUE(n::equal_bits(first.logdet,second.logdet));
 auto inverse=m::MafInverse(p,first.latent);m::MafInverse(p,first.latent);EXPECT_EQ(p.flatten(),before);
 EXPECT_TRUE(n::equal_bits(p.get("extra",{7},n::init::normal(.1)),other.get("extra",{7},n::init::normal(.1))));
}
TEST(NnMafDensity, OwningStageConditionerAndInverseGraphsSurviveParentAndStoreCollection) {
 auto x=n::from_vector(std::vector<double>{.1,.2,.3,.4},{2,2}).requires_grad_();n::Tensor mean,scale,stage,inverse;
 {n::ParameterStore p(9603,n::DType::Float64);auto path=m::MafDensity(p,x);auto cond=m::MafConditioner(p,x,1);
  mean=cond.mean;scale=path.coupling_scale3;stage=path.conditioner_input2;inverse=m::MafInverse(p,path.latent).value;}
 for(auto value:{mean,scale,stage,inverse})EXPECT_TRUE(n::grad(n::sum(value),{x},n::GradOptions{true,true,false})[0].defined());
}
TEST(NnMafDensity, InvalidInputMaskOptionsAndDtypesRejectBeforeLeafCreation) {
 n::ParameterStore p(9603);
 for(auto x:{n::Tensor(),n::zeros({2}),n::zeros({1,3}),n::zeros({0,2}),n::zeros({1,2},n::DType::Int32),n::full({1,2},NAN),n::zeros({1,2},n::DType::Float64)}){
  EXPECT_THROW(m::MafDensity(p,x),std::invalid_argument);
  EXPECT_THROW(m::MafInverse(p,x),std::invalid_argument);
  EXPECT_THROW(m::MafConditioner(p,x,0),std::invalid_argument);EXPECT_EQ(p.size(),0);}
 for(int block:{-1,4}){EXPECT_THROW(m::MafConditioner(p,n::zeros({2,2}),block),std::invalid_argument);
  EXPECT_THROW(m::MafMasks(block),std::invalid_argument);}
 EXPECT_THROW(m::MafMasks(0,n::DType::Int32),std::invalid_argument);EXPECT_EQ(p.size(),0);
}
TEST(NnMafDensity, NonfiniteStoredValuesAndUnrepresentableGlobalScalesRejectWithoutWrites) {
 n::ParameterStore p(9603);auto x=n::zeros({2,2});m::MafDensity(p,x);
 for(double value:{1000.,-1000.,double(NAN)}){auto scale=n::full({2},value);p.set("log_scale",scale);
  EXPECT_THROW(m::MafDensity(p,x),std::invalid_argument);
  EXPECT_THROW(m::MafInverse(p,x),std::invalid_argument);EXPECT_TRUE(n::equal_bits(p.at("log_scale"),scale));}
 p.set("log_scale",n::zeros({2}));auto bad=n::full({16,4},INFINITY);p.set("autoregressive0/heads/W",bad);
 EXPECT_THROW(m::MafDensity(p,x),std::invalid_argument);
 EXPECT_THROW(m::MafInverse(p,x),std::invalid_argument);EXPECT_TRUE(n::equal_bits(p.at("autoregressive0/heads/W"),bad));
}
