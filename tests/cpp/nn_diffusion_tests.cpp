#include <gtest/gtest.h>
#include <dm/dm.hpp>
namespace n=dm::nn;namespace m=n::models;
namespace {
n::Tensor Times(){return n::from_vector(std::vector<std::int32_t>{0,31},{2});}
n::Tensor States(n::DType dtype=n::DType::Float64){return n::reshape(n::arange(0,128,1,dtype)/128.-.5,{2,8,8,1});}
}
TEST(NnDiffusion, CosineSourceScheduleAndConfigurableTransitionCounts) {
 auto s=m::CosineDiffusionSchedule();EXPECT_EQ(s.beta.shape(),(n::Shape{32}));EXPECT_EQ(s.beta.dtype(),n::DType::Float64);EXPECT_FALSE(s.alpha_bar.requires_grad());
 EXPECT_DOUBLE_EQ(s.alpha_bar.to_vector<double>().back(),2.369970271136327e-6);EXPECT_EQ(s.posterior_variance.to_vector<double>()[0],0);EXPECT_EQ(s.beta.to_vector<double>().back(),.999);
 for(int steps:{2,8,32,64}) {
  auto a=m::CosineDiffusionSchedule(steps);auto alpha=a.alpha.to_vector<double>(),bar=a.alpha_bar.to_vector<double>(),previous=a.alpha_bar_previous.to_vector<double>();double product=1.;
  for(int i=0;i<steps;++i){EXPECT_DOUBLE_EQ(previous[i],product);product*=alpha[i];EXPECT_DOUBLE_EQ(bar[i],product);EXPECT_GT(product,0);}
 }
 auto other=m::CosineDiffusionSchedule(8,.05,.8);EXPECT_LE(n::max(other.beta).item(),.8);EXPECT_EQ(other.beta.size(0),8);
}
TEST(NnDiffusion, CoefficientSelectionSupportsStridesRepeatedIndicesAndLiveTablePartials) {
 auto table=n::from_vector(std::vector<double>{.1,.2,.3,.4},{4}).requires_grad_();auto t=n::from_vector(std::vector<std::int64_t>{3,0,3},{3});
 auto out=m::DiffusionExtract(table,t,2,n::DType::Float64);EXPECT_EQ(out.shape(),(n::Shape{3,1}));EXPECT_EQ(out.to_vector<double>(),(std::vector<double>{.4,.1,.4}));
 auto g=n::grad(n::sum(n::square(out)),{table},n::GradOptions{true,true,false})[0];EXPECT_EQ(g.to_vector<double>(),(std::vector<double>{.2,0,0,1.6}));
 auto higher=n::grad(n::sum(g),{table})[0];EXPECT_EQ(higher.to_vector<double>(),(std::vector<double>{2,0,0,4}));
 auto index=n::slice_axis(n::from_vector(std::vector<std::int32_t>{0,2,3,1},{4}),0,0,4,2);EXPECT_EQ(m::DiffusionExtract(table,index,1).size(0),2);
}
TEST(NnDiffusion, StructuralTimeFeaturesRetainFloat32PhaseEvenInDouble) {
 auto t=Times();auto f=m::DiffusionTimeEmbedding(t),d=m::DiffusionTimeEmbedding(t,31.f,n::DType::Float64);
 EXPECT_TRUE(n::equal_bits(n::cast(f,n::DType::Float64),d));EXPECT_FALSE(d.requires_grad());
 auto values=d.to_vector<double>();for(int j=0;j<4;++j){EXPECT_EQ(values[j],0);EXPECT_EQ(values[j+4],1);}
 EXPECT_NEAR(values[8],double(float(std::sin(float(std::acos(-1.))))),1e-14);EXPECT_GT(std::abs(values[8]),1e-8);
 auto normalized=m::DiffusionTimeEmbedding(n::from_vector(std::vector<std::int32_t>{1},{1}),1.f);EXPECT_EQ(normalized.shape(),(n::Shape{1,8}));
}
TEST(NnDiffusion, ForwardNoisingAndPosteriorKeepSourceRoundedCoefficientsInBothDtypes) {
 auto s=m::CosineDiffusionSchedule();auto t=n::from_vector(std::vector<std::int32_t>{1,17},{2});
 for(auto type:{n::DType::Float32,n::DType::Float64}) {
  auto clean=n::full({2,3},.25,type),noise=n::full({2,3},-.1,type);auto out=m::DiffusionNoised(clean,noise,t,s);
  auto signal=n::cast(m::DiffusionExtract(s.sqrt_alpha_bar,t,2),type),sigma=n::cast(m::DiffusionExtract(s.sqrt_one_minus_alpha_bar,t,2),type);
  EXPECT_TRUE(n::equal_bits(out,signal*clean+sigma*noise));auto rec=m::DdpmReverse(out,t,noise,n::zeros({2,3},type),s);EXPECT_LT(n::max(n::abs(rec.x0_raw-clean)).item(),type==n::DType::Float32?1e-7:1e-15);
  if(type==n::DType::Float64){auto full=m::DiffusionExtract(s.sqrt_alpha_bar,t,2,type)*clean+m::DiffusionExtract(s.sqrt_one_minus_alpha_bar,t,2,type)*noise;EXPECT_GT(n::max(n::abs(full-out)).item(),1e-10);}
 }
}
TEST(NnDiffusion, LastTransitionHasExactlyZeroInnovationEffectAndClippingRetainsRawValues) {
 auto s=m::CosineDiffusionSchedule();auto time=n::from_vector(std::vector<std::int32_t>{0,31},{2});auto x=n::from_vector(std::vector<double>{-.25,.25,2,-2},{2,2});auto eps=n::zeros({2,2},n::DType::Float64);
 auto a=m::DdpmReverse(x,time,eps,n::ones({2,2},n::DType::Float64),s),b=m::DdpmReverse(x,time,eps,n::full({2,2},123.,n::DType::Float64),s);
 EXPECT_EQ(a.variance.to_vector<double>()[0],0);EXPECT_TRUE(n::equal_bits(n::slice_axis(a.value,0,0,1),n::slice_axis(b.value,0,0,1)));
 auto raw=a.x0_raw.to_vector<double>(),clipped=a.x0_clipped.to_vector<double>();EXPECT_GT(raw[2],100);EXPECT_LT(raw[3],-100);EXPECT_EQ(clipped[2],1);EXPECT_EQ(clipped[3],-1);EXPECT_GT(n::max(n::abs(a.value-b.value)).item(),1);
}
TEST(NnDiffusion, SourceMlpCensusInitializationOrderAndWarmRngImmutability) {
 n::ParameterStore p(11003,n::DType::Float64),q(11003,n::DType::Float64);
 for(const std::string name:{"hidden1","hidden2","output"}){q.get(name+"/W",{name=="hidden1"?72:32,name=="output"?64:32},n::init::normal(.1));q.get(name+"/b",{name=="output"?64:32},n::init::zeros());}
 auto out=m::DdpmEpsilon(p,States(),Times());EXPECT_EQ(p.count(),5504);EXPECT_EQ(p.size(),6);EXPECT_EQ(out.hidden2.shape(),(n::Shape{2,32}));
 for(auto name:p.names())EXPECT_TRUE(n::equal_bits(p.at(name),q.at(name)))<<name;
 auto before=p.flatten();m::DdpmEpsilon(p,States(),Times());EXPECT_EQ(before,p.flatten());EXPECT_TRUE(n::equal_bits(p.get("extra",{7},n::init::normal(.1)),q.get("extra",{7},n::init::normal(.1))));
}
TEST(NnDiffusion, All5504ParameterCoordinatesMatchFiniteDifferencesAndHigherStatePaths) {
 n::ParameterStore p(11003,n::DType::Float64);auto x=States().requires_grad_(),t=Times();m::DdpmEpsilon(p,x,t);
 auto objective=[&](){auto pred=m::DdpmEpsilon(p,x,t);return n::mean(n::square(pred.epsilon-.2));};
 auto gradients=n::grad(objective(),p.parameters(),n::GradOptions{true,true,false});auto mixed=n::grad(n::sum(gradients[0]),{x})[0];EXPECT_GT(n::max(n::abs(mixed)).item(),1e-6);
 auto names=p.names();auto params=p.parameters();for(std::size_t k=0;k<names.size();++k){auto original=params[k].detach().clone();auto v=original.to_vector<double>(),g=gradients[k].to_vector<double>();
  for(std::size_t at=0;at<v.size();++at){auto hi=v,lo=v;hi[at]+=1e-5;lo[at]-=1e-5;p.set(names[k],n::from_vector(hi,original.shape()));double high=objective().item();p.set(names[k],n::from_vector(lo,original.shape()));double low=objective().item();p.set(names[k],original);EXPECT_NEAR((high-low)/2e-5,g[at],2e-10)<<names[k]<<":"<<at;}
 }
}
TEST(NnDiffusion, PureForwardAndReverseFieldsKeepApplicableHigherPathsAfterCollection) {
 auto clean=n::full({2,3},.2,n::DType::Float64).requires_grad_(),eps=n::full({2,3},.1,n::DType::Float64).requires_grad_();auto t=n::from_vector(std::vector<std::int32_t>{4,17},{2});n::Tensor state,raw,mean,value;
 {auto s=m::CosineDiffusionSchedule();state=m::DiffusionNoised(clean,eps,t,s);auto reverse=m::DdpmReverse(state,t,eps,n::ones({2,3},n::DType::Float64),s);raw=reverse.x0_raw;mean=reverse.mean;value=reverse.value;}
 for(auto field:{state,raw,mean,value}){auto g=n::grad(n::sum(n::square(field)),{clean},n::GradOptions{true,true,false})[0];EXPECT_GT(n::max(n::abs(g)).item(),1e-5);EXPECT_TRUE(n::grad(n::sum(g),{clean},n::GradOptions{true,false,false})[0].defined());}
}
TEST(NnDiffusion, PredictorFlatFieldsSurviveStoreAndParentCollectionWithHigherGraphs) {
 auto x=States().requires_grad_();n::Tensor epsilon,h1,h2,embedding;
 {n::ParameterStore p(11003,n::DType::Float64);auto o=m::DdpmEpsilon(p,x,Times());epsilon=o.epsilon;h1=o.hidden1;h2=o.hidden2;embedding=o.time_embedding;}
 for(auto field:{epsilon,h1,h2}){auto g=n::grad(n::sum(n::square(field)),{x},n::GradOptions{true,true,false})[0];EXPECT_TRUE(n::grad(n::sum(g),{x},n::GradOptions{true,false,false})[0].defined());}
 EXPECT_EQ(embedding.shape(),(n::Shape{2,8}));EXPECT_FALSE(embedding.requires_grad());
}
TEST(NnDiffusion, StridedStatesAndIntegerTimesMatchContiguousCallsWithoutMutation) {
 n::ParameterStore p(11003);auto x=n::transpose(States(n::DType::Float32),{0,2,1,3});auto before=x.clone();auto t=n::slice_axis(n::from_vector(std::vector<std::int32_t>{0,7,31,9},{4}),0,0,4,2);
 auto a=m::DdpmEpsilon(p,x,t),b=m::DdpmEpsilon(p,x.contiguous(),t.contiguous());EXPECT_TRUE(n::equal_bits(a.epsilon,b.epsilon));EXPECT_TRUE(n::equal_bits(x,before));
 auto s=m::CosineDiffusionSchedule();EXPECT_TRUE(n::equal_bits(m::DiffusionNoised(x,x,t,s),m::DiffusionNoised(x.contiguous(),x.contiguous(),t.contiguous(),s)));
}
TEST(NnDiffusion, ExplicitHostOwnedAdamUpdatesTheEpsilonModel) {
 n::ParameterStore p(11003);auto x=States(n::DType::Float32),t=Times();m::DdpmEpsilon(p,x,t);n::OptimizerConfig c;c.kind=n::OptimizerKind::Adam;n::Optimizer optimizer(p,.003,c);auto before=p.flatten();
 auto o=m::DdpmEpsilon(p,x,t);auto gradients=n::grad(n::mean(n::square(o.epsilon-.2)),p.parameters());optimizer.step(gradients);EXPECT_EQ(optimizer.iterations(),1);EXPECT_NE(before,p.flatten());
}
TEST(NnDiffusion, InvalidSchedulesIndicesAndEmbeddingOptionsReject) {
 for(int count:{-1,0,1}){EXPECT_THROW(m::CosineDiffusionSchedule(count),std::invalid_argument);}
 for(double offset:{-1.,double(NAN),double(INFINITY),1e300}){EXPECT_THROW(m::CosineDiffusionSchedule(32,offset),std::invalid_argument);}
 for(double cap:{0.,1.,-1.,double(NAN)}){EXPECT_THROW(m::CosineDiffusionSchedule(32,.008,cap),std::invalid_argument);}
 auto s=m::CosineDiffusionSchedule();for(auto t:{n::Tensor(),n::zeros({0},n::DType::Int32),n::zeros({1}),n::zeros({1,1},n::DType::Int32),n::from_vector(std::vector<std::int32_t>{-1},{1}),n::from_vector(std::vector<std::int32_t>{32},{1})}){
  EXPECT_THROW(m::DiffusionExtract(s.beta,t),std::invalid_argument);
 }
 for(float maximum:{0.f,-1.f,float(NAN),float(INFINITY)}){EXPECT_THROW(m::DiffusionTimeEmbedding(Times(),maximum),std::invalid_argument);}
 EXPECT_THROW(m::DiffusionExtract(s.beta,Times(),0),std::invalid_argument);
 EXPECT_THROW(m::DiffusionTimeEmbedding(Times(),31.f,n::DType::Int32),std::invalid_argument);
}
TEST(NnDiffusion, InvalidPredictorGeometryTimeAndDtypeDoNotCreateLeaves) {
 n::ParameterStore p(11003);for(auto x:{n::Tensor(),n::zeros({2,64}),n::zeros({2,8,8,2}),n::zeros({0,8,8,1}),n::zeros({2,8,8,1},n::DType::Int32),n::full({2,8,8,1},NAN),States()}){
  EXPECT_THROW(m::DdpmEpsilon(p,x,Times()),std::invalid_argument);EXPECT_EQ(p.size(),0);
 }
 for(auto t:{n::zeros({1},n::DType::Int32),n::zeros({2}),n::full({2},32,n::DType::Int32)}){
  EXPECT_THROW(m::DdpmEpsilon(p,States(n::DType::Float32),t),std::invalid_argument);EXPECT_EQ(p.size(),0);
 }
}
TEST(NnDiffusion, InvalidPosteriorTablesAndUnsafeComputedMathRejectBeforeClipping) {
 auto s=m::CosineDiffusionSchedule();auto x=n::zeros({2,3}),t=Times();
 EXPECT_THROW(m::DdpmReverse(x,t,x,x,m::DiffusionScheduleOutput{}),std::invalid_argument);
 EXPECT_THROW(m::DiffusionNoised(x,n::zeros({2,2}),t,s),std::invalid_argument);
 s.posterior_variance=n::full({32},-1,n::DType::Float64);EXPECT_THROW(m::DdpmReverse(x,t,x,x,s),std::invalid_argument);
 s=m::CosineDiffusionSchedule();s.sqrt_alpha_bar=n::zeros({32},n::DType::Float64);EXPECT_THROW(m::DiffusionNoised(x,x,t,s),std::invalid_argument);
 s=m::CosineDiffusionSchedule();EXPECT_THROW(m::DdpmReverse(n::full({2,3},1e38),t,n::full({2,3},-1e38),x,s),std::invalid_argument);
 n::ParameterStore p(11003);m::DdpmEpsilon(p,States(n::DType::Float32),t);p.set("hidden1/W",n::full({72,32},INFINITY));EXPECT_THROW(m::DdpmEpsilon(p,States(n::DType::Float32),t),std::invalid_argument);
}
