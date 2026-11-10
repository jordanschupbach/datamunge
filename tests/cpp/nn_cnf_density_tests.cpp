#include <gtest/gtest.h>
#include <dm/dm.hpp>
namespace n=dm::nn;namespace m=n::models;
namespace {
n::Tensor Points(){return n::from_vector(std::vector<double>{.4,-.7,1.2,.3,-1,2},{3,2});}
n::Tensor ActualLogdet(const n::Tensor& value,const n::Tensor& input){
 auto keep=n::GradOptions{true,true,false};auto a=n::grad(n::sum(n::slice_axis(value,1,0,1)),{input},keep)[0];auto b=n::grad(n::sum(n::slice_axis(value,1,1,2)),{input},keep)[0];
 auto determinant=n::select(a,1,0)*n::select(b,1,1)-n::select(a,1,1)*n::select(b,1,0);EXPECT_GT(n::min(determinant).item(),0);return n::log(determinant);
}
void SetScalarTanh(n::ParameterStore& p,double output_weight){
 auto type=p.dtype();m::CnfField(p,n::zeros({1,2},type),0.f);std::vector<double> wi(24,0),wo(16,0);wi[0]=1;wo[0]=output_weight;
 p.set("field/hidden/W",n::cast(n::from_vector(wi,{3,8}),type));p.set("field/output/W",n::cast(n::from_vector(wo,{8,2}),type));p.set("field/hidden/b",n::zeros({8},type));p.set("field/output/b",n::zeros({2},type));
}
}
TEST(NnCnfDensity, BothDtypesSourceCensusAndNumericalRoundtrip){
 for(auto dtype:{n::DType::Float32,n::DType::Float64}){
  n::ParameterStore p(10003,dtype);auto x=n::cast(Points(),dtype);auto out=m::CnfDensity(p,x);auto rec=m::CnfInverse(p,out.latent);
  EXPECT_EQ(p.count(),52);EXPECT_EQ(p.size(),5);EXPECT_TRUE(n::equal_bits(out.unscaled,out.stage3));EXPECT_EQ(out.stage_velocity1.shape(),(n::Shape{3,2}));EXPECT_EQ(out.stage_divergence2.shape(),(n::Shape{3}));
  double tolerance=dtype==n::DType::Float32?5e-6:1e-7;EXPECT_LT(n::max(n::abs(rec.value-x)).item(),tolerance);EXPECT_LT(n::max(n::abs(rec.logdet+out.logdet)).item(),tolerance);
 }
}
TEST(NnCnfDensity, KernelFirstInitializationAndFieldOnlyCensusPreserveSourceRng){
 n::ParameterStore p(10003,n::DType::Float64),q(10003,n::DType::Float64);auto wi=q.get("field/hidden/W",{3,8},n::init::normal(.3)),wo=q.get("field/output/W",{8,2},n::init::normal(.1));
 m::CnfField(p,n::zeros({1,2},n::DType::Float64),.5f);EXPECT_EQ(p.count(),50);EXPECT_EQ(p.size(),4);EXPECT_TRUE(n::equal_bits(wi,p.at("field/hidden/W")));EXPECT_TRUE(n::equal_bits(wo,p.at("field/output/W")));
 m::CnfDensity(p,n::zeros({1,2},n::DType::Float64));EXPECT_TRUE(n::equal_bits(p.get("extra",{7},n::init::normal(.1)),q.get("extra",{7},n::init::normal(.1))));
}
TEST(NnCnfDensity, AllTraceModesMatchActualFieldJacobianContractionsAndHigherPaths){
 n::ParameterStore p(10003,n::DType::Float64);auto x=Points().requires_grad_();auto keep=n::GradOptions{true,true,false};
 auto f=m::CnfField(p,x,.5f);auto a=n::grad(n::sum(n::slice_axis(f.velocity,1,0,1)),{x},keep)[0],b=n::grad(n::sum(n::slice_axis(f.velocity,1,1,2)),{x},keep)[0];
 auto exact=n::select(a,1,0)+n::select(b,1,1),plus=n::sum(a+b,{1}),minus=n::select(a,1,0)-n::select(a,1,1)-n::select(b,1,0)+n::select(b,1,1);
 EXPECT_LT(n::max(n::abs(exact-f.exact_trace)).item(),1e-16);EXPECT_LT(n::max(n::abs(plus-f.hutchinson_plus)).item(),1e-16);EXPECT_LT(n::max(n::abs(minus-f.hutchinson_minus)).item(),1e-16);
 EXPECT_GT(n::max(n::abs(plus-exact)).item(),1e-4);
 std::vector<n::Tensor> expected{f.exact_trace,f.hutchinson_plus,f.hutchinson_minus,.5*(f.hutchinson_plus+f.hutchinson_minus)};
 for(int mode=0;mode<4;++mode){auto out=m::CnfField(p,x,.5f,mode);EXPECT_TRUE(n::equal_bits(out.divergence,expected[mode]));auto g=n::grad(n::sum(out.divergence),{x},keep)[0];EXPECT_TRUE(n::grad(n::sum(g),{x},keep)[0].defined());}
}
TEST(NnCnfDensity, FieldAll50ParameterCoordinatesMatchFiniteDifferencesInEveryTraceMode){
 for(int mode=0;mode<4;++mode){
  n::ParameterStore p(10003,n::DType::Float64);auto x=Points();m::CnfField(p,x,.5f,mode);
  auto objective=[&](){auto f=m::CnfField(p,x,.5f,mode);return n::mean(n::sum(n::square(f.velocity),{1}),{0})+.3*n::mean(f.divergence,{0});};
  auto gradients=n::grad(objective(),p.parameters());auto names=p.names();auto params=p.parameters();
  for(std::size_t i=0;i<names.size();++i){auto original=params[i].detach().clone();auto values=original.to_vector<double>(),g=gradients[i].to_vector<double>();
   for(std::size_t at=0;at<values.size();++at){auto hi=values,lo=values;hi[at]+=1e-5;lo[at]-=1e-5;p.set(names[i],n::from_vector(hi,original.shape()));double high=objective().item();p.set(names[i],n::from_vector(lo,original.shape()));double low=objective().item();p.set(names[i],original);EXPECT_NEAR((high-low)/2e-5,g[at],2e-10)<<mode<<names[i]<<":"<<at;}
  }
 }
}
TEST(NnCnfDensity, MeanNllAll52CoordinatesAndHigherGraphsThroughTheFullUnrolledSolver){
 n::ParameterStore p(10003,n::DType::Float64);auto x=Points().requires_grad_();m::CnfDensity(p,x);
 auto objective=[&](){auto out=m::CnfDensity(p,x);return m::NormalFlowDensity(out.latent,out.logdet).loss;};
 auto gradients=n::grad(objective(),p.parameters(),n::GradOptions{true,true,false});auto higher=n::grad(n::sum(gradients[0]),{x})[0];EXPECT_GT(n::max(n::abs(higher)).item(),1e-8);
 auto names=p.names();auto params=p.parameters();for(std::size_t i=0;i<names.size();++i){auto original=params[i].detach().clone();auto values=original.to_vector<double>(),g=gradients[i].to_vector<double>();
  for(std::size_t at=0;at<values.size();++at){auto hi=values,lo=values;hi[at]+=1e-5;lo[at]-=1e-5;p.set(names[i],n::from_vector(hi,original.shape()));double high=objective().item();p.set(names[i],n::from_vector(lo,original.shape()));double low=objective().item();p.set(names[i],original);EXPECT_NEAR((high-low)/2e-5,g[at],3e-9)<<names[i]<<":"<<at;}
 }
}
TEST(NnCnfDensity, InverseObjectiveAll52CoordinatesAndHigherGraphsUseReverseRk4){
 n::ParameterStore p(10003,n::DType::Float64);auto z=Points().requires_grad_();m::CnfDensity(p,z);
 auto objective=[&](){auto out=m::CnfInverse(p,z);return n::mean(n::sum(n::square(out.value),{1}),{0})+.3*n::mean(out.logdet,{0});};
 auto gradients=n::grad(objective(),p.parameters(),n::GradOptions{true,true,false});EXPECT_TRUE(n::grad(n::sum(gradients[0]),{z})[0].defined());
 auto names=p.names();auto params=p.parameters();for(std::size_t i=0;i<names.size();++i){auto original=params[i].detach().clone();auto values=original.to_vector<double>(),g=gradients[i].to_vector<double>();
  for(std::size_t at=0;at<values.size();++at){auto hi=values,lo=values;hi[at]+=1e-5;lo[at]-=1e-5;p.set(names[i],n::from_vector(hi,original.shape()));double high=objective().item();p.set(names[i],n::from_vector(lo,original.shape()));double low=objective().item();p.set(names[i],original);EXPECT_NEAR((high-low)/2e-5,g[at],3e-9)<<names[i]<<":"<<at;}
 }
}
TEST(NnCnfDensity, BothDiscreteMapJacobiansAgreeWithIntegratedTracesWithinNumericalResiduals){
 n::ParameterStore p(10003,n::DType::Float64);auto x=Points().requires_grad_();m::CnfDensity(p,x);p.set("log_scale",n::from_vector(std::vector<double>{.2,-.3},{2}));
 auto forward=m::CnfDensity(p,x);auto z=n::from_vector(std::vector<double>{-.8,1.2,.4,-1.3},{2,2}).requires_grad_();auto inverse=m::CnfInverse(p,z);
 EXPECT_LT(n::max(n::abs(ActualLogdet(forward.latent,x)-forward.logdet)).item(),1e-7);EXPECT_LT(n::max(n::abs(ActualLogdet(inverse.value,z)-inverse.logdet)).item(),1e-7);
 EXPECT_GT((n::max(forward.logdet)-n::min(forward.logdet)).item(),1e-5);auto back=m::CnfDensity(p,inverse.value);EXPECT_LT(n::max(n::abs(back.logdet+inverse.logdet)).item(),1e-7);
 // Single fixed probes leave states unchanged but generally are not full logdets.
 auto exact=m::CnfDensity(p,x,8,0),plus=m::CnfDensity(p,x,8,1),minus=m::CnfDensity(p,x,8,2),paired=m::CnfDensity(p,x,8,3);
 EXPECT_TRUE(n::equal_bits(exact.latent,plus.latent));EXPECT_TRUE(n::equal_bits(exact.latent,minus.latent));EXPECT_TRUE(n::equal_bits(exact.latent,paired.latent));
 EXPECT_GT(n::max(n::abs(plus.logdet-exact.logdet)).item(),1e-4);EXPECT_LT(n::max(n::abs(.5*(plus.logdet+minus.logdet)-paired.logdet)).item(),1e-14);
}
TEST(NnCnfDensity, EveryQuarterEndpointAndCumulativeTraceIsRetainedInBothDirections){
 for(bool inverse:{false,true})for(int steps:{4,8,12,16}){
  n::ParameterStore p(10003,n::DType::Float64);auto path=m::CnfIntegrate(p,Points(),inverse,steps);
  std::vector<n::Tensor> states{path.stage0,path.stage1,path.stage2,path.stage3},determinants{path.stage_logdet0,path.stage_logdet1,path.stage_logdet2,path.stage_logdet3},divergences{path.stage_divergence0,path.stage_divergence1,path.stage_divergence2,path.stage_divergence3},velocities{path.stage_velocity0,path.stage_velocity1,path.stage_velocity2,path.stage_velocity3};
  float dt=(inverse?-1.f:1.f)/steps;
  for(int q=0;q<4;++q){int last=(q+1)*(steps/4)-1;float t=inverse?1.f+dt*last:dt*last;auto f=m::CnfField(p,states[q],t+dt);EXPECT_TRUE(n::equal_bits(f.velocity,velocities[q]));EXPECT_TRUE(n::equal_bits(f.divergence,divergences[q]));}
  EXPECT_TRUE(n::equal_bits(path.stage3,path.unscaled));EXPECT_TRUE(n::equal_bits(path.latent,path.unscaled));EXPECT_TRUE(n::equal_bits(path.logdet,path.stage_logdet3));EXPECT_GT(n::max(n::abs(determinants[3]-determinants[0])).item(),1e-5);
 }
}
TEST(NnCnfDensity, SourceFloat32Rk4CoefficientsRemainRoundedInDoubleEvenForConstantVelocity){
 n::ParameterStore p(10003,n::DType::Float64);SetScalarTanh(p,0.);p.set("field/output/b",n::from_vector(std::vector<double>{.8,-.3},{2}));auto x=Points();auto out=m::CnfIntegrate(p,x);auto velocity=m::CnfField(p,x,0.f).velocity;
 auto expected=x.to_vector<double>(),v=velocity.to_vector<double>();const float dt=1.f/8,coefficient=dt/6.f;
 for(int step=0;step<8;++step)for(std::size_t i=0;i<expected.size();++i)expected[i]+=double(coefficient)*((v[i]+v[i])+2.*(v[i]+v[i]));
 EXPECT_LT(n::max(n::abs(out.latent-n::from_vector(expected,{3,2}))).item(),2e-15);EXPECT_GT(n::max(n::abs(out.latent-(x+velocity))).item(),1e-10);EXPECT_EQ(n::max(n::abs(out.logdet)).item(),0);
}
TEST(NnCnfDensity, FieldTimeIsStructuralFloat32BeforeDoublePromotion){
 n::ParameterStore p(10003,n::DType::Float64);SetScalarTanh(p,1.);std::vector<double> wi(24,0);wi[16]=1;p.set("field/hidden/W",n::from_vector(wi,{3,8}));double time=.1+1e-10;auto f=m::CnfField(p,n::zeros({1,2},n::DType::Float64),float(time));
 EXPECT_NEAR(f.velocity.to_vector<double>()[0],.3*std::tanh(double(float(time))),1e-16);EXPECT_GT(std::abs(f.velocity.to_vector<double>()[0]-.3*std::tanh(time)),1e-12);EXPECT_EQ(f.divergence.item(),0);
}
TEST(NnCnfDensity, AnalyticScalarTanhOdeChecksRefinementAndNumericalJacobianResidual){
 n::ParameterStore p(10003,n::DType::Float64);SetScalarTanh(p,3.);auto x=n::from_vector(std::vector<double>{-1,.2,-.3,.1,0,-.5,.4,.3,1.2,.7},{5,2}).requires_grad_();std::vector<double> wanted=x.to_vector<double>(),wanted_ld;
 for(int r=0;r<5;++r){double original=wanted[r*2];wanted[r*2]=std::asinh(std::exp(.9)*std::sinh(original));wanted_ld.push_back(.9+std::log(std::cosh(original))-std::log(std::cosh(wanted[r*2])));}
 double first=0,last=0;for(int steps:{4,8,16,32}){auto out=m::CnfIntegrate(p,x,false,steps);double error=n::max(n::abs(out.latent-n::from_vector(wanted,{5,2}))).item();if(steps==4)first=error;last=error;EXPECT_LT(n::max(n::abs(out.logdet-n::from_vector(wanted_ld,{5}))).item(),2e-5);EXPECT_LT(n::max(n::abs(ActualLogdet(out.latent,x)-out.logdet)).item(),2e-5);}
 EXPECT_LT(last,first);EXPECT_LT(last,1e-7);
}
TEST(NnCnfDensity, StridesWarmInferenceImmutabilityAndNoAdditionalRngDraws){
 n::ParameterStore p(10003,n::DType::Float64),q(10003,n::DType::Float64);auto x=Points();auto a=m::CnfDensity(p,x);m::CnfDensity(q,x);auto before=p.flatten();
 auto strided=n::transpose(n::transpose(x,{1,0}).contiguous(),{1,0});auto b=m::CnfDensity(p,strided);EXPECT_TRUE(n::equal_bits(a.latent,b.latent));EXPECT_TRUE(n::equal_bits(a.logdet,b.logdet));m::CnfInverse(p,a.latent);m::CnfField(p,x,.3f);EXPECT_EQ(p.flatten(),before);
 EXPECT_TRUE(n::equal_bits(p.get("extra",{7},n::init::normal(.1)),q.get("extra",{7},n::init::normal(.1))));
}
TEST(NnCnfDensity, OwningFieldQuarterAndReverseGraphsSurviveParentAndStoreCollection){
 auto x=Points().requires_grad_();n::Tensor velocity,trace,quarter,cumulative,endpoint,inverse;
 {n::ParameterStore p(10003,n::DType::Float64);auto f=m::CnfField(p,x,.5f);auto path=m::CnfDensity(p,x);velocity=f.velocity;trace=f.exact_trace;quarter=path.stage1;cumulative=path.stage_logdet2;endpoint=path.stage_divergence3;inverse=m::CnfInverse(p,path.latent).value;}
 for(auto value:{velocity,trace,quarter,cumulative,endpoint,inverse}){auto keep=n::GradOptions{true,true,false};auto first=n::grad(n::sum(value),{x},keep)[0];EXPECT_TRUE(first.defined());EXPECT_TRUE(n::grad(n::sum(n::square(first)),{x},keep)[0].defined());}
}
TEST(NnCnfDensity, InvalidInputsStepsModesTimesAndUnsafeStoredMathReject){
 n::ParameterStore p(10003);for(auto x:{n::Tensor(),n::zeros({2}),n::zeros({1,3}),n::zeros({0,2}),n::zeros({1,2},n::DType::Int32),n::full({1,2},NAN),n::zeros({1,2},n::DType::Float64)}){
  EXPECT_THROW(m::CnfDensity(p,x),std::invalid_argument);
  EXPECT_THROW(m::CnfInverse(p,x),std::invalid_argument);EXPECT_EQ(p.size(),0);}
 auto x=n::zeros({1,2});for(int steps:{-4,0,1,2,3,5,6,7,10}){
  EXPECT_THROW(m::CnfDensity(p,x,steps),std::invalid_argument);
  EXPECT_THROW(m::CnfInverse(p,x,steps),std::invalid_argument);EXPECT_EQ(p.size(),0);}
 for(int mode:{-1,4}){
  EXPECT_THROW(m::CnfField(p,x,.5f,mode),std::invalid_argument);
  EXPECT_THROW(m::CnfIntegrate(p,x,false,8,mode),std::invalid_argument);EXPECT_EQ(p.size(),0);}
 EXPECT_THROW(m::CnfField(p,x,NAN),std::invalid_argument);EXPECT_EQ(p.size(),0);m::CnfDensity(p,x);
 for(double scale:{1000.,-1000.,double(NAN)}){auto bad=n::full({2},scale);p.set("log_scale",bad);
  EXPECT_THROW(m::CnfDensity(p,x),std::invalid_argument);
  EXPECT_THROW(m::CnfInverse(p,x),std::invalid_argument);EXPECT_TRUE(n::equal_bits(p.at("log_scale"),bad));}
 p.set("log_scale",n::zeros({2}));p.set("field/hidden/W",n::full({3,8},INFINITY));EXPECT_THROW(m::CnfField(p,x,.5f),std::invalid_argument);
 n::ParameterStore q(10003,n::DType::Float64);m::CnfField(q,Points(),.5f);q.set("field/hidden/W",n::full({3,8},1e300,n::DType::Float64));EXPECT_THROW(m::CnfField(q,n::full({1,2},1e300,n::DType::Float64),.5f),std::invalid_argument);
}
