#include <gtest/gtest.h>
#include <dm/dm.hpp>
namespace n=dm::nn;namespace m=n::models;
TEST(NnGlowDensity, CalibrationKnownMomentsPopulationRidgeFinalCastAndStoppedGraphs) {
 auto x=n::from_vector(std::vector<float>{1,2,3,6},{2,2}).requires_grad_();auto fit=m::FlowActNormCalibration(x);
 EXPECT_EQ(fit.mean.to_vector<double>(),(std::vector<double>{2,4}));EXPECT_EQ(fit.variance.to_vector<double>(),(std::vector<double>{1,4}));
 EXPECT_EQ(fit.bias.to_vector<float>(),(std::vector<float>{-2,-4}));EXPECT_EQ(fit.log_scale.dtype(),n::DType::Float32);
 EXPECT_EQ(fit.log_scale.to_vector<float>()[0],float(-.5*std::log(1+1e-6)));
 for(auto value:{fit.mean,fit.variance,fit.bias,fit.log_scale})EXPECT_FALSE(value.requires_grad());
 auto strided=n::transpose(n::transpose(x,{1,0}).contiguous(),{1,0});auto second=m::FlowActNormCalibration(strided);
 EXPECT_TRUE(n::equal_bits(fit.mean,second.mean));EXPECT_TRUE(n::equal_bits(fit.log_scale,second.log_scale));
 auto constant=m::FlowActNormCalibration(n::full({1,3},7,n::DType::Float64));
 EXPECT_EQ(constant.variance.to_vector<double>(),(std::vector<double>{0,0,0}));
 EXPECT_NEAR(constant.log_scale.to_vector<double>()[0],-.5*std::log(1e-6),1e-15);
}
TEST(NnGlowDensity, ActNormHasNoAutomaticCalibrationAndExplicitInstallationIsImmutable) {
 n::ParameterStore p(9503,n::DType::Float64);auto x=n::from_vector(std::vector<double>{1,2,3,6},{2,2});
 auto identity=m::FlowActNorm(p,"norm",x);EXPECT_TRUE(n::equal_bits(identity.value,x));EXPECT_EQ(identity.logdet.item(),0);
 auto fit=m::FlowActNormCalibration(x);p.set("norm/bias",fit.bias);p.set("norm/log_scale",fit.log_scale);auto before=p.flatten();
 auto out=m::FlowActNorm(p,"norm",x),inverse=m::FlowActNorm(p,"norm",out.value,true);
 EXPECT_LT(n::max(n::abs(inverse.value-x)).item(),2e-15);EXPECT_EQ((inverse.logdet+out.logdet).item(),0);
 EXPECT_LT(n::max(n::abs(n::mean(out.value,{0}))).item(),1e-15);
 EXPECT_EQ(p.count(),4);EXPECT_EQ(p.flatten(),before);
 m::FlowActNorm(p,"norm",n::full({2,2},100,n::DType::Float64));EXPECT_EQ(p.flatten(),before);
}
TEST(NnGlowDensity, SignedMixingLogAbsDetInverseAndWeightGradientMatchClosedForm) {
 n::ParameterStore p(1,n::DType::Float64);auto x=n::from_vector(std::vector<double>{.2,-.7,1.1,.4,-.3,2},{3,2});
 p.set("mix/W",n::from_vector(std::vector<double>{1,2,3,4},{2,2}));auto before=p.flatten();
 auto out=m::FlowMix2D(p,"mix",x),inverse=m::FlowMix2D(p,"mix",out.value,.35,true);
 EXPECT_NEAR(out.logdet.item(),std::log(2.),1e-15);EXPECT_EQ((inverse.logdet+out.logdet).item(),0);
 EXPECT_LT(n::max(n::abs(inverse.value-x)).item(),3e-15);
 auto gradient=n::grad(n::sum(n::square(out.value))-out.logdet,{p.at("mix/W")},n::GradOptions{true,true,false})[0];
 auto data=x.to_vector<double>(),y=out.value.to_vector<double>(),g=gradient.to_vector<double>();
 const std::vector<double> inverse_transpose{-2,1.5,1,-.5};
 for(int i=0;i<2;++i)for(int j=0;j<2;++j){double expected=-inverse_transpose[i*2+j];for(int r=0;r<3;++r)expected+=2*data[r*2+i]*y[r*2+j];EXPECT_NEAR(g[i*2+j],expected,2e-14);}
 auto higher=n::grad(n::sum(gradient),{p.at("mix/W")})[0];EXPECT_GT(n::max(n::abs(higher)).item(),1e-5);EXPECT_EQ(p.flatten(),before);
}
TEST(NnGlowDensity, RotationInitializationRoundsToFloat32ForBothDtypesWithoutRandomDraws) {
 for(auto dtype:{n::DType::Float32,n::DType::Float64}){
  n::ParameterStore p(9503,dtype),reference(9503,dtype);m::FlowMix2D(p,"mix",n::zeros({2,2},dtype),.75);
  auto actual=p.at("mix/W").to_vector<double>();std::vector<float> expected{float(std::cos(.75)),float(-std::sin(.75)),float(std::sin(.75)),float(std::cos(.75))};
  for(int i=0;i<4;++i)EXPECT_EQ(actual[i],double(expected[i]));
  auto a=p.get("random",{9},n::init::normal(.1)),b=reference.get("random",{9},n::init::normal(.1));EXPECT_TRUE(n::equal_bits(a,b));
 }
}
TEST(NnGlowDensity, SequentialCalibrationUsesEachStagesActualCurrentInputAndOwns296Values) {
 n::ParameterStore p(9503);auto calibration=m::BananaPoints(m::GaussianNoise({2},32,9203));
 m::GlowDensity(p,calibration);EXPECT_EQ(p.count(),296);EXPECT_EQ(p.size(),28);
 for(int i=0;i<4;++i){auto out=m::GlowDensity(p,calibration);
  auto input=i==0?out.actnorm_input0:i==1?out.actnorm_input1:i==2?out.actnorm_input2:out.actnorm_input3;
  auto fit=m::FlowActNormCalibration(input);p.set("actnorm"+std::to_string(i)+"/bias",fit.bias);p.set("actnorm"+std::to_string(i)+"/log_scale",fit.log_scale);
 }
 auto out=m::GlowDensity(p,calibration);auto before=p.flatten();
 for(auto value:{out.actnorm_output0,out.actnorm_output1,out.actnorm_output2,out.actnorm_output3}){
  EXPECT_LT(n::max(n::abs(n::mean(value,{0}))).item(),3e-7);EXPECT_LT(n::max(n::abs(n::mean(n::square(value),{0})-1.)).item(),1e-5);}
 auto rec=m::GlowInverse(p,out.latent);EXPECT_LT(n::max(n::abs(rec.value-calibration)).item(),5e-6);
 EXPECT_LT(n::max(n::abs(rec.logdet+out.logdet)).item(),5e-6);EXPECT_EQ(p.flatten(),before);
}
TEST(NnGlowDensity, BothActualJacobiansMatchPerPointLogAbsDetEvenWithNegativeMixing) {
 n::ParameterStore p(9503,n::DType::Float64);auto x=n::from_vector(std::vector<double>{.4,-.7,1.2,.3,-1,2},{3,2}).requires_grad_();
 m::GlowDensity(p,x);p.set("mix0/W",n::from_vector(std::vector<double>{1,2,3,4},{2,2}));auto path=m::GlowDensity(p,x);
 auto z=n::from_vector(std::vector<double>{.8,-1.2,-2,.9},{2,2}).requires_grad_();auto inverse=m::GlowInverse(p,z);
 for(auto entry:{std::make_tuple(path.latent,path.logdet,x),std::make_tuple(inverse.value,inverse.logdet,z)}){
  auto value=std::get<0>(entry),ld=std::get<1>(entry),input=std::get<2>(entry);
  auto a=n::grad(n::sum(n::slice_axis(value,1,0,1)),{input},n::GradOptions{true,false,false})[0].to_vector<double>();
  auto b=n::grad(n::sum(n::slice_axis(value,1,1,2)),{input})[0].to_vector<double>();
  for(int r=0;r<input.size(0);++r){double det=a[r*2]*b[r*2+1]-a[r*2+1]*b[r*2];EXPECT_LT(det,0);EXPECT_NEAR(std::log(std::abs(det)),ld.to_vector<double>()[r],3e-15);}
 }
}
TEST(NnGlowDensity, MeanNllPartialsMatchAll28LeafFiniteDifferencesAndRetainHigherPaths) {
 n::ParameterStore p(9503,n::DType::Float64);auto x=n::from_vector(std::vector<double>{.4,-.7,1.2,.3,-1,2},{3,2}).requires_grad_();
 auto path=m::GlowDensity(p,x);auto loss=m::NormalFlowDensity(path.latent,path.logdet).loss;
 auto gradients=n::grad(loss,p.parameters(),n::GradOptions{true,true,false});
 auto higher=n::grad(n::sum(gradients[0]),{x})[0];EXPECT_GT(n::max(n::abs(higher)).item(),1e-8);
 auto names=p.names();auto parameters=p.parameters();ASSERT_EQ(names.size(),28);
 for(std::size_t i=0;i<names.size();++i){
  auto original=parameters[i].detach().clone();auto hi=original.to_vector<double>(),lo=hi;hi[0]+=1e-5;lo[0]-=1e-5;
  p.set(names[i],n::from_vector(hi,original.shape()));auto a=m::GlowDensity(p,x);double high=m::NormalFlowDensity(a.latent,a.logdet).loss.item();
  p.set(names[i],n::from_vector(lo,original.shape()));auto b=m::GlowDensity(p,x);double low=m::NormalFlowDensity(b.latent,b.logdet).loss.item();p.set(names[i],original);
  EXPECT_NEAR((high-low)/2e-5,gradients[i].to_vector<double>()[0],3e-9)<<names[i];
 }
}
TEST(NnGlowDensity, StridedInputsAndOwningActNormMixAndScaleGraphsSurviveCollection) {
 auto x=n::from_vector(std::vector<double>{.1,.2,.3,.4,.5,.6},{3,2}).requires_grad_();n::Tensor norm,mix,scale;
 {n::ParameterStore p(9503,n::DType::Float64);auto first=m::GlowDensity(p,x);auto before=p.flatten();
  auto strided=n::transpose(n::transpose(x,{1,0}).contiguous(),{1,0});auto second=m::GlowDensity(p,strided);
  EXPECT_TRUE(n::equal_bits(first.latent,second.latent));EXPECT_TRUE(n::equal_bits(first.logdet,second.logdet));
  norm=second.actnorm_output0;mix=second.mixed2;scale=second.coupling_scale3;EXPECT_EQ(p.flatten(),before);}
 for(auto value:{norm,mix,scale})EXPECT_TRUE(n::grad(n::sum(value),{x},n::GradOptions{true,false,false})[0].defined());
}
TEST(NnGlowDensity, InvalidMatricesNamesAnglesAndCalibrationOptionsRejectBeforeLeafCreation) {
 n::ParameterStore p(9503);
 for(auto x:{n::Tensor(),n::zeros({2}),n::zeros({1,3}),n::zeros({0,2}),n::zeros({1,2},n::DType::Int32),n::full({1,2},NAN),n::zeros({1,2},n::DType::Float64)}){
  EXPECT_THROW(m::GlowDensity(p,x),std::invalid_argument);
  EXPECT_THROW(m::GlowInverse(p,x),std::invalid_argument);EXPECT_EQ(p.size(),0);}
 EXPECT_THROW(m::FlowActNorm(p,"",n::zeros({2,2})),std::invalid_argument);
 EXPECT_THROW(m::FlowMix2D(p,"",n::zeros({2,2})),std::invalid_argument);
 EXPECT_THROW(m::FlowMix2D(p,"mix",n::zeros({2,2}),NAN),std::invalid_argument);EXPECT_EQ(p.size(),0);
 for(double epsilon:{0.,-1.,double(INFINITY),double(NAN)})EXPECT_THROW(m::FlowActNormCalibration(n::zeros({2,2}),epsilon),std::invalid_argument);
 EXPECT_THROW(m::FlowActNormCalibration(n::zeros({0,2})),std::invalid_argument);
}
TEST(NnGlowDensity, SingularNonfiniteAndUnrepresentableDeterminantsOrScalesRejectWithoutWrites) {
 n::ParameterStore p(9503);auto x=n::zeros({2,2});m::GlowDensity(p,x);
 for(auto values:{std::vector<float>{1,2,2,4},std::vector<float>{1e30f,0,0,1e30f},std::vector<float>{1e-30f,0,0,1e-30f},std::vector<float>{INFINITY,0,0,1}}){
  auto matrix=n::from_vector(values,{2,2});p.set("mix0/W",matrix);
  EXPECT_THROW(m::GlowDensity(p,x),std::invalid_argument);
  EXPECT_THROW(m::GlowInverse(p,x),std::invalid_argument);EXPECT_TRUE(n::equal_bits(p.at("mix0/W"),matrix));}
 p.set("mix0/W",n::from_vector(std::vector<float>{1,0,0,1},{2,2}));
 for(double value:{1000.,-1000.,double(NAN)}){
  p.set("actnorm0/log_scale",n::full({2},value));EXPECT_THROW(m::GlowDensity(p,x),std::invalid_argument);
  EXPECT_THROW(m::GlowInverse(p,x),std::invalid_argument);EXPECT_TRUE(n::equal_bits(p.at("actnorm0/log_scale"),n::full({2},value)));}
}
