#include <gtest/gtest.h>
#include <dm/dm.hpp>
namespace n=dm::nn;namespace m=n::models;
namespace {
n::Tensor Scalar(double value){return n::full({},value,n::DType::Float64);}
n::Tensor From(const std::vector<double>& values,const n::Shape& shape){return n::reshape(n::from_vector(values,{std::int64_t(values.size())}),shape);}
void CheckJacobian(const n::Tensor& value,const n::Tensor& ld,const n::Tensor& input,double tolerance=2e-12){
 auto keep=n::GradOptions{true,false,false};
 auto a=n::grad(n::sum(n::slice_axis(value,1,0,1)),{input},keep)[0].to_vector<double>();
 auto b=n::grad(n::sum(n::slice_axis(value,1,1,2)),{input},keep)[0].to_vector<double>();
 for(int r=0;r<input.size(0);++r){double determinant=a[r*2]*b[r*2+1]-a[r*2+1]*b[r*2];EXPECT_GT(determinant,0);EXPECT_NEAR(std::log(determinant),ld.to_vector<double>()[r],tolerance);}
}
}
TEST(NnPlanarRadialDensity, BothFamiliesSourceCensusRoundtripAndStageMetadata){
 for(bool radial:{false,true})for(auto dtype:{n::DType::Float32,n::DType::Float64}){
  n::ParameterStore p(9903,dtype);auto x=n::cast(n::from_vector(std::vector<double>{.4,-.7,1.2,.3,-1,2},{3,2}),dtype);
  auto path=m::PlanarRadialDensity(p,x,radial);auto inverse=m::PlanarRadialInverse(p,path.latent,radial);
  EXPECT_EQ(p.count(),radial?18:22);EXPECT_EQ(p.size(),radial?13:21);double tolerance=dtype==n::DType::Float32?3e-6:2e-13;
  EXPECT_LT(n::max(n::abs(inverse.value-x)).item(),tolerance);EXPECT_LT(n::max(n::abs(inverse.logdet+path.logdet)).item(),tolerance);
  EXPECT_TRUE(n::equal_bits(path.unscaled,path.flow3));EXPECT_EQ(path.flow_factor2.shape(),(n::Shape{3,1}));EXPECT_EQ(path.flow_coordinate1.shape(),(n::Shape{3,1}));
  for(auto factor:{path.flow_factor0,path.flow_factor1,path.flow_factor2,path.flow_factor3})EXPECT_GT(n::min(factor).item(),0);
 }
}
TEST(NnPlanarRadialDensity, SourcePlanarAnglesAndAlternatingPerpendicularsPreserveFloatArithmetic){
 n::ParameterStore p(9903,n::DType::Float64);m::PlanarRadialDensity(p,n::zeros({1,2},n::DType::Float64));
 for(int i=0;i<4;++i){auto prefix="flow"+std::to_string(i)+"/";EXPECT_EQ(p.at(prefix+"angle").item(),double(.4f+.65f*i));EXPECT_EQ(p.at(prefix+"perpendicular").item(),double(.02f*(i%2?-1:1)));}
}
TEST(NnPlanarRadialDensity, BothActualWholeMapJacobiansMatchOwnSpatialLogdets){
 for(bool radial:{false,true}){
  n::ParameterStore p(9903,n::DType::Float64);auto x=n::from_vector(std::vector<double>{.4,-.7,1.2,.3,-1,2},{3,2}).requires_grad_();m::PlanarRadialDensity(p,x,radial);
  p.set("log_scale",n::from_vector(std::vector<double>{.2,-.3},{2}));auto path=m::PlanarRadialDensity(p,x,radial);
  auto z=n::from_vector(std::vector<double>{-.8,1.2,.4,-1.3},{2,2}).requires_grad_();auto inverse=m::PlanarRadialInverse(p,z,radial);
  CheckJacobian(path.latent,path.logdet,x);CheckJacobian(inverse.value,inverse.logdet,z);
  EXPECT_GT((n::max(path.logdet)-n::min(path.logdet)).item(),1e-6);
  auto rec=m::PlanarRadialDensity(p,inverse.value,radial);EXPECT_LT(n::max(n::abs(rec.logdet+inverse.logdet)).item(),2e-12);
 }
}
TEST(NnPlanarRadialDensity, MeanNllAll34LeafFiniteDifferencesAndHigherInputPaths){
 for(bool radial:{false,true}){
  n::ParameterStore p(9903,n::DType::Float64);auto x=n::from_vector(std::vector<double>{.4,-.7,1.2,.3,-1,2},{3,2}).requires_grad_();auto path=m::PlanarRadialDensity(p,x,radial);
  auto gradients=n::grad(m::NormalFlowDensity(path.latent,path.logdet).loss,p.parameters(),n::GradOptions{true,true,false});
  auto higher=n::grad(n::sum(gradients[0]),{x})[0];EXPECT_GT(n::max(n::abs(higher)).item(),1e-10);
  auto names=p.names();auto parameters=p.parameters();ASSERT_EQ(names.size(),radial?13:21);
  for(std::size_t i=0;i<names.size();++i){auto original=parameters[i].detach().clone();auto hi=original.to_vector<double>(),lo=hi,g=gradients[i].to_vector<double>();
   auto ix=std::max_element(g.begin(),g.end(),[](double a,double b){return std::abs(a)<std::abs(b);})-g.begin();hi[ix]+=1e-5;lo[ix]-=1e-5;
   p.set(names[i],From(hi,original.shape()));auto a=m::PlanarRadialDensity(p,x,radial);double high=m::NormalFlowDensity(a.latent,a.logdet).loss.item();
   p.set(names[i],From(lo,original.shape()));auto b=m::PlanarRadialDensity(p,x,radial);double low=m::NormalFlowDensity(b.latent,b.logdet).loss.item();p.set(names[i],original);
   EXPECT_NEAR((high-low)/2e-5,g[ix],3e-9)<<names[i];
  }
 }
}
TEST(NnPlanarRadialDensity, InverseObjectiveAll40ParameterCoordinatesMatchFiniteDifferences){
 for(bool radial:{false,true}){
  n::ParameterStore p(9903,n::DType::Float64);auto z=n::from_vector(std::vector<double>{.4,-.7,1.2,.3,-1,2},{3,2});m::PlanarRadialDensity(p,z,radial);
  auto objective=[&](){auto inv=m::PlanarRadialInverse(p,z,radial);return n::mean(n::sum(n::square(inv.value),{1}),{0})+.3*n::mean(inv.logdet,{0});};
  auto gradients=n::grad(objective(),p.parameters());auto names=p.names();auto parameters=p.parameters();
  for(std::size_t i=0;i<names.size();++i){auto original=parameters[i].detach().clone();auto values=original.to_vector<double>(),g=gradients[i].to_vector<double>();
   for(std::size_t ix=0;ix<values.size();++ix){auto hi=values,lo=values;hi[ix]+=1e-5;lo[ix]-=1e-5;
    p.set(names[i],From(hi,original.shape()));double high=objective().item();p.set(names[i],From(lo,original.shape()));double low=objective().item();p.set(names[i],original);
    EXPECT_NEAR((high-low)/2e-5,g[ix],5e-9)<<names[i]<<":"<<ix;}
  }
 }
}
TEST(NnPlanarRadialDensity, PureStrongPlanarContractionHingeAndBothJacobians){
 auto x=n::from_vector(std::vector<double>{-.1*std::cos(.6)/std::exp(.3),-.1*std::sin(.6)/std::exp(.3),0,0,1,0,0,1,-1,-1,3,-2,-3,2,10,-10,-10,10},{9,2}).requires_grad_();
 auto angle=Scalar(.6),frequency=Scalar(.3),projection=Scalar(-6),q=Scalar(.7),bias=Scalar(.1);
 auto f=m::PlanarFlow(x,angle,frequency,projection,q,bias),iv=m::PlanarFlow(x,angle,frequency,projection,q,bias,true),rec=m::PlanarFlow(f.value,angle,frequency,projection,q,bias,true);
 EXPECT_LT(n::max(n::abs(rec.value-x)).item(),2e-14);EXPECT_NEAR(f.coordinate.to_vector<double>()[0],0.,5e-17);EXPECT_LT(f.factor.to_vector<double>()[0],.11);
 CheckJacobian(f.value,f.logdet,x);CheckJacobian(iv.value,iv.logdet,x,2e-9);
}
TEST(NnPlanarRadialDensity, PurePlanarInputAndAllScalarPartialsThroughBothDirections){
 auto x=n::from_vector(std::vector<double>{.4,-.7,1.2,.3,-1,2},{3,2}).requires_grad_();std::vector<n::Tensor> parameters;
 for(double value:{.6,.3,-1.,.7,.1})parameters.push_back(Scalar(value).requires_grad_());
 auto objective=[&](const n::Tensor& points,const std::vector<n::Tensor>& p){auto a=m::PlanarFlow(points,p[0],p[1],p[2],p[3],p[4]),b=m::PlanarFlow(points,p[0],p[1],p[2],p[3],p[4],true);return n::sum(n::square(a.value))-.2*n::sum(a.logdet)+.3*(n::sum(n::square(b.value))-n::sum(b.logdet));};
 auto inputs=parameters;inputs.insert(inputs.begin(),x);auto g=n::grad(objective(x,parameters),inputs,n::GradOptions{true,true,false});EXPECT_TRUE(n::grad(n::sum(g[1]),{x})[0].defined());
 auto values=x.to_vector<double>();for(int i=0;i<6;++i){auto hi=values,lo=values;hi[i]+=1e-5;lo[i]-=1e-5;EXPECT_NEAR((objective(n::from_vector(hi,{3,2}),parameters).item()-objective(n::from_vector(lo,{3,2}),parameters).item())/2e-5,g[0].to_vector<double>()[i],2e-8);}
 for(int i=0;i<5;++i){auto hi=parameters,lo=parameters;hi[i]=Scalar(parameters[i].item()+1e-5);lo[i]=Scalar(parameters[i].item()-1e-5);EXPECT_NEAR((objective(x,hi).item()-objective(x,lo).item())/2e-5,g[i+1].item(),2e-8);}
}
TEST(NnPlanarRadialDensity, PlanarInverseHigherGraphFollowsStoppedCorrectionRatherThanExactImplicitHessian){
 auto x=n::from_vector(std::vector<double>{.3,-.2},{1,2});auto bias=Scalar(.1).requires_grad_();auto projection=Scalar(-1.);
 auto inv=m::PlanarFlow(x,Scalar(0),Scalar(0),projection,Scalar(0),bias,true);auto keep=n::GradOptions{true,true,false};
 auto first=n::grad(n::sum(n::slice_axis(inv.value,1,0,1)),{bias},keep)[0];auto second=n::grad(first,{bias})[0];
 double p=std::log1p(std::exp(-1.))-.9,t=inv.coordinate.item(),a=std::tanh(t),sech=1-a*a;
 double low=.4-(std::abs(p)+1),high=.4+(std::abs(p)+1);
 for(int i=0;i<32;++i){double mid=.5*(low+high);if((mid+p*std::tanh(mid))-.4<0)low=mid;else high=mid;}
 double root=.5*(low+high),den=1+p*(1-std::pow(std::tanh(root),2));
 EXPECT_NEAR(first.item(),-p*sech/den,2e-13);EXPECT_NEAR(second.item(),2*p*a*sech/(den*den),2e-12);
 double true_second=2*p*a*sech/(den*den)-p*sech*(2*p*a*sech/(den*den*den));EXPECT_GT(std::abs(second.item()-true_second),1e-4);
}
TEST(NnPlanarRadialDensity, PureRadialStrongContractionBothQuadraticBranchesAndFiniteCenterJacobians){
 auto center=n::from_vector(std::vector<double>{.3,-.2},{2});auto x=n::from_vector(std::vector<double>{.3,-.2,.31,-.19,0,0,1,0,0,1,10,-10,-10,10},{7,2}).requires_grad_();
 auto alpha=Scalar(.4),beta=Scalar(std::log(std::expm1(.12)));auto f=m::RadialFlow(x,center,alpha,beta),iv=m::RadialFlow(x,center,alpha,beta,true),rec=m::RadialFlow(f.value,center,alpha,beta,true);
 EXPECT_LT(n::max(n::abs(rec.value-x)).item(),3e-14);EXPECT_EQ(f.coordinate.to_vector<double>()[0],0);EXPECT_EQ(iv.coordinate.to_vector<double>()[0],0);
 CheckJacobian(f.value,f.logdet,x);CheckJacobian(iv.value,iv.logdet,x);
 auto keep=n::GradOptions{true,false,false};auto row=n::grad(n::sum(n::slice_axis(f.value,1,0,1)),{x},keep)[0].to_vector<double>();
 EXPECT_NEAR(row[0],f.factor.to_vector<double>()[0],1e-15);EXPECT_EQ(row[1],0);
 auto radial_gradient=n::grad(n::sum(f.coordinate),{x},keep)[0].to_vector<double>();EXPECT_EQ(radial_gradient[0],0);EXPECT_EQ(radial_gradient[1],0);
 auto ld_gradient=n::grad(n::sum(f.logdet),{x},keep)[0].to_vector<double>();EXPECT_EQ(ld_gradient[0],0);EXPECT_EQ(ld_gradient[1],0);
}
TEST(NnPlanarRadialDensity, PureRadialAllInputCenterAndShapeParameterPartialsInBothDirections){
 auto x=n::from_vector(std::vector<double>{.4,-.7,1.2,.3,-1,2},{3,2}).requires_grad_();auto center=n::from_vector(std::vector<double>{.1,-.2},{2}).requires_grad_();auto alpha=Scalar(.4).requires_grad_(),beta=Scalar(-1.).requires_grad_();
 auto objective=[](const n::Tensor& x,const n::Tensor& c,const n::Tensor& a,const n::Tensor& b){auto f=m::RadialFlow(x,c,a,b),i=m::RadialFlow(x,c,a,b,true);return n::sum(n::square(f.value))-.2*n::sum(f.logdet)+.3*(n::sum(n::square(i.value))-n::sum(i.logdet));};
 auto g=n::grad(objective(x,center,alpha,beta),{x,center,alpha,beta},n::GradOptions{true,true,false});EXPECT_TRUE(n::grad(n::sum(g[1]),{x})[0].defined());
 std::vector<n::Tensor> inputs{x,center,alpha,beta};for(int k=0;k<4;++k){auto values=inputs[k].to_vector<double>();for(std::size_t i=0;i<values.size();++i){auto hv=values,lv=values;hv[i]+=1e-5;lv[i]-=1e-5;auto hi=inputs,lo=inputs;hi[k]=From(hv,inputs[k].shape());lo[k]=From(lv,inputs[k].shape());double fd=(objective(hi[0],hi[1],hi[2],hi[3]).item()-objective(lo[0],lo[1],lo[2],lo[3]).item())/2e-5;EXPECT_NEAR(fd,g[k].to_vector<double>()[i],2e-8);}}
}
TEST(NnPlanarRadialDensity, StridesWarmImmutabilityAndNoAdditionalRngDraws){
 for(bool radial:{false,true}){
  n::ParameterStore p(9903,n::DType::Float64),other(9903,n::DType::Float64);auto x=n::from_vector(std::vector<double>{.4,-.7,1.2,.3,-1,2},{3,2});auto first=m::PlanarRadialDensity(p,x,radial);m::PlanarRadialDensity(other,x,radial);auto before=p.flatten();
  auto strided=n::transpose(n::transpose(x,{1,0}).contiguous(),{1,0});auto second=m::PlanarRadialDensity(p,strided,radial);EXPECT_TRUE(n::equal_bits(first.latent,second.latent));EXPECT_TRUE(n::equal_bits(first.logdet,second.logdet));
  m::PlanarRadialInverse(p,first.latent,radial);m::PlanarRadialInverse(p,first.latent,radial);EXPECT_EQ(p.flatten(),before);
  EXPECT_TRUE(n::equal_bits(p.get("extra",{7},n::init::normal(.1)),other.get("extra",{7},n::init::normal(.1))));
 }
}
TEST(NnPlanarRadialDensity, StageFactorCoordinateAndInverseGraphsSurviveParentAndStoreCollection){
 for(bool radial:{false,true}){
  auto x=n::from_vector(std::vector<double>{.4,-.7,1.2,.3},{2,2}).requires_grad_();n::Tensor factor,coordinate,stage,inverse;
  {n::ParameterStore p(9903,n::DType::Float64);auto path=m::PlanarRadialDensity(p,x,radial);factor=path.flow_factor2;coordinate=path.flow_coordinate1;stage=path.flow3;inverse=m::PlanarRadialInverse(p,path.latent,radial).value;}
  for(auto field:{factor,coordinate,stage,inverse}){auto keep=n::GradOptions{true,true,false};auto g=n::grad(n::sum(field),{x},keep)[0];EXPECT_TRUE(g.defined());EXPECT_TRUE(n::grad(n::sum(n::square(g)),{x},keep)[0].defined());}
 }
}
TEST(NnPlanarRadialDensity, InvalidShapesDtypesAndBlocksRejectBeforeLeafCreation){
 n::ParameterStore p(9903);for(auto x:{n::Tensor(),n::zeros({2}),n::zeros({1,3}),n::zeros({0,2}),n::zeros({1,2},n::DType::Int32),n::full({1,2},NAN),n::zeros({1,2},n::DType::Float64)}){
  EXPECT_THROW(m::PlanarRadialDensity(p,x),std::invalid_argument);
  EXPECT_THROW(m::PlanarRadialInverse(p,x,true),std::invalid_argument);EXPECT_EQ(p.size(),0);}
 for(int i:{-1,4}){EXPECT_THROW(m::ElementaryFlow(p,n::zeros({1,2}),i),std::invalid_argument);}
 EXPECT_EQ(p.size(),0);auto x=n::ones({1,2},n::DType::Float64),zero=Scalar(0);
 EXPECT_THROW(m::PlanarFlow(x,n::zeros({1},n::DType::Float64),zero,zero,zero,zero),std::invalid_argument);
 EXPECT_THROW(m::PlanarFlow(x,n::full({},0.),zero,zero,zero,zero),std::invalid_argument);
 EXPECT_THROW(m::RadialFlow(x,n::zeros({1},n::DType::Float64),zero,zero),std::invalid_argument);
 EXPECT_THROW(m::RadialFlow(x,n::zeros({2},n::DType::Float64),n::full({},0.),zero),std::invalid_argument);
}
TEST(NnPlanarRadialDensity, UnrepresentableFrequencyRadiusProjectionAndGlobalScaleReject){
 auto x=n::ones({2,2},n::DType::Float64),zero=Scalar(0),center=n::zeros({2},n::DType::Float64);
 for(double v:{1000.,-1000.,double(NAN)}){EXPECT_THROW(m::PlanarFlow(x,zero,Scalar(v),zero,zero,zero),std::invalid_argument);}
 EXPECT_THROW(m::PlanarFlow(x,zero,zero,Scalar(1e300),zero,zero,true),std::invalid_argument);
 EXPECT_THROW(m::RadialFlow(x,center,zero,Scalar(-1000.)),std::invalid_argument);
 EXPECT_THROW(m::RadialFlow(n::full({2,2},1e300,n::DType::Float64),center,zero,zero),std::invalid_argument);
 for(bool radial:{false,true}){n::ParameterStore p(9903,n::DType::Float64);m::PlanarRadialDensity(p,x,radial);auto bad=n::full({2},1000,n::DType::Float64);p.set("log_scale",bad);
  EXPECT_THROW(m::PlanarRadialDensity(p,x,radial),std::invalid_argument);
  EXPECT_THROW(m::PlanarRadialInverse(p,x,radial),std::invalid_argument);EXPECT_TRUE(n::equal_bits(p.at("log_scale"),bad));}
}
