#include <gtest/gtest.h>
#include <dm/dm.hpp>
namespace n=dm::nn;namespace m=n::models;
TEST(NnSplineDensity, SourceCensusAndAnalyticInverseRecoverBothAlternatingOrders) {
 for(auto dtype:{n::DType::Float32,n::DType::Float64}){
  n::ParameterStore p(9803,dtype);auto x=n::cast(n::from_vector(std::vector<double>{.2,-.7,1.1,.4,-.3,2},{3,2}),dtype);
  auto path=m::SplineDensity(p,x);auto inverse=m::SplineInverse(p,path.latent);EXPECT_EQ(p.count(),878);EXPECT_EQ(p.size(),17);
  double tolerance=dtype==n::DType::Float32?5e-6:2e-13;
  EXPECT_LT(n::max(n::abs(inverse.value-x)).item(),tolerance);
  EXPECT_LT(n::max(n::abs(inverse.logdet+path.logdet)).item(),tolerance);
  EXPECT_TRUE(n::equal_bits(path.unscaled,path.coupling3));
  for(auto widths:{path.coupling_widths0,path.coupling_widths1,path.coupling_widths2,path.coupling_widths3})
   EXPECT_LT(n::max(n::abs(n::sum(widths,{1})-8.)).item(),dtype==n::DType::Float32?2e-6:2e-14);
 }
}
TEST(NnSplineDensity, BothActualJacobiansAndSpatialDeterminantsMatchTheirOwnPaths) {
 n::ParameterStore p(9803,n::DType::Float64);
 auto x=n::from_vector(std::vector<double>{.2,-.7,1.1,.4,-.3,2},{3,2}).requires_grad_();auto path=m::SplineDensity(p,x);
 p.set("log_scale",n::from_vector(std::vector<double>{.2,-.3},{2}));path=m::SplineDensity(p,x);
 auto z=n::from_vector(std::vector<double>{-.8,1.2,.4,-1.3},{2,2}).requires_grad_();auto inverse=m::SplineInverse(p,z);
 for(auto entry:{std::make_tuple(path.latent,path.logdet,x),std::make_tuple(inverse.value,inverse.logdet,z)}){
  auto value=std::get<0>(entry),ld=std::get<1>(entry),input=std::get<2>(entry);
  auto a=n::grad(n::sum(n::slice_axis(value,1,0,1)),{input},n::GradOptions{true,false,false})[0].to_vector<double>();
  auto b=n::grad(n::sum(n::slice_axis(value,1,1,2)),{input})[0].to_vector<double>();
  for(int r=0;r<input.size(0);++r){double det=a[r*2]*b[r*2+1]-a[r*2+1]*b[r*2];EXPECT_GT(det,0);EXPECT_NEAR(std::log(det),ld.to_vector<double>()[r],3e-14);}
 }
 EXPECT_GT((n::max(path.logdet)-n::min(path.logdet)).item(),1e-5);
 auto density=m::SplineDensity(p,inverse.value);EXPECT_LT(n::max(n::abs(density.logdet+inverse.logdet)).item(),3e-14);
}
TEST(NnSplineDensity, MeanNllMatchesAll17LeafFiniteDifferencesAndRetainsHigherPaths) {
 n::ParameterStore p(9803,n::DType::Float64);auto x=n::from_vector(std::vector<double>{.4,-.7,1.2,.3,-1,2},{3,2}).requires_grad_();
 auto path=m::SplineDensity(p,x);auto loss=m::NormalFlowDensity(path.latent,path.logdet).loss;
 auto gradients=n::grad(loss,p.parameters(),n::GradOptions{true,true,false});
 auto higher=n::grad(n::sum(gradients[0]),{x})[0];EXPECT_GT(n::max(n::abs(higher)).item(),1e-8);
 auto names=p.names();auto parameters=p.parameters();ASSERT_EQ(names.size(),17);
 for(std::size_t i=0;i<names.size();++i){auto original=parameters[i].detach().clone();auto hi=original.to_vector<double>(),lo=hi;auto g=gradients[i].to_vector<double>();
  auto at=std::max_element(g.begin(),g.end(),[](double a,double b){return std::abs(a)<std::abs(b);})-g.begin();hi[at]+=1e-5;lo[at]-=1e-5;
  p.set(names[i],n::from_vector(hi,original.shape()));auto a=m::SplineDensity(p,x);double high=m::NormalFlowDensity(a.latent,a.logdet).loss.item();
  p.set(names[i],n::from_vector(lo,original.shape()));auto b=m::SplineDensity(p,x);double low=m::NormalFlowDensity(b.latent,b.logdet).loss.item();p.set(names[i],original);
  EXPECT_NEAR((high-low)/2e-5,g[at],3e-9)<<names[i];
 }
}
TEST(NnSplineDensity, StridesWarmImmutabilityAndNoAdditionalRngDraws) {
 n::ParameterStore p(9803,n::DType::Float64),other(9803,n::DType::Float64);
 auto x=n::from_vector(std::vector<double>{.1,.2,.3,.4,.5,.6},{3,2});auto first=m::SplineDensity(p,x);m::SplineDensity(other,x);auto before=p.flatten();
 auto strided=n::transpose(n::transpose(x,{1,0}).contiguous(),{1,0});auto second=m::SplineDensity(p,strided);
 EXPECT_TRUE(n::equal_bits(first.latent,second.latent));EXPECT_TRUE(n::equal_bits(first.logdet,second.logdet));
 auto inverse=m::SplineInverse(p,first.latent);m::SplineInverse(p,first.latent);EXPECT_EQ(p.flatten(),before);
 EXPECT_TRUE(n::equal_bits(p.get("extra",{7},n::init::normal(.1)),other.get("extra",{7},n::init::normal(.1))));
}
TEST(NnSplineDensity, OwningStageConditionerAndInverseGraphsSurviveParentAndStoreCollection) {
 auto x=n::from_vector(std::vector<double>{.1,.2,.3,.4},{2,2}).requires_grad_();n::Tensor mean,scale,stage,inverse;
 {n::ParameterStore p(9803,n::DType::Float64);auto path=m::SplineDensity(p,x);
  mean=n::slice_axis(path.coupling_heights1,1,0,1);scale=n::slice_axis(path.coupling_widths3,1,0,1);stage=path.coupling2;inverse=m::SplineInverse(p,path.latent).value;}
 for(auto value:{mean,scale,stage,inverse})EXPECT_TRUE(n::grad(n::sum(value),{x},n::GradOptions{true,true,false})[0].defined());
}
TEST(NnSplineDensity, InvalidInputMaskOptionsAndDtypesRejectBeforeLeafCreation) {
 n::ParameterStore p(9803);
 for(auto x:{n::Tensor(),n::zeros({2}),n::zeros({1,3}),n::zeros({0,2}),n::zeros({1,2},n::DType::Int32),n::full({1,2},NAN),n::zeros({1,2},n::DType::Float64)}){
  EXPECT_THROW(m::SplineDensity(p,x),std::invalid_argument);
  EXPECT_THROW(m::SplineInverse(p,x),std::invalid_argument);
  EXPECT_EQ(p.size(),0);}
 for(int block:{-1,4}){EXPECT_THROW(m::SplineCoupling(p,n::zeros({2,2}),block),std::invalid_argument);}
 EXPECT_EQ(p.size(),0);
}
TEST(NnSplineDensity, NonfiniteStoredValuesAndUnrepresentableGlobalScalesRejectWithoutWrites) {
 n::ParameterStore p(9803);auto x=n::zeros({2,2});m::SplineDensity(p,x);
 for(double value:{1000.,-1000.,double(NAN)}){auto scale=n::full({2},value);p.set("log_scale",scale);
  EXPECT_THROW(m::SplineDensity(p,x),std::invalid_argument);
  EXPECT_THROW(m::SplineInverse(p,x),std::invalid_argument);EXPECT_TRUE(n::equal_bits(p.at("log_scale"),scale));}
 p.set("log_scale",n::zeros({2}));auto bad=n::full({16,11},INFINITY);p.set("coupling0/heads/W",bad);
 EXPECT_THROW(m::SplineDensity(p,x),std::invalid_argument);
 EXPECT_THROW(m::SplineInverse(p,x),std::invalid_argument);EXPECT_TRUE(n::equal_bits(p.at("coupling0/heads/W"),bad));
}
TEST(NnSplineDensity, PureSplineGenericBinsBoundsIdentityAndNonuniformHeadPartials) {
 for(auto dtype:{n::DType::Float32,n::DType::Float64})for(int bins:{2,4,7})for(double bound:{2.5,4.}){
  auto x=n::cast(n::from_vector(std::vector<double>{-bound-1,-bound,-1,.13,bound,bound+1},{6,1}),dtype);
  auto heads=n::zeros({6,3*bins-1},dtype);auto f=m::RationalQuadraticSpline(x,heads,false,bound);
  auto inverse=m::RationalQuadraticSpline(f.value,heads,true,bound);double tolerance=dtype==n::DType::Float32?2e-6:8e-15;
  EXPECT_LT(n::max(n::abs(f.value-x)).item(),tolerance);EXPECT_LT(n::max(n::abs(inverse.value-x)).item(),tolerance);
  EXPECT_LT(n::max(n::abs(f.logdet)).item(),tolerance);EXPECT_EQ(f.widths.shape(),(n::Shape{6,bins}));
  EXPECT_EQ(f.derivatives.shape(),(n::Shape{6,bins+1}));EXPECT_GT(n::min(f.widths).item(),0);EXPECT_GT(n::min(f.heights).item(),0);
  EXPECT_EQ(f.value.to_vector<double>()[0],x.to_vector<double>()[0]);EXPECT_EQ(f.value.to_vector<double>()[5],x.to_vector<double>()[5]);
 }
 // Nonuniform public configurations also check every head's local derivative.
 for(int bins:{2,7}){
  auto x=n::from_vector(std::vector<double>{-3,-1.6,-.4,.37,1.9,3},{6,1}).requires_grad_();
  std::vector<double> raw(6*(3*bins-1));for(std::size_t i=0;i<raw.size();++i)raw[i]=.3*std::sin(double(i));
  auto heads=n::from_vector(raw,{6,3*bins-1}).requires_grad_();auto f=m::RationalQuadraticSpline(x,heads,false,2.5,.03,.02);
  auto rec=m::RationalQuadraticSpline(f.value,heads,true,2.5,.03,.02);EXPECT_LT(n::max(n::abs(rec.value-x)).item(),1e-14);
  auto j=n::grad(n::sum(f.value),{x},n::GradOptions{true,false,false})[0];EXPECT_GT(n::min(j).item(),0);
  EXPECT_LT(n::max(n::abs(n::reshape(n::log(j),{-1})-f.logdet)).item(),1e-14);
  auto loss=n::sum(n::square(f.value))-.2*n::sum(f.logdet);auto g=n::grad(loss,{heads})[0].to_vector<double>();
  for(int k=0;k<3*bins-1;++k){auto hi=raw,lo=raw;auto index=2*(3*bins-1)+k;hi[index]+=1e-5;lo[index]-=1e-5;
   auto h=m::RationalQuadraticSpline(x,n::from_vector(hi,{6,3*bins-1}),false,2.5,.03,.02);
   auto l=m::RationalQuadraticSpline(x,n::from_vector(lo,{6,3*bins-1}),false,2.5,.03,.02);
   double fd=((n::sum(n::square(h.value))-.2*n::sum(h.logdet))-(n::sum(n::square(l.value))-.2*n::sum(l.logdet))).item()/2e-5;
   EXPECT_NEAR(fd,g[index],4e-9);
  }
 }
}
TEST(NnSplineDensity, RoundedSourceDerivativeOffsetIsExplicitInDoubleAndNoRngIsUsed) {
 auto x=n::full({2,1},.25,n::DType::Float64),heads=n::zeros({2,11},n::DType::Float64);
 auto precise=m::RationalQuadraticSpline(x,heads),source=m::RationalQuadraticSpline(x,heads,false,4.,.01,.001,true);
 double expected=.001+std::log1p(std::exp(double(float(std::log(std::expm1(.999))))));
 EXPECT_NEAR(source.derivatives.to_vector<double>()[1],expected,2e-16);
 EXPECT_GT(std::abs(source.derivatives.to_vector<double>()[1]-precise.derivatives.to_vector<double>()[1]),1e-10);
 EXPECT_TRUE(n::equal_bits(source.value,m::RationalQuadraticSpline(x,heads,false,4.,.01,.001,true).value));
}
TEST(NnSplineDensity, PureSplinePositiveActualDerivativesMatchLogdetsAndC1TailKnots) {
 auto x=n::from_vector(std::vector<double>{-6,-4,-3.999,-2,0,2,3.999,4,6},{9,1}).requires_grad_();
 std::vector<double> values;for(int i=0;i<9;++i)for(double v:{0.,0.,0.,0.,.2,-.3,.1,.4,-.2,.3,.6})values.push_back(v);
 auto heads=n::from_vector(values,{9,11});auto f=m::RationalQuadraticSpline(x,heads),iv=m::RationalQuadraticSpline(x,heads,true);
 auto keep=n::GradOptions{true,true,false};auto j=n::grad(n::sum(f.value),{x},keep)[0],ij=n::grad(n::sum(iv.value),{x},keep)[0];
 EXPECT_GT(n::min(j).item(),0);EXPECT_GT(n::min(ij).item(),0);
 EXPECT_LT(n::max(n::abs(n::reshape(n::log(j),{-1})-f.logdet)).item(),4e-15);
 EXPECT_LT(n::max(n::abs(n::reshape(n::log(ij),{-1})-iv.logdet)).item(),4e-15);
 auto rec=m::RationalQuadraticSpline(f.value,heads,true);EXPECT_LT(n::max(n::abs(rec.value-x)).item(),5e-15);
 for(int i:{0,1,7,8}){EXPECT_NEAR(j.to_vector<double>()[i],1.,2e-15);EXPECT_EQ(f.value.to_vector<double>()[i],x.to_vector<double>()[i]);}
}
TEST(NnSplineDensity, PureForwardInverseObjectivePartialsMatchInputAndAllHeadFiniteDifferencesAndHigherPaths) {
 std::vector<double> points{-5,-2.4,-.7,.3,1.2,3.6,5},values;for(int i=0;i<7;++i)for(double v:{0.,0.,0.,0.,.2,-.3,.1,.4,-.2,.3,.6})values.push_back(v);
 auto x=n::from_vector(points,{7,1}).requires_grad_(),heads=n::from_vector(values,{7,11}).requires_grad_();
 auto objective=[](const n::Tensor& a,const n::Tensor& b){auto f=m::RationalQuadraticSpline(a,b),iv=m::RationalQuadraticSpline(a,b,true);return n::sum(n::square(f.value))-n::sum(f.logdet)+.3*(n::sum(n::square(iv.value))-n::sum(iv.logdet));};
 auto keep=n::GradOptions{true,true,false};auto gradients=n::grad(objective(x,heads),{x,heads},keep);
 auto higher=n::grad(n::sum(gradients[1]),{x},keep)[0];EXPECT_GT(n::max(n::abs(higher)).item(),1e-5);
 for(int i=0;i<7;++i){auto hi=points,lo=points;hi[i]+=1e-5;lo[i]-=1e-5;
  double fd=(objective(n::from_vector(hi,{7,1}),heads).item()-objective(n::from_vector(lo,{7,1}),heads).item())/2e-5;
  EXPECT_NEAR(fd,gradients[0].to_vector<double>()[i],3e-8);}
 for(int k=0;k<11;++k){auto hi=values,lo=values;hi[11+k]+=1e-5;lo[11+k]-=1e-5;
  double fd=(objective(x,n::from_vector(hi,{7,11})).item()-objective(x,n::from_vector(lo,{7,11})).item())/2e-5;
  EXPECT_NEAR(fd,gradients[1].to_vector<double>()[11+k],3e-8);}
 for(int row:{0,6})for(int k=0;k<11;++k)EXPECT_EQ(gradients[1].to_vector<double>()[row*11+k],0);
}
TEST(NnSplineDensity, CouplingsPreserveRawConditioningCoordinatesAndExactStrictTails) {
 for(int block=0;block<4;++block){n::ParameterStore p(9803,n::DType::Float64);int c=block%2;
  std::vector<double> values=c?std::vector<double>{-6,.4,6,-.3}:std::vector<double>{.4,-6,-.3,6};auto x=n::from_vector(values,{2,2});
  auto f=m::SplineCoupling(p,x,block),iv=m::SplineCoupling(p,f.value,block,true);EXPECT_EQ(p.count(),219);EXPECT_EQ(p.size(),4);
  EXPECT_TRUE(n::equal_bits(f.value,x));EXPECT_TRUE(n::equal_bits(iv.value,x));EXPECT_EQ(n::max(n::abs(f.logdet)).item(),0);
  EXPECT_EQ(f.widths.shape(),(n::Shape{2,4}));EXPECT_EQ(f.derivatives.shape(),(n::Shape{2,5}));
 }
}
TEST(NnSplineDensity, PureSplineAcceptsStridesAndRetainsHeadBinGraphsAfterParentCollection) {
 auto x=n::from_vector(std::vector<double>{-.7,.3,1.1},{3,1}).requires_grad_();auto heads=n::full({3,11},.15,n::DType::Float64).requires_grad_();n::Tensor value,width,derivative;
 {auto strided=n::transpose(n::transpose(heads,{1,0}).contiguous(),{1,0});auto a=m::RationalQuadraticSpline(x,heads),b=m::RationalQuadraticSpline(x,strided);
  EXPECT_TRUE(n::equal_bits(a.value,b.value));EXPECT_TRUE(n::equal_bits(a.logdet,b.logdet));value=b.value;width=n::slice_axis(b.widths,1,0,1);derivative=n::slice_axis(b.derivatives,1,1,2);}
 auto keep=n::GradOptions{true,true,false};
 for(auto field:{value,width,derivative}){auto g=n::grad(n::sum(field),{heads},keep)[0];EXPECT_TRUE(g.defined());EXPECT_TRUE(n::grad(n::sum(n::square(g)),{heads},keep)[0].defined());}
}
TEST(NnSplineDensity, PureInvalidGeometryOptionsAndUnrepresentableBinsOrQuadraticsReject) {
 auto x=n::zeros({2,1}),heads=n::zeros({2,11});
 for(auto bad:{n::Tensor(),n::zeros({0,1}),n::zeros({2}),n::zeros({2,2}),n::zeros({2,1},n::DType::Int32),n::full({2,1},NAN)})EXPECT_THROW(m::RationalQuadraticSpline(bad,heads),std::invalid_argument);
 for(auto bad:{n::zeros({1,11}),n::zeros({2,4}),n::zeros({2,6}),n::zeros({2,11},n::DType::Float64),n::full({2,11},INFINITY)})EXPECT_THROW(m::RationalQuadraticSpline(x,bad),std::invalid_argument);
 for(double bound:{0.,-1.,double(INFINITY),double(NAN)})EXPECT_THROW(m::RationalQuadraticSpline(x,heads,false,bound),std::invalid_argument);
 for(double fraction:{0.,-.01,.25,.4,double(NAN)})EXPECT_THROW(m::RationalQuadraticSpline(x,heads,false,4,fraction),std::invalid_argument);
 for(double floor:{0.,-1.,1.,double(INFINITY)})EXPECT_THROW(m::RationalQuadraticSpline(x,heads,false,4,.01,floor),std::invalid_argument);
 auto d=n::zeros({1,1},n::DType::Float64),h=n::zeros({1,11},n::DType::Float64);
 for(double bound:{1e-200,1e200})EXPECT_THROW(m::RationalQuadraticSpline(d,h,true,bound),std::invalid_argument);
 auto extreme=n::from_vector(std::vector<float>{100,-100,-100,-100,0,0,0,0,0,0,0},{1,11});
 EXPECT_THROW(m::RationalQuadraticSpline(n::zeros({1,1}),extreme,false,4,1e-10),std::invalid_argument);
 EXPECT_THROW(m::RationalQuadraticSpline(x,heads,false,4,1e-46),std::invalid_argument);
}
