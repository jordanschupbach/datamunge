#include <gtest/gtest.h>
#include <dm/dm.hpp>
namespace n=dm::nn;namespace m=n::models;
namespace {
n::Tensor Labels(std::int64_t count){return n::zeros({count},n::DType::Int64);}
double Difference(const n::Tensor& a,const n::Tensor& b){return n::max(n::abs(a-b)).item();}
}
TEST(NnGatedPixels, MasksPreserveBothStreamGeometriesAndRawPixelExclusion) {
 for(bool vertical:{false,true})for(bool first:{false,true}) {
  auto a=m::GatedPixelMask(2,vertical,first,3,n::DType::Float64);
  EXPECT_EQ(a.shape(),(n::Shape{vertical?3:1,3,2,3}));
  auto v=a.to_vector<double>();int ones=0;
  for(int y=0;y<(vertical?3:1);++y)for(int x=0;x<3;++x)for(int c=0;c<6;++c){
   bool allowed=vertical?(y==0||(!first&&y==1)):(x==0||(!first&&x==1));
   EXPECT_EQ(v[(y*3+x)*6+c],allowed?1:0);ones+=allowed;
  }
  EXPECT_EQ(n::sum(a).item(),ones);EXPECT_FALSE(a.requires_grad());
 }
 EXPECT_THROW(m::GatedPixelMask(0,true,true),std::invalid_argument);
 EXPECT_THROW(m::GatedPixelMask(INT64_MAX,false,false),std::invalid_argument);
 EXPECT_THROW(m::GatedPixelMask(1,true,true,8,n::DType::Int64),std::invalid_argument);
}
TEST(NnGatedPixels, GateHasCorrectHalvesAndLiveHigherPaths) {
 auto t=n::DType::Float64;
 auto x=n::tensor({-1.,0.,1.,2.,0.,1.,-1.,2.},{1,1,1,8},t).requires_grad_();
 auto y=m::GatedPixelGate(x);auto values=y.to_vector<double>();
 const double a[4]={-1,0,1,2},b[4]={0,1,-1,2};
 for(int k=0;k<4;++k)EXPECT_NEAR(values[k],std::tanh(a[k])/(1+std::exp(-b[k])),1e-15);
 n::GradOptions graph;graph.create_graph=true;
 auto g=n::grad(n::sum(y),{x},graph)[0];auto second=n::grad(n::sum(n::square(g)),{x})[0];
 EXPECT_GT(n::max(n::abs(second)).item(),.01);
 EXPECT_THROW(m::GatedPixelGate(n::zeros({1,2,3,4})),std::invalid_argument);
}
TEST(NnGatedPixels, OddRectangularMaskedConvolutionUsesAllHorizontalTaps) {
 auto t=n::DType::Float64;n::ParameterStore p(1,t);
 auto x=n::tensor({1.,2.,4.,8.},{1,1,4,1},t).requires_grad_();
 auto mask=m::GatedPixelMask(1,false,false,1,t);
 m::MaskedPixelConv(p,"head",x,mask);
 p.set("head/W",n::ones({1,3,1,1},t));p.set("head/b",n::zeros({1},t));
 auto y=m::MaskedPixelConv(p,"head",x,mask);
 EXPECT_TRUE(n::equal_bits(y,n::tensor({1.,3.,6.,12.},{1,1,4,1},t)));
 auto g=n::grad(n::sum(y),p.parameters());
 EXPECT_EQ(g[0].to_vector<double>()[2],0);
 EXPECT_THROW(m::MaskedPixelConv(p,"even",x,n::ones({2,3,1,1},t)),std::invalid_argument);
}
TEST(NnGatedPixels, BothBinaryCensusesAndReusableRgbContextPreserveNamedLeaves) {
 auto t=n::DType::Float64;
 for(bool conditional:{false,true}) {
  n::ParameterStore p(8501,t);auto x=n::full({2,5,7,1},.2,t);
  auto o=m::GatedPixelCnn(p,x,Labels(2),conditional);
  EXPECT_EQ(p.count(),conditional?1553:1073);EXPECT_EQ(p.size(),conditional?29:23);
  EXPECT_EQ(o.logits.shape(),x.shape());
  for(const auto& h:{o.vertical0,o.horizontal0,o.vertical1,o.horizontal1,o.vertical,o.horizontal})
   EXPECT_EQ(h.shape(),(n::Shape{2,5,7,4}));
  EXPECT_EQ(p.contains("block0/condition_vertical"),conditional);
  EXPECT_FALSE(p.contains("block0/vertical_to_horizontal/b"));
 }
 n::ParameterStore rgb(8609,t);
 auto out=m::GatedPixelConvolutional(rgb,n::zeros({2,4,6,3},t),{},false,3,30);
 EXPECT_EQ(out.logits.shape(),(n::Shape{2,4,6,30}));EXPECT_EQ(rgb.count(),1410);EXPECT_EQ(rgb.size(),23);
}
TEST(NnGatedPixels, ValidLabelsAreIgnoredOnlyInUnconditionalModeAndInputsRejectBeforeLeaves) {
 auto t=n::DType::Float64;auto x=n::full({2,3,5,1},.25,t);
 auto labels=Labels(2),shifted=n::ones({2},n::DType::Int64);
 for(bool conditional:{false,true}) {
  n::ParameterStore p(8501,t);
  auto a=m::GatedPixelCnn(p,x,labels,conditional).logits;
  auto before=p.flatten();auto b=m::GatedPixelCnn(p,x,shifted,conditional).logits;
  if(conditional)EXPECT_GT(Difference(a,b),1e-6);else EXPECT_TRUE(n::equal_bits(a,b));
  EXPECT_EQ(before,p.flatten());
 }
 n::ParameterStore fresh(1,t);
 EXPECT_THROW(m::GatedPixelCnn(fresh,x,{},false),std::invalid_argument);
 EXPECT_THROW(m::GatedPixelCnn(fresh,x,n::full({2},10,n::DType::Int64),false),std::invalid_argument);
 EXPECT_THROW(m::GatedPixelCnn(fresh,x,n::zeros({2},t),true),std::invalid_argument);
 EXPECT_THROW(m::GatedPixelConvolutional(fresh,x,{},true),std::invalid_argument);
 EXPECT_THROW(m::GatedPixelConvolutional(fresh,x,{},false,1,INT64_MAX),std::invalid_argument);
 EXPECT_EQ(fresh.size(),0);
}
TEST(NnGatedPixels, All64SitesAreStrictlyCausalWhileVerticalLinkOpensBlindSpot) {
 auto t=n::DType::Float64;
 for(bool conditional:{false,true}) {
  n::ParameterStore p(8501,t);auto x=n::full({1,8,8,1},.25,t).requires_grad_();
  auto labels=Labels(1);m::GatedPixelCnn(p,x,labels,conditional);
  for(const auto& name:p.names())p.set(name,n::full(p.at(name).shape(),.1,t));
  auto logits=n::reshape(m::GatedPixelCnn(p,x,labels,conditional).logits,{64});
  n::GradOptions keep;keep.retain_graph=true;
  auto original=logits.to_vector<double>();
  for(int site=0;site<64;++site) {
   auto g=n::grad(n::sum(n::slice_axis(logits,0,site,site+1)),{x},keep)[0].to_vector<double>();
   for(int k=site;k<64;++k)EXPECT_EQ(g[k],0);
   auto changed=x.to_vector<double>();for(int k=site;k<64;++k)changed[k]=1-changed[k];
   auto flipped=n::reshape(m::GatedPixelCnn(p,n::from_vector(changed,x.shape()),labels,conditional).logits,{64}).to_vector<double>();
   for(int k=0;k<=site;++k)EXPECT_EQ(original[k],flipped[k]);
   if(site==34)EXPECT_GT(std::abs(g[28]),1e-10);
  }
 }
}
TEST(NnGatedPixels, ImageWeightHigherPathsAndFieldsSurviveStoreLifetime) {
 auto t=n::DType::Float64;auto x=n::full({1,3,5,1},.25,t).requires_grad_();
 auto logits=[&](){n::ParameterStore p(8501,t);return m::GatedPixelCnn(p,x,Labels(1),true).logits;}();
 n::GradOptions graph;graph.create_graph=true;
 auto g=n::grad(n::mean(n::square(logits)),{x},graph)[0];
 EXPECT_GT(n::max(n::abs(g)).item(),1e-10);
 EXPECT_GT(n::max(n::abs(n::grad(n::sum(g),{x})[0])).item(),1e-12);
}
TEST(NnGatedPixels, All320MaskedCoordinatesHaveZeroPartialsAndRemainFixedUnderAdam) {
 auto t=n::DType::Float64;
 for(bool conditional:{false,true}) {
  n::ParameterStore p(8501,t);auto x=n::full({2,4,5,1},.2,t);auto labels=Labels(2);
  m::GatedPixelCnn(p,x,labels,conditional);
  std::map<std::string,n::Tensor> initial;
  for(const auto& name:p.names())initial[name]=p.at(name).detach().clone();
  n::OptimizerConfig config;config.kind=n::OptimizerKind::Adam;n::Optimizer optimizer(p,.003,config);
  int masked=0;
  for(int step=0;step<3;++step) {
   auto o=m::GatedPixelCnn(p,x,labels,conditional);
   auto gradient=n::grad(m::BernoulliPixels(o.logits,n::ones(x.shape(),t)).loss,p.parameters());
   optimizer.step(gradient);
   for(std::size_t k=0;k<p.names().size();++k) {
    auto name=p.names()[k];bool vertical=name.find("/vertical/")!=std::string::npos;
    bool horizontal=name.find("/horizontal/")!=std::string::npos;
    if(!name.ends_with("/W")||!(vertical||horizontal))continue;
    auto mask=m::GatedPixelMask(name.starts_with("block0/")?1:4,vertical,name.starts_with("block0/"),8,t).to_vector<double>();
    auto g=gradient[k].to_vector<double>(),before=initial.at(name).to_vector<double>(),after=p.at(name).to_vector<double>();
    for(std::size_t j=0;j<mask.size();++j)if(mask[j]==0){EXPECT_EQ(g[j],0);EXPECT_EQ(before[j],after[j]);if(!step)++masked;}
   }
  }
  EXPECT_EQ(masked,320);EXPECT_EQ(optimizer.iterations(),3);
 }
}
