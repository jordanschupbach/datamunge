#include <gtest/gtest.h>
#include <dm/dm.hpp>
#include <datamunge/nn/autoregressive_pixels.hpp>
namespace n=dm::nn;
namespace m=n::models;

TEST(NnAutoregressivePixels, MasksHaveExactRasterStructureAndOverflowGuards) {
  for(bool b:{false,true}) {
    auto mask=m::CausalPixelMask(2,3,b,n::DType::Float64);
    EXPECT_EQ(mask.shape(),(n::Shape{3,3,2,3}));
    auto v=mask.to_vector<double>();
    for(int i=0;i<54;++i)EXPECT_EQ(v[i],i<(b?30:24)?1.:0.);
    EXPECT_FALSE(mask.requires_grad());
  }
  EXPECT_THROW(m::CausalPixelMask(0,8,false),std::invalid_argument);
  EXPECT_THROW(m::CausalPixelMask(INT64_MAX,2,false),std::invalid_argument);
  EXPECT_THROW(m::CausalPixelMask(1,1,false,n::DType::Int64),std::invalid_argument);
}
TEST(NnAutoregressivePixels, BinaryPreparationRoundsBeforeThresholdAndStopsInput) {
  auto x=n::full({2,28,28,1},.5).requires_grad_();
  auto out=m::BinaryPixels(x);
  EXPECT_TRUE(n::equal_bits(out.gray,n::full({2,8,8,1},.5)));
  EXPECT_TRUE(n::equal_bits(out.images,n::ones({2,8,8,1})));
  EXPECT_FALSE(out.gray.requires_grad());EXPECT_FALSE(out.images.requires_grad());
  auto v=std::vector<float>(784,0);
  v[1*28+1]=1;v[1*28+2]=.5;v[2*28+1]=.25;v[2*28+2]=.125;
  auto check=m::BinaryPixels(n::from_vector(v,{1,28,28,1}));
  EXPECT_EQ(check.gray.to_vector<float>()[0],float(.75*.75+.75*.25*.5+.25*.75*.25+.25*.25*.125));
  auto bad=n::full({1,28,28,1},1.1);
  EXPECT_THROW(m::BinaryPixels(bad),std::invalid_argument);
  EXPECT_THROW(m::BinaryPixels(n::zeros({1,8,8,1})),std::invalid_argument);
  EXPECT_THROW(m::BinaryPixels(n::zeros({1,28,28,1},n::DType::Float64)),std::invalid_argument);
}
TEST(NnAutoregressivePixels, UniformBankIsFullMidpointStreamIndependentOfPrefix) {
  auto u=m::BinaryPixelUniforms(3,8423);
  std::mt19937 rng(8423);auto v=u.to_vector<double>();
  for(double value:v)EXPECT_EQ(value,(double(rng())+.5)/4294967296.);
  EXPECT_EQ(u.shape(),(n::Shape{3,64}));EXPECT_FALSE(u.requires_grad());
  EXPECT_TRUE(n::equal_bits(u,m::BinaryPixelUniforms(3,8423)));
  EXPECT_FALSE(n::equal_bits(u,m::BinaryPixelUniforms(3,8424)));
  EXPECT_THROW(m::BinaryPixelUniforms(0,1),std::invalid_argument);
  EXPECT_THROW(m::BinaryPixelUniforms(INT64_MAX,1),std::invalid_argument);
}
TEST(NnAutoregressivePixels, BinaryLikelihoodHasStableUnitsAndHigherLogitDerivative) {
  auto t=n::DType::Float64;
  auto z=n::zeros({2,3,5,1},t).requires_grad_();
  auto out=m::BernoulliPixels(z,n::ones(z.shape(),t));
  EXPECT_NEAR(out.loss.item(),std::log(2.),1e-14);
  for(double value:out.per_image_nll.to_vector<double>())EXPECT_NEAR(value,15*std::log(2.),1e-13);
  n::GradOptions options;options.create_graph=true;
  auto g=n::grad(out.loss,{z},options)[0];
  auto h=n::grad(n::sum(g),{z})[0];
  for(double value:g.to_vector<double>())EXPECT_NEAR(value,-.5/30,1e-15);
  for(double value:h.to_vector<double>())EXPECT_NEAR(value,.25/30,1e-15);
  auto extreme=m::BernoulliPixels(n::tensor({-1000.,1000.},{1,1,2,1},t),n::tensor({1.,0.},{1,1,2,1},t));
  EXPECT_EQ(extreme.loss.item(),1000);
  EXPECT_THROW(m::BernoulliPixels(z,n::full(z.shape(),.5,t)),std::invalid_argument);
  EXPECT_THROW(m::BernoulliPixels(z,n::zeros({1,3,5,1},t)),std::invalid_argument);
}
TEST(NnAutoregressivePixels, CensusBoundariesRectanglesStridesAndInvalidBeforeLeaves) {
  n::ParameterStore p(8401,n::DType::Float64);
  auto x=n::transpose(n::full({2,5,7,1},.2,n::DType::Float64),{0,2,1,3});
  auto out=m::PixelCnn(p,x);
  EXPECT_EQ(p.count(),1841);EXPECT_EQ(p.size(),10);
  EXPECT_EQ(out.logits.shape(),(n::Shape{2,7,5,1}));
  for(const auto& v:{out.input_features,out.head,out.hidden0,out.hidden1,out.hidden})
    EXPECT_EQ(v.shape(),(n::Shape{2,7,5,8}));
  EXPECT_TRUE(n::equal_bits(out.logits,m::PixelCnn(p,x.contiguous()).logits));
  auto names=p.names();auto before=std::vector<n::Tensor>();
  for(const auto& name:names)before.push_back(p.at(name).clone());
  m::PixelCnn(p,x);
  for(std::size_t k=0;k<names.size();++k)EXPECT_TRUE(n::equal_bits(before[k],p.at(names[k])));
  n::ParameterStore fresh(1);
  EXPECT_THROW(m::PixelCnn(fresh,n::zeros({1,8,8,2})),std::invalid_argument);
  EXPECT_EQ(fresh.size(),0);
  EXPECT_THROW(m::MaskedPixelConv(fresh,"",n::zeros({1,8,8,1}),m::CausalPixelMask(1,8,false)),std::invalid_argument);
  EXPECT_EQ(fresh.size(),0);
  EXPECT_THROW(m::MaskedPixelConv(fresh,"head",n::zeros({1,8,8,1}),n::full({3,3,1,8},.5)),std::invalid_argument);
  EXPECT_EQ(fresh.size(),0);
}
TEST(NnAutoregressivePixels, StrictCausalityAndBlindSpotHoldAcrossAll64Sites) {
  auto t=n::DType::Float64;n::ParameterStore p(8401,t);
  auto x=n::full({1,8,8,1},.25,t).requires_grad_();
  m::PixelCnn(p,x);
  for(const auto& name:p.names())p.set(name,n::full(p.at(name).shape(),.1,t));
  auto z=n::reshape(m::PixelCnn(p,x).logits,{64});
  n::GradOptions keep;keep.retain_graph=true;
  for(int j=0;j<64;++j) {
    auto g=n::grad(n::sum(n::slice_axis(z,0,j,j+1)),{x},keep)[0].to_vector<double>();
    for(int k=j;k<64;++k)EXPECT_EQ(g[k],0);
    auto v=x.to_vector<double>();for(int k=j;k<64;++k)v[k]=1-v[k];
    auto flip=n::reshape(m::PixelCnn(p,n::from_vector(v,x.shape())).logits,{64}).to_vector<double>();
    auto original=z.to_vector<double>();
    for(int k=0;k<=j;++k)EXPECT_EQ(flip[k],original[k]);
    if(j==34)EXPECT_EQ(g[28],0); // Ordinary masked-convolution blind spot.
  }
}
TEST(NnAutoregressivePixels, MaskedKernelGradientAndAdamWritesStayZero) {
  auto t=n::DType::Float64;n::ParameterStore p(8401,t);
  auto x=n::full({2,4,5,1},.25,t);
  m::PixelCnn(p,x);
  std::vector<n::Tensor> initial;
  for(const auto& name:p.names())initial.push_back(p.at(name).clone());
  n::OptimizerConfig config;config.kind=n::OptimizerKind::Adam;
  n::Optimizer optimizer(p,.003,config);
  int masked=0;
  for(int step=0;step<3;++step) {
    auto out=m::BernoulliPixels(m::PixelCnn(p,x).logits,n::ones(x.shape(),t));
    auto g=n::grad(out.loss,p.parameters());
    optimizer.step(g);
    for(std::size_t k=0;k<p.names().size();++k) {
      const auto name=p.names()[k];
      if(name=="output/W" || !name.ends_with("/W"))continue;
      auto mask=m::CausalPixelMask(name=="input/W"?1:8,8,name!="input/W",t).to_vector<double>();
      auto grad=g[k].to_vector<double>(),before=initial[k].to_vector<double>(),after=p.at(name).to_vector<double>();
      for(std::size_t j=0;j<mask.size();++j)if(!mask[j]) {
        EXPECT_EQ(grad[j],0);EXPECT_EQ(before[j],after[j]);if(!step)++masked;
      }
    }
  }
  EXPECT_EQ(masked,808);EXPECT_EQ(optimizer.iterations(),3);
}
