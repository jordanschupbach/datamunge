#include <gtest/gtest.h>
#include <dm/dm.hpp>
namespace n=dm::nn;namespace m=n::models;
TEST(NnMadeDensity, DegreeComparisonsHandleZeroStrictnessMixedIntegerWidthsAndErrors) {
 auto a=n::from_vector(std::vector<std::int32_t>{0,1,2},{3});
 auto b=n::from_vector(std::vector<std::int64_t>{0,2},{2});
 auto inclusive=m::AutoregressiveDegreeMask(a,b,false,n::DType::Float64);
 auto strict=m::AutoregressiveDegreeMask(a,b,true,n::DType::Float64);
 EXPECT_TRUE(n::equal_bits(inclusive,n::tensor({1.,1.,0.,1.,0.,1.},{3,2},n::DType::Float64)));
 EXPECT_TRUE(n::equal_bits(strict,n::tensor({0.,1.,0.,1.,0.,0.},{3,2},n::DType::Float64)));
 EXPECT_FALSE(strict.requires_grad());
 EXPECT_THROW(m::AutoregressiveDegreeMask(n::ones({3}),b),std::invalid_argument);
 EXPECT_THROW(m::AutoregressiveDegreeMask(n::from_vector(std::vector<std::int32_t>{-1},{1}),b),std::invalid_argument);
 EXPECT_THROW(m::AutoregressiveDegreeMask(a,b,false,n::DType::Int64),std::invalid_argument);
}
TEST(NnMadeDensity, SourceInventoryCensusAndEveryStructuralFuturePathAreExact) {
 for(bool reverse:{false,true}) {
  auto out=m::MadeMasks(reverse,n::DType::Float64);
  auto d=out.input_degrees.to_vector<std::int32_t>(),h1=out.hidden1_degrees.to_vector<std::int32_t>(),h2=out.hidden2_degrees.to_vector<std::int32_t>();
  for(int j=0;j<64;++j)EXPECT_EQ(d[j],reverse?64-j:j+1);
  for(int j=0;j<16;++j){EXPECT_EQ(h1[j],1+j*62/15);EXPECT_EQ(h2[j],h1[(j*5)%16]);}
  int blocked=0;
  for(const auto& mask:{out.input_mask,out.hidden_mask,out.output_mask})for(double v:mask.to_vector<double>())if(v==0)++blocked;
  EXPECT_EQ(blocked,1144);EXPECT_EQ(out.input_degrees.dtype(),n::DType::Int32);
  auto paths=n::matmul(n::matmul(out.input_mask,out.hidden_mask),out.output_mask).to_vector<double>();
  for(int incoming=0;incoming<64;++incoming)for(int outgoing=0;outgoing<64;++outgoing)
   if(d[incoming]>=d[outgoing])EXPECT_EQ(paths[incoming*64+outgoing],0);
 }
 EXPECT_THROW(m::MadeMasks(false,n::DType::Int64),std::invalid_argument);
}
TEST(NnMadeDensity, MaskedDenseKeepsLiveInputWeightPathsAndStopsStructuralMask) {
 auto t=n::DType::Float64;n::ParameterStore p(1,t);
 auto x=n::tensor({1.,2.,3.,4.},{2,2},t).requires_grad_();
 auto mask=n::tensor({1.,0.,0.,1.},{2,2},t).requires_grad_();
 m::MaskedDensityDense(p,"toy",x,mask);p.set("toy/W",n::ones({2,2},t));p.set("toy/b",n::zeros({2},t));
 auto y=m::MaskedDensityDense(p,"toy",x,mask);EXPECT_TRUE(n::equal_bits(y,x));
 n::GradOptions graph;graph.create_graph=true;graph.allow_unused=true;
 auto gradient=n::grad(n::sum(n::square(y)),{x,p.at("toy/W"),mask},graph);
 EXPECT_FALSE(gradient[2].defined());
 EXPECT_TRUE(n::equal_bits(gradient[0],2*x));
 auto g=gradient[1].to_vector<double>();EXPECT_EQ(g[1],0);EXPECT_EQ(g[2],0);
 EXPECT_GT(n::max(n::abs(n::grad(n::sum(gradient[0]),{p.at("toy/W")})[0])).item(),1);
}
TEST(NnMadeDensity, BothOrdersHaveSameRawInitializationAndValidateBeforeLeaves) {
 auto t=n::DType::Float64;n::ParameterStore a(8803,t),b(8803,t);
 auto image=n::full({2,8,8,1},.25,t);
 auto x=m::MadeDensity(a,image,false),y=m::MadeDensity(b,image,true);
 EXPECT_EQ(a.count(),2400);EXPECT_EQ(a.size(),6);EXPECT_EQ(a.flatten(),b.flatten());
 EXPECT_EQ(x.hidden1.shape(),(n::Shape{2,16}));EXPECT_EQ(x.hidden2.shape(),(n::Shape{2,16}));EXPECT_EQ(x.logits.shape(),image.shape());
 auto before=a.flatten();m::MadeDensity(a,image,true);EXPECT_EQ(before,a.flatten());
 n::ParameterStore fresh(1,t);
 EXPECT_THROW(m::MadeDensity(fresh,n::zeros({1,4,16,1},t)),std::invalid_argument);
 EXPECT_THROW(m::MadeDensity(fresh,n::zeros({1,8,8,2},t)),std::invalid_argument);
 EXPECT_THROW(m::MaskedDensityDense(fresh,"",n::zeros({2,3},t),n::ones({3,4},t)),std::invalid_argument);
 EXPECT_THROW(m::MaskedDensityDense(fresh,"x",n::zeros({2,3},t),n::full({3,4},.5,t)),std::invalid_argument);
 EXPECT_EQ(fresh.size(),0);
}
TEST(NnMadeDensity, BothDegreeOrdersHaveAll64ExactCausalAndFutureFlipGuards) {
 auto t=n::DType::Float64;
 for(bool reverse:{false,true}) {
  n::ParameterStore p(8803,t);auto image=n::full({1,8,8,1},.25,t).requires_grad_();
  auto logits=n::reshape(m::MadeDensity(p,image,reverse).logits,{64});auto original=logits.to_vector<double>();
  n::GradOptions keep;keep.retain_graph=true;
  for(int ordinal=0;ordinal<64;++ordinal) {
   int site=reverse?63-ordinal:ordinal;
   auto gradient=n::grad(n::sum(n::slice_axis(logits,0,site,site+1)),{image},keep)[0].to_vector<double>();
   auto changed=image.to_vector<double>();
   for(int j=0;j<64;++j)if((reverse?63-j:j)>=ordinal){EXPECT_EQ(gradient[j],0);changed[j]=1-changed[j];}
   auto flipped=n::reshape(m::MadeDensity(p,n::from_vector(changed,image.shape()),reverse).logits,{64}).to_vector<double>();
   for(int j=0;j<64;++j)if((reverse?63-j:j)<=ordinal)EXPECT_EQ(original[j],flipped[j]);
  }
 }
}
TEST(NnMadeDensity, All1144MaskedCoordinatesStayFixedUnderAdamForEitherOrder) {
 auto t=n::DType::Float64;
 for(bool reverse:{false,true}) {
  n::ParameterStore p(8803,t);auto image=n::full({2,8,8,1},.2,t);m::MadeDensity(p,image,reverse);
  auto inv=m::MadeMasks(reverse,t);std::map<std::string,n::Tensor> masks{{"input/W",inv.input_mask},{"hidden/W",inv.hidden_mask},{"output/W",inv.output_mask}},initial;
  for(const auto& name:p.names())initial[name]=p.at(name).detach().clone();
  n::OptimizerConfig config;config.kind=n::OptimizerKind::Adam;n::Optimizer optimizer(p,.003,config);
  for(int step=0;step<3;++step) {
   auto gradient=n::grad(m::BernoulliPixels(m::MadeDensity(p,image,reverse).logits,n::ones(image.shape(),t)).loss,p.parameters());
   optimizer.step(gradient);int blocked=0;
   for(std::size_t k=0;k<p.names().size();++k) {
    auto name=p.names()[k];if(!masks.count(name))continue;
    auto mask=masks.at(name).to_vector<double>(),g=gradient[k].to_vector<double>(),before=initial.at(name).to_vector<double>(),after=p.at(name).to_vector<double>();
    for(std::size_t j=0;j<mask.size();++j)if(mask[j]==0){EXPECT_EQ(g[j],0);EXPECT_EQ(before[j],after[j]);++blocked;}
   }
   EXPECT_EQ(blocked,1144);
  }
 }
}
TEST(NnMadeDensity, MixtureNormalizesJointTablesAndHasResponsibilityGradientAndHessian) {
 auto t=n::DType::Float64;
 auto nll=(-n::log(n::tensor({.4,.1,.1,.4,.1,.4,.4,.1},{4,2},t))).requires_grad_();
 auto out=m::EqualDensityMixture(nll);
 for(double v:out.per_image_nll.to_vector<double>())EXPECT_NEAR(v,std::log(4.),1e-14);
 EXPECT_NEAR(n::sum(n::exp(-out.per_image_nll)).item(),1,1e-14);
 EXPECT_LT(out.mean_nll.item(),n::mean(nll).item());
 n::GradOptions graph;graph.create_graph=true;
 auto gradient=n::grad(n::sum(out.per_image_nll),{nll},graph)[0];
 EXPECT_LE(n::max(n::abs(gradient-out.responsibilities)).item(),1e-14);
 auto second=n::grad(n::sum(n::slice_axis(gradient,1,0,1)),{nll})[0].to_vector<double>();
 for(int row=0;row<4;++row){EXPECT_NEAR(second[row*2],-.16,1e-14);EXPECT_NEAR(second[row*2+1],.16,1e-14);}
}
TEST(NnMadeDensity, MixtureHandlesExtremeSignedNllsSingleComponentsAndInvalidInputs) {
 auto t=n::DType::Float64;
 auto values=n::tensor({1000.,1002.,-1002.,-1000.},{2,2},t);
 auto out=m::EqualDensityMixture(values);
 EXPECT_NEAR(out.per_image_nll.to_vector<double>()[0],1000-std::log1p(std::exp(-2.))+std::log(2.),1e-13);
 EXPECT_NEAR(out.per_image_nll.to_vector<double>()[1],-1002-std::log1p(std::exp(-2.))+std::log(2.),1e-13);
 auto one=m::EqualDensityMixture(n::tensor({-2.,3.},{2,1}));
 EXPECT_EQ(one.per_image_nll.dtype(),t);EXPECT_TRUE(n::equal_bits(one.per_image_nll,n::tensor({-2.,3.},{2},t)));
 EXPECT_THROW(m::EqualDensityMixture(n::zeros({2})),std::invalid_argument);
 EXPECT_THROW(m::EqualDensityMixture(n::full({2,2},std::nan(""),t)),std::invalid_argument);
 EXPECT_THROW(m::EqualDensityMixture(n::zeros({2,2},n::DType::Int64)),std::invalid_argument);
}
TEST(NnMadeDensity, OwningModelFieldsRetainHigherImagePathsAfterStoreCollection) {
 auto t=n::DType::Float64;auto image=n::full({1,8,8,1},.25,t).requires_grad_();
 auto out=[&](){n::ParameterStore p(8803,t);return m::MadeDensity(p,image,true);}();
 n::GradOptions graph;graph.create_graph=true;
 auto first=n::grad(n::mean(n::square(out.logits)),{image},graph)[0];
 EXPECT_GT(n::max(n::abs(first)).item(),1e-10);
 EXPECT_GT(n::max(n::abs(n::grad(n::sum(first),{image})[0])).item(),1e-12);
}
