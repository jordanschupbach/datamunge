#include <gtest/gtest.h>
#include <dm/dm.hpp>
namespace n=dm::nn;namespace m=n::models;

TEST(NnRecurrentPixels, StepUsesInputForgetOutputCandidateOrderAndPlusOneForgetBias) {
 auto t=n::DType::Float64;
 auto a=n::zeros({1,1,3,16},t),h=n::zeros({1,1,3,4},t),c=n::full({1,1,3,4},2,t),w=n::zeros({1,3,4,16},t);
 auto out=m::PixelRowLstmStep(a,h,c,w);
 const double f=1/(1+std::exp(-1.));
 for(double v:out.input_gate.to_vector<double>())EXPECT_EQ(v,.5);
 for(double v:out.forget_gate.to_vector<double>())EXPECT_NEAR(v,f,1e-15);
 for(double v:out.output_gate.to_vector<double>())EXPECT_EQ(v,.5);
 for(double v:out.candidate.to_vector<double>())EXPECT_EQ(v,0);
 for(double v:out.cell.to_vector<double>())EXPECT_NEAR(v,2*f,1e-15);
 for(double v:out.hidden.to_vector<double>())EXPECT_NEAR(v,.5*std::tanh(2*f),1e-15);
}
TEST(NnRecurrentPixels, RecurrentKernelReadsNeighboringColumnsOnlyFromExplicitPreviousRow) {
 auto t=n::DType::Float64;
 auto out=m::PixelRowLstmStep(n::zeros({1,1,3,16},t),n::ones({1,1,3,4},t),n::zeros({1,1,3,4},t),n::ones({1,3,4,16},t));
 auto g=out.gates.to_vector<double>();
 for(int x=0;x<3;++x)for(int k=0;k<16;++k)EXPECT_EQ(g[x*16+k],x==1?12:8);
}
TEST(NnRecurrentPixels, StepHasFiniteDifferencePartialsForFeaturesBothStatesAndKernel) {
 auto t=n::DType::Float64;
 std::vector<n::Tensor> input{
  n::reshape(n::arange(0,48,1,t)/100.-.2,{1,1,3,16}),
  n::full({1,1,3,4},.3,t),n::full({1,1,3,4},-.2,t),
  n::reshape(n::arange(0,192,1,t)/1000.-.05,{1,3,4,16})};
 for(auto& x:input)x.requires_grad_();
 auto objective=[](const std::vector<n::Tensor>& x){auto o=m::PixelRowLstmStep(x[0],x[1],x[2],x[3]);return n::sum(n::square(o.hidden))+.25*n::sum(o.cell);};
 n::GradOptions graph;graph.create_graph=true;
 auto gradient=n::grad(objective(input),input,graph);
 for(std::size_t k=0;k<input.size();++k) {
  for(auto j:std::set<std::int64_t>{0,input[k].numel()/2,input[k].numel()-1}) {
   auto values=input[k].to_vector<double>();auto plus=input,minus=input;
   values[j]+=1e-5;plus[k]=n::from_vector(values,input[k].shape());
   values[j]-=2e-5;minus[k]=n::from_vector(values,input[k].shape());
   const double finite=(objective(plus).item()-objective(minus).item())/2e-5;
   EXPECT_NEAR(gradient[k].to_vector<double>()[j],finite,1e-9);
  }
 }
 auto second=n::grad(n::sum(gradient[2]),{input[0]})[0];
 EXPECT_GT(n::max(n::abs(second)).item(),1e-4);
}
TEST(NnRecurrentPixels, ModelCensusAndConnectedZeroStatesPreserveAllSevenBoundaries) {
 auto t=n::DType::Float64;n::ParameterStore p(8303,t);
 auto image=n::full({2,3,5,1},.25,t).requires_grad_();auto out=m::PixelRnn(p,image);
 EXPECT_EQ(p.count(),357);EXPECT_EQ(p.size(),5);
 EXPECT_EQ(p.names(),(std::vector<std::string>{"input/W","input/b","row/W","output/W","output/b"}));
 EXPECT_EQ(out.logits.shape(),image.shape());
 EXPECT_EQ(out.input_gates.shape(),(n::Shape{2,3,5,16}));EXPECT_EQ(out.row_gates.shape(),out.input_gates.shape());
 EXPECT_EQ(out.hidden.shape(),(n::Shape{2,3,5,4}));EXPECT_EQ(out.cell.shape(),out.hidden.shape());
 EXPECT_EQ(out.initial_hidden.shape(),(n::Shape{2,1,5,4}));
 EXPECT_TRUE(n::equal_bits(out.initial_hidden,out.initial_cell));EXPECT_TRUE(out.initial_hidden.requires_grad());
 EXPECT_EQ(n::max(n::abs(out.initial_hidden)).item(),0);
 auto zero=n::grad(n::sum(out.initial_hidden)+n::sum(out.initial_cell),{image})[0];
 EXPECT_EQ(n::max(n::abs(zero)).item(),0);
 EXPECT_FALSE(p.contains("row/b"));EXPECT_FALSE(p.contains("initial_hidden"));
}
TEST(NnRecurrentPixels, EveryRasterSiteHasExactFutureZerosAndFiniteChangeInvariance) {
 auto t=n::DType::Float64;n::ParameterStore p(8303,t);
 auto image=n::full({1,8,8,1},.25,t).requires_grad_();
 auto logits=n::reshape(m::PixelRnn(p,image).logits,{64});auto original=logits.to_vector<double>();
 n::GradOptions keep;keep.retain_graph=true;
 for(int site=0;site<64;++site) {
  auto g=n::grad(n::sum(n::slice_axis(logits,0,site,site+1)),{image},keep)[0].to_vector<double>();
  for(int k=site;k<64;++k)EXPECT_EQ(g[k],0);
  auto changed=image.to_vector<double>();for(int k=site;k<64;++k)changed[k]=1-changed[k];
  auto flipped=n::reshape(m::PixelRnn(p,n::from_vector(changed,image.shape())).logits,{64}).to_vector<double>();
  for(int k=0;k<=site;++k)EXPECT_EQ(flipped[k],original[k]);
 }
}
TEST(NnRecurrentPixels, AllEightyMaskedCoordinatesRemainFixedAcrossAdamWrites) {
 auto t=n::DType::Float64;n::ParameterStore p(8303,t);auto x=n::full({2,4,5,1},.2,t);
 m::PixelRnn(p,x);auto initial=p.at("input/W").to_vector<double>();
 auto mask=m::CausalPixelMask(1,16,false,t).to_vector<double>();
 n::OptimizerConfig config;config.kind=n::OptimizerKind::Adam;n::Optimizer optimizer(p,.003,config);
 for(int step=0;step<3;++step) {
  auto gradient=n::grad(m::BernoulliPixels(m::PixelRnn(p,x).logits,n::ones(x.shape(),t)).loss,p.parameters());
  optimizer.step(gradient);auto g=gradient[0].to_vector<double>(),after=p.at("input/W").to_vector<double>();int zeros=0;
  for(std::size_t k=0;k<mask.size();++k)if(mask[k]==0){EXPECT_EQ(g[k],0);EXPECT_EQ(after[k],initial[k]);++zeros;}
  EXPECT_EQ(zeros,80);
 }
 EXPECT_EQ(optimizer.iterations(),3);
}
TEST(NnRecurrentPixels, OwningModelGraphRetainsHigherImagePathsAfterStoreCollection) {
 auto t=n::DType::Float64;auto image=n::full({1,3,5,1},.2,t).requires_grad_();
 auto out=[&](){n::ParameterStore p(8303,t);return m::PixelRnn(p,image);}();
 n::GradOptions graph;graph.create_graph=true;
 auto first=n::grad(n::mean(n::square(out.logits)),{image},graph)[0];
 EXPECT_GT(n::max(n::abs(first)).item(),1e-10);
 auto second=n::grad(n::sum(first),{image})[0];EXPECT_GT(n::max(n::abs(second)).item(),1e-12);
}
TEST(NnRecurrentPixels, RectanglesStridesBatchIndependenceAndInvalidInputsBeforeLeaves) {
 auto t=n::DType::Float64;n::ParameterStore p(8303,t);
 auto x=n::transpose(n::full({2,3,5,1},.25,t),{0,2,1,3});
 auto out=m::PixelRnn(p,x);auto before=p.flatten();
 EXPECT_TRUE(n::equal_bits(out.logits,m::PixelRnn(p,x.contiguous()).logits));
 EXPECT_TRUE(n::equal_bits(n::take_rows(out.logits,{0}),m::PixelRnn(p,n::take_rows(x,{0})).logits));
 EXPECT_EQ(before,p.flatten());
 n::ParameterStore fresh(1,t);
 EXPECT_THROW(m::PixelRnn(fresh,n::zeros({1,3,5,2},t)),std::invalid_argument);
 EXPECT_THROW(m::PixelRnn(fresh,n::zeros({1,3,5,1})),std::invalid_argument);
 EXPECT_THROW(m::PixelRnn(fresh,n::full({1,3,5,1},std::nan(""),t)),std::invalid_argument);
 EXPECT_EQ(fresh.size(),0);
 EXPECT_THROW(m::PixelRowLstmStep(n::zeros({1,2,3,16},t),n::zeros({1,1,3,4},t),n::zeros({1,1,3,4},t),n::zeros({1,3,4,16},t)),std::invalid_argument);
 EXPECT_THROW(m::PixelRowLstmStep(n::zeros({1,1,3,16},t),n::zeros({1,1,2,4},t),n::zeros({1,1,3,4},t),n::zeros({1,3,4,16},t)),std::invalid_argument);
 EXPECT_THROW(m::PixelRowLstmStep(n::zeros({1,1,3,16},t),n::zeros({1,1,3,4},t),n::zeros({1,1,3,4},t),n::zeros({3,3,4,16},t)),std::invalid_argument);
}
