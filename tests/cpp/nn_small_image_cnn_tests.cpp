#include <datamunge/nn/bar_images.hpp>
#include <datamunge/nn/zplus_relevance.hpp>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/losses.hpp>
#include <datamunge/nn/metrics.hpp>
#include <datamunge/nn/optim.hpp>
#include <datamunge/nn/vision_plots.hpp>
#include <datamunge/nn/model_plot.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
namespace {
namespace n=datamunge::nn;namespace m=n::models;
void close(const n::Tensor& x,const std::vector<double>& expected,double tolerance=1e-10){const auto actual=x.to_vector<double>();ASSERT_EQ(actual.size(),expected.size());for(std::size_t i=0;i<actual.size();++i)EXPECT_NEAR(actual[i],expected[i],tolerance);}
n::Tensor image(){return n::tensor({1,2,0,0,1,3,2,0,1},{1,3,3,1},n::DType::Float64);}
m::SmallImageCnnConfig hand_config(){m::SmallImageCnnConfig c;c.filters=1;c.kernel=2;c.pool_size=1;c.hidden=1;return c;}
void hand_store(n::ParameterStore& p) {
  (void)m::SmallImageCnn(p,image(),hand_config());
  p.set("cnn/conv/W",n::tensor({1,0,0,1},{2,2,1,1},n::DType::Float64));p.set("cnn/conv/b",n::zeros({1},n::DType::Float64));
  p.set("cnn/hidden/W",n::ones({4,1},n::DType::Float64));p.set("cnn/hidden/b",n::zeros({1},n::DType::Float64));
  p.set("cnn/output/W",n::ones({1,2},n::DType::Float64));p.set("cnn/output/b",n::tensor({0,-20},{2},n::DType::Float64));
}
}
TEST(SmallImageCnn, NoiselessBarMasksLabelsPositionsAndShuffleAreIndependentOwnedData) {
  n::BarImageOptions o;o.side=6;o.thickness=2;o.noise_stddev=0;o.min_position=2;o.max_position=2;o.dtype=n::DType::Float64;auto data=n::make_bar_images(2,7,o);EXPECT_EQ(data.images.shape(),(n::Shape{4,6,6,1}));EXPECT_EQ(data.images.to_vector<double>(),data.object_mask.to_vector<double>());EXPECT_EQ(data.labels.to_vector<std::int64_t>(),(std::vector<std::int64_t>{0,1,0,1}));
  auto sorted=data.shuffled_rows;std::sort(sorted.begin(),sorted.end());EXPECT_EQ(sorted,(std::vector<std::int64_t>{0,1,2,3}));close(n::sum(data.object_mask,{1,2,3}),{12,12,12,12});
  const auto values=data.images.to_vector<double>();for(int row=0;row<4;++row)for(int y=0;y<6;++y)for(int x=0;x<6;++x)EXPECT_EQ(values[(row*6+y)*6+x],((row%2?x:y)>=2 && (row%2?x:y)<4)?1:0);
  data.images.zero_();EXPECT_EQ(n::sum(data.object_mask).item(),48);
}
TEST(SmallImageCnn, NativeGeneratorMatchesOriginalPerImageNormalStreamAndPostGenerationShuffle) {
  n::BarImageOptions o;o.dtype=n::DType::Float64;const auto data=n::make_bar_images(300,20240815,o);const auto pixels=data.images.to_vector<double>(),masks=data.object_mask.to_vector<double>();const auto positions=data.positions.to_vector<std::int64_t>();std::mt19937_64 engine(20240815);std::uniform_int_distribution<std::size_t> position(0,10);
  for(std::size_t row=0;row<600;++row){const auto pos=position(engine);EXPECT_EQ(positions[row],pos);std::normal_distribution<double> noise(0,.08);for(std::size_t y=0;y<12;++y)for(std::size_t x=0;x<12;++x){const auto coordinate=row%2?x:y;const bool occupied=coordinate==pos || coordinate==pos+1;const auto index=(row*12+y)*12+x;EXPECT_EQ(pixels[index],std::clamp(double(occupied)+noise(engine),0.0,1.0));EXPECT_EQ(masks[index],occupied?1:0);}}
  std::vector<std::int64_t> order(600);std::iota(order.begin(),order.end(),0);std::shuffle(order.begin(),order.end(),engine);EXPECT_EQ(order,data.shuffled_rows);
}
TEST(SmallImageCnn, NoisyDataAreReproducibleBoundedAndCarryTrueMasks) {
  n::BarImageOptions o;o.noise_stddev=1;const auto a=n::make_bar_images(10,3,o),b=n::make_bar_images(10,3,o),other=n::make_bar_images(10,4,o);EXPECT_EQ(a.images.to_vector<float>(),b.images.to_vector<float>());EXPECT_NE(a.images.to_vector<float>(),other.images.to_vector<float>());const auto values=a.images.to_vector<float>(),mask=a.object_mask.to_vector<float>();bool noisy_object=false;for(std::size_t i=0;i<values.size();++i){EXPECT_GE(values[i],0);EXPECT_LE(values[i],1);noisy_object|=mask[i]==1 && values[i]<0.5;}EXPECT_TRUE(noisy_object);close(n::sum(a.object_mask,{1,2,3}),std::vector<double>(20,24),1e-6);
}
TEST(SmallImageCnn, BarInputDomainsAndCompleteDimensionsRejectBeforeAllocation) {
  EXPECT_THROW(n::make_bar_images(0),std::invalid_argument);
  n::BarImageOptions o;o.thickness=13;
  EXPECT_THROW(n::make_bar_images(1,0,o),std::invalid_argument);
  o={};o.min_position=11;
  EXPECT_THROW(n::make_bar_images(1,0,o),std::invalid_argument);
  o={};o.noise_stddev=std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(n::make_bar_images(1,0,o),std::invalid_argument);
  o={};o.dtype=n::DType::Int64;
  EXPECT_THROW(n::make_bar_images(1,0,o),std::invalid_argument);
  EXPECT_THROW(n::make_bar_images(std::numeric_limits<std::int64_t>::max()),std::length_error);
  o={};o.side=std::numeric_limits<std::int64_t>::max();
  EXPECT_THROW(n::make_bar_images(1,0,o),std::length_error);
}
TEST(SmallImageCnn, OriginalLocalGlobalAndDenseParameterCountsFollowActualTensorLeaves) {
  const auto x=n::zeros({2,12,12,1});n::ParameterStore main(1),local(1),global(1),dense(1);m::SmallImageCnnConfig c;const auto a=m::SmallImageCnn(main,x,c);EXPECT_EQ(main.count(),6578);EXPECT_EQ(main.parameters().size(),6u);EXPECT_EQ(a.conv_pre.shape(),(n::Shape{2,10,10,8}));EXPECT_EQ(a.pooled.shape(),(n::Shape{2,5,5,8}));EXPECT_EQ(a.flat.shape(),(n::Shape{2,200}));EXPECT_EQ(a.logits.shape(),(n::Shape{2,2}));c.hidden=16;(void)m::SmallImageCnn(local,x,c);EXPECT_EQ(local.count(),3330);c.global_pool=true;const auto b=m::SmallImageCnn(global,x,c);EXPECT_EQ(global.count(),258);EXPECT_EQ(b.pooled.shape(),(n::Shape{2,1,1,8}));(void)m::DenseImageBaseline(dense,x);EXPECT_EQ(dense.count(),4706);
}
TEST(SmallImageCnn, EntireForwardMatchesAHandComputedSharedDiagonalKernelAndDenseHead) {
  n::ParameterStore p(1,n::DType::Float64);hand_store(p);const auto out=m::SmallImageCnn(p,image(),hand_config());close(out.conv_pre,{2,5,0,2});close(out.conv_relu,{2,5,0,2});close(out.pooled,{2,5,0,2});close(out.hidden,{9});close(out.logits,{9,-11});EXPECT_EQ(out.pool_argmax.to_vector<std::int64_t>(),(std::vector<std::int64_t>{0,1,2,3}));
}
TEST(SmallImageCnn, SharedKernelGradientSumsContributionsFromEverySpatialApplication) {
  auto kernel=n::tensor({1,0,0,1},{2,2,1,1},n::DType::Float64);kernel.requires_grad_();n::Conv2DOptions o;o.padding=n::Padding::Valid;const auto y=n::conv2d(image(),kernel,o);const auto derivative=n::grad(n::sum(y),{kernel})[0];close(y,{2,5,0,2});close(derivative,{4,6,3,5});
}
TEST(SmallImageCnn, AutogradOfEveryLearnedValueMatchesCentralFiniteDifferencesThroughMaxPooling) {
  n::ParameterStore p(9,n::DType::Float64);m::SmallImageCnnConfig c;c.filters=2;c.kernel=2;c.pool_size=2;c.hidden=3;std::vector<double> values(25);std::mt19937_64 random(7);std::uniform_real_distribution<double> uniform(.1,1);for(auto& value:values)value=uniform(random);const auto x=n::from_vector(values,{1,5,5,1});(void)m::SmallImageCnn(p,x,c);
  for(const auto& name:p.names()){const auto leaf=p.at(name);std::vector<double> v(leaf.numel());for(std::size_t i=0;i<v.size();++i)v[i]=.1+.013*(i+1);p.set(name,n::from_vector(v,leaf.shape()));}
  const auto leaves=p.parameters();const auto loss=n::sum(n::square(m::SmallImageCnn(p,x,c).logits));const auto gradients=n::grad(loss,leaves);n::NoGradGuard guard;
  for(std::size_t j=0;j<leaves.size();++j){auto leaf=leaves[j];auto base=leaf.to_vector<double>();const auto analytic=gradients[j].to_vector<double>();for(std::size_t i=0;i<base.size();++i){auto up=base,down=base;up[i]+=1e-5;down[i]-=1e-5;leaf.copy_(n::from_vector(up,leaf.shape()));const auto hi=n::sum(n::square(m::SmallImageCnn(p,x,c).logits)).item();leaf.copy_(n::from_vector(down,leaf.shape()));const auto lo=n::sum(n::square(m::SmallImageCnn(p,x,c).logits)).item();leaf.copy_(n::from_vector(base,leaf.shape()));EXPECT_NEAR(analytic[i],(hi-lo)/2e-5,1e-5);}}
}
TEST(SmallImageCnn, LocalMaxPoolingIsNotInvariantToEveryShiftSmallerThanItsWindow) {
  const auto a=n::tensor({0,1,0,0},{1,1,4,1}),b=n::tensor({0,0,1,0},{1,1,4,1});close(n::max_pool2d(a,{1,2},{1,2}),{1,0});close(n::max_pool2d(b,{1,2},{1,2}),{0,1});
}
TEST(SmallImageCnn, InteriorCleanTranslationsPreserveGlobalMaxWhileAnExplicitLocalHeadReadsPosition) {
  n::BarImageOptions o;o.dtype=n::DType::Float64;o.noise_stddev=0;o.min_position=2;o.max_position=2;const auto a=n::make_bar_images(1,1,o).images;o.min_position=6;o.max_position=6;const auto b=n::make_bar_images(1,1,o).images;
  m::SmallImageCnnConfig c;c.filters=1;c.kernel=1;c.hidden=1;c.global_pool=true;n::ParameterStore global(1,n::DType::Float64);(void)m::SmallImageCnn(global,a,c);global.set("cnn/conv/W",n::ones({1,1,1,1},n::DType::Float64));global.set("cnn/conv/b",n::zeros({1},n::DType::Float64));global.set("cnn/hidden/W",n::ones({1,1},n::DType::Float64));global.set("cnn/hidden/b",n::zeros({1},n::DType::Float64));global.set("cnn/output/W",n::tensor({1,0},{1,2},n::DType::Float64));global.set("cnn/output/b",n::zeros({2},n::DType::Float64));close(m::SmallImageCnn(global,a,c).logits,{1,0,1,0});close(m::SmallImageCnn(global,b,c).logits,{1,0,1,0});
  c.global_pool=false;n::ParameterStore local(1,n::DType::Float64);(void)m::SmallImageCnn(local,a,c);for(const auto& name:global.names())if(name!="cnn/hidden/W")local.set(name,global.at(name));std::vector<double> w(36);w[6]=1;local.set("cnn/hidden/W",n::from_vector(w,{36,1}));EXPECT_EQ(m::SmallImageCnn(local,a,c).logits.to_vector<double>()[0],1);EXPECT_EQ(m::SmallImageCnn(local,b,c).logits.to_vector<double>()[0],0);
}
TEST(SmallImageCnn, DenseZPlusMatchesPositiveConnectionFractionsAndAllowsSignedIncomingScores) {
  const auto a=n::tensor({1,3},{1,2},n::DType::Float64),w=n::ones({2,1},n::DType::Float64);close(n::zplus_dense(a,w,n::tensor({4},{1,1},n::DType::Float64)),{1,3},1e-8);close(n::zplus_dense(a,w,n::tensor({-4},{1,1},n::DType::Float64)),{-1,-3},1e-8);close(n::zplus_dense(a,w,n::tensor({4},{1,1},n::DType::Float64),1),{.8,2.4});
  close(n::zplus_dense(a,-w,n::tensor({4},{1,1},n::DType::Float64)),{0,0});
}
TEST(SmallImageCnn, ConvolutionZPlusMatchesAnIndependentSignedNestedLoopRedistribution) {
  constexpr int N=2,H=3,W=4,C=2,F=2,K=2,OH=2,OW=3;std::vector<double> x(N*H*W*C),weight(K*K*C*F),r(N*OH*OW*F);for(std::size_t i=0;i<x.size();++i)x[i]=.1+.03*(i%9);for(std::size_t i=0;i<weight.size();++i)weight[i]=i%3==0?-.2:.05*(i+1);for(std::size_t i=0;i<r.size();++i)r[i]=(i%2?-1:1)*(.3+.02*i);std::vector<double> expected(x.size());
  for(int n=0;n<N;++n)for(int y=0;y<OH;++y)for(int z=0;z<OW;++z)for(int f=0;f<F;++f){double denom=.01;for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)for(int c=0;c<C;++c)denom+=x[((n*H+y+ky)*W+z+kx)*C+c]*std::max(0.0,weight[((ky*K+kx)*C+c)*F+f]);const auto score=r[((n*OH+y)*OW+z)*F+f];for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)for(int c=0;c<C;++c){const auto id=((n*H+y+ky)*W+z+kx)*C+c;expected[id]+=score*x[id]*std::max(0.0,weight[((ky*K+kx)*C+c)*F+f])/denom;}}
  n::Conv2DOptions options;options.padding=n::Padding::Valid;close(n::zplus_conv2d(n::from_vector(x,{N,H,W,C}),n::from_vector(weight,{K,K,C,F}),n::from_vector(r,{N,OH,OW,F}),options,.01),expected,1e-12);
}
TEST(SmallImageCnn, ZPlusRejectsInvalidDomainsUnrepresentableStabilizersAndNonfiniteArithmetic) {
  EXPECT_THROW(n::zplus_dense(n::tensor({-1},{1,1}),n::ones({1,1}),n::ones({1,1})),std::invalid_argument);
  EXPECT_THROW(n::zplus_dense(n::ones({1,1}),n::ones({1,1}),n::ones({1,1}),0),std::invalid_argument);
  EXPECT_THROW(n::zplus_dense(n::ones({1,1}),n::ones({1,1}),n::ones({1,1}),1e-300),std::invalid_argument);
  EXPECT_THROW(n::zplus_dense(n::ones({1,2}),n::ones({1,1}),n::ones({1,1})),std::invalid_argument);
  EXPECT_THROW(n::zplus_dense(n::ones({1,1}),n::ones({1,1},n::DType::Float64),n::ones({1,1})),std::invalid_argument);
  EXPECT_THROW(n::zplus_dense(n::tensor({std::numeric_limits<double>::max()},{1,1},n::DType::Float64),n::tensor({2},{1,1},n::DType::Float64),n::ones({1,1},n::DType::Float64)),std::overflow_error);
  EXPECT_THROW(n::zplus_conv2d(n::ones({1,2,2,1}),n::ones({1,1,1,1}),n::ones({1,1,1,1})),std::invalid_argument);
}
TEST(SmallImageCnn, WholeCnnRedistributionMatchesTheHandComputedPixelPathsAndSignedSeed) {
  n::ParameterStore p(1,n::DType::Float64);hand_store(p);const auto positive=m::SmallImageCnnZPlus(p,image(),{0},hand_config());close(positive.logits,{9,-11});close(positive.input_relevance,{1,2,0,0,2,3,0,0,1},1e-8);const auto negative=m::SmallImageCnnZPlus(p,image(),{1},hand_config());EXPECT_LT(n::sum(negative.input_relevance).item(),0);EXPECT_NEAR(n::sum(negative.input_relevance).item(),-11,1e-7);EXPECT_FALSE(positive.input_relevance.requires_grad());
  p.set("cnn/output/W",-n::ones({1,2},n::DType::Float64));p.set("cnn/output/b",n::tensor({10,10},{2},n::DType::Float64));const auto zero=m::SmallImageCnnZPlus(p,image(),{0},hand_config());EXPECT_EQ(n::sum(zero.output_relevance).item(),1);EXPECT_EQ(n::sum(zero.input_relevance).item(),0);
}
TEST(SmallImageCnn, ActualFirstPoolWinnerRoutesAllRelevanceInATiedWindow) {
  m::SmallImageCnnConfig c;c.filters=1;c.kernel=1;c.hidden=1;c.pool_size=2;n::ParameterStore p(1,n::DType::Float64);const auto x=n::ones({1,2,2,1},n::DType::Float64);(void)m::SmallImageCnn(p,x,c);for(const auto& name:p.names())p.set(name,name.ends_with("/b")?n::zeros(p.at(name).shape(),n::DType::Float64):n::ones(p.at(name).shape(),n::DType::Float64));const auto r=m::SmallImageCnnZPlus(p,x,{0},c);close(r.conv_relevance,{1,0,0,0},1e-8);close(r.input_relevance,{1,0,0,0},1e-8);
}
TEST(SmallImageCnn, ExplanationNeverCreatesMissingLeavesAndChecksTargetsAndFiniteData) {
  n::ParameterStore empty(1);const auto x=n::ones({1,12,12,1});
  EXPECT_THROW(m::SmallImageCnnZPlus(empty,x),std::invalid_argument);
  EXPECT_EQ(empty.count(),0);
  (void)m::SmallImageCnn(empty,x);
  EXPECT_THROW(m::SmallImageCnnZPlus(empty,x,{2}),std::invalid_argument);
  EXPECT_THROW(m::SmallImageCnnZPlus(empty,x,{0,1}),std::invalid_argument);
  EXPECT_THROW(m::SmallImageCnnZPlus(empty,-x,{0}),std::invalid_argument);
  const auto predicted=m::SmallImageCnnZPlus(empty,x);EXPECT_EQ(predicted.target_ids.to_vector<std::int64_t>(),n::argmax(predicted.logits,1).to_vector<std::int64_t>());
  empty.set("cnn/output/b",n::tensor({std::numeric_limits<double>::infinity(),0},{2}));
  EXPECT_THROW(m::SmallImageCnnZPlus(empty,x,{0}),std::invalid_argument);
}
TEST(SmallImageCnn, ExplainingPreservesParametersInputGradientsAndCallerGradMode) {
  n::ParameterStore p(1,n::DType::Float64);hand_store(p);auto x=image();x.requires_grad_();n::sum(m::SmallImageCnn(p,x,hand_config()).logits).backward();std::vector<std::vector<double>> values,gradients;for(const auto& leaf:p.parameters()){values.push_back(leaf.to_vector<double>());gradients.push_back(leaf.grad().to_vector<double>());}const auto input_gradient=x.grad().to_vector<double>();
  {n::NoGradGuard guard;(void)m::SmallImageCnnZPlus(p,x,{0},hand_config());EXPECT_FALSE(n::is_grad_enabled());}
  EXPECT_TRUE(n::is_grad_enabled());const auto leaves=p.parameters();for(std::size_t i=0;i<leaves.size();++i){EXPECT_EQ(values[i],leaves[i].to_vector<double>());EXPECT_EQ(gradients[i],leaves[i].grad().to_vector<double>());}EXPECT_EQ(input_gradient,x.grad().to_vector<double>());
}
TEST(SmallImageCnn, InvalidModelGeometryAndDtypesRejectBeforeCreatingAnyLeaf) {
  n::ParameterStore p(1);const auto x=n::ones({1,3,3,1});m::SmallImageCnnConfig c;c.kernel=4;
  EXPECT_THROW(m::SmallImageCnn(p,x,c),std::invalid_argument);
  c={};c.filters=0;
  EXPECT_THROW(m::SmallImageCnn(p,x,c),std::invalid_argument);
  c={};c.pool_size=0;c.global_pool=true;
  EXPECT_THROW(m::SmallImageCnn(p,x,c),std::invalid_argument);
  EXPECT_THROW(m::SmallImageCnn(p,n::ones({1,3,3,1},n::DType::Float64)),std::invalid_argument);
  EXPECT_THROW(m::DenseImageBaseline(p,n::ones({1,3,3}),3,2),std::invalid_argument);
  c={};c.kernel=2;c.filters=std::numeric_limits<std::int64_t>::max();c.pool_size=1;
  EXPECT_THROW(m::SmallImageCnn(p,x,c),std::overflow_error);
  EXPECT_EQ(p.count(),0);
}
TEST(SmallImageCnn, ExplicitTensorAdamLearnsBarsAndUpdatesSharedConvolutionWeights) {
  n::BarImageOptions options;options.side=8;options.noise_stddev=.02;const auto data=n::make_bar_images(16,7,options);m::SmallImageCnnConfig c;c.filters=4;c.hidden=8;c.global_pool=true;n::ParameterStore p(1);(void)m::SmallImageCnn(p,data.images,c);const auto initial_kernel=p.at("cnn/conv/W").to_vector<float>();const auto initial=n::sparse_softmax_cross_entropy(m::SmallImageCnn(p,data.images,c).logits,data.labels).item();n::OptimizerConfig config;config.kind=n::OptimizerKind::Adam;n::Optimizer optimizer(p,.02,config);
  for(int step=0;step<80;++step)optimizer.minimize(n::sparse_softmax_cross_entropy(m::SmallImageCnn(p,data.images,c).logits,data.labels));n::NoGradGuard guard;const auto logits=m::SmallImageCnn(p,data.images,c).logits;const auto final=n::sparse_softmax_cross_entropy(logits,data.labels).item();EXPECT_LT(final,.5*initial);EXPECT_GE(n::accuracy(logits,data.labels),.9);EXPECT_NE(initial_kernel,p.at("cnn/conv/W").to_vector<float>());
}
TEST(SmallImageCnn, NativeImageDiagramRetainsSquareCellsAndRejectsInvalidPresentationData) {
  const auto picture=n::diagram_tensor_images(n::tensor({0,.25,.5,.75,1,2},{1,2,3,1}),1,{"six pixels"});const auto svg=picture.to_svg();std::smatch match;ASSERT_TRUE(std::regex_search(svg,match,std::regex("<polygon points=\"([^\"]+)\"")));auto coordinates=match[1].str();std::replace(coordinates.begin(),coordinates.end(),',',' ');std::istringstream input(coordinates);std::vector<double> points;for(double value;input>>value;)points.push_back(value);ASSERT_EQ(points.size(),8u);EXPECT_GT(std::fabs(points[2]-points[0]),0);EXPECT_NEAR(std::fabs(points[2]-points[0]),std::fabs(points[5]-points[3]),1e-10);EXPECT_NE(svg.find("six pixels"),std::string::npos);EXPECT_EQ(svg.find("data:image"),std::string::npos);
  EXPECT_THROW(n::diagram_tensor_images(n::ones({1,2,2,2})),std::invalid_argument);
  EXPECT_THROW(n::diagram_tensor_images(n::ones({2,2,2,1}),1,{"only one"}),std::invalid_argument);
  EXPECT_THROW(n::diagram_tensor_images(n::ones({1,2,2,1}),1,{},1,0),std::invalid_argument);
  EXPECT_THROW(n::diagram_tensor_images(n::tensor({std::numeric_limits<double>::quiet_NaN()},{1,1,1,1})),std::invalid_argument);
}
TEST(SmallImageCnn, ExecutedArchitectureDiagramsCountEveryActualLeafForAllReportedHeads) {
  const std::vector<std::int64_t> counts{6578,3330,258,4706};
  for(int variant=0;variant<4;++variant){n::ParameterStore p(1);m::SmallImageCnnConfig c;c.hidden=variant==0?32:16;c.global_pool=variant==2;n::Module module([&](const n::Tensor& input){return variant==3?m::DenseImageBaseline(p,input):m::SmallImageCnn(p,input,c).logits;},"Actual small image classifier");const auto diagram=module.plot(n::zeros({1,12,12,1}));EXPECT_EQ(diagram.parameter_count(),counts[variant]);EXPECT_EQ(p.count(),counts[variant]);EXPECT_EQ(diagram.to_svg().find("data:image/"),std::string::npos);}
}
