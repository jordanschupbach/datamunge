#include <gtest/gtest.h>
#include <dm/dm.hpp>
namespace n=dm::nn;namespace m=n::models;
TEST(NnLogisticPixels, IntegerAreaRoundingStridesDtypesAndInvalidInputs) {
 auto raw=n::from_vector(std::vector<std::int32_t>{0,0,0,1,1,1,0,0,0,1,1,1},{1,2,2,3});
 for(auto x:{raw,n::cast(n::transpose(raw,{0,2,1,3}),n::DType::Int64)}){
  auto out=m::RgbAreaPixels(x,2);EXPECT_EQ(out.codes.shape(),(n::Shape{1,1,1,3}));
  for(auto c:out.codes.to_vector<std::int32_t>())EXPECT_EQ(c,1);
  for(auto c:out.images.to_vector<float>())EXPECT_EQ(c,float(1./127.5-1));
  EXPECT_FALSE(out.images.requires_grad());EXPECT_FALSE(out.codes.requires_grad());
 }
 EXPECT_THROW(m::RgbAreaPixels(n::zeros({1,4,4,3})),std::invalid_argument);
 EXPECT_THROW(m::RgbAreaPixels(raw,0),std::invalid_argument);
 EXPECT_THROW(m::RgbAreaPixels(raw,3),std::invalid_argument);
 EXPECT_THROW(m::RgbAreaPixels(n::full({1,4,4,3},256,n::DType::Int64)),std::invalid_argument);
 EXPECT_THROW(m::RgbAreaPixels(n::full({1,4,4,3},-1,n::DType::Int32)),std::invalid_argument);
 EXPECT_THROW(m::RgbAreaPixels(n::zeros({1,4,4,1},n::DType::Int32)),std::invalid_argument);
 auto rectangular=m::RgbAreaPixels(n::full({2,4,6,3},255,n::DType::Int64),2);
 EXPECT_EQ(rectangular.images.shape(),(n::Shape{2,2,3,3}));EXPECT_EQ(n::min(rectangular.images).item(),1);
}
TEST(NnLogisticPixels, EveryByteBinAndInfiniteTailNormalizeForBothFloatingTypes) {
 std::vector<double> x(256*3);for(int i=0;i<256;++i)for(int c=0;c<3;++c)x[i*3+c]=double(i)/127.5-1;
 for(auto t:{n::DType::Float32,n::DType::Float64}){
  auto out=m::DiscretizedLogisticRGB(n::cast(n::from_vector(x,{256,3}),t),n::zeros({256,10},t),1);
  auto totals=n::sum(n::exp(out.channel_log_mass),{0}).to_vector<double>();
  for(double mass:totals)EXPECT_NEAR(mass,1,2e-5);
  EXPECT_EQ(out.pixel_log_probability.shape(),(n::Shape{256}));EXPECT_EQ(out.means.shape(),(n::Shape{256,1,3}));
  EXPECT_EQ(out.mean_channel_nll.dtype(),t);
 }
}
TEST(NnLogisticPixels, AnalyticCenterMassEndpointTailsAndStoppedMaximumKeepHigherPaths) {
 auto t=n::DType::Float64;auto p=n::zeros({1,10},t).requires_grad_();auto x=n::zeros({1,3},t).requires_grad_();
 auto out=m::DiscretizedLogisticRGB(x,p,1);const double step=float(1.f/255);
 const double expected=-std::log(2/(1+std::exp(-step))-1);
 EXPECT_NEAR(out.mean_channel_nll.item(),expected,1e-12);
 n::GradOptions go;go.create_graph=true;
 auto first=n::grad(out.mean_channel_nll,{p,x},go);
 EXPECT_TRUE(first[0].defined());EXPECT_TRUE(first[1].defined());
 auto second=n::grad(n::sum(first[0]),{p})[0];EXPECT_TRUE(second.defined());
 for(double value:second.to_vector<double>())EXPECT_TRUE(std::isfinite(value));
 auto edge=m::DiscretizedLogisticRGB(n::full({1,3},-1,t),n::zeros({1,10},t),1);
 EXPECT_NEAR(edge.mean_channel_nll.item(),std::log1p(std::exp(1-step)),1e-12);
}
TEST(NnLogisticPixels, ParameterGradientsCoupledMeansAndClippedScalesMatchFiniteDifferences) {
 auto t=n::DType::Float64;std::vector<double> values(30);for(int i=0;i<30;++i)values[i]=.3*std::sin(i*.37);
 values[4]=-8;values[14]=-.4;values[24]=2;
 auto p=n::from_vector(values,{1,30}).requires_grad_();auto x=n::tensor({-.6,.2,.8},{1,3},t);
 auto out=m::DiscretizedLogisticRGB(x,p);auto gradient=n::grad(out.mean_channel_nll,{p})[0].to_vector<double>();
 auto means=out.means.to_vector<double>();EXPECT_NEAR(means[1],values[2]+std::tanh(values[7])*(-.6),1e-14);
 EXPECT_NEAR(means[2],values[3]+std::tanh(values[8])*(-.6)+std::tanh(values[9])*.2,1e-14);
 EXPECT_EQ(gradient[4],0);
 for(int j=0;j<30;++j){auto hi=values,lo=values;hi[j]+=1e-5;lo[j]-=1e-5;
  double fd=(m::DiscretizedLogisticRGB(x,n::from_vector(hi,{1,30})).mean_channel_nll.item()-m::DiscretizedLogisticRGB(x,n::from_vector(lo,{1,30})).mean_channel_nll.item())/2e-5;
  EXPECT_NEAR(gradient[j],fd,2e-7)<<j;
 }
}
TEST(NnLogisticPixels, LikelihoodValidatesShapesDtypesRangeAndNonfiniteFields) {
 auto t=n::DType::Float64;auto x=n::zeros({2,3},t),p=n::zeros({2,30},t);
 EXPECT_THROW(m::DiscretizedLogisticRGB(x,p,0),std::invalid_argument);
 EXPECT_THROW(m::DiscretizedLogisticRGB(x,n::zeros({1,30},t)),std::invalid_argument);
 EXPECT_THROW(m::DiscretizedLogisticRGB(x,n::zeros({2,20},t)),std::invalid_argument);
 EXPECT_THROW(m::DiscretizedLogisticRGB(x,n::zeros({2,30})),std::invalid_argument);
 EXPECT_THROW(m::DiscretizedLogisticRGB(n::full({2,3},1.001,t),p),std::invalid_argument);
 EXPECT_THROW(m::DiscretizedLogisticRGB(x,n::full({2,30},INFINITY,t)),std::invalid_argument);
 EXPECT_THROW(m::DiscretizedLogisticRGB(x,n::full({2,30},10000,t)),std::invalid_argument);
 EXPECT_EQ(m::DiscretizedLogisticRGB(n::zeros({3},t),n::zeros({10},t),1).pixel_log_probability.shape(),(n::Shape{1}));
 auto strided=m::DiscretizedLogisticRGB(n::transpose(n::zeros({2,4,3},t),{1,0,2}),n::transpose(n::zeros({2,4,30},t),{1,0,2}));
 EXPECT_EQ(strided.pixel_log_probability.shape(),(n::Shape{8}));
}
TEST(NnLogisticPixels, CompactModelCensusInvalidBeforeLeavesAndEveryWholePixelCausalPath) {
 auto t=n::DType::Float64;n::ParameterStore p(8609,t);
 EXPECT_THROW(m::PixelCnnPP(p,n::full({1,2,3,3},1.1,t)),std::invalid_argument);EXPECT_EQ(p.size(),0);
 auto x=n::full({1,2,3,3},.25,t).requires_grad_();auto out=m::PixelCnnPP(p,x);
 EXPECT_EQ(p.count(),1410);EXPECT_EQ(p.size(),23);EXPECT_EQ(out.parameters.shape(),(n::Shape{1,2,3,30}));
 EXPECT_NEAR(out.per_image_nll.item()/18,out.loss.item(),1e-12);
 n::GradOptions keep;keep.retain_graph=true;
 for(int site=0;site<6;++site){
  auto gradient=n::grad(n::sum(n::select(n::reshape(out.parameters,{1,6,30}),1,site)),{x},keep)[0].to_vector<double>();
  for(int j=site*3;j<18;++j)EXPECT_EQ(gradient[j],0);
  auto changed=n::where(n::greater_equal(n::reshape(n::arange(0,6),{1,2,3,1}),site),-x,x);
  auto head=m::PixelCnnPP(p,changed).parameters;
  EXPECT_TRUE(n::equal_bits(n::slice_axis(n::reshape(head,{1,6,30}),1,0,site+1),n::slice_axis(n::reshape(out.parameters,{1,6,30}),1,0,site+1)));
 }
}
TEST(NnLogisticPixels, MaskedContextCoordinatesStayFixedAcrossActualAdamWrites) {
 n::ParameterStore p(8609);auto x=n::full({2,2,3,3},.2);m::PixelCnnPP(p,x);auto initial=p.flatten();
 n::Optimizer optimizer(p,.003,{n::OptimizerKind::Adam});std::map<std::string,n::Tensor> masks;
 for(int block=0;block<3;++block)for(bool vertical:{false,true}){
  std::string key="block"+std::to_string(block)+(vertical?"/vertical/W":"/horizontal/W");
  masks[key]=m::GatedPixelMask(block?4:3,vertical,block==0);
 }
 int blocked=0;for(auto& [key,mask]:masks)for(float v:mask.to_vector<float>())if(v==0)++blocked;EXPECT_EQ(blocked,448);
 for(int step=0;step<3;++step){auto out=m::PixelCnnPP(p,x);auto grads=n::grad(out.loss,p.parameters());optimizer.step(grads);}
 for(auto& [key,mask]:masks){auto blocked_mask=n::equal(mask,0);
  // Recreate the exact seeded initializer to compare forbidden coordinates.
  n::ParameterStore fresh(8609);m::PixelCnnPP(fresh,x);
  EXPECT_TRUE(n::equal_bits(n::where(blocked_mask,p.at(key),n::zeros_like(mask)),n::where(blocked_mask,fresh.at(key),n::zeros_like(mask))));
 }
 EXPECT_NE(p.flatten(),initial);
}
TEST(NnLogisticPixels, FlatRecordsRetainGraphsBeyondParentAndStoreLifetime) {
 auto t=n::DType::Float64;auto image=n::full({1,2,2,3},.25,t).requires_grad_();n::Tensor loss,means;
 {n::ParameterStore p(8609,t);auto out=m::PixelCnnPP(p,image);loss=out.loss;means=out.means;}
 EXPECT_EQ(means.shape(),(n::Shape{4,3,3}));n::GradOptions go;go.create_graph=true;
 auto first=n::grad(loss,{image},go)[0];auto second=n::grad(n::sum(first),{image})[0];
 for(double v:second.to_vector<double>())EXPECT_TRUE(std::isfinite(v));
}
