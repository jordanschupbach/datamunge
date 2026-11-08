#include <datamunge/nn/image_explainers.hpp>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/ops.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
namespace {
namespace n=datamunge::nn;
void close(const n::Tensor& x,const std::vector<double>& expected,double tolerance=1e-10){const auto actual=x.to_vector<double>();
    ASSERT_EQ(actual.size(),expected.size());for(std::size_t i=0;i<actual.size();++i)EXPECT_NEAR(actual[i],expected[i],tolerance);}
n::ImagePatchOptions line(int p){n::ImagePatchOptions o;o.patch_rows=1;o.patch_cols=p;o.batch_size=3;return o;}
n::Tensor image(int p){return n::ones({1,1,p,1},n::DType::Float64);}
n::Tensor flags(const std::vector<double>& values,std::int64_t rows,std::int64_t p){return n::cast(n::from_vector(values,{rows,p}),n::DType::Bool);}
n::ImageScoreBatch game(const std::vector<double>& scores,int p){return [scores,p](const n::Tensor& x){auto values=x.to_vector<double>();std::vector<double> out(x.size(0));for(std::size_t row=0;row<out.size();++row){std::size_t mask=0;for(int j=0;j<p;++j)if(values[row*p+j]>.5)mask|=std::size_t{1}<<j;out[row]=scores.at(mask);}return n::from_vector(out);};}
std::vector<double> permutation_oracle(const std::vector<double>& scores,int p){
    std::vector<int> order(p);std::iota(order.begin(),order.end(),0);std::vector<long double> total(p,0);std::size_t count=0;
    do{std::size_t mask=0;for(int j:order){auto next=mask|(std::size_t{1}<<j);total[j]+=static_cast<long double>(scores[next])-scores[mask];mask=next;}++count;}while(std::next_permutation(order.begin(),order.end()));
    std::vector<double> out;for(auto t:total)out.push_back(t/count);return out;
}
std::vector<double> independent_ridge(const n::TensorImageLime& fit,double ridge){
    // Long-double pivoted elimination is independent of tensor Gram/Cholesky kernels.
    const auto z=fit.masks.to_vector<bool>();const auto y=fit.scores.to_vector<double>();const auto w=fit.proximity_weights.to_vector<double>();const auto p=fit.masks.size(1),d=p+1,N=fit.masks.size(0);
    std::vector<std::vector<long double>> a(d,std::vector<long double>(d+1,0));
    for(std::int64_t row=0;row<N;++row){std::vector<long double> x(d,1);for(std::int64_t j=0;j<p;++j)x[j+1]=z[row*p+j];for(std::int64_t i=0;i<d;++i){a[i][d]+=w[row]*x[i]*y[row];for(std::int64_t j=0;j<d;++j)a[i][j]+=w[row]*x[i]*x[j];}}
    for(std::int64_t j=1;j<d;++j)a[j][j]+=ridge;
    for(std::int64_t i=0;i<d;++i){auto pivot=i;for(auto j=i+1;j<d;++j)if(std::abs(a[j][i])>std::abs(a[pivot][i]))pivot=j;std::swap(a[i],a[pivot]);auto divisor=a[i][i];for(auto j=i;j<=d;++j)a[i][j]/=divisor;for(std::int64_t row=0;row<d;++row)if(row!=i){auto factor=a[row][i];for(auto j=i;j<=d;++j)a[row][j]-=factor*a[i][j];}}
    std::vector<double> out;for(auto& row:a)out.push_back(row[d]);return out;
}
}
TEST(TensorImageMask, UnevenGridChannelsSignedValuesAndOwnership){
    n::ImagePatchOptions o;o.patch_rows=2;o.patch_cols=2;o.baseline=-3;
    std::vector<double> v(3*5*2);std::iota(v.begin(),v.end(),-10);auto x=n::from_vector(v,{1,3,5,2});auto z=flags({1,0,0,1,0,1,1,0},2,4);
    auto result=n::mask_image_patches(x,z,o);auto actual=result.to_vector<double>();
    for(int b=0;b<2;++b)for(int row=0;row<3;++row)for(int col=0;col<5;++col)for(int c=0;c<2;++c){int patch=(row*2/3)*2+col*2/5;bool on=b?(patch==1||patch==2):(patch==0||patch==3);
    EXPECT_EQ(actual[((b*3+row)*5+col)*2+c],on?v[(row*5+col)*2+c]:-3);}
    x.zero_();z.zero_();
    EXPECT_EQ(result.to_vector<double>(),actual);
}
TEST(TensorImageMask, NoncontiguousInputsAndFloat32BaselineRounding){
    auto base=n::tensor({1,2,3,4,5,6},{1,2,3,1});auto x=n::transpose(base,{0,2,1,3});auto o=line(2);o.baseline=.1;
    auto y=n::mask_image_patches(x,flags({1,0},1,2),o);close(y,{1,double(float(.1)),2,double(float(.1)),3,double(float(.1))},1e-7);
}
TEST(TensorShapley, TinyInteractionSignedDummyAndActualQueryOrder){
    std::vector<double> scores{0,2,3,11,-1,1,2,10};auto o=line(3);std::size_t seen=0,calls=0;
    auto f=[&](const n::Tensor& x){EXPECT_FALSE(n::is_grad_enabled());auto z=x.to_vector<double>();for(std::int64_t row=0;row<x.size(0);++row){for(int p=0;p<3;++p)EXPECT_EQ(z[row*3+p],double(((seen+row)>>p)&1));}seen+=x.size(0);++calls;return game(scores,3)(x);};
    const auto e=n::exact_image_shapley(image(3),f,o);close(e.patch_values,{5,6,-1});close(e.coalition_scores,scores);
    EXPECT_EQ(e.work.evaluations,8);
    EXPECT_EQ(e.work.predict_calls,3);
    EXPECT_EQ(e.work.marginal_terms,12);
    EXPECT_EQ(calls,3);
    EXPECT_EQ(seen,8);
    EXPECT_NEAR(e.efficiency_residual,0,1e-14);
}
TEST(TensorShapley, RandomGamesMatchIndependentPermutationAverage){
    std::mt19937_64 rng(20261107);
    for(int fixture=0;fixture<224;++fixture){int p=1+fixture%7;std::vector<double> scores(std::size_t{1}<<p);for(auto& v:scores)v=double(int(rng()%2001)-1000)/16.;auto o=line(p);o.batch_size=1+rng()%19;
        auto e=n::exact_image_shapley(image(p),game(scores,p),o);close(e.patch_values,permutation_oracle(scores,p),2e-12);close(e.coalition_scores,scores);
    EXPECT_NEAR(e.efficiency_residual,0,2e-12);
    EXPECT_EQ(e.work.evaluations,scores.size());
    EXPECT_EQ(e.work.predict_calls,(scores.size()+o.batch_size-1)/o.batch_size);
    EXPECT_EQ(e.work.marginal_terms,p*scores.size()/2);}
}
TEST(TensorShapley, EverySevenPlayerUnanimityGameSatisfiesSymmetryAndDummy){
    const int p=7;for(unsigned group=1;group<128;++group){std::vector<double> scores(128);for(unsigned s=0;s<128;++s)scores[s]=(s&group)==group?1:0;
        const auto e=n::exact_image_shapley(image(p),game(scores,p),line(p));auto phi=e.patch_values.to_vector<double>();int members=std::popcount(group);for(int j=0;j<p;++j)EXPECT_NEAR(phi[j],group&(1u<<j)?1./members:0,1e-13);}
}
TEST(TensorShapley, AdditivityAndBaselineChangeTheSpecifiedGame){
    const auto x=n::tensor({2,3},{1,1,2,1},n::DType::Float64);auto o=line(2);o.baseline=1;
    auto f=[](const n::Tensor& x){auto z=n::reshape(x,{x.size(0),2});return 3+2*n::select(z,1,0)-4*n::select(z,1,1);};
    auto e=n::exact_image_shapley(x,f,o);close(e.patch_values,{2,-8});
    EXPECT_EQ(e.base_value,1);
    EXPECT_EQ(e.full_value,-5);
    auto doubled=n::exact_image_shapley(x,[&](const n::Tensor& y){return 2*f(y)+7;},o);close(doubled.patch_values,{4,-16});
    EXPECT_EQ(doubled.base_value,9);
}
TEST(TensorShapley, QuotaAndShapeErrorsHappenBeforePrediction){
    std::size_t calls=0;n::ImageScoreBatch f=[&](const n::Tensor& x){++calls;return n::sum(x,{1,2,3});};auto o=line(3);o.max_evaluations=7;
    EXPECT_THROW(n::exact_image_shapley(image(3),f,o),std::length_error);
    EXPECT_EQ(calls,0);
    o.max_evaluations=8;
    EXPECT_EQ(n::exact_image_shapley(image(3),f,o).work.evaluations,8);
    o.max_evaluations=0;
    EXPECT_EQ(n::exact_image_shapley(image(3),f,o).work.evaluations,8);
    calls=0;auto big=line(21);
    EXPECT_THROW(n::exact_image_shapley(image(21),f,big),std::invalid_argument);
    EXPECT_EQ(calls,0);
    o.batch_size=0;
    EXPECT_THROW(n::exact_image_shapley(image(3),f,o),std::invalid_argument);
    EXPECT_EQ(calls,0);
}
TEST(TensorImageInputs, RejectEmptyMalformedGridDataBaselineAndMasks){
    const auto x=image(2);auto o=line(2);auto f=[](const n::Tensor& z){return n::sum(z,{1,2,3});};
    EXPECT_THROW(n::exact_image_shapley(n::Tensor{},f,o),std::invalid_argument);
    EXPECT_THROW(n::exact_image_shapley(n::ones({2,1,2,1}),f,o),std::invalid_argument);
    EXPECT_THROW(n::exact_image_shapley(n::ones({1,2}),f,o),std::invalid_argument);
    EXPECT_THROW(n::exact_image_shapley(n::ones({1,1,2,1},n::DType::Int64),f,o),std::invalid_argument);
    EXPECT_THROW(n::exact_image_shapley(n::full({1,1,2,1},std::numeric_limits<double>::quiet_NaN(),n::DType::Float64),f,o),std::invalid_argument);
    for(int rows:{0,2}){o.patch_rows=rows;
    EXPECT_THROW(n::exact_image_shapley(x,f,o),std::invalid_argument);}
    o=line(2);o.patch_cols=3;
    EXPECT_THROW(n::exact_image_shapley(x,f,o),std::invalid_argument);
    o=line(2);o.baseline=std::numeric_limits<double>::infinity();
    EXPECT_THROW(n::exact_image_shapley(x,f,o),std::invalid_argument);
    o.baseline=std::numeric_limits<double>::max();
    EXPECT_THROW(n::exact_image_shapley(n::ones({1,1,2,1}),f,o),std::invalid_argument);
    o=line(2);
    EXPECT_THROW(n::mask_image_patches(x,n::ones({1,2}),o),std::invalid_argument);
    EXPECT_THROW(n::mask_image_patches(x,flags({1},1,1),o),std::invalid_argument);
    EXPECT_THROW(n::exact_image_shapley(x,{},o),std::invalid_argument);
}
TEST(TensorImageCallbacks, ValidateOutputAndPropagateExternalExceptions){
    auto o=line(2);const auto x=image(2);
    for(int mode=0;mode<5;++mode){auto bad=[mode](const n::Tensor& b){if(mode==0)return n::scalar(0);if(mode==1)return n::zeros({b.size(0),1});if(mode==2)return n::zeros({b.size(0)},n::DType::Int64);if(mode==3)return n::full({b.size(0)},std::numeric_limits<double>::infinity());return n::Tensor{};};
    EXPECT_THROW(n::exact_image_shapley(x,bad,o),std::invalid_argument);}
    std::size_t calls=0;n::ImageScoreBatch throwing=[&](const n::Tensor& b){if(++calls==2)throw std::logic_error("callback state");return n::sum(b,{1,2,3});};
    EXPECT_THROW(n::exact_image_shapley(x,throwing,o),std::logic_error);
    EXPECT_EQ(calls,2);
    EXPECT_TRUE(n::is_grad_enabled());
    auto extreme=[](const n::Tensor& b){return (2*n::sum(b,{1,2,3})-1)*(std::numeric_limits<double>::max()/2);};
    // Scores can be finite while the full-minus-empty difference is not.
    o=line(1);
    EXPECT_THROW(n::exact_image_shapley(image(1),[&](const n::Tensor& b){return (2*n::sum(b,{1,2,3})-1)*std::numeric_limits<double>::max();},o),std::overflow_error);
}
TEST(TensorImageCallbacks, NoGradientMutationAndOwnedScoresSurviveReuse){
    auto x=image(2);x.requires_grad_();x.set_grad(n::full(x.shape(),7,n::DType::Float64));auto saved=x.to_vector<double>();auto o=line(2);o.batch_size=1;auto shared=n::zeros({1},n::DType::Float64);int calls=0;
    auto e=n::exact_image_shapley(x,[&](const n::Tensor& b){EXPECT_FALSE(n::is_grad_enabled());shared.fill_(calls++);return shared;},o);
    close(e.coalition_scores,{0,1,2,3});shared.fill_(99);close(e.coalition_scores,{0,1,2,3});close(x.grad(),{7,7});
    EXPECT_EQ(x.to_vector<double>(),saved);
    EXPECT_FALSE(e.patch_values.requires_grad());
    EXPECT_FALSE(e.coalition_scores.requires_grad());
}
TEST(TensorLime, ActualSamplesWeightsFitMatchIndependentPivotedSolver){
    const std::vector<double> scores{0,2,3,11,-1,1,2,10};
    for(int seed=0;seed<16;++seed){n::ImageLimeOptions o;o.patches=line(3);o.samples=64;o.seed=seed;o.kernel_width=.25+seed*.05;o.ridge=.01;
        const auto e=n::lime_image(image(3),game(scores,3),o);const auto expected=independent_ridge(e,o.ridge);
    EXPECT_NEAR(e.intercept,expected[0],1e-10);close(e.patch_values,{expected[1],expected[2],expected[3]},1e-10);
        auto z=e.masks.to_vector<bool>();auto y=e.scores.to_vector<double>();auto w=e.proximity_weights.to_vector<double>();for(int p=0;p<3;++p){EXPECT_TRUE(z[p]);
    EXPECT_FALSE(z[3+p]);}
        for(int row=0;row<64;++row){int off=0,mask=0;for(int p=0;p<3;++p){off+=!z[row*3+p];if(z[row*3+p])mask|=1<<p;}EXPECT_EQ(y[row],scores[mask]);
    EXPECT_NEAR(w[row],std::exp(-std::pow(off/3./o.kernel_width,2)),1e-15);}
        EXPECT_EQ(e.work.evaluations,64);
    EXPECT_EQ(e.work.predict_calls,22);
    EXPECT_EQ(e.work.marginal_terms,0);
    EXPECT_EQ(e.full_value,10);
    EXPECT_NEAR(e.fitted_full,e.intercept+n::sum(e.patch_values).item(),1e-14);
    }
}
TEST(TensorLime, AdditiveKnownCoefficientsAndBatchInvariance){
    auto f=[](const n::Tensor& x){auto z=n::reshape(x,{x.size(0),3});return 7+2*n::select(z,1,0)-4*n::select(z,1,1)+n::select(z,1,2);};n::ImageLimeOptions o;o.patches=line(3);o.samples=512;o.kernel_width=1;o.ridge=1e-8;
    const auto a=n::lime_image(image(3),f,o);close(a.patch_values,{2,-4,1},1e-8);
    EXPECT_NEAR(a.intercept,7,1e-8);
    EXPECT_NEAR(a.r_squared,1,1e-12);
    o.patches.batch_size=1000;const auto b=n::lime_image(image(3),f,o);
    EXPECT_EQ(a.masks.to_vector<bool>(),b.masks.to_vector<bool>());close(a.patch_values,b.patch_values.to_vector<double>(),1e-13);
    EXPECT_EQ(b.work.predict_calls,1);
    auto constant=n::lime_image(image(3),[](const n::Tensor& x){return n::full({x.size(0)},3,n::DType::Float64);},o);
    EXPECT_EQ(constant.r_squared,0);close(constant.patch_values,{0,0,0},1e-11);
    EXPECT_NEAR(constant.intercept,3,1e-12);
}
TEST(TensorLime, InvalidFitOptionsQuotaGeometryAndCallbackReject){
    auto f=[](const n::Tensor& x){return n::sum(x,{1,2,3});};n::ImageLimeOptions o;o.patches=line(2);o.samples=8;
    for(double v:{0.,-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}){o.kernel_width=v;
    EXPECT_THROW(n::lime_image(image(2),f,o),std::invalid_argument);}
    o.kernel_width=.25;for(double v:{0.,-1.,std::numeric_limits<double>::infinity()}){o.ridge=v;
    EXPECT_THROW(n::lime_image(image(2),f,o),std::invalid_argument);}
    o.ridge=.001;o.samples=2;
    EXPECT_THROW(n::lime_image(image(2),f,o),std::invalid_argument);
    o.samples=8;o.patches.max_evaluations=7;
    EXPECT_THROW(n::lime_image(image(2),f,o),std::length_error);
    o.patches.max_evaluations=0;o.samples=std::numeric_limits<std::size_t>::max();
    EXPECT_THROW(n::lime_image(image(2),f,o),std::length_error);
    o.samples=8;
    EXPECT_THROW(n::lime_image(image(2),[](const n::Tensor& x){return n::zeros({x.size(0),1});},o),std::invalid_argument);
    o.kernel_width=std::numeric_limits<double>::min();
    EXPECT_NO_THROW(n::lime_image(image(2),f,o));
}
TEST(TensorAttributionMaps, UnevenAreasConserveSignedPatchScoresAndOptionalRepeat){
    auto p=n::tensor({2,-3,5,-7},{2,2},n::DType::Float64);auto a=n::expand_image_attributions(p,3,5);auto b=n::expand_image_attributions(p,3,5,false);auto v=a.to_vector<double>();auto repeated=b.to_vector<double>();const std::vector<double> areas{6,4,3,2};
    for(int row=0;row<3;++row)for(int col=0;col<5;++col){int patch=(row*2/3)*2+col*2/5;
    EXPECT_EQ(repeated[row*5+col],p.to_vector<double>()[patch]);
    EXPECT_NEAR(v[row*5+col],p.to_vector<double>()[patch]/areas[patch],1e-15);}
    EXPECT_NEAR(n::sum(a).item(),-3,1e-14);p.zero_();
    EXPECT_NEAR(n::sum(a).item(),-3,1e-14);
    EXPECT_THROW(n::expand_image_attributions(n::ones({2,2}),1,3),std::invalid_argument);
    EXPECT_THROW(n::expand_image_attributions(n::ones({2,2}),std::numeric_limits<std::int64_t>::max(),2),std::length_error);
}
TEST(TensorAttributionMaps, SignedCorrelationClippingUndefinedAndExtremeMagnitudes){
    auto a=n::tensor({-2,-1,0,1,2},{5},n::DType::Float64);
    EXPECT_NEAR(n::attribution_correlation(a,a),1,1e-15);
    EXPECT_NEAR(n::attribution_correlation(a,-a),-1,1e-15);
    EXPECT_NEAR(n::attribution_correlation(a,7+3*a),1,1e-15);
    EXPECT_TRUE(std::isnan(n::attribution_correlation(a,n::ones({5}))));
    EXPECT_TRUE(std::isnan(n::attribution_correlation(-n::ones({5}),a,true)));
    EXPECT_NEAR(n::attribution_correlation(a,a*(std::numeric_limits<double>::max()/4)),1,1e-15);
    EXPECT_THROW(n::attribution_correlation(a,n::ones({1,5})),std::invalid_argument);
}

TEST(TensorAttributionMaps, NativeSignedAndIntensityKeysHaveOwnedGeometry){
    auto x=n::tensor({-2,0,2,0,0.5,1},{2,1,3,1},n::DType::Float64);
    const auto figure=n::diagram_image_maps(x,2,{"signed","intensity"},{true,false});
    const auto svg=figure.to_svg();
    EXPECT_NE(svg.find("rgb(128,128,128)"),std::string::npos);
    EXPECT_FALSE(svg.empty());
    EXPECT_NE(svg.find("signed"),std::string::npos);
    EXPECT_NE(svg.find("intensity"),std::string::npos);
    EXPECT_EQ(svg.find("data:image"),std::string::npos);x.zero_();
    EXPECT_EQ(figure.to_svg(),svg);
    EXPECT_THROW(n::diagram_image_maps(n::zeros({1,2,2,1}),0),std::invalid_argument);
    EXPECT_THROW(n::diagram_image_maps(n::zeros({1,2,2,1}),1,{"a","b"}),std::invalid_argument);
    EXPECT_THROW(n::diagram_image_maps(n::zeros({1,1001,1000,1})),std::length_error);
}
