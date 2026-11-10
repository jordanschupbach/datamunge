#include <gtest/gtest.h>
#include <dm/dm.hpp>
namespace n=dm::nn;namespace m=n::models;
TEST(NnWaveGeneration, FullCausalHeadMatchesAllRingWrapsAndResetDilationCycles) {
 n::NoGradGuard guard;
 for(auto c:{m::WaveStackConfig{2,3,3,2,5},m::WaveStackConfig{4,8,5,1,256}}){
  n::ParameterStore p(541);m::WaveStack(p,n::one_hot(n::zeros({1,2},n::DType::Int32),c.classes),c);
  auto before=p.flatten();m::WaveStackCache cache(p,c);std::vector<std::int32_t> history;
  for(int step=0;step<90;++step){int code=(step*37+128)%c.classes;history.push_back(code);
   auto cached=cache.next(code);const auto start=std::max(0,int(history.size())-m::WaveStackField(c));
   std::vector<std::int32_t> tail(history.begin()+start,history.end());
   auto native=m::WaveStack(p,n::one_hot(n::from_vector(tail,{1,std::int64_t(tail.size())}),c.classes),c).logits;
   auto last=n::reshape(n::select(native,1,native.size(1)-1),{c.classes});
   EXPECT_LT(n::max(n::abs(last-cached)).item(),2e-6)<<step;
   EXPECT_EQ(cache.consumed(),step+1);EXPECT_FALSE(cached.requires_grad());EXPECT_EQ(cached.dtype(),n::DType::Float32);
  }
  EXPECT_EQ(p.flatten(),before);
 }
}
TEST(NnWaveGeneration, WeightSnapshotsIgnoreLaterWritesAndUnrelatedLeaves) {
 n::NoGradGuard guard;m::WaveStackConfig c{2,3,2,1,5};n::ParameterStore p(541);
 m::WaveStack(p,n::one_hot(n::zeros({1,2},n::DType::Int32),5),c);
 p.set("unrelated",n::full({1},NAN));m::WaveStackCache cache(p,c),reference(p,c);
 for(auto name:p.names())p.set(name,n::zeros(p.at(name).shape()));
 for(int code:{1,4,0,2,3,1,4,2})EXPECT_TRUE(n::equal_bits(cache.next(code),reference.next(code)));
}
TEST(NnWaveGeneration, NativeCloneAndCopyForkStateAndResetReplayExactly) {
 n::NoGradGuard guard;m::WaveStackConfig c{2,3,3,2,5};n::ParameterStore p(541);
 m::WaveStack(p,n::one_hot(n::zeros({1,1},n::DType::Int32),5),c);m::WaveStackCache a(p,c);
 for(int code:{1,2,0,4,3,4,0})a.next(code);auto b=a.clone();auto copy=a;
 EXPECT_EQ(b.consumed(),7);std::vector<n::Tensor> old;
 for(int step=0;step<50;++step){int code=step%5;auto head=a.next(code);old.push_back(head);
  EXPECT_TRUE(n::equal_bits(head,b.next(code)));EXPECT_TRUE(n::equal_bits(head,copy.next(code)));}
 auto values=old[0].to_vector<float>();b.reset();EXPECT_EQ(b.consumed(),0);EXPECT_EQ(a.consumed(),57);
 m::WaveStackCache fresh(p,c);for(int code:{4,3,0,1,2})EXPECT_TRUE(n::equal_bits(b.next(code),fresh.next(code)));
 EXPECT_EQ(old[0].to_vector<float>(),values);
}
TEST(NnWaveGeneration, MissingShapeDtypeNonfiniteAndArchitectureErrorsRejectWithoutFitting) {
 m::WaveStackConfig c{2,3,2,1,5};n::ParameterStore empty(541);
 EXPECT_THROW(m::WaveStackCache(empty,c),std::invalid_argument);EXPECT_EQ(empty.size(),0);
 auto invalid=c;invalid.levels=0;
 EXPECT_THROW(m::WaveStackCache(empty,invalid),std::invalid_argument);
 invalid=c;invalid.residual=std::numeric_limits<int>::max();
 EXPECT_THROW(m::WaveStackCache(empty,invalid),std::invalid_argument);
 n::ParameterStore fp64(541,n::DType::Float64);
 EXPECT_THROW(m::WaveStackCache(fp64,c),std::invalid_argument);
 {n::NoGradGuard guard;m::WaveStack(empty,n::one_hot(n::zeros({1,1},n::DType::Int32),5),c);}
 auto original=empty.at("output/b").detach().clone();empty.set("output/b",n::full({5},NAN));
 EXPECT_THROW(m::WaveStackCache(empty,c),std::invalid_argument);empty.set("output/b",original);
 auto mismatch=c;mismatch.classes=6;
 EXPECT_THROW(m::WaveStackCache(empty,mismatch),std::invalid_argument);
}
TEST(NnWaveGeneration, InvalidCodeDoesNotConsumeOrChangeAnyHistory) {
 n::NoGradGuard guard;m::WaveStackConfig c{2,3,3,1,5};n::ParameterStore p(541);
 m::WaveStack(p,n::one_hot(n::zeros({1,1},n::DType::Int32),5),c);m::WaveStackCache a(p,c);
 a.next(1);a.next(2);auto reference=a.clone();
 EXPECT_THROW(a.next(-1),std::invalid_argument);
 EXPECT_THROW(a.next(5),std::invalid_argument);EXPECT_EQ(a.consumed(),2);
 for(int code:{0,4,3,1,0,2,4})EXPECT_TRUE(n::equal_bits(a.next(code),reference.next(code)));
}
TEST(NnWaveGeneration, NonfiniteCandidateRollsBackBeforeAnyRingWriteOrPreviousCodeChange) {
 n::NoGradGuard guard;m::WaveStackConfig c{1,1,1,1,2};n::ParameterStore p(541);
 m::WaveStack(p,n::one_hot(n::zeros({1,1},n::DType::Int32),2),c);
 for(auto name:p.names())p.set(name,n::zeros(p.at(name).shape()));
 const float huge=std::numeric_limits<float>::max();
 p.set("input/W",n::from_vector(std::vector<float>{huge,0,huge,0},{1,2,2,1}));
 m::WaveStackCache a(p,c);a.next(0);auto reference=a.clone();
 EXPECT_THROW(a.next(0),std::invalid_argument);EXPECT_EQ(a.consumed(),1);
 EXPECT_TRUE(n::equal_bits(a.next(1),reference.next(1)));EXPECT_EQ(a.consumed(),2);
 a.reset();EXPECT_TRUE(n::equal_bits(a.next(0),n::zeros({2})));
 // A finite ring candidate followed by an overflowing output must also roll
 // back. Wrong ring/previous-code writes would change or overflow the next head.
 n::ParameterStore q(541);m::WaveStack(q,n::one_hot(n::zeros({1,1},n::DType::Int32),2),c);
 for(auto name:q.names())q.set(name,n::zeros(q.at(name).shape()));
 q.set("input/W",n::from_vector(std::vector<float>{.2f,.01f,10.f,.1f},{1,2,2,1}));
 q.set("block1/gate/W",n::from_vector(std::vector<float>{1,0,1,0},{1,2,1,2}));
 q.set("block1/skip/W",n::ones({1,1,1,1}));q.set("post/W",n::full({1,1,1,1},4));
 q.set("output/W",n::full({1,1,1,2},huge));
 m::WaveStackCache finite(q,c);auto first=finite.next(1);auto untouched=finite.clone();
 EXPECT_THROW(finite.next(0),std::invalid_argument);
 EXPECT_EQ(finite.consumed(),1);
 auto after=finite.next(1);EXPECT_TRUE(n::equal_bits(after,untouched.next(1)));
 EXPECT_GT(n::min(after).item(),1e37);finite.reset();EXPECT_TRUE(n::equal_bits(first,finite.next(1)));
}
TEST(NnWaveGeneration, SnapshotAndOwningHeadsOutliveStoreAndCache) {
 m::WaveStackConfig c{2,3,2,1,5};n::Tensor head;std::unique_ptr<m::WaveStackCache> cache;
 {n::ParameterStore p(541);m::WaveStack(p,n::one_hot(n::zeros({1,1},n::DType::Int32),5),c);
  cache=std::make_unique<m::WaveStackCache>(p,c);head=cache->next(3);}
 auto before=head.to_vector<float>();for(int code:{0,1,4,2,3,1})cache->next(code);
 EXPECT_EQ(head.to_vector<float>(),before);cache.reset();EXPECT_EQ(head.to_vector<float>(),before);EXPECT_FALSE(head.requires_grad());
}
TEST(NnWaveGeneration, GenericClassCountsAndSingleGateNeedNoFinalResidualLeaf) {
 n::NoGradGuard guard;
 for(int classes:{2,7,256}){m::WaveStackConfig c{1,2,1,1,classes};n::ParameterStore p(541);
  m::WaveStack(p,n::one_hot(n::zeros({1,1},n::DType::Int32),classes),c);
  EXPECT_FALSE(p.contains("block1/residual/W"));m::WaveStackCache cache(p,c);
  EXPECT_EQ(cache.next(classes-1).shape(),(n::Shape{classes}));
 }
}
TEST(NnWaveGeneration, StridedRequiredLeavesAreCopiedToAnIndependentContiguousSnapshot) {
 n::NoGradGuard guard;m::WaveStackConfig c{2,3,2,1,5};n::ParameterStore p(541),strided(0);
 m::WaveStack(p,n::one_hot(n::zeros({1,2},n::DType::Int32),5),c);
 for(auto name:p.names()){
  auto value=p.at(name);auto view=n::select(n::stack({value,value},value.dim()),value.dim(),0);
  EXPECT_FALSE(view.is_contiguous());strided.set(name,view);
 }
 m::WaveStackCache a(p,c),b(strided,c);
 for(int code:{0,1,4,2,3,4,0,2,4,1,0})EXPECT_TRUE(n::equal_bits(a.next(code),b.next(code)));
}
