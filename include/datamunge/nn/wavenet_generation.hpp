#pragma once
/// @file wavenet_generation.hpp
/// @brief Owning Float32 per-layer circular state for causal WaveNet next-code inference.
#include "datamunge/nn/wavenet_stack_models.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>

namespace datamunge::nn::models {
/// @brief Own fixed WaveStack weights and circular histories for one categorical code stream.
///
/// Snapshot required finite Float32 CPU leaves from a warmed ParameterStore;
/// later store writes/collection cannot affect this cache. Each block retains
/// its OWN input at its dilation, not a raw-code history or another block's
/// residual. Input uses previous/current code taps; the last residual projection
/// is absent. Skip summation, ReLU/post/ReLU/head match WaveStack. One next(code)
/// consumes one code and returns the logits for the following code. The source
/// scalar Float32 multiply/add order and tanh(filter)/(1.f+exp(-gate)) arithmetic
/// are retained; cached/full-convolution results may differ by roundoff.
///
/// Fixed weights and mutable state own all storage. C++ copies and clone() fork
/// independent streams at the current position. Output Tensors own detached
/// storage and outlive the cache. No fitting, optimizer, RNG, categorical draw,
/// codec, parameter writes or autograd path. Distinct caches are independent;
/// callers synchronize access to one cache. Reset restores the true beginning
/// with zero feature histories and no previous code, not code-zero padding.
/// Memory is copied weights plus R*cycles*(2^levels-1) floats for rings. Per-code
/// work is O(R*classes+blocks*(R^2+R*skip)+skip^2+skip*classes), independent of
/// generated length; scratch is O(blocks*R+skip+classes). Failed invalid-code,
/// count-limit or nonfinite-candidate steps preserve the previous state.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;namespace m=n::models;
/// m::WaveStackConfig c{4,8,5,1,256};n::ParameterStore p(541);
/// m::WaveStack(p,n::one_hot(n::zeros({1,2},n::DType::Int32),256),c);
/// m::WaveStackCache cache(p,c);auto logits=cache.next(128);
/// // Owning detached Float32[256]; cache.consumed()==1. Choose/feed the next code explicitly.
/// @endcode
/// @par References
/// - van den Oord et al.(2016), WaveNet: A Generative Model for Raw Audio,
/// https://arxiv.org/abs/1609.03499 . Circular block-input histories implement
/// the repository's two-tap unconditional teaching stack, not conditioned TTS.
/// @see WaveStack(), WaveStackField(), WaveNll(), next(), reset(), clone()
class WaveStackCache {
 public:
  /// @brief Snapshot the required Float32 leaves and initialize empty per-layer histories.
  ///
  /// Store/config must describe the already-warmed WaveStack architecture.
  /// Required keys are input, block1..L/gate, block1..L/skip, block1..L-1/residual,
  /// post and output, each /W and /b. W shape[1,kernel,in,out], b[out]; input/gate
  /// kernel2, others1. Unrelated store leaves are ignored. Strided CPU leaves
  /// are copied contiguously, preserving Float32 values; no leaf or RNG is created.
  /// Validate widths, levels1..12,cycles1..4,classes>=2 and representable2*R.
  /// Invalid construction publishes no partial object. Initial consumed()==0.
  /// @param parameters Read-only warmed Float32 store with all required finite CPU leaves.
  /// @param config Explicit matching WaveStackConfig, copied by value; no implicit architecture inference.
  /// @throws std::invalid_argument For invalid config, missing/misshaped/nonfinite/non-CPU or non-Float32 leaves.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// namespace n=dm::nn;namespace m=n::models;
  /// m::WaveStackConfig c{1,2,2,1,3};n::ParameterStore p(17);
  /// m::WaveStack(p,n::one_hot(n::zeros({1,1},n::DType::Int32),3),c);
  /// m::WaveStackCache cache(p,c); // Copies learned values; store and cache are independent.
  /// @endcode
  /// @par References
  /// - van den Oord et al.(2016), https://arxiv.org/abs/1609.03499 . Matches the
  /// canonical teaching WaveStack checkpoint names and two-tap architecture.
  /// @see WaveStackCache, WaveStack(), next(), reset()
  WaveStackCache(const ParameterStore& parameters,const WaveStackConfig& config):config_(config) {
    (void)WaveStackField(config_);
    if(parameters.dtype()!=DType::Float32 || config_.residual>std::numeric_limits<int>::max()/2)
      throw std::invalid_argument("WaveStackCache requires Float32 weights and representable doubled residual width");
    const auto r=config_.residual,s=config_.skip,c=config_.classes,blocks=config_.levels*config_.cycles;
    Layer(parameters,"input",2,c,r);
    for(int i=0;i<blocks;++i){
      const auto name="block"+std::to_string(i+1);
      Layer(parameters,name+"/gate",2,r,2*r);Layer(parameters,name+"/skip",1,r,s);
      if(i+1<blocks)Layer(parameters,name+"/residual",1,r,r);
    }
    Layer(parameters,"post",1,s,s);Layer(parameters,"output",1,s,c);
    for(int i=0;i<blocks;++i)ring_.emplace_back(static_cast<std::size_t>(1<<(i%config_.levels))*r,0.f);
  }

  /// @brief Consume one observed/generated code and return detached scores for the next code.
  ///
  /// Valid code0..classes-1 and consumed()<INT64_MAX. Reads each layer's old
  /// input at time%dilation, computes the current block and accumulates skips
  /// in source Float32 scalar order. No sampling or softmax: feed every prefix
  /// code, then choose a code from the returned head and feed it back explicitly.
  /// Candidate ring writes are committed only after a finite head/state and an
  /// owning output Tensor have been constructed. Input errors, overflow yielding
  /// nonfinite state/head and count-limit errors preserve all previous history.
  /// Outputs have no weight/input autograd connection, even when source leaves
  /// require gradients. Existing returned heads remain valid across next/reset.
  /// @param code Integer code in[0,classes); code128 is the source cold-start seed for256 classes.
  /// @return Owning detached Float32 CPU[classes] logits predicting the following code.
  /// @throws std::invalid_argument For an out-of-range code or nonfinite candidate state/head.
  /// @throws std::overflow_error When consumed() already equals INT64_MAX.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// namespace n=dm::nn;namespace m=n::models;
  /// m::WaveStackConfig c{1,2,2,1,3};n::ParameterStore p(17);
  /// m::WaveStack(p,n::one_hot(n::zeros({1,1},n::DType::Int32),3),c);
  /// m::WaveStackCache cache(p,c);auto first=cache.next(1);auto second=cache.next(2);
  /// // first/second each own three logits; consumed()==2; no categorical sampling is hidden.
  /// @endcode
  /// @par References
  /// - van den Oord et al.(2016), https://arxiv.org/abs/1609.03499 . Per-layer
  /// own-input rings implement the source teaching stack's causal dilations.
  /// @see WaveStackCache, consumed(), reset(), WaveStack()
  Tensor next(int code) {
    if(code<0 || code>=config_.classes)throw std::invalid_argument("WaveStackCache code outside class range");
    if(time_==std::numeric_limits<std::int64_t>::max())throw std::overflow_error("WaveStackCache consumed-code limit");
    const int r=config_.residual,s=config_.skip,blocks=config_.levels*config_.cycles;
    std::vector<float> x(r),skip(s,0.f);std::vector<std::vector<float>> pending;pending.reserve(blocks);
    const auto& iw=weights_.at("input/W");const auto& ib=weights_.at("input/b");
    for(int j=0;j<r;++j)x[j]=(previous_<0?0:iw[static_cast<std::size_t>(previous_)*r+j])+iw[(static_cast<std::size_t>(config_.classes)+code)*r+j]+ib[j];
    for(int block=0;block<blocks;++block){
      const int dilation=1<<(block%config_.levels);const auto slot=static_cast<std::size_t>(time_%dilation);
      const auto name="block"+std::to_string(block+1);const auto& w=weights_.at(name+"/gate/W");const auto& b=weights_.at(name+"/gate/b");
      std::vector<float> joined(2*r),gate(r);
      for(int j=0;j<2*r;++j){float sum=0;
        for(int i=0;i<r;++i)sum+=ring_[block][slot*r+i]*w[static_cast<std::size_t>(i)*(2*r)+j];
        for(int i=0;i<r;++i)sum+=x[i]*w[(static_cast<std::size_t>(r)+i)*(2*r)+j];
        joined[j]=sum+b[j];
      }
      pending.push_back(x);
      for(int i=0;i<r;++i)gate[i]=std::tanh(joined[i])/(1.f+std::exp(-joined[r+i]));
      auto contribution=Point(name+"/skip",gate,s);for(int j=0;j<s;++j)skip[j]+=contribution[j];
      if(block+1<blocks){auto residual=Point(name+"/residual",gate,r);for(int j=0;j<r;++j)x[j]+=residual[j];}
    }
    for(auto& v:skip)v=std::max(0.f,v);
    auto hidden=Point("post",skip,s);for(auto& v:hidden)v=std::max(0.f,v);
    auto head=Point("output",hidden,config_.classes);
    for(const auto& state:pending)for(float v:state)if(!std::isfinite(v))throw std::invalid_argument("WaveStackCache candidate state is nonfinite");
    for(float v:head)if(!std::isfinite(v))throw std::invalid_argument("WaveStackCache candidate head is nonfinite");
    auto result=from_vector(head,{config_.classes});
    for(int block=0;block<blocks;++block){const auto slot=static_cast<std::size_t>(time_%(1<<(block%config_.levels)));
      std::copy(pending[block].begin(),pending[block].end(),ring_[block].begin()+slot*r);}
    previous_=code;++time_;return result;
  }

  /// @brief Clear every feature ring and previous-code tap without changing the weight snapshot.
  ///
  /// Restores consumed()==0 and the true beginning-of-stream state. Replaying
  /// the same codes reproduces the same heads; earlier output Tensors and clones
  /// remain unchanged. O(total ring floats), no allocation, fitting or RNG.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// namespace n=dm::nn;namespace m=n::models;
  /// m::WaveStackConfig c{1,2,1,1,3};n::ParameterStore p(17);
  /// m::WaveStack(p,n::one_hot(n::zeros({1,1},n::DType::Int32),3),c);
  /// m::WaveStackCache cache(p,c);auto a=cache.next(1);cache.reset();auto b=cache.next(1);
  /// // dm::nn::equal_bits(a,b) is true; consumed()==1 after the replayed code.
  /// @endcode
  /// @see next(), consumed(), clone()
  void reset() noexcept {for(auto& values:ring_)std::fill(values.begin(),values.end(),0.f);previous_=-1;time_=0;}

  /// @brief Count successfully consumed codes since construction or the most recent reset.
  /// @return Nonnegative Int64 count; failed next calls do not increment it.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// namespace n=dm::nn;namespace m=n::models;
  /// m::WaveStackConfig c{1,1,1,1,2};n::ParameterStore p(17);
  /// m::WaveStack(p,n::one_hot(n::zeros({1,1},n::DType::Int32),2),c);
  /// m::WaveStackCache cache(p,c);auto before=cache.consumed();cache.next(0);auto after=cache.consumed();
  /// // before==0, after==1.
  /// @endcode
  /// @see next(), reset()
  std::int64_t consumed() const noexcept {return time_;}

  /// @brief Fork an independent weight snapshot and all current circular state by value.
  ///
  /// Equivalent to the native C++ value copy, including consumed count and
  /// previous code. The two caches can continue/reset independently without
  /// shared mutable storage. O(weight floats+ring floats) copy, no fitting/RNG.
  /// @return Owning independent WaveStackCache at the same stream position.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// namespace n=dm::nn;namespace m=n::models;
  /// m::WaveStackConfig c{1,2,1,1,3};n::ParameterStore p(17);
  /// m::WaveStack(p,n::one_hot(n::zeros({1,1},n::DType::Int32),3),c);
  /// m::WaveStackCache a(p,c);a.next(1);auto b=a.clone();
  /// auto x=a.next(2),y=b.next(2); // equal_bits(x,y); both consumed()==2.
  /// @endcode
  /// @see next(), reset(), consumed()
  WaveStackCache clone() const {return *this;}

 private:
  /// @brief Copy one validated source layer's kernel/bias into this unpublished snapshot.
  /// @param p Already Float32 store; required leaves must exist and be finite CPU data.
  /// @param name Source prefix, followed by /W and /b.
  /// @param kernel Positive validated causal kernel width1 or2.
  /// @param in Positive validated input width.
  /// @param out Positive validated output width.
  /// @throws std::invalid_argument For missing/misshaped/nonfinite/non-CPU/non-Float32 leaves.
  void Layer(const ParameterStore& p,const std::string& name,int kernel,int in,int out) {
    for(const auto& [key,shape]:std::vector<std::pair<std::string,Shape>>{{name+"/W",{1,kernel,in,out}},{name+"/b",{out}}}){
      if(!p.contains(key))throw std::invalid_argument("WaveStackCache missing leaf: "+key);
      auto value=p.at(key);
      if(!value.defined() || value.device().backend!=compute::Backend::CPU || value.dtype()!=DType::Float32 || value.shape()!=shape)
        throw std::invalid_argument("WaveStackCache incompatible leaf: "+key);
      auto data=value.contiguous().to_vector<float>();
      for(float v:data)if(!std::isfinite(v))throw std::invalid_argument("WaveStackCache nonfinite leaf: "+key);
      weights_.emplace(key,std::move(data));
    }
  }
  /// @brief Apply a validated1x1 source projection using scalar Float32 accumulation order.
  /// @param name Existing layer prefix in the owned weight map.
  /// @param x Input vector matching that layer's validated width.
  /// @param out Positive validated output width.
  /// @return Independent Float32 output vector; candidate finiteness is checked by next().
  std::vector<float> Point(const std::string& name,const std::vector<float>& x,int out) const {
    std::vector<float> result(out);const auto& w=weights_.at(name+"/W");const auto& b=weights_.at(name+"/b");
    for(int j=0;j<out;++j){float sum=0;for(std::size_t i=0;i<x.size();++i)sum+=x[i]*w[i*out+j];result[j]=sum+b[j];}
    return result;
  }
  std::map<std::string,std::vector<float>> weights_; ///< Owning fixed required Float32 kernels/biases, independent of caller storage.
  WaveStackConfig config_;                         ///< Copied immutable validated architecture metadata.
  std::int64_t time_=0;                             ///< Successfully consumed code count; modulo each layer's dilation indexes its ring.
  int previous_=-1;                                 ///< Previous raw code for input tap0; minus1 denotes the true beginning.
  std::vector<std::vector<float>> ring_;             ///< One owning circular history of each block's own Float32 inputs.
};
} // namespace datamunge::nn::models
