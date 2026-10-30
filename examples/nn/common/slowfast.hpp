#pragma once
#include <datamunge/nn/slowfast_models.hpp>
#include "common/factorized_video.hpp"
#include "common/tf_replay.hpp"
namespace slowfast {namespace nn=datamunge::nn;using nn::Tensor;
using Config=datamunge::nn::models::SlowFastConfig;
using Path=datamunge::nn::models::SlowFastOutput;
using datamunge::nn::models::SlowFastIndices;
using datamunge::nn::models::SlowFastSample;
using datamunge::nn::models::SlowFastJoin;
using datamunge::nn::models::SlowFastUnit;
using datamunge::nn::models::SlowFast;
inline auto Indices(int frames,int alpha=8){return SlowFastIndices(frames,alpha);}
inline auto Sample(const Tensor&rgb,int frames,int alpha=8){return SlowFastSample(rgb,frames,alpha);}
inline auto Join(const Tensor&slow,const Tensor&lateral,double scale=1){return SlowFastJoin(slow,lateral,scale);}
inline auto Unit(nn::ParameterStore&p,nn::ParameterStore&s,const std::string&name,const Tensor&x,int out,int kt,int kh,int kw,int st,int sh,int sw,bool relu,bool training,bool zero,bool update){return SlowFastUnit(p,s,name,x,out,kt,kh,kw,st,sh,sw,relu,training,zero,update);}
inline auto Forward(nn::ParameterStore&p,nn::ParameterStore&s,const Tensor&rgb,const Config&cfg={},bool training=false,bool update_stats=false,nn::Generator*dropout=nullptr){return SlowFast(p,s,rgb,cfg,training,update_stats,dropout);}
struct Model {
  nn::ParameterStore parameters{751}, state{753};
  nn::Generator      dropout{743};
  Model() {
    nn::NoGradGuard guard;
    Logits(nn::zeros({1, 16, 32, 32, 3}), false, false);
  }
  Tensor Logits(const Tensor& rgb, bool training, bool update_stats) {
    return Forward(parameters, state, rgb, {8, 3, 16}, training, update_stats, &dropout).logits;
  }
};

} // namespace slowfast
