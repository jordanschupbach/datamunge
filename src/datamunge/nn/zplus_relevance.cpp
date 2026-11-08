/// @file zplus_relevance.cpp
/// @brief Tensor-only signed stabilized positive-connection redistribution and actual small-CNN max-unpool explanations.
#include <datamunge/nn/zplus_relevance.hpp>
#include <datamunge/nn/autograd.hpp>
#include <cmath>
#include <stdexcept>
namespace datamunge::nn {
/// @brief Private shape/domain, numerical-range and stabilizer checks for detached attribution.
namespace zplus_detail {
/// @brief Require defined nonempty finite floating tensor data.
/// @param x Borrowed candidate tensor.
/// @throws std::invalid_argument If undefined, empty, nonfloating or nonfinite.
void finite(const Tensor& x){if(!x.defined() || x.numel()<1 || (x.dtype()!=DType::Float32 && x.dtype()!=DType::Float64) || !all(is_finite(x)).item())throw std::invalid_argument("zplus: nonempty finite floating data required");}
/// @brief Require nonnegative activations in addition to finite floating data.
/// @param x Borrowed activation tensor.
/// @throws std::invalid_argument If any value violates the activation domain.
void activation(const Tensor& x){finite(x);if(!all(greater_equal(x,0)).item())throw std::invalid_argument("zplus: nonnegative activations required");}
/// @brief Reject nonfinite computed products, denominators or relevance maps.
/// @param x Computed floating tensor.
/// @return x unchanged.
/// @throws std::overflow_error If any computed value is nonfinite.
Tensor computed(const Tensor& x){if(!all(is_finite(x)).item())throw std::overflow_error("zplus: nonfinite computed arithmetic");return x;}
/// @brief Validate a stabilizer before dtype rounding can erase positivity.
/// @param epsilon Requested finite positive stabilizer.
/// @param dtype Validated Float32 or Float64 tensor dtype.
/// @throws std::invalid_argument If nonpositive/nonfinite or unrepresentable positive in Float32.
void stabilizer(double epsilon,DType dtype){if(!std::isfinite(epsilon) || epsilon<=0 || (dtype==DType::Float32 && (!std::isfinite(static_cast<float>(epsilon)) || static_cast<float>(epsilon)<=0)))throw std::invalid_argument("zplus: positive dtype-representable epsilon required");}
}
Tensor zplus_dense(const Tensor& a,const Tensor& weight,const Tensor& relevance,double epsilon) {
  NoGradGuard guard;zplus_detail::activation(a);zplus_detail::finite(weight);zplus_detail::finite(relevance);zplus_detail::stabilizer(epsilon,a.dtype());
  if(a.dim()!=2 || weight.dim()!=2 || relevance.dim()!=2 || a.size(1)!=weight.size(0) || relevance.size(0)!=a.size(0) || relevance.size(1)!=weight.size(1) || a.dtype()!=weight.dtype() || a.dtype()!=relevance.dtype())throw std::invalid_argument("zplus dense: incompatible matrix shapes/dtypes");
  const auto positive=relu(weight.detach());const auto z=zplus_detail::computed(matmul(a.detach(),positive));const auto denominator=zplus_detail::computed(z+epsilon);return zplus_detail::computed(a.detach()*matmul(relevance.detach()/denominator,positive,false,true));
}
Tensor zplus_conv2d(const Tensor& a,const Tensor& kernel,const Tensor& relevance,const Conv2DOptions& options,double epsilon) {
  NoGradGuard guard;zplus_detail::activation(a);zplus_detail::finite(kernel);zplus_detail::finite(relevance);zplus_detail::stabilizer(epsilon,a.dtype());
  if(a.dim()!=4 || kernel.dim()!=4 || relevance.dim()!=4 || a.dtype()!=kernel.dtype() || a.dtype()!=relevance.dtype())throw std::invalid_argument("zplus convolution: incompatible rank/dtype");
  const auto positive=relu(kernel.detach());const auto z=zplus_detail::computed(conv2d(a.detach(),positive,options));
  if(z.shape()!=relevance.shape())throw std::invalid_argument("zplus convolution: relevance must match forward shape");
  const auto denominator=zplus_detail::computed(z+epsilon);return zplus_detail::computed(a.detach()*conv2d_backprop_input(a.shape(),positive,relevance.detach()/denominator,options));
}
namespace models {
CnnZPlusExplanation SmallImageCnnZPlus(ParameterStore& p,const Tensor& image,const std::vector<std::int64_t>& targets,SmallImageCnnConfig config,const std::string& prefix,double epsilon) {
  NoGradGuard guard;zplus_detail::activation(image);zplus_detail::stabilizer(epsilon,image.dtype());
  if(image.dim()!=4 || image.size(0)<1 || config.classes<2 || (!targets.empty() && targets.size()!=static_cast<std::size_t>(image.size(0))))throw std::invalid_argument("small CNN zplus: incompatible image/target count");
  for(const auto target:targets)if(target<0 || target>=config.classes)throw std::invalid_argument("small CNN zplus: target outside class range");
  for(const auto& suffix:{"/conv/W","/conv/b","/hidden/W","/hidden/b","/output/W","/output/b"}){if(!p.contains(prefix+suffix))throw std::invalid_argument("small CNN zplus: populate all model leaves before explaining");zplus_detail::finite(p.at(prefix+suffix));}
  const auto forward=SmallImageCnn(p,image,config,prefix);CnnZPlusExplanation out;out.logits=zplus_detail::computed(forward.logits);out.target_ids=targets.empty()?argmax(out.logits,1):from_vector(targets);out.output_relevance=one_hot(out.target_ids,config.classes,1,0,image.dtype())*out.logits;
  out.hidden_relevance=zplus_dense(forward.hidden,p.at(prefix+"/output/W"),out.output_relevance,epsilon);
  out.pooled_relevance=reshape(zplus_dense(forward.flat,p.at(prefix+"/hidden/W"),out.hidden_relevance,epsilon),forward.pooled.shape());
  out.conv_relevance=max_unpool2d(out.pooled_relevance,forward.pool_argmax,forward.conv_relu.shape(),false);
  Conv2DOptions options;options.padding=Padding::Valid;out.input_relevance=zplus_conv2d(image,p.at(prefix+"/conv/W"),out.conv_relevance,options,epsilon);return out;
}
}
}
