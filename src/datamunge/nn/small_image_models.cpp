/// @file small_image_models.cpp
/// @brief Tensor-only small-image forward models with explicit nonlinear endpoints and max-pool winners.
#include <datamunge/nn/small_image_models.hpp>
#include <stdexcept>
namespace datamunge::nn::models {
/// @brief Private whole-shape checks shared by the compact CNN and dense baseline.
namespace small_image_detail {
/// @brief Validate common positive NHWC geometry and a matching learnable floating store.
/// @param p Borrowed parameter-store dtype.
/// @param image Borrowed image tensor.
/// @param hidden Positive hidden width.
/// @param classes Categorical width at least two.
/// @param prefix Nonempty parameter prefix.
/// @throws std::invalid_argument If any common shape/dtype/configuration field is invalid.
void input(const ParameterStore& p,const Tensor& image,std::int64_t hidden,std::int64_t classes,const std::string& prefix) {
  if(!image.defined() || image.dim()!=4 || image.numel()<1 || image.size(0)<1 || image.size(1)<1 || image.size(2)<1 || image.size(3)<1 || (image.dtype()!=DType::Float32 && image.dtype()!=DType::Float64) || image.dtype()!=p.dtype() || hidden<1 || classes<2 || prefix.empty())throw std::invalid_argument("small image model: incompatible positive NHWC, dtype, head or prefix");
}
}
SmallImageCnnOutputs SmallImageCnn(ParameterStore& p,const Tensor& image,SmallImageCnnConfig config,const std::string& prefix) {
  small_image_detail::input(p,image,config.hidden,config.classes,prefix);
  if(config.filters<1 || config.kernel<1 || config.kernel>image.size(1) || config.kernel>image.size(2) || config.pool_size<1)throw std::invalid_argument("small image CNN: invalid filter/pool geometry");
  const auto h=image.size(1)-config.kernel+1,w=image.size(2)-config.kernel+1;
  const auto ph=config.global_pool?1:h/config.pool_size,pw=config.global_pool?1:w/config.pool_size;
  if(ph<1 || pw<1)throw std::invalid_argument("small image CNN: local pool exceeds map");
  const auto flat=shape_numel({ph,pw,config.filters});
  (void)shape_numel({config.kernel,config.kernel,image.size(3),config.filters});(void)shape_numel({flat,config.hidden});(void)shape_numel({config.hidden,config.classes});
  SmallImageCnnOutputs out;out.image=image;ConvLayerOptions conv;conv.padding=Padding::Valid;conv.init=init::he_uniform();
  out.conv_pre=conv2d(p,prefix+"/conv",image,config.filters,config.kernel,conv);out.conv_relu=relu(out.conv_pre);
  const std::array<std::int64_t,2> window=config.global_pool?std::array<std::int64_t,2>{h,w}:std::array<std::int64_t,2>{config.pool_size,config.pool_size};
  const auto pool=max_pool2d_with_argmax(out.conv_relu,window,window,Padding::Valid,false);out.pooled=pool.output;out.pool_argmax=pool.argmax;out.flat=reshape(out.pooled,{image.size(0),flat});
  DenseOptions hidden;hidden.init=init::he_uniform();out.hidden_pre=dense(p,prefix+"/hidden",out.flat,config.hidden,hidden);out.hidden=relu(out.hidden_pre);out.logits=dense(p,prefix+"/output",out.hidden,config.classes);return out;
}
Tensor DenseImageBaseline(ParameterStore& p,const Tensor& image,std::int64_t hidden,std::int64_t classes,const std::string& prefix) {
  small_image_detail::input(p,image,hidden,classes,prefix);const auto flat=shape_numel({image.size(1),image.size(2),image.size(3)});(void)shape_numel({flat,hidden});(void)shape_numel({hidden,classes});DenseOptions options;options.init=init::he_uniform();auto features=relu(dense(p,prefix+"/hidden",reshape(image,{image.size(0),flat}),hidden,options));return dense(p,prefix+"/output",features,classes);
}
}
