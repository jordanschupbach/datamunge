/// @file
/// @brief Pixel Q networks with CUDA matrix products and host replay tensors.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "datamunge/nn/pixel_q.hpp"

namespace datamunge::nn::models {

namespace pixel_q_detail {

/// @brief Multiply two host matrices on CUDA and return a differentiable host result.
/// @param cuda CUDA context with native matrix multiplication.
/// @param a Left matrix.
/// @param b Right matrix.
/// @return Host result with transfer and CUDA matrix-product adjoints.
inline Tensor CudaProduct(const compute::Context& cuda,const Tensor& a,const Tensor& b) {
  return matmul(a.to(cuda),b.to(cuda)).to(a.context());
}

/// @brief Apply a source pixel convolution using CUDA products over host im2col patches.
/// @param cuda CUDA matrix context.
/// @param p Host parameter store.
/// @param name Source parameter prefix.
/// @param x NHWC input.
/// @param filters Output channels.
/// @param kernel Square kernel size.
/// @param stride Spatial stride.
/// @param same Whether to use SAME padding.
/// @return Activated NHWC outputs.
inline Tensor CudaPixelConv(const compute::Context& cuda,ParameterStore& p,const std::string& name,
    const Tensor& x,int filters,int kernel,int stride,bool same) {
  const auto batch=x.size(0),height=x.size(1),width=x.size(2),channels=x.size(3);
  const auto oh=same ? (height+stride-1)/stride : (height-kernel)/stride+1;
  const auto ow=same ? (width+stride-1)/stride : (width-kernel)/stride+1;
  const auto dh=same ? std::max<std::int64_t>(0,(oh-1)*stride+kernel-height) : 0;
  const auto dw=same ? std::max<std::int64_t>(0,(ow-1)*stride+kernel-width) : 0;
  const auto padded=pad(x,{{0,0},{dh/2,dh-dh/2},{dw/2,dw-dw/2},{0,0}});
  const auto features=kernel*kernel*channels;
  std::vector<std::int64_t> indices;
  indices.reserve(oh*ow*features);
  for (std::int64_t y=0;y<oh;++y) for (std::int64_t col=0;col<ow;++col)
    for (int ky=0;ky<kernel;++ky) for (int kx=0;kx<kernel;++kx)
      for (std::int64_t c=0;c<channels;++c)
        indices.push_back(((y*stride+ky)*(width+dw)+col*stride+kx)*channels+c);
  const auto columns=gather(padded.reshape({batch,-1}),from_vector(indices),1)
      .reshape({batch*oh*ow,features});
  const auto weights=p.get(name+"/W",{kernel,kernel,channels,filters},init::glorot_uniform());
  const auto bias=p.get(name+"/b",{filters},init::zeros());
  return relu(bias_add(CudaProduct(cuda,columns,weights.reshape({features,filters})),bias))
      .reshape({batch,oh,ow,filters});
}

/// @brief Apply the two source dense readout layers using CUDA products.
/// @param cuda CUDA matrix context.
/// @param p Host parameter store.
/// @param x Last convolutional activations.
/// @param hidden Hidden width.
/// @param actions Number of Q values per row.
/// @return Host Q values.
inline Tensor CudaPixelReadout(const compute::Context& cuda,ParameterStore& p,
    const Tensor& x,int hidden,int actions) {
  const auto dense_gpu=[&](const std::string& name,const Tensor& input,int units) {
    const auto weight=p.get(name+"/W",{input.size(1),units},init::glorot_uniform());
    const auto bias=p.get(name+"/b",{units},init::zeros());
    return bias_add(CudaProduct(cuda,input,weight),bias);
  };
  return dense_gpu("q",relu(dense_gpu("fc",x.reshape({x.size(0),-1}),hidden)),actions);
}

/// @brief Require an available CUDA matrix context for pixel-network products.
/// @param cuda Requested matrix context.
/// @throws std::invalid_argument Context is not a CUDA matrix backend.
inline void CheckCudaPixelContext(const compute::Context& cuda) {
  if (cuda.device().backend!=compute::Backend::CUDA || !cuda.supports_matmul())
    throw std::invalid_argument("pixel Q: CUDA matrix context required");
}

} // namespace pixel_q_detail

/// @brief Apply the Snake pixel Q graph with CUDA convolution and dense products.
/// @param cuda CUDA context with native matrix multiplication.
/// @param p Host Float32 parameter store shared with SnakePixelQ().
/// @param x Host Float32 [batch,108] plane-major observations.
/// @param actions Number of discrete actions.
/// @return Host Q values. Backpropagation also runs matrix products on CUDA.
/// @throws std::invalid_argument Invalid device or input geometry.
inline Tensor SnakePixelQCuda(const compute::Context& cuda,ParameterStore& p,
    const Tensor& x,int actions=4) {
  using namespace pixel_q_detail;
  CheckCudaPixelContext(cuda);
  Check(x,108,actions);
  const auto image=transpose(x.reshape({-1,3,6,6}),{0,2,3,1});
  const auto c1=CudaPixelConv(cuda,p,"conv1",image,16,3,1,true);
  const auto c2=CudaPixelConv(cuda,p,"conv2",c1,32,3,1,true);
  return CudaPixelReadout(cuda,p,c2,128,actions);
}

/// @brief Apply the Nature pixel Q graph with CUDA convolution and dense products.
/// @param cuda CUDA context with native matrix multiplication.
/// @param p Host Float32 parameter store shared with NaturePixelQ().
/// @param x Host Float32 [batch,28224] frame-major observations.
/// @param actions Number of discrete actions.
/// @return Host Q values. Backpropagation also runs matrix products on CUDA.
/// @throws std::invalid_argument Invalid device or input geometry.
inline Tensor NaturePixelQCuda(const compute::Context& cuda,ParameterStore& p,
    const Tensor& x,int actions) {
  using namespace pixel_q_detail;
  CheckCudaPixelContext(cuda);
  Check(x,4*84*84,actions);
  const auto image=transpose(x.reshape({-1,4,84,84}),{0,2,3,1});
  const auto c1=CudaPixelConv(cuda,p,"conv1",image,32,8,4,false);
  const auto c2=CudaPixelConv(cuda,p,"conv2",c1,64,4,2,false);
  const auto c3=CudaPixelConv(cuda,p,"conv3",c2,64,3,1,false);
  return CudaPixelReadout(cuda,p,c3,512,actions);
}

} // namespace datamunge::nn::models
