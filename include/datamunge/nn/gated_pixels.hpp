#pragma once
/// @file gated_pixels.hpp
/// @brief Explicit dual-stream gated raster context and optional class-table conditioning.
#include "datamunge/nn/autoregressive_pixels.hpp"

namespace datamunge::nn::models {
/// @brief Private leaf-order-preserving biased/unbiased1x1 projection.
namespace gated_pixel_detail {
/// @brief Apply one source normal(.1)1x1 kernel with an optional zero bias.
/// @param p Matching floating caller-owned store; missing leaves initialize.
/// @param name Leaf prefix followed by /W and optional /b.
/// @param x Already validated finite NHWC input with four channels.
/// @param outputs Positive validated output width.
/// @param bias True to create/add zero bias, false for vertical-to-horizontal links.
/// @return Owning live[N,H,W,outputs] projection.
/// @throws std::invalid_argument For incompatible stored leaves or primitive geometry.
/// @throws std::logic_error For unsupported primitive execution.
inline Tensor Project(ParameterStore& p,const std::string& name,const Tensor& x,
                      std::int64_t outputs,bool bias=true) {
  auto w=p.get(name+"/W",{1,1,4,outputs},init::normal(.1));
  auto y=conv2d(x,w);
  if(bias)y=bias_add(y,p.get(name+"/b",{outputs},init::zeros()));
  return y;
}
} // namespace gated_pixel_detail

/// @brief Construct the vertical or horizontal structural mask for a gated raster block.
///
/// Positive representable Cin/Cout, Float32(default)/64, exact zeros/ones.
/// Vertical shape[3,3,Cin,Cout]: first block allows only the top row, later
/// blocks allow top and current rows. Horizontal shape[1,3,Cin,Cout]: first
/// allows left only, later left and center. Masks exclude the whole current
/// input pixel in the first block, including all channels; there is no RGB
/// channel ordering within that pixel. Later vertical current-row features are
/// already strictly above the current raw row. Owning detached CPU data, no
/// model/store/global RNG writes. Work/storage O(CinCout), local independent state.
/// @param input_channels Positive Cin with representable kernel product.
/// @param vertical True for3x3 vertical stream; false for1x3 horizontal stream.
/// @param first True for first-block raw-image exclusion, false for later causal features.
/// @param output_channels Positive Cout, default8 for two width4 gate halves.
/// @param dtype Float32(default) or Float64 output dtype.
/// @return Owning detached structural[3,3,Cin,Cout] or[1,3,Cin,Cout] mask.
/// @throws std::invalid_argument For nonpositive/overflowing widths or nonfloating dtype.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto a=dm::nn::models::GatedPixelMask(1,true,true);
/// auto b=dm::nn::models::GatedPixelMask(4,false,false);
/// // a[3,3,1,8] contains24 ones; b[1,3,4,8] contains64 ones.
/// @endcode
/// @par References
/// - van den Oord et al.(2016), Conditional Image Generation with PixelCNN
/// Decoders, https://arxiv.org/abs/1606.05328 . Compact source dual-stream masks.
/// @see GatedPixelConvolutional(), CausalPixelMask(), MaskedPixelConv()
inline Tensor GatedPixelMask(std::int64_t input_channels,bool vertical,bool first,
                             std::int64_t output_channels=8,DType dtype=DType::Float32) {
  const auto limit=std::numeric_limits<std::int64_t>::max();
  if(input_channels<=0 || output_channels<=0 || input_channels>limit/9 ||
     output_channels>limit/(9*input_channels) || (dtype!=DType::Float32 && dtype!=DType::Float64))
    throw std::invalid_argument("GatedPixelMask requires representable positive floating widths");
  const auto height=vertical?3:1;
  std::vector<float> value(static_cast<std::size_t>(height*3*input_channels*output_channels),0.f);
  for(std::int64_t y=0;y<height;++y)
    for(std::int64_t x=0;x<3;++x)
      if(vertical?(y==0 || (!first && y==1)):(x==0 || (!first && x==1))) {
        auto start=(y*3+x)*input_channels*output_channels;
        std::fill(value.begin()+start,value.begin()+start+input_channels*output_channels,1.f);
      }
  return cast(from_vector(value,{height,3,input_channels,output_channels}),dtype);
}

/// @brief Multiply a width-four tanh feature by its paired sigmoid gate.
///
/// Finite Float32/64 CPU preactivation[N,H,W,8], positive axes, arbitrary
/// strides. First four channels are tanh-filter logits, last four sigmoid-gate
/// logits. Return tanh(first)*sigmoid(last), retaining applicable first/higher
/// paths to both halves. Output is bounded in[-1,1]. Owning live[N,H,W,4], no
/// fitting, RNG, store mutation or hidden no-grad. Work/storage O(NHW); local
/// calls share no mutable state. This gate alone does not enforce causality.
/// @param preactivation Finite floating CPU[N,H,W,8] concatenated filter/gate logits.
/// @return Owning live floating CPU[N,H,W,4] gated features.
/// @throws std::invalid_argument For incompatible geometry/dtype or nonfinite values.
/// @throws std::logic_error For unsupported storage or primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto y=n::models::GatedPixelGate(n::zeros({2,3,5,8}));
/// // y[2,3,5,4] is zero: tanh(0)*sigmoid(0)=0.
/// @endcode
/// @par References
/// - van den Oord et al.(2016), Conditional Image Generation with PixelCNN
/// Decoders, https://arxiv.org/abs/1606.05328 . Paired tanh/sigmoid gate.
/// @see GatedPixelConvolutional(), GatedPixelOutput
inline Tensor GatedPixelGate(const Tensor& preactivation) {
  autoregressive_pixel_detail::Image(preactivation,8,preactivation.dtype());
  return tanh(slice_axis(preactivation,3,0,4))*sigmoid(slice_axis(preactivation,3,4,8));
}

/// @brief Own the seven live boundaries of the three-block dual-stream gated context.
///
/// Flat Tensor fields own graph/storage after parent/store collection. Each
/// vertical field is a width4 gated map; each horizontal field includes its
/// width4 projection and, above block0, a residual from the previous horizontal
/// block. No optimizer, model object or hidden sample stream is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(8501);
/// auto o=n::models::GatedPixelConvolutional(p,n::zeros({1,8,8,1}));
/// // Six width4 maps plus logits[1,8,8,1];1073 stored values in23 leaves.
/// @endcode
/// @par References
/// - van den Oord et al.(2016), Conditional Image Generation with PixelCNN
/// Decoders, https://arxiv.org/abs/1606.05328 . Three-block width4 teaching variant.
/// @see GatedPixelConvolutional(), GatedPixelCnn()
struct GatedPixelOutput {
  Tensor vertical0;   ///< Owning live first gated vertical map[N,H,W,4].
  Tensor horizontal0; ///< Owning live first projected horizontal map[N,H,W,4].
  Tensor vertical1;   ///< Owning live second gated vertical map[N,H,W,4].
  Tensor horizontal1; ///< Owning live second horizontal residual map[N,H,W,4].
  Tensor vertical;    ///< Owning live third gated vertical map[N,H,W,4].
  Tensor horizontal;  ///< Owning live third horizontal residual map[N,H,W,4].
  Tensor logits;      ///< Owning live unbounded1x1 context parameters[N,H,W,output_channels].
};

/// @brief Evaluate a three-block gated raster context with optional class-table offsets.
///
/// Matching finite Float32/64 CPU image[N,H,W,input_channels], positive axes,
/// arbitrary strides and signed pixels. Three blocks use separate biased masked
/// vertical3x3 and horizontal1x3 convolutions to8 logits, paired tanh/sigmoid
/// gates to4 features, an unbiased1x1 vertical4->horizontal8 link and biased
/// horizontal1x1 projection4->4. Blocks1/2 add a horizontal residual. A biased
///1x1 head4->output_channels yields unbounded logits/context parameters.
/// First masks exclude the whole current raw pixel; later vertical masks allow
/// the current causal feature row and later horizontal masks allow the center.
/// This opens the ordinary masked-CNN blind spot while retaining strict raster
/// causality, including all current RGB channels for input_channels=3.
///
/// Conditional mode adds six learned[10,8] tables, one per block/stream, to
/// preactivation logits before gating; labels Int32/64 CPU[N] in[0,10) required.
/// Unconditional mode accepts omitted labels, or validates supplied labels and
/// ignores their values. Labels are discrete; all applicable image/weight first/
/// higher paths remain live. Missing W/tables normal(0,.1), biases zero, source
/// seed8501. Gray binary head:1073 values/23 leaves unconditional,1553/29
/// conditional;320 masked coordinates in either. RGB3/30 context:1410/23
/// unconditional,448 masked coordinates, no within-pixel density in this API.
/// Only missing leaves initialize; architecture mode must remain fixed for one
/// store. Unused pre-existing leaves are not removed. All supplied inputs/config
/// validate before leaves; stored-leaf/math errors may follow creation. No
/// normalization, dropout, pooling, batch coupling, optimizer, sampling, fitting,
/// existing-leaf writes or hidden no-grad. Seven owning fields retain graphs/
/// storage. Work O(NHW*(input_channels+output_channels)), fixed hidden width;
/// synchronize shared mutable stores.
/// @param parameters Caller-owned matching floating store, source seed8501.
/// @param image Finite signed CPU[N,H,W,input_channels] image/canvas.
/// @param labels Optional Int32/64 CPU[N] classes0..9; required if conditional.
/// @param conditional False(default) for image-only context; true for six learned class tables.
/// @param input_channels Positive input width, default1; RGB context uses3.
/// @param output_channels Positive head width, default1; source RGB mixture context uses30.
/// @return Owning seven-field GatedPixelOutput with two stream maps per block and head logits.
/// @throws std::invalid_argument For incompatible inputs/config/products/labels or stored leaves.
/// @throws std::logic_error For unsupported storage or primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(8609);
/// auto o=n::models::GatedPixelConvolutional(p,n::zeros({2,8,8,3}),{},false,3,30);
/// // Whole-pixel causal context[2,8,8,30]; a separate RGB likelihood consumes it.
/// @endcode
/// @par References
/// - van den Oord et al.(2016), Conditional Image Generation with PixelCNN
/// Decoders, https://arxiv.org/abs/1606.05328 . Compact three-block width4 variant;
/// RGB context is reused by the repository PixelCNN++ teaching likelihood.
/// @see GatedPixelOutput, GatedPixelCnn(), GatedPixelMask(), GatedPixelGate()
inline GatedPixelOutput GatedPixelConvolutional(
    ParameterStore& parameters,const Tensor& image,const Tensor& labels={},
    bool conditional=false,std::int64_t input_channels=1,std::int64_t output_channels=1) {
  const auto limit=std::numeric_limits<std::int64_t>::max();
  const std::int64_t fixed=conditional?1452:972;
  if(input_channels<=0 || output_channels<=0 || input_channels>(limit-fixed)/96 ||
     output_channels>(limit-fixed-96*input_channels)/5)
    throw std::invalid_argument("Gated context widths must have representable positive parameter products");
  autoregressive_pixel_detail::Image(image,input_channels,parameters.dtype());
  const auto pixels=image.numel()/input_channels;
  if(pixels>limit/std::max<std::int64_t>(8,output_channels))
    throw std::invalid_argument("Gated context activation product overflows");
  if(conditional || labels.defined())gan_detail::Labels(labels,image.size(0),10);
  GatedPixelOutput out;
  Tensor v=image,h=image;
  for(int i=0;i<3;++i) {
    const auto name="block"+std::to_string(i);
    const auto channels=i?4:input_channels;
    auto pv=MaskedPixelConv(parameters,name+"/vertical",v,GatedPixelMask(channels,true,!i,8,parameters.dtype()));
    auto ph=MaskedPixelConv(parameters,name+"/horizontal",h,GatedPixelMask(channels,false,!i,8,parameters.dtype()));
    if(conditional) {
      auto tv=parameters.get(name+"/condition_vertical",{10,8},init::normal(.1));
      pv=pv+reshape(matmul(cast(one_hot(labels,10),parameters.dtype()),tv),{-1,1,1,8});
      auto th=parameters.get(name+"/condition_horizontal",{10,8},init::normal(.1));
      ph=ph+reshape(matmul(cast(one_hot(labels,10),parameters.dtype()),th),{-1,1,1,8});
    }
    v=GatedPixelGate(pv);
    const auto link=gated_pixel_detail::Project(parameters,name+"/vertical_to_horizontal",v,8,false);
    const auto gh=GatedPixelGate(ph+link);
    const auto projected=gated_pixel_detail::Project(parameters,name+"/project",gh,4);
    h=i?h+projected:projected;
    if(i==0){out.vertical0=v;out.horizontal0=h;}
    if(i==1){out.vertical1=v;out.horizontal1=h;}
  }
  out.vertical=v;out.horizontal=h;
  out.logits=gated_pixel_detail::Project(parameters,"output",h,output_channels);
  gan_detail::Finite(out.logits);
  return out;
}

/// @brief Evaluate the binary gated model while validating labels in both source variants.
///
/// Equivalent to GatedPixelConvolutional with1 input and1 output, except labels
/// are mandatory and validated even for unconditional mode. Finite matching
/// floating CPU image[N,H,W,1], positive rectangles, labels Int32/64[N]0..9.
/// Conditional false ignores valid labels; true adds six learned class tables.
/// Source seed8501;1073/23 or1553/29 values/leaves,320 masked coordinates.
/// Inputs validate before leaves. Seven owning live stream/logit fields retain
/// applicable first/higher image/weight paths after parent/store collection.
/// No likelihood, fitting, sampling, optimizer, mutation of existing leaves or
/// hidden no-grad; no batch coupling. Complexity/ownership/concurrency follow
/// GatedPixelConvolutional; synchronize shared mutable stores.
/// @param parameters Caller-owned matching floating store, source seed8501.
/// @param image Finite signed CPU[N,H,W,1] binary canvas or differentiable gray input.
/// @param labels Required Int32/64 CPU[N] classes0..9 in both modes.
/// @param conditional True for learned class conditioning; false for validated ignored labels.
/// @return Owning live seven-field GatedPixelOutput, logits[N,H,W,1].
/// @throws std::invalid_argument For invalid input geometry/dtype/products/classes or stored leaves.
/// @throws std::logic_error For unsupported storage or primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(8501);
/// auto o=n::models::GatedPixelCnn(p,n::zeros({2,8,8,1}),n::zeros({2},n::DType::Int64),true);
/// // Conditional binary head:1553 values/29 leaves; all labels are valid digit classes.
/// @endcode
/// @par References
/// - van den Oord et al.(2016), Conditional Image Generation with PixelCNN
/// Decoders, https://arxiv.org/abs/1606.05328 . Source teaching program validates
/// labels even in its unconditional variant, which creates no class tables.
/// @see GatedPixelConvolutional(), GatedPixelOutput, BernoulliPixels()
inline GatedPixelOutput GatedPixelCnn(ParameterStore& parameters,const Tensor& image,
                                     const Tensor& labels,bool conditional) {
  autoregressive_pixel_detail::Image(image,1,parameters.dtype());
  gan_detail::Labels(labels,image.size(0),10);
  return GatedPixelConvolutional(parameters,image,labels,conditional,1,1);
}
} // namespace datamunge::nn::models
