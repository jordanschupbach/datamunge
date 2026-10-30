// SWIG bindings for the shared NN example components (examples/nn/common/*.hpp).
//
// The example .cpp files own their experiment loops; these headers provide the
// model forward functions, data readers, metrics and checks those loops call.
// Binding them lets the R and Python examples run the same main loop over the
// same C++. Everything here is header-only, so no library change is needed.
//
// Every C++ namespace is exported with a "<namespace>__" prefix (SWIG flattens
// namespaces); pydatamunge.nn_components and datamungerNN::nn_components regroup
// the names, so C++ image_training::ImageBatch is image_training.ImageBatch in
// Python and nn_components$image_training$ImageBatch in R. Callback-typed
// functions (std::function parameters such as LogitsBuilder) are not bound: a
// model written in R or Python cannot be handed to C++ (R has no directors), so
// the example scripts write out those loops themselves. Functions returning a
// callback are wrapped as small callable handles (LogitsModel).
//
// scripts/nn_example_audit.py treats the headers listed below as bound.
// bound-header: runtime.hpp
// bound-header: paths.hpp
// bound-header: dataset_assets.hpp
// bound-header: sentence_assets.hpp
// bound-header: checks.hpp
// bound-header: image_training.hpp
// bound-header: resnet.hpp
// bound-header: resnet_demo.hpp
// bound-header: architecture_checks.hpp
// bound-header: mobile_checks.hpp
// bound-header: grouped.hpp
// bound-header: lrn.hpp
// bound-header: inception.hpp
// bound-header: normalized_conv.hpp
// bound-header: deconvnet.hpp
// bound-header: alexnet.hpp
// bound-header: googlenet.hpp
// bound-header: inception23.hpp
// bound-header: inception4.hpp
// bound-header: nin.hpp
// bound-header: vgg.hpp
// bound-header: xception.hpp
// bound-header: zfnet.hpp
// bound-header: preact_resnet.hpp
// bound-header: densenet.hpp
// bound-header: dpn.hpp
// bound-header: pyramidnet.hpp
// bound-header: res2net.hpp
// bound-header: resnext.hpp
// bound-header: revnet.hpp
// bound-header: stochastic_depth.hpp
// bound-header: wide_resnet.hpp
// bound-header: mobilenet.hpp
// bound-header: mbconv.hpp
// bound-header: condensenet.hpp
// bound-header: ghostnet.hpp
// bound-header: regnet.hpp
// bound-header: shufflenet.hpp
// bound-header: squeezenet.hpp
// bound-header: convnext.hpp
// bound-header: deformable.hpp
// bound-header: non_local.hpp
// bound-header: octconv.hpp
// bound-header: repvgg.hpp
// bound-header: attention_cnn.hpp
// bound-header: mnist.hpp
// bound-header: cnn_baselines.hpp

%{
#include "common/runtime.hpp"
#include "common/paths.hpp"
#include "common/sentence_assets.hpp"
#include "common/checks.hpp"
#include "common/image_training.hpp"
#include "common/resnet.hpp"
#include "common/resnet_demo.hpp"
#include "common/architecture_checks.hpp"
#include "common/mobile_checks.hpp"
#include "common/grouped.hpp"
#include "common/lrn.hpp"
#include "common/inception.hpp"
#include "common/normalized_conv.hpp"
#include "common/deconvnet.hpp"
#include "common/alexnet.hpp"
#include "common/googlenet.hpp"
#include "common/inception23.hpp"
#include "common/inception4.hpp"
#include "common/nin.hpp"
#include "common/vgg.hpp"
#include "common/xception.hpp"
#include "common/zfnet.hpp"
#include "common/preact_resnet.hpp"
#include "common/densenet.hpp"
#include "common/dpn.hpp"
#include "common/pyramidnet.hpp"
#include "common/res2net.hpp"
#include "common/resnext.hpp"
#include "common/revnet.hpp"
#include "common/stochastic_depth.hpp"
#include "common/wide_resnet.hpp"
#include "common/mobilenet.hpp"
#include "common/mbconv.hpp"
#include "common/condensenet.hpp"
#include "common/ghostnet.hpp"
#include "common/regnet.hpp"
#include "common/shufflenet.hpp"
#include "common/squeezenet.hpp"
#include "common/convnext.hpp"
#include "common/deformable.hpp"
#include "common/non_local.hpp"
#include "common/octconv.hpp"
#include "common/repvgg.hpp"
#include "common/attention_cnn.hpp"
#include "common/mnist.hpp"
#include "common/cnn_baselines.hpp"
%}

// Memory-network forward/loss/evaluation components. The adapter uses the
// existing NamedTensors boundary; host languages own gradients and Adam.
// bound-header: memory_task.hpp
// bound-header: memory_validation.hpp
// bound-header: memory_data.hpp
// bound-header: ntm.hpp
// bound-header: dnc.hpp
// bound-header: memnet.hpp
// binding-adapter: ntm memory_components.hpp
// binding-adapter: dnc memory_components.hpp
// binding-adapter: memnet memory_components.hpp
// binding-adapter: key_value memory_components.hpp
%{
#include "common/memory_components.hpp"
%}
%rename("memory_components__%(strip:[memory_components::])s", regextarget=1, fullname=1) "^memory_components::[^:]+$";
%include "common/memory_components.hpp"

// Component calls print through std::cout; flush after each call so their output
// interleaves correctly with the host language's own (buffered) printing.
%exception {
  try { $action }
  catch (const std::exception& e) { std::cout.flush(); SWIG_exception(SWIG_RuntimeError, e.what()); }
  std::cout.flush();
}

// ---- namespace prefixes (members keep their names: the regex excludes nested scopes) ----
%rename("example_runtime__%(strip:[example_runtime::])s", regextarget=1, fullname=1) "^example_runtime::[^:]+$";
%rename("paths__%(strip:[paths::])s", regextarget=1, fullname=1) "^paths::[^:]+$";
%rename("dataset_assets__%(strip:[dataset_assets::])s", regextarget=1, fullname=1) "^dataset_assets::[^:]+$";
%rename("check__%(strip:[check::])s", regextarget=1, fullname=1) "^check::[^:]+$";
%rename("image_training__%(strip:[image_training::])s", regextarget=1, fullname=1) "^image_training::[^:]+$";
%rename("resnet__%(strip:[resnet::])s", regextarget=1, fullname=1) "^resnet::[^:]+$";
%rename("resnet_demo__%(strip:[resnet_demo::])s", regextarget=1, fullname=1) "^resnet_demo::[^:]+$";
%rename("architecture__%(strip:[architecture::])s", regextarget=1, fullname=1) "^architecture::[^:]+$";
%rename("mobile_checks__%(strip:[mobile_checks::])s", regextarget=1, fullname=1) "^mobile_checks::[^:]+$";
%rename("nn_components__%(strip:[nn_components::])s", regextarget=1, fullname=1) "^nn_components::[^:]+$";
%rename("grouped__%(strip:[grouped::])s", regextarget=1, fullname=1) "^grouped::[^:]+$";
%rename("lrn__%(strip:[lrn::])s", regextarget=1, fullname=1) "^lrn::[^:]+$";
%rename("inception__%(strip:[inception::])s", regextarget=1, fullname=1) "^inception::[^:]+$";
%rename("normalized_conv__%(strip:[normalized_conv::])s", regextarget=1, fullname=1) "^normalized_conv::[^:]+$";
%rename("deconvnet__%(strip:[deconvnet::])s", regextarget=1, fullname=1) "^deconvnet::[^:]+$";
%rename("alexnet__%(strip:[alexnet::])s", regextarget=1, fullname=1) "^alexnet::[^:]+$";
%rename("googlenet__%(strip:[googlenet::])s", regextarget=1, fullname=1) "^googlenet::[^:]+$";
%rename("inception23__%(strip:[inception23::])s", regextarget=1, fullname=1) "^inception23::[^:]+$";
%rename("inception4__%(strip:[inception4::])s", regextarget=1, fullname=1) "^inception4::[^:]+$";
%rename("nin__%(strip:[nin::])s", regextarget=1, fullname=1) "^nin::[^:]+$";
%rename("vgg__%(strip:[vgg::])s", regextarget=1, fullname=1) "^vgg::[^:]+$";
%rename("xception__%(strip:[xception::])s", regextarget=1, fullname=1) "^xception::[^:]+$";
%rename("zfnet__%(strip:[zfnet::])s", regextarget=1, fullname=1) "^zfnet::[^:]+$";
%rename("preact__%(strip:[preact::])s", regextarget=1, fullname=1) "^preact::[^:]+$";
%rename("densenet__%(strip:[densenet::])s", regextarget=1, fullname=1) "^densenet::[^:]+$";
%rename("dpn__%(strip:[dpn::])s", regextarget=1, fullname=1) "^dpn::[^:]+$";
%rename("pyramidnet__%(strip:[pyramidnet::])s", regextarget=1, fullname=1) "^pyramidnet::[^:]+$";
%rename("res2net__%(strip:[res2net::])s", regextarget=1, fullname=1) "^res2net::[^:]+$";
%rename("resnext__%(strip:[resnext::])s", regextarget=1, fullname=1) "^resnext::[^:]+$";
%rename("revnet__%(strip:[revnet::])s", regextarget=1, fullname=1) "^revnet::[^:]+$";
%rename("stochastic_depth__%(strip:[stochastic_depth::])s", regextarget=1, fullname=1) "^stochastic_depth::[^:]+$";
%rename("wide_resnet__%(strip:[wide_resnet::])s", regextarget=1, fullname=1) "^wide_resnet::[^:]+$";
%rename("mobilenet__%(strip:[mobilenet::])s", regextarget=1, fullname=1) "^mobilenet::[^:]+$";
%rename("mbconv__%(strip:[mbconv::])s", regextarget=1, fullname=1) "^mbconv::[^:]+$";
%rename("condensenet__%(strip:[condensenet::])s", regextarget=1, fullname=1) "^condensenet::[^:]+$";
%rename("ghostnet__%(strip:[ghostnet::])s", regextarget=1, fullname=1) "^ghostnet::[^:]+$";
%rename("regnet__%(strip:[regnet::])s", regextarget=1, fullname=1) "^regnet::[^:]+$";
%rename("shufflenet__%(strip:[shufflenet::])s", regextarget=1, fullname=1) "^shufflenet::[^:]+$";
%rename("squeezenet__%(strip:[squeezenet::])s", regextarget=1, fullname=1) "^squeezenet::[^:]+$";
%rename("convnext__%(strip:[convnext::])s", regextarget=1, fullname=1) "^convnext::[^:]+$";
%rename("deformable__%(strip:[deformable::])s", regextarget=1, fullname=1) "^deformable::[^:]+$";
%rename("non_local__%(strip:[non_local::])s", regextarget=1, fullname=1) "^non_local::[^:]+$";
%rename("octconv__%(strip:[octconv::])s", regextarget=1, fullname=1) "^octconv::[^:]+$";
%rename("repvgg__%(strip:[repvgg::])s", regextarget=1, fullname=1) "^repvgg::[^:]+$";
%rename("attention_cnn__%(strip:[attention_cnn::])s", regextarget=1, fullname=1) "^attention_cnn::[^:]+$";
%rename("mnist__%(strip:[mnist::])s", regextarget=1, fullname=1) "^mnist::[^:]+$";
%rename("cnn_baselines__%(strip:[cnn_baselines::])s", regextarget=1, fullname=1) "^cnn_baselines::[^:]+$";

%template(ImageDatasetPair) std::pair<datamunge::nn::ImageDataset, datamunge::nn::ImageDataset>;

// ---- runtime.hpp ----
%include "common/runtime.hpp"
%template(example_runtime__Limit) example_runtime::Limit<int>;
%template(example_runtime__LimitInt64) example_runtime::Limit<std::int64_t>;

// ---- paths.hpp: std::filesystem::path is exposed as std::string ----
%include "common/dataset_assets.hpp"
%include "common/sentence_assets.hpp"
%ignore paths::DataRoot;
%ignore paths::Dataset;
%ignore paths::OutputRoot;
%ignore paths::OutputDir;
%include "common/paths.hpp"
%rename(paths__DataRoot) paths::DataRootString;
%rename(paths__Dataset) paths::DatasetString;
%rename(paths__OutputRoot) paths::OutputRootString;
%rename(paths__OutputDir) paths::OutputDirString;
%inline %{
namespace paths {
std::string DataRootString() { return DataRoot().string(); }
std::string DatasetString(const std::string& name) { return Dataset(name).string(); }
std::string OutputRootString() { return OutputRoot().string(); }
std::string OutputDirString(const std::string& relative) { return OutputDir(relative).string(); }
}
%}

// ---- checks.hpp ----
%ignore check::Throws;
%ignore Header;
%include "common/checks.hpp"

// ---- image_training.hpp ----
%ignore image_training::LogitsBuilder;
%ignore image_training::TrainingBuilder;
%ignore image_training::TrainingNetwork;
%ignore image_training::EvaluateImages;
%ignore image_training::TrainingObjective;
%ignore image_training::EpochSeconds;
%ignore image_training::ParseCifarOption;
%ignore image_training::LoadCifar10;
%include "common/image_training.hpp"
%rename(image_training__LoadCifar10) image_training::LoadCifar10String;
%inline %{
namespace image_training {
std::pair<datamunge::nn::ImageDataset, datamunge::nn::ImageDataset> LoadCifar10String(const std::string& directory) {
  return LoadCifar10(std::filesystem::path(directory));
}
}
%}

// ---- resnet.hpp ----
%ignore resnet::ResNetBlockCounts;
%ignore resnet::ResNetOutputs::stages;
%include "common/resnet.hpp"
%extend resnet::ResNetOutputs {
  datamunge::nn::Tensor stage(int index) const {
    if (index < 0 || index >= 4) throw std::out_of_range("ResNet stage index 0..3");
    return $self->stages[static_cast<std::size_t>(index)];
  }
}

// ---- callable handles for functions that return a model builder ----
%inline %{
namespace nn_components {
/// A C++ image_training::LogitsBuilder callable from R and Python:
/// model.call(params, state, images, training) -> logits.
struct LogitsModel {
  image_training::LogitsBuilder builder;
  datamunge::nn::Tensor call(datamunge::nn::ParameterStore& params, datamunge::nn::ParameterStore& state,
                             const datamunge::nn::Tensor& images, bool training) const {
    if (!builder) throw std::runtime_error("empty LogitsModel");
    return builder(params, state, images, training);
  }
};
}
%}
#ifdef SWIGPYTHON
%extend nn_components::LogitsModel {
  %pythoncode %{
    def __call__(self, params, state, images, training=False):
        return self.call(params, state, images, training)
  %}
}
#endif

// ---- resnet_demo.hpp ----
%ignore resnet_demo::CompactResNet;
%include "common/resnet_demo.hpp"
%rename(resnet_demo__CompactResNet) resnet_demo::CompactResNetModel;
%inline %{
namespace resnet_demo {
nn_components::LogitsModel CompactResNetModel(int depth) { return {CompactResNet(depth)}; }
}
%}

// ---- architecture_checks.hpp / mobile_checks.hpp ----
%include "common/architecture_checks.hpp"
%include "common/mobile_checks.hpp"

// ---- architecture components used by the CNN examples ----
// Builder contexts hold references (no default constructor); the model functions
// taking a ParameterStore are the bound entry points.
%nodefaultctor inception23::InceptionBuilder;
%nodefaultctor revnet::RevContext;
%nodefaultctor mobilenet::MobileContext;
%nodefaultctor condensenet::CondenseContext;
%nodefaultctor shufflenet::ShuffleContext;
// Fixed-size std::array members are exposed through indexed accessors; functions
// taking or returning std::array, std::function hooks or nn::ReversibleBlock are not bound.
%define NN_COMPONENT_TENSOR_ARRAY(Class, member, N)
%ignore Class::member;
%rename(member) Class::member ## _at;
%extend Class {
  datamunge::nn::Tensor member ## _at(int index) const {
    if (index < 0 || index >= N) throw std::out_of_range(#member " index must be in 0.." #N);
    return $self->member[static_cast<std::size_t>(index)];
  }
}
%enddef

// SWIG's R struct-slot declarations use the original C++ member spelling even
// after %rename. Hide the reserved `in` slot and expose explicit accessors.
%define NN_COMPONENT_INPUT_CHANNELS(Class)
%ignore Class::in;
%extend Class {
  int input_channels() const { return $self->in; }
  void set_input_channels(int value) { $self->in = value; }
}
%enddef

%include "common/grouped.hpp"
%include "common/lrn.hpp"
NN_COMPONENT_TENSOR_ARRAY(inception::InceptionOutputs, branches, 4)
%include "common/inception.hpp"
%include "common/normalized_conv.hpp"
%include "common/deconvnet.hpp"
NN_COMPONENT_TENSOR_ARRAY(alexnet::AlexNetOutputs, convolution, 5)
NN_COMPONENT_TENSOR_ARRAY(alexnet::AlexNetOutputs, pooling, 3)
%include "common/alexnet.hpp"
%ignore googlenet::GoogLeNetWidths;
NN_COMPONENT_TENSOR_ARRAY(googlenet::GoogLeNetOutputs, stem, 3)
NN_COMPONENT_TENSOR_ARRAY(googlenet::GoogLeNetOutputs, inception, 9)
%include "common/googlenet.hpp"
%ignore inception23::InceptionBuilder::V2Module;
%ignore inception23::InceptionBuilder::V2Reduction;
%include "common/inception23.hpp"
%include "common/inception4.hpp"
NN_COMPONENT_TENSOR_ARRAY(nin::NinOutputs, blocks, 3)
NN_COMPONENT_TENSOR_ARRAY(nin::NinOutputs, pooling, 2)
%include "common/nin.hpp"
%ignore vgg::VggBlockDepths;
NN_COMPONENT_TENSOR_ARRAY(vgg::VggOutputs, pooling, 5)
%include "common/vgg.hpp"
%include "common/xception.hpp"
NN_COMPONENT_TENSOR_ARRAY(zfnet::ZFNetOutputs, convolution, 5)
NN_COMPONENT_TENSOR_ARRAY(zfnet::ZFNetOutputs, pooling, 3)
NN_COMPONENT_TENSOR_ARRAY(zfnet::ZFNetOutputs, normalized, 2)
%include "common/zfnet.hpp"
NN_COMPONENT_TENSOR_ARRAY(preact::PreActResNetOutputs, stages, 3)
%include "common/preact_resnet.hpp"
%include "common/densenet.hpp"
%ignore dpn::DPNLayout::blocks;
%ignore dpn::DPNLayout::inc;
%rename(blocks) dpn::DPNLayout::blocks_at;
%rename(inc) dpn::DPNLayout::inc_at;
%extend dpn::DPNLayout {
  int blocks_at(int stage) const { return $self->blocks.at(static_cast<std::size_t>(stage)); }
  int inc_at(int stage) const { return $self->inc.at(static_cast<std::size_t>(stage)); }
}
%template(dpn__DPNBlockOutputsVector) std::vector<dpn::DPNBlockOutputs>;
%template(dpn__DPNBlockOutputsVectorVector) std::vector<std::vector<dpn::DPNBlockOutputs> >;
%ignore dpn::DPNOutputs::stages;
%rename(stages) dpn::DPNOutputs::stages_at;
%extend dpn::DPNOutputs {
  dpn::DPNPaths stages_at(int stage) const { return $self->stages.at(static_cast<std::size_t>(stage)); }
}
%include "common/dpn.hpp"
%include "common/pyramidnet.hpp"
%include "common/res2net.hpp"
%include "common/resnext.hpp"
%ignore revnet::RevCoupling;
// `in` is an R keyword; use the same descriptive field in both languages.
NN_COMPONENT_INPUT_CHANNELS(revnet::RevNetUnit)
%include "common/revnet.hpp"
%include "common/stochastic_depth.hpp"
NN_COMPONENT_TENSOR_ARRAY(wide_resnet::WideResNetOutputs, stages, 3)
%include "common/wide_resnet.hpp"
NN_COMPONENT_INPUT_CHANNELS(mobilenet::MobileNetV3Block)
%template(mobilenet__MobileLayerVector) std::vector<mobilenet::MobileLayer>;
%template(mobilenet__MobileNetV3BlockVector) std::vector<mobilenet::MobileNetV3Block>;
%include "common/mobilenet.hpp"
%include "common/mbconv.hpp"
%include "common/condensenet.hpp"
// CondenseContext holds references; this owning handle builds one over host-owned stores
// (its layer list is kept inside) so LearnedGroupConv / Condense can be called from the ports.
%inline %{
namespace condensenet {
struct CondenseContextHandle {
  std::vector<mobilenet::MobileLayer> layers;
  CondenseContext context;
  CondenseContextHandle(datamunge::nn::ParameterStore& p, datamunge::nn::ParameterStore& state, bool training,
                        double momentum = .9, double progress = 0)
      : context{p, state, layers, training, momentum, progress} {}
  CondenseContext& get() { return context; }
};
LearnedGroupConvOutputs LearnedGroupConvHandle(CondenseContextHandle& handle, const std::string& name, const datamunge::nn::Tensor& x,
                                               std::int64_t outputs, int groups, int condense) {
  return LearnedGroupConv(handle.context, name, x, outputs, groups, condense);
}
}
%}
%rename(condensenet__CondenseContextHandle) condensenet::CondenseContextHandle;
%rename(condensenet__LearnedGroupConvHandle) condensenet::LearnedGroupConvHandle;
%include "common/ghostnet.hpp"
%include "common/regnet.hpp"
%ignore shufflenet::ShuffleNetOutputs::layers;
%include "common/shufflenet.hpp"
%extend shufflenet::ShuffleNetOutputs {
  /// Recorded convolutions: count, output shape and multiply-adds per output position of layer i.
  int layer_count() const { return int($self->layers.size()); }
  std::vector<std::int64_t> layer_shape(int index) const { return $self->layers.at(static_cast<std::size_t>(index)).first; }
  std::int64_t layer_cost(int index) const { return $self->layers.at(static_cast<std::size_t>(index)).second; }
}
// ShuffleContext holds references; this owning handle builds one over host-owned stores.
%inline %{
namespace shufflenet {
struct ShuffleContextHandle {
  std::vector<std::pair<datamunge::nn::Shape, std::int64_t>> layers;
  ShuffleContext context;
  ShuffleContextHandle(datamunge::nn::ParameterStore& p, datamunge::nn::ParameterStore& state, bool training,
                       double momentum = .9, bool update_stats = true, bool record_layers = false)
      : context{p, state, training, momentum, update_stats, record_layers ? &layers : nullptr} {}
  ShuffleContext& get() { return context; }
  int layer_count() const { return int(layers.size()); }
  std::vector<std::int64_t> layer_shape(int index) const { return layers.at(static_cast<std::size_t>(index)).first; }
  std::int64_t layer_cost(int index) const { return layers.at(static_cast<std::size_t>(index)).second; }
};
}
%}
%rename(shufflenet__ShuffleContextHandle) shufflenet::ShuffleContextHandle;
%include "common/squeezenet.hpp"
%include "common/convnext.hpp"
%include "common/deformable.hpp"
%include "common/non_local.hpp"
%include "common/octconv.hpp"
NN_COMPONENT_INPUT_CHANNELS(repvgg::RepVGGBlockSpec)
%include "common/repvgg.hpp"
%ignore attention_cnn::Conv2Hook;
%ignore attention_cnn::AfterBlockHook;
%ignore attention_cnn::AttentionResNetConfig::conv2;
%ignore attention_cnn::AttentionResNetConfig::after_block;
%ignore attention_cnn::SelectiveKernelHook;
%include "common/attention_cnn.hpp"
// The selective-kernel hook is a std::function; install it from the bound languages through this method.
%extend attention_cnn::AttentionResNetConfig {
  void use_selective_kernel(int cardinality, int reduction, double momentum = .9, bool update_stats = true,
                            std::int64_t minimum = 32) {
    $self->conv2 = attention_cnn::SelectiveKernelHook(cardinality, reduction, momentum, update_stats, minimum);
  }
  bool has_conv2_hook() const { return static_cast<bool>($self->conv2); }
  bool has_after_block_hook() const { return static_cast<bool>($self->after_block); }
}
%ignore mnist::LogitsBuilder;
%ignore mnist::Evaluate;
%include "common/mnist.hpp"
%include "common/cnn_baselines.hpp"

// Architecture diagnostics used by native language examples.
%{
#include "common/cnn_validation.hpp"
%}
%rename("alexnet_checks__%(strip:[alexnet_checks::])s", regextarget=1, fullname=1) "^alexnet_checks::[^:]+$";
%rename("xception_checks__%(strip:[xception_checks::])s", regextarget=1, fullname=1) "^xception_checks::[^:]+$";
%rename("densenet_checks__%(strip:[densenet_checks::])s", regextarget=1, fullname=1) "^densenet_checks::[^:]+$";
%rename("ghostnet_checks__%(strip:[ghostnet_checks::])s", regextarget=1, fullname=1) "^ghostnet_checks::[^:]+$";
%rename("mobilenet_v2_checks__%(strip:[mobilenet_v2_checks::])s", regextarget=1, fullname=1) "^mobilenet_v2_checks::[^:]+$";
%rename("mnasnet_checks__%(strip:[mnasnet_checks::])s", regextarget=1, fullname=1) "^mnasnet_checks::[^:]+$";
%rename("convnext_checks__%(strip:[convnext_checks::])s", regextarget=1, fullname=1) "^convnext_checks::[^:]+$";
%rename("efficientnet_checks__%(strip:[efficientnet_checks::])s", regextarget=1, fullname=1) "^efficientnet_checks::[^:]+$";
%rename("efficientnet_v2_checks__%(strip:[efficientnet_v2_checks::])s", regextarget=1, fullname=1) "^efficientnet_v2_checks::[^:]+$";
%ignore alexnet_checks::CompactArchitecture;
%include "common/cnn_validation.hpp"

// Architecture diagnostics used by native language examples.
%{
#include "common/residual_validation.hpp"
%}
%rename("preactivation_resnet_checks__%(strip:[preactivation_resnet_checks::])s", regextarget=1, fullname=1) "^preactivation_resnet_checks::[^:]+$";
%rename("pyramidnet_checks__%(strip:[pyramidnet_checks::])s", regextarget=1, fullname=1) "^pyramidnet_checks::[^:]+$";
%rename("res2net_checks__%(strip:[res2net_checks::])s", regextarget=1, fullname=1) "^res2net_checks::[^:]+$";
%rename("resnext_checks__%(strip:[resnext_checks::])s", regextarget=1, fullname=1) "^resnext_checks::[^:]+$";
%rename("revnet_checks__%(strip:[revnet_checks::])s", regextarget=1, fullname=1) "^revnet_checks::[^:]+$";
%rename("cbam_checks__%(strip:[cbam_checks::])s", regextarget=1, fullname=1) "^cbam_checks::[^:]+$";
%rename("non_local_checks__%(strip:[non_local_checks::])s", regextarget=1, fullname=1) "^non_local_checks::[^:]+$";
%rename("octconv_checks__%(strip:[octconv_checks::])s", regextarget=1, fullname=1) "^octconv_checks::[^:]+$";
%rename("repvgg_checks__%(strip:[repvgg_checks::])s", regextarget=1, fullname=1) "^repvgg_checks::[^:]+$";
%rename("sknet_checks__%(strip:[sknet_checks::])s", regextarget=1, fullname=1) "^sknet_checks::[^:]+$";
%ignore pyramidnet_checks::PatternValues;
%ignore pyramidnet_checks::Pattern;
%ignore revnet_checks::Pattern;
%ignore revnet_checks::RelativeDifference;
%ignore revnet_checks::Batch;
%ignore revnet_checks::Loss;
%ignore revnet_checks::PeakMiB;
%ignore non_local_checks::Project;
%ignore non_local_checks::NonLocalHook;
%ignore octconv_checks::Convolve;
%ignore octconv_checks::Pool;
%ignore repvgg_checks::Perturb;
%ignore repvgg_checks::MaxDifference;
%include "common/residual_validation.hpp"

// Hard competition autoencoders: bind the canonical forward/mask and checks.
// The host examples own Adam, epochs, diagnostics/checkpoint/report ordering.
%{
#include "common/competitive_autoencoder.hpp"
%}
%rename("ae__%(strip:[ae::])s", regextarget=1, fullname=1) "^ae::[^:]+$";
// These other autoencoder model wrappers still bundle optimizer steps. Their
// component forward functions remain available; no bundled step is exported.
%ignore ae::ConvolutionalAutoencoderModel;
%ignore ae::VariationalAutoencoderModel;
%ignore ae::UndercompleteModel;
%ignore ae::CompetitiveAutoencoderModel;
%ignore ae::MakeAdam;
%ignore ae::VaeScore;
// These internal strings contain arbitrary tensor bytes, not text. Expose
// equality directly; R strings cannot represent embedded NUL bytes.
%ignore ae::Bytes;
%ignore ae::Snapshot;
// Filesystem/stream entry points use portable string adapters below.
%ignore ae::LoadSaliency;
%ignore ae::CopyManifests;
%ignore ae::PanelRow;
%ignore ae::SaveData;
%ignore ae::DigitGallery;
%ignore ae::LoadFrozen;
%ignore ae::CaseStrip3;
%ignore ae::Mosaic3;
%include "common/autoencoder.hpp"
%template(ae__SaliencyDataVector) std::vector<ae::SaliencyData>;
%include "common/competitive_autoencoder.hpp"
%rename(ae__LoadSaliency) ae::LoadSaliencyString;
%rename(ae__SaveData) ae::SaveDataString;
%inline %{
namespace ae {
bool SameTensorBits(const nn::Tensor& first, const nn::Tensor& second) {
  return nn::equal_bits(first, second);
}
std::vector<SaliencyData> LoadSaliencyString(const std::string& root) {
  return LoadSaliency(root);
}
void SaveDataString(const std::string& file, const std::vector<SaliencyData>& pools) {
  SaveData(file, pools);
}
}
%}

// ---- semantic segmentation components (1.7) ----
// Procedural shape scenes, batching, balanced pixel CE, IoU/policy checks and the
// FCN / U-Net / SegNet / HRNet / PSPNet / DeepLab model functions. Callback-typed
// entry points (SegmentationScore, SegmentationPreview, DenseCrfPreview take a
// LogitsFn) are not bound: the host examples write those evaluation loops out
// over SegmentationBatch + SegmentationConfusion / SegmentationPreviewLogits below.
// bound-header: segmentation.hpp
// bound-header: fcn.hpp
// bound-header: unet.hpp
// bound-header: unet_variants.hpp
// bound-header: segnet.hpp
// bound-header: hrnet.hpp
// bound-header: pspnet.hpp
// bound-header: deeplab.hpp
// bound-header: deeplab_v3.hpp
// bound-header: dense_crf.hpp
// bound-header: segmentation_losses.hpp
%{
#include "common/segmentation.hpp"
#include "common/fcn.hpp"
#include "common/unet.hpp"
#include "common/unet_variants.hpp"
#include "common/segnet.hpp"
#include "common/hrnet.hpp"
#include "common/deeplab.hpp"
#include "common/deeplab_v3.hpp"
#include "common/pspnet.hpp"
#include "common/dense_crf.hpp"
#include "common/segmentation_losses.hpp"
%}
%rename("seg__%(strip:[seg::])s", regextarget=1, fullname=1) "^seg::[^:]+$";
%rename("fcn__%(strip:[fcn::])s", regextarget=1, fullname=1) "^fcn::[^:]+$";
%rename("unet__%(strip:[unet::])s", regextarget=1, fullname=1) "^unet::[^:]+$";
%rename("segnet__%(strip:[segnet::])s", regextarget=1, fullname=1) "^segnet::[^:]+$";
%rename("hrnet__%(strip:[hrnet::])s", regextarget=1, fullname=1) "^hrnet::[^:]+$";
%rename("pspnet__%(strip:[pspnet::])s", regextarget=1, fullname=1) "^pspnet::[^:]+$";
%rename("deeplab__%(strip:[deeplab::])s", regextarget=1, fullname=1) "^deeplab::[^:]+$";
// Internal helpers of segmentation_losses.hpp stay private to C++.
%rename("$ignore", regextarget=1, fullname=1) "^seg::detail::.*$";

// Fixed-size std::array<int, N> members are exposed through indexed get/set accessors.
%define NN_COMPONENT_INT_ARRAY(Class, member, N)
%ignore Class::member;
%rename(member) Class::member ## _at;
%rename(set_ ## member) Class::set_ ## member ## _at;
%extend Class {
  int member ## _at(int index) const {
    if (index < 0 || index >= N) throw std::out_of_range(#member " index must be in 0.." #N);
    return $self->member[static_cast<std::size_t>(index)];
  }
  void set_ ## member ## _at(int index, int value) {
    if (index < 0 || index >= N) throw std::out_of_range(#member " index must be in 0.." #N);
    $self->member[static_cast<std::size_t>(index)] = value;
  }
}
%enddef

%ignore seg::SegmentationBuilder;
%ignore seg::LogitsFn;
%ignore seg::SegmentationConfig::inspect;
%ignore seg::SegmentationScore;
%ignore seg::SegmentationPreview;
%ignore seg::kSegmentationPalette;
%ignore seg::SegmentationDirectory;
%ignore seg::DenseCrfPreview;
%ignore seg::SquaredDistance2D;
%include "common/segmentation.hpp"
%include "common/fcn.hpp"
%include "common/unet.hpp"
%include "common/unet_variants.hpp"
%template(segnet__IndexedPoolVector) std::vector<segnet::IndexedPool>;
%include "common/segnet.hpp"
NN_COMPONENT_INT_ARRAY(hrnet::HRNetConfig, modules, 3)
%include "common/hrnet.hpp"
%ignore deeplab::AtrousScorePyramid;
NN_COMPONENT_INT_ARRAY(deeplab::DeepLabV2Config, rates, 4)
%include "common/deeplab.hpp"
NN_COMPONENT_INT_ARRAY(deeplab::DeepLabV3Config, rates, 3)
%include "common/deeplab_v3.hpp"
%include "common/pspnet.hpp"
%include "common/dense_crf.hpp"
%include "common/segmentation_losses.hpp"
%rename(seg__SegmentationDirectory) seg::SegmentationDirectoryString;
%rename(seg__SquaredDistance2D) seg::SquaredDistance2DInt;
%rename(deeplab__AtrousScorePyramid) deeplab::AtrousScorePyramidVector;
%inline %{
namespace seg {
std::string SegmentationDirectoryString(const std::string& name) { return SegmentationDirectory(name).string(); }
/// Confusion counts (truth rows, prediction columns, row-major) of logits [B, H, W, classes]
/// against labels [B, H, W]; the host example sums these over its evaluation batches and
/// passes the total to ConfusionMeanIou.
std::vector<std::int64_t> SegmentationConfusion(const nn::Tensor& logits, const nn::Tensor& labels, int classes) {
  nn::NoGradGuard           guard;
  std::vector<std::int64_t> confusion(std::size_t(classes) * classes, 0);
  const auto predictions = nn::argmax(logits, 3).to_vector<std::int64_t>();
  const auto targets     = labels.to_vector<std::int32_t>();
  check::Require(predictions.size() == targets.size(), "prediction/mask pixel counts agree");
  for (std::size_t i = 0; i < targets.size(); ++i) {
    const int target = targets[i], prediction = int(predictions[i]);
    check::Require(prediction >= 0 && prediction < classes, "valid segmentation prediction");
    ++confusion[std::size_t(target * classes + prediction)];
  }
  return confusion;
}
/// SegmentationPreview from the host example's own logits of rows 0..min(4, count)-1.
void SegmentationPreviewLogits(const nn::Tensor& logits, const SegmentationData& data, const std::string& filename) {
  const nn::Tensor fixed = logits.detach();
  SegmentationPreview([&](const nn::Tensor&) { return fixed; }, data, filename);
}
std::vector<double> SquaredDistance2DInt(const std::vector<int>& seeds, int height, int width) {
  std::vector<bool> flags(seeds.begin(), seeds.end());
  return SquaredDistance2D(flags, height, width);
}
}
namespace deeplab {
std::vector<nn::Tensor> AtrousScorePyramidVector(nn::ParameterStore& p, const std::string& name, const nn::Tensor& features,
                                                 int classes, const std::vector<int>& rates) {
  if (rates.size() != 4) throw std::invalid_argument("AtrousScorePyramid takes four rates");
  return AtrousScorePyramid(p, name, features, classes, std::array<int, 4>{rates[0], rates[1], rates[2], rates[3]});
}
}
%}

// ---- 1-D convolution (1.9) and interpretability (1.11) components ----
// Shared sentence / time-series / sensor / character / TCN / WaveNet-stack model
// functions, their data readers and metrics, the TF-replay comparison helpers and
// the saliency classifier with the interpretability building blocks. Model classes
// keep their forward/loss/predict/save/restore members; their bundled optimizers
// (std::unique_ptr<nn::Optimizer>, KimAdadelta slots) are not exported, so the host
// examples own every gradient and update. Callback-typed functions (FixParameters,
// HostGradientError, VanillaInputGradient, DreamAscent, OcclusionSensitivity,
// IntegratedGradients, SmoothGrad) and std::map-valued helpers are not bound: the
// host examples write those loops out over the bound pieces, and std::filesystem
// paths are taken as strings through the adapters below.
// bound-header: kim_cnn.hpp
// bound-header: sentence.hpp
// bound-header: tf_replay.hpp
// bound-header: time_series.hpp
// bound-header: ecg_sensor.hpp
// bound-header: character_cnn.hpp
// bound-header: character.hpp
// bound-header: tcn.hpp
// bound-header: wavenet_stack.hpp
// bound-header: saliency.hpp
// bound-header: dream.hpp
// bound-header: feature_visualization.hpp
// bound-header: grad_cam.hpp
// bound-header: occlusion.hpp
// bound-header: integrated_gradients.hpp
// bound-header: smoothgrad.hpp
// bound-header: guided_backprop.hpp
%{
#include "common/kim_cnn.hpp"
#include "common/tf_replay.hpp"
#include "common/sentence.hpp"
#include "common/time_series.hpp"
#include "common/ecg_sensor.hpp"
#include "common/character_cnn.hpp"
#include "common/character.hpp"
#include "common/tcn.hpp"
#include "common/wavenet_stack.hpp"
#include "common/saliency.hpp"
#include "common/dream.hpp"
#include "common/feature_visualization.hpp"
#include "common/grad_cam.hpp"
#include "common/occlusion.hpp"
#include "common/integrated_gradients.hpp"
#include "common/smoothgrad.hpp"
#include "common/guided_backprop.hpp"
%}
%rename("sentence__%(strip:[sentence::])s", regextarget=1, fullname=1) "^sentence::[^:]+$";
%rename("replay__%(strip:[replay::])s", regextarget=1, fullname=1) "^replay::[^:]+$";
%rename("series__%(strip:[series::])s", regextarget=1, fullname=1) "^series::[^:]+$";
%rename("sensor__%(strip:[sensor::])s", regextarget=1, fullname=1) "^sensor::[^:]+$";
%rename("charcnn__%(strip:[charcnn::])s", regextarget=1, fullname=1) "^charcnn::[^:]+$";
%rename("tcn__%(strip:[tcn::])s", regextarget=1, fullname=1) "^tcn::[^:]+$";
%rename("wavestack__%(strip:[wavestack::])s", regextarget=1, fullname=1) "^wavestack::[^:]+$";
%rename("sal__%(strip:[sal::])s", regextarget=1, fullname=1) "^sal::[^:]+$";
%rename("dream__%(strip:[dream::])s", regextarget=1, fullname=1) "^dream::[^:]+$";
%rename("featurevis__%(strip:[featurevis::])s", regextarget=1, fullname=1) "^featurevis::[^:]+$";
%rename("cam__%(strip:[cam::])s", regextarget=1, fullname=1) "^cam::[^:]+$";
%rename("occlusion__%(strip:[occlusion::])s", regextarget=1, fullname=1) "^occlusion::[^:]+$";
%rename("ig__%(strip:[ig::])s", regextarget=1, fullname=1) "^ig::[^:]+$";
%rename("smooth__%(strip:[smooth::])s", regextarget=1, fullname=1) "^smooth::[^:]+$";
%rename("guided__%(strip:[guided::])s", regextarget=1, fullname=1) "^guided::[^:]+$";

// kim_cnn.hpp / sentence.hpp
%immutable sentence::KimAdadelta::slots;
%ignore sentence::FixParameters;
%ignore sentence::HostGradientError;
%ignore sentence::RequireLiveGradients;
%ignore sentence::Snapshot;
%ignore sentence::ExportArrays;
%ignore sentence::CopyIfExists;
%ignore sentence::LoadSentences;
%immutable sentence::SentenceModel::parameters;
%immutable sentence::SentenceModel::dropout;
%ignore sentence::SentenceModel::optimizer;
%ignore sentence::SentenceModel::Snapshot;
%template(sentence__SentenceDataVector) std::vector<sentence::SentenceData>;
%include "common/kim_cnn.hpp"
// tf_replay.hpp: the std::map-valued helpers stay in C++ (ports use nn.load_npz / NamedTensors).
%ignore replay::Map;
%ignore replay::TfModels;
%ignore replay::ToMap;
%ignore replay::ToNamed;
%ignore replay::Rows;
%ignore replay::Split;
%ignore replay::SameBits;
%ignore replay::Arrays;
%ignore replay::Named;
%ignore replay::LoadInto;
%ignore replay::MaxDiff;
%ignore replay::History;
%ignore replay::CopyAssets;
%include "common/tf_replay.hpp"
%include "common/sentence.hpp"
// time_series.hpp
%ignore series::LoadSeries;
%immutable series::SeriesModel::parameters;
%immutable series::SeriesModel::state;
%ignore series::SeriesModel::optimizer;
%ignore series::SeriesModel::Snapshot;
%ignore series::SeriesModel::Export;
%template(series__SeriesDataVector) std::vector<series::SeriesData>;
%include "common/time_series.hpp"
// ecg_sensor.hpp
%ignore sensor::LoadSensor;
%ignore sensor::CompareWithTflow;
%immutable sensor::SensorModel::parameters;
%immutable sensor::SensorModel::state;
%ignore sensor::SensorModel::optimizer;
%ignore sensor::SensorModel::Snapshot;
%ignore sensor::SensorModel::Export;
%include "common/ecg_sensor.hpp"
// character_cnn.hpp / character.hpp
%ignore charcnn::LoadCharacters;
%immutable charcnn::CharacterModel::parameters;
%immutable charcnn::CharacterModel::dropout;
%ignore charcnn::CharacterModel::optimizer;
%ignore charcnn::CharacterModel::Snapshot;
%include "common/character_cnn.hpp"
%include "common/character.hpp"
// tcn.hpp
%immutable tcn::TcnModel::parameters;
%immutable tcn::TcnModel::state;
%immutable tcn::TcnModel::dropout;
%ignore tcn::TcnModel::optimizer;
%ignore tcn::TcnModel::Snapshot;
%ignore tcn::TcnModel::Export;
%include "common/tcn.hpp"
// wavenet_stack.hpp: Loss(d, update=true) bundles an optimizer step; only the evaluation form is exported.
%ignore wavestack::LoadWaveStack;
%immutable wavestack::WaveStackModel::parameters;
%ignore wavestack::WaveStackModel::optimizer;
%ignore wavestack::WaveStackModel::Loss;
%ignore wavestack::WaveStackModel::Snapshot;
%ignore wavestack::WaveStackModel::Export;
%template(wavestack__WaveDataVector) std::vector<wavestack::WaveData>;
%include "common/wavenet_stack.hpp"
// saliency.hpp: the std::function gate (guided_backprop's GuidedRelu) and std::map arrays are
// adapted below; SaliencyModel::Update bundles an Adam step and is not exported.
%ignore sal::Arrays;
%ignore sal::Gate;
%ignore sal::Bytes;
%ignore sal::LoadArrays;
%ignore sal::SaveArrays;
%ignore sal::SaliencyConvFeatures;
%ignore sal::SaliencyCnn;
%ignore sal::VanillaInputGradient;
%ignore sal::SaliencyModel::SaliencyModel(bool, std::uint64_t, sal::Gate);
%ignore sal::SaliencyModel::gate;
%ignore sal::SaliencyModel::optimizer;
%ignore sal::SaliencyModel::Update;
%ignore sal::SaliencyModel::Snapshot;
%ignore sal::SaliencyModel::Export;
%ignore sal::SaliencyModel::Assign;
%immutable sal::SaliencyModel::parameters;
%ignore sal::Canvas::at;
%ignore sal::SaliencyPng;
%ignore sal::SignedColor;
%ignore sal::SourceDir;
%ignore sal::DefaultSource;
%ignore sal::SourceWeights;
%ignore sal::RestoreSource;
%ignore sal::CopyManifests;
%ignore sal::SaliencyGallery;
%include "common/saliency.hpp"
%extend sal::SaliencyModel {
  /// Export() as an .npz file.
  void ExportFile(const std::string& file) const { $self->Export(std::filesystem::path(file)); }
  /// Assign() from named arrays (nn.load_npz's result).
  void AssignNamed(const datamunge::nn::NamedTensors& weights) {
    $self->Assign(sal::Arrays(weights.begin(), weights.end()));
  }
}
// dream.hpp / feature_visualization.hpp / grad_cam.hpp / occlusion.hpp / integrated_gradients.hpp / smoothgrad.hpp
%ignore dream::DreamFunction;
%ignore dream::DreamAscent;
%include "common/dream.hpp"
%include "common/feature_visualization.hpp"
%include "common/grad_cam.hpp"
%ignore occlusion::PredictionFunction;
%ignore occlusion::OcclusionSensitivity;
%ignore occlusion::Export;
%include "common/occlusion.hpp"
%ignore ig::InputGradientFunction;
%ignore ig::IntegratedGradients;
%include "common/integrated_gradients.hpp"
%ignore smooth::SmoothGrad;
%include "common/smoothgrad.hpp"
%ignore guided::GuidedReluCount;
%ignore guided::GuidedReluFunction;
%include "common/guided_backprop.hpp"

// String-path / callback-free adapters for the functions ignored above.
%rename(sentence__RequireLiveGradients) sentence::RequireLiveGradientsNamed;
%rename(sentence__ExportArrays) sentence::ExportArraysFile;
%rename(sentence__CopyIfExists) sentence::CopyIfExistsFile;
%rename(sentence__LoadSentences) sentence::LoadSentencesDir;
%rename(sentence__Snapshot) sentence::SnapshotNamed;
%rename(replay__TfModels) replay::TfModelsString;
%rename(replay__SameBits) replay::SameBitsTensor;
%rename(replay__LoadInto) replay::LoadIntoFile;
%rename(replay__MaxDiff) replay::MaxDiffFile;
%rename(replay__History) replay::HistoryFile;
%rename(replay__CopyAssets) replay::CopyAssetsDir;
%rename(series__LoadSeries) series::LoadSeriesDir;
%rename(sensor__LoadSensor) sensor::LoadSensorDir;
%rename(sensor__CompareWithTflow) sensor::CompareWithTflowDir;
%rename(charcnn__LoadCharacters) charcnn::LoadCharactersDir;
%rename(wavestack__LoadWaveStack) wavestack::LoadWaveStackDir;
%rename(sal__SaliencyConvFeatures) sal::SaliencyConvFeaturesRelu;
%rename(sal__SaliencyCnn) sal::SaliencyCnnRelu;
%rename(sal__GuidedSaliencyConvFeatures) sal::SaliencyConvFeaturesGuided;
%rename(sal__LoadArrays) sal::LoadArraysNamed;
%rename(sal__SaveArrays) sal::SaveArraysNamed;
%rename(sal__SaliencyPng) sal::SaliencyPngFile;
%rename(sal__SignedColor) sal::SignedColorVector;
%rename(sal__SourceDir) sal::SourceDirString;
%rename(sal__DefaultSource) sal::DefaultSourceString;
%rename(sal__RestoreSource) sal::RestoreSourceDir;
%rename(sal__SaliencyGallery) sal::SaliencyGalleryDir;
%inline %{
namespace sentence {
/// RequireLiveGradients with the parameters named in `may_vanish` excused from the nonzero check.
int RequireLiveGradientsNamed(const nn::ParameterStore& p, const std::vector<Tensor>& gradients, const std::string& what,
                              const std::vector<std::string>& may_vanish = {}, double floor = 0) {
  if (may_vanish.empty()) return RequireLiveGradients(p, gradients, what, {}, floor);
  return RequireLiveGradients(
      p, gradients, what,
      [&](const std::string& name) { return std::find(may_vanish.begin(), may_vanish.end(), name) != may_vanish.end(); }, floor);
}
void ExportArraysFile(const std::string& path, const nn::ParameterStore& p) { ExportArrays(std::filesystem::path(path), p); }
void ExportArraysFile(const std::string& path, const nn::ParameterStore& p, const nn::ParameterStore& s) {
  ExportArrays(std::filesystem::path(path), p, &s);
}
void CopyIfExistsFile(const std::string& from, const std::string& to) {
  CopyIfExists(std::filesystem::path(from), std::filesystem::path(to));
}
SentenceSplits LoadSentencesDir(const std::string& root) { return LoadSentences(std::filesystem::path(root)); }
/// Snapshot of one or two stores as named tensors ("<k>:<name>", k = store index).
nn::NamedTensors SnapshotNamed(const nn::ParameterStore& p) { return replay::ToNamed(Snapshot({&p})); }
nn::NamedTensors SnapshotNamed(const nn::ParameterStore& p, const nn::ParameterStore& s) {
  return replay::ToNamed(Snapshot({&p, &s}));
}
}
namespace replay {
std::string TfModelsString(const std::string& name) { return TfModels(name).string(); }
bool SameBitsTensor(const Tensor& a, const Tensor& b) { return SameBits(a, b); }
/// Bit-identical named tensors (same keys, same bytes).
bool SameBitsTensor(const nn::NamedTensors& a, const nn::NamedTensors& b) { return SameBits(ToMap(a), ToMap(b)); }
void LoadIntoFile(nn::ParameterStore& p, const std::string& file) { LoadInto(p, std::filesystem::path(file)); }
/// max |a[k] - file[k]| over the names of `a`.
double MaxDiffFile(const nn::NamedTensors& a, const std::string& file) { return MaxDiff(ToMap(a), std::filesystem::path(file)); }
double MaxDiffFile(const nn::NamedTensors& a, const nn::NamedTensors& b) { return MaxDiff(ToMap(a), ToMap(b)); }
std::vector<double> HistoryFile(const std::string& file, int column = 1) { return History(std::filesystem::path(file), column); }
void CopyAssetsDir(const std::string& from, const std::string& to, const std::vector<std::string>& files) {
  CopyAssets(std::filesystem::path(from), std::filesystem::path(to), files);
}
}
namespace series {
SeriesSplits LoadSeriesDir(const std::string& root) { return LoadSeries(std::filesystem::path(root)); }
}
namespace sensor {
SensorSplits LoadSensorDir(const std::string& root) { return LoadSensor(std::filesystem::path(root)); }
void CompareWithTflowDir(const std::string& root, const std::string& tf_dir) {
  CompareWithTflow(std::filesystem::path(root), std::filesystem::path(tf_dir));
}
}
namespace charcnn {
CharacterSplits LoadCharactersDir(const std::string& root) { return LoadCharacters(std::filesystem::path(root)); }
}
namespace wavestack {
WaveSplits LoadWaveStackDir(const std::string& root) { return LoadWaveStack(std::filesystem::path(root)); }
}
namespace sal {
SaliencyCnnOutput SaliencyConvFeaturesRelu(nn::ParameterStore& p, const Tensor& image) { return SaliencyConvFeatures(p, image); }
SaliencyCnnOutput SaliencyCnnRelu(nn::ParameterStore& p, const Tensor& image) { return SaliencyCnn(p, image); }
/// Convolutional features with guided_backprop's GuidedRelu gate (the full guided network is guided::GuidedSaliencyCnn).
SaliencyCnnOutput SaliencyConvFeaturesGuided(nn::ParameterStore& p, const Tensor& image) {
  return SaliencyConvFeatures(p, image, Gate(guided::GuidedRelu));
}
nn::NamedTensors LoadArraysNamed(const std::string& file) {
  const auto arrays = LoadArrays(std::filesystem::path(file));
  return nn::NamedTensors(arrays.begin(), arrays.end());
}
void SaveArraysNamed(const std::string& file, const nn::NamedTensors& arrays) {
  SaveArrays(std::filesystem::path(file), Arrays(arrays.begin(), arrays.end()));
}
void SaliencyPngFile(const std::string& path, const Canvas& canvas) { SaliencyPng(std::filesystem::path(path), canvas); }
std::vector<double> SignedColorVector(double g) {
  const auto rgb = SignedColor(g);
  return {rgb[0], rgb[1], rgb[2]};
}
std::string SourceDirString(const std::string& value) { return SourceDir(value).string(); }
std::string DefaultSourceString() { return DefaultSource().string(); }
/// RestoreSource; returns whether `source` is our own run (best.npz present) rather than tflow's record.
bool RestoreSourceDir(const std::string& source, SaliencyModel& model, SaliencyModel& initial) {
  return RestoreSource(std::filesystem::path(source), model, initial).ours;
}
void SaliencyGalleryDir(const std::string& directory, const SaliencyData& examples, const Tensor& targets, const Tensor& runners,
                        const Tensor& gradients, const Tensor& alternative, const Tensor& random, double peak) {
  SaliencyGallery(std::filesystem::path(directory), examples, targets, runners, gradients, alternative, random, peak);
}
}
%}

// ---- object detection components (1.6): scenes, box utilities, RPN / RoI heads, RetinaNet,
// FCOS, SSD, CenterNet, DETR, EfficientDet / BiFPN, mosaic and affine scene transforms ----
// Boxes are std::array<float, 4> ([ymin, xmin, ymax, xmax)) exposed as detector.Box. Raw
// float-pointer decoders and the std::mt19937_64 sampling helpers get vector / engine-handle
// adapters below; the templated `Detect*` chunked-inference helpers and DetectorHooks
// (std::function) are not bound: the host examples write those loops out over the bound
// forward functions and Decode* functions.
// bound-header: detection.hpp
// bound-header: detection_scenes.hpp
// bound-header: fpn.hpp
// bound-header: two_stage.hpp
// bound-header: retinanet.hpp
// bound-header: fcos.hpp
// bound-header: ssd.hpp
// bound-header: centernet.hpp
// bound-header: detr.hpp
// bound-header: hungarian.hpp
// bound-header: efficientdet.hpp
// bound-header: bifpn.hpp
// bound-header: detection_mosaic.hpp
// bound-header: detection_affine.hpp
%{
#include "common/detection.hpp"
#include "common/detection_scenes.hpp"
#include "common/fpn.hpp"
#include "common/two_stage.hpp"
#include "common/retinanet.hpp"
#include "common/fcos.hpp"
#include "common/ssd.hpp"
#include "common/centernet.hpp"
#include "common/hungarian.hpp"
#include "common/detr.hpp"
#include "common/bifpn.hpp"
#include "common/efficientdet.hpp"
#include "common/detection_mosaic.hpp"
#include "common/detection_affine.hpp"
%}
%rename("detector__%(strip:[detector::])s", regextarget=1, fullname=1) "^detector::[^:]+$";
%rename("fpn__%(strip:[fpn::])s", regextarget=1, fullname=1) "^fpn::[^:]+$";
%rename("retinanet__%(strip:[retinanet::])s", regextarget=1, fullname=1) "^retinanet::[^:]+$";
%rename("fcos__%(strip:[fcos::])s", regextarget=1, fullname=1) "^fcos::[^:]+$";
%rename("ssd__%(strip:[ssd::])s", regextarget=1, fullname=1) "^ssd::[^:]+$";
%rename("centernet__%(strip:[centernet::])s", regextarget=1, fullname=1) "^centernet::[^:]+$";
%rename("detr__%(strip:[detr::])s", regextarget=1, fullname=1) "^detr::[^:]+$";
%rename("hungarian__%(strip:[hungarian::])s", regextarget=1, fullname=1) "^hungarian::[^:]+$";
%rename("efficientdet__%(strip:[efficientdet::])s", regextarget=1, fullname=1) "^efficientdet::[^:]+$";
%rename("bifpn__%(strip:[bifpn::])s", regextarget=1, fullname=1) "^bifpn::[^:]+$";

#ifdef SWIGR
// R has no std_array.i: declare the fixed-size array interface the box types need
// (size / get / set); Python gets SWIG's full std::array wrapper. Without the feature below,
// declaring the template makes SWIG route the by-value std::array<std::int64_t, N> option
// arguments (nn.i NN_R_ARRAY typemaps) through SwigValueWrapper, which has no operator[].
%feature("novaluewrapper") std::array;
namespace std {
template <class T, size_t N> struct array {
  array();
  size_t size() const;
  %extend {
    T get(int index) const {
      if (index < 0 || index >= int(N)) throw std::out_of_range("array index");
      return (*$self)[static_cast<size_t>(index)];
    }
    void set(int index, T value) {
      if (index < 0 || index >= int(N)) throw std::out_of_range("array index");
      (*$self)[static_cast<size_t>(index)] = value;
    }
  }
};
}
#endif
%template(detector__Box) std::array<float, 4>;
%template(detector__BoxVector) std::vector<std::array<float, 4> >;
%template(detector__BoxVectorVector) std::vector<std::vector<std::array<float, 4> > >;
%template(detector__BoxVariances) std::array<double, 4>;
%template(IntVectorVector) std::vector<std::vector<int> >;
%template(DoubleVectorVector) std::vector<std::vector<double> >;
%template(IntQuad) std::array<int, 4>;
%template(IntQuadVector) std::vector<std::array<int, 4> >;

// detection.hpp
%ignore detector::DetectionScenes::image;
%ignore detector::ClassAp;
%ignore detector::DecodeBox;
%ignore detector::DetectorHooks;
%ignore detector::TrainingDirectory;
%include "common/detection.hpp"
%template(detector__DetectionVector) std::vector<detector::Detection>;
%template(detector__RpnLevelVector) std::vector<detector::RpnLevel>;
// detection_scenes.hpp
%ignore detector::LoadDigits;
%ignore detector::StandardScenes;
%ignore detector::CropScene;
%ignore detector::DecodeBoxWith;
%include "common/detection_scenes.hpp"
%template(detector__ScenePair) std::pair<detector::DetectionScenes, detector::DetectionScenes>;
%include "common/fpn.hpp"
// two_stage.hpp: the std::mt19937_64 samplers take the detector.Mt19937_64 engine handle below.
%ignore detector::SampleRois;
%ignore detector::MakeRpnTargets;
%ignore detector::RpnProposals;
%include "common/two_stage.hpp"
%include "common/retinanet.hpp"
%include "common/fcos.hpp"
%template(fcos__FcosPointVector) std::vector<fcos::FcosPoint>;
%ignore ssd::SsdLossOutputs::selected_negatives;
%include "common/ssd.hpp"
%extend ssd::SsdLossOutputs {
  /// Hard-negative selection mask [N * A] as 0/1 integers.
  std::vector<int> selected_negatives_mask() const {
    return std::vector<int>($self->selected_negatives.begin(), $self->selected_negatives.end());
  }
}
NN_COMPONENT_INT_ARRAY(centernet::CenterNetConfig, upsample_widths, 3)
%ignore centernet::Grid::at;
%include "common/centernet.hpp"
%include "common/hungarian.hpp"
%ignore detr::DetrCorners;
%include "common/detr.hpp"
%include "common/bifpn.hpp"
%include "common/efficientdet.hpp"
%include "common/detection_mosaic.hpp"
%template(detector__DetectionTransformVector) std::vector<detector::DetectionTransform>;
%include "common/detection_affine.hpp"

%rename(detector__ClassAp) detector::ClassApVector;
%rename(detector__DecodeBox) detector::DecodeBoxVector;
%rename(detector__DecodeBoxWith) detector::DecodeBoxWithVector;
%rename(detector__TrainingDirectory) detector::TrainingDirectoryString;
%rename(detector__LoadDigits) detector::LoadDigitsDir;
%rename(detector__StandardScenes) detector::StandardScenesDir;
%rename(detector__CropScene) detector::CropSceneTensor;
%rename(detector__SampleRois) detector::SampleRoisEngine;
%rename(detector__MakeRpnTargets) detector::MakeRpnTargetsEngine;
%rename(detector__RpnProposals) detector::RpnProposalsVector;
%rename(detr__DetrCorners) detr::DetrCornersVector;
%inline %{
namespace detector {
/// [ymin, xmin, ymax, xmax) from four values; BoxValues reads one back as a plain vector.
Box MakeBox(float ymin, float xmin, float ymax, float xmax) { return {ymin, xmin, ymax, xmax}; }
std::vector<float> BoxValues(const Box& box) { return {box[0], box[1], box[2], box[3]}; }
nn::BoxVariances MakeBoxVariances(double y, double x, double height, double width) { return {y, x, height, width}; }
/// A std::mt19937_64 the host example owns, for the samplers that draw from one
/// (SampleRois, MakeRpnTargets) and the std::shuffle / uniform draws around them.
struct Mt19937_64 {
  std::mt19937_64 engine;
  explicit Mt19937_64(std::uint64_t seed) : engine(seed) {}
  std::uint64_t next() { return engine(); }
  /// std::uniform_int_distribution<int>(low, high)(engine).
  int uniform_int(int low, int high) { return std::uniform_int_distribution<int>(low, high)(engine); }
  /// std::uniform_real_distribution<float>(low, high)(engine).
  float uniform_float(float low, float high) { return std::uniform_real_distribution<float>(low, high)(engine); }
  /// std::shuffle of the values with this engine.
  std::vector<int> shuffle(std::vector<int> values) {
    std::shuffle(values.begin(), values.end(), engine);
    return values;
  }
};
std::vector<double> ClassApVector(const std::vector<Detection>& detections, const DetectionScenes& scenes,
                                  const std::vector<int>& rows, int label, float threshold, bool voc) {
  const auto ap = ClassAp(detections, scenes, rows, label, threshold, voc);
  return {ap[0], ap[1], ap[2]};
}
Box DecodeBoxVector(const Box& anchor, const std::vector<float>& delta) {
  if (delta.size() != 4) throw std::invalid_argument("box delta takes four values");
  return DecodeBox(anchor, delta.data());
}
Box DecodeBoxWithVector(const Box& anchor, const std::vector<float>& delta, const nn::BoxVariances& variance) {
  if (delta.size() != 4) throw std::invalid_argument("box delta takes four values");
  return DecodeBoxWith(anchor, delta.data(), variance);
}
std::string TrainingDirectoryString(const std::string& name) { return TrainingDirectory(name).string(); }
Digits LoadDigitsDir(const std::string& directory, const std::string& split) {
  return LoadDigits(std::filesystem::path(directory), split);
}
std::pair<DetectionScenes, DetectionScenes> StandardScenesDir(const std::string& directory, int size = 64) {
  return StandardScenes(std::filesystem::path(directory), size);
}
/// CropScene as a [size, size] float tensor in [-1, 1].
nn::Tensor CropSceneTensor(const DetectionScenes& scenes, int row, const Box& box, int size) {
  std::vector<float> out(std::size_t(size) * size);
  CropScene(scenes, row, box, size, out.data());
  return nn::from_vector(out, {size, size});
}
void SampleRoisEngine(const DetectionScenes& scenes, int row, int index, const std::vector<Box>& candidates, int total,
                      int background, int stride, int feature_size, Mt19937_64& random, RoiBatch& batch) {
  SampleRois(scenes, row, index, candidates, total, background, stride, feature_size, random.engine, batch);
}
RpnTargets MakeRpnTargetsEngine(const std::vector<Box>& anchors, const DetectionScenes& scenes, const std::vector<int>& rows,
                                Mt19937_64& random, int per_image = 128) {
  return MakeRpnTargets(anchors, scenes, rows, random.engine, per_image);
}
/// RpnProposals of one image from its objectness [A] and deltas [A * 4] values.
std::vector<Box> RpnProposalsVector(const std::vector<float>& objectness, const std::vector<float>& deltas,
                                    const std::vector<Box>& anchors, int height, int width, int before = 300, int after = 100) {
  if (objectness.size() != anchors.size() || deltas.size() != 4 * anchors.size())
    throw std::invalid_argument("RPN proposals take one objectness and four deltas per anchor");
  return RpnProposals(objectness.data(), deltas.data(), anchors, height, width, before, after);
}
}
namespace detr {
detector::Box DetrCornersVector(const std::vector<float>& center_size) {
  if (center_size.size() != 4) throw std::invalid_argument("DETR boxes take four values");
  return DetrCorners(center_size.data());
}
}
%}

// ---- YOLO family, Mask R-CNN / instance / panoptic / YOLACT, detection evaluation, misc ------
// Callable templates (DetectYolo*, DetectPixelRois, DetectYolact) stay C++-only: the headers now
// expose their per-level/per-chunk decoders (DecodeYoloV3Level, DecodePixelRois, DecodeYolact ...)
// and the host example writes the chunk loop. Raw-pointer decoders take a value vector + offset.
// bound-header: yolo_v1.hpp
// bound-header: yolo_v2.hpp
// bound-header: yolo_v3.hpp
// bound-header: yolo_v5.hpp
// bound-header: word_tree.hpp
// bound-header: mask_rcnn.hpp
// bound-header: instance_segmentation.hpp
// bound-header: panoptic.hpp
// bound-header: panoptic_fpn.hpp
// bound-header: panoptic_scenes.hpp
// bound-header: yolact.hpp
// bound-header: detection_evaluation.hpp
// bound-header: detection_metrics.hpp
// bound-header: highway.hpp
// bound-header: stn.hpp
// bound-header: capsnet.hpp
// bound-header: im2col.hpp
// bound-header: receptive_field.hpp
%{
#include "common/word_tree.hpp"
#include "common/yolo_v1.hpp"
#include "common/yolo_v2.hpp"
#include "common/yolo_v3.hpp"
#include "common/yolo_v5.hpp"
#include "common/instance_segmentation.hpp"
#include "common/mask_rcnn.hpp"
#include "common/panoptic.hpp"
#include "common/panoptic_fpn.hpp"
#include "common/panoptic_scenes.hpp"
#include "common/yolact.hpp"
#include "common/detection_evaluation.hpp"
#include "common/detection_metrics.hpp"
#include "common/highway.hpp"
#include "common/stn.hpp"
#include "common/capsnet.hpp"
#include "common/im2col.hpp"
#include "common/receptive_field.hpp"
%}
%rename("yolo__%(strip:[yolo::])s", regextarget=1, fullname=1) "^yolo::[^:]+$";
%rename("wordtree__%(strip:[wordtree::])s", regextarget=1, fullname=1) "^wordtree::[^:]+$";
%rename("maskrcnn__%(strip:[maskrcnn::])s", regextarget=1, fullname=1) "^maskrcnn::[^:]+$";
%rename("instance__%(strip:[instance::])s", regextarget=1, fullname=1) "^instance::[^:]+$";
%rename("panoptic__%(strip:[panoptic::])s", regextarget=1, fullname=1) "^panoptic::[^:]+$";
%rename("yolact__%(strip:[yolact::])s", regextarget=1, fullname=1) "^yolact::[^:]+$";
%rename("detection_evaluation__%(strip:[detection_evaluation::])s", regextarget=1, fullname=1) "^detection_evaluation::[^:]+$";
%rename("detection__%(strip:[detection::])s", regextarget=1, fullname=1) "^detection::[^:]+$";
%rename("highway__%(strip:[highway::])s", regextarget=1, fullname=1) "^highway::[^:]+$";
%rename("stn__%(strip:[stn::])s", regextarget=1, fullname=1) "^stn::[^:]+$";
%rename("capsnet__%(strip:[capsnet::])s", regextarget=1, fullname=1) "^capsnet::[^:]+$";
%rename("im2col__%(strip:[im2col::])s", regextarget=1, fullname=1) "^im2col::[^:]+$";
%rename("receptive__%(strip:[receptive::])s", regextarget=1, fullname=1) "^receptive::[^:]+$";

%template(IntPair) std::pair<int, int>;
%template(IntPairVector) std::vector<std::pair<int, int> >;
%template(TensorTriple) std::array<datamunge::nn::Tensor, 3>;
%template(yolo__Dimension) std::array<float, 2>;
%template(yolo__DimensionVector) std::vector<std::array<float, 2> >;
%template(IntBoxPair) std::pair<int, std::array<float, 4> >;
%template(IntBoxPairVector) std::vector<std::pair<int, std::array<float, 4> > >;

// word_tree.hpp: the parent -> children map is read through group(parent).
%ignore wordtree::WordTree::groups;
%include "common/word_tree.hpp"
%extend wordtree::WordTree {
  /// Children of `parent` (-1 for the roots); empty when it has none.
  std::vector<int> group(int parent) const {
    const auto it = $self->groups.find(parent);
    return it == $self->groups.end() ? std::vector<int>{} : it->second;
  }
}

// yolo_v1.hpp: the classifier's optimizer step lives in the host example (Loss + WeightPenalty).
%immutable yolo::YoloV1Classifier::parameters;
%immutable yolo::YoloV1Classifier::state;
%ignore yolo::YoloV1PretrainedWeights::parameters;
%ignore yolo::YoloV1PretrainedWeights::state;
%ignore yolo::DetectYoloV1;
%ignore yolo::WriteYoloV1Bootstrap;
%include "common/yolo_v1.hpp"
%extend yolo::YoloV1PretrainedWeights {
  std::vector<std::string> parameter_names() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : $self->parameters) names.push_back(name);
    return names;
  }
  std::vector<std::string> state_names() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : $self->state) names.push_back(name);
    return names;
  }
  nn::Tensor parameter(const std::string& name) const { return $self->parameters.at(name); }
  nn::Tensor state_value(const std::string& name) const { return $self->state.at(name); }
}
// yolo_v2.hpp
%ignore yolo::DecodeYoloV2;
%ignore yolo::DetectYoloV2;
%include "common/yolo_v2.hpp"
// yolo_v3.hpp
%ignore yolo::YoloV3Targets::levels;
%ignore yolo::MakeYoloV3Targets;
%ignore yolo::DetectYoloV3;
%include "common/yolo_v3.hpp"
%extend yolo::YoloV3Targets {
  /// Targets of head `level` (0 coarse .. 2 fine).
  yolo::YoloV3LevelTargets level(int index) const {
    if (index < 0 || index > 2) throw std::out_of_range("YOLO v3 has three head levels");
    return $self->levels[static_cast<std::size_t>(index)];
  }
}
// yolo_v5.hpp: CspContext holds references; the host example owns one through CspContextHandle.
%nodefaultctor yolo::CspContext;
%nodefaultdtor yolo::CspContext;
%ignore yolo::YoloV5Targets::levels;
%ignore yolo::MakeYoloV5Targets;
%ignore yolo::DecodeYoloV5;
%ignore yolo::DetectYoloV5;
%include "common/yolo_v5.hpp"
%extend yolo::YoloV5Targets {
  yolo::YoloV5LevelTargets level(int index) const {
    if (index < 0 || index > 2) throw std::out_of_range("YOLO v5 has three head levels");
    return $self->levels[static_cast<std::size_t>(index)];
  }
}
// instance_segmentation.hpp
%ignore instance::InstanceScenes::instance_masks;
%ignore instance::PasteMask;
%ignore instance::ExportInstanceMasks;
%include "common/instance_segmentation.hpp"
%template(instance__MaskDetectionVector) std::vector<instance::MaskDetection>;
%extend instance::InstanceScenes {
  /// Number of object supports of `scene`.
  int instance_count(int scene) const { return int($self->instance_masks.at(static_cast<std::size_t>(scene)).size()); }
  /// H x W 0/1 support of object `object` of `scene`.
  instance::BinaryMask instance_mask(int scene, int object) const {
    return $self->instance_masks.at(static_cast<std::size_t>(scene)).at(static_cast<std::size_t>(object));
  }
}
// mask_rcnn.hpp
%ignore maskrcnn::DetectPixelRois;
%include "common/mask_rcnn.hpp"
%template(maskrcnn__PositiveMaskTargets) maskrcnn::PositiveMaskTargets<instance::InstanceScenes>;
// panoptic.hpp: the id -> segment / id -> PQ statistic maps are read through accessors and PqStatsHandle.
%ignore panoptic::PanopticMap::segments;
%ignore panoptic::PanopticAreas;
%ignore panoptic::PanopticStats;
%ignore panoptic::AddPanopticStats;
%ignore panoptic::PanopticQuality;
%include "common/panoptic.hpp"
%template(panoptic__PanopticMapVector) std::vector<panoptic::PanopticMap>;
%extend panoptic::PanopticMap {
  std::vector<int> segment_ids() const {
    std::vector<int> ids;
    for (const auto& [id, _] : $self->segments) ids.push_back(id);
    return ids;
  }
  panoptic::PanopticSegment segment(int id) const { return $self->segments.at(id); }
  bool has_segment(int id) const { return $self->segments.count(id) != 0; }
}
%include "common/panoptic_fpn.hpp"
%ignore panoptic::PanopticPreview;
%include "common/panoptic_scenes.hpp"
// yolact.hpp
%ignore yolact::YolactLoss::negatives;
%ignore yolact::MakeYolactTargets;
%ignore yolact::DetectYolact;
%include "common/yolact.hpp"
%extend yolact::YolactLoss {
  /// Mined hard negatives [B * A] as 0/1 integers.
  std::vector<int> negatives_mask() const { return std::vector<int>($self->negatives.begin(), $self->negatives.end()); }
}
// detection_evaluation.hpp
%ignore detection_evaluation::EvaluationImage::filename;
%ignore detection_evaluation::DatasetAp::per_class50;
%ignore detection_evaluation::LoadEvaluationImages;
%ignore detection_evaluation::LoadEvaluationTruth;
%ignore detection_evaluation::LoadPredictionsJson;
%include "common/detection_evaluation.hpp"
%template(detection_evaluation__EvaluationTruthVector) std::vector<detection_evaluation::EvaluationTruth>;
%template(detection_evaluation__EvaluationImageVector) std::vector<detection_evaluation::EvaluationImage>;
%extend detection_evaluation::EvaluationImage {
  std::string filename_string() const { return $self->filename.string(); }
}
%extend detection_evaluation::DatasetAp {
  std::vector<int> per_class50_labels() const {
    std::vector<int> labels;
    for (const auto& [label, _] : $self->per_class50) labels.push_back(label);
    return labels;
  }
  double per_class50_value(int label) const { return $self->per_class50.at(label); }
}
// detection_metrics.hpp (its own double-valued detection::Box / Detection, distinct from detector::)
%include "common/detection_metrics.hpp"
%template(detection__GroundTruthVector) std::vector<detection::GroundTruth>;
%template(detection__DetectionVector) std::vector<detection::Detection>;
%include "common/highway.hpp"
%include "common/stn.hpp"
// capsnet.hpp: the decoder closure is called through Reconstruct(mask).
%ignore capsnet::CapsNetOutputs::reconstruct;
%include "common/capsnet.hpp"
%extend capsnet::CapsNetOutputs {
  /// Decoder output [N, 784] in (0, 1) from the digit capsules masked by `mask` [N, 10].
  nn::Tensor Reconstruct(const nn::Tensor& mask) const { return $self->reconstruct(mask); }
}
%include "common/im2col.hpp"
%include "common/receptive_field.hpp"

%rename(yolo__WriteYoloV1Bootstrap) yolo::WriteYoloV1BootstrapPath;
%rename(yolo__DecodeYoloV2) yolo::DecodeYoloV2Values;
%rename(yolo__MakeYoloV3Targets) yolo::MakeYoloV3TargetsTensors;
%rename(yolo__MakeYoloV5Targets) yolo::MakeYoloV5TargetsTensors;
%rename(yolo__DecodeYoloV5) yolo::DecodeYoloV5Values;
%rename(instance__PasteMask) instance::PasteMaskValues;
%rename(instance__ExportInstanceMasks) instance::ExportInstanceMasksPath;
%rename(panoptic__PanopticAreas) panoptic::PanopticAreasList;
%rename(panoptic__PanopticStats) panoptic::PanopticStatsHandle;
%rename(panoptic__PanopticQuality) panoptic::PanopticQualityHandle;
%rename(panoptic__PanopticPreview) panoptic::PanopticPreviewPath;
%rename(yolact__MakeYolactTargets) yolact::MakeYolactTargetsEngine;
%rename(detection_evaluation__LoadEvaluationImages) detection_evaluation::LoadEvaluationImagesDir;
%rename(detection_evaluation__LoadEvaluationTruth) detection_evaluation::LoadEvaluationTruthDir;
%rename(detection_evaluation__LoadPredictionsJson) detection_evaluation::LoadPredictionsJsonFile;
%inline %{
namespace yolo {
/// Height/width prior from two values.
Dimension MakeDimension(float height, float width) { return {height, width}; }
std::vector<float> DimensionValues(const Dimension& d) { return {d[0], d[1]}; }
void WriteYoloV1BootstrapPath(const YoloV1PretrainedWeights& weights, const YoloV1Config& cfg, const std::string& prefix) {
  WriteYoloV1Bootstrap(weights, cfg, std::filesystem::path(prefix));
}
/// DecodeYoloV2 of the cell values starting at `offset` in `raw` (a prediction's to_vector values).
Box DecodeYoloV2Values(const std::vector<float>& raw, int offset, int row, int col, int cells, const Dimension& prior, int height, int width) {
  if (offset < 0 || std::size_t(offset) + 5 > raw.size()) throw std::out_of_range("YOLO v2 cell offset");
  return DecodeYoloV2(raw.data() + offset, row, col, cells, prior, height, width);
}
/// MakeYoloV3Targets from the three raw head tensors (coarse, middle, fine).
YoloV3Targets MakeYoloV3TargetsTensors(const nn::Tensor& coarse, const nn::Tensor& middle, const nn::Tensor& fine,
                                       const std::vector<Dimension>& priors, const DetectionScenes& scenes, const std::vector<int>& rows) {
  return MakeYoloV3Targets({coarse.to_vector<float>(), middle.to_vector<float>(), fine.to_vector<float>()},
                           {coarse.shape(), middle.shape(), fine.shape()}, priors, scenes, rows);
}
/// MakeYoloV5Targets from the three raw head tensors' shapes (coarse, middle, fine).
YoloV5Targets MakeYoloV5TargetsTensors(const nn::Tensor& coarse, const nn::Tensor& middle, const nn::Tensor& fine,
                                       const std::vector<Dimension>& priors, const DetectionScenes& scenes, const std::vector<int>& rows) {
  return MakeYoloV5Targets({coarse.shape(), middle.shape(), fine.shape()}, priors, scenes, rows);
}
Box DecodeYoloV5Values(const std::vector<float>& raw, int offset, int row, int col, int cells, const Dimension& prior, int stride, int height,
                       int width) {
  if (offset < 0 || std::size_t(offset) + 5 > raw.size()) throw std::out_of_range("YOLO v5 cell offset");
  return DecodeYoloV5(raw.data() + offset, row, col, cells, prior, stride, height, width);
}
/// A CspContext over the host example's stores (the struct itself holds references).
struct CspContextHandle {
  CspContext context;
  CspContextHandle(nn::ParameterStore& parameters, nn::ParameterStore& state, bool training = false, bool update_stats = true)
      : context{parameters, state, training, update_stats} {}
  CspContext& get() { return context; }
};
}
namespace instance {
/// PasteMask of the `side * side * classes` probabilities starting at `offset` in `probabilities`.
BinaryMask PasteMaskValues(const std::vector<float>& probabilities, int offset, int side, int classes, int label, const Box& box,
                           int height, int width) {
  if (offset < 0 || std::size_t(offset) + std::size_t(side) * side * classes > probabilities.size())
    throw std::out_of_range("mask probabilities offset");
  return PasteMask(probabilities.data() + offset, side, classes, label, box, height, width);
}
void ExportInstanceMasksPath(const InstanceScenes& scenes, const std::vector<MaskDetection>& detections, const std::string& filename) {
  ExportInstanceMasks(scenes, detections, std::filesystem::path(filename));
}
}
namespace panoptic {
/// [segment id, pixel area] rows of PanopticAreas.
std::vector<std::vector<int>> PanopticAreasList(const PanopticMap& map) {
  std::vector<std::vector<int>> rows;
  for (const auto& [id, area] : PanopticAreas(map)) rows.push_back({id, area});
  return rows;
}
/// Per-class PQ statistics (a std::map<int, PqStat> the host example accumulates with add()).
struct PqStatsHandle {
  PqStats stats;
  std::vector<int> labels() const {
    std::vector<int> result;
    for (const auto& [label, _] : stats) result.push_back(label);
    return result;
  }
  PqStat at(int label) const { return stats.at(label); }
  bool has(int label) const { return stats.count(label) != 0; }
  /// AddPanopticStats(this, sample).
  void add(const PqStatsHandle& sample) { AddPanopticStats(stats, sample.stats); }
};
PqStatsHandle PanopticStatsHandle(const PanopticMap& truth, const PanopticMap& prediction) {
  PqStatsHandle handle;
  handle.stats = PanopticStats(truth, prediction);
  return handle;
}
/// PanopticQuality with `things` given as 0/1 per class.
PqResult PanopticQualityHandle(const PqStatsHandle& stats, const std::vector<int>& things, int subset = -1) {
  return PanopticQuality(stats.stats, std::vector<bool>(things.begin(), things.end()), subset);
}
void PanopticPreviewPath(const PanopticScenes& data, const std::vector<std::vector<int>>& semantic, const std::vector<PanopticMap>& predicted,
                         const std::string& filename) {
  PanopticPreview(data, semantic, predicted, std::filesystem::path(filename));
}
}
namespace yolact {
/// A std::mt19937 the host example owns for MakeYolactTargets' positive-mask subsampling.
struct Mt19937 {
  std::mt19937 engine;
  explicit Mt19937(std::uint32_t seed) : engine(seed) {}
  std::uint32_t next() { return engine(); }
};
YolactTargets MakeYolactTargetsEngine(const YolactOutputs& net, const std::vector<Box>& anchors, const instance::InstanceScenes& scenes,
                                      const std::vector<int>& rows, int classes, Mt19937& random) {
  return MakeYolactTargets(net, anchors, scenes, rows, classes, random.engine);
}
}
namespace detection_evaluation {
std::vector<EvaluationImage> LoadEvaluationImagesDir(const std::string& directory) {
  return LoadEvaluationImages(std::filesystem::path(directory));
}
std::vector<EvaluationTruth> LoadEvaluationTruthDir(const std::string& directory) {
  return LoadEvaluationTruth(std::filesystem::path(directory));
}
std::vector<Detection> LoadPredictionsJsonFile(const std::string& path) { return LoadPredictionsJson(std::filesystem::path(path)); }
}
%}
