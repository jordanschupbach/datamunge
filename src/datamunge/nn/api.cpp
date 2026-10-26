// Binding-friendly facade over datamunge::nn (see api.hpp).

#include "datamunge/nn/api.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <sstream>
#include <stdexcept>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/conv.hpp"
#include "datamunge/nn/io.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/losses.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/optim.hpp"
#include "datamunge/nn/parallel.hpp"
#include "datamunge/nn/params.hpp"
#include "datamunge/nn/random.hpp"

namespace datamunge::nn {

namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

DType parse_dtype(const std::string& name) {
  const std::string n = lower(name);
  if (n == "float32" || n == "float") return DType::Float32;
  if (n == "float64" || n == "double") return DType::Float64;
  if (n == "int32") return DType::Int32;
  if (n == "int64") return DType::Int64;
  if (n == "bool") return DType::Bool;
  throw std::invalid_argument("unknown dtype '" + name + "' (use float32, float64, int32, int64 or bool)");
}

Shape to_shape(const std::vector<int>& v) { return Shape(v.begin(), v.end()); }
Axes  to_axes(const std::vector<int>& v) { return Axes(v.begin(), v.end()); }

Activation parse_activation(const std::string& name) {
  const std::string n = lower(name);
  if (n == "none" || n == "linear" || n.empty()) return Activation::None;
  for (int k = 0; k <= static_cast<int>(Activation::Tanh); ++k) {
    const auto a = static_cast<Activation>(k);
    if (activation_name(a) == n) return a;
  }
  throw std::invalid_argument("unknown activation '" + name + "'");
}

Padding parse_padding(const std::string& name) {
  const std::string n = lower(name);
  if (n == "same") return Padding::Same;
  if (n == "valid") return Padding::Valid;
  throw std::invalid_argument("padding must be 'same' or 'valid', got '" + name + "'");
}

OptimizerKind parse_optimizer(const std::string& name) {
  const std::string n = lower(name);
  for (int k = 0; k <= static_cast<int>(OptimizerKind::Adadelta); ++k) {
    const auto kind = static_cast<OptimizerKind>(k);
    if (lower(optimizer_name(kind)) == n) return kind;
  }
  throw std::invalid_argument("unknown optimizer '" + name + "'");
}

const Tensor& need(const NnTensor& t, const char* what) {
  if (!t.defined()) throw std::invalid_argument(std::string(what) + ": tensor is undefined");
  return t.tensor();
}

}  // namespace

// ---- NnTensor ------------------------------------------------------------------------------

NnTensor::NnTensor() = default;
NnTensor::NnTensor(Tensor tensor) : tensor_(std::move(tensor)) {}

NnTensor::NnTensor(const std::vector<double>& values, const std::vector<int>& shape, const std::string& dtype)
    : tensor_(cast(from_vector(values, to_shape(shape)), parse_dtype(dtype))) {}

NnTensor NnTensor::zeros(const std::vector<int>& shape, const std::string& dtype) {
  return NnTensor(nn::zeros(to_shape(shape), parse_dtype(dtype)));
}
NnTensor NnTensor::ones(const std::vector<int>& shape, const std::string& dtype) {
  return NnTensor(nn::ones(to_shape(shape), parse_dtype(dtype)));
}
NnTensor NnTensor::full(const std::vector<int>& shape, double value, const std::string& dtype) {
  return NnTensor(nn::full(to_shape(shape), value, parse_dtype(dtype)));
}
NnTensor NnTensor::arange(double start, double stop, double step, const std::string& dtype) {
  return NnTensor(cast(nn::arange(start, stop, step, DType::Float64), parse_dtype(dtype)));
}
NnTensor NnTensor::randn(const std::vector<int>& shape, unsigned seed, double mean, double std, const std::string& dtype) {
  Generator gen(seed);
  return NnTensor(nn::normal(to_shape(shape), mean, std, gen, parse_dtype(dtype)));
}
NnTensor NnTensor::uniform(const std::vector<int>& shape, unsigned seed, double low, double high,
                           const std::string& dtype) {
  Generator gen(seed);
  return NnTensor(nn::uniform(to_shape(shape), low, high, gen, parse_dtype(dtype)));
}
NnTensor NnTensor::load_npy(const std::string& path) { return NnTensor(nn::load_npy(path)); }
void     NnTensor::save_npy(const std::string& path) const { nn::save_npy(path, need(*this, "save_npy")); }

bool                NnTensor::defined() const { return tensor_.defined(); }
std::vector<double> NnTensor::values() const { return need(*this, "values").to_vector<double>(); }
std::vector<int>    NnTensor::shape() const {
  const Shape& s = need(*this, "shape").shape();
  return std::vector<int>(s.begin(), s.end());
}
std::string NnTensor::dtype() const { return dtype_name(need(*this, "dtype").dtype()); }
int         NnTensor::ndim() const { return static_cast<int>(need(*this, "ndim").dim()); }
int         NnTensor::numel() const { return static_cast<int>(need(*this, "numel").numel()); }
double      NnTensor::item() const { return need(*this, "item").item<double>(); }
double      NnTensor::at(const std::vector<int>& index) const {
  return need(*this, "at").at(std::vector<std::int64_t>(index.begin(), index.end()));
}

std::string NnTensor::str() const {
  if (!defined()) return "NnTensor(undefined)";
  std::ostringstream os;
  os << "NnTensor(shape=" << shape_str(tensor_.shape()) << ", dtype=" << dtype() << ", values=[";
  const auto v = values();
  for (std::size_t i = 0; i < v.size() && i < 12; ++i) os << (i ? ", " : "") << v[i];
  if (v.size() > 12) os << ", ...";
  os << "])";
  return os.str();
}

NnTensor NnTensor::astype(const std::string& dtype) const { return NnTensor(cast(need(*this, "astype"), parse_dtype(dtype))); }
NnTensor NnTensor::reshape(const std::vector<int>& shape) const {
  return NnTensor(nn::reshape(need(*this, "reshape"), to_shape(shape)));
}
NnTensor NnTensor::transpose(const std::vector<int>& perm) const {
  return NnTensor(nn::transpose(need(*this, "transpose"), to_axes(perm)));
}
NnTensor NnTensor::slice(int axis, int start, int stop) const {
  return NnTensor(slice_axis(need(*this, "slice"), axis, start, stop));
}
NnTensor NnTensor::concat(const NnTensor& other, int axis) const {
  return NnTensor(nn::concat({need(*this, "concat"), need(other, "concat")}, axis));
}

NnTensor NnTensor::add(const NnTensor& o) const { return NnTensor(nn::add(need(*this, "add"), need(o, "add"))); }
NnTensor NnTensor::sub(const NnTensor& o) const { return NnTensor(nn::sub(need(*this, "sub"), need(o, "sub"))); }
NnTensor NnTensor::mul(const NnTensor& o) const { return NnTensor(nn::mul(need(*this, "mul"), need(o, "mul"))); }
NnTensor NnTensor::div(const NnTensor& o) const { return NnTensor(nn::div(need(*this, "div"), need(o, "div"))); }
NnTensor NnTensor::add_scalar(double v) const { return NnTensor(nn::add(need(*this, "add_scalar"), v)); }
NnTensor NnTensor::mul_scalar(double v) const { return NnTensor(nn::mul(need(*this, "mul_scalar"), v)); }
NnTensor NnTensor::pow(double e) const { return NnTensor(nn::pow(need(*this, "pow"), e)); }
NnTensor NnTensor::neg() const { return NnTensor(nn::neg(need(*this, "neg"))); }
NnTensor NnTensor::matmul(const NnTensor& o, bool ta, bool tb) const {
  return NnTensor(nn::matmul(need(*this, "matmul"), need(o, "matmul"), ta, tb));
}

NnTensor NnTensor::exp() const { return NnTensor(nn::exp(need(*this, "exp"))); }
NnTensor NnTensor::log() const { return NnTensor(nn::log(need(*this, "log"))); }
NnTensor NnTensor::sqrt() const { return NnTensor(nn::sqrt(need(*this, "sqrt"))); }
NnTensor NnTensor::abs() const { return NnTensor(nn::abs(need(*this, "abs"))); }
NnTensor NnTensor::square() const { return NnTensor(nn::square(need(*this, "square"))); }
NnTensor NnTensor::tanh() const { return NnTensor(nn::tanh(need(*this, "tanh"))); }
NnTensor NnTensor::sigmoid() const { return NnTensor(nn::sigmoid(need(*this, "sigmoid"))); }
NnTensor NnTensor::relu() const { return NnTensor(nn::relu(need(*this, "relu"))); }
NnTensor NnTensor::activate(const std::string& a) const {
  return NnTensor(nn::activate(need(*this, "activate"), parse_activation(a)));
}
NnTensor NnTensor::softmax(int axis) const { return NnTensor(nn::softmax(need(*this, "softmax"), axis)); }
NnTensor NnTensor::log_softmax(int axis) const { return NnTensor(nn::log_softmax(need(*this, "log_softmax"), axis)); }

NnTensor NnTensor::sum(const std::vector<int>& axes, bool keepdims) const {
  return NnTensor(nn::sum(need(*this, "sum"), to_axes(axes), keepdims));
}
NnTensor NnTensor::mean(const std::vector<int>& axes, bool keepdims) const {
  return NnTensor(nn::mean(need(*this, "mean"), to_axes(axes), keepdims));
}
NnTensor NnTensor::max(const std::vector<int>& axes, bool keepdims) const {
  return NnTensor(nn::max(need(*this, "max"), to_axes(axes), keepdims));
}
NnTensor NnTensor::min(const std::vector<int>& axes, bool keepdims) const {
  return NnTensor(nn::min(need(*this, "min"), to_axes(axes), keepdims));
}
NnTensor NnTensor::argmax(int axis) const { return NnTensor(nn::argmax(need(*this, "argmax"), axis)); }

void NnTensor::set_requires_grad(bool value) { tensor_.requires_grad_(value); }
bool NnTensor::requires_grad() const { return defined() && tensor_.requires_grad(); }
void NnTensor::backward() { need(*this, "backward").backward(); }
NnTensor NnTensor::grad() const { return NnTensor(need(*this, "grad").grad()); }
void     NnTensor::zero_grad() { tensor_.zero_grad(); }
NnTensor NnTensor::detach() const { return NnTensor(need(*this, "detach").detach()); }

// ---- NnModel -------------------------------------------------------------------------------

namespace {

enum class LayerKind { Dense, Conv2d, MaxPool, AvgPool, GlobalAvgPool, BatchNorm, LayerNorm, Dropout, Activation, Flatten, Reshape };

struct LayerSpec {
  LayerKind        kind;
  std::string      name;
  int              units{0}, kernel{0}, stride{1};
  double           value{0.0}, value2{0.0};
  std::string      padding{"same"}, activation{"none"};
  std::vector<int> shape;
};

std::string kind_name(LayerKind k) {
  switch (k) {
    case LayerKind::Dense: return "dense";
    case LayerKind::Conv2d: return "conv";
    case LayerKind::MaxPool: return "maxpool";
    case LayerKind::AvgPool: return "avgpool";
    case LayerKind::GlobalAvgPool: return "globalavgpool";
    case LayerKind::BatchNorm: return "batchnorm";
    case LayerKind::LayerNorm: return "layernorm";
    case LayerKind::Dropout: return "dropout";
    case LayerKind::Activation: return "activation";
    case LayerKind::Flatten: return "flatten";
    case LayerKind::Reshape: return "reshape";
  }
  return "?";
}

}  // namespace

struct NnModel::Impl {
  explicit Impl(unsigned seed) : params(seed), stats(seed + 1), dropout_gen(seed + 2) {}
  ParameterStore                 params, stats;
  Generator                      dropout_gen;
  std::vector<LayerSpec>         layers;
  std::map<std::string, Tensor>  grads;
  std::unique_ptr<Optimizer>     optimizer;
  std::string                    optimizer_kind{"adam"};
  double                         learning_rate{1e-3};

  void add(LayerSpec spec) {
    spec.name = kind_name(spec.kind) + std::to_string(layers.size());
    layers.push_back(std::move(spec));
  }

  Tensor run(const Tensor& input, bool training) {
    Tensor x = input;
    if (!is_floating(x.dtype())) x = cast(x, params.dtype());
    for (const auto& l : layers) {
      switch (l.kind) {
        case LayerKind::Dense: x = dense(params, l.name, x, l.units, parse_activation(l.activation)); break;
        case LayerKind::Conv2d: {
          ConvLayerOptions o;
          o.stride     = l.stride;
          o.padding    = parse_padding(l.padding);
          o.activation = parse_activation(l.activation);
          x            = conv2d(params, l.name, x, l.units, l.kernel, o);
          break;
        }
        case LayerKind::MaxPool: x = max_pool2d(x, l.kernel, l.stride, parse_padding(l.padding)); break;
        case LayerKind::AvgPool: x = avg_pool2d(x, l.kernel, l.stride, parse_padding(l.padding)); break;
        case LayerKind::GlobalAvgPool: x = global_avg_pool2d(x); break;
        case LayerKind::BatchNorm: {
          BatchNormOptions o;
          o.momentum = l.value;
          o.eps      = l.value2;
          x          = batch_norm(params, stats, l.name, x, training, o);
          break;
        }
        case LayerKind::LayerNorm: x = layer_norm(params, l.name, x, l.value); break;
        case LayerKind::Dropout: x = dropout(x, l.value, training, dropout_gen); break;
        case LayerKind::Activation: x = activate(x, parse_activation(l.activation)); break;
        case LayerKind::Flatten: x = nn::reshape(x, {x.shape().empty() ? 1 : x.shape()[0], -1}); break;
        case LayerKind::Reshape: {
          Shape s{x.shape().empty() ? 1 : x.shape()[0]};
          s.insert(s.end(), l.shape.begin(), l.shape.end());
          x = nn::reshape(x, s);
          break;
        }
      }
    }
    return x;
  }

  Tensor loss_of(const Tensor& x, const Tensor& y, const std::string& name, bool training) {
    const Tensor      out = run(x, training);
    const std::string n   = lower(name);
    if (n == "mse") return mse(out, cast(y, out.dtype()));
    if (n == "softmax_cross_entropy" || n == "sparse_softmax_cross_entropy" || n == "cross_entropy")
      return sparse_softmax_cross_entropy(out, cast(y, DType::Int64));
    if (n == "sigmoid_cross_entropy" || n == "binary_cross_entropy")
      return sigmoid_cross_entropy(out, cast(nn::reshape(y, out.shape()), out.dtype()));
    throw std::invalid_argument("unknown loss '" + name + "' (mse, softmax_cross_entropy, sigmoid_cross_entropy)");
  }
};

NnModel::NnModel(unsigned seed) : impl_(std::make_unique<Impl>(seed)) {}
NnModel::~NnModel() = default;

void NnModel::add_dense(int units, const std::string& activation) {
  parse_activation(activation);
  LayerSpec l{LayerKind::Dense};
  l.units      = units;
  l.activation = activation;
  impl_->add(l);
}
void NnModel::add_conv2d(int filters, int kernel, int stride, const std::string& padding, const std::string& activation) {
  parse_activation(activation);
  parse_padding(padding);
  LayerSpec l{LayerKind::Conv2d};
  l.units = filters, l.kernel = kernel, l.stride = stride, l.padding = padding, l.activation = activation;
  impl_->add(l);
}
void NnModel::add_max_pool2d(int window, int stride, const std::string& padding) {
  parse_padding(padding);
  LayerSpec l{LayerKind::MaxPool};
  l.kernel = window, l.stride = stride, l.padding = padding;
  impl_->add(l);
}
void NnModel::add_avg_pool2d(int window, int stride, const std::string& padding) {
  parse_padding(padding);
  LayerSpec l{LayerKind::AvgPool};
  l.kernel = window, l.stride = stride, l.padding = padding;
  impl_->add(l);
}
void NnModel::add_global_avg_pool2d() { impl_->add(LayerSpec{LayerKind::GlobalAvgPool}); }
void NnModel::add_batch_norm(double momentum, double eps) {
  LayerSpec l{LayerKind::BatchNorm};
  l.value = momentum, l.value2 = eps;
  impl_->add(l);
}
void NnModel::add_layer_norm(double eps) {
  LayerSpec l{LayerKind::LayerNorm};
  l.value = eps;
  impl_->add(l);
}
void NnModel::add_dropout(double rate) {
  if (rate < 0.0 || rate >= 1.0) throw std::invalid_argument("add_dropout: rate must be in [0, 1)");
  LayerSpec l{LayerKind::Dropout};
  l.value = rate;
  impl_->add(l);
}
void NnModel::add_activation(const std::string& activation) {
  parse_activation(activation);
  LayerSpec l{LayerKind::Activation};
  l.activation = activation;
  impl_->add(l);
}
void NnModel::add_flatten() { impl_->add(LayerSpec{LayerKind::Flatten}); }
void NnModel::add_reshape(const std::vector<int>& shape) {
  LayerSpec l{LayerKind::Reshape};
  l.shape = shape;
  impl_->add(l);
}

int NnModel::num_layers() const { return static_cast<int>(impl_->layers.size()); }

std::string NnModel::summary() const {
  std::ostringstream os;
  for (const auto& l : impl_->layers) {
    os << l.name;
    for (const auto& name : impl_->params.names())
      if (name.rfind(l.name + "/", 0) == 0) os << "  " << name << shape_str(impl_->params.at(name).shape());
    os << '\n';
  }
  os << "parameters: " << impl_->params.count() << '\n';
  return os.str();
}

NnTensor NnModel::forward(const NnTensor& x, bool training) { return NnTensor(impl_->run(need(x, "forward"), training)); }

NnTensor NnModel::predict(const NnTensor& x) {
  NoGradGuard guard;
  return NnTensor(impl_->run(need(x, "predict"), false));
}

double NnModel::loss(const NnTensor& x, const NnTensor& y, const std::string& loss, bool training) {
  NoGradGuard guard;
  return impl_->loss_of(need(x, "loss"), need(y, "loss"), loss, training).item<double>();
}

double NnModel::compute_gradients(const NnTensor& x, const NnTensor& y, const std::string& loss) {
  const Tensor l      = impl_->loss_of(need(x, "compute_gradients"), need(y, "compute_gradients"), loss, true);
  const auto   params = impl_->params.parameters();
  const auto   grads  = grad(l, params, GradOptions{false, false, true});
  impl_->grads.clear();
  const auto& names = impl_->params.names();
  for (std::size_t i = 0; i < names.size(); ++i) impl_->grads[names[i]] = grads[i];
  return l.item<double>();
}

NnTensor NnModel::gradient(const std::string& name) const {
  const auto it = impl_->grads.find(name);
  if (it == impl_->grads.end()) throw std::out_of_range("no gradient for '" + name + "' (call compute_gradients first)");
  return NnTensor(it->second);
}

void NnModel::set_optimizer(const std::string& kind, double learning_rate) {
  parse_optimizer(kind);
  impl_->optimizer_kind = kind;
  impl_->learning_rate  = learning_rate;
  impl_->optimizer.reset();
}

double NnModel::train_step(const NnTensor& x, const NnTensor& y, const std::string& loss) {
  const double value = compute_gradients(x, y, loss);
  if (!impl_->optimizer) {
    OptimizerConfig config;
    config.kind       = parse_optimizer(impl_->optimizer_kind);
    impl_->optimizer  = std::make_unique<Optimizer>(impl_->params, impl_->learning_rate, config);
  }
  std::vector<Tensor> grads;
  for (const auto& name : impl_->params.names()) grads.push_back(impl_->grads[name]);
  impl_->optimizer->step(grads);
  return value;
}

std::vector<std::string> NnModel::parameter_names() const { return impl_->params.names(); }
NnTensor NnModel::parameter(const std::string& name) const { return NnTensor(impl_->params.at(name).detach()); }
void     NnModel::set_parameter(const std::string& name, const NnTensor& value) {
  NoGradGuard guard;
  impl_->params.set(name, cast(need(value, "set_parameter"), impl_->params.dtype()));
}
int NnModel::parameter_count() const { return static_cast<int>(impl_->params.count()); }

void NnModel::save(const std::string& path) const {
  NamedTensors arrays;
  for (const auto& name : impl_->params.names()) arrays.emplace_back(name, impl_->params.at(name).detach());
  for (const auto& name : impl_->stats.names()) arrays.emplace_back("state/" + name, impl_->stats.at(name).detach());
  save_npz(path, arrays);
}

void NnModel::load(const std::string& path) {
  NoGradGuard guard;
  for (const auto& [name, value] : load_npz(path)) {
    if (name.rfind("state/", 0) == 0)
      impl_->stats.set(name.substr(6), cast(value, impl_->stats.dtype()));
    else
      impl_->params.set(name, cast(value, impl_->params.dtype()));
  }
  impl_->optimizer.reset();
}

std::string nn_gemm_backend() { return gemm_backend(); }
int         nn_get_num_threads() { return get_num_threads(); }
void        nn_set_num_threads(int threads) { set_num_threads(threads); }

}  // namespace datamunge::nn
