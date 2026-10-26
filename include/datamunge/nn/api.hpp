#pragma once

/// @file api.hpp
/// @brief A small, binding-friendly surface over `datamunge::nn` for the SWIG
///        language bindings (Python, R, ...).
///
/// It deliberately uses only types every binding already understands: values
/// travel as `std::vector<double>`, shapes and axes as `std::vector<int>`,
/// dtypes and names as strings. C++ code should use the full `nn` API instead.
///
///   NnTensor: create and convert tensors, a core set of differentiable ops,
///             and reverse-mode gradients (`set_requires_grad`, `backward`, `grad`).
///   NnModel:  a sequential model built from layer specs; forward inference,
///             losses, gradients by parameter name, optimizer steps, and .npz
///             save / load (weights trained in Python or C++ load directly).

#include <memory>
#include <string>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

class NnTensor {
 public:
  /// An empty (undefined) tensor.
  NnTensor();
  /// `values` in row-major order filling `shape`; dtype "float32", "float64", "int32", "int64" or "bool".
  NnTensor(const std::vector<double>& values, const std::vector<int>& shape, const std::string& dtype = "float32");

  static NnTensor zeros(const std::vector<int>& shape, const std::string& dtype = "float32");
  static NnTensor ones(const std::vector<int>& shape, const std::string& dtype = "float32");
  static NnTensor full(const std::vector<int>& shape, double value, const std::string& dtype = "float32");
  /// start, start + step, ... below stop.
  static NnTensor arange(double start, double stop, double step = 1.0, const std::string& dtype = "float32");
  /// Seeded N(mean, std^2) and U(low, high) samples.
  static NnTensor randn(const std::vector<int>& shape, unsigned seed, double mean = 0.0, double std = 1.0,
                        const std::string& dtype = "float32");
  static NnTensor uniform(const std::vector<int>& shape, unsigned seed, double low = 0.0, double high = 1.0,
                          const std::string& dtype = "float32");
  /// NumPy .npy files.
  static NnTensor load_npy(const std::string& path);
  void            save_npy(const std::string& path) const;

  // ---- inspection --------------------------------------------------------------------
  bool                defined() const;
  std::vector<double> values() const;
  std::vector<int>    shape() const;
  std::string         dtype() const;
  int                 ndim() const;
  int                 numel() const;
  /// The value of a one-element tensor.
  double              item() const;
  /// The element at a full index.
  double              at(const std::vector<int>& index) const;
  /// A short printable description with the values.
  std::string         str() const;

  // ---- shape and conversion ----------------------------------------------------------
  NnTensor astype(const std::string& dtype) const;
  NnTensor reshape(const std::vector<int>& shape) const;
  /// Permutes axes; an empty `perm` reverses them.
  NnTensor transpose(const std::vector<int>& perm = std::vector<int>()) const;
  /// x[start:stop] along `axis`.
  NnTensor slice(int axis, int start, int stop) const;
  NnTensor concat(const NnTensor& other, int axis) const;

  // ---- arithmetic (broadcasting) -----------------------------------------------------
  NnTensor add(const NnTensor& other) const;
  NnTensor sub(const NnTensor& other) const;
  NnTensor mul(const NnTensor& other) const;
  NnTensor div(const NnTensor& other) const;
  NnTensor add_scalar(double value) const;
  NnTensor mul_scalar(double value) const;
  NnTensor pow(double exponent) const;
  NnTensor neg() const;
  NnTensor matmul(const NnTensor& other, bool transpose_a = false, bool transpose_b = false) const;

  // ---- elementwise functions ---------------------------------------------------------
  NnTensor exp() const;
  NnTensor log() const;
  NnTensor sqrt() const;
  NnTensor abs() const;
  NnTensor square() const;
  NnTensor tanh() const;
  NnTensor sigmoid() const;
  NnTensor relu() const;
  /// Any activation by name: "relu", "gelu", "silu", "softplus", "elu", "leaky_relu", ...
  NnTensor activate(const std::string& activation) const;
  NnTensor softmax(int axis = -1) const;
  NnTensor log_softmax(int axis = -1) const;

  // ---- reductions (empty axes = all) ---------------------------------------------------
  NnTensor sum(const std::vector<int>& axes = std::vector<int>(), bool keepdims = false) const;
  NnTensor mean(const std::vector<int>& axes = std::vector<int>(), bool keepdims = false) const;
  NnTensor max(const std::vector<int>& axes = std::vector<int>(), bool keepdims = false) const;
  NnTensor min(const std::vector<int>& axes = std::vector<int>(), bool keepdims = false) const;
  NnTensor argmax(int axis = -1) const;

  // ---- autograd --------------------------------------------------------------------------
  /// Marks a float tensor as a leaf whose gradient `backward` accumulates.
  void     set_requires_grad(bool requires_grad = true);
  bool     requires_grad() const;
  /// Back-propagates from this one-element tensor into every leaf's grad().
  void     backward();
  /// The accumulated gradient (undefined if none).
  NnTensor grad() const;
  void     zero_grad();
  /// The same values without gradient history.
  NnTensor detach() const;

#ifndef SWIG
  explicit NnTensor(Tensor tensor);
  const Tensor& tensor() const { return tensor_; }
#endif

 private:
  Tensor tensor_;
};

/// A sequential network. Layers are added by spec and own parameters named
/// "<kind><index>/..." (e.g. "dense0/W", "conv1/b"); batch-norm moving
/// statistics live in a second store. Parameters are created on the first
/// forward pass, when the input width is known.
class NnModel {
 public:
  explicit NnModel(unsigned seed = 0);
  ~NnModel();
  NnModel(const NnModel&)            = delete;
  NnModel& operator=(const NnModel&) = delete;

  /// Fully connected layer; activation "none", "relu", "tanh", "sigmoid", "gelu", ...
  void add_dense(int units, const std::string& activation = "none");
  /// 2-D convolution on NHWC input; padding "same" or "valid".
  void add_conv2d(int filters, int kernel, int stride = 1, const std::string& padding = "same",
                  const std::string& activation = "none");
  void add_max_pool2d(int window, int stride, const std::string& padding = "valid");
  void add_avg_pool2d(int window, int stride, const std::string& padding = "valid");
  void add_global_avg_pool2d();
  /// Batch normalization over the last axis (momentum 0.99, eps 1e-3, TF defaults).
  void add_batch_norm(double momentum = 0.99, double eps = 1e-3);
  void add_layer_norm(double eps = 1e-5);
  /// Inverted dropout, active only when training.
  void add_dropout(double rate);
  void add_activation(const std::string& activation);
  /// Collapses every axis but the first.
  void add_flatten();
  /// Reshapes each example (the batch axis is kept): e.g. {28, 28, 1}.
  void add_reshape(const std::vector<int>& shape);

  int         num_layers() const;
  /// One line per layer with its parameter shapes (after the first forward pass).
  std::string summary() const;

  /// Runs the network. `training` enables dropout and batch statistics (and
  /// updates batch-norm moving averages).
  NnTensor forward(const NnTensor& x, bool training = false);
  /// forward(x, false) without recording gradients.
  NnTensor predict(const NnTensor& x);

  /// Loss of the network on (x, y): "mse", "softmax_cross_entropy" (y holds
  /// integer class labels) or "sigmoid_cross_entropy" (y in [0, 1]).
  double loss(const NnTensor& x, const NnTensor& y, const std::string& loss = "mse", bool training = false);
  /// Computes the loss in training mode and stores the gradient of every
  /// parameter (read them with `gradient`). Returns the loss.
  double compute_gradients(const NnTensor& x, const NnTensor& y, const std::string& loss = "mse");
  NnTensor gradient(const std::string& name) const;

  /// Optimizer for `train_step`: "sgd", "momentum", "adam", "adamw", "rmsprop", "adagrad", ...
  void   set_optimizer(const std::string& kind = "adam", double learning_rate = 1e-3);
  /// compute_gradients followed by one optimizer update. Returns the loss.
  double train_step(const NnTensor& x, const NnTensor& y, const std::string& loss = "mse");

  std::vector<std::string> parameter_names() const;
  NnTensor                 parameter(const std::string& name) const;
  void                     set_parameter(const std::string& name, const NnTensor& value);
  int                      parameter_count() const;

  /// Parameters and batch-norm statistics as one .npz archive ("state/..." for statistics).
  void save(const std::string& path) const;
  /// Loads an archive written by save() (or any .npz with the same names).
  void load(const std::string& path);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// The GEMM backend nn uses ("accelerate", "openblas", "cblas" or "builtin").
std::string nn_gemm_backend();
/// Threads used by nn kernels.
int  nn_get_num_threads();
void nn_set_num_threads(int threads);

}  // namespace datamunge::nn
