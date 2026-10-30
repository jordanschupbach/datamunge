#pragma once

/// @file params.hpp
/// @brief Named parameters (TF Variables) and weight initializers.
///
/// A ParameterStore holds the learnable tensors of a model by name.
/// `get(name, shape, init)` creates the parameter on first use and returns the
/// same tensor afterwards, so calling a layer twice with the same name shares
/// its weights (tflow's `nn::Params`). Parameters are float leaves with
/// `requires_grad`; optimizers update them in place.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "datamunge/nn/random.hpp"
#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

/// Draws an initial value of the given shape and dtype.
using Initializer = std::function<Tensor(const Shape& shape, DType dtype, Generator& gen)>;

/// @brief Factories for shape- and dtype-aware parameter initialization callbacks.
namespace init {

/// Fan-in and fan-out of a weight shape: [in, out] for dense weights and
/// [kh, kw, in, out] (receptive field times channels) for convolutions, as in TF.
/// The convention is positional, as in Keras: fan_in always uses shape[-2]
/// and fan_out shape[-1]. For a transposed-convolution kernel [kh, kw, out, in]
/// (conv2d_transpose's layout) that makes fan_in = kh kw out_channels and
/// fan_out = kh kw in_channels, so he_* / glorot_* / lecun_* scale such
/// kernels by the output channels, exactly as Keras Conv2DTranspose does.
std::pair<double, double> fans(const Shape& shape);

/// @brief Returns an initializer that fills every element with zero without drawing randomness.
Initializer zeros();
/// @brief Returns an initializer that fills every element with one without drawing randomness.
Initializer ones();
/// @brief Returns an initializer that fills with value converted to the requested dtype.
Initializer constant(double value);
/// U(low, high)
Initializer uniform(double low, double high);
/// U(-limit, limit)
Initializer uniform(double limit);
/// @brief Returns a normal initializer with the supplied standard deviation and mean.
Initializer normal(double stddev, double mean = 0.0);
/// Normal redrawn beyond two standard deviations (TF TruncatedNormal).
Initializer truncated_normal(double stddev, double mean = 0.0);
/// U(-a, a), a = sqrt(6 / (fan_in + fan_out))
Initializer glorot_uniform();
/// N(0, 2 / (fan_in + fan_out))
Initializer glorot_normal();
/// U(-a, a), a = sqrt(6 / fan_in)
Initializer he_uniform();
/// N(0, 2 / fan_in)
Initializer he_normal();
/// U(-a, a), a = sqrt(3 / fan_in)
Initializer lecun_uniform();
/// N(0, 1 / fan_in)
Initializer lecun_normal();
/// A (semi-)orthogonal matrix from the QR decomposition of a Gaussian one,
/// over the last axis vs. the rest (Saxe et al. 2014), scaled by `gain`.
Initializer orthogonal(double gain = 1.0);

}  // namespace init

/// Named learnable tensors in creation order.
class ParameterStore {
 public:
  /// @brief Creates an empty store with its own seed-derived initializer stream and floating dtype.
  /// @throws std::invalid_argument If dtype is not Float32 or Float64.
  explicit ParameterStore(std::uint64_t seed = 0, DType dtype = DType::Float32);

  // Stores alias their tensors, so copying would share parameters
  // silently. Use clone() for an independent copy.
  /// @brief Disallowed to prevent an accidental shallow copy of learnable tensors; use clone().
  ParameterStore(const ParameterStore&)            = delete;
  /// @brief Disallowed to prevent silently sharing parameters during copy assignment.
  ParameterStore& operator=(const ParameterStore&) = delete;
  /// @brief Transfers parameter names, tensor ownership, and initializer stream state.
  ParameterStore(ParameterStore&&)                 = default;
  /// @brief Replaces this store by moving parameter handles and initializer state from another.
  ParameterStore& operator=(ParameterStore&&)      = default;

  /// The parameter `name`, created from `init` on first use. Later calls
  /// must ask for the same shape.
  Tensor get(const std::string& name, const Shape& shape, const Initializer& init);
  /// An existing parameter (throws if missing).
  [[nodiscard]] Tensor at(const std::string& name) const;
  /// @brief Returns whether a parameter with this exact case-sensitive name already exists.
  [[nodiscard]] bool   contains(const std::string& name) const;
  /// Adds `value` as a new parameter, or overwrites an existing one in place
  /// (shapes must match).
  void set(const std::string& name, const Tensor& value);

  /// @brief Borrows parameter names in creation order; later insertion may invalidate references.
  [[nodiscard]] const std::vector<std::string>& names() const noexcept { return names_; }
  /// @brief Returns a new vector of handles aliasing the store's learnable tensors, in creation order.
  [[nodiscard]] std::vector<Tensor>             parameters() const;
  /// @brief Returns the number of named tensors, rather than their total scalar element count.
  [[nodiscard]] std::size_t                     size() const noexcept { return names_.size(); }
  /// Total number of scalar parameters.
  [[nodiscard]] std::int64_t                    count() const;
  /// @brief Returns the floating dtype to which all stored parameters are converted.
  [[nodiscard]] DType                           dtype() const noexcept { return dtype_; }
  /// Random stream used by initializers (and handy for dropout masks).
  Generator&                                    generator() noexcept { return gen_; }

  /// this <- tau * src + (1 - tau) * this for every parameter of `src`
  /// (matched by name; all must exist here with the same shape). tau = 1 is a
  /// hard copy (target-network sync), tau < 1 Polyak averaging.
  void copy_from(const ParameterStore& src, double tau = 1.0);
  /// Deep copy with the same names, values and seed state.
  [[nodiscard]] ParameterStore clone() const;

  /// All parameters concatenated in creation order (for black-box optimizers
  /// such as CMA-ES or CEM in `datamunge::optim`).
  [[nodiscard]] std::vector<double> flatten() const;
  /// Inverse of flatten().
  void unflatten(const std::vector<double>& values);

  /// Drops the accumulated gradients of every parameter.
  void zero_grad();

  /// Writes every parameter to an .npz archive (one array per name).
  void save(const std::string& path) const;
  /// Reads an .npz archive written by save(). With `strict`, the archive must
  /// contain exactly the parameters of this store; otherwise missing
  /// parameters are created and extra arrays are added.
  void load(const std::string& path, bool strict = true);

 private:
  /// @brief Owned parameter keys, preserving creation order and aligned with values_.
  std::vector<std::string>              names_;
  /// @brief Open-addressed buckets: zero is empty; other entries are indices into names_/values_ plus one.
  std::vector<std::size_t>              index_;
  /// @brief Returns the matching bucket's parameter index, or values_.size() when absent.
  std::size_t find_index(const std::string& name) const;
  /// @brief Learnable leaf tensors owned in creation order, also shared with returned handles.
  std::vector<Tensor>                   values_;
  /// @brief Mutable initializer/dropout stream belonging to this store.
  Generator                             gen_;
  /// @brief Common Float32 or Float64 representation of learnable parameters.
  DType                                 dtype_;
};

}  // namespace datamunge::nn
