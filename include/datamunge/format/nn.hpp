#pragma once

/// @file nn.hpp
/// @brief std::format / std::print support for datamunge::nn.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// Tensors print through Tensor::str() (dtype, shape and a capped, nested
/// value preview). Options and result structs print one line,
/// `TypeName(field=value, ...)`, with tensor fields shown as `dtype[shape]`.
/// Containers (ParameterStore, NnModel) list their entries one per line.
/// Random generators keep their state private and print just their name.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::nn::tensor({{1, 2, 3}, {4, 5, 6.5}}));
/// // float32[2, 3] [[1, 2, 3], [4, 5, 6.5]]
/// @endcode

#include <datamunge/format/linalg.hpp>
#include <datamunge/format/text_formatter.hpp>
#include <datamunge/nn/api.hpp>
#include <datamunge/nn/mt19937.hpp>
#include <datamunge/nn/nn.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <string>
#include <vector>

/// @brief Core library APIs and their algorithm, data, and numerical submodules.
namespace datamunge::format_detail {

/// @brief Helpers for the nn formatters.
namespace nn_fmt {

/// @brief Short tensor description for struct fields: "float32[2, 3]", or "undefined".
/// @param t The tensor.
/// @return Dtype and shape, without values.
inline std::string tshape(const nn::Tensor& t) {
  if (!t.defined()) return "undefined";
  return nn::dtype_name(t.dtype()) + nn::shape_str(t.shape());
}

/// @brief Short complex-tensor description: "complex(float32[4])", or "undefined".
/// @param t The complex tensor.
/// @return Dtype and shape of its parts.
inline std::string cshape(const nn::ComplexTensor& t) {
  if (!t.real.defined()) return "undefined";
  return "complex(" + tshape(t.real) + ")";
}

/// @brief One line per (name, tensor) pair, e.g. "  dense/W: float32[4, 2]", capped at `max_lines`.
/// @param names Entry names.
/// @param tensors Matching tensors.
/// @param max_lines Entries listed before eliding.
/// @return The lines, each prefixed by a newline.
inline std::string tensor_lines(const std::vector<std::string>& names, const std::vector<nn::Tensor>& tensors,
                                std::size_t max_lines = 20) {
  std::string out;
  for (std::size_t i = 0; i < names.size() && i < tensors.size(); ++i) {
    if (i == max_lines) {
      out += std::format("\n  ... ({} more)", names.size() - max_lines);
      break;
    }
    out += std::format("\n  {}: {}", names[i], tshape(tensors[i]));
  }
  return out;
}

/// @brief OptimizerConfig fields that the configured rule actually uses, as "k=v, ..." (may be empty).
/// @param c The configuration.
/// @return Comma-separated fields.
inline std::string optimizer_fields(const nn::OptimizerConfig& c) {
  using K = nn::OptimizerKind;
  switch (c.kind) {
    case K::Sgd: return {};
    case K::Momentum:
    case K::Nesterov: return std::format("momentum={}", c.momentum);
    case K::Adagrad: return std::format("initial_accumulator={}, epsilon={}", c.initial_accumulator, c.epsilon);
    case K::RMSProp:
    case K::Adadelta: return std::format("rho={}, epsilon={}", c.rho, c.epsilon);
    case K::Adam:
    case K::Nadam:
    case K::AMSGrad:
    case K::Adamax: return std::format("beta1={}, beta2={}, epsilon={}", c.beta1, c.beta2, c.epsilon);
    case K::AdamW:
    case K::Lamb:
      return std::format("beta1={}, beta2={}, epsilon={}, weight_decay={}", c.beta1, c.beta2, c.epsilon, c.weight_decay);
    case K::Lars:
      return std::format("momentum={}, weight_decay={}, trust_coefficient={}", c.momentum, c.weight_decay,
                         c.trust_coefficient);
    case K::Lion: return std::format("beta1={}, beta2={}, weight_decay={}", c.beta1, c.beta2, c.weight_decay);
    case K::Adafactor:
      return std::format(
          "beta1={}, factor_epsilon={}, parameter_epsilon={}, update_clip={}, decay_exponent={}, relative_step={}, "
          "scale_parameter={}",
          c.beta1, c.factor_epsilon, c.parameter_epsilon, c.update_clip, c.decay_exponent, c.relative_step,
          c.scale_parameter);
  }
  return {};
}

/// @brief The library's optimizer display name, or "OptimizerKind(<int>)" for an unknown value.
/// @param kind The update rule.
/// @return E.g. "AdamW".
inline std::string optimizer_kind(nn::OptimizerKind kind) {
  try {
    return nn::optimizer_name(kind);
  } catch (const std::exception&) {
    return std::format("OptimizerKind({})", static_cast<int>(kind));
  }
}

/// @brief Description of an autograd graph node: name, input/output metadata and next nodes.
/// @param node The node.
/// @return E.g. "Node(Mul, inputs=[float32[2]], outputs=[float32[2]], next=[AccumulateGrad])".
inline std::string node(const nn::Node& node) {
  auto metas = [](const std::vector<nn::TensorMeta>& ms) {
    std::vector<std::string> parts;
    for (const auto& m : ms) parts.push_back(nn::dtype_name(m.dtype) + nn::shape_str(m.shape));
    return preview(parts);
  };
  std::vector<std::string> next;
  for (const auto& e : node.next_edges()) next.push_back(e.node ? e.node->name() : "None");
  return std::format("Node({}, inputs={}, outputs={}, next={}{})", node.name(), metas(node.input_meta()),
                     metas(node.output_meta()), preview(next), node.released() ? ", released" : "");
}

}  // namespace nn_fmt
}  // namespace datamunge::format_detail

// ---- enums -------------------------------------------------------------------

/// @brief Prints a DType as the library's lowercase name ("float32", "int64", ...).
template <>
struct std::formatter<datamunge::nn::DType> : datamunge::format_detail::text_formatter<datamunge::nn::DType> {
  /// @brief The dtype name from nn::dtype_name().
  /// @param d The dtype.
  /// @return E.g. "float32"; "DType(<int>)" for an unknown value.
  static std::string text(datamunge::nn::DType d) {
    const std::string name = datamunge::nn::dtype_name(d);
    return name == "unknown" ? std::format("DType({})", static_cast<int>(d)) : name;
  }
};

/// @brief Prints an Activation as the library's name ("relu", "gelu_tanh", "identity" for None).
template <>
struct std::formatter<datamunge::nn::Activation> : datamunge::format_detail::text_formatter<datamunge::nn::Activation> {
  /// @brief The activation name from nn::activation_name().
  /// @param a The activation.
  /// @return E.g. "relu"; "Activation(<int>)" for an unknown value.
  static std::string text(datamunge::nn::Activation a) {
    try {
      return datamunge::nn::activation_name(a);
    } catch (const std::exception&) {
      return std::format("Activation({})", static_cast<int>(a));
    }
  }
};

/// @brief Prints an OptimizerKind as the library's display name ("Adam", "AdamW", ...).
template <>
struct std::formatter<datamunge::nn::OptimizerKind>
    : datamunge::format_detail::text_formatter<datamunge::nn::OptimizerKind> {
  /// @brief The rule name from nn::optimizer_name().
  /// @param k The optimizer kind.
  /// @return E.g. "AdamW"; "OptimizerKind(<int>)" for an unknown value.
  static std::string text(datamunge::nn::OptimizerKind k) { return datamunge::format_detail::nn_fmt::optimizer_kind(k); }
};

/// @brief Prints an AttentionImpl as its enumerator name (Auto, Plain, Fused).
template <>
struct std::formatter<datamunge::nn::AttentionImpl> : datamunge::format_detail::text_formatter<datamunge::nn::AttentionImpl> {
  /// @brief The enumerator name.
  /// @param value The value.
  /// @return E.g. "Auto"; "AttentionImpl(<int>)" for an unknown value.
  static std::string text(datamunge::nn::AttentionImpl value) {
    using E = datamunge::nn::AttentionImpl;
    switch (value) {
      case E::Auto: return "Auto";
      case E::Plain: return "Plain";
      case E::Fused: return "Fused";
    }
    return std::format("AttentionImpl({})", static_cast<int>(value));
  }
};

/// @brief Prints a RotaryLayout as its enumerator name (Interleaved, HalfSplit).
template <>
struct std::formatter<datamunge::nn::RotaryLayout> : datamunge::format_detail::text_formatter<datamunge::nn::RotaryLayout> {
  /// @brief The enumerator name.
  /// @param value The value.
  /// @return E.g. "Interleaved"; "RotaryLayout(<int>)" for an unknown value.
  static std::string text(datamunge::nn::RotaryLayout value) {
    using E = datamunge::nn::RotaryLayout;
    switch (value) {
      case E::Interleaved: return "Interleaved";
      case E::HalfSplit: return "HalfSplit";
    }
    return std::format("RotaryLayout({})", static_cast<int>(value));
  }
};

/// @brief Prints a WavFormat as its enumerator name (Pcm16, Pcm24, Pcm32, Pcm8, Float32).
template <>
struct std::formatter<datamunge::nn::WavFormat> : datamunge::format_detail::text_formatter<datamunge::nn::WavFormat> {
  /// @brief The enumerator name.
  /// @param value The value.
  /// @return E.g. "Pcm16"; "WavFormat(<int>)" for an unknown value.
  static std::string text(datamunge::nn::WavFormat value) {
    using E = datamunge::nn::WavFormat;
    switch (value) {
      case E::Pcm16: return "Pcm16";
      case E::Pcm24: return "Pcm24";
      case E::Pcm32: return "Pcm32";
      case E::Pcm8: return "Pcm8";
      case E::Float32: return "Float32";
    }
    return std::format("WavFormat({})", static_cast<int>(value));
  }
};

/// @brief Prints a Padding as its enumerator name (Same, Valid, Explicit).
template <>
struct std::formatter<datamunge::nn::Padding> : datamunge::format_detail::text_formatter<datamunge::nn::Padding> {
  /// @brief The enumerator name.
  /// @param value The value.
  /// @return E.g. "Same"; "Padding(<int>)" for an unknown value.
  static std::string text(datamunge::nn::Padding value) {
    using E = datamunge::nn::Padding;
    switch (value) {
      case E::Same: return "Same";
      case E::Valid: return "Valid";
      case E::Explicit: return "Explicit";
    }
    return std::format("Padding({})", static_cast<int>(value));
  }
};

/// @brief Prints a ResizeMethod as its enumerator name (Bilinear, Nearest).
template <>
struct std::formatter<datamunge::nn::ResizeMethod> : datamunge::format_detail::text_formatter<datamunge::nn::ResizeMethod> {
  /// @brief The enumerator name.
  /// @param value The value.
  /// @return E.g. "Bilinear"; "ResizeMethod(<int>)" for an unknown value.
  static std::string text(datamunge::nn::ResizeMethod value) {
    using E = datamunge::nn::ResizeMethod;
    switch (value) {
      case E::Bilinear: return "Bilinear";
      case E::Nearest: return "Nearest";
    }
    return std::format("ResizeMethod({})", static_cast<int>(value));
  }
};

/// @brief Prints a DepthToSpaceMode as its enumerator name (DCR, CRD).
template <>
struct std::formatter<datamunge::nn::DepthToSpaceMode> : datamunge::format_detail::text_formatter<datamunge::nn::DepthToSpaceMode> {
  /// @brief The enumerator name.
  /// @param value The value.
  /// @return E.g. "DCR"; "DepthToSpaceMode(<int>)" for an unknown value.
  static std::string text(datamunge::nn::DepthToSpaceMode value) {
    using E = datamunge::nn::DepthToSpaceMode;
    switch (value) {
      case E::DCR: return "DCR";
      case E::CRD: return "CRD";
    }
    return std::format("DepthToSpaceMode({})", static_cast<int>(value));
  }
};

/// @brief Prints a GridSampleMode as its enumerator name (Bilinear, Nearest).
template <>
struct std::formatter<datamunge::nn::GridSampleMode> : datamunge::format_detail::text_formatter<datamunge::nn::GridSampleMode> {
  /// @brief The enumerator name.
  /// @param value The value.
  /// @return E.g. "Bilinear"; "GridSampleMode(<int>)" for an unknown value.
  static std::string text(datamunge::nn::GridSampleMode value) {
    using E = datamunge::nn::GridSampleMode;
    switch (value) {
      case E::Bilinear: return "Bilinear";
      case E::Nearest: return "Nearest";
    }
    return std::format("GridSampleMode({})", static_cast<int>(value));
  }
};

/// @brief Prints a GridSamplePadding as its enumerator name (Zeros, Border).
template <>
struct std::formatter<datamunge::nn::GridSamplePadding> : datamunge::format_detail::text_formatter<datamunge::nn::GridSamplePadding> {
  /// @brief The enumerator name.
  /// @param value The value.
  /// @return E.g. "Zeros"; "GridSamplePadding(<int>)" for an unknown value.
  static std::string text(datamunge::nn::GridSamplePadding value) {
    using E = datamunge::nn::GridSamplePadding;
    switch (value) {
      case E::Zeros: return "Zeros";
      case E::Border: return "Border";
    }
    return std::format("GridSamplePadding({})", static_cast<int>(value));
  }
};

/// @brief Prints a Reduction as its enumerator name (None, Mean, Sum).
template <>
struct std::formatter<datamunge::nn::Reduction> : datamunge::format_detail::text_formatter<datamunge::nn::Reduction> {
  /// @brief The enumerator name.
  /// @param value The value.
  /// @return E.g. "None"; "Reduction(<int>)" for an unknown value.
  static std::string text(datamunge::nn::Reduction value) {
    using E = datamunge::nn::Reduction;
    switch (value) {
      case E::None: return "None";
      case E::Mean: return "Mean";
      case E::Sum: return "Sum";
    }
    return std::format("Reduction({})", static_cast<int>(value));
  }
};

/// @brief Prints a LinalgBackend as its enumerator name (Auto, Builtin, Lapack).
template <>
struct std::formatter<datamunge::nn::LinalgBackend> : datamunge::format_detail::text_formatter<datamunge::nn::LinalgBackend> {
  /// @brief The enumerator name.
  /// @param value The value.
  /// @return E.g. "Auto"; "LinalgBackend(<int>)" for an unknown value.
  static std::string text(datamunge::nn::LinalgBackend value) {
    using E = datamunge::nn::LinalgBackend;
    switch (value) {
      case E::Auto: return "Auto";
      case E::Builtin: return "Builtin";
      case E::Lapack: return "Lapack";
    }
    return std::format("LinalgBackend({})", static_cast<int>(value));
  }
};

/// @brief Prints a PadMode as its enumerator name (Constant, Reflect, Symmetric).
template <>
struct std::formatter<datamunge::nn::PadMode> : datamunge::format_detail::text_formatter<datamunge::nn::PadMode> {
  /// @brief The enumerator name.
  /// @param value The value.
  /// @return E.g. "Constant"; "PadMode(<int>)" for an unknown value.
  static std::string text(datamunge::nn::PadMode value) {
    using E = datamunge::nn::PadMode;
    switch (value) {
      case E::Constant: return "Constant";
      case E::Reflect: return "Reflect";
      case E::Symmetric: return "Symmetric";
    }
    return std::format("PadMode({})", static_cast<int>(value));
  }
};

/// @brief Prints a RnnKind as its enumerator name (Simple, Lstm, Gru).
template <>
struct std::formatter<datamunge::nn::RnnKind> : datamunge::format_detail::text_formatter<datamunge::nn::RnnKind> {
  /// @brief The enumerator name.
  /// @param value The value.
  /// @return E.g. "Simple"; "RnnKind(<int>)" for an unknown value.
  static std::string text(datamunge::nn::RnnKind value) {
    using E = datamunge::nn::RnnKind;
    switch (value) {
      case E::Simple: return "Simple";
      case E::Lstm: return "Lstm";
      case E::Gru: return "Gru";
    }
    return std::format("RnnKind({})", static_cast<int>(value));
  }
};

/// @brief Prints an nn::Tensor through Tensor::str(): dtype, shape and a capped nested value preview ("Tensor(undefined)" if empty).
template <>
struct std::formatter<datamunge::nn::Tensor> : datamunge::format_detail::text_formatter<datamunge::nn::Tensor> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return E.g. "float32[2, 2] [[1, 2], [3, 4]]", with " (requires_grad)" when tracked.
  static std::string text(const datamunge::nn::Tensor& x) {
    return x.str();
  }
};

/// @brief Prints an NnTensor (the bindings-facing tensor) through NnTensor::str().
template <>
struct std::formatter<datamunge::nn::NnTensor> : datamunge::format_detail::text_formatter<datamunge::nn::NnTensor> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return E.g. "NnTensor(shape=[2, 2], dtype=float32, values=[1, 2, 3, 4])".
  static std::string text(const datamunge::nn::NnTensor& x) {
    return x.str();
  }
};

/// @brief Prints an NnModel as a header line plus its summary() (one layer per line, then the parameter count).
template <>
struct std::formatter<datamunge::nn::NnModel> : datamunge::format_detail::text_formatter<datamunge::nn::NnModel> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return "NnModel(layers=n)" followed by the summary lines, indented.
  static std::string text(const datamunge::nn::NnModel& x) {
    std::string out = std::format("NnModel(layers={})", x.num_layers());
    std::string summary = x.summary();
    while (!summary.empty() && summary.back() == '\n') summary.pop_back();
    std::size_t start = 0;
    while (start <= summary.size() && !summary.empty()) {
      const auto end = summary.find('\n', start);
      out += "\n  " + summary.substr(start, end == std::string::npos ? std::string::npos : end - start);
      if (end == std::string::npos) break;
      start = end + 1;
    }
    return out;
  }
};

/// @brief Prints a ParameterStore PyTorch-style: a header, then one "name: dtype[shape]" line per parameter.
template <>
struct std::formatter<datamunge::nn::ParameterStore> : datamunge::format_detail::text_formatter<datamunge::nn::ParameterStore> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return Header "ParameterStore(params=n, count=c, dtype=d)" then up to 20 parameter lines.
  static std::string text(const datamunge::nn::ParameterStore& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("ParameterStore(params={}, count={}, dtype={})", x.size(), x.count(),
                       datamunge::nn::dtype_name(x.dtype())) +
           F::tensor_lines(x.names(), x.parameters());
  }
};

/// @brief Prints decoded audio as "Wav(samples=float32[frames, channels], sample_rate=r)".
template <>
struct std::formatter<datamunge::nn::Wav> : datamunge::format_detail::text_formatter<datamunge::nn::Wav> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::Wav& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("Wav(samples={}, sample_rate={})", F::tshape(x.samples), x.sample_rate);
  }
};

/// @brief Prints an augmentation rectangle as "Rect(top=, left=, height=, width=)".
template <>
struct std::formatter<datamunge::nn::augment::Rect> : datamunge::format_detail::text_formatter<datamunge::nn::augment::Rect> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::augment::Rect& x) {
    return std::format("Rect(top={}, left={}, height={}, width={})", x.top, x.left, x.height, x.width);
  }
};

/// @brief Prints a mixup/CutMix batch as "MixedBatch(images=.., labels=.., weight=w)".
template <>
struct std::formatter<datamunge::nn::augment::MixedBatch> : datamunge::format_detail::text_formatter<datamunge::nn::augment::MixedBatch> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::augment::MixedBatch& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("MixedBatch(images={}, labels={}, weight={})", F::tshape(x.images), F::tshape(x.labels), x.weight);
  }
};

/// @brief Prints an autograd edge as "Edge(node=Name, input_nr=k)" (node None for a non-differentiable input).
template <>
struct std::formatter<datamunge::nn::Edge> : datamunge::format_detail::text_formatter<datamunge::nn::Edge> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::Edge& x) {
    return std::format("Edge(node={}, input_nr={})", x.node ? x.node->name() : std::string("None"), x.input_nr);
  }
};

/// @brief Prints tensor metadata as "TensorMeta(float32[2, 3])".
template <>
struct std::formatter<datamunge::nn::TensorMeta> : datamunge::format_detail::text_formatter<datamunge::nn::TensorMeta> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::TensorMeta& x) {
    return std::format("TensorMeta({}{})", datamunge::nn::dtype_name(x.dtype), datamunge::nn::shape_str(x.shape));
  }
};

/// @brief Prints an autograd graph node: its name, input/output metadata and next nodes.
template <>
struct std::formatter<datamunge::nn::Node> : datamunge::format_detail::text_formatter<datamunge::nn::Node> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::Node& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return F::node(x);
  }
};

/// @brief Prints a leaf-gradient accumulator node like any autograd Node.
template <>
struct std::formatter<datamunge::nn::AccumulateGrad> : datamunge::format_detail::text_formatter<datamunge::nn::AccumulateGrad> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::AccumulateGrad& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return F::node(x);
  }
};

/// @brief Prints a saved tensor as "SavedTensor(float32[..])", or "SavedTensor(stale)" if it was modified in place.
template <>
struct std::formatter<datamunge::nn::SavedTensor> : datamunge::format_detail::text_formatter<datamunge::nn::SavedTensor> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::SavedTensor& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    try {
      return std::format("SavedTensor({})", F::tshape(x.get()));
    } catch (const std::exception&) {
      return "SavedTensor(stale)";
    }
  }
};

/// @brief Prints reverse-mode options as "GradOptions(retain_graph=, create_graph=, allow_unused=)".
template <>
struct std::formatter<datamunge::nn::GradOptions> : datamunge::format_detail::text_formatter<datamunge::nn::GradOptions> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::GradOptions& x) {
    return std::format("GradOptions(retain_graph={}, create_graph={}, allow_unused={})", x.retain_graph, x.create_graph, x.allow_unused);
  }
};

/// @brief Prints a custom differentiable function as "Function(name)".
template <>
struct std::formatter<datamunge::nn::Function> : datamunge::format_detail::text_formatter<datamunge::nn::Function> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::Function& x) {
    return std::format("Function({})", x.name());
  }
};

/// @brief Prints checkpointing options as "CheckpointOptions(preserve_rng_state=, generators=n)".
template <>
struct std::formatter<datamunge::nn::CheckpointOptions> : datamunge::format_detail::text_formatter<datamunge::nn::CheckpointOptions> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::CheckpointOptions& x) {
    return std::format("CheckpointOptions(preserve_rng_state={}, generators={})", x.preserve_rng_state, x.generators.size());
  }
};

/// @brief Prints a reversible block as "ReversibleBlock(f=set|empty, g=set|empty)".
template <>
struct std::formatter<datamunge::nn::ReversibleBlock> : datamunge::format_detail::text_formatter<datamunge::nn::ReversibleBlock> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::ReversibleBlock& x) {
    return std::format("ReversibleBlock(f={}, g={})", x.f ? "set" : "empty", x.g ? "set" : "empty");
  }
};

/// @brief Prints attention options as "AttentionOptions(causal=, scale=, dropout=, need_weights=, impl=, mask=..)".
template <>
struct std::formatter<datamunge::nn::AttentionOptions> : datamunge::format_detail::text_formatter<datamunge::nn::AttentionOptions> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::AttentionOptions& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("AttentionOptions(causal={}, scale={}, dropout={}, need_weights={}, impl={}, mask={})", x.causal, x.scale, x.dropout, x.need_weights, x.impl, F::tshape(x.mask));
  }
};

/// @brief Prints an attention result as "Attended(output=.., weights=..)".
template <>
struct std::formatter<datamunge::nn::Attended> : datamunge::format_detail::text_formatter<datamunge::nn::Attended> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::Attended& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("Attended(output={}, weights={})", F::tshape(x.output), F::tshape(x.weights));
  }
};

/// @brief Prints a KV cache as "KvCache(length=n, keys=..)".
template <>
struct std::formatter<datamunge::nn::KvCache> : datamunge::format_detail::text_formatter<datamunge::nn::KvCache> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::KvCache& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("KvCache(length={}, keys={}, values={})", x.length(), F::tshape(x.keys()), F::tshape(x.values()));
  }
};

/// @brief Prints multi-head attention options (the initializer and generator are omitted).
template <>
struct std::formatter<datamunge::nn::MultiHeadOptions> : datamunge::format_detail::text_formatter<datamunge::nn::MultiHeadOptions> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::MultiHeadOptions& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format(
        "MultiHeadOptions(causal={}, kv_heads={}, bias={}, dropout={}, training={}, need_weights={}, impl={}, rotary={}, "
        "rotary_base={}, rotary_layout={}, cache={}, mask={})",
        x.causal, x.kv_heads, x.bias, x.dropout, x.training, x.need_weights, x.impl, x.rotary, x.rotary_base,
        x.rotary_layout, x.cache ? "set" : "none", F::tshape(x.mask));
  }
};

/// @brief Prints an image dataset as "ImageDataset(images=float32[N, H, W, C], labels=int64[N])".
template <>
struct std::formatter<datamunge::nn::ImageDataset> : datamunge::format_detail::text_formatter<datamunge::nn::ImageDataset> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::ImageDataset& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("ImageDataset(images={}, labels={})", F::tshape(x.images), F::tshape(x.labels));
  }
};

/// @brief Prints a labelled image file as "ImageFile(path, label=k)".
template <>
struct std::formatter<datamunge::nn::ImageFile> : datamunge::format_detail::text_formatter<datamunge::nn::ImageFile> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::ImageFile& x) {
    return std::format("ImageFile({}, label={})", x.path, x.label);
  }
};

/// @brief Prints an image-folder scan as "ImageFolder(classes=[...], images=n)".
template <>
struct std::formatter<datamunge::nn::ImageFolder> : datamunge::format_detail::text_formatter<datamunge::nn::ImageFolder> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::ImageFolder& x) {
    return std::format("ImageFolder(classes={}, images={})", datamunge::format_detail::preview(x.class_names), x.images.size());
  }
};

/// @brief Prints a spatial size as "ImageSize(height, width)".
template <>
struct std::formatter<datamunge::nn::ImageSize> : datamunge::format_detail::text_formatter<datamunge::nn::ImageSize> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::ImageSize& x) {
    return std::format("ImageSize({}, {})", x.height, x.width);
  }
};

/// @brief Prints grid-sample options as "GridSampleOptions(mode=, padding=, align_corners=)".
template <>
struct std::formatter<datamunge::nn::GridSampleOptions> : datamunge::format_detail::text_formatter<datamunge::nn::GridSampleOptions> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::GridSampleOptions& x) {
    return std::format("GridSampleOptions(mode={}, padding={}, align_corners={})", x.mode, x.padding, x.align_corners);
  }
};

/// @brief Prints deformable-convolution geometry as "DeformableOptions(kernel=hxw, stride=, pad=hxw, dilation=)".
template <>
struct std::formatter<datamunge::nn::DeformableOptions> : datamunge::format_detail::text_formatter<datamunge::nn::DeformableOptions> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::DeformableOptions& x) {
    return std::format("DeformableOptions(kernel={}x{}, stride={}, pad={}x{}, dilation={})", x.kernel_h, x.kernel_w, x.stride, x.pad_h, x.pad_w, x.dilation);
  }
};

/// @brief Prints gradient-check tolerances as "GradcheckOptions(eps=, atol=, rtol=)".
template <>
struct std::formatter<datamunge::nn::GradcheckOptions> : datamunge::format_detail::text_formatter<datamunge::nn::GradcheckOptions> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::GradcheckOptions& x) {
    return std::format("GradcheckOptions(eps={}, atol={}, rtol={})", x.eps, x.atol, x.rtol);
  }
};

/// @brief Prints a gradient-check outcome as "GradcheckResult(ok=, max_abs_error=[, message=..])".
template <>
struct std::formatter<datamunge::nn::GradcheckResult> : datamunge::format_detail::text_formatter<datamunge::nn::GradcheckResult> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::GradcheckResult& x) {
    std::string out = std::format("GradcheckResult(ok={}, max_abs_error={}", x.ok, x.max_abs_error);
    if (!x.message.empty()) out += std::format(", message={:?}", x.message);
    return out + ")";
  }
};

/// @brief Prints 2-D convolution options as "Conv2DOptions(strides=[..], dilations=[..], padding=, groups=)".
template <>
struct std::formatter<datamunge::nn::Conv2DOptions> : datamunge::format_detail::text_formatter<datamunge::nn::Conv2DOptions> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line (explicit paddings only for Padding::Explicit).
  static std::string text(const datamunge::nn::Conv2DOptions& x) {
    std::string out = std::format("Conv2DOptions(strides={}, dilations={}, padding={}", x.strides, x.dilations, x.padding);
    if (x.padding == datamunge::nn::Padding::Explicit) out += std::format(", explicit_paddings={}", x.explicit_paddings);
    return out + std::format(", groups={})", x.groups);
  }
};

/// @brief Prints 3-D convolution options as "Conv3DOptions(strides=[..], dilations=[..], padding=, groups=)".
template <>
struct std::formatter<datamunge::nn::Conv3DOptions> : datamunge::format_detail::text_formatter<datamunge::nn::Conv3DOptions> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line (explicit paddings only for Padding::Explicit).
  static std::string text(const datamunge::nn::Conv3DOptions& x) {
    std::string out = std::format("Conv3DOptions(strides={}, dilations={}, padding={}", x.strides, x.dilations, x.padding);
    if (x.padding == datamunge::nn::Padding::Explicit) out += std::format(", explicit_paddings={}", x.explicit_paddings);
    return out + std::format(", groups={})", x.groups);
  }
};

/// @brief Prints a pooling-with-argmax result as "PoolWithArgmax(output=.., argmax=..)".
template <>
struct std::formatter<datamunge::nn::PoolWithArgmax> : datamunge::format_detail::text_formatter<datamunge::nn::PoolWithArgmax> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::PoolWithArgmax& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("PoolWithArgmax(output={}, argmax={})", F::tshape(x.output), F::tshape(x.argmax));
  }
};

/// @brief Prints a batch-norm result as "BatchNormResult(y=.., mean=.., variance=..)".
template <>
struct std::formatter<datamunge::nn::BatchNormResult> : datamunge::format_detail::text_formatter<datamunge::nn::BatchNormResult> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::BatchNormResult& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("BatchNormResult(y={}, mean={}, variance={})", F::tshape(x.y), F::tshape(x.mean), F::tshape(x.variance));
  }
};

/// @brief Prints dense-layer options as "DenseOptions(activation=, bias=, bias_value=, alpha=)" (the initializer is omitted).
template <>
struct std::formatter<datamunge::nn::DenseOptions> : datamunge::format_detail::text_formatter<datamunge::nn::DenseOptions> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::DenseOptions& x) {
    return std::format("DenseOptions(activation={}, bias={}, bias_value={}, alpha={})", x.activation, x.bias, x.bias_value, x.alpha);
  }
};

/// @brief Prints convolution-layer options (the initializer is omitted).
template <>
struct std::formatter<datamunge::nn::ConvLayerOptions> : datamunge::format_detail::text_formatter<datamunge::nn::ConvLayerOptions> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::ConvLayerOptions& x) {
    return std::format("ConvLayerOptions(stride={}, dilation={}, padding={}, activation={}, bias={}, alpha={})", x.stride, x.dilation, x.padding, x.activation, x.bias, x.alpha);
  }
};

/// @brief Prints batch-norm options as "BatchNormOptions(momentum=, eps=, scale=, center=, update_stats=)".
template <>
struct std::formatter<datamunge::nn::BatchNormOptions> : datamunge::format_detail::text_formatter<datamunge::nn::BatchNormOptions> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::BatchNormOptions& x) {
    return std::format("BatchNormOptions(momentum={}, eps={}, scale={}, center={}, update_stats={})", x.momentum, x.eps, x.scale, x.center, x.update_stats);
  }
};

/// @brief Prints an LSTM state as "LstmState(h=.., c=..)".
template <>
struct std::formatter<datamunge::nn::LstmState> : datamunge::format_detail::text_formatter<datamunge::nn::LstmState> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::LstmState& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("LstmState(h={}, c={})", F::tshape(x.h), F::tshape(x.c));
  }
};

/// @brief Prints a non-max-suppression result as "NmsResult(valid=k, indices=.., scores=..)".
template <>
struct std::formatter<datamunge::nn::NmsResult> : datamunge::format_detail::text_formatter<datamunge::nn::NmsResult> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::NmsResult& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("NmsResult(valid={}, indices={}, scores={})", x.valid, F::tshape(x.indices), F::tshape(x.scores));
  }
};

/// @brief Prints a complex tensor as its dtype/shape plus both parts' value previews.
template <>
struct std::formatter<datamunge::nn::ComplexTensor> : datamunge::format_detail::text_formatter<datamunge::nn::ComplexTensor> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return Three lines: "ComplexTensor dtype[shape]", then "real: .." and "imag: ..".
  static std::string text(const datamunge::nn::ComplexTensor& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    if (!x.real.defined()) return "ComplexTensor(undefined)";
    return std::format("ComplexTensor {}\nreal: {}\nimag: {}", F::tshape(x.real), x.real.values_str(),
                       x.imag.defined() ? x.imag.values_str() : std::string("undefined"));
  }
};

/// @brief Prints classification metrics as "ClassMetrics(precision=, recall=, f1=, iou=, support=)".
template <>
struct std::formatter<datamunge::nn::ClassMetrics> : datamunge::format_detail::text_formatter<datamunge::nn::ClassMetrics> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::ClassMetrics& x) {
    return std::format("ClassMetrics(precision={}, recall={}, f1={}, iou={}, support={})", x.precision, x.recall, x.f1, x.iou, x.support);
  }
};

/// @brief Prints a confusion matrix as a header (classes, total, accuracy) plus a truth-by-prediction count grid.
template <>
struct std::formatter<datamunge::nn::ConfusionMatrix> : datamunge::format_detail::text_formatter<datamunge::nn::ConfusionMatrix> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return Header line, then the grid (rows are truth, columns predictions).
  static std::string text(const datamunge::nn::ConfusionMatrix& x) {
    namespace g = datamunge::format_detail::grid_fmt;
    std::uint64_t total = 0, correct = 0;
    for (std::size_t i = 0; i < x.cells().size(); ++i)
      for (std::size_t j = 0; j < x.cells()[i].size(); ++j) {
        total += x.cells()[i][j];
        if (i == j) correct += x.cells()[i][j];
      }
    std::string head = std::format("ConfusionMatrix(classes={}, total={}", x.classes(), total);
    if (total > 0) head += std::format(", accuracy={}", static_cast<double>(correct) / static_cast<double>(total));
    head += ")";
    return g::with_body(head, g::grid(x.classes(), x.classes(), [&](std::size_t i, std::size_t j) {
                          return std::format("{}", x.cells()[i][j]);
                        }));
  }
};

/// @brief Prints a 32-bit Mersenne Twister stream as "Mt19937" (its seed and state are private).
template <>
struct std::formatter<datamunge::nn::Mt19937> : datamunge::format_detail::text_formatter<datamunge::nn::Mt19937> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return "Mt19937".
  static std::string text([[maybe_unused]] const datamunge::nn::Mt19937& x) {
    return "Mt19937";
  }
};

/// @brief Prints a 64-bit Mersenne Twister stream as "Mt19937_64" (its seed and state are private).
template <>
struct std::formatter<datamunge::nn::Mt19937_64> : datamunge::format_detail::text_formatter<datamunge::nn::Mt19937_64> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return "Mt19937_64".
  static std::string text([[maybe_unused]] const datamunge::nn::Mt19937_64& x) {
    return "Mt19937_64";
  }
};

/// @brief Prints an nn random stream as "Generator" (its seed and state are private; formatting never advances it).
template <>
struct std::formatter<datamunge::nn::Generator> : datamunge::format_detail::text_formatter<datamunge::nn::Generator> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return "Generator".
  static std::string text([[maybe_unused]] const datamunge::nn::Generator& x) {
    return "Generator";
  }
};

/// @brief Prints a memory scope as "PeakMemoryScope(baseline=b, peak_above_baseline=p, live_above_baseline=l)" (bytes).
template <>
struct std::formatter<datamunge::nn::PeakMemoryScope> : datamunge::format_detail::text_formatter<datamunge::nn::PeakMemoryScope> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::PeakMemoryScope& x) {
    return std::format("PeakMemoryScope(baseline={}, peak_above_baseline={}, live_above_baseline={})", x.baseline(), x.peak_above_baseline(), x.live_above_baseline());
  }
};

/// @brief Prints an optimizer configuration with the fields its rule uses, e.g. "OptimizerConfig(Adam, beta1=0.9, beta2=0.999, epsilon=1e-08)".
template <>
struct std::formatter<datamunge::nn::OptimizerConfig> : datamunge::format_detail::text_formatter<datamunge::nn::OptimizerConfig> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::OptimizerConfig& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    const std::string fields = F::optimizer_fields(x);
    return std::format("OptimizerConfig({}{}{})", F::optimizer_kind(x.kind), fields.empty() ? "" : ", ", fields);
  }
};

/// @brief Prints an optimizer as "Optimizer(Adam, lr=, iterations=, parameters=, <rule fields>)".
template <>
struct std::formatter<datamunge::nn::Optimizer> : datamunge::format_detail::text_formatter<datamunge::nn::Optimizer> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::Optimizer& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    const std::string fields = F::optimizer_fields(x.config());
    return std::format("Optimizer({}, lr={}, iterations={}, parameters={}{}{})", F::optimizer_kind(x.config().kind),
                       x.learning_rate(), x.iterations(), x.parameters().size(), fields.empty() ? "" : ", ", fields);
  }
};

/// @brief Prints a gradient accumulator as "GradientAccumulator(count=n, total_weight=w)".
template <>
struct std::formatter<datamunge::nn::GradientAccumulator> : datamunge::format_detail::text_formatter<datamunge::nn::GradientAccumulator> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::GradientAccumulator& x) {
    return std::format("GradientAccumulator(count={}, total_weight={})", x.count(), x.total_weight());
  }
};

/// @brief Prints a QR result as "QrResult(q=.., r=..)".
template <>
struct std::formatter<datamunge::nn::QrResult> : datamunge::format_detail::text_formatter<datamunge::nn::QrResult> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::QrResult& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("QrResult(q={}, r={})", F::tshape(x.q), F::tshape(x.r));
  }
};

/// @brief Prints an SVD result as "SvdResult(s=.., u=.., v=..)".
template <>
struct std::formatter<datamunge::nn::SvdResult> : datamunge::format_detail::text_formatter<datamunge::nn::SvdResult> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::SvdResult& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("SvdResult(s={}, u={}, v={})", F::tshape(x.s), F::tshape(x.u), F::tshape(x.v));
  }
};

/// @brief Prints a symmetric eigendecomposition as "EighResult(e=.., v=..)".
template <>
struct std::formatter<datamunge::nn::EighResult> : datamunge::format_detail::text_formatter<datamunge::nn::EighResult> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::EighResult& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("EighResult(e={}, v={})", F::tshape(x.e), F::tshape(x.v));
  }
};

/// @brief Prints a sign/log-determinant result as "SlogdetResult(sign=.., log_abs_det=..)" with values.
template <>
struct std::formatter<datamunge::nn::SlogdetResult> : datamunge::format_detail::text_formatter<datamunge::nn::SlogdetResult> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::SlogdetResult& x) {
    return std::format("SlogdetResult(sign={}, log_abs_det={})", x.sign.defined() ? x.sign.values_str() : "undefined", x.log_abs_det.defined() ? x.log_abs_det.values_str() : "undefined");
  }
};

/// @brief Prints a general eigendecomposition as "EigResult(e=complex(..), v=complex(..))".
template <>
struct std::formatter<datamunge::nn::EigResult> : datamunge::format_detail::text_formatter<datamunge::nn::EigResult> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::EigResult& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("EigResult(e={}, v={})", F::cshape(x.e), F::cshape(x.v));
  }
};

/// @brief Prints CTC options as "CtcOptions(blank=, time_major=, ignore_longer_outputs_than_inputs=)".
template <>
struct std::formatter<datamunge::nn::CtcOptions> : datamunge::format_detail::text_formatter<datamunge::nn::CtcOptions> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::CtcOptions& x) {
    return std::format("CtcOptions(blank={}, time_major={}, ignore_longer_outputs_than_inputs={})", x.blank, x.time_major, x.ignore_longer_outputs_than_inputs);
  }
};

/// @brief Prints a CTC loss result as "CtcLossResult(loss=.., gradient=..)".
template <>
struct std::formatter<datamunge::nn::CtcLossResult> : datamunge::format_detail::text_formatter<datamunge::nn::CtcLossResult> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::CtcLossResult& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("CtcLossResult(loss={}, gradient={})", F::tshape(x.loss), F::tshape(x.gradient));
  }
};

/// @brief Prints a CTC decoding as "CtcDecoded(sequences=[[..], ..], dense=..)".
template <>
struct std::formatter<datamunge::nn::CtcDecoded> : datamunge::format_detail::text_formatter<datamunge::nn::CtcDecoded> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::CtcDecoded& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("CtcDecoded(sequences={}, dense={})", datamunge::format_detail::preview(x.sequences), F::tshape(x.dense));
  }
};

/// @brief Prints a CRF decoding as "CrfDecoded(tags=.., scores=..)".
template <>
struct std::formatter<datamunge::nn::CrfDecoded> : datamunge::format_detail::text_formatter<datamunge::nn::CrfDecoded> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::CrfDecoded& x) {
    namespace F = datamunge::format_detail::nn_fmt;
    return std::format("CrfDecoded(tags={}, scores={})", F::tshape(x.tags), F::tshape(x.scores));
  }
};

/// @brief Prints a batch order as "BatchOrder(batches_per_epoch=, epoch=, cursor=)".
template <>
struct std::formatter<datamunge::nn::BatchOrder> : datamunge::format_detail::text_formatter<datamunge::nn::BatchOrder> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::BatchOrder& x) {
    return std::format("BatchOrder(batches_per_epoch={}, epoch={}, cursor={})", x.batches_per_epoch(), x.epoch(), x.cursor());
  }
};

/// @brief Prints early-stopping state as "EarlyStopping(best=, best_epoch=, misses=)" (best omitted before any observation).
template <>
struct std::formatter<datamunge::nn::EarlyStopping> : datamunge::format_detail::text_formatter<datamunge::nn::EarlyStopping> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::EarlyStopping& x) {
    if (x.best_epoch() < 0) return std::format("EarlyStopping(best_epoch=-1, misses={})", x.misses());
    return std::format("EarlyStopping(best={}, best_epoch={}, misses={})", x.best(), x.best_epoch(), x.misses());
  }
};

/// @brief Prints an early-stopping decision as "Decision(improved=, stop=)".
template <>
struct std::formatter<datamunge::nn::EarlyStopping::Decision> : datamunge::format_detail::text_formatter<datamunge::nn::EarlyStopping::Decision> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::EarlyStopping::Decision& x) {
    return std::format("Decision(improved={}, stop={})", x.improved, x.stop);
  }
};

/// @brief Prints a weighted mean as "WeightedMean(value=v, count=n)" (value omitted while empty).
template <>
struct std::formatter<datamunge::nn::WeightedMean> : datamunge::format_detail::text_formatter<datamunge::nn::WeightedMean> {
  /// @brief The printed text.
  /// @param x The value to print.
  /// @return One line.
  static std::string text(const datamunge::nn::WeightedMean& x) {
    if (x.count() == 0) return "WeightedMean(count=0)";
    return std::format("WeightedMean(value={}, count={})", x.value(), x.count());
  }
};
