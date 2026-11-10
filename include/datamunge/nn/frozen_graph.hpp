#pragma once
/// @file frozen_graph.hpp
/// @brief Native read-only execution of the numeric static subset of TensorFlow GraphDef.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "datamunge/nn/io.hpp"

namespace datamunge::nn {
/// @brief Own an immutable frozen GraphDef and execute its supported numeric subgraphs natively.
///
/// No TensorFlow runtime, protobuf library, variables, training, network access or
/// random sampling. Numeric constants retain exact Float32/64, Int32/64 or Bool
/// bits; scalar rank is preserved. Unsupported constant types/operations may be
/// present outside fetched ancestry, allowing feeds to bypass JPEG decoding.
/// evaluate() documents the supported operations. Models are caller-supplied;
/// this class does not certify their identity or preprocessing. Copies share an
/// immutable parsed model; evaluations use independent local working storage and
/// are safe concurrently when caller feeds are not concurrently mutated. Outputs
/// are detached owning copies. Mutable internal constant storage never escapes.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const std::vector<unsigned char>
/// bytes{10,30,10,1,99,18,5,67,111,110,115,116,42,18,10,5,118,97,108,117,101,18,9,66,7,8,1,45,0,0,0,64};
/// dm::nn::FrozenGraph g(bytes); // Standard GraphDef containing scalar Const c=2.
/// const auto out=g.evaluate({}, {"c:0"}); // out[0].second.item() == 2.
/// @endcode
/// @par References
/// - TensorFlow GraphDef/NodeDef/AttrValue/TensorProto schemas:
/// https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/framework .
/// - Protocol Buffers wire encoding, https://protobuf.dev/programming-guides/encoding/ .
/// @see evaluate(), constant(), node_names()
class FrozenGraph {
 public:
  /// @brief Parse owned model bytes and decode supported immutable numeric constants.
  ///
  /// The byte buffer may be destroyed after construction. The positive bound
  /// limits both serialized bytes and total decoded numeric constant storage,
  /// default512MiB. Rejects malformed lengths/varints/shapes, duplicate names,
  /// missing referenced nodes, nonzero serialized input ports and excessive node count(100000).
  /// Numeric constants accept little-endian tensor_content or typed repeated fields; missing typed
  /// values fill zero and a short nonempty list repeats its last value. Unknown
  /// metadata fields are skipped. Unsupported dtype constants are deferred until
  /// requested. Cycles and operation support are checked for the fetched graph.
  /// @param bytes Complete GraphDef wire bytes, nonempty and within max_bytes.
  /// @param max_bytes Positive byte/decoded-constant bound, default512MiB.
  /// @throws std::invalid_argument For malformed/inconsistent input or exceeded bounds.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// dm::nn::FrozenGraph g(std::vector<unsigned
  /// char>{10,30,10,1,99,18,5,67,111,110,115,116,42,18,10,5,118,97,108,117,101,18,9,66,7,8,1,45,0,0,0,64});
  /// // One node, scalar Float32 c=2, without TensorFlow installed.
  /// @endcode
  /// @par References
  /// - TensorFlow TensorProto,
  /// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto .
  /// @see FrozenGraph, load()
  explicit FrozenGraph(
      const std::vector<unsigned char>& bytes,
      std::uint64_t                     max_bytes = 512ULL * 1024 * 1024);

  /// @brief Load and parse a bounded regular local GraphDef file without downloading anything.
  /// @param path Readable regular file with complete GraphDef bytes.
  /// @param max_bytes Positive serialized/decoded-constant byte limit, default512MiB.
  /// @return Owning immutable parsed FrozenGraph, independent of subsequent file changes.
  /// @throws std::invalid_argument For malformed model bytes or exceeded bounds.
  /// @throws std::runtime_error For missing/nonregular files or read errors.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// #include <fstream>
  /// const std::vector<unsigned char>
  /// b{10,30,10,1,99,18,5,67,111,110,115,116,42,18,10,5,118,97,108,117,101,18,9,66,7,8,1,45,0,0,0,64};
  /// std::ofstream f("build/frozen-constant.pb",std::ios::binary);f.write(reinterpret_cast<const
  /// char*>(b.data()),b.size());f.close(); auto
  /// graph=dm::nn::FrozenGraph::load("build/frozen-constant.pb");
  /// @endcode
  /// @see FrozenGraph(), evaluate()
  static FrozenGraph load(const std::string& path, std::uint64_t max_bytes = 512ULL * 1024 * 1024);

  /// @brief List every model node name in serialized order, including unsupported unused nodes.
  /// @return Owned copied names; no mutable model metadata is exposed.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// dm::nn::FrozenGraph g(std::vector<unsigned
  /// char>{10,30,10,1,99,18,5,67,111,110,115,116,42,18,10,5,118,97,108,117,101,18,9,66,7,8,1,45,0,0,0,64});
  /// auto names=g.node_names(); // {"c"}.
  /// @endcode
  /// @see operation(), constant()
  std::vector<std::string> node_names() const;

  /// @brief Read an exact node's declared operation without executing it.
  /// @param name Existing node name, optionally suffixed:0; other output ports rejected.
  /// @return Owned operation name, for example "Const" or "Conv2D".
  /// @throws std::invalid_argument For unknown names or unsupported output-port syntax.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// dm::nn::FrozenGraph g(std::vector<unsigned
  /// char>{10,30,10,1,99,18,5,67,111,110,115,116,42,18,10,5,118,97,108,117,101,18,9,66,7,8,1,45,0,0,0,64});
  /// auto op=g.operation("c:0"); // "Const".
  /// @endcode
  /// @see node_names(), evaluate()
  std::string operation(const std::string& name) const;

  /// @brief Copy one decoded numeric constant while keeping internal model storage immutable.
  /// @param name Existing numeric Const node name, optionally suffixed:0.
  /// @return Detached owning CPU constant copy with exact dtype, shape and value bits.
  /// @throws std::invalid_argument For unknown names, non-Const nodes or unsupported constant
  /// dtype.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// dm::nn::FrozenGraph g(std::vector<unsigned
  /// char>{10,30,10,1,99,18,5,67,111,110,115,116,42,18,10,5,118,97,108,117,101,18,9,66,7,8,1,45,0,0,0,64});
  /// auto c=g.constant("c"); // Float32 scalar2, independent mutable storage.
  /// @endcode
  /// @see evaluate(), operation()
  Tensor constant(const std::string& name) const;

  /// @brief Execute only fetched numeric ancestry with explicit feed overrides and no autograd.
  ///
  /// Supports Const, Placeholder, Identity, StopGradient, CheckNumerics, Relu,
  /// Add/AddV2, Sub, Mul, Div/RealDiv, BiasAdd, MatMul, Reshape, Squeeze,
  /// ExpandDims, Concat/ConcatV2, Softmax, ResizeBilinear, Conv2D, AvgPool,
  /// MaxPool and legacy BatchNormWithGlobalNormalization. Spatial ops use NHWC,
  /// SAME/VALID; no batch/channel stride or pooling. Conv spatial dilation is
  /// supported. BiasAdd is last-axis; MatMul honors transpose flags; resize
  /// accepts Float32 and honors align_corners/half_pixel_centers. Division and
  /// Softmax require floating inputs; spatial ops require Float32/64. Legacy BN preserves epsilon's
  /// serialized Float32 value and scale_after_normalization. Control dependencies
  /// execute too, including CheckNumerics, and cycles/depth>1024 are rejected.
  /// Feed overrides cut all input/control ancestry of that node. JPEG/string,
  /// variable, function, random and multi-output execution is unsupported.
  ///
  /// Feeds are defined CPU tensors with native numeric dtype; static dtype attrs
  /// are checked where available. No shape inference or pretrained identity is
  /// assumed; primitive geometry checks propagate. NaNs are permitted unless a
  /// reachable CheckNumerics rejects them. Inputs/internal weights are unchanged.
  /// First/higher paths always stop. Independent local work storage releases
  /// intermediates after their last consumer; fetch order and spelling are retained,
  /// including duplicate fetches. Empty fetches return empty, after feed checks.
  /// @param feeds Named feed tensors; unique node names optionally suffixed:0.
  /// @param fetches Requested existing output-zero names, in result order.
  /// @return Detached owning result copies named exactly as requested.
  /// @throws std::invalid_argument For invalid feeds/names, unsupported reachable ops/attrs,
  /// cycles, nonfinite CheckNumerics values or primitive argument errors.
  /// @throws std::logic_error For non-CPU storage or primitive runtime restrictions.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// namespace n=dm::nn;
  /// n::FrozenGraph g(std::vector<unsigned
  /// char>{10,30,10,1,99,18,5,67,111,110,115,116,42,18,10,5,118,97,108,117,101,18,9,66,7,8,1,45,0,0,0,64});
  /// auto out=g.evaluate({{"c",n::full({},7)}},{"c:0","c"}); // Two detached scalar7 copies.
  /// @endcode
  /// @par References
  /// - TensorFlow operation contracts, https://www.tensorflow.org/api_docs/python/tf/raw_ops .
  /// @see FrozenGraph, constant(), datamunge::nn::NoGradGuard
  NamedTensors evaluate(const NamedTensors& feeds, const std::vector<std::string>& fetches) const;

 private:
  /// @brief Immutable parsed node/attribute/constant implementation, shared by value copies.
  struct Impl;
  std::shared_ptr<const Impl> impl_; ///< Shared immutable model owner; no caller-owned references.
};
} // namespace datamunge::nn
