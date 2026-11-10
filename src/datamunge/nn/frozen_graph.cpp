/// @file frozen_graph.cpp
/// @brief Checked GraphDef wire decoding and read-only native numeric graph evaluation.
#include "datamunge/nn/frozen_graph.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <string_view>
#include <utility>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/conv.hpp"
#include "datamunge/nn/image_ops.hpp"

namespace datamunge::nn {
/// @brief Private bounded protobuf decoding and pure operation dispatch.
namespace {
/// @brief Reject a malformed or unsupported frozen-graph contract with context.
/// @param message Specific parse, endpoint or operation problem.
/// @throws std::invalid_argument Always.
[[noreturn]] void Bad(const std::string& message) {
  throw std::invalid_argument("FrozenGraph: " + message);
}
/// @brief Borrow a bounded protobuf message and consume its primitive wire fields.
struct Reader {
  std::string_view bytes;  ///< Borrowed immutable message bytes, outliving the reader.
  std::size_t      at = 0; ///< First unread offset, always bounded by bytes.size().

  /// @brief Test whether all bytes were consumed.
  /// @return True exactly at the message end.
  bool done() const { return at == bytes.size(); }
  /// @brief Consume an unsigned varint with strict tenth-byte overflow checking.
  /// @return Exact uint64 bit pattern, including signed two's-complement encodings.
  /// @throws std::invalid_argument For truncation or overflow.
  std::uint64_t varint() {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 10; ++i) {
      if (at == bytes.size())
        Bad("truncated varint");
      auto b = static_cast<unsigned char>(bytes[at++]);
      if (i == 9 && b > 1)
        Bad("varint overflow");
      value |= std::uint64_t(b & 127) << (7 * i);
      if (!(b & 128))
        return value;
    }
    Bad("overlong varint");
  }
  /// @brief Split the next valid protobuf field/wire tag.
  /// @return Positive field number and wire type0,1,2 or5.
  /// @throws std::invalid_argument For invalid tags or unsupported groups.
  std::pair<unsigned, unsigned> tag() {
    auto key = varint();
    if ((key >> 3) == 0 || (key >> 3) >= (1ULL << 29) ||
        ((key & 7) != 0 && (key & 7) != 1 && (key & 7) != 2 && (key & 7) != 5))
      Bad("invalid protobuf field tag");
    return {unsigned(key >> 3), unsigned(key & 7)};
  }
  /// @brief Borrow the next exact number of bytes and advance safely.
  /// @param count Number of bytes to consume.
  /// @return Borrowed field slice.
  /// @throws std::invalid_argument For a truncated field.
  std::string_view take(std::uint64_t count) {
    if (count > bytes.size() - at)
      Bad("truncated field");
    auto out = bytes.substr(at, std::size_t(count));
    at += std::size_t(count);
    return out;
  }
  /// @brief Borrow one length-delimited field.
  /// @return Bounded field bytes.
  /// @throws std::invalid_argument For invalid length or truncation.
  std::string_view blob() { return take(varint()); }
  /// @brief Read a little-endian fixed-width bit pattern without host-endian assumptions.
  /// @param width Four or eight bytes, or one byte for Bool tensor content.
  /// @return Exact unsigned field bits.
  /// @throws std::invalid_argument For truncation.
  std::uint64_t fixed(unsigned width) {
    auto          raw  = take(width);
    std::uint64_t bits = 0;
    for (unsigned i = 0; i < width; ++i)
      bits |= std::uint64_t(static_cast<unsigned char>(raw[i])) << (8 * i);
    return bits;
  }
  /// @brief Skip one unknown supported field while preserving length checks.
  /// @param wire Wire type0,1,2 or5.
  /// @throws std::invalid_argument For truncation or invalid wire type.
  void skip(unsigned wire) {
    if (wire == 0)
      varint();
    else if (wire == 1)
      take(8);
    else if (wire == 2)
      blob();
    else if (wire == 5)
      take(4);
    else
      Bad("unsupported wire type");
  }
};
/// @brief Require the schema's wire type for a recognized field.
/// @param actual Actual field wire code.
/// @param expected Schema wire code.
/// @throws std::invalid_argument For mismatched wire codes.
void Wire(unsigned actual, unsigned expected) {
  if (actual != expected)
    Bad("wrong field wire type");
}
/// @brief Map supported TensorFlow numeric dtype codes into native dtypes.
/// @param code TensorFlow enum value.
/// @return Corresponding native dtype.
/// @throws std::invalid_argument For an unsupported dtype.
DType Type(std::int64_t code) {
  if (code == 1)
    return DType::Float32;
  if (code == 2)
    return DType::Float64;
  if (code == 3)
    return DType::Int32;
  if (code == 9)
    return DType::Int64;
  if (code == 10)
    return DType::Bool;
  Bad("unsupported numeric dtype " + std::to_string(code));
}
/// @brief Decode a fully known tensor shape with bounded rank and nonnegative axes.
/// @param bytes TensorShapeProto message.
/// @return Exact shape, empty for a scalar.
/// @throws std::invalid_argument For unknown rank, negative axes or excessive rank.
Shape ParseShape(std::string_view bytes) {
  Reader r{bytes};
  Shape  shape;
  while (!r.done()) {
    auto [f, w] = r.tag();
    if (f == 2) {
      Wire(w, 2);
      Reader       dim{r.blob()};
      std::int64_t size = 0;
      while (!dim.done()) {
        auto [df, dw] = dim.tag();
        if (df == 1) {
          Wire(dw, 0);
          size = std::bit_cast<std::int64_t>(dim.varint());
        } else
          dim.skip(dw);
      }
      if (size < 0 || shape.size() == 32)
        Bad("unsupported constant shape");
      shape.push_back(size);
    } else if (f == 3) {
      Wire(w, 0);
      if (r.varint())
        Bad("unknown constant rank");
    } else
      r.skip(w);
  }
  return shape;
}
/// @brief Decode one numeric TensorProto within the cumulative constant-storage budget.
/// @param bytes Complete TensorProto message.
/// @param remaining Remaining decoded-constant bytes, reduced before allocation.
/// @return Owned exact tensor, or undefined for an unsupported dtype.
/// @throws std::invalid_argument For malformed fields, shape/value mismatch or budget overflow.
Tensor ParseTensor(std::string_view bytes, std::uint64_t& remaining) {
  Reader                                         r{bytes};
  std::int64_t                                   code = 0;
  Shape                                          shape;
  std::string_view                               content;
  std::map<unsigned, std::vector<std::uint64_t>> values;
  while (!r.done()) {
    auto [f, w] = r.tag();
    if (f == 1) {
      Wire(w, 0);
      code = std::bit_cast<std::int64_t>(r.varint());
    } else if (f == 2) {
      Wire(w, 2);
      shape = ParseShape(r.blob());
    } else if (f == 4) {
      Wire(w, 2);
      content = r.blob();
    } else if (f == 5 || f == 6 || f == 7 || f == 10 || f == 11) {
      unsigned scalar_wire = f == 5 ? 5 : f == 6 ? 1 : 0;
      /// @brief Consume a typed scalar from either packed or unpacked storage.
      /// @param from Reader positioned at one numeric value.
      /// @return Exact serialized integer/floating bit pattern.
      auto read_value = [&](Reader& from) {
        return scalar_wire == 0 ? from.varint() : from.fixed(scalar_wire == 5 ? 4 : 8);
      };
      if (w == 2) {
        Reader packed{r.blob()};
        while (!packed.done())
          values[f].push_back(read_value(packed));
      } else {
        Wire(w, scalar_wire);
        values[f].push_back(read_value(r));
      }
    } else
      r.skip(w);
  }
  if (code == 0)
    Bad("missing constant dtype");
  if (code != 1 && code != 2 && code != 3 && code != 9 && code != 10)
    return {};
  auto          dtype = Type(code);
  auto          width = dtype_size(dtype);
  std::uint64_t count = 1;
  for (auto size : shape) {
    if (size && count > std::uint64_t(INT64_MAX) / std::uint64_t(size))
      Bad("constant product overflow");
    count *= std::uint64_t(size);
  }
  if (count > remaining / width)
    Bad("decoded constants exceed byte limit");
  remaining -= count * width;
  unsigned    field = code == 1 ? 5 : code == 2 ? 6 : code == 3 ? 7 : code == 9 ? 10 : 11;
  const auto& typed = values[field];
  if ((!content.empty() && content.size() != count * width) || typed.size() > count)
    Bad("constant payload does not match shape");
  Tensor out = empty(shape, dtype);
  Reader raw{content};
  for (std::uint64_t i = 0; i < count; ++i) {
    std::uint64_t bits = !content.empty() ? raw.fixed(unsigned(width))
                         : typed.empty()  ? 0
                                          : typed[std::min(std::size_t(i), typed.size() - 1)];
    if (code == 1)
      out.data_ptr<float>()[i] = std::bit_cast<float>(std::uint32_t(bits));
    else if (code == 2)
      out.data_ptr<double>()[i] = std::bit_cast<double>(bits);
    else if (code == 3)
      out.data_ptr<std::int32_t>()[i] = std::bit_cast<std::int32_t>(std::uint32_t(bits));
    else if (code == 9)
      out.data_ptr<std::int64_t>()[i] = std::bit_cast<std::int64_t>(bits);
    else {
      if (bits > 1)
        Bad("invalid Bool constant");
      out.data_ptr<bool>()[i] = bits != 0;
    }
  }
  return out;
}
/// @brief Scalar/list attribute storage for the supported static operation set.
struct Attr {
  unsigned         kind    = 0; ///< Last recognized AttrValue oneof tag.
  std::int64_t     integer = 0; ///< Integer, bool or dtype bit pattern.
  float            real    = 0; ///< Exact serialized Float32 attribute.
  std::string      text;        ///< String attribute, including embedded bytes.
  Shape            list;        ///< Packed/unpacked signed int64 list, preserving order.
  std::string_view tensor;      ///< Borrowed TensorProto bytes, used only during model parsing.
};
/// @brief Decode supported oneof attributes with last-value semantics.
/// @param bytes AttrValue message.
/// @return Owned numeric/string metadata plus borrowed tensor payload.
/// @throws std::invalid_argument For malformed recognized fields.
Attr ParseAttr(std::string_view bytes) {
  Reader r{bytes};
  Attr   a;
  while (!r.done()) {
    auto [f, w] = r.tag();
    if (f == 1) {
      Wire(w, 2);
      a      = {};
      a.kind = 1;
      Reader list{r.blob()};
      while (!list.done()) {
        auto [lf, lw] = list.tag();
        if (lf != 3) {
          list.skip(lw);
          continue;
        }
        if (lw == 2) {
          Reader packed{list.blob()};
          while (!packed.done())
            a.list.push_back(std::bit_cast<std::int64_t>(packed.varint()));
        } else {
          Wire(lw, 0);
          a.list.push_back(std::bit_cast<std::int64_t>(list.varint()));
        }
      }
    } else if (f == 2 || f == 8) {
      Wire(w, 2);
      a          = {};
      a.kind     = f;
      auto value = r.blob();
      if (f == 2)
        a.text = value;
      else
        a.tensor = value;
    } else if (f == 3 || f == 5 || f == 6) {
      Wire(w, 0);
      a         = {};
      a.kind    = f;
      a.integer = std::bit_cast<std::int64_t>(r.varint());
    } else if (f == 4) {
      Wire(w, 5);
      a      = {};
      a.kind = 4;
      a.real = std::bit_cast<float>(std::uint32_t(r.fixed(4)));
    } else
      r.skip(w);
  }
  return a;
}
/// @brief Exact name, input endpoint strings and owned attributes for one operation.
struct GraphNode {
  std::string                 name;   ///< Unique serialized operation name.
  std::string                 op;     ///< Declared operation kind.
  std::vector<std::string>    inputs; ///< Input and control endpoint strings in original order.
  std::map<std::string, Attr> attrs;  ///< Last map entry per attribute name.
  Tensor                      value;  ///< Immutable decoded numeric Const, otherwise undefined.
};
/// @brief Check TensorFlow's serialized operation-name grammar using ASCII only.
/// @param name Candidate node name.
/// @return True for [A-Za-z0-9.][A-Za-z0-9_>./]*.
bool ValidName(const std::string& name) {
  /// @brief Recognize an ASCII letter without consulting locale state.
  /// @param c Input character.
  /// @return True for A..Z or a..z.
  auto alpha = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
  /// @brief Recognize an ASCII decimal digit.
  /// @param c Input character.
  /// @return True for0..9.
  auto digit = [](char c) { return c >= '0' && c <= '9'; };
  if (name.empty() || !(alpha(name[0]) || digit(name[0]) || name[0] == '.'))
    return false;
  for (char c : name)
    if (!(alpha(c) || digit(c) || c == '_' || c == '>' || c == '.' || c == '/'))
      return false;
  return true;
}
/// @brief Parse one NodeDef and decode its supported numeric constant, if any.
/// @param bytes Complete NodeDef message.
/// @param remaining Remaining cumulative decoded constant budget.
/// @return Owned node metadata and immutable numeric storage.
/// @throws std::invalid_argument For malformed fields, invalid names or invalid constants.
GraphNode ParseNode(std::string_view bytes, std::uint64_t& remaining) {
  Reader    r{bytes};
  GraphNode n;
  while (!r.done()) {
    auto [f, w] = r.tag();
    if (f == 1 || f == 2 || f == 3) {
      Wire(w, 2);
      std::string text(r.blob());
      if (f == 1)
        n.name = std::move(text);
      else if (f == 2)
        n.op = std::move(text);
      else
        n.inputs.push_back(std::move(text));
    } else if (f == 5) {
      Wire(w, 2);
      Reader      entry{r.blob()};
      std::string key;
      Attr        attr;
      while (!entry.done()) {
        auto [ef, ew] = entry.tag();
        if (ef == 1) {
          Wire(ew, 2);
          key = entry.blob();
        } else if (ef == 2) {
          Wire(ew, 2);
          attr = ParseAttr(entry.blob());
        } else
          entry.skip(ew);
      }
      if (key.empty())
        Bad("empty attribute name");
      n.attrs[key] = std::move(attr);
    } else
      r.skip(w);
  }
  if (!ValidName(n.name) || n.op.empty())
    Bad("invalid node name or operation");
  if (n.op == "Const") {
    auto it = n.attrs.find("value");
    if (it == n.attrs.end() || it->second.kind != 8)
      Bad("Const without TensorProto value");
    n.value       = ParseTensor(it->second.tensor, remaining);
    auto declared = n.attrs.find("dtype");
    if (declared != n.attrs.end() &&
        (declared->second.kind != 6 ||
         (n.value.defined() && n.value.dtype() != Type(declared->second.integer))))
      Bad(n.name + ": Const dtype conflicts with TensorProto");
  }
  // Payload views must not outlive the byte buffer passed into the constructor.
  for (auto& [key, a] : n.attrs)
    a.tensor = {};
  return n;
}
/// @brief Normalize a single output-zero data endpoint, rejecting other ports and controls.
/// @param text GraphNode name or node:0.
/// @return Canonical node name.
/// @throws std::invalid_argument For empty, control or unsupported port names.
std::string Name(const std::string& text) {
  if (text.empty() || text.front() == '^' || text.find('\0') != std::string::npos)
    Bad("invalid endpoint");
  auto colon = text.find(':');
  if (colon == std::string::npos)
    return text;
  if (colon == 0 || text.substr(colon) != ":0")
    Bad("only output port zero is supported: " + text);
  return text.substr(0, colon);
}
/// @brief Read a correctly typed scalar integer/bool attribute or its explicit default.
/// @param n Operation metadata.
/// @param key Attribute name.
/// @param kind Expected oneof tag3(integer),5(bool) or6(dtype).
/// @param fallback Value when absent.
/// @return Exact stored attribute or fallback.
/// @throws std::invalid_argument For wrong attribute kind.
std::int64_t
Integer(const GraphNode& n, const std::string& key, unsigned kind, std::int64_t fallback) {
  auto it = n.attrs.find(key);
  if (it == n.attrs.end())
    return fallback;
  if (it->second.kind != kind)
    Bad(n.name + ": wrong attribute " + key);
  if (kind == 5 && it->second.integer != 0 && it->second.integer != 1)
    Bad("invalid bool attribute");
  return it->second.integer;
}
/// @brief Read a string attribute with its explicit default.
/// @param n Operation metadata.
/// @param key Attribute name.
/// @param fallback Default string when absent.
/// @return Stored string or fallback.
/// @throws std::invalid_argument For wrong attribute kind.
std::string Text(const GraphNode& n, const std::string& key, const std::string& fallback) {
  auto it = n.attrs.find(key);
  if (it == n.attrs.end())
    return fallback;
  if (it->second.kind != 2)
    Bad(n.name + ": wrong string attribute " + key);
  return it->second.text;
}
/// @brief Read one integer list attribute with an explicit fallback.
/// @param n Operation metadata.
/// @param key Attribute name.
/// @param fallback Default list when absent.
/// @return Stored list or fallback.
/// @throws std::invalid_argument For wrong attribute kind.
Shape List(const GraphNode& n, const std::string& key, Shape fallback) {
  auto it = n.attrs.find(key);
  if (it == n.attrs.end())
    return fallback;
  if (it->second.kind != 1)
    Bad(n.name + ": wrong list attribute " + key);
  return it->second.list;
}
/// @brief Decode supported SAME/VALID spatial padding.
/// @param n Operation metadata with required padding string.
/// @return Corresponding native padding.
/// @throws std::invalid_argument For missing/unsupported padding.
Padding SpatialPadding(const GraphNode& n) {
  auto value = Text(n, "padding", "");
  if (value == "SAME")
    return Padding::Same;
  if (value == "VALID")
    return Padding::Valid;
  Bad(n.name + ": unsupported spatial padding");
}
/// @brief Extract positive spatial settings from an NHWC four-element attribute.
/// @param n Operation metadata.
/// @param key strides, dilations or ksize.
/// @param fallback Default four-element list for optional attrs.
/// @return Positive height/width values, with batch/channel exactly one.
/// @throws std::invalid_argument For invalid list geometry or non-NHWC settings.
std::array<std::int64_t, 2> Spatial(const GraphNode& n, const std::string& key, Shape fallback) {
  auto list = List(n, key, std::move(fallback));
  if (list.size() != 4 || list[0] != 1 || list[3] != 1 || list[1] <= 0 || list[2] <= 0 ||
      Text(n, "data_format", "NHWC") != "NHWC")
    Bad(n.name + ": requires positive NHWC " + key);
  return {list[1], list[2]};
}
/// @brief Convert an Int32/64 tensor to exact metadata values without floating conversion.
/// @param tensor Integer scalar/vector metadata input.
/// @return Exact signed values in logical order.
/// @throws std::invalid_argument For noninteger metadata dtype.
Shape Ints(const Tensor& tensor) {
  if (!is_integral(tensor.dtype()))
    Bad("integer metadata tensor required");
  return tensor.contiguous().to_vector<std::int64_t>();
}
/// @brief Read exactly one integer metadata scalar, accepting only rank zero.
/// @param tensor Integer scalar tensor.
/// @return Exact axis value.
/// @throws std::invalid_argument For wrong dtype or shape.
std::int64_t Axis(const Tensor& tensor) {
  auto values = Ints(tensor);
  if (tensor.dim() != 0 || values.size() != 1)
    Bad("axis must be an integer scalar");
  return values[0];
}
/// @brief Recognize only pure supported static numeric operations.
/// @param op Serialized operation kind.
/// @return True when native dispatch implements it.
bool Supported(const std::string& op) {
  const std::set<std::string> names{
      "Const",
      "Placeholder",
      "Identity",
      "StopGradient",
      "CheckNumerics",
      "Relu",
      "Add",
      "AddV2",
      "Sub",
      "Mul",
      "Div",
      "RealDiv",
      "BiasAdd",
      "MatMul",
      "Reshape",
      "Squeeze",
      "ExpandDims",
      "Concat",
      "ConcatV2",
      "Softmax",
      "ResizeBilinear",
      "Conv2D",
      "AvgPool",
      "MaxPool",
      "BatchNormWithGlobalNormalization"};
  return names.count(op) != 0;
}
/// @brief Execute one preflighted pure numeric operation through native tensor kernels.
/// @param n Immutable operation metadata.
/// @param x Ordered data inputs, excluding completed control dependencies.
/// @return New tensor or immutable input/constant alias, cloned before public return.
/// @throws std::invalid_argument For unsupported settings, arity, CheckNumerics or primitive
/// errors.
Tensor Apply(const GraphNode& n, const std::vector<Tensor>& x) {
  const auto& op = n.op;
  /// @brief Require the exact data input count, excluding already-executed control edges.
  /// @param count Required input count.
  auto arity = [&](std::size_t count) {
    if (x.size() != count)
      Bad(n.name + ": input count");
  };
  if (op == "Const") {
    arity(0);
    if (!n.value.defined())
      Bad(n.name + ": unsupported constant dtype");
    return n.value;
  }
  if (op == "Placeholder")
    Bad(n.name + ": missing placeholder feed");
  const auto declared = Integer(n, "T", 6, 0);
  if (declared) {
    const auto type = Type(declared);
    for (std::size_t i = 0; i < x.size(); ++i) {
      bool metadata = (op == "Concat" && i == 0) || (op == "ConcatV2" && i + 1 == x.size()) ||
                      ((op == "Reshape" || op == "ExpandDims" || op == "ResizeBilinear") && i == 1);
      if (!metadata && x[i].dtype() != type)
        Bad(n.name + ": input dtype conflicts with T");
    }
  }
  if (op == "Identity" || op == "StopGradient" || op == "CheckNumerics" || op == "Relu" ||
      op == "Softmax" || op == "Squeeze") {
    arity(1);
    if (op == "CheckNumerics") {
      if (!is_floating(x[0].dtype()))
        Bad(n.name + ": floating CheckNumerics required");
      for (double v : x[0].contiguous().to_vector<double>())
        if (!std::isfinite(v))
          Bad(n.name + ": " + Text(n, "message", "nonfinite input"));
    }
    if (op == "Relu") {
      if (x[0].dtype() == DType::Bool)
        Bad(n.name + ": numeric Relu required");
      return relu(x[0]);
    }
    if (op == "Softmax") {
      if (!is_floating(x[0].dtype()))
        Bad(n.name + ": floating Softmax required");
      return softmax(x[0], -1);
    }
    if (op == "Squeeze")
      return squeeze(x[0], List(n, "squeeze_dims", {}));
    return x[0];
  }
  if (op == "Concat" || op == "ConcatV2") {
    auto count = Integer(n, "N", 3, -1);
    if (count < 2 || std::uint64_t(count) + 1 != x.size())
      Bad(n.name + ": concat input count");
    std::vector<Tensor> items;
    std::int64_t        axis = 0;
    if (op == "Concat") {
      axis = Axis(x.front());
      items.assign(x.begin() + 1, x.end());
    } else {
      axis = Axis(x.back());
      items.assign(x.begin(), x.end() - 1);
    }
    return concat(items, axis);
  }
  if (op == "Conv2D" || op == "AvgPool" || op == "MaxPool") {
    if (x.empty() || !is_floating(x[0].dtype()))
      Bad(n.name + ": floating spatial input required");
    if (op == "Conv2D") {
      arity(2);
      Conv2DOptions o;
      o.strides   = Spatial(n, "strides", {});
      o.dilations = Spatial(n, "dilations", {1, 1, 1, 1});
      o.padding   = SpatialPadding(n);
      return conv2d(x[0], x[1], o);
    }
    arity(1);
    auto window = Spatial(n, "ksize", {}), stride = Spatial(n, "strides", {});
    return op == "AvgPool" ? avg_pool2d(x[0], window, stride, SpatialPadding(n))
                           : max_pool2d(x[0], window, stride, SpatialPadding(n));
  }
  if (op == "BatchNormWithGlobalNormalization") {
    arity(5);
    if (x[0].dim() != 4 || !is_floating(x[0].dtype()))
      Bad(n.name + ": floating NHWC BN required");
    for (int i = 1; i < 5; ++i)
      if (x[i].shape() != Shape({x[0].size(3)}) || x[i].dtype() != x[0].dtype())
        Bad(n.name + ": BN channel/dtype alignment");
    auto it = n.attrs.find("variance_epsilon");
    if (it == n.attrs.end() || it->second.kind != 4 || !std::isfinite(it->second.real) ||
        it->second.real < 0)
      Bad(n.name + ": invalid BN epsilon");
    // Match TensorFlow's old kernel: scale=rsqrt(var+eps), optional gamma,
    // then x*scale+(beta-mean*scale), rather than (x-mean)*scale+beta.
    Tensor scale = rsqrt(x[2] + double(it->second.real));
    if (Integer(n, "scale_after_normalization", 5, 0))
      scale = scale * x[4];
    return x[0] * scale + (x[3] - x[1] * scale);
  }
  arity(2);
  if (op == "Add" || op == "AddV2" || op == "Sub" || op == "Mul") {
    if (x[0].dtype() == DType::Bool || x[0].dtype() != x[1].dtype())
      Bad(n.name + ": matching numeric arithmetic required");
    if (op == "Add" || op == "AddV2")
      return x[0] + x[1];
    if (op == "Sub")
      return x[0] - x[1];
    return x[0] * x[1];
  }
  if (op == "Div" || op == "RealDiv") {
    if (!is_floating(x[0].dtype()) || x[0].dtype() != x[1].dtype())
      Bad(n.name + ": floating division required");
    return x[0] / x[1];
  }
  if (op == "BiasAdd") {
    if (Text(n, "data_format", "NHWC") != "NHWC" || x[0].dim() < 2 ||
        x[1].shape() != Shape({x[0].shape().back()}) || x[0].dtype() != x[1].dtype())
      Bad(n.name + ": last-axis BiasAdd alignment");
    return x[0] + x[1];
  }
  if (op == "MatMul")
    return matmul(x[0], x[1], Integer(n, "transpose_a", 5, 0), Integer(n, "transpose_b", 5, 0));
  if (op == "Reshape") {
    if (x[1].dim() != 1)
      Bad(n.name + ": reshape shape must be a vector");
    return reshape(x[0], Ints(x[1]));
  }
  if (op == "ExpandDims")
    return expand_dims(x[0], Axis(x[1]));
  if (op == "ResizeBilinear") {
    if (x[0].dtype() != DType::Float32)
      Bad(n.name + ": only Float32 ResizeBilinear is supported");
    if (x[1].shape() != Shape({2}))
      Bad(n.name + ": resize size must have two entries");
    auto size = Ints(x[1]);
    return resize_bilinear(
        x[0],
        {size[0], size[1]},
        Integer(n, "align_corners", 5, 0),
        Integer(n, "half_pixel_centers", 5, 0));
  }
  Bad(n.name + ": unsupported operation " + op);
}
} // namespace

/// @brief Immutable parsed model and exact endpoint lookup table.
struct FrozenGraph::Impl {
  std::vector<GraphNode> nodes; ///< Nodes in serialized order, independent of topological order.
  std::map<std::string, std::size_t> index; ///< Unique name to owned node position.

  /// @brief Resolve an existing output-zero endpoint.
  /// @param text GraphNode name or node:0.
  /// @return Stable node index.
  /// @throws std::invalid_argument For invalid syntax or unknown name.
  std::size_t id(const std::string& text) const {
    auto name = Name(text);
    auto it   = index.find(name);
    if (it == index.end())
      Bad("unknown endpoint " + text);
    return it->second;
  }
};

FrozenGraph::FrozenGraph(const std::vector<unsigned char>& bytes, std::uint64_t max_bytes) {
  if (bytes.empty() || !max_bytes || bytes.size() > max_bytes)
    Bad("serialized graph exceeds positive byte limit");
  Reader        r{std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size())};
  auto          out       = std::make_shared<Impl>();
  std::uint64_t remaining = max_bytes;
  while (!r.done()) {
    auto [f, w] = r.tag();
    if (f != 1) {
      r.skip(w);
      continue;
    }
    Wire(w, 2);
    if (out->nodes.size() == 100000)
      Bad("too many nodes");
    GraphNode node = ParseNode(r.blob(), remaining);
    if (!out->index.emplace(node.name, out->nodes.size()).second)
      Bad("duplicate node name " + node.name);
    out->nodes.push_back(std::move(node));
  }
  if (out->nodes.empty())
    Bad("graph has no nodes");
  for (const auto& n : out->nodes)
    for (const auto& input : n.inputs)
      out->id(!input.empty() && input.front() == '^' ? input.substr(1) : input);
  impl_ = std::move(out);
}
FrozenGraph FrozenGraph::load(const std::string& path, std::uint64_t max_bytes) {
  if (!max_bytes)
    Bad("byte limit must be positive");
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error))
    throw std::runtime_error("FrozenGraph: not a regular file: " + path);
  auto size = std::filesystem::file_size(path, error);
  if (error)
    throw std::runtime_error("FrozenGraph: cannot size file: " + path);
  if (size > max_bytes || size > std::uint64_t(std::numeric_limits<std::streamsize>::max()))
    Bad("file exceeds byte limit");
  std::ifstream              in(path, std::ios::binary);
  std::vector<unsigned char> bytes(std::size_t(size), 0);
  if (!in.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(size)) ||
      in.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("FrozenGraph: incomplete or changed file: " + path);
  return FrozenGraph(bytes, max_bytes);
}
std::vector<std::string> FrozenGraph::node_names() const {
  std::vector<std::string> out;
  for (const auto& n : impl_->nodes)
    out.push_back(n.name);
  return out;
}
std::string FrozenGraph::operation(const std::string& name) const {
  return impl_->nodes[impl_->id(name)].op;
}
Tensor FrozenGraph::constant(const std::string& name) const {
  const auto& n = impl_->nodes[impl_->id(name)];
  if (n.op != "Const" || !n.value.defined())
    Bad(name + ": not a supported numeric constant");
  return n.value.detach().clone();
}
NamedTensors FrozenGraph::evaluate(
    const NamedTensors&             feeds,
    const std::vector<std::string>& fetches) const {
  NoGradGuard                guard;
  const auto&                g = *impl_;
  std::vector<Tensor>        value(g.nodes.size());
  std::vector<unsigned char> visited(g.nodes.size(), 0);
  std::vector<std::size_t>   order, uses(g.nodes.size(), 0), roots;
  for (const auto& [key, tensor] : feeds) {
    auto id = g.id(key);
    if (value[id].defined() || !tensor.defined())
      Bad("duplicate or undefined feed " + key);
    const auto& node = g.nodes[id];
    auto code = Integer(node, node.op == "Const" || node.op == "Placeholder" ? "dtype" : "T", 6, 0);
    if (code && tensor.dtype() != Type(code))
      Bad(key + ": feed dtype mismatch");
    if (tensor.device().backend != compute::Backend::CPU)
      throw std::logic_error("FrozenGraph: feeds require CPU storage");
    value[id] = tensor.detach();
  }
  /// @brief Preflight fetched ancestry, preserving control edges and checking bounded recursion.
  /// @param id Requested node index.
  /// @param depth Current dependency depth, at most1024.
  std::function<void(std::size_t, unsigned)> visit = [&](std::size_t id, unsigned depth) {
    if (value[id].defined() || visited[id] == 2)
      return;
    if (visited[id] == 1)
      Bad("cycle in fetched graph");
    if (depth > 1024)
      Bad("dependency depth exceeds1024");
    const auto& node = g.nodes[id];
    if (!Supported(node.op))
      Bad(node.name + ": unsupported operation " + node.op);
    visited[id] = 1;
    for (const auto& input : node.inputs) {
      auto child = g.id(input.front() == '^' ? input.substr(1) : input);
      ++uses[child];
      visit(child, depth + 1);
    }
    visited[id] = 2;
    order.push_back(id);
  };
  for (const auto& key : fetches) {
    auto id = g.id(key);
    roots.push_back(id);
    ++uses[id];
    visit(id, 0);
  }
  for (auto id : order) {
    const auto&         node = g.nodes[id];
    std::vector<Tensor> inputs;
    for (const auto& input : node.inputs)
      if (input.front() != '^')
        inputs.push_back(value[g.id(input)]);
    value[id] = Apply(node, inputs);
    for (const auto& input : node.inputs) {
      auto child = g.id(input.front() == '^' ? input.substr(1) : input);
      if (--uses[child] == 0)
        value[child] = {};
    }
  }
  NamedTensors out;
  for (std::size_t i = 0; i < roots.size(); ++i)
    out.emplace_back(fetches[i], value[roots[i]].detach().clone());
  return out;
}
} // namespace datamunge::nn
