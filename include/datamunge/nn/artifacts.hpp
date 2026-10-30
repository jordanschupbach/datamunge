#pragma once

/// @file artifacts.hpp
/// @brief Typed named-tensor restoration, byte-exact strings and eager inference snapshots.

#include <map>
#include <set>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/io.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/params.hpp"

namespace datamunge::nn {

/// @brief Restore arbitrary typed tensor handles after validating every assignment.
///
/// Unlike ParameterStore, this accepts integer and boolean state. Targets are
/// existing tensor handles; their storage is updated in place without recording
/// gradients. Names must be unique in both lists. Validation failure changes no
/// target. Concurrent mutation of the same tensors is unsupported.
/// @param targets Unique names and defined destination tensors.
/// @param source Unique names and defined snapshot values; never mutated.
/// @param strict If true (default), reject extra source names too.
/// @throws std::invalid_argument On duplicate/missing names, undefined tensors,
///         differing dtypes/shapes or extra strict-mode source names.
/// @par Example
/// @code
/// auto counter=dm::nn::scalar(0,dm::nn::DType::Int64);
/// dm::nn::restore_tensors({{"step",counter}},{{"step",dm::nn::scalar(7,dm::nn::DType::Int64)}});
/// @endcode
/// @see load_npz(), restore_training_state()
inline void
restore_tensors(const NamedTensors& targets, const NamedTensors& source, bool strict = true) {
  std::map<std::string, Tensor> values;
  for (const auto& [name, value] : source)
    if (!value.defined() || !values.emplace(name, value).second)
      throw std::invalid_argument("restore_tensors: undefined or duplicate source");
  std::set<std::string> names;
  NamedTensors          staged;
  for (const auto& [name, target] : targets) {
    if (!target.defined() || !names.insert(name).second)
      throw std::invalid_argument("restore_tensors: undefined or duplicate target");
    const auto it = values.find(name);
    if (it == values.end() || it->second.dtype() != target.dtype() ||
        it->second.shape() != target.shape())
      throw std::invalid_argument("restore_tensors: incompatible tensor " + name);
    // Stage deep copies as targets may alias source tensors or one another.
    staged.emplace_back(name, it->second.detach().clone());
  }
  if (strict && values.size() != names.size())
    throw std::invalid_argument("restore_tensors: extra source names");
  NoGradGuard guard;
  for (std::size_t i = 0; i < targets.size(); ++i) {
    auto target = targets[i].second;
    target.copy_(staged[i].second);
  }
}

/// @brief Encode arbitrary byte strings as numeric tensors without a string dtype.
/// @param words Strings, including empty, UTF-8 and embedded NUL bytes.
/// @return Entries "bytes" (Int32 concatenated bytes) and "offsets" (Int64
///         cumulative positions, length words.size()+1); both rank one.
/// @par Example
/// @code
/// const auto arrays=dm::nn::encode_strings({"café",std::string("a\0b",3)});
/// const auto words=dm::nn::decode_strings(arrays);
/// @endcode
/// @see decode_strings(), save_npz()
inline NamedTensors encode_strings(const std::vector<std::string>& words) {
  std::vector<std::int32_t> bytes;
  std::vector<std::int64_t> offsets{0};
  for (const auto& word : words) {
    for (unsigned char c : word)
      bytes.push_back(c);
    offsets.push_back(static_cast<std::int64_t>(bytes.size()));
  }
  return {
      {"bytes", from_vector(bytes, {static_cast<std::int64_t>(bytes.size())})},
      {"offsets", from_vector(offsets, {static_cast<std::int64_t>(offsets.size())})}};
}

/// @brief Decode checked byte-string arrays produced by encode_strings().
/// @param arrays Exactly two unique entries named bytes and offsets, with the
///        ranks/dtypes above. Offsets start at zero, are nondecreasing and end
///        at byte count; byte values must be in 0..255.
/// @return Owned original strings, preserving all bytes; no UTF-8 validation.
/// @throws std::invalid_argument On invalid schema, offsets or byte values.
/// @par Example
/// @code
/// const auto words=dm::nn::decode_strings(dm::nn::encode_strings({"", "abc"}));
/// @endcode
/// @see encode_strings()
inline std::vector<std::string> decode_strings(const NamedTensors& arrays) {
  Tensor b, o;
  for (const auto& [name, value] : arrays) {
    if (name == "bytes" && !b.defined())
      b = value;
    else if (name == "offsets" && !o.defined())
      o = value;
    else
      throw std::invalid_argument("decode_strings: unexpected or duplicate name");
  }
  if (!b.defined() || !o.defined() || b.dtype() != DType::Int32 || o.dtype() != DType::Int64 ||
      b.dim() != 1 || o.dim() != 1 || o.numel() < 1)
    throw std::invalid_argument("decode_strings: invalid array schema");
  const auto bytes   = b.to_vector<std::int32_t>();
  const auto offsets = o.to_vector<std::int64_t>();
  if (offsets.front() != 0 || offsets.back() != static_cast<std::int64_t>(bytes.size()))
    throw std::invalid_argument("decode_strings: invalid endpoint offsets");
  for (auto v : bytes)
    if (v < 0 || v > 255)
      throw std::invalid_argument("decode_strings: invalid byte");
  std::vector<std::string> out;
  for (std::size_t i = 1; i < offsets.size(); ++i) {
    if (offsets[i] < offsets[i - 1] || offsets[i] > static_cast<std::int64_t>(bytes.size()))
      throw std::invalid_argument("decode_strings: invalid offsets");
    std::string word;
    for (auto j = offsets[i - 1]; j < offsets[i]; ++j)
      word.push_back(static_cast<char>(bytes[j]));
    out.push_back(std::move(word));
  }
  return out;
}

/// @brief Snapshot store parameters reachable from executed eager outputs.
///
/// Uses defined reverse-mode VJPs to detect graph connectivity, not nonzero
/// derivative values. This observes only the executed branch; it is not static
/// control-flow analysis or GraphDef export. Retains the original graph.
/// @param store Source named floating parameters, unchanged.
/// @param outputs Nonempty list of already evaluated, differentiable outputs.
/// @return Detached deep copies in store order; no optimizer state or code.
/// @throws std::invalid_argument On empty outputs or invalid autograd inputs.
/// @par Example
/// @code
/// dm::nn::ParameterStore s;
/// const auto w=s.get("w",{},dm::nn::init::ones());
/// const auto snapshot=dm::nn::inference_parameters(s,{w*2});
/// @endcode
/// @see save_npz(), tape()
inline NamedTensors inference_parameters(
    const ParameterStore&      store,
    const std::vector<Tensor>& outputs) {
  if (outputs.empty())
    throw std::invalid_argument("inference_parameters: no outputs");
  const auto        params = store.parameters();
  std::vector<bool> used(params.size(), false);
  GradOptions       options;
  options.allow_unused = true;
  options.retain_graph = true;
  for (const auto& output : outputs) {
    const auto gradients = grad(sum(output), params, options);
    for (std::size_t i = 0; i < used.size(); ++i)
      used[i] = used[i] || gradients[i].defined();
  }
  NamedTensors out;
  for (std::size_t i = 0; i < used.size(); ++i)
    if (used[i])
      out.emplace_back(store.names()[i], params[i].detach().clone());
  return out;
}

/// @brief Evaluate a typed affine serving contract with a dynamic batch dimension.
/// @param features Floating [batch,input_width] tensor.
/// @param weights [input_width,output_width], same floating dtype as features.
/// @param bias [output_width], same dtype as features.
/// @return Newly allocated [batch,output_width] predictions without autograd history.
/// @throws std::invalid_argument On dtype or shape mismatch.
/// @par Example
/// @code
/// const auto
/// y=dm::nn::affine_inference(dm::nn::tensor({{2,-3}}),dm::nn::tensor({.75,-1.5},{2,1}),dm::nn::tensor({.25}));
/// // 6.25
/// @endcode
/// @see load_npz(), restore_tensors()
inline Tensor affine_inference(const Tensor& features, const Tensor& weights, const Tensor& bias) {
  if (!features.defined() || !weights.defined() || !bias.defined() ||
      !is_floating(features.dtype()) || features.dtype() != weights.dtype() ||
      features.dtype() != bias.dtype() || features.dim() != 2 || weights.dim() != 2 ||
      bias.dim() != 1 || features.size(1) != weights.size(0) || weights.size(1) != bias.size(0))
    throw std::invalid_argument("affine_inference: incompatible serving contract");
  NoGradGuard guard;
  return matmul(features, weights) + bias;
}

} // namespace datamunge::nn
