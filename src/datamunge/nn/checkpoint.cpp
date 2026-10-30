/// @file checkpoint.cpp
/// @brief Saving and all-or-nothing restoring of parameters, optimizer slots and batch order.

#include "datamunge/nn/checkpoint.hpp"

#include "datamunge/nn/ops.hpp"

#include <array>
#include <map>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace datamunge::nn {

/// @brief Helpers for checkpoint entry names and slot shapes.
namespace {

constexpr std::string_view kParams = "params/";        ///< Prefix of parameter entries.
constexpr std::string_view kOptimizer = "optimizer/";  ///< Prefix of optimizer-slot entries.
constexpr std::string_view kOrder = "batch_order/state";  ///< Name of the batch-order entry.

/// @brief Throws std::invalid_argument with a "restore_training_state: " prefix.
[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument("restore_training_state: " + message); }

/// @brief Expected shape of optimizer slot `kind` for parameter shape `p`, or throws for an unknown kind.
Shape slot_shape(const std::string& kind, const Shape& p) {
  static constexpr std::array<std::string_view, 9> same{"velocity", "accumulator", "variance", "accum", "accum_update", "m", "u", "v", "vmax"};
  for (auto k : same)
    if (kind == k) return p;
  if (kind == "row" || kind == "col") {
    if (p.size() < 2) fail("slot kind '" + kind + "' needs a parameter of rank >= 2");
    std::int64_t rows = 1;
    for (std::size_t i = 0; i + 1 < p.size(); ++i) rows *= p[i];
    return kind == "row" ? Shape{rows} : Shape{p.back()};
  }
  fail("unknown optimizer slot kind '" + kind + "'");
}

}  // namespace

NamedTensors training_state(const ParameterStore& store, const Optimizer* optimizer, const BatchOrder* order) {
  NamedTensors out;
  for (const auto& name : store.names()) out.emplace_back(std::string(kParams) + name, store.at(name).detach().clone());
  if (optimizer)
    for (const auto& [name, value] : optimizer->state()) out.emplace_back(std::string(kOptimizer) + name, value.clone());
  if (order) {
    const std::string text = order->save_state();
    std::vector<std::int32_t> codes(text.begin(), text.end());
    out.emplace_back(std::string(kOrder), from_vector(codes, {static_cast<std::int64_t>(codes.size())}));
  }
  return out;
}

void restore_training_state(const NamedTensors& state, ParameterStore& store, Optimizer* optimizer, BatchOrder* order) {
  // 1. Sort the entries and check every one before changing anything.
  std::map<std::string, Tensor> params;
  NamedTensors slots;
  const Tensor* order_codes = nullptr;
  bool has_step = false;
  for (const auto& [name, value] : state) {
    if (name.starts_with(kParams)) {
      if (!params.emplace(name.substr(kParams.size()), value).second) fail("duplicate entry '" + name + "'");
    } else if (name.starts_with(kOptimizer)) {
      slots.emplace_back(name.substr(kOptimizer.size()), value);
    } else if (name == kOrder) {
      order_codes = &value;
    } else {
      fail("unexpected entry '" + name + "'");
    }
  }
  for (const auto& name : store.names()) {
    const auto it = params.find(name);
    if (it == params.end()) fail("missing parameter '" + name + "'");
    if (it->second.shape() != store.at(name).shape()) fail("parameter '" + name + "' has the wrong shape");
  }
  if (params.size() != store.size()) fail("checkpoint has parameters the store does not have");
  if (optimizer) {
    const auto& ps = optimizer->parameters();
    for (const auto& [name, value] : slots) {
      if (name == "step") {
        if (value.numel() != 1) fail("optimizer step must be a scalar");
        has_step = true;
        continue;
      }
      if (name == "nadam_product") {
        if (value.numel() != 1) fail("nadam_product must be a scalar");
        continue;
      }
      const auto slash = name.rfind('/');
      if (slash == std::string::npos || slash + 1 == name.size()) fail("bad optimizer slot name '" + name + "'");
      const std::string index_text = name.substr(slash + 1);
      for (char c : index_text)
        if (c < '0' || c > '9') fail("bad optimizer slot index in '" + name + "'");
      const auto index = std::stoull(index_text);
      if (std::to_string(index) != index_text || index >= ps.size()) fail("optimizer slot '" + name + "' is out of range");
      if (value.shape() != slot_shape(name.substr(0, slash), ps[index].shape())) fail("optimizer slot '" + name + "' has the wrong shape");
    }
    if (!has_step) fail("checkpoint has no optimizer state");
  }
  BatchOrder* probe_target = nullptr;
  std::unique_ptr<BatchOrder> probe;
  if (order) {
    if (!order_codes) fail("checkpoint has no batch order");
    const auto codes = cast(*order_codes, DType::Int32).to_vector<std::int32_t>();
    std::string text;
    for (auto c : codes) {
      if (c < 0 || c > 255) fail("batch order text is not a byte string");
      text.push_back(static_cast<char>(c));
    }
    probe = std::make_unique<BatchOrder>(*order);
    probe->restore_state(text);  // throws std::invalid_argument for other settings; *order is untouched
    probe_target = order;
  }
  // 2. Apply. None of these steps can fail for a checkpoint that passed the checks.
  for (const auto& name : store.names()) store.set(name, params.at(name));
  if (optimizer) optimizer->load_state(slots);
  if (probe_target) *probe_target = *probe;
}

void save_training_checkpoint(const std::string& path, const ParameterStore& store, const Optimizer* optimizer, const BatchOrder* order) {
  save_npz(path, training_state(store, optimizer, order));
}

void load_training_checkpoint(const std::string& path, ParameterStore& store, Optimizer* optimizer, BatchOrder* order) {
  restore_training_state(load_npz(path), store, optimizer, order);
}

}  // namespace datamunge::nn
