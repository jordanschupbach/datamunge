// Portable bindings for the memory-network model components. NamedTensors
// already has native Python/R marshalling in the NN API; use it at the boundary
// instead of exposing std::function callbacks or std::map proxy objects.
// No optimizer or experiment loop lives in this adapter.
#pragma once

#include "common/dnc.hpp"
#include "common/memnet.hpp"
#include "common/memory_data.hpp"
#include "common/memory_validation.hpp"
#include "common/ntm.hpp"

namespace memory_components {
namespace nn = datamunge::nn;

enum class Kind { Ntm, Dnc, MemNet, KeyValue };

class Model {
 public:
  explicit Model(Kind kind, const nn::NamedTensors& example) : kind_(kind) {
    const auto batch = ToMap(example);
    switch (kind) {
      case Kind::Ntm:
        model_ = memory_nets::NtmModel(false, batch);
        break;
      case Kind::Dnc:
        model_ = memory_nets::DncModel(false, batch);
        break;
      case Kind::MemNet:
        model_ = memory_nets::MemNetModel(false, batch);
        break;
      case Kind::KeyValue:
        model_ = memory_nets::KeyValueModel(false, batch);
        break;
      default:
        throw std::invalid_argument("unknown memory model kind");
    }
  }

  nn::ParameterStore& parameters() { return model_->store; }

  nn::NamedTensors Leaves(const nn::NamedTensors& batch) const {
    return ToNamed(model_->leaves(ToMap(batch)));
  }

  nn::NamedTensors Forward(const nn::NamedTensors& leaves, const nn::NamedTensors& batch) {
    return ToNamed(model_->forward(model_->store, ToMap(leaves), ToMap(batch)));
  }

  nn::Tensor Loss(const nn::NamedTensors& outputs, const nn::NamedTensors& batch) const {
    return model_->loss(ToMap(outputs), ToMap(batch));
  }

  // The canonical one-batch diagnostics, including derivatives of encoded
  // inputs. The language example controls when and on which split to call it.
  nn::NamedTensors Evaluate(const nn::NamedTensors& batch) {
    return ToNamed(model_->Evaluate(ToMap(batch)));
  }

  nn::Tensor       Predict(const nn::NamedTensors& batch) { return model_->Predict(ToMap(batch)); }
  nn::NamedTensors Arrays() const { return model_->Arrays(); }
  void             Load(const std::string& file) { model_->LoadArrays(file); }

  std::vector<std::string> StateNames() const {
    if (kind_ == Kind::Ntm)
      return {"h", "c", "memory", "read", "read_weights", "write_weights"};
    if (kind_ == Kind::Dnc)
      return memory_nets::DncNames();
    throw std::invalid_argument("streaming state is defined for NTM and DNC");
  }
  nn::NamedTensors Step(const nn::Tensor& input, const nn::NamedTensors& previous) {
    nn::NoGradGuard    guard;
    const auto         p = ToMap(previous);
    memory_nets::Batch fields;
    if (kind_ == Kind::Ntm) {
      const memory_nets::NtmState state{
          p.at("h"),
          p.at("c"),
          p.at("memory"),
          p.at("read"),
          p.at("read_weights"),
          p.at("write_weights")};
      const auto step = memory_nets::NtmCell(model_->store, input, state);
      fields          = {
          {"logits", step.logits},
          {"interface", step.interface},
          {"h", step.state.h},
          {"c", step.state.c},
          {"memory", step.state.memory},
          {"read", step.state.read},
          {"read_weights", step.state.read_weights},
          {"write_weights", step.state.write_weights}};
    } else if (kind_ == Kind::Dnc) {
      std::vector<nn::Tensor> values;
      for (const auto& name : memory_nets::DncNames())
        values.push_back(p.at(name));
      const auto step =
          memory_nets::DncCell(model_->store, input, memory_nets::DncState::From(values));
      fields = {
          {"logits", step.logits},
          {"interface", step.interface},
          {"allocation", step.allocation},
          {"forward", step.forward},
          {"backward", step.backward}};
      const auto states = step.state.Values();
      for (std::size_t i = 0; i < states.size(); ++i)
        fields[memory_nets::DncNames()[i]] = states[i];
    } else
      throw std::invalid_argument("streaming step is defined for NTM and DNC");
    return ToNamed(fields);
  }

 private:
  Kind                                      kind_;
  std::unique_ptr<memory_nets::MemoryModel> model_;

  static memory_nets::Batch ToMap(const nn::NamedTensors& records) {
    memory_nets::Batch result;
    for (const auto& [name, tensor] : records) {
      if (!result.emplace(name, tensor).second)
        throw std::invalid_argument("duplicate memory batch field: " + name);
    }
    return result;
  }
  static nn::NamedTensors ToNamed(const memory_nets::Batch& records) {
    return {records.begin(), records.end()};
  }
};
inline void Fixtures(Kind kind, const std::string& directory) {
  if (kind == Kind::Ntm)
    memory_validation::NtmFixtures(directory);
  else if (kind == Kind::Dnc)
    memory_validation::DncFixtures(directory);
}
inline void CompareRecords(
    const std::string&              root,
    const std::string&              tf,
    const std::string&              init,
    const std::vector<std::string>& records,
    const std::string&              test_key) {
  rnn_cells::DemoOptions options;
  options.out  = root;
  options.tf   = tf;
  options.init = init;
  rnn_cells::CompareWithTflow(root, options, records, test_key);
}
inline double CompareNpz(const std::string& first, const std::string& second) {
  return rnn_cells::CompareNpz(first, second);
}
inline void PrepareData(const std::string& directory, bool overwrite = false) {
  memory_data::Prepare(directory, overwrite);
}
inline void PrepareDataset(const std::string& directory, bool copy_task) {
  memory_data::PrepareDataset(directory, copy_task);
}
} // namespace memory_components
