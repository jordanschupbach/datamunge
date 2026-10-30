#pragma once

/// @file checkpoint.hpp
/// @brief Training checkpoints: parameters, optimizer slots and batch order in one archive, restored all-or-nothing.
///
/// A checkpoint is a list of named tensors (an .npz archive on disk) with three groups of entries:
/// - "params/<name>" for every parameter of a ParameterStore;
/// - "optimizer/<slot>" for every entry of Optimizer::state(), including the int64 step counter;
/// - "batch_order/state", the text of BatchOrder::save_state() stored as int32 character codes.
///
/// restore_training_state() checks the whole checkpoint against the current model first: the parameter names and shapes,
/// each optimizer slot's name, index and shape, and the batch order's settings. Only then does it change anything, so
/// a failed restore leaves the store, the optimizer and the batch order exactly as they were. Resuming from a full
/// checkpoint reproduces an uninterrupted run exactly when training is otherwise deterministic.

#include "datamunge/nn/io.hpp"
#include "datamunge/nn/optim.hpp"
#include "datamunge/nn/params.hpp"
#include "datamunge/nn/training.hpp"

#include <string>

namespace datamunge::nn {

/// @brief Collects the training state as named tensors (deep copies).
/// @param store  Parameters, saved as "params/<name>".
/// @param optimizer  Optional optimizer, saved as "optimizer/<slot>"; nullptr leaves it out.
/// @param order  Optional batch order, saved as "batch_order/state"; nullptr leaves it out.
/// @return The checkpoint entries, parameters first.
[[nodiscard]] NamedTensors training_state(const ParameterStore& store, const Optimizer* optimizer = nullptr,
                                          const BatchOrder* order = nullptr);

/// @brief Restores a training state produced by training_state(), all-or-nothing.
///
/// The checkpoint must hold exactly the store's parameters, with the same shapes. If `optimizer` is given, the
/// checkpoint must hold an "optimizer/step" entry, and every slot "<kind>/<i>" must name a known slot kind, a parameter
/// index of the optimizer and the shape that kind has for that parameter (Adafactor's "row" and "col" are the row and
/// column counts of the parameter viewed as a matrix over its last axis). If `order` is given, the checkpoint must hold
/// a batch order with the same settings. Entries for components that are not passed are ignored, so passing only the
/// store restores the weights alone.
/// @param state  Checkpoint entries.
/// @param store  Parameters to overwrite.
/// @param optimizer  Optional optimizer whose state is replaced.
/// @param order  Optional batch order whose position and random stream are replaced.
/// @throws std::invalid_argument  If the checkpoint does not match; nothing is changed in that case.
void restore_training_state(const NamedTensors& state, ParameterStore& store, Optimizer* optimizer = nullptr,
                            BatchOrder* order = nullptr);

/// @brief Writes training_state() to an .npz archive.
/// @param path  Output file.
/// @param store  Parameters.
/// @param optimizer  Optional optimizer.
/// @param order  Optional batch order.
void save_training_checkpoint(const std::string& path, const ParameterStore& store, const Optimizer* optimizer = nullptr,
                              const BatchOrder* order = nullptr);

/// @brief Reads an .npz archive and restores it with restore_training_state().
/// @param path  Archive written by save_training_checkpoint().
/// @param store  Parameters to overwrite.
/// @param optimizer  Optional optimizer whose state is replaced.
/// @param order  Optional batch order whose state is replaced.
/// @throws std::runtime_error  If the file cannot be read.
/// @throws std::invalid_argument  If the checkpoint does not match; nothing is changed in either case.
void load_training_checkpoint(const std::string& path, ParameterStore& store, Optimizer* optimizer = nullptr,
                              BatchOrder* order = nullptr);

}  // namespace datamunge::nn
