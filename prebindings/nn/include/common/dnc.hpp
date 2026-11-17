// Differentiable neural computer allocation, freeing, temporal links and memory
// access. Port of tflow's common/nn/dnc.h (4 slots x 4 wide memory, LSTM(8)
// controller, one write head with allocation + content addressing, one read
// head mixing backward / forward temporal links and content; 799 parameters).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/ntm.hpp"

namespace memory_nets {

/// Softmax over slots of softplus(strength) * cosine(memory rows, key), with
/// tflow's 1e-6 guards. memory [B, 4, 4], key [B, 4], strength [B, 1].
using nn::models::DncContent;

/// Allocation weights from usage [B, 4]: slots sorted by (stop-gradient) usage
/// with stable ties, a_sorted[j] = (1 - u_j) prod_{k<j} u_k, mapped back to
/// slot order. Gradients flow through the sorted usage values only.
using nn::models::DncAllocation;

using DncLink=nn::models::DncLink;

/// L' = (1 - w_i - w_j) L + w_i p_j without the diagonal; p' = (1 - sum w) p + w.
using nn::models::DncTemporal;

using DncState=nn::models::DncState;

using nn::models::DncNames;

using DncAccess=nn::models::DncAccess;

/// Memory access from a 24-wide interface: write key(4) strength(1) erase(4)
/// add(4) allocation gate(1) write gate(1) free gate(1), read key(4)
/// strength(1) read modes(3).
using nn::models::DncMemory;

using DncStep=nn::models::DncStep;

using nn::models::DncCell;

/// Nine-step unroll from a zero state with valid steps t <= 2 * length
/// (tflow's nn::Dnc). Fields: logits, interface, allocation, forward, backward
/// [B, 9, .] (zero on invalid steps) and the nine state traces [B, 10, ...].
inline Outputs Dnc(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths) {
  return nn::models::DncSequence(p,input,lengths);
}

/// tflow's dnc Model: seed 45203, Adam .005.
inline std::unique_ptr<MemoryModel> DncModel(bool training, const Batch& example) { return CopyModel(45203, Dnc, training, example); }

}  // namespace memory_nets
