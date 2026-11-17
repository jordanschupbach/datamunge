#pragma once
/// @file memory_networks.hpp
/// @brief Native source-preset differentiable addressing, copy controller and two-hop question
/// answering.
#include <datamunge/nn/recurrent_components.hpp>
namespace datamunge::nn::models {
/// @brief Internal geometry and column-slice helpers for the fixed memory presets.
namespace memory_network_detail {
/// @brief Validate a finite floating CPU tensor of the exact preset shape.
/// @param x Candidate floating tensor.
/// @param shape Required positive geometry.
/// @throws std::invalid_argument Invalid CPU dtype, finite value or shape.
inline void Geometry(const Tensor& x, const Shape& shape) {
  recurrent_component_detail::Floating(x);
  if (x.shape() != shape)
    throw std::invalid_argument("memory network shape");
}
/// @brief Slice columns of a validated rank-two tensor, retaining derivatives.
/// @param x Floating CPU[N,D].
/// @param begin Nonnegative first column.
/// @param width Positive number of columns within D.
/// @return Owning live CPU[N,width] slice.
inline Tensor Cols(const Tensor& x, std::int64_t begin, std::int64_t width) {
  return slice(x, {0, begin}, {-1, width});
}
} // namespace memory_network_detail
/// @brief Own the flat live NtmAddressOutput tensor endpoints.
///
/// Handles own storage and retain applicable first/higher paths after input/store collection.
/// No fitting, mutable cache or optimizer. Family-inapplicable fields are undefined.
/// @par Example
/// @code
/// dm::nn::models::NtmAddressOutput value;
/// @endcode
/// @see NtmAddressing(), AdjacentMemoryNetwork()
/// @par References
/// - Graves et al. (2014), Neural Turing Machines, https://arxiv.org/abs/1410.5401 ; Sukhbaatar et
/// al. (2015), End-To-End Memory Networks, https://arxiv.org/abs/1503.08895 .
struct NtmAddressOutput {
  Tensor content;      ///< CPU[N,4] content softmax.
  Tensor interpolated; ///< CPU[N,4] gated content/previous mixture.
  Tensor shifted;      ///< CPU[N,4] three-tap circular convolution.
  Tensor weights;      ///< CPU[N,4] sharpened normalized addressing weights.
};
/// @brief Own the flat live NtmState tensor endpoints.
///
/// Handles own storage and retain applicable first/higher paths after input/store collection.
/// No fitting, mutable cache or optimizer. Family-inapplicable fields are undefined.
/// @par Example
/// @code
/// dm::nn::models::NtmState value;
/// @endcode
/// @see NtmAddressing(), AdjacentMemoryNetwork()
/// @par References
/// - Graves et al. (2014), Neural Turing Machines, https://arxiv.org/abs/1410.5401 ; Sukhbaatar et
/// al. (2015), End-To-End Memory Networks, https://arxiv.org/abs/1503.08895 .
struct NtmState {
  Tensor h;             ///< CPU[N,8] controller hidden state.
  Tensor c;             ///< CPU[N,8] controller cell memory.
  Tensor memory;        ///< CPU[N,4,4] memory slots.
  Tensor read;          ///< CPU[N,4] previous read vector.
  Tensor read_weights;  ///< CPU[N,4] previous read-head distribution.
  Tensor write_weights; ///< CPU[N,4] previous write-head distribution.
};
/// @brief Own the flat live NtmCellOutput tensor endpoints.
///
/// Handles own storage and retain applicable first/higher paths after input/store collection.
/// No fitting, mutable cache or optimizer. Family-inapplicable fields are undefined.
/// @par Example
/// @code
/// dm::nn::models::NtmCellOutput value;
/// @endcode
/// @see NtmAddressing(), AdjacentMemoryNetwork()
/// @par References
/// - Graves et al. (2014), Neural Turing Machines, https://arxiv.org/abs/1410.5401 ; Sukhbaatar et
/// al. (2015), End-To-End Memory Networks, https://arxiv.org/abs/1503.08895 .
struct NtmCellOutput {
  Tensor h;             ///< CPU[N,8] next hidden state.
  Tensor c;             ///< CPU[N,8] next cell memory.
  Tensor memory;        ///< CPU[N,4,4] written memory.
  Tensor read;          ///< CPU[N,4] next read vector.
  Tensor read_weights;  ///< CPU[N,4] next read distribution.
  Tensor write_weights; ///< CPU[N,4] next write distribution.
  Tensor logits;        ///< CPU[N,3] binary copy scores.
  Tensor interface;     ///< CPU[N,28] live controller interface.
};
/// @brief Own the flat live NtmSequenceOutput tensor endpoints.
///
/// Handles own storage and retain applicable first/higher paths after input/store collection.
/// No fitting, mutable cache or optimizer. Family-inapplicable fields are undefined.
/// @par Example
/// @code
/// dm::nn::models::NtmSequenceOutput value;
/// @endcode
/// @see NtmAddressing(), AdjacentMemoryNetwork()
/// @par References
/// - Graves et al. (2014), Neural Turing Machines, https://arxiv.org/abs/1410.5401 ; Sukhbaatar et
/// al. (2015), End-To-End Memory Networks, https://arxiv.org/abs/1503.08895 .
struct NtmSequenceOutput {
  Tensor logits;        ///< CPU[N,9,3] copy scores; invalid steps zero.
  Tensor h;             ///< CPU[N,10,8] hidden history.
  Tensor c;             ///< CPU[N,10,8] memory history.
  Tensor memory;        ///< CPU[N,10,4,4] slot history.
  Tensor read;          ///< CPU[N,10,4] read history.
  Tensor read_weights;  ///< CPU[N,10,4] read-head history.
  Tensor write_weights; ///< CPU[N,10,4] write-head history.
  Tensor interface;     ///< CPU[N,9,28] controller interfaces; invalid steps zero.
};
/// @brief Own the flat live QaMemoryOutput tensor endpoints.
///
/// Handles own storage and retain applicable first/higher paths after input/store collection.
/// No fitting, mutable cache or optimizer. Family-inapplicable fields are undefined.
/// @par Example
/// @code
/// dm::nn::models::QaMemoryOutput value;
/// @endcode
/// @see NtmAddressing(), AdjacentMemoryNetwork()
/// @par References
/// - Graves et al. (2014), Neural Turing Machines, https://arxiv.org/abs/1410.5401 ; Sukhbaatar et
/// al. (2015), End-To-End Memory Networks, https://arxiv.org/abs/1503.08895 .
struct QaMemoryOutput {
  Tensor logits;    ///< CPU[N,4] answer scores.
  Tensor states;    ///< CPU[N,3,8] query and two updated states.
  Tensor attention; ///< CPU[N,2,6] hop weights; empty memories all zero.
  Tensor reads;     ///< CPU[N,2,8] read vectors.
  Tensor m0;        ///< CPU[N,6,8] first adjacent embedding; undefined for key-value.
  Tensor m1;        ///< CPU[N,6,8] shared middle embedding; undefined for key-value.
  Tensor m2;        ///< CPU[N,6,8] answer-tied embedding; undefined for key-value.
  Tensor keys;      ///< CPU[N,6,8] key vectors; undefined for adjacent embedding.
  Tensor values;    ///< CPU[N,6,8] value vectors; undefined for adjacent embedding.
};
/// @brief Compute content, interpolation, circular shift and sharpening.
///
/// Source epsilon1e-6 regularizes cosine norms;1e-8 regularizes sharpening. No mutation, fitting or
/// RNG; O(N*16). Native nonsmooth conventions apply at norm extrema.
///
/// @param memory Finite floating CPU[N,4,4] memory slots.
/// @param parameters Same-dtype finite CPU[N,10]:key4,beta1,gate1,shift3,gamma1.
/// @param previous Same-dtype nonnegative finite CPU[N,4] previous address; not implicitly
/// normalized.
/// @return Owning live four-stage address record.
/// @throws std::invalid_argument Invalid finite CPU dtype, geometry or option domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto x=n::zeros({1,4,4});auto
/// y=m::NtmAddressing(x,n::zeros({1,10}),n::full({1,4},.25));
/// @endcode
/// @par References
/// - Graves, Wayne and Danihelka (2014), Neural Turing Machines, https://arxiv.org/abs/1410.5401.
/// This compact source preset uses four4-wide slots and one read/write head.
/// @see MaskedTanhSequence()
inline NtmAddressOutput
NtmAddressing(const Tensor& memory, const Tensor& parameters, const Tensor& previous) {
  recurrent_component_detail::Floating(memory);
  if (memory.dim() != 3)
    throw std::invalid_argument("address memory rank");
  auto rows = memory.size(0);
  memory_network_detail::Geometry(memory, {rows, 4, 4});
  memory_network_detail::Geometry(parameters, {rows, 10});
  memory_network_detail::Geometry(previous, {rows, 4});
  if (parameters.dtype() != memory.dtype() || previous.dtype() != memory.dtype())
    throw std::invalid_argument("address dtype");
  for (auto v : previous.to_vector<double>())
    if (v < 0)
      throw std::invalid_argument("negative previous address");

  auto part = [&](std::int64_t b, std::int64_t w) {
    return memory_network_detail::Cols(parameters, b, w);
  };
  const Tensor key          = part(0, 4);
  const Tensor dot          = sum(mul(memory, expand_dims(key, 1)), {2});
  const Tensor mn           = sqrt(add(sum(square(memory), {2}), 1e-6));
  const Tensor kn           = sqrt(add(sum(square(key), {1}, true), 1e-6));
  const Tensor cosine       = div(dot, mul(mn, kn));
  const Tensor beta         = softplus(part(4, 1));
  const Tensor content      = softmax(mul(beta, cosine), -1);
  const Tensor gate         = sigmoid(part(5, 1));
  const Tensor interpolated = add(mul(gate, content), mul(sub(1.0, gate), previous));
  const Tensor shift        = softmax(part(6, 3), -1);
  const Tensor left         = concat(
      {memory_network_detail::Cols(interpolated, 1, 3),
               memory_network_detail::Cols(interpolated, 0, 1)},
      1);
  const Tensor right = concat(
      {memory_network_detail::Cols(interpolated, 3, 1),
       memory_network_detail::Cols(interpolated, 0, 3)},
      1);
  const Tensor shifted =
      add(add(mul(left, memory_network_detail::Cols(shift, 0, 1)),
              mul(interpolated, memory_network_detail::Cols(shift, 1, 1))),
          mul(right, memory_network_detail::Cols(shift, 2, 1)));
  const Tensor gamma     = add(softplus(part(9, 1)), 1.0);
  const Tensor sharpened = exp(mul(gamma, log(add(shifted, 1e-8))));
  const Tensor weights   = div(sharpened, sum(sharpened, {1}, true));
  return {content, interpolated, shifted, weights};
}
/// @brief Advance one explicit source NTM controller and memory state.
///
/// Writes erase/add memory, then reads the written memory using an address computed from previous
/// memory. No optimizer, RNG bank, cached state or fitting. Warm store immutable; synchronize leaf
/// creation/writes; failed initialization can leave earlier leaves. Work O(N*8*40).
///
/// @param p Caller Float32/64 store; missing controller/interface/output leaves initialized in
/// source order.
/// @param input Finite CPU[N,4] source input, matching store dtype.
/// @param previous Explicit finite matching hidden/cell, slots, read and nonnegative previous-head
/// states.
/// @return Flat owning live state, copy logits and interface.
/// @throws std::invalid_argument Invalid finite CPU dtype, geometry or option domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);auto
/// a=n::full({1,4},.25);m::NtmState
/// v{n::zeros({1,8}),n::zeros({1,8}),n::zeros({1,4,4}),n::zeros({1,4}),a,a};auto
/// y=m::NtmCell(p,n::zeros({1,4}),v);
/// @endcode
/// @par References
/// - Graves, Wayne and Danihelka (2014), Neural Turing Machines, https://arxiv.org/abs/1410.5401.
/// This compact source preset uses four4-wide slots and one read/write head.
/// @see MaskedTanhSequence()
inline NtmCellOutput NtmCell(ParameterStore& p, const Tensor& input, const NtmState& previous) {
  recurrent_component_detail::Floating(input);
  if (input.dim() != 2)
    throw std::invalid_argument("copy input rank");
  auto rows = input.size(0);
  memory_network_detail::Geometry(input, {rows, 4});
  for (const auto& state : std::vector<Tensor>{previous.h, previous.c})
    memory_network_detail::Geometry(state, {rows, 8});
  memory_network_detail::Geometry(previous.memory, {rows, 4, 4});
  for (const auto& state :
       std::vector<Tensor>{previous.read, previous.read_weights, previous.write_weights})
    memory_network_detail::Geometry(state, {rows, 4});
  if (input.dtype() != p.dtype())
    throw std::invalid_argument("copy store dtype");
  for (const auto& state : std::vector<Tensor>{
           previous.h,
           previous.c,
           previous.memory,
           previous.read,
           previous.read_weights,
           previous.write_weights})
    if (state.dtype() != input.dtype())
      throw std::invalid_argument("copy state dtype");

  const auto controller =
      lstm_step(p, "controller", concat({input, previous.read}, 1), {previous.h, previous.c});
  const Tensor interface = dense(p, "interface", controller.h, 28);
  const auto   write     = NtmAddressing(
      previous.memory, memory_network_detail::Cols(interface, 0, 10), previous.write_weights);
  const auto read = NtmAddressing(
      previous.memory, memory_network_detail::Cols(interface, 10, 10), previous.read_weights);
  const Tensor erase    = sigmoid(memory_network_detail::Cols(interface, 20, 4));
  const Tensor addition = tanh(memory_network_detail::Cols(interface, 24, 4));
  const Tensor weights  = expand_dims(write.weights, 2);
  const Tensor memory =
      add(mul(previous.memory, sub(1.0, mul(weights, expand_dims(erase, 1)))),
          mul(weights, expand_dims(addition, 1)));
  const Tensor value  = sum(mul(memory, expand_dims(read.weights, 2)), {1});
  const Tensor logits = dense(p, "output", concat({controller.h, value}, 1), 3);
  return {
      controller.h, controller.c, memory, value, read.weights, write.weights, logits, interface};
}
/// @brief Unroll the nine-step source NTM copy preset.
///
/// Zero controller state, learned initial slots and slot0 initial heads are explicit source
/// conventions. No fitting/optimizer/objective; sequence caller owns training. All histories
/// include the initial state. Work O(9*N*8*40). Warm store immutable; synchronize missing
/// leaves/writes.
///
/// @param p Caller-owned Float32 source store; learned initial4x4 memory plus controller leaves.
/// @param x Finite Float32 CPU[N,9,4] copy input.
/// @param lengths Int32/64 CPU[N] copy lengths in[0,4]; steps through2*length inclusive remain
/// valid.
/// @return Flat live logits, interfaces and all state histories.
/// @throws std::invalid_argument Invalid finite CPU dtype, geometry or option domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);auto
/// y=m::NtmSequence(p,n::zeros({1,9,4}),n::zeros({1},n::DType::Int32));
/// @endcode
/// @par References
/// - Graves, Wayne and Danihelka (2014), Neural Turing Machines, https://arxiv.org/abs/1410.5401.
/// This compact source preset uses four4-wide slots and one read/write head.
/// @see MaskedTanhSequence()
inline NtmSequenceOutput NtmSequence(ParameterStore& p, const Tensor& x, const Tensor& lengths) {
  recurrent_component_detail::Floating(x);
  if (x.dim() != 3)
    throw std::invalid_argument("copy sequence rank");
  memory_network_detail::Geometry(x, {x.size(0), 9, 4});
  recurrent_component_detail::Lengths(lengths, x.size(0), 4);
  if (x.dtype() != DType::Float32 || p.dtype() != DType::Float32)
    throw std::invalid_argument("copy sequence source Float32 preset");

  const std::int64_t batch   = x.shape()[0];
  const Tensor       initial = p.get("memory/initial", {4, 4}, init::normal(.01));
  const Tensor       memory  = add(zeros({batch, 4, 4}), initial);
  const Tensor       weights =
      add(zeros({batch, 4}), from_vector(std::vector<float>{1, 0, 0, 0}, {1, 4}));
  const Tensor        read = sum(mul(memory, expand_dims(weights, 2)), {1});
  NtmState            state{zeros({batch, 8}), zeros({batch, 8}), memory, read, weights, weights};
  std::vector<Tensor> hs{state.h}, cs{state.c}, ms{memory}, rs{read}, rw{weights}, ww{weights},
      logits, interfaces;
  const auto steps = time_steps(x);
  for (int t = 0; t < 9; ++t) {
    const auto   next = NtmCell(p, steps[t], state);
    const Tensor valid =
        expand_dims(greater(add(mul(cast(lengths, DType::Float32), 2.0), 1.0), double(t)), 1);
    auto choose = [&](const Tensor& a, const Tensor& b) { return where(valid, a, b); };
    state       = {
        choose(next.h, state.h),
        choose(next.c, state.c),
        where(expand_dims(valid, 2), next.memory, state.memory),
        choose(next.read, state.read),
        choose(next.read_weights, state.read_weights),
        choose(next.write_weights, state.write_weights)};
    logits.push_back(choose(next.logits, zeros_like(next.logits)));
    interfaces.push_back(choose(next.interface, zeros_like(next.interface)));
    hs.push_back(state.h);
    cs.push_back(state.c);
    ms.push_back(state.memory);
    rs.push_back(state.read);
    rw.push_back(state.read_weights);
    ww.push_back(state.write_weights);
  }
  NtmSequenceOutput result;
  result.logits        = stack_time(logits);
  result.h             = stack_time(hs);
  result.c             = stack_time(cs);
  result.memory        = stack(ms, 1);
  result.read          = stack_time(rs);
  result.read_weights  = stack_time(rw);
  result.write_weights = stack_time(ww);
  result.interface     = stack_time(interfaces);
  return result;
}
/// @brief Compute source HopWeights structural attention data.
///
/// Compact six-fact source primitive. Empty memory reads zero; no fitting or mutation.
///
/// @param scores Finite CPU[N,6] attention scores.
/// @param valid Boolean matching valid-slot mask.
/// @return Live normalized weights or detached valid mask.
/// @throws std::invalid_argument Invalid finite CPU dtype, geometry or option domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto v=m::FactValid(n::zeros({1},n::DType::Int32));auto
/// w=m::HopWeights(n::zeros({1,6}),v);
/// @endcode
/// @par References
/// - Sukhbaatar et al. (2015), End-To-End Memory Networks, https://arxiv.org/abs/1503.08895.
/// @see MaskedTanhSequence()
inline Tensor HopWeights(const Tensor& scores, const Tensor& valid) {
  recurrent_component_detail::Floating(scores);
  if (scores.dim() != 2 || scores.size(1) != 6 || !valid.defined() ||
      valid.shape() != scores.shape() || valid.dtype() != DType::Bool ||
      valid.device().backend != compute::Backend::CPU)
    throw std::invalid_argument("memory attention scores/mask");
  const Tensor masked = where(valid, scores, add(mul(scores, 0.0), -1e9));
  const Tensor raw    = mul(softmax(masked, -1), cast(valid, DType::Float32));
  return div(raw, maximum(sum(raw, {1}, true), full({1}, 1.0)));
}
/// @brief Compute source FactValid structural attention data.
///
/// Compact six-fact source primitive. Empty memory reads zero; no fitting or mutation.
///
/// @param lengths Validated integer CPU[N] lengths in[0,6].
/// @return Live normalized weights or detached valid mask.
/// @throws std::invalid_argument Invalid finite CPU dtype, geometry or option domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto v=m::FactValid(n::zeros({1},n::DType::Int32));
/// @endcode
/// @par References
/// - Sukhbaatar et al. (2015), End-To-End Memory Networks, https://arxiv.org/abs/1503.08895.
/// @see MaskedTanhSequence()
inline Tensor FactValid(const Tensor& lengths) {
  if (!lengths.defined() || lengths.dim() != 1 || lengths.size(0) < 1)
    throw std::invalid_argument("fact lengths rank");
  recurrent_component_detail::Lengths(lengths, lengths.size(0), 6);
  return less(
      expand_dims(arange(0, 6, 1, DType::Float32), 0),
      expand_dims(cast(lengths, DType::Float32), 1));
}
/// @brief Evaluate the source two-hop AdjacentMemoryNetwork preset.
///
/// No labels, fitting, optimizer or caches. Empty fact sets read exactly zero; padding facts cannot
/// affect outputs or derivatives. Adjacent embedding uses312values; key-value uses264 including
/// identity-initialized hop rotations. Warm store immutable; synchronize missing leaves/writes;
/// initialization failure may leave earlier leaves. First/higher paths retained. Work
/// O(N*6*2*13*8).
///
/// @param p Caller-owned Float32 source store; missing leaves created in source order.
/// @param facts Finite Float32 CPU[N,6,2,13] encoded facts; continuous inputs allowed for
/// derivatives.
/// @param query Finite Float32 CPU[N,13] encoded query.
/// @param lengths Int32/64 CPU[N] valid facts in[0,6].
/// @return Flat live answer logits, hop states, weights, reads and family embeddings.
/// @throws std::invalid_argument Invalid finite CPU dtype, geometry or option domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);auto
/// y=m::AdjacentMemoryNetwork(p,n::zeros({1,6,2,13}),n::zeros({1,13}),n::zeros({1},n::DType::Int32));
/// @endcode
/// @par References
/// - Sukhbaatar et al. (2015), End-To-End Memory Networks, https://arxiv.org/abs/1503.08895.
/// @see MaskedTanhSequence()
inline QaMemoryOutput AdjacentMemoryNetwork(
    ParameterStore& p,
    const Tensor&   facts,
    const Tensor&   query,
    const Tensor&   lengths) {
  recurrent_component_detail::Floating(facts);
  if (facts.dim() != 4)
    throw std::invalid_argument("memory fact rank");
  auto rows = facts.size(0);
  memory_network_detail::Geometry(facts, {rows, 6, 2, 13});
  memory_network_detail::Geometry(query, {rows, 13});
  recurrent_component_detail::Lengths(lengths, rows, 6);
  if (facts.dtype() != DType::Float32 || query.dtype() != DType::Float32 ||
      p.dtype() != DType::Float32)
    throw std::invalid_argument("memory QA source Float32 preset");

  std::vector<float> positions(16);
  for (int j = 0; j < 2; ++j)
    for (int k = 0; k < 8; ++k)
      positions[j * 8 + k] = j == 0 ? .5f : (k + 1) / 8.f;
  const Tensor        pos   = from_vector(positions, {1, 1, 2, 8});
  const Tensor        valid = FactValid(lengths);
  std::vector<Tensor> tables, memory;
  for (int i = 0; i < 3; ++i) {
    const Tensor table = p.get("embedding" + std::to_string(i) + "/W", {13, 8}, init::normal(.1));
    tables.push_back(table);
    const Tensor embedded = reshape(matmul(reshape(facts, {-1, 13}), table), {-1, 6, 2, 8});
    const Tensor sentence = sum(mul(embedded, pos), {2});
    memory.push_back(where(expand_dims(valid, 2), sentence, zeros_like(sentence)));
  }
  Tensor              u = matmul(query, tables[0]);
  std::vector<Tensor> states{u}, attention, reads;
  for (int hop = 0; hop < 2; ++hop) {
    const Tensor weights = HopWeights(sum(mul(memory[hop], expand_dims(u, 1)), {2}), valid);
    const Tensor read    = sum(mul(memory[hop + 1], expand_dims(weights, 2)), {1});
    u                    = add(u, read);
    states.push_back(u);
    attention.push_back(weights);
    reads.push_back(read);
  }
  const Tensor   answers = slice(tables[2], {8, 0}, {4, 8});
  QaMemoryOutput result;
  result.logits    = matmul(u, answers, false, true);
  result.states    = stack_time(states);
  result.attention = stack_time(attention);
  result.reads     = stack_time(reads);
  result.m0        = memory[0];
  result.m1        = memory[1];
  result.m2        = memory[2];
  return result;
}
/// @brief Evaluate the source two-hop KeyValueMemoryNetwork preset.
///
/// No labels, fitting, optimizer or caches. Empty fact sets read exactly zero; padding facts cannot
/// affect outputs or derivatives. Adjacent embedding uses312values; key-value uses264 including
/// identity-initialized hop rotations. Warm store immutable; synchronize missing leaves/writes;
/// initialization failure may leave earlier leaves. First/higher paths retained. Work
/// O(N*6*2*13*8).
///
/// @param p Caller-owned Float32 source store; missing leaves created in source order.
/// @param facts Finite Float32 CPU[N,6,2,13] encoded facts; continuous inputs allowed for
/// derivatives.
/// @param query Finite Float32 CPU[N,13] encoded query.
/// @param lengths Int32/64 CPU[N] valid facts in[0,6].
/// @return Flat live answer logits, hop states, weights, reads and family embeddings.
/// @throws std::invalid_argument Invalid finite CPU dtype, geometry or option domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(1);auto
/// y=m::KeyValueMemoryNetwork(p,n::zeros({1,6,2,13}),n::zeros({1,13}),n::zeros({1},n::DType::Int32));
/// @endcode
/// @par References
/// - Miller et al. (2016), Key-Value Memory Networks for Directly Reading Documents,
/// https://arxiv.org/abs/1606.03126.
/// @see MaskedTanhSequence()
inline QaMemoryOutput KeyValueMemoryNetwork(
    ParameterStore& p,
    const Tensor&   facts,
    const Tensor&   query,
    const Tensor&   lengths) {
  recurrent_component_detail::Floating(facts);
  if (facts.dim() != 4)
    throw std::invalid_argument("memory fact rank");
  auto rows = facts.size(0);
  memory_network_detail::Geometry(facts, {rows, 6, 2, 13});
  memory_network_detail::Geometry(query, {rows, 13});
  recurrent_component_detail::Lengths(lengths, rows, 6);
  if (facts.dtype() != DType::Float32 || query.dtype() != DType::Float32 ||
      p.dtype() != DType::Float32)
    throw std::invalid_argument("memory QA source Float32 preset");

  const Tensor embedding = p.get("embedding/A", {13, 8}, init::normal(.1));
  const Tensor answers   = p.get("answers/B", {4, 8}, init::normal(.1));
  const Tensor valid     = FactValid(lengths);
  auto         encode    = [&](std::int64_t role) {
    const Tensor onehot  = reshape(slice(facts, {0, 0, role, 0}, {-1, 6, 1, 13}), {-1, 13});
    const Tensor vectors = reshape(matmul(onehot, embedding), {-1, 6, 8});
    return where(expand_dims(valid, 2), vectors, zeros_like(vectors));
  };
  const Tensor        keys = encode(0), values = encode(1);
  Tensor              u = matmul(query, embedding);
  std::vector<Tensor> states{u}, attention, reads;
  for (int hop = 0; hop < 2; ++hop) {
    const Tensor weights = HopWeights(sum(mul(keys, expand_dims(u, 1)), {2}), valid);
    const Tensor read    = sum(mul(values, expand_dims(weights, 2)), {1});
    const Tensor rotation =
        p.get("hop" + std::to_string(hop) + "/R", {8, 8}, [](const Shape&, DType d, Generator&) {
          return eye(8, d);
        });
    u = matmul(add(u, read), rotation);
    states.push_back(u);
    attention.push_back(weights);
    reads.push_back(read);
  }
  QaMemoryOutput result;
  result.logits    = matmul(u, answers, false, true);
  result.states    = stack_time(states);
  result.attention = stack_time(attention);
  result.reads     = stack_time(reads);
  result.keys      = keys;
  result.values    = values;
  return result;
}
} // namespace datamunge::nn::models
