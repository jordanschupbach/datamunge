#pragma once
/// @file sequence_language_graphs.hpp
/// @brief Original word recurrence and masked IMDB/POS/NER sequence graph endpoints.
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/sequence.hpp>
namespace datamunge::nn::models {

/// @brief Source word_lm parameter names, fixed dimensions and live sequence graph components.
namespace word_lm {

namespace nn = datamunge::nn; ///< Native tensor/operation namespace.
using nn::Tensor;             ///< Native shared-storage graph tensor.

constexpr int kVocab = 10000; ///< Original source kVocab dimension.
constexpr int kUnits = 8;     ///< Original source kUnits dimension.
constexpr int kSteps = 16;    ///< Original source kSteps dimension.

/// @brief Create/reuse the six original PTB word-RNN parameter tensors.
///
/// Pure live forward graph, with no objective, optimizer, collection or fitting/generation loop.
/// ParameterStore is borrowed and must outlive dependent graphs; first calls initialize missing
/// source-named leaves in original order, later calls reuse them. Callers synchronize store
/// initialization/mutation. Tensor copies share native storage.
///
/// @param p Borrowed store; source names, widths and normal initializers retained.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);dm::nn::models::word_lm::Parameters(p);
/// @endcode
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 ; Lafferty et al. (2001), Conditional Random Fields,
/// https://repository.upenn.edu/handle/20.500.14332/20034 .
/// @see Sequence()
inline void Parameters(nn::ParameterStore& p) {
  p.get("embedding/W", {kVocab, kUnits}, nn::init::normal(.1));
  p.get("recurrent/Wx", {kUnits, kUnits}, nn::init::normal(.1));
  p.get("recurrent/Wh", {kUnits, kUnits}, nn::init::normal(.15));
  p.get("recurrent/b", {kUnits}, nn::init::zeros());
  p.get("output/W", {kUnits, kVocab}, nn::init::normal(.1));
  p.get("output/b", {kVocab}, nn::init::zeros());
}

/// @brief Live source StepPath endpoints with owning tensor handles.
///
/// Tensor copies share storage and preserve derivative paths. Field names retain the source
/// diagnostic naming and layout.
///
/// @par Example
/// @code
/// dm::nn::models::word_lm::StepPath value;
/// @endcode
/// @see datamunge::nn::Tensor
struct StepPath {
  Tensor hidden; ///< Live hidden endpoint; source shape follows the owning graph.
  Tensor logits; ///< Live logits endpoint; source shape follows the owning graph.
};

/// One word: one-hot [B, 10000] and state [B, 8] -> next state, logits [B, 10000].
/// @brief Build one explicit-state full-softmax word-RNN transition.
///
/// Pure live forward graph, with no objective, optimizer, collection or fitting/generation loop.
/// ParameterStore is borrowed and must outlive dependent graphs; first calls initialize missing
/// source-named leaves in original order, later calls reuse them. Callers synchronize store
/// initialization/mutation. Tensor copies share native storage.
///
/// @param p Borrowed store previously initialized with Parameters().
/// @param onehot Live floating[B,10000] encoded words, including soft inputs.
/// @param state Live floating[B,8] previous hidden state.
/// @return Live hidden[B,8] and logits[B,10000].
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);dm::nn::models::word_lm::Parameters(p);
/// auto path=dm::nn::models::word_lm::Step(p,dm::nn::zeros({1,10000}),dm::nn::zeros({1,8}));
/// @endcode
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 ; Lafferty et al. (2001), Conditional Random Fields,
/// https://repository.upenn.edu/handle/20.500.14332/20034 .
/// @see Sequence()
inline StepPath Step(nn::ParameterStore& p, const Tensor& onehot, const Tensor& state) {
  const Tensor h = nn::rnn_step(p, "recurrent", nn::matmul(onehot, p.at("embedding/W")), state);
  return {h, nn::dense(p, "output", h, kVocab)};
}

/// @brief Live source Path endpoints with owning tensor handles.
///
/// Tensor copies share storage and preserve derivative paths. Field names retain the source
/// diagnostic naming and layout.
///
/// @par Example
/// @code
/// dm::nn::models::word_lm::Path value;
/// @endcode
/// @see datamunge::nn::Tensor
struct Path {
  Tensor logits; ///< Live logits endpoint; source shape follows the owning graph.
  Tensor hidden; ///< Live hidden endpoint; source shape follows the owning graph.  // [B, 16,
                 ///< 10000], [B, 16, 8] (states after each word)
};

/// Full zero-state unroll over one-hot words [B, 16, 10000].
/// @brief Build the original zero-state16-token word recurrence.
///
/// Pure live forward graph, with no objective, optimizer, collection or fitting/generation loop.
/// ParameterStore is borrowed and must outlive dependent graphs; first calls initialize missing
/// source-named leaves in original order, later calls reuse them. Callers synchronize store
/// initialization/mutation. Tensor copies share native storage.
///
/// @param p Borrowed parameter store; source leaves created if missing.
/// @param encoded Live floating[B,16,10000] encoded words.
/// @return Live post-token hidden[B,16,8] and logits[B,16,10000].
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);auto
/// path=dm::nn::models::word_lm::Sequence(p,dm::nn::zeros({1,16,10000}));
/// @endcode
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 ; Lafferty et al. (2001), Conditional Random Fields,
/// https://repository.upenn.edu/handle/20.500.14332/20034 .
/// @see Step()
inline Path Sequence(nn::ParameterStore& p, const Tensor& encoded) {
  Parameters(p);
  Tensor              h = nn::zeros({encoded.shape()[0], kUnits}, encoded.dtype());
  std::vector<Tensor> states, scores;
  for (const auto& x : nn::time_steps(encoded)) {
    const auto step = Step(p, x, h);
    h               = step.hidden;
    states.push_back(h);
    scores.push_back(step.logits);
  }
  return {nn::stack_time(scores), nn::stack_time(states)};
}

} // namespace word_lm

/// @brief Source imdb parameter names, fixed dimensions and live sequence graph components.
namespace imdb {

namespace nn = datamunge::nn; ///< Native tensor/operation namespace.
using nn::Tensor;             ///< Native shared-storage graph tensor.

constexpr int kSteps = 48;  ///< Original source kSteps dimension.
constexpr int kVocab = 256; ///< Original source kVocab dimension.
constexpr int kUnits = 8;   ///< Original source kUnits dimension.

/// @brief Live source Path endpoints with owning tensor handles.
///
/// Tensor copies share storage and preserve derivative paths. Field names retain the source
/// diagnostic naming and layout.
///
/// @par Example
/// @code
/// dm::nn::models::imdb::Path value;
/// @endcode
/// @see datamunge::nn::Tensor
struct Path {
  Tensor logits;      ///< Live logits endpoint; source shape follows the owning graph.
  Tensor states;      ///< Live states endpoint; source shape follows the owning graph.
  Tensor cell_states; ///< Live cell_states endpoint; source shape follows the owning graph.
  Tensor final_state; ///< Live final_state endpoint; source shape follows the owning graph.
  Tensor final_cell;  ///< Live final_cell endpoint; source shape follows the owning graph.
  Tensor gates;       ///< Live gates endpoint; source shape follows the owning graph.
  Tensor embeddings;  ///< Live embeddings endpoint; source shape follows the owning graph.
};

/// The four LSTM gates (i, f, g, o) after their nonlinearities, [B, 32].
/// @brief Expose original i/f/g/o activations for one IMDB LSTM step.
///
/// Pure live forward graph, with no objective, optimizer, collection or fitting/generation loop.
/// ParameterStore is borrowed and must outlive dependent graphs; first calls initialize missing
/// source-named leaves in original order, later calls reuse them. Callers synchronize store
/// initialization/mutation. Tensor copies share native storage.
///
/// @param p Borrowed store with initialized cell/Wx,Wh,b leaves.
/// @param x Live embedded token[B,8].
/// @param h Live previous hidden state[B,8].
/// @return Concatenated nonlinear gates[B,32].
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);auto
/// path=dm::nn::models::imdb::ImdbLstm(p,dm::nn::zeros({1,48,256}),dm::nn::from_vector(std::vector<int>{48}),dm::nn::zeros({1,8}),dm::nn::zeros({1,8}));
/// auto gates=dm::nn::models::imdb::Gates(p,dm::nn::zeros({1,8}),dm::nn::zeros({1,8}));
/// @endcode
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 ; Lafferty et al. (2001), Conditional Random Fields,
/// https://repository.upenn.edu/handle/20.500.14332/20034 .
/// @see ImdbLstm()
inline Tensor Gates(nn::ParameterStore& p, const Tensor& x, const Tensor& h) {
  const Tensor pre = nn::bias_add(
      nn::add(nn::matmul(x, p.at("cell/Wx")), nn::matmul(h, p.at("cell/Wh"))), p.at("cell/b"));
  const auto parts = nn::split(pre, 4, 1);
  return nn::concat(
      {nn::sigmoid(parts[0]), nn::sigmoid(parts[1]), nn::tanh(parts[2]), nn::sigmoid(parts[3])}, 1);
}

/// @brief Build original length-masked IMDB LSTM sentiment logits and state diagnostics.
///
/// Pure live forward graph, with no objective, optimizer, collection or fitting/generation loop.
/// ParameterStore is borrowed and must outlive dependent graphs; first calls initialize missing
/// source-named leaves in original order, later calls reuse them. Callers synchronize store
/// initialization/mutation. Tensor copies share native storage. State freezes after each length;
/// padded gates are zero. Zero-length rows keep their initial state.
///
/// @param p Borrowed store; embedding/W,cell/* and head/* source leaves.
/// @param encoded Live floating[B,48,256] encoded review tokens.
/// @param lengths Integer[B] source prefix lengths in[0,48].
/// @param initial Live floating[B,8] initial hidden state.
/// @param initial_cell Live floating[B,8] initial cell state.
/// @return Live logits[B,2], states/cells[B,49,8], gates[B,48,32], embeddings[B,48,8] and final
/// h/c[B,8].
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);auto
/// path=dm::nn::models::imdb::ImdbLstm(p,dm::nn::zeros({1,48,256}),dm::nn::from_vector(std::vector<int>{48}),dm::nn::zeros({1,8}),dm::nn::zeros({1,8}));
/// @endcode
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 ; Lafferty et al. (2001), Conditional Random Fields,
/// https://repository.upenn.edu/handle/20.500.14332/20034 .
/// @see Gates()
inline Path ImdbLstm(
    nn::ParameterStore& p,
    const Tensor&       encoded,
    const Tensor&       lengths,
    const Tensor&       initial,
    const Tensor&       initial_cell) {
  const auto   B  = encoded.shape()[0];
  const Tensor ew = p.get("embedding/W", {kVocab, 8}, nn::init::normal(.1));
  Path         out;
  out.embeddings = nn::reshape(nn::matmul(nn::reshape(encoded, {-1, kVocab}), ew), {B, kSteps, 8});
  const auto          xs = nn::time_steps(out.embeddings);
  nn::LstmState       s{initial, initial_cell};
  std::vector<Tensor> states{s.h}, cells{s.c}, gates;
  for (int t = 0; t < kSteps; ++t) {
    const auto   proposed = nn::lstm_step(p, "cell", xs[t], s, 1.0);
    const Tensor gate     = Gates(p, xs[t], s.h);
    const Tensor valid    = nn::expand_dims(nn::greater(lengths, double(t)), 1);
    s                     = {nn::where(valid, proposed.h, s.h), nn::where(valid, proposed.c, s.c)};
    states.push_back(s.h);
    cells.push_back(s.c);
    gates.push_back(nn::where(valid, gate, 0.0));
  }
  const Tensor hw = p.get("head/W", {kUnits, 2}, nn::init::normal(.1));
  const Tensor hb = p.get("head/b", {2}, nn::init::zeros());
  out.logits      = nn::bias_add(nn::matmul(s.h, hw), hb);
  out.states      = nn::stack_time(states);
  out.cell_states = nn::stack_time(cells);
  out.final_state = s.h;
  out.final_cell  = s.c;
  out.gates       = nn::stack_time(gates);
  return out;
}

} // namespace imdb

/// @brief Source bilstm_tagging parameter names, fixed dimensions and live sequence graph
/// components.
namespace bilstm_tagging {

namespace nn = datamunge::nn; ///< Native tensor/operation namespace.
using nn::Tensor;             ///< Native shared-storage graph tensor.

constexpr int kSteps = 24;  ///< Original source kSteps dimension.
constexpr int kVocab = 512; ///< Original source kVocab dimension.
constexpr int kUnits = 8;   ///< Original source kUnits dimension.
constexpr int kPos   = 45;  ///< Original source kPos dimension.
constexpr int kNer   = 9;   ///< Original source kNer dimension.

/// The four LSTM gates (i, f, g, o) after their nonlinearities, [B, 32].
/// @brief Expose one named bidirectional LSTM gate vector.
///
/// Pure live forward graph, with no objective, optimizer, collection or fitting/generation loop.
/// ParameterStore is borrowed and must outlive dependent graphs; first calls initialize missing
/// source-named leaves in original order, later calls reuse them. Callers synchronize store
/// initialization/mutation. Tensor copies share native storage.
///
/// @param p Borrowed store with initialized named LSTM leaves.
/// @param name Source forward or backward recurrent prefix.
/// @param x Live embedded token[B,8].
/// @param h Live previous hidden state[B,8].
/// @return Concatenated i/f/g/o nonlinearities[B,32].
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);auto lengths=dm::nn::from_vector(std::vector<int>{24});auto
/// encoded=dm::nn::zeros({1,24,512});std::vector<dm::nn::Tensor> initial(4,dm::nn::zeros({1,8}));
/// auto path=dm::nn::models::bilstm_tagging::BiLstmFeatures(p,encoded,lengths,initial);auto
/// gates=dm::nn::models::bilstm_tagging::LstmGates(p,"forward",dm::nn::zeros({1,8}),dm::nn::zeros({1,8}));
/// @endcode
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 ; Lafferty et al. (2001), Conditional Random Fields,
/// https://repository.upenn.edu/handle/20.500.14332/20034 .
/// @see BiLstmFeatures()
inline Tensor
LstmGates(nn::ParameterStore& p, const std::string& name, const Tensor& x, const Tensor& h) {
  const Tensor pre = nn::bias_add(
      nn::add(nn::matmul(x, p.at(name + "/Wx")), nn::matmul(h, p.at(name + "/Wh"))),
      p.at(name + "/b"));
  const auto parts = nn::split(pre, 4, 1);
  return nn::concat(
      {nn::sigmoid(parts[0]), nn::sigmoid(parts[1]), nn::tanh(parts[2]), nn::sigmoid(parts[3])}, 1);
}

/// @brief Live source Features endpoints with owning tensor handles.
///
/// Tensor copies share storage and preserve derivative paths. Field names retain the source
/// diagnostic naming and layout.
///
/// @par Example
/// @code
/// dm::nn::models::bilstm_tagging::Features value;
/// @endcode
/// @see datamunge::nn::Tensor
struct Features {
  Tensor embeddings;     ///< Live embeddings endpoint; source shape follows the owning graph.
  Tensor forward_states; ///< Live forward_states endpoint; source shape follows the owning graph.
  Tensor forward_cells;  ///< Live forward_cells endpoint; source shape follows the owning graph.
  Tensor backward_processing; ///< Live backward_processing endpoint; source shape follows the
                              ///< owning graph.
  Tensor backward_cells; ///< Live backward_cells endpoint; source shape follows the owning graph.
  Tensor
      backward_aligned;  ///< Live backward_aligned endpoint; source shape follows the owning graph.
  Tensor features;       ///< Live features endpoint; source shape follows the owning graph.
  Tensor forward_gates;  ///< Live forward_gates endpoint; source shape follows the owning graph.
  Tensor backward_gates; ///< Live backward_gates endpoint; source shape follows the owning graph.
};

/// [B, 1] bool: lengths > t.
/// @brief Form one length-based timestep visibility mask.
///
/// @param lengths Integer[B] lengths supplied by the caller.
/// @param t Zero-based timestep.
/// @return Boolean[B,1] values lengths>t.
/// @par Example
/// @code
/// auto valid=dm::nn::models::bilstm_tagging::ValidAt(dm::nn::from_vector(std::vector<int>{2}),1);
/// @endcode
/// @see BiLstmFeatures()
inline Tensor ValidAt(const Tensor& lengths, int t) {
  return nn::expand_dims(nn::greater(lengths, double(t)), 1);
}

/// initial = {forward h, forward c, backward h, backward c}, each [B, 8].
/// @brief Build independently masked forward/backward LSTMs with original-time alignment.
///
/// Pure live forward graph, with no objective, optimizer, collection or fitting/generation loop.
/// ParameterStore is borrowed and must outlive dependent graphs; first calls initialize missing
/// source-named leaves in original order, later calls reuse them. Callers synchronize store
/// initialization/mutation. Tensor copies share native storage. Backward state processing
/// runs24..1, then aligns each output to its original token index. State freezes past the source
/// length.
///
/// @param p Borrowed store with exact embedding/forward/backward/head names.
/// @param encoded Live floating[B,24,512] encoded words.
/// @param lengths Integer[B] complete sentence lengths in[0,24].
/// @param initial Exactly four floating[B,8] states: forward h,c then backward h,c.
/// @return Live embeddings[B,24,8], chronological forward states, reverse-processing/back-aligned
/// endpoints, cells and gates; concatenated features[B,24,16].
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);auto lengths=dm::nn::from_vector(std::vector<int>{24});auto
/// encoded=dm::nn::zeros({1,24,512});std::vector<dm::nn::Tensor> initial(4,dm::nn::zeros({1,8}));
/// auto path=dm::nn::models::bilstm_tagging::BiLstmFeatures(p,encoded,lengths,initial);
/// @endcode
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 ; Lafferty et al. (2001), Conditional Random Fields,
/// https://repository.upenn.edu/handle/20.500.14332/20034 .
/// @see BiLstmTagging()
inline Features BiLstmFeatures(
    nn::ParameterStore&        p,
    const Tensor&              encoded,
    const Tensor&              lengths,
    const std::vector<Tensor>& initial) {
  const auto   B  = encoded.shape()[0];
  const Tensor ew = p.get("embedding/W", {kVocab, 8}, nn::init::normal(.1));
  Features     out;
  out.embeddings = nn::reshape(nn::matmul(nn::reshape(encoded, {-1, kVocab}), ew), {B, kSteps, 8});
  const auto          xs = nn::time_steps(out.embeddings);
  nn::LstmState       s{initial[0], initial[1]};
  std::vector<Tensor> forward{s.h}, fcells{s.c}, fgates, backward{initial[2]}, bcells{initial[3]},
      aligned(kSteps), bgates(kSteps);
  for (int t = 0; t < kSteps; ++t) {
    const auto   next  = nn::lstm_step(p, "forward", xs[t], s, 1.0);
    const Tensor gate  = LstmGates(p, "forward", xs[t], s.h);
    const Tensor valid = ValidAt(lengths, t);
    s                  = {nn::where(valid, next.h, s.h), nn::where(valid, next.c, s.c)};
    forward.push_back(s.h);
    fcells.push_back(s.c);
    fgates.push_back(nn::where(valid, gate, 0.0));
  }
  s = {initial[2], initial[3]};
  for (int t = kSteps - 1; t >= 0; --t) {
    const auto   next  = nn::lstm_step(p, "backward", xs[t], s, 1.0);
    const Tensor gate  = LstmGates(p, "backward", xs[t], s.h);
    const Tensor valid = ValidAt(lengths, t);
    s                  = {nn::where(valid, next.h, s.h), nn::where(valid, next.c, s.c)};
    backward.push_back(s.h);
    bcells.push_back(s.c);
    aligned[t] = nn::where(valid, s.h, 0.0);
    bgates[t]  = nn::where(valid, gate, 0.0);
  }
  std::vector<Tensor> features;
  for (int t = 0; t < kSteps; ++t)
    features.push_back(
        nn::where(ValidAt(lengths, t), nn::concat({forward[t + 1], aligned[t]}, 1), 0.0));
  out.forward_states      = nn::stack_time(forward);
  out.forward_cells       = nn::stack_time(fcells);
  out.backward_processing = nn::stack_time(backward);
  out.backward_cells      = nn::stack_time(bcells);
  out.backward_aligned    = nn::stack_time(aligned);
  out.features            = nn::stack_time(features);
  out.forward_gates       = nn::stack_time(fgates);
  out.backward_gates      = nn::stack_time(bgates);
  return out;
}

/// @brief Live source Path endpoints with owning tensor handles.
///
/// Tensor copies share storage and preserve derivative paths. Field names retain the source
/// diagnostic naming and layout.
///
/// @par Example
/// @code
/// dm::nn::models::bilstm_tagging::Path value;
/// @endcode
/// @see datamunge::nn::Tensor
struct Path {
  Features context; ///< Live bidirectional state, gate and alignment endpoints.
  Tensor   pos_logits; ///< Live pos_logits endpoint; source shape follows the owning graph.
  Tensor   ner_logits; ///< Live ner_logits endpoint; source shape follows the owning graph.
};

/// @brief Build the original joint POS45/NER9 masked linear heads.
///
/// Pure live forward graph, with no objective, optimizer, collection or fitting/generation loop.
/// ParameterStore is borrowed and must outlive dependent graphs; first calls initialize missing
/// source-named leaves in original order, later calls reuse them. Callers synchronize store
/// initialization/mutation. Tensor copies share native storage.
///
/// @param p Borrowed store with exact embedding/forward/backward/head names.
/// @param encoded Live floating[B,24,512] encoded words.
/// @param lengths Integer[B] complete sentence lengths in[0,24].
/// @param initial Exactly four floating[B,8] states: forward h,c then backward h,c.
/// @return Live context diagnostics plus POS logits[B,24,45] and NER logits[B,24,9]; padded logits
/// are zero.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);auto lengths=dm::nn::from_vector(std::vector<int>{24});auto
/// encoded=dm::nn::zeros({1,24,512});std::vector<dm::nn::Tensor> initial(4,dm::nn::zeros({1,8}));
/// auto path=dm::nn::models::bilstm_tagging::BiLstmTagging(p,encoded,lengths,initial);
/// @endcode
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 ; Lafferty et al. (2001), Conditional Random Fields,
/// https://repository.upenn.edu/handle/20.500.14332/20034 .
/// @see BiLstmFeatures()
inline Path BiLstmTagging(
    nn::ParameterStore&        p,
    const Tensor&              encoded,
    const Tensor&              lengths,
    const std::vector<Tensor>& initial) {
  Path out;
  out.context = BiLstmFeatures(p, encoded, lengths, initial);
  auto head   = [&](const std::string& name, std::int64_t k) {
    const Tensor W = p.get(name + "/W", {16, k}, nn::init::normal(.1));
    const Tensor b = p.get(name + "/b", {k}, nn::init::zeros());
    return nn::reshape(
        nn::bias_add(nn::matmul(nn::reshape(out.context.features, {-1, 16}), W), b),
        {-1, kSteps, k});
  };
  const Tensor pos   = head("pos", kPos);
  const Tensor ner   = head("ner", kNer);
  const Tensor valid = nn::expand_dims(
      nn::less(
          nn::reshape(nn::arange(0, kSteps, 1, nn::DType::Int32), {1, -1}),
          nn::reshape(lengths, {-1, 1})),
      2);
  out.pos_logits = nn::where(valid, pos, 0.0);
  out.ner_logits = nn::where(valid, ner, 0.0);
  return out;
}

} // namespace bilstm_tagging

/// @brief Source bilstm_crf parameter names, fixed dimensions and live sequence graph components.
namespace bilstm_crf {

namespace nn = datamunge::nn; ///< Native tensor/operation namespace.
using nn::Tensor;             ///< Native shared-storage graph tensor.

constexpr std::int64_t kSteps = 24;  ///< Original source kSteps dimension.
constexpr std::int64_t kVocab = 512; ///< Original source kVocab dimension.
constexpr std::int64_t kDim   = 8;   ///< Original source kDim dimension.
constexpr std::int64_t kUnits = 8;   ///< Original source kUnits dimension.
constexpr std::int64_t kTags  = 9;   ///< Original source kTags dimension.

/// @brief Live source TaggingBatch endpoints with owning tensor handles.
///
/// Tensor copies share storage and preserve derivative paths. Field names retain the source
/// diagnostic naming and layout.
///
/// @par Example
/// @code
/// dm::nn::models::bilstm_crf::TaggingBatch value;
/// @endcode
/// @see datamunge::nn::Tensor
struct TaggingBatch {
  Tensor input;      ///< Live input endpoint; source shape follows the owning graph.
  Tensor lengths;    ///< Live lengths endpoint; source shape follows the owning graph.
  Tensor pos;        ///< Live pos endpoint; source shape follows the owning graph.
  Tensor ner;        ///< Live ner endpoint; source shape follows the owning graph.
  Tensor source_ids; ///< Live source_ids endpoint; source shape follows the owning graph.  //
                     ///< int32: [B, 24], [B], [B, 24], [B, 24], [B]
};

/// [B, 1] bool: lengths > t.
/// @brief Form the source CRF emission encoder timestep mask.
///
/// @param lengths Integer[B] sentence lengths.
/// @param t Zero-based timestep.
/// @return Boolean[B,1] values lengths>t.
/// @par Example
/// @code
/// auto mask=dm::nn::models::bilstm_crf::ValidAt(dm::nn::from_vector(std::vector<int>{2}),1);
/// @endcode
/// @see Encode()
inline Tensor ValidAt(const Tensor& lengths, std::int64_t t) {
  return nn::reshape(nn::greater(lengths, static_cast<double>(t)), {-1, 1});
}

/// @brief Live source Path endpoints with owning tensor handles.
///
/// Tensor copies share storage and preserve derivative paths. Field names retain the source
/// diagnostic naming and layout.
///
/// @par Example
/// @code
/// dm::nn::models::bilstm_crf::Path value;
/// @endcode
/// @see datamunge::nn::Tensor
struct Path {
  Tensor embeddings;     ///< Live embeddings endpoint; source shape follows the owning graph.
  Tensor forward_states; ///< Live forward_states endpoint; source shape follows the owning graph.
  Tensor
      backward_aligned; ///< Live backward_aligned endpoint; source shape follows the owning graph.
  Tensor features;      ///< Live features endpoint; source shape follows the owning graph.
  Tensor unaries;       ///< Live unaries endpoint; source shape follows the owning graph.
};

/// Label-free forward pass: words and lengths only.
/// @brief Build source length-masked bidirectional NER emissions before the CRF objective.
///
/// Pure live forward graph, with no objective, optimizer, collection or fitting/generation loop.
/// ParameterStore is borrowed and must outlive dependent graphs; first calls initialize missing
/// source-named leaves in original order, later calls reuse them. Callers synchronize store
/// initialization/mutation. Tensor copies share native storage. Padded unary scores are zero. CRF
/// likelihood, objectives and optimizers remain caller code.
///
/// @param p Borrowed store with source embedding, recurrent and emission leaves.
/// @param input Integer[B,24] word IDs in[0,511].
/// @param lengths Integer[B] complete sentence lengths in[0,24].
/// @return Live embeddings, forward/backward original-time aligned states, features[B,24,16] and
/// unary scores[B,24,9].
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);auto
/// path=dm::nn::models::bilstm_crf::Encode(p,dm::nn::zeros({1,24},dm::nn::DType::Int32),dm::nn::from_vector(std::vector<int>{24}));
/// @endcode
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 ; Lafferty et al. (2001), Conditional Random Fields,
/// https://repository.upenn.edu/handle/20.500.14332/20034 .
/// @see CrfParams(), datamunge::nn::crf_log_norm()
inline Path Encode(nn::ParameterStore& p, const Tensor& input, const Tensor& lengths) {
  Path         out;
  const Tensor table     = p.get("embedding/W", {kVocab, kDim}, nn::init::normal(.1));
  out.embeddings         = nn::gather(table, input); // == one_hot(input) @ W
  const auto          xs = nn::time_steps(out.embeddings);
  const auto          B  = input.shape()[0];
  nn::LstmState       s  = nn::lstm_zero_state(B, kUnits);
  std::vector<Tensor> forward{s.h}, aligned(kSteps);
  for (std::int64_t t = 0; t < kSteps; ++t) {
    const auto next  = nn::lstm_step(p, "forward", xs[t], s, 1.0);
    const auto valid = ValidAt(lengths, t);
    s                = {nn::where(valid, next.h, s.h), nn::where(valid, next.c, s.c)};
    forward.push_back(s.h);
  }
  s = nn::lstm_zero_state(B, kUnits);
  for (std::int64_t t = kSteps - 1; t >= 0; --t) {
    const auto next  = nn::lstm_step(p, "backward", xs[t], s, 1.0);
    const auto valid = ValidAt(lengths, t);
    s                = {nn::where(valid, next.h, s.h), nn::where(valid, next.c, s.c)};
    aligned[t]       = nn::where(valid, s.h, 0.0);
  }
  std::vector<Tensor> features;
  for (std::int64_t t = 0; t < kSteps; ++t)
    features.push_back(
        nn::where(ValidAt(lengths, t), nn::concat({forward[t + 1], aligned[t]}, 1), 0.0));
  out.forward_states   = nn::stack(forward, 1);
  out.backward_aligned = nn::stack(aligned, 1);
  out.features         = nn::stack(features, 1);
  nn::DenseOptions head;
  head.init           = nn::init::normal(.1);
  const Tensor raw    = nn::dense(p, "ner", out.features, kTags, head);
  const Tensor valid3 = nn::expand_dims(
      nn::less(
          nn::reshape(nn::arange(0, kSteps, 1, nn::DType::Int32), {1, -1}),
          nn::reshape(lengths, {-1, 1})),
      2);
  out.unaries = nn::where(valid3, raw, 0.0);
  return out;
}

/// @brief Live source Crf endpoints with owning tensor handles.
///
/// Tensor copies share storage and preserve derivative paths. Field names retain the source
/// diagnostic naming and layout.
///
/// @par Example
/// @code
/// dm::nn::models::bilstm_crf::Crf value;
/// @endcode
/// @see datamunge::nn::Tensor
struct Crf {
  Tensor transition; ///< Live transition endpoint; source shape follows the owning graph.
  Tensor start;      ///< Live start endpoint; source shape follows the owning graph.
  Tensor end;        ///< Live end endpoint; source shape follows the owning graph.
};
/// @brief Create/reuse the original nine-tag CRF transition/start/end leaves.
///
/// Pure live forward graph, with no objective, optimizer, collection or fitting/generation loop.
/// ParameterStore is borrowed and must outlive dependent graphs; first calls initialize missing
/// source-named leaves in original order, later calls reuse them. Callers synchronize store
/// initialization/mutation. Tensor copies share native storage. Transition initialization has
/// normal std.05; start/end initialize to zero.
///
/// @param p Borrowed source parameter store.
/// @return Live transition[from,to][9,9], start[9] and end[9] scores.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);auto crf=dm::nn::models::bilstm_crf::CrfParams(p);
/// @endcode
/// @par References
/// - Hochreiter and Schmidhuber (1997), Long Short-Term Memory,
/// https://doi.org/10.1162/neco.1997.9.8.1735 ; Lafferty et al. (2001), Conditional Random Fields,
/// https://repository.upenn.edu/handle/20.500.14332/20034 .
/// @see datamunge::nn::crf_sequence_score(), datamunge::nn::crf_decode()
inline Crf CrfParams(nn::ParameterStore& p) {
  return {
      p.get("crf/transition", {kTags, kTags}, nn::init::normal(.05)),
      p.get("crf/start", {kTags}, nn::init::zeros()),
      p.get("crf/end", {kTags}, nn::init::zeros())};
}

} // namespace bilstm_crf
} // namespace datamunge::nn::models
