#pragma once
/// @file sequence_vae_data.hpp
/// @brief Checked exhaustive sentence-grammar pools, vocabulary and decoded-prefix display.
#include <filesystem>
#include <fstream>
#include <set>

#include "datamunge/nn/io.hpp"
#include "datamunge/nn/sequence_variational_autoencoder.hpp"

namespace datamunge::nn::models {
/// @brief Return an independent copy of the fixed sentence-VAE nineteen-word vocabulary.
///
/// IDs are PAD0,BOS1,EOS2, followed by article,colour,shape,verb,direction and
/// optional adverb words. The result is an owning mutable copy; changing it does
/// not change models, preparation or later calls. No I/O, fitting, RNG or shared
/// mutable state; O(19) work/storage.
/// @return Owning vocabulary in exact token-ID order.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto words=dm::nn::models::SequenceVocabulary();
/// // words[0]="[pad]", words[1]="[bos]", words[2]="[eos]", words[18]="slowly".
/// @endcode
/// @see SequenceText(), LoadSequenceVaeData()
inline std::vector<std::string> SequenceVocabulary() {
  return {
      "[pad]",
      "[bos]",
      "[eos]",
      "a",
      "the",
      "red",
      "green",
      "blue",
      "circle",
      "square",
      "triangle",
      "moves",
      "rolls",
      "left",
      "right",
      "up",
      "down",
      "quickly",
      "slowly"};
}

/// @brief Own one aligned partition of the exhaustive local sentence grammar.
///
/// All handles own tensor lifetimes; copying shares storage. Loaded pools use
/// Int32 exclusively. Only tokens/explicit prefixes enter the sentence model;
/// factors, lengths and source IDs are audit metadata. No optimizer or RNG state.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto
/// root=dm::datasets::prepare_example_data("sequence-vae-assets","build/local-datasets"); const
/// auto pools=dm::nn::models::LoadSequenceVaeData(root+"/sequence_vae"); const auto fit=pools[0];
/// // 288 rows; validation/test have72 each.
/// @endcode
/// @see LoadSequenceVaeData(), SequenceDataArrays()
struct SequenceVaeData {
  Tensor tokens;  ///< Int32[N,7] targets, including EOS2 and right PAD0 when length6.
  Tensor inputs;  ///< Int32[N,7] BOS1 followed by previous targets only.
  Tensor lengths; ///< Int32[N] EOS-inclusive lengths6 or7, excluding PAD.
  Tensor factors; ///< Int32[N,6] article/colour/shape/verb/direction/adverb indices.
  Tensor source;  ///< Int32[N] unique mixed-radix grammar identity in[0,431].
};

/// @brief Private strict corpus validation, separate from token-model arithmetic.
namespace sequence_grammar_detail {
/// @brief Reject a failed sentence-data invariant before returning any partial corpus.
/// @param condition Required invariant.
/// @param message Diagnostic reason.
/// @throws std::invalid_argument When condition is false.
inline void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::invalid_argument("sentence grammar: " + message);
}
/// @brief Find one required named array while allowing unrelated extra arrays.
/// @param arrays Owning loaded archive handles.
/// @param key Required unique field name.
/// @return Owning tensor handle sharing the loaded array's storage.
/// @throws std::invalid_argument For a missing or repeated required name.
inline Tensor Find(const NamedTensors& arrays, const std::string& key) {
  Tensor out;
  for (const auto& [name, value] : arrays)
    if (name == key) {
      Require(!out.defined(), "repeated array " + key);
      out = value;
    }
  Require(out.defined(), "missing array " + key);
  return out;
}
/// @brief Validate every token, identity, split, factor marginal and target-prefix relation.
/// @param pools Exactly three fitting/validation/test pools; row order is preserved.
/// @throws std::invalid_argument For any schema, grammar, split, identity or balance violation.
inline void Validate(const std::vector<SequenceVaeData>& pools) {
  Require(pools.size() == 3, "exactly fit/validation/test pools required");
  std::set<int>              sources;
  std::set<std::vector<int>> sequences;
  const int                  card[]{2, 3, 3, 2, 4, 3};
  for (int split = 0; split < 3; ++split) {
    const auto&        d = pools[split];
    const std::int64_t n = split == 0 ? 288 : 72;
    Require(
        d.tokens.defined() && d.inputs.defined() && d.lengths.defined() && d.factors.defined() &&
            d.source.defined(),
        "all five fields required");
    for (const auto& t : {d.tokens, d.inputs, d.lengths, d.factors, d.source})
      Require(t.dtype() == DType::Int32, "all corpus arrays must be Int32");
    Require(
        d.tokens.shape() == Shape{n, 7} && d.inputs.shape() == Shape{n, 7} &&
            d.lengths.shape() == Shape{n} && d.factors.shape() == Shape{n, 6} &&
            d.source.shape() == Shape{n},
        "exact pool dimensions required");
    const auto tokens  = d.tokens.contiguous().to_vector<int>(),
               inputs  = d.inputs.contiguous().to_vector<int>(),
               lengths = d.lengths.contiguous().to_vector<int>(),
               factors = d.factors.contiguous().to_vector<int>(),
               source  = d.source.contiguous().to_vector<int>();
    int counts[6][4]{}, shorts = 0, total = 0;
    for (std::int64_t b = 0; b < n; ++b) {
      int f[6];
      for (int j = 0; j < 6; ++j) {
        f[j] = factors[b * 6 + j];
        Require(f[j] >= 0 && f[j] < card[j], "factor out of range");
        ++counts[j][f[j]];
      }
      const int id      = (((((f[0] * 3 + f[1]) * 3 + f[2]) * 2 + f[3]) * 4 + f[4]) * 3 + f[5]);
      const int residue = 2 * ((f[1] + f[2] + f[5]) % 3) + (f[0] + f[3] + f[4]) % 2;
      Require(source[b] == id && sources.insert(id).second, "incorrect or duplicate source ID");
      Require(
          split == 2   ? residue == 0
          : split == 1 ? residue == 1
                       : residue >= 2,
          "incorrect held-out combination split");
      std::vector<int> expected{3 + f[0], 5 + f[1], 8 + f[2], 11 + f[3], 13 + f[4]};
      if (f[5])
        expected.push_back(16 + f[5]);
      expected.push_back(2);
      const int length = int(expected.size());
      Require(lengths[b] == length, "incorrect EOS-inclusive length");
      shorts += length == 6;
      total += length;
      expected.resize(7, 0);
      for (int t = 0; t < 7; ++t) {
        Require(tokens[b * 7 + t] == expected[t], "token/EOS/PAD does not regenerate from factors");
        Require(
            inputs[b * 7 + t] == (t ? expected[t - 1] : 1),
            "prefix must be BOS then PREVIOUS target");
      }
      Require(sequences.insert(expected).second, "duplicate sentence");
    }
    for (int j = 0; j < 6; ++j)
      for (int k = 0; k < card[j]; ++k)
        Require(counts[j][k] == n / card[j], "unbalanced factor marginal");
    Require(
        shorts == n / 3 && total == n * 20 / 3, "incorrect adverb lengths or valid-token count");
  }
  Require(
      sources.size() == 432 && sequences.size() == 432, "exhaustive432 unique sentences required");
}
/// @brief Extract a selected nonempty integer row without inspecting unrelated rows.
/// @param tokens Int32/Int64 matrix[B,T], B,T>0.
/// @param row Zero-based selected row in[0,B).
/// @return Owning int64 token IDs in selected order; values are checked by each caller.
/// @throws std::invalid_argument For invalid shape, dtype or row.
inline std::vector<std::int64_t> Row(const Tensor& tokens, std::int64_t row) {
  Require(
      tokens.defined() && tokens.dim() == 2 &&
          (tokens.dtype() == DType::Int32 || tokens.dtype() == DType::Int64),
      "integer token matrix required");
  Require(
      tokens.size(0) > 0 && tokens.size(1) > 0 && row >= 0 && row < tokens.size(0),
      "invalid selected row");
  return select(tokens, 0, row).contiguous().to_vector<std::int64_t>();
}
} // namespace sequence_grammar_detail

/// @brief Load and exhaustively check the local432-sentence VAE grammar without reordering rows.
///
/// Reads root/sentences.npz and vocabulary.txt; never prepares, downloads, shuffles
/// or fits. Vocabulary has exactly the19 LF-delimited words in SequenceVocabulary;
/// a final newline is optional. Config is Int32 with exactly two elements19,7
/// (any shape). Required arrays are the five SequenceVaeData fields for fit288,
/// validation72,test72; extra arrays are ignored. Missing/ambiguous fields fail.
/// All fields must be Int32 with exact shapes. Every target/EOS/PAD, previous-
/// target prefix, factor, mixed-radix source ID, unique sentence, held-out residue,
/// per-factor balance and six/seven-token length is regenerated and checked.
/// Cardinalities are2,3,3,2,4,3; residue=2*((colour+shape+adverb)%3)
/// +(article+verb+direction)%2 selects test0,validation1,fit2..5. Valid row
/// permutations are retained, including supplied nonascending data. Manifest JSON
/// is not read; this validates numeric grammar, not external-framework provenance.
/// Owns loaded handles after files close. No mutation, RNG or shared mutable
/// state. O(432*7) numeric work plus archive I/O, with bounded corpus validation.
/// @param root Directory containing sentences.npz and vocabulary.txt.
/// @return Owning three pools ordered fitting, validation, test, preserving each row order.
/// @throws std::invalid_argument For schema, vocabulary or grammar violations.
/// @throws std::runtime_error For archive or vocabulary I/O/format errors.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto
/// root=dm::datasets::prepare_example_data("sequence-vae-assets","build/local-datasets"); const
/// auto pools=dm::nn::models::LoadSequenceVaeData(root+"/sequence_vae");
/// // Every one of the432 sentences and its previous-target prefix has been checked.
/// @endcode
/// @see SequenceVaeData, SequenceDataArrays(), SequenceVocabulary()
inline std::vector<SequenceVaeData> LoadSequenceVaeData(const std::string& root) {
  namespace fs = std::filesystem;
  using namespace sequence_grammar_detail;
  const auto    arrays = load_npz((fs::path(root) / "sentences.npz").string());
  std::ifstream text(fs::path(root) / "vocabulary.txt");
  if (!text)
    throw std::runtime_error("cannot read sentence vocabulary");
  std::vector<std::string> words;
  std::string              word;
  while (std::getline(text, word))
    words.push_back(word);
  if (text.bad())
    throw std::runtime_error("cannot read complete sentence vocabulary");
  Require(words == SequenceVocabulary(), "exact fixed vocabulary PAD0/BOS1/EOS2 required");
  const auto config = Find(arrays, "config");
  Require(config.dtype() == DType::Int32 && config.numel() == 2, "config must be Int32{19,7}");
  Require(
      config.contiguous().to_vector<int>() == std::vector<int>{19, 7}, "config must contain19,7");
  std::vector<SequenceVaeData> pools;
  for (const std::string name : {"fit", "validation", "test"})
    pools.push_back(
        {Find(arrays, name + "_tokens"),
         Find(arrays, name + "_decoder_inputs"),
         Find(arrays, name + "_lengths"),
         Find(arrays, name + "_factors"),
         Find(arrays, name + "_source_ids")});
  Validate(pools);
  return pools;
}

/// @brief Display a selected decoded prefix through its first EOS, retaining special-token text.
///
/// Accepts a nonempty Int32/64[B,T] matrix and row in[0,B). Checks IDs0..18 in
/// that row only through the first EOS2, including EOS. Later values are ignored.
/// Joins words with single spaces, retaining literal [pad]/[bos]/[eos]; if no EOS
/// is present, displays all T words. No filtering of malformed grammar, RNG,
/// mutation or shared state. Owns string; O(T) work/storage. Strides accepted.
/// @param tokens Integer token matrix[B,T], positive B,T.
/// @param row Zero-based selected row.
/// @return Owning human-readable prefix, including literal [eos] when encountered.
/// @throws std::invalid_argument For invalid shape/dtype/row or prefix IDs outside0..18.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto ids=dm::nn::from_vector(std::vector<int>{3,5,8,11,13,2,0},{1,7});
/// const auto text=dm::nn::models::SequenceText(ids,0);
/// // "a red circle moves left [eos]"
/// @endcode
/// @see SequenceGrammarValid(), SequenceVocabulary()
inline std::string SequenceText(const Tensor& tokens, std::int64_t row) {
  const auto  ids   = sequence_grammar_detail::Row(tokens, row);
  const auto  words = SequenceVocabulary();
  std::string text;
  for (auto id : ids) {
    sequence_grammar_detail::Require(id >= 0 && id < 19, "display prefix ID outside vocabulary");
    if (!text.empty())
      text += ' ';
    text += words[id];
    if (id == 2)
      break;
  }
  return text;
}

/// @brief Test whether a selected prefix before its first EOS follows the fixed sentence grammar.
///
/// Int32/64[B,T], positive B,T, selected row in[0,B). A valid prefix has article,
/// colour, shape, verb, direction, optionally quickly/slowly, then EOS2. Missing
/// EOS, wrong order, PAD/BOS before EOS or out-of-role IDs return false. Values
/// AFTER the first EOS are ignored, including non-PAD suffixes: this is the
/// original prefix-grammar diagnostic, not a strict generated-suffix validator.
/// No mutation, RNG or gradient; owns bool, O(T) work/storage, strides accepted.
/// @param tokens Nonempty integer token matrix.
/// @param row Zero-based selected row.
/// @return True exactly for a grammatical EOS-terminated prefix, ignoring later columns.
/// @throws std::invalid_argument For invalid shape, dtype or selected row.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto ids=dm::nn::from_vector(std::vector<int>{3,5,8,11,13,2,0},{1,7});
/// const bool valid=dm::nn::models::SequenceGrammarValid(ids,0); // true
/// @endcode
/// @see SequenceText(), SequenceGenerate()
inline bool SequenceGrammarValid(const Tensor& tokens, std::int64_t row) {
  const auto                values = sequence_grammar_detail::Row(tokens, row);
  std::vector<std::int64_t> ids;
  bool                      eos = false;
  for (auto id : values) {
    if (id == 2) {
      eos = true;
      break;
    }
    ids.push_back(id);
  }
  if (!eos || (ids.size() != 5 && ids.size() != 6))
    return false;
  return ids[0] >= 3 && ids[0] <= 4 && ids[1] >= 5 && ids[1] <= 7 && ids[2] >= 8 && ids[2] <= 10 &&
         ids[3] >= 11 && ids[3] <= 12 && ids[4] >= 13 && ids[4] <= 16 &&
         (ids.size() == 5 || (ids[5] >= 17 && ids[5] <= 18));
}

/// @brief Export three fully validated sentence pools under the original fifteen experiment-array
/// names.
///
/// Performs the same exhaustive numeric checks as LoadSequenceVaeData, without
/// reading files or vocabulary. Returns five arrays per split, fit/validation/test
/// order; config is deliberately absent from the experiment artifact protocol.
/// Handles own lifetimes and share storage with pools; later caller mutation is
/// visible through shared handles. No reordering, copying of numeric storage,
/// hidden fitting or RNG. Validation work O(432*7), plus fifteen output handles.
/// @param pools Exactly three valid fitting/validation/test pools, strides accepted.
/// @return Owning named tensor handles preserving source row order and values.
/// @throws std::invalid_argument For any missing field, schema, grammar or split violation.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto
/// root=dm::datasets::prepare_example_data("sequence-vae-assets","build/local-datasets"); const
/// auto
/// arrays=dm::nn::models::SequenceDataArrays(dm::nn::models::LoadSequenceVaeData(root+"/sequence_vae"));
/// // arrays.size()==15; these are the experiment data.npz fields, without config.
/// @endcode
/// @see LoadSequenceVaeData(), SequenceVaeData
inline NamedTensors SequenceDataArrays(const std::vector<SequenceVaeData>& pools) {
  sequence_grammar_detail::Validate(pools);
  NamedTensors out;
  for (int split = 0; split < 3; ++split) {
    const std::string name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    const auto&       d    = pools[split];
    out.emplace_back(name + "_tokens", d.tokens);
    out.emplace_back(name + "_decoder_inputs", d.inputs);
    out.emplace_back(name + "_lengths", d.lengths);
    out.emplace_back(name + "_factors", d.factors);
    out.emplace_back(name + "_source_ids", d.source);
  }
  return out;
}
} // namespace datamunge::nn::models
