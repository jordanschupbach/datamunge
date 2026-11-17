#pragma once
/// @file published_bert.hpp
/// @brief Actual published Small BERT serving and explicit ASCII WordPiece packing.
#include <datamunge/nn/completion_graphs.hpp>
#include <datamunge/nn/io.hpp>
#include <filesystem>
#include <fstream>
namespace datamunge::nn::models {
/// @brief wordpiece source-equation components; no optimizer or training driver.
namespace wordpiece {
namespace nn = datamunge::nn;
/// @brief Own the published uncased BERT vocabulary and tokenize explicit ASCII requests.
///
/// Reads30522 vocabulary lines and owns the token strings and lookup. Vocabulary/lookup
/// are public for source compatibility; keep them unchanged while using the tokenizer.
/// Concurrent const tokenization is safe when those containers are not mutated.
/// This is the ASCII BasicTokenizer/WordPiece subset; it rejects non-ASCII rather than
/// claiming Unicode normalization. Special IDs PAD0,UNK100,CLS101,SEP102,MASK103 are checked.
/// @throws std::invalid_argument Invalid vocabulary, non-ASCII input, token/type/mask IDs or packed
/// geometry.
/// @throws std::runtime_error For file/array decoding failures where files are read.
/// @par Example
/// @code
/// auto root=dm::datasets::prepare_example_data("nn-completion-assets");
/// dm::nn::models::wordpiece::AsciiWordPiece
/// tokenizer(std::filesystem::path(root)/"pretrained_bert_serve/vocab.txt"); auto
/// row=tokenizer.Pack("Hello TensorFlow!");
/// @endcode
/// @par References
/// - Google Research, BERT tokenization source,
/// https://github.com/google-research/bert/blob/master/tokenization.py .
/// @see dm::datasets::prepare_example_data(), dm::nn::load_safetensors()
class AsciiWordPiece {
 public:
  /// @brief Owned token strings in original vocabulary index order; keep immutable during use.
  std::vector<std::string> vocabulary;
  /// @brief Owned token-to-ID map; keep consistent with vocabulary and immutable during use.
  std::map<std::string, int> lookup;
  /// @brief Read and validate the30522-token uncased BERT vocabulary.
  ///
  /// Strings and lookup are copied into this instance and outlive the closed file.
  /// Validates line count and five original special-token identities; callers authenticate
  /// the complete source file independently, or use the pinned native adapter.
  /// @param path Readable vocabulary file with one original token per line.
  /// @throws std::invalid_argument Invalid vocabulary, non-ASCII input, token/type/mask IDs or
  /// packed geometry.
  /// @throws std::runtime_error For file/array decoding failures where files are read.
  /// @par Example
  /// @code
  /// auto root=dm::datasets::prepare_example_data("nn-completion-assets");
  /// dm::nn::models::wordpiece::AsciiWordPiece
  /// tokenizer(std::filesystem::path(root)/"pretrained_bert_serve/vocab.txt"); auto
  /// row=tokenizer.Pack("Hello TensorFlow!");
  /// @endcode
  /// @par References
  /// - Google Research, BERT tokenization source,
  /// https://github.com/google-research/bert/blob/master/tokenization.py .
  /// @see dm::datasets::prepare_example_data(), dm::nn::load_safetensors()
  explicit AsciiWordPiece(const std::filesystem::path& path) {
    std::ifstream f(path);
    component_detail::Require(bool(f), "WordPiece vocabulary file");
    std::string token;
    while (std::getline(f, token)) {
      lookup.emplace(token, vocabulary.size());
      vocabulary.push_back(token);
    }
    component_detail::Require(
        vocabulary.size() == 30522 && vocabulary[0] == "[PAD]" && vocabulary[100] == "[UNK]" &&
            vocabulary[101] == "[CLS]" && vocabulary[102] == "[SEP]" && vocabulary[103] == "[MASK]",
        "published BERT uncased vocabulary and IDs");
  }
  /// @brief Return uncased ASCII BasicTokenizer and greedy WordPiece IDs.
  ///
  /// Lowercases ASCII letters, splits ASCII whitespace and punctuation, then chooses
  /// longest vocabulary pieces with ## prefixes after the first piece. A failed word or
  /// one longer than100characters becomes one UNK100. Empty input returns an empty vector.
  /// The instance and input string are unchanged.
  /// @param text ASCII request bytes; any byte>=128 is rejected.
  /// @return Owning token IDs, without CLS/SEP/padding.
  /// @throws std::invalid_argument Invalid vocabulary, non-ASCII input, token/type/mask IDs or
  /// packed geometry.
  /// @throws std::runtime_error For file/array decoding failures where files are read.
  /// @par Example
  /// @code
  /// auto root=dm::datasets::prepare_example_data("nn-completion-assets");
  /// dm::nn::models::wordpiece::AsciiWordPiece
  /// tokenizer(std::filesystem::path(root)/"pretrained_bert_serve/vocab.txt"); auto
  /// row=tokenizer.Pack("Hello TensorFlow!");
  /// @endcode
  /// @par References
  /// - Google Research, BERT tokenization source,
  /// https://github.com/google-research/bert/blob/master/tokenization.py .
  /// @see dm::datasets::prepare_example_data(), dm::nn::load_safetensors()
  std::vector<int> Tokenize(const std::string& text) const {
    std::vector<std::string> basic;
    std::string              word;
    auto                     flush = [&]() {
      if (!word.empty()) {
        basic.push_back(word);
        word.clear();
      }
    };
    for (unsigned char c : text) {
      component_detail::Require(
          c < 128,
          "ASCII WordPiece demo rejects non-ASCII; use "
          "full BERT Unicode tokenizer for such text");
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        flush();
        continue;
      }
      if (c < 32 || c == 127)
        continue;
      if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) ||
          (c >= 123 && c <= 126)) {
        flush();
        basic.emplace_back(1, char(c));
      } else
        word.push_back(c >= 'A' && c <= 'Z' ? char(c + 32) : char(c));
    }
    flush();
    std::vector<int> out;
    for (const auto& token : basic) {
      if (token.size() > 100) {
        out.push_back(100);
        continue;
      }
      std::vector<int> pieces;
      size_t           start = 0;
      bool             bad   = false;
      while (start < token.size()) {
        size_t end    = token.size();
        int    chosen = -1;
        for (; end > start; --end) {
          auto piece = (start ? "##" : "") + token.substr(start, end - start);
          auto it    = lookup.find(piece);
          if (it != lookup.end()) {
            chosen = it->second;
            break;
          }
        }
        if (chosen < 0) {
          bad = true;
          break;
        }
        pieces.push_back(chosen);
        start = end;
      }
      if (bad)
        out.push_back(100);
      else
        out.insert(out.end(), pieces.begin(), pieces.end());
    }
    return out;
  }
  /// @brief Own packed Int32 request IDs, visibility and segment types.
  ///
  /// Tensor handles own shared storage; a default instance has undefined handles.
  /// Pack() returns three independent CPU Int32[1,width] endpoints. PAD0 fills unused
  /// positions, mask is0/1, and types are0 for first segment/padding or1 for the second.
  /// @throws std::invalid_argument Invalid vocabulary, non-ASCII input, token/type/mask IDs or
  /// packed geometry.
  /// @throws std::runtime_error For file/array decoding failures where files are read.
  /// @par Example
  /// @code
  /// auto root=dm::datasets::prepare_example_data("nn-completion-assets");
  /// dm::nn::models::wordpiece::AsciiWordPiece
  /// tokenizer(std::filesystem::path(root)/"pretrained_bert_serve/vocab.txt"); auto
  /// row=tokenizer.Pack("Hello TensorFlow!");
  /// @endcode
  /// @par References
  /// - Google Research, BERT tokenization source,
  /// https://github.com/google-research/bert/blob/master/tokenization.py .
  /// @see dm::datasets::prepare_example_data(), dm::nn::load_safetensors()
  struct Packed {
    /// @brief ids, mask, types: live source graph endpoint/storage; tensor copies share native
    /// storage.
    nn::Tensor ids;   ///< Live source endpoint; owning handle shares storage and derivatives.
    nn::Tensor mask;  ///< Live source endpoint; owning handle shares storage and derivatives.
    nn::Tensor types; ///< Live source endpoint; owning handle shares storage and derivatives.
  };
  /// @brief Pack one or two ASCII sentences using original BERT special tokens.
  ///
  /// Tokenizes both requested segments, then repeatedly truncates the longer side
  /// (ties truncate the second) to reserve CLS/SEP. Single requests use CLS A SEP;
  /// pairs use CLS A SEP B SEP. Output owns storage and leaves this tokenizer unchanged.
  /// @param first ASCII first sentence, possibly empty.
  /// @param second ASCII second sentence, defaultempty; ignored unless pair=true.
  /// @param pair Defaultfalse; true packs the second segment, including an empty one.
  /// @param width Output width, default32;2..512 for single requests or3..512 for pairs.
  /// @return Owned Int32[1,width] IDs, binary mask and0/1 segment types.
  /// @throws std::invalid_argument Invalid vocabulary, non-ASCII input, token/type/mask IDs or
  /// packed geometry.
  /// @throws std::runtime_error For file/array decoding failures where files are read.
  /// @par Example
  /// @code
  /// auto root=dm::datasets::prepare_example_data("nn-completion-assets");
  /// dm::nn::models::wordpiece::AsciiWordPiece
  /// tokenizer(std::filesystem::path(root)/"pretrained_bert_serve/vocab.txt"); auto
  /// row=tokenizer.Pack("Hello TensorFlow!");
  /// @endcode
  /// @par References
  /// - Google Research, BERT tokenization source,
  /// https://github.com/google-research/bert/blob/master/tokenization.py .
  /// @see dm::datasets::prepare_example_data(), dm::nn::load_safetensors()
  Packed Pack(
      const std::string& first,
      const std::string& second = "",
      bool               pair   = false,
      int                width  = 32) const {
    component_detail::Require(
        width >= (pair ? 3 : 2) && width <= 512, "BERT packed width within position capacity");
    auto a = Tokenize(first), b = pair ? Tokenize(second) : std::vector<int>{};
    while (int(a.size() + b.size()) > width - (pair ? 3 : 2)) {
      if (a.size() > b.size())
        a.pop_back();
      else
        b.pop_back();
    }
    std::vector<int> ids{101}, types{0};
    ids.insert(ids.end(), a.begin(), a.end());
    types.insert(types.end(), a.size(), 0);
    ids.push_back(102);
    types.push_back(0);
    if (pair) {
      ids.insert(ids.end(), b.begin(), b.end());
      types.insert(types.end(), b.size(), 1);
      ids.push_back(102);
      types.push_back(1);
    }
    std::vector<int> packed_ids(width), packed_mask(width), packed_types(width);
    for (size_t i = 0; i < ids.size(); ++i) {
      packed_ids[i]   = ids[i];
      packed_mask[i]  = 1;
      packed_types[i] = types[i];
    }
    Packed result{
        nn::from_vector(packed_ids, {1, width}),
        nn::from_vector(packed_mask, {1, width}),
        nn::from_vector(packed_types, {1, width})};
    return result;
  }
};
} // namespace wordpiece

/// @brief pretrained bert source-equation components; no optimizer or training driver.
namespace pretrained_bert {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief A: owning source-compatible native type spelling.
using A = std::map<std::string, Tensor>;
/// @brief Read a local native NPZ into a name-to-tensor map.
///
/// Returned owning tensors outlive the closed file. No download, authentication or
/// optimizer is performed here; the native preparation adapter authenticates publication.
/// @param file Readable native NPZ archive.
/// @return Owning CPU tensor handles keyed by source names.
/// @throws std::invalid_argument Malformed native array metadata or unsupported numeric storage.
/// @throws std::runtime_error For file/array decoding failures where files are read.
/// @par Example
/// @code
/// auto root=dm::datasets::prepare_example_data("nn-completion-assets");
/// dm::nn::models::pretrained_bert::Model
/// model(std::filesystem::path(root)/"pretrained_bert_serve/pretrained_arrays.npz"); auto
/// y=model.Run({{"input_word_ids",dm::nn::from_vector(std::vector<int>{101,7592,102},{1,3})},{"input_mask",dm::nn::ones({1,3},dm::nn::DType::Int32)},{"input_type_ids",dm::nn::zeros({1,3},dm::nn::DType::Int32)}});
/// @endcode
/// @par References
/// - Google Research, published bert_uncased_L-2_H-128_A-2,
/// https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/tree/30b0a37ccaaa32f332884b96992754e246e48c5f
/// .
/// @see dm::datasets::prepare_example_data(), dm::nn::load_safetensors()
inline A Load(const std::filesystem::path& file) {
  auto arrays = nn::load_npz(file.string());
  return {arrays.begin(), arrays.end()};
}
/// @brief Own and serve the published two-layer128-wide Small BERT equations.
///
/// Owns44 Float32 arrays totalling4433210 stored values, including the unused MLM head.
/// The constructor checks count/dtype; publication identity is authenticated by the
/// preparation adapter, not by those counts. Run() uses original key-only masking,
/// nonzero padded query states, epsilon1e-12 and the source tanh GELU approximation.
/// There is no optimizer, mutable inference cache or gradient API. Concurrent const
/// serving is safe; Arrays() returns deep copies, so internal weights never escape.
/// @throws std::invalid_argument Wrong stored tensor count/dtype/total parameter count or invalid
/// serving input.
/// @throws std::runtime_error For file/array decoding failures where files are read.
/// @par Example
/// @code
/// auto root=dm::datasets::prepare_example_data("nn-completion-assets");
/// dm::nn::models::pretrained_bert::Model
/// model(std::filesystem::path(root)/"pretrained_bert_serve/pretrained_arrays.npz"); auto
/// y=model.Run({{"input_word_ids",dm::nn::from_vector(std::vector<int>{101,7592,102},{1,3})},{"input_mask",dm::nn::ones({1,3},dm::nn::DType::Int32)},{"input_type_ids",dm::nn::zeros({1,3},dm::nn::DType::Int32)}});
/// @endcode
/// @par References
/// - Google Research, published bert_uncased_L-2_H-128_A-2,
/// https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/tree/30b0a37ccaaa32f332884b96992754e246e48c5f
/// .
/// @see dm::datasets::prepare_example_data(), dm::nn::load_safetensors()
class Model {
  /// @brief Private owning stored Float32 arrays, never exposed without cloning.
  A weights;

 public:
  /// @brief Load owning published-source arrays from a native NPZ boundary.
  ///
  /// Copies the archive into private tensor storage; the file may be closed/removed
  /// after construction. Checks44 Float32 arrays and4433210 elements. Required names
  /// and individual shapes are used by Run(); use the authenticated native adapter
  /// to obtain the supported complete inventory.
  /// @param file Readable NPZ in the native published parameter spelling/layout.
  /// @throws std::invalid_argument Wrong array count, dtype or total element count.
  /// @throws std::runtime_error For file/array decoding failures where files are read.
  /// @par Example
  /// @code
  /// auto root=dm::datasets::prepare_example_data("nn-completion-assets");
  /// dm::nn::models::pretrained_bert::Model
  /// model(std::filesystem::path(root)/"pretrained_bert_serve/pretrained_arrays.npz"); auto
  /// y=model.Run({{"input_word_ids",dm::nn::from_vector(std::vector<int>{101,7592,102},{1,3})},{"input_mask",dm::nn::ones({1,3},dm::nn::DType::Int32)},{"input_type_ids",dm::nn::zeros({1,3},dm::nn::DType::Int32)}});
  /// @endcode
  /// @par References
  /// - Google Research, published bert_uncased_L-2_H-128_A-2,
  /// https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/tree/30b0a37ccaaa32f332884b96992754e246e48c5f
  /// .
  /// @see dm::datasets::prepare_example_data(), dm::nn::load_safetensors()
  explicit Model(const std::filesystem::path& file) : weights(Load(file)) {
    component_detail::Require(weights.size() == 44, "44 published stored tensors");
    std::int64_t count = 0;
    for (const auto& [k, v] : weights) {
      component_detail::Require(v.dtype() == nn::DType::Float32, "float32 published weights");
      count += v.numel();
    }
    component_detail::Require(count == 4433210, "actual pretrained variable count");
  }
  /// @brief Deep-copy every stored array without exposing mutable internal weights.
  ///
  /// Preserves source names, shapes and Float32 bits. Copies own storage independently
  /// of this model; no forward execution or file access occurs.
  /// @return Map of44 owning cloned tensors.
  /// @throws std::invalid_argument Unsupported primitive storage/backend for cloning.
  /// @throws std::runtime_error For file/array decoding failures where files are read.
  /// @par Example
  /// @code
  /// auto root=dm::datasets::prepare_example_data("nn-completion-assets");
  /// dm::nn::models::pretrained_bert::Model
  /// model(std::filesystem::path(root)/"pretrained_bert_serve/pretrained_arrays.npz"); auto
  /// y=model.Run({{"input_word_ids",dm::nn::from_vector(std::vector<int>{101,7592,102},{1,3})},{"input_mask",dm::nn::ones({1,3},dm::nn::DType::Int32)},{"input_type_ids",dm::nn::zeros({1,3},dm::nn::DType::Int32)}});
  /// @endcode
  /// @par References
  /// - Google Research, published bert_uncased_L-2_H-128_A-2,
  /// https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/tree/30b0a37ccaaa32f332884b96992754e246e48c5f
  /// .
  /// @see dm::datasets::prepare_example_data(), dm::nn::load_safetensors()
  A Arrays() const {
    A out;
    for (const auto& [k, v] : weights)
      out[k] = v.clone();
    return out;
  }
  /// @brief Serve original five Small BERT endpoints with key-only attention masks.
  ///
  /// Inputs are CPU Int32[N,T],N>0,T in[2,512], with identical shapes. Each row must
  /// have at least one visible key; holes are allowed. Values of masked query states
  /// are retained. Computes two128-wide, two-head,512-FF encoder layers and a tanh
  /// pooler. Inputs and stored weights are unchanged. Outputs are owning detached
  /// Float32 tensors; repeated serving is deterministic and no training graph remains.
  /// @param b input_word_ids in[0,30521], input_mask binary, input_type_ids in{0,1}.
  /// @return bert_encoder and bert_encoder_3 pooled[N,128]; bert_encoder_1 layer0[N,T,128];
  /// bert_encoder_2 and bert_encoder_4 layer1[N,T,128].
  /// @throws std::invalid_argument Missing/mismatched keys, dtype/rank/width, token/type/mask
  /// values or empty visible row. Missing malformed weight names may throw std::out_of_range.
  /// @throws std::runtime_error For file/array decoding failures where files are read.
  /// @par Example
  /// @code
  /// auto root=dm::datasets::prepare_example_data("nn-completion-assets");
  /// dm::nn::models::pretrained_bert::Model
  /// model(std::filesystem::path(root)/"pretrained_bert_serve/pretrained_arrays.npz"); auto
  /// y=model.Run({{"input_word_ids",dm::nn::from_vector(std::vector<int>{101,7592,102},{1,3})},{"input_mask",dm::nn::ones({1,3},dm::nn::DType::Int32)},{"input_type_ids",dm::nn::zeros({1,3},dm::nn::DType::Int32)}});
  /// @endcode
  /// @par References
  /// - Google Research, published bert_uncased_L-2_H-128_A-2,
  /// https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/tree/30b0a37ccaaa32f332884b96992754e246e48c5f
  /// .
  /// @see dm::datasets::prepare_example_data(), dm::nn::load_safetensors()
  A Run(const A& b) const {
    for (auto key : {"input_word_ids", "input_mask", "input_type_ids"})
      component_detail::Require(
          b.count(key) && b.at(key).dtype() == nn::DType::Int32 && b.at(key).dim() == 2,
          "BERT input keys/int32/rank2");
    auto ids = b.at("input_word_ids"), mask = b.at("input_mask"), types = b.at("input_type_ids");
    int  n = ids.size(0), t = ids.size(1);
    component_detail::Require(
        ids.shape() == mask.shape() && ids.shape() == types.shape() && n > 0 && t >= 2 && t <= 512,
        "matching BERT input shapes/max512");
    auto iv = ids.to_vector<int>(), mv = mask.to_vector<int>(), tv = types.to_vector<int>();
    for (int row = 0; row < n; ++row) {
      int active = 0;
      for (int j = 0; j < t; ++j) {
        int i = row * t + j;
        component_detail::Require(iv[i] >= 0 && iv[i] < 30522, "BERT vocabulary range");
        component_detail::Require(mv[i] == 0 || mv[i] == 1, "binary attention mask");
        component_detail::Require(tv[i] == 0 || tv[i] == 1, "two segment IDs");
        active += mv[i];
      }
      component_detail::Require(active > 0, "at least one visible token");
    }
    nn::NoGradGuard guard;
    auto            w    = [&](const std::string& key) { return weights.at(key); };
    auto            norm = [&](const Tensor& x, const std::string& prefix) {
      auto c = x - nn::mean(x, {-1}, true);
      return c / nn::sqrt(nn::mean(nn::square(c), {-1}, true) + 1e-12) * w(prefix + "/gamma") +
             w(prefix + "/beta");
    };
    auto h = norm(
        nn::gather(w("word_embeddings/embeddings"), ids) +
            nn::slice(w("position_embedding/embeddings"), {0, 0}, {t, 128}) +
            nn::gather(w("type_embeddings/embeddings"), types),
        "embeddings/layer_norm");
    std::vector<Tensor> states;
    auto bias = nn::reshape((1. - nn::cast(mask, nn::DType::Float32)) * -1e9, {n, 1, 1, t});
    for (int l = 0; l < 2; ++l) {
      auto prefix  = "transformer/layer_" + std::to_string(l);
      auto project = [&](const std::string& name) {
        auto p     = prefix + "/self_attention/" + name;
        auto value = nn::matmul(h, nn::reshape(w(p + "/kernel"), {128, 128})) +
                     nn::reshape(w(p + "/bias"), {128});
        return nn::transpose(nn::reshape(value, {n, t, 2, 64}), {0, 2, 1, 3});
      };
      auto q = project("query"), k = project("key"), v = project("value");
      auto probability = nn::softmax(nn::matmul(q * .125, k, false, true) + bias, -1);
      auto context =
          nn::reshape(nn::transpose(nn::matmul(probability, v), {0, 2, 1, 3}), {n, t, 128});
      auto out =
          nn::matmul(
              context,
              nn::reshape(w(prefix + "/self_attention/attention_output/kernel"), {128, 128})) +
          w(prefix + "/self_attention/attention_output/bias");
      h      = norm(h + out, prefix + "/self_attention_layer_norm");
      auto f = nn::gelu(
          nn::matmul(h, w(prefix + "/intermediate/kernel")) + w(prefix + "/intermediate/bias"),
          true);
      out = nn::matmul(f, w(prefix + "/output/kernel")) + w(prefix + "/output/bias");
      h   = norm(h + out, prefix + "/output_layer_norm");
      states.push_back(h);
    }
    auto cls = nn::squeeze(nn::slice(h, {0, 0, 0}, {n, 1, 128}), {1});
    auto pooled =
        nn::tanh(nn::matmul(cls, w("pooler_transform/kernel")) + w("pooler_transform/bias"));
    return {
        {"bert_encoder", pooled},
        {"bert_encoder_1", states[0]},
        {"bert_encoder_2", states[1]},
        {"bert_encoder_3", pooled},
        {"bert_encoder_4", states[1]}};
  }
};
} // namespace pretrained_bert

} // namespace datamunge::nn::models
