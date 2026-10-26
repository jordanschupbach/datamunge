// Whitespace, UTF-8 character, learned BPE and vocabulary-driven WordPiece
// (port of tflow's common/data/tokenization.h; PadTokens returns nn tensors).
//
// Whitespace splitting uses ASCII space/tab/newline characters and preserves
// case and punctuation. Character tokens are Unicode scalar values (not bytes
// or grapheme clusters); malformed, overlong and surrogate UTF-8 is rejected.
// No Unicode normalization, accent stripping or language-specific splitting.
//
// BPE learns weighted adjacent-pair frequencies within words, including an
// explicit </w> end marker. Ties break lexicographically. Apply learned merges
// in their ranked order; unseen characters become [UNK]. This clear reference
// implementation targets small corpora, not large tokenizer-training jobs.
// WordPiece uses longest-match-first with ## continuation pieces and replaces
// a whole word by [UNK] if any suffix fails. Its vocabulary is supplied, not
// learned here. IDs are stable vocabulary order, with PAD=0 and UNK=1.
#pragma once

#include <algorithm>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace data::text {

inline std::vector<std::string> Characters(const std::string& text) {
  std::vector<std::string> tokens;
  for (size_t i = 0; i < text.size();) {
    const auto lead = static_cast<uint8_t>(text[i]);
    size_t count = 0;
    uint32_t value = 0, minimum = 0;
    if (lead < 0x80) { count = 1; value = lead; }
    else if (lead >= 0xc2 && lead <= 0xdf) { count = 2; value = lead & 31; minimum = 0x80; }
    else if (lead >= 0xe0 && lead <= 0xef) { count = 3; value = lead & 15; minimum = 0x800; }
    else if (lead >= 0xf0 && lead <= 0xf4) { count = 4; value = lead & 7; minimum = 0x10000; }
    else throw std::invalid_argument("invalid UTF-8 leading byte");
    if (text.size() - i < count) throw std::invalid_argument("truncated UTF-8");
    for (size_t j = 1; j < count; ++j) {
      const auto byte = static_cast<uint8_t>(text[i + j]);
      if ((byte & 0xc0) != 0x80) throw std::invalid_argument("invalid UTF-8 continuation");
      value = (value << 6) | (byte & 63);
    }
    if (value < minimum || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) throw std::invalid_argument("invalid Unicode scalar");
    tokens.push_back(text.substr(i, count)); i += count;
  }
  return tokens;
}

inline std::vector<std::string> Whitespace(const std::string& text) {
  Characters(text); // validate UTF-8 even though splitting operates on ASCII bytes
  std::vector<std::string> words;
  size_t start = 0;
  while ((start = text.find_first_not_of(" \t\r\n\f\v", start)) != std::string::npos) {
    const size_t end = text.find_first_of(" \t\r\n\f\v", start);
    words.push_back(text.substr(start, end == std::string::npos ? end : end - start));
    if (end == std::string::npos) break;
    start = end;
  }
  return words;
}

class Bpe {
 public:
  using Pair = std::pair<std::string, std::string>;
  static Bpe Train(const std::map<std::string, uint64_t>& frequencies, size_t max_merges, uint64_t min_count = 2) {
    if (frequencies.empty() || !min_count) throw std::invalid_argument("BPE needs a corpus and positive minimum count");
    Bpe model;
    std::vector<std::pair<std::vector<std::string>, uint64_t>> words;
    for (const auto& entry : frequencies) {
      if (!entry.second || Whitespace(entry.first) != std::vector<std::string>{entry.first}) throw std::invalid_argument("BPE corpus keys must be nonempty words");
      auto symbols = Characters(entry.first);
      model.alphabet_.insert(symbols.begin(), symbols.end());
      symbols.push_back("</w>"); words.emplace_back(std::move(symbols), entry.second);
    }
    for (size_t iteration = 0; iteration < max_merges; ++iteration) {
      std::map<Pair, uint64_t> counts;
      for (const auto& word : words) for (size_t i = 1; i < word.first.size(); ++i) {
        auto& count = counts[{word.first[i - 1], word.first[i]}];
        if (word.second > std::numeric_limits<uint64_t>::max() - count) throw std::invalid_argument("BPE pair frequency overflow");
        count += word.second;
      }
      if (counts.empty()) break;
      auto best = counts.begin();
      for (auto it = counts.begin(); it != counts.end(); ++it) if (it->second > best->second) best = it;
      if (best->second < min_count) break;
      model.merges_.push_back(best->first);
      for (auto& word : words) word.first = Merge(word.first, best->first);
    }
    return model;
  }
  std::vector<std::string> EncodeWord(const std::string& word) const {
    auto symbols = Characters(word);
    if (symbols.empty()) return {};
    for (auto& symbol : symbols) if (!alphabet_.count(symbol)) symbol = "[UNK]";
    symbols.push_back("</w>");
    for (const auto& pair : merges_) symbols = Merge(symbols, pair);
    return symbols;
  }
  std::string Save() const {
    std::ostringstream out;
    out << "TFLOW_BPE_V1 " << alphabet_.size() << ' ' << merges_.size() << '\n';
    for (const auto& symbol : alphabet_) out << std::quoted(symbol) << '\n';
    for (const auto& pair : merges_) out << std::quoted(pair.first) << ' ' << std::quoted(pair.second) << '\n';
    return out.str();
  }
  static Bpe Load(const std::string& serialized) {
    std::istringstream in(serialized);
    std::string magic;
    size_t chars = 0, merges = 0;
    if (!(in >> magic >> chars >> merges) || magic != "TFLOW_BPE_V1" || !chars || chars > 1000000 || merges > 1000000) throw std::invalid_argument("invalid BPE model header");
    Bpe model;
    for (size_t i = 0; i < chars; ++i) {
      std::string symbol;
      if (!(in >> std::quoted(symbol)) || Characters(symbol).size() != 1 || !model.alphabet_.insert(symbol).second) throw std::invalid_argument("invalid BPE alphabet");
    }
    auto known = model.alphabet_; known.insert("</w>");
    for (size_t i = 0; i < merges; ++i) {
      Pair pair;
      if (!(in >> std::quoted(pair.first) >> std::quoted(pair.second)) || !known.count(pair.first) || !known.count(pair.second)) throw std::invalid_argument("invalid BPE merge dependency");
      known.insert(pair.first + pair.second); model.merges_.push_back(std::move(pair));
    }
    in >> std::ws;
    if (!in.eof()) throw std::invalid_argument("trailing BPE model data");
    return model;
  }
  const std::vector<Pair>& merges() const { return merges_; }
 private:
  static std::vector<std::string> Merge(const std::vector<std::string>& symbols, const Pair& pair) {
    std::vector<std::string> result;
    for (size_t i = 0; i < symbols.size(); ++i) {
      if (i + 1 < symbols.size() && symbols[i] == pair.first && symbols[i + 1] == pair.second) {
        result.push_back(pair.first + pair.second); ++i;
      } else result.push_back(symbols[i]);
    }
    return result;
  }
  std::set<std::string> alphabet_;
  std::vector<Pair> merges_;
};

class WordPiece {
 public:
  explicit WordPiece(std::vector<std::string> vocabulary, size_t max_chars = 100) : vocabulary_(std::move(vocabulary)), max_chars_(max_chars) {
    if (vocabulary_.size() < 2 || vocabulary_[0] != "[PAD]" || vocabulary_[1] != "[UNK]" || !max_chars) throw std::invalid_argument("WordPiece needs PAD=0, UNK=1 and positive max length");
    for (size_t i = 0; i < vocabulary_.size(); ++i) {
      Characters(vocabulary_[i]);
      if (vocabulary_[i].empty() || !ids_.emplace(vocabulary_[i], i).second) throw std::invalid_argument("empty/duplicate WordPiece token");
    }
  }
  std::vector<std::string> EncodeWord(const std::string& word) const {
    const auto chars = Characters(word);
    if (chars.size() > max_chars_) return {"[UNK]"};
    std::vector<std::string> tokens;
    for (size_t start = 0; start < chars.size();) {
      std::string found;
      size_t end = chars.size();
      for (; end > start; --end) {
        std::string candidate = start ? "##" : "";
        for (size_t i = start; i < end; ++i) candidate += chars[i];
        if (ids_.count(candidate)) { found = std::move(candidate); break; }
      }
      if (found.empty()) return {"[UNK]"};
      tokens.push_back(std::move(found)); start = end;
    }
    return tokens;
  }
  std::vector<int32_t> Encode(const std::string& sentence) const {
    std::vector<int32_t> ids;
    for (const auto& word : Whitespace(sentence)) for (const auto& token : EncodeWord(word)) ids.push_back(ids_.at(token));
    return ids;
  }
  const std::vector<std::string>& vocabulary() const { return vocabulary_; }
 private:
  std::vector<std::string> vocabulary_;
  std::map<std::string, int32_t> ids_;
  size_t max_chars_;
};

/// Right-truncated, right-padded token IDs: ids int32 [N, max_length]
/// (PAD = 0) and mask float32 [N, max_length] (1 for real tokens).
struct TokenBatch { datamunge::nn::Tensor ids, mask; };
inline TokenBatch PadTokens(const std::vector<std::vector<int32_t>>& sequences, size_t max_length) {
  if (sequences.empty() || !max_length) throw std::invalid_argument("token batching needs sequences and positive max length");
  std::vector<int32_t> ids(sequences.size() * max_length, 0);
  std::vector<float> mask(sequences.size() * max_length, 0.f);
  for (size_t i = 0; i < sequences.size(); ++i) for (size_t j = 0; j < std::min(max_length, sequences[i].size()); ++j) {
    if (sequences[i][j] < 0) throw std::invalid_argument("negative token ID");
    ids[i * max_length + j] = sequences[i][j]; mask[i * max_length + j] = 1;
  }
  const datamunge::nn::Shape flat{int64_t(ids.size())}, shape{int64_t(sequences.size()), int64_t(max_length)};
  return {datamunge::nn::from_vector(ids, flat).reshape(shape), datamunge::nn::from_vector(mask, flat).reshape(shape)};
}

}  // namespace data::text
