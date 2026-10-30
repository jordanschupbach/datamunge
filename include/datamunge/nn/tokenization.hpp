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

/// @file tokenization.hpp
/// @brief UTF-8, byte-pair and WordPiece tokenization with masked batching.

#include <algorithm>
#include <cstdint>
#include <datamunge/nn/ops.hpp>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/// @brief UTF-8 token boundaries, learned merges and masked integer batches.
namespace datamunge::nn::data::text {

/// @brief Split valid UTF-8 into Unicode scalar strings.
/// @param text UTF-8 bytes, without normalisation.
/// @return Owned token per scalar; empty input gives no tokens.
/// @throws std::invalid_argument On malformed, overlong, surrogate or out-of-range UTF-8.
///
/// @par Example
/// @code
/// namespace t = dm::nn::data::text;
/// const auto chars = t::Characters("café"); // 4 scalar values
/// @endcode
/// @see Whitespace()
inline std::vector<std::string> Characters(const std::string& text) {
  std::vector<std::string> tokens;
  for (size_t i = 0; i < text.size();) {
    const auto lead  = static_cast<uint8_t>(text[i]);
    size_t     count = 0;
    uint32_t   value = 0, minimum = 0;
    if (lead < 0x80) {
      count = 1;
      value = lead;
    } else if (lead >= 0xc2 && lead <= 0xdf) {
      count   = 2;
      value   = lead & 31;
      minimum = 0x80;
    } else if (lead >= 0xe0 && lead <= 0xef) {
      count   = 3;
      value   = lead & 15;
      minimum = 0x800;
    } else if (lead >= 0xf0 && lead <= 0xf4) {
      count   = 4;
      value   = lead & 7;
      minimum = 0x10000;
    } else
      throw std::invalid_argument("invalid UTF-8 leading byte");
    if (text.size() - i < count)
      throw std::invalid_argument("truncated UTF-8");
    for (size_t j = 1; j < count; ++j) {
      const auto byte = static_cast<uint8_t>(text[i + j]);
      if ((byte & 0xc0) != 0x80)
        throw std::invalid_argument("invalid UTF-8 continuation");
      value = (value << 6) | (byte & 63);
    }
    if (value < minimum || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff))
      throw std::invalid_argument("invalid Unicode scalar");
    tokens.push_back(text.substr(i, count));
    i += count;
  }
  return tokens;
}

/// @brief Split valid UTF-8 at ASCII whitespace while preserving case and punctuation.
/// @param text UTF-8 text; ASCII space, tab, CR, LF, formfeed and vertical tab delimit words.
/// @return Nonempty words in order.
/// @throws std::invalid_argument On invalid UTF-8.
///
/// @par Example
/// @code
/// namespace t = dm::nn::data::text;
/// const auto words = t::Whitespace("a  b"); // {"a","b"}
/// @endcode
/// @see Characters()
inline std::vector<std::string> Whitespace(const std::string& text) {
  Characters(text); // validate UTF-8 even though splitting operates on ASCII bytes
  std::vector<std::string> words;
  size_t                   start = 0;
  while ((start = text.find_first_not_of(" \t\r\n\f\v", start)) != std::string::npos) {
    const size_t end = text.find_first_of(" \t\r\n\f\v", start);
    words.push_back(text.substr(start, end == std::string::npos ? end : end - start));
    if (end == std::string::npos)
      break;
    start = end;
  }
  return words;
}

/// @brief Small-corpus frequency-weighted byte-pair encoder over Unicode scalar strings.
///
/// End-of-word markers are explicit. Pair-frequency ties use lexicographic order.
/// Training rescans all word tokens at each merge; encoding applies ranked merges.
/// Unknown scalar characters become [UNK]; there is no Unicode normalisation.
///
/// @par Example
/// @code
/// namespace t = dm::nn::data::text;
/// const auto b = t::Bpe::Train({{"aaa",3},{"ab",1}}, 1);
/// const auto pieces = b.EncodeWord("aa"); // {"aa","</w>"}
/// @endcode
///
/// @par References
/// - Sennrich, Haddow and Birch (2016), https://aclanthology.org/P16-1162/
/// @see WordPiece
class Bpe {
 public:
  /// @brief Two adjacent token strings defining one ranked merge.
  using Pair = std::pair<std::string, std::string>;
  /// @brief Learn ranked non-overlapping adjacent-pair merges.
  /// @param frequencies Nonempty map of nonempty single words to positive integer weights.
  /// @param max_merges Maximum merge count; zero retains the initial alphabet.
  /// @param min_count Minimum weighted count, positive; default 2.
  /// @return Owned learned alphabet and merge list.
  /// @throws std::invalid_argument On invalid corpus, UTF-8, zero counts or count overflow.
  ///
  /// @par Example
  /// @code
  /// namespace t = dm::nn::data::text;
  /// const auto b = t::Bpe::Train({{"aaa",3},{"ab",1}},1);
  /// @endcode
  /// @see EncodeWord()
  static Bpe Train(
      const std::map<std::string, uint64_t>& frequencies,
      size_t                                 max_merges,
      uint64_t                               min_count = 2) {
    if (frequencies.empty() || !min_count)
      throw std::invalid_argument("BPE needs a corpus and positive minimum count");
    Bpe                                                        model;
    std::vector<std::pair<std::vector<std::string>, uint64_t>> words;
    for (const auto& entry : frequencies) {
      if (!entry.second || Whitespace(entry.first) != std::vector<std::string>{entry.first})
        throw std::invalid_argument("BPE corpus keys must be nonempty words");
      auto symbols = Characters(entry.first);
      model.alphabet_.insert(symbols.begin(), symbols.end());
      symbols.push_back("</w>");
      words.emplace_back(std::move(symbols), entry.second);
    }
    for (size_t iteration = 0; iteration < max_merges; ++iteration) {
      std::map<Pair, uint64_t> counts;
      for (const auto& word : words)
        for (size_t i = 1; i < word.first.size(); ++i) {
          auto& count = counts[{word.first[i - 1], word.first[i]}];
          if (word.second > std::numeric_limits<uint64_t>::max() - count)
            throw std::invalid_argument("BPE pair frequency overflow");
          count += word.second;
        }
      if (counts.empty())
        break;
      auto best = counts.begin();
      for (auto it = counts.begin(); it != counts.end(); ++it)
        if (it->second > best->second)
          best = it;
      if (best->second < min_count)
        break;
      model.merges_.push_back(best->first);
      for (auto& word : words)
        word.first = Merge(word.first, best->first);
    }
    return model;
  }
  /// @brief Encode one word using learned merge order.
  /// @param word UTF-8 input; caller supplies one word.
  /// @return Token strings including the end marker; empty word gives none.
  /// @throws std::invalid_argument On invalid UTF-8.
  ///
  /// @par Example
  /// @code
  /// namespace t = dm::nn::data::text;
  /// const auto b = t::Bpe::Train({{"aaa",3}},1);
  /// const auto tokens = b.EncodeWord("aa");
  /// @endcode
  /// @see Train()
  std::vector<std::string> EncodeWord(const std::string& word) const {
    auto symbols = Characters(word);
    if (symbols.empty())
      return {};
    for (auto& symbol : symbols)
      if (!alphabet_.count(symbol))
        symbol = "[UNK]";
    symbols.push_back("</w>");
    for (const auto& pair : merges_)
      symbols = Merge(symbols, pair);
    return symbols;
  }
  /// @brief Serialize the alphabet and ranked merges to a quoted text format.
  ///
  /// The legacy magic is retained for compatibility with existing example artifacts.
  /// @return Owned TFLOW_BPE_V1 text; vocabulary order is deterministic.
  ///
  /// @par Example
  /// @code
  /// namespace t = dm::nn::data::text;
  /// const auto b = t::Bpe::Train({{"aaa",3}},1);
  /// const auto copy = t::Bpe::Load(b.Save());
  /// @endcode
  /// @see Load()
  std::string Save() const {
    std::ostringstream out;
    out << "TFLOW_BPE_V1 " << alphabet_.size() << ' ' << merges_.size() << '\n';
    for (const auto& symbol : alphabet_)
      out << std::quoted(symbol) << '\n';
    for (const auto& pair : merges_)
      out << std::quoted(pair.first) << ' ' << std::quoted(pair.second) << '\n';
    return out.str();
  }
  /// @brief Restore an encoder after checking alphabet and merge dependencies.
  /// @param serialized Complete TFLOW_BPE_V1 text, including quoted strings.
  /// @return Owned equivalent encoder.
  /// @throws std::invalid_argument On malformed magic, counts, UTF-8, alphabet, dependencies or
  /// trailing data.
  ///
  /// @par Example
  /// @code
  /// namespace t = dm::nn::data::text;
  /// const auto b = t::Bpe::Train({{"aaa",3}},1);
  /// const auto copy = t::Bpe::Load(b.Save());
  /// @endcode
  /// @see Save()
  static Bpe Load(const std::string& serialized) {
    std::istringstream in(serialized);
    std::string        magic;
    size_t             chars = 0, merges = 0;
    if (!(in >> magic >> chars >> merges) || magic != "TFLOW_BPE_V1" || !chars || chars > 1000000 ||
        merges > 1000000)
      throw std::invalid_argument("invalid BPE model header");
    Bpe model;
    for (size_t i = 0; i < chars; ++i) {
      std::string symbol;
      if (!(in >> std::quoted(symbol)) || Characters(symbol).size() != 1 ||
          !model.alphabet_.insert(symbol).second)
        throw std::invalid_argument("invalid BPE alphabet");
    }
    auto known = model.alphabet_;
    known.insert("</w>");
    for (size_t i = 0; i < merges; ++i) {
      Pair pair;
      if (!(in >> std::quoted(pair.first) >> std::quoted(pair.second)) ||
          !known.count(pair.first) || !known.count(pair.second))
        throw std::invalid_argument("invalid BPE merge dependency");
      known.insert(pair.first + pair.second);
      model.merges_.push_back(std::move(pair));
    }
    in >> std::ws;
    if (!in.eof())
      throw std::invalid_argument("trailing BPE model data");
    return model;
  }
  /// @brief Inspect ranked merges without mutation.
  /// @return Borrowed list valid for the encoder lifetime.
  const std::vector<Pair>& merges() const { return merges_; }

 private:
  /// @brief Replace non-overlapping occurrences of one adjacent pair.
  /// @param symbols Input token sequence.
  /// @param pair Adjacent strings to concatenate.
  /// @return New sequence after a left-to-right pass.
  static std::vector<std::string> Merge(const std::vector<std::string>& symbols, const Pair& pair) {
    std::vector<std::string> result;
    for (size_t i = 0; i < symbols.size(); ++i) {
      if (i + 1 < symbols.size() && symbols[i] == pair.first && symbols[i + 1] == pair.second) {
        result.push_back(pair.first + pair.second);
        ++i;
      } else
        result.push_back(symbols[i]);
    }
    return result;
  }
  std::set<std::string> alphabet_; ///< Valid Unicode scalar strings learned from corpus words.
  std::vector<Pair>     merges_;   ///< Ranked adjacent-pair rules in application order.
};

/// @brief Fixed-vocabulary greedy longest-match WordPiece encoder.
///
/// PAD=0 and UNK=1. Continuation pieces start with ##. Any failed suffix
/// replaces the whole word by UNK. Vocabulary is supplied, not trained.
///
/// @par Example
/// @code
/// namespace t = dm::nn::data::text;
/// const t::WordPiece w({"[PAD]","[UNK]","play","##ing"});
/// const auto ids = w.Encode("playing"); // {2,3}
/// @endcode
///
/// @par References
/// - Devlin et al., BERT tokenizer,
/// https://github.com/google-research/bert/blob/master/tokenization.py
/// @see Bpe
class WordPiece {
 public:
  /// @brief Validate and store a caller-ordered vocabulary.
  /// @param vocabulary Unique nonempty valid UTF-8 pieces with PAD and UNK first.
  /// @param max_chars Maximum Unicode scalar count per word, positive; default 100.
  /// @throws std::invalid_argument On invalid vocabulary, UTF-8 or length.
  ///
  /// @par Example
  /// @code
  /// namespace t = dm::nn::data::text;
  /// const t::WordPiece w({"[PAD]","[UNK]","a","##b"});
  /// @endcode
  /// @see Encode()
  explicit WordPiece(std::vector<std::string> vocabulary, size_t max_chars = 100)
      : vocabulary_(std::move(vocabulary)), max_chars_(max_chars) {
    if (vocabulary_.size() < 2 || vocabulary_[0] != "[PAD]" || vocabulary_[1] != "[UNK]" ||
        !max_chars)
      throw std::invalid_argument("WordPiece needs PAD=0, UNK=1 and positive max length");
    for (size_t i = 0; i < vocabulary_.size(); ++i) {
      Characters(vocabulary_[i]);
      if (vocabulary_[i].empty() || !ids_.emplace(vocabulary_[i], i).second)
        throw std::invalid_argument("empty/duplicate WordPiece token");
    }
  }
  /// @brief Greedily split a single word into vocabulary pieces.
  /// @param word UTF-8 input; no normalisation or lowercasing.
  /// @return Pieces, or one UNK on a failed suffix or exceeded limit; empty word gives none.
  /// @throws std::invalid_argument On invalid UTF-8.
  ///
  /// @par Example
  /// @code
  /// namespace t = dm::nn::data::text;
  /// const t::WordPiece w({"[PAD]","[UNK]","play","##ing"});
  /// const auto pieces = w.EncodeWord("playing");
  /// @endcode
  /// @see Encode()
  std::vector<std::string> EncodeWord(const std::string& word) const {
    const auto chars = Characters(word);
    if (chars.size() > max_chars_)
      return {"[UNK]"};
    std::vector<std::string> tokens;
    for (size_t start = 0; start < chars.size();) {
      std::string found;
      size_t      end = chars.size();
      for (; end > start; --end) {
        std::string candidate = start ? "##" : "";
        for (size_t i = start; i < end; ++i)
          candidate += chars[i];
        if (ids_.count(candidate)) {
          found = std::move(candidate);
          break;
        }
      }
      if (found.empty())
        return {"[UNK]"};
      tokens.push_back(std::move(found));
      start = end;
    }
    return tokens;
  }
  /// @brief Split ASCII-whitespace words and concatenate their WordPiece IDs.
  /// @param sentence Valid UTF-8 sentence, preserving case and punctuation.
  /// @return IDs in vocabulary order; no padding or special sentence tokens.
  /// @throws std::invalid_argument On invalid UTF-8.
  ///
  /// @par Example
  /// @code
  /// namespace t = dm::nn::data::text;
  /// const t::WordPiece w({"[PAD]","[UNK]","a"});
  /// const auto ids = w.Encode("a z"); // {2,1}
  /// @endcode
  /// @see PadTokens()
  std::vector<int32_t> Encode(const std::string& sentence) const {
    std::vector<int32_t> ids;
    for (const auto& word : Whitespace(sentence))
      for (const auto& token : EncodeWord(word))
        ids.push_back(ids_.at(token));
    return ids;
  }
  /// @brief Inspect the stable caller-supplied token order.
  /// @return Borrowed pieces; index equals token ID, valid for the object lifetime.
  const std::vector<std::string>& vocabulary() const { return vocabulary_; }

 private:
  std::vector<std::string>       vocabulary_; ///< Caller-ordered pieces; array index is token ID.
  std::map<std::string, int32_t> ids_;        ///< Unique exact piece-to-ID lookup.
  size_t max_chars_; ///< Maximum scalar count accepted before whole-word UNK.
};

/// Right-truncated, right-padded token IDs: ids int32 [N, max_length]
/// (PAD = 0) and mask float32 [N, max_length] (1 for real tokens).
/// @brief Owned right-padded token IDs and explicit mask.
///
/// ids is Int32 [N,L] and mask Float32 [N,L]. Mask is one for retained
/// tokens, including a caller-supplied zero ID, and zero for padding.
///
/// @par Example
/// @code
/// namespace t = dm::nn::data::text;
/// const auto batch = t::PadTokens({{2,3},{}},3); // [2,3] IDs and mask
/// @endcode
/// @see PadTokens()
struct TokenBatch {
  datamunge::nn::Tensor ids;  ///< Int32 [N,L] retained IDs and zero padding.
  datamunge::nn::Tensor mask; ///< Float32 [N,L], one at retained token positions.
};
/// @brief Right-truncate and right-pad integer token sequences.
///
/// No vocabulary range check is possible here. Discarded suffix IDs are not inspected.
/// @param sequences Nonempty sequence list; individual sequences may be empty.
/// @param max_length Positive retained length and output width.
/// @return Owned Int32 IDs and Float32 mask; padding ID is zero.
/// @throws std::invalid_argument On empty batch, zero width or a negative retained ID.
///
/// @par Example
/// @code
/// namespace t = dm::nn::data::text;
/// const auto b = t::PadTokens({{2,3},{1}},3);
/// @endcode
/// @see TokenBatch
inline TokenBatch PadTokens(const std::vector<std::vector<int32_t>>& sequences, size_t max_length) {
  if (sequences.empty() || !max_length)
    throw std::invalid_argument("token batching needs sequences and positive max length");
  std::vector<int32_t> ids(sequences.size() * max_length, 0);
  std::vector<float>   mask(sequences.size() * max_length, 0.f);
  for (size_t i = 0; i < sequences.size(); ++i)
    for (size_t j = 0; j < std::min(max_length, sequences[i].size()); ++j) {
      if (sequences[i][j] < 0)
        throw std::invalid_argument("negative token ID");
      ids[i * max_length + j]  = sequences[i][j];
      mask[i * max_length + j] = 1;
    }
  const datamunge::nn::Shape flat{int64_t(ids.size())},
      shape{int64_t(sequences.size()), int64_t(max_length)};
  return {
      datamunge::nn::from_vector(ids, flat).reshape(shape),
      datamunge::nn::from_vector(mask, flat).reshape(shape)};
}

} // namespace datamunge::nn::data::text
