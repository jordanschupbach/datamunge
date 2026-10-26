// ASCII subset of BERT uncased BasicTokenizer + greedy WordPiece and packing.
// Non-ASCII input is rejected explicitly; this is not a Unicode tokenizer.
#pragma once
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "common/checks.hpp"
namespace wordpiece {
namespace nn = datamunge::nn;
class AsciiWordPiece {
 public:
  std::vector<std::string>   vocabulary;
  std::map<std::string, int> lookup;
  explicit AsciiWordPiece(const std::filesystem::path& path) {
    std::ifstream f(path);
    check::Require(bool(f), "WordPiece vocabulary file");
    std::string token;
    while (std::getline(f, token)) {
      lookup.emplace(token, vocabulary.size());
      vocabulary.push_back(token);
    }
    check::Require(
        vocabulary.size() == 30522 && vocabulary[0] == "[PAD]" && vocabulary[100] == "[UNK]" &&
            vocabulary[101] == "[CLS]" && vocabulary[102] == "[SEP]" && vocabulary[103] == "[MASK]",
        "published BERT uncased vocabulary and IDs");
  }
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
      check::Require(
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
  struct Packed {
    nn::Tensor ids, mask, types;
  };
  Packed Pack(
      const std::string& first,
      const std::string& second = "",
      bool               pair   = false,
      int                width  = 32) const {
    check::Require(
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
