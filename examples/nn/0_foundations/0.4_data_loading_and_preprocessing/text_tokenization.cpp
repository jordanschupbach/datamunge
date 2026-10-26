// CMD: just run-nn text_tokenization
// Whitespace, Unicode characters, learned BPE and greedy WordPiece.
//
// Port of tflow's text_tokenization. Learn ten BPE merges from a weighted toy
// vocabulary, inspect segmentation of unseen word combinations, and reload a
// saved merge model. Test Unicode scalar boundaries and invalid UTF-8,
// deterministic tie-breaking, WordPiece continuation markers, longest matches
// and whole-word unknown fallback.
//
// Convert WordPiece tokens to stable IDs, truncate/right-pad a batch and feed
// it to an nn embedding lookup with masked mean pooling. An empty sentence
// must produce a zero vector even when the PAD embedding is nonzero.
// Preprocessing preserves case/punctuation and uses ASCII whitespace, without
// Unicode normalization. This is not a full BERT basic tokenizer.
// References: https://arxiv.org/abs/1508.07909 and the original BERT tokenizer
// at https://github.com/google-research/bert/blob/master/tokenization.py.
//
// Differences from TF: the tokenizers live in common/tokenization.hpp
// (tflow's common/data/tokenization.h) and PadTokens returns nn tensors; the
// lookup and pooling are eager nn ops (gather, mul, sum, maximum, div). The
// BPE model is written to output/nn/fixtures/text/bpe.model.

#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "common/binary.hpp"
#include "common/checks.hpp"
#include "common/paths.hpp"
#include "common/tokenization.hpp"

namespace nn    = datamunge::nn;
namespace token = data::text;
using nn::Tensor;

namespace {

void Print(const std::vector<std::string>& tokens) {
  for (const auto& value : tokens) std::cout << '[' << value << "] ";
  std::cout << '\n';
}

void Basic() {
  const auto words = token::Whitespace("  café\tnewer\nlowest  ");
  check::Require(words == std::vector<std::string>({"café", "newer", "lowest"}), "whitespace boundaries");
  const auto chars = token::Characters("café🙂");
  check::Require(chars == std::vector<std::string>({"c", "a", "f", "é", "🙂"}), "Unicode scalar tokenization");
  check::Require(token::Characters("e\u0301").size() == 2, "combining marks remain distinct scalars");
  int failures = 0;
  for (const auto& invalid : std::vector<std::string>{"\xc0\x80", "\xed\xa0\x80", "\xf4\x90\x80\x80", "\xe2\x82", "\x80", "\xe2x\x80"}) {
    bool rejected = false;
    try { token::Characters(invalid); } catch (const std::invalid_argument&) { rejected = true; }
    check::Require(rejected, "malformed UTF-8 must fail"); ++failures;
  }
  std::cout << "  whitespace: "; Print(words);
  std::cout << "  characters: "; Print(chars);
  std::cout << "  " << failures << " invalid UTF-8 cases rejected\n";
}

void Bpe() {
  const auto bpe = token::Bpe::Train({{"low", 5}, {"lower", 2}, {"newest", 6}, {"widest", 3}}, 10);
  check::Require(bpe.merges().size() == 10 && bpe.merges()[0] == token::Bpe::Pair{"e", "s"}, "weighted pair counts and deterministic tie break");
  std::cout << "  learned merges:";
  for (const auto& pair : bpe.merges()) std::cout << " (" << pair.first << ',' << pair.second << ')';
  std::cout << '\n';
  const auto path = paths::OutputDir("fixtures/text") / "bpe.model";
  const auto serialized = bpe.Save();
  data::WriteBinary(path, data::Bytes(serialized.begin(), serialized.end()));
  const auto bytes = data::ReadBinary(path);
  const auto restored = token::Bpe::Load(std::string(bytes.begin(), bytes.end()));
  check::Require(restored.Save() == serialized, "BPE persistence preserves merge ranks/alphabet");
  for (const std::string word : {"low", "lower", "lowest", "newer", "widest"}) {
    const auto pieces = bpe.EncodeWord(word);
    check::Require(restored.EncodeWord(word) == pieces, "reloaded BPE segmentation");
    std::string joined;
    for (const auto& piece : pieces) joined += piece;
    check::Require(joined == word + "</w>", "BPE preserves known character sequence and word boundary");
    std::cout << "  " << word << ": "; Print(pieces);
  }
  const auto unknown = bpe.EncodeWord("λ");
  check::Require(unknown == std::vector<std::string>({"[UNK]", "</w>"}), "BPE unseen character fallback");
  bool rejected = false;
  try { token::Bpe::Load(serialized + " garbage"); } catch (const std::invalid_argument&) { rejected = true; }
  check::Require(rejected, "invalid saved model rejected");
}

void WordPieceAndBatch() {
  const token::WordPiece wp({"[PAD]", "[UNK]", "low", "##est", "lower", "new", "##er", "wide", "##st", "caf", "##é", "##s"});
  check::Require(wp.EncodeWord("lowest") == std::vector<std::string>({"low", "##est"}), "WordPiece continuations");
  check::Require(wp.EncodeWord("lower") == std::vector<std::string>({"lower"}), "WordPiece longest match");
  check::Require(wp.EncodeWord("lowerx") == std::vector<std::string>({"[UNK]"}), "whole word fallback after failed suffix");
  check::Require(wp.EncodeWord("café") == std::vector<std::string>({"caf", "##é"}), "WordPiece Unicode boundaries");
  check::Require(wp.EncodeWord(std::string(101, 'a')) == std::vector<std::string>({"[UNK]"}), "maximum word length");
  for (const std::string word : {"lowest", "lower", "newer", "café", "lowerx"}) {
    std::cout << "  " << word << ": "; Print(wp.EncodeWord(word));
  }
  const auto batch = token::PadTokens({wp.Encode("lowest newer"), wp.Encode("unknown"), wp.Encode("")}, 3);
  check::Require(batch.ids.shape() == nn::Shape({3, 3}), "padded token shape");
  const auto ids = batch.ids.to_vector<int32_t>();
  check::Require(ids[0] == 2 && ids[1] == 3 && ids[2] == 5, "right truncation and stable IDs");
  check::Near(batch.mask.to_vector<float>()[3 + 1], 0, 0, "padding mask");
  const auto         vocab = static_cast<std::int64_t>(wp.vocabulary().size());
  std::vector<float> table(vocab * 2);
  for (std::int64_t i = 0; i < vocab; ++i) {
    table[2 * i]     = static_cast<float>(i);
    table[2 * i + 1] = static_cast<float>(100 + i);
  }
  const Tensor embeddings  = nn::from_vector(table, {vocab * 2}).reshape({vocab, 2});
  const Tensor lookup      = nn::gather(embeddings, batch.ids, 0);
  const Tensor mask        = nn::expand_dims(batch.mask, 2);
  const Tensor sum         = nn::sum(nn::mul(lookup, mask), {1});
  const Tensor denominator = nn::maximum(nn::sum(mask, {1}), nn::scalar(1));
  const auto   pooled      = nn::div(sum, denominator).to_vector<float>();
  check::Near(pooled[0], 10. / 3, 1e-6, "truncated mean embedding");
  check::Near(pooled[1], 100 + 10. / 3, 1e-5, "embedding feature two");
  check::Near(pooled[2], 1, 0, "unknown embedding");
  for (int c = 0; c < 2; ++c) check::Near(pooled[4 + c], 0, 0, "empty sentence ignores PAD embedding");
  std::cout << "  token batch [3,3], retained lengths=(3,1,0)\n"
            << "  masked mean embeddings=(" << pooled[0] << ", " << pooled[1]
            << "), (1,101), (0,0)\n";
}

}  // namespace

int main() {
  try {
    nn::set_num_threads(2);
    std::cout << "== Whitespace and characters\n";
    Basic();
    std::cout << "\n== Learned BPE\n";
    Bpe();
    std::cout << "\n== WordPiece and nn token batching\n";
    WordPieceAndBatch();
    std::cout << "All tokenization checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
