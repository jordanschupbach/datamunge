#include <gtest/gtest.h>

#include <datamunge/algorithms/lz77.hpp>
#include <datamunge/algorithms/lzw.hpp>
#include <random>
using namespace datamunge::algorithms;
namespace {
std::vector<int> phrase_reference(const std::string& input) {
  std::vector<std::string> dict;
  for (int i = 0; i < 256; ++i)
    dict.emplace_back(1, char(i));
  std::vector<int> out;
  for (std::size_t pos = 0; pos < input.size();) {
    std::size_t length = 0;
    int         best   = -1;
    for (std::size_t i = 0; i < dict.size(); ++i)
      if (dict[i].size() > length && input.compare(pos, dict[i].size(), dict[i]) == 0) {
        best   = int(i);
        length = dict[i].size();
      }
    out.push_back(best);
    if (pos + length < input.size())
      dict.push_back(input.substr(pos, length + 1));
    pos += length;
  }
  return out;
}
} // namespace
TEST(LZWTrace, ExhaustiveShortBinaryWordsAndActualLearnedEntries) {
  for (int n = 0; n <= 9; ++n)
    for (int word = 0; word < (1 << n); ++word) {
      std::string input(n, 'A');
      for (int i = 0; i < n; ++i)
        input[i] += (word >> i) & 1;
      auto e = lzw_encode_trace(input);
      EXPECT_EQ(e.analysis.codes, phrase_reference(input));
      EXPECT_EQ(e.analysis.codes, lzw_encode(input));
      auto d = lzw_decode_trace(e.analysis.codes);
      EXPECT_EQ(d.analysis.data, input);
      EXPECT_EQ(lzw_decode(e.analysis.codes), input);
      std::size_t entries = e.input.size(), codes = 0;
      for (auto& s : e.steps) {
        entries += 1 + s.prefix.size() + s.candidate.size();
        if (s.emitted_code >= 0)
          ++codes;
        if (!s.flush) {
          EXPECT_EQ(s.byte, static_cast<unsigned char>(input[s.input_index]));
          EXPECT_EQ(s.candidate, s.prefix + input[s.input_index]);
        }
      }
      EXPECT_EQ(entries, e.captured_entries);
      EXPECT_EQ(codes, e.analysis.codes.size());
      EXPECT_EQ(e.analysis.work.input_bytes, input.size());
      EXPECT_EQ(e.analysis.work.emissions, codes);
      std::size_t decoded_entries = d.input.size(), offset = 0;
      for (std::size_t k = 0; k < d.steps.size(); ++k) {
        auto& s = d.steps[k];
        EXPECT_EQ(s.output_begin, offset);
        EXPECT_EQ(s.entry, input.substr(offset, s.entry.size()));
        offset += s.entry.size();
        EXPECT_EQ(s.output_end, offset);
        if (k) {
          EXPECT_EQ(s.learned_entry, d.steps[k - 1].entry + s.entry.front());
          EXPECT_EQ(s.learned_code, 255 + k);
        }
        decoded_entries += 1 + s.entry.size() + s.learned_entry.size();
      }
      EXPECT_EQ(decoded_entries, d.captured_entries);
      EXPECT_EQ(offset, input.size());
    }
}
TEST(LZWTrace, OriginalFixturesSpecialCaseAndEveryByte) {
  auto classic = lzw_encode(lzw_classic_text());
  EXPECT_EQ(lzw_classic_text().size(), 24);
  EXPECT_EQ(
      classic,
      (std::vector<int>{84, 79, 66, 69, 79, 82, 78, 79, 84, 256, 258, 260, 265, 259, 261, 263}));
  EXPECT_EQ(lzw_decode(classic), lzw_classic_text());
  auto pangram = lz77_demo_text();
  EXPECT_EQ(lzw_encode(pangram).size(), 65);
  EXPECT_EQ(lzw_decode(lzw_encode(pangram)), pangram);
  auto special = lzw_decode_trace({65, 256, 257, 258});
  EXPECT_EQ(special.analysis.data, std::string(10, 'A'));
  EXPECT_EQ(special.analysis.work.special_cases, 3);
  EXPECT_EQ(special.analysis.work.dictionary_bytes, 265);
  for (std::size_t i = 1; i < special.steps.size(); ++i)
    EXPECT_TRUE(special.steps[i].special);
  std::string bytes;
  for (int i = 0; i < 256; ++i)
    bytes.push_back(char(i));
  bytes += bytes;
  EXPECT_EQ(lzw_decode(lzw_encode(bytes)), bytes);
  std::mt19937_64 rng(987);
  for (int run = 0; run < 200; ++run) {
    std::string s(rng() % 50, '\0');
    for (auto& c : s)
      c = char(rng() % 256);
    EXPECT_EQ(lzw_decode(lzw_encode(s)), s);
    EXPECT_EQ(lzw_encode(s), phrase_reference(s));
  }
}
TEST(LZWTrace, CompleteInvalidCodeGuardsCapsEmptyAndOwnership) {
  for (auto codes :
       std::vector<std::vector<int>>{{256}, {-1}, {65, 257}, {65, -1}, {65, 256, 258}}) {
    EXPECT_THROW(lzw_decode(codes), std::invalid_argument);
    EXPECT_THROW(lzw_decode_trace(codes, 1, 1), std::invalid_argument);
  }
  auto e = lzw_encode_trace("AAA");
  EXPECT_THROW(lzw_encode_trace("AAA", e.captured_entries - 1), std::length_error);
  EXPECT_EQ(lzw_encode_trace("AAA", e.captured_entries).captured_entries, e.captured_entries);
  auto d = lzw_decode_trace({65, 256});
  EXPECT_THROW(lzw_decode_trace({65, 256}, 2), std::length_error);
  EXPECT_EQ(lzw_decode_trace({65, 256}, 3, d.captured_entries).analysis.data, "AAA");
  EXPECT_THROW(lzw_decode_trace({65, 256}, 0, d.captured_entries - 1), std::length_error);
  std::string input = "AAA";
  auto        own   = lzw_encode_trace(input, 0);
  input[0]          = 'B';
  EXPECT_EQ(own.input, "AAA");
  auto codes   = own.analysis.codes;
  auto decoded = lzw_decode_trace(codes, 0, 0);
  codes[0]     = 66;
  EXPECT_EQ(decoded.input[0], 65);
  EXPECT_EQ(decoded.analysis.data, "AAA");
  EXPECT_TRUE(lzw_encode_trace("").steps.empty());
  EXPECT_TRUE(lzw_decode_trace({}).steps.empty());
  EXPECT_EQ(lzw_decode_analyze({}).work.dictionary_bytes, 0);
}
