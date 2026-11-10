#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <datamunge/datasets/assets.hpp>
#include <datamunge/nn/sequence_vae_data.hpp>
#include <datamunge/nn/training.hpp>
namespace n  = datamunge::nn;
namespace m  = n::models;
namespace fs = std::filesystem;
namespace {
class SentenceGrammarData : public testing::Test {
 protected:
  fs::path        root;
  n::NamedTensors original;
  void            SetUp() override {
    static std::atomic<unsigned> serial{0};
    root = fs::temp_directory_path() /
           ("dm-sequence-grammar-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
            std::to_string(serial++));
    datamunge::datasets::prepare_example_data("sequence-vae-assets", root.string());
    root /= "sequence_vae";
    original = n::load_npz((root / "sentences.npz").string());
  }
  void TearDown() override {
    std::error_code ec;
    fs::remove_all(root.parent_path(), ec);
  }
  void save(const n::NamedTensors& records) {
    n::save_npz((root / "sentences.npz").string(), records);
  }
  void vocabulary(const std::string& text) {
    std::ofstream out(root / "vocabulary.txt");
    out << text;
    ASSERT_TRUE(out);
  }
  n::Tensor& field(n::NamedTensors& records, const std::string& name) {
    for (auto& r : records)
      if (r.first == name)
        return r.second;
    throw std::runtime_error("test fixture field absent");
  }
};
} // namespace
TEST_F(SentenceGrammarData, FullCorpusOwnedPoolsAndOriginalFifteenArrayProtocol) {
  auto pools = m::LoadSequenceVaeData(root.string());
  ASSERT_EQ(pools.size(), 3u);
  EXPECT_EQ(pools[0].tokens.shape(), (n::Shape{288, 7}));
  EXPECT_EQ(pools[1].source.numel(), 72);
  EXPECT_EQ(pools[2].source.numel(), 72);
  auto arrays = m::SequenceDataArrays(pools);
  ASSERT_EQ(arrays.size(), 15u);
  for (const auto& [key, value] : arrays) {
    auto expected = m::sequence_grammar_detail::Find(original, key);
    EXPECT_TRUE(n::equal_bits(value, expected));
  }
  auto fit = pools[0];
  pools.clear();
  fs::remove(root / "sentences.npz");
  EXPECT_EQ(fit.tokens.size(0), 288);
  EXPECT_EQ(fit.factors.size(1), 6);
  for (std::int64_t row = 0; row < 288; ++row)
    EXPECT_TRUE(m::SequenceGrammarValid(fit.tokens, row));
}
TEST_F(SentenceGrammarData, LoaderPreservesValidPermutationsAndAcceptsConfigShapeAndExtras) {
  auto records             = original;
  field(records, "config") = n::reshape(field(records, "config"), {1, 2});
  records.emplace_back("unrelated", n::ones({3}));
  for (const std::string split : {"fit", "validation", "test"}) {
    const int                 count = split == "fit" ? 288 : 72;
    std::vector<std::int64_t> rows;
    for (int i = count - 1; i >= 0; --i)
      rows.push_back(i);
    for (const std::string suffix :
         {"_tokens", "_decoder_inputs", "_lengths", "_factors", "_source_ids"})
      field(records, split + suffix) = n::take_rows(field(records, split + suffix), rows);
  }
  save(records);
  const auto pools = m::LoadSequenceVaeData(root.string());
  EXPECT_TRUE(n::equal_bits(pools[0].source, field(records, "fit_source_ids")));
  EXPECT_GT(pools[0].source.to_vector<int>()[0], pools[0].source.to_vector<int>()[287]);
  EXPECT_EQ(m::SequenceDataArrays(pools).size(), 15u);
  auto strided      = pools;
  strided[0].tokens = n::transpose(n::transpose(strided[0].tokens, {1, 0}).contiguous(), {1, 0});
  EXPECT_TRUE(n::equal_bits(m::SequenceDataArrays(strided)[0].second, pools[0].tokens));
}
TEST_F(SentenceGrammarData, EveryRequiredFieldIsCheckedAndInvalidCorporaReturnNoPartialResult) {
  for (const std::string name :
       {"config",
        "fit_tokens",
        "validation_tokens",
        "test_tokens",
        "fit_decoder_inputs",
        "fit_lengths",
        "fit_factors",
        "fit_source_ids"}) {
    SCOPED_TRACE(name);
    auto records         = original;
    auto value           = field(records, name).to_vector<int>();
    value[0]             = value[0] == 0 ? 1 : 0;
    field(records, name) = n::from_vector(value, field(records, name).shape());
    save(records);
    EXPECT_THROW(m::LoadSequenceVaeData(root.string()), std::invalid_argument);
    records              = original;
    field(records, name) = n::cast(field(records, name), n::DType::Int64);
    save(records);
    EXPECT_THROW(m::LoadSequenceVaeData(root.string()), std::invalid_argument);
  }
  auto records = original;
  records.erase(records.begin());
  save(records);
  EXPECT_THROW(m::LoadSequenceVaeData(root.string()), std::invalid_argument);
  records = original;
  records.emplace_back("fit_tokens", field(records, "fit_tokens"));
  EXPECT_THROW(m::sequence_grammar_detail::Find(records, "fit_tokens"), std::invalid_argument);
  auto pools = m::LoadSequenceVaeData(
      (datamunge::datasets::prepare_example_data(
           "sequence-vae-assets", (root.parent_path() / "second").string()) +
       "/sequence_vae"));
  EXPECT_THROW(m::SequenceDataArrays({pools[0]}), std::invalid_argument);
  pools[0].source = pools[1].source;
  EXPECT_THROW(m::SequenceDataArrays(pools), std::invalid_argument);
}
TEST_F(SentenceGrammarData, VocabularyIsCopiedAndExactFileContractHasClearIoFailures) {
  auto words = m::SequenceVocabulary();
  ASSERT_EQ(words.size(), 19u);
  words[0] = "changed";
  EXPECT_EQ(m::SequenceVocabulary()[0], "[pad]");
  std::string text;
  for (const auto& word : m::SequenceVocabulary()) {
    if (!text.empty())
      text += '\n';
    text += word;
  }
  vocabulary(text);
  EXPECT_NO_THROW(m::LoadSequenceVaeData(root.string()));
  vocabulary(text + "\n\n");
  EXPECT_THROW(m::LoadSequenceVaeData(root.string()), std::invalid_argument);
  vocabulary("wrong\n");
  EXPECT_THROW(m::LoadSequenceVaeData(root.string()), std::invalid_argument);
  fs::remove(root / "vocabulary.txt");
  EXPECT_THROW(m::LoadSequenceVaeData(root.string()), std::runtime_error);
  fs::remove(root / "sentences.npz");
  EXPECT_ANY_THROW(m::LoadSequenceVaeData(root.string()));
}
TEST_F(SentenceGrammarData, PrefixDisplayGrammarSuffixPolicyAndInvalidSelections) {
  const auto tokens = n::from_vector(
      std::vector<std::int64_t>{3, 5, 8, 11, 13, 2, 999, 4, 7, 10, 12, 16, 18, 2}, {2, 7});
  EXPECT_EQ(m::SequenceText(tokens, 0), "a red circle moves left [eos]");
  EXPECT_EQ(m::SequenceText(tokens, 1), "the blue triangle rolls down slowly [eos]");
  EXPECT_TRUE(m::SequenceGrammarValid(tokens, 0));
  EXPECT_TRUE(m::SequenceGrammarValid(tokens, 1));
  const auto incomplete = n::from_vector(std::vector<int>{0, 1, 3, 5, 8, 11, 13}, {1, 7});
  EXPECT_FALSE(m::SequenceGrammarValid(incomplete, 0));
  EXPECT_EQ(m::SequenceText(incomplete, 0), "[pad] [bos] a red circle moves left");
  const auto bad = n::full({1, 7}, 99, n::DType::Int32);
  EXPECT_FALSE(m::SequenceGrammarValid(bad, 0));
  EXPECT_THROW(m::SequenceText(bad, 0), std::invalid_argument);
  EXPECT_THROW(m::SequenceText(tokens, -1), std::invalid_argument);
  EXPECT_THROW(m::SequenceGrammarValid(tokens, 2), std::invalid_argument);
  EXPECT_THROW(m::SequenceText(n::zeros({1, 7}), 0), std::invalid_argument);
  auto strided = n::transpose(n::transpose(tokens, {1, 0}).contiguous(), {1, 0});
  EXPECT_EQ(m::SequenceText(strided, 1), m::SequenceText(tokens, 1));
}
