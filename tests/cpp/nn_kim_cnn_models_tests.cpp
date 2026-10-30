
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#include <regex>
#define main original_kim_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.9_1d_convolutions/kim_sentence_cnn.cpp"
#undef main
#include "../fixtures/nn_sentence_reference.hpp"
TEST(KimSentence, IndependentAll80WideConvolutionAndEmbeddingParameterDerivatives) {
  HostChecks();
}
TEST(KimSentence, BuiltinAdadeltaAllSlotsAndExplicitHeadProjection) {
  OptimizerChecks();
}
TEST(KimSentence, SeededDropoutEveryValueAndSharedMaskAdjoint) {
  DropoutChecks();
}
TEST(KimSentence, ExtremeStableLossEveryLogitDerivative) {
  LossChecks();
}
TEST(KimSentence, FullEveryBranchAndEveryLearnedGradient) {
  nn::set_num_threads(2);
  FullChecks();
}
TEST(KimSentence, EveryOriginalWordLengthLabelSourceAndFitOnlyVocabulary) {
  namespace m = dm::nn::models;
  const std::filesystem::path root = "build/cnn-datasets/sentence";
  const auto data = m::LoadSentences(root);
  const auto archive = dataset_assets::detail::Gunzip(root / "rt-polaritydata.tar.gz");
  std::vector<std::string> source;
  for (const std::string kind : {"neg", "pos"}) {
    std::istringstream lines(
        dataset_assets::detail::TarText(archive, "rt-polaritydata/rt-polarity." + kind));
    std::string line;
    while (std::getline(lines, line))
      source.push_back(line);
  }
  ASSERT_EQ(source.size(), 10662);
  std::vector<std::string> vocabulary;
  std::map<std::string, int> ids;
  std::ifstream vocab(root / "vocabulary.txt");
  std::string word;
  while (std::getline(vocab, word)) {
    ids[word] = int(vocabulary.size());
    vocabulary.push_back(word);
  }
  ASSERT_EQ(vocabulary.size(), data.config.vocabulary);
  const std::regex pattern("[a-z0-9']+");
  std::map<std::string, int> frequencies, owner;
  int records = 0;
  for (int split = 0; split < 3; ++split) {
    const auto &d = data.pools[split];
    const auto tokens = d.tokens.to_vector<int>(), lengths = d.lengths.to_vector<int>(),
               labels = d.labels.to_vector<int>(), sources = d.source_ids.to_vector<int>();
    for (int row = 0; row < d.size(); ++row) {
      const int id = sources[row];
      EXPECT_EQ(labels[row], id >= 5331);
      auto raw = source[id];
      for (char &c : raw)
        if (c >= 'A' && c <= 'Z')
          c += 32;
      std::vector<std::string> words;
      for (std::sregex_iterator at(raw.begin(), raw.end(), pattern), end;
           at != end && words.size() < 64; ++at)
        words.push_back(at->str());
      ASSERT_EQ(lengths[row], words.size());
      std::string key;
      for (int t = 0; t < 64; ++t) {
        int expected = 0;
        if (t < int(words.size())) {
          expected = ids.count(words[t]) ? ids.at(words[t]) : 1;
          if (split == 0)
            ++frequencies[words[t]];
          key += words[t] + " ";
        }
        EXPECT_EQ(tokens[row * 64 + t], expected);
      }
      if (owner.count(key))
        EXPECT_EQ(owner.at(key), split);
      else
        owner[key] = split;
      ++records;
    }
  }
  std::vector<std::pair<std::string, int>> ranked(frequencies.begin(), frequencies.end());
  std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
    return a.second == b.second ? a.first < b.first : a.second > b.second;
  });
  if (ranked.size() > 7998)
    ranked.resize(7998);
  ASSERT_EQ(ranked.size() + 2, vocabulary.size());
  for (std::size_t i = 0; i < ranked.size(); ++i)
    EXPECT_EQ(vocabulary[i + 2], ranked[i].first);
  EXPECT_EQ(records, 1400);
}
TEST(KimSentence, EveryRealCountNbScoreAndVisibleSgdParity) {
  namespace m = dm::nn::models;
  const auto data = m::LoadSentences("build/cnn-datasets/sentence");
  const auto &fit = data.pools[0];
  const auto &test = data.pools[2];
  const int v = data.config.vocabulary;
  const auto x = m::SentenceWordCounts(fit, v), y = m::SentenceWordCounts(test, v),
             labels = nn::one_hot(fit.labels, 2, 1., 0., nn::DType::Float64),
             target = nn::matmul(nn::transpose(labels, {1, 0}), x) + 1.,
             prior_target = nn::sum(labels, {0}) + 1.;
  const auto token_values = fit.tokens.to_vector<int>(),
             lengths = fit.lengths.to_vector<int>();
  const auto counts = x.to_vector<double>();
  for (int row = 0; row < fit.size(); ++row) {
    std::vector<int> literal(v);
    for (int t = 0; t < lengths[row]; ++t)
      ++literal[token_values[row * 64 + t]];
    for (int token = 0; token < v; ++token)
      EXPECT_EQ(counts[row * v + token], literal[token]);
  }
  nn::ParameterStore p(331, nn::DType::Float64);
  const auto c = p.get("token_counts", {2, v}, nn::init::zeros()),
             class_counts = p.get("class_counts", {2}, nn::init::zeros());
  nn::OptimizerConfig options;
  options.kind = nn::OptimizerKind::Sgd;
  nn::Optimizer sgd(p, .5, options);
  sgd.minimize(nn::sum(nn::square(c - target)) +
               nn::sum(nn::square(class_counts - prior_target)));
  const auto logits =
      (nn::matmul(y, nn::transpose(nn::log(c / nn::sum(c, {1}, true)), {1, 0})) +
       nn::log(class_counts / nn::sum(class_counts)))
          .to(nn::DType::Float32);
  EXPECT_EQ(nn::max(nn::abs(logits - m::SentenceNB(fit, test, v))).item(), 0.);
  EXPECT_LE(nn::max(nn::abs(logits - word_reference::SentenceNB(fit, test, v))).item(), 2e-5);
}
TEST(KimSentence, StructuralPadGradientsAndObservedOnlyTensorControls) {
  namespace m = dm::nn::models;
  nn::set_num_threads(2);
  const auto data = m::LoadSentences("build/cnn-datasets/sentence");
  const auto d = m::SentenceRows(data.pools[2], {0, 1});
  nn::ParameterStore p(317);
  const auto out = m::KimCnn(p, d.tokens, d.lengths, data.config);
  const auto g = nn::grad(nn::sum(nn::square(out.logits)), {out.embedding})[0];
  EXPECT_EQ(nn::max(nn::abs(nn::select(g, 0, 0))).item(), 0.);
  nn::NoGradGuard guard;
  const auto active = m::LengthMask(d.lengths, 64, nn::DType::Bool),
             padding_changed = nn::where(active, d.tokens, 1.),
             observed_changed = nn::where(active, 1., d.tokens);
  EXPECT_EQ(nn::max(nn::abs(m::KimCnn(p, padding_changed, d.lengths, data.config).logits -
                            out.logits))
                .item(),
            0.);
  EXPECT_GT(nn::max(nn::abs(m::KimCnn(p, observed_changed, d.lengths, data.config).logits -
                            out.logits))
                .item(),
            1e-6);
}
