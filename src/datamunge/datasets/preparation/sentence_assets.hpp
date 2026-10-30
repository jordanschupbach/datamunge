#pragma once
/// @file sentence_assets.hpp
/// @brief Original deterministic MR vocabulary and duplicate-group partition preparation.
#include <algorithm>
#include <datamunge/nn/nn.hpp>
#include <map>
#include <numeric>
#include <random>

#include "support.hpp"
/// @brief Private original MR preparation implementation.
namespace datamunge::datasets::detail::sentence {
/// @brief Tokenization, duplicate grouping and fitting-only vocabulary helpers.
namespace detail {
namespace fs = std::filesystem;
namespace nn = datamunge::nn;
/// @brief Apply ASCII lowercase word/apostrophe tokenization and retain the first64 tokens.
/// @param line Original Latin1 source review.
/// @return Ordered observed token strings.
inline std::vector<std::string> SentenceWords(const std::string& line) {
  std::vector<std::string> words;
  std::string              word;
  const auto               finish = [&] {
    if (!word.empty()) {
      words.push_back(word);
      word.clear();
    }
  };
  // The source is Latin-1. Define an ASCII tokenizer explicitly rather than
  // depending on the process locale or a host-language Unicode implementation.
  for (unsigned char c : line) {
    if (c >= 'A' && c <= 'Z')
      c += 'a' - 'A';
    if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '\'')
      word += char(c);
    else
      finish();
  }
  finish();
  if (words.size() > 64)
    words.resize(64);
  return words;
}

/// @brief Internal observed review with its original source identity.
struct Review {
  std::vector<std::string> words;  ///< Truncated observed word sequence.
  int                      label;  ///< Original binary sentiment label.
  int                      source; ///< Original zero-based corpus line identity.
};

/// @brief Save original duplicate-group selections and a fitting-only vocabulary.
/// @param archive Directory holding the two extracted original source text files.
/// @param stage Output directory for tensors, vocabulary and provenance.
inline void WriteSentenceData(const fs::path& archive, const fs::path& stage) {
  std::vector<Review>                             reviews;
  std::map<std::string, std::vector<std::size_t>> groups;
  int                                             source = 0;
  for (int label = 0; label < 2; ++label) {
    std::istringstream input(
        datamunge::datasets::detail::read(
            archive / "rt-polaritydata" / ("rt-polarity." + std::string(label ? "pos" : "neg"))));
    std::string line;
    int         lines = 0;
    while (std::getline(input, line)) {
      auto      words = SentenceWords(line);
      const int id    = source++;
      ++lines;
      if (words.empty())
        continue;
      std::string key;
      for (const auto& word : words) {
        key += word;
        key += ' ';
      }
      groups[key].push_back(reviews.size());
      reviews.push_back({std::move(words), label, id});
    }
    if (lines != 5331)
      throw std::runtime_error("MR v1.0 requires 5331 sentences per class");
  }
  // One representative per duplicate group. Ambiguous groups are excluded;
  // neither exact texts nor equal truncated token sequences cross splits.
  std::vector<std::size_t> candidates[2];
  for (const auto& [key, group] : groups) {
    const int label = reviews[group.front()].label;
    if (std::all_of(
            group.begin(), group.end(), [&](auto index) { return reviews[index].label == label; }))
      candidates[label].push_back(group.front());
  }
  std::mt19937                                    random(337);
  std::map<std::string, std::vector<std::size_t>> selected;
  for (int label = 0; label < 2; ++label) {
    std::shuffle(candidates[label].begin(), candidates[label].end(), random);
    if (candidates[label].size() < 700)
      throw std::runtime_error("insufficient unique MR sentences");
    std::size_t at = 0;
    for (const auto& [split, count] :
         std::vector<std::pair<std::string, int>>{{"fit", 500}, {"validation", 100}, {"test", 100}})
      for (int i = 0; i < count; ++i)
        selected[split].push_back(candidates[label][at++]);
  }
  std::map<std::string, int> frequencies;
  for (auto index : selected["fit"])
    for (const auto& word : reviews[index].words)
      ++frequencies[word];
  std::vector<std::pair<std::string, int>> ranked(frequencies.begin(), frequencies.end());
  std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
    return a.second == b.second ? a.first < b.first : a.second > b.second;
  });
  if (ranked.size() > 7998)
    ranked.resize(7998);
  std::map<std::string, int> vocabulary;
  std::ofstream              words(stage / "vocabulary.txt");
  words << "<PAD>\n<UNK>\n";
  for (std::size_t i = 0; i < ranked.size(); ++i) {
    vocabulary[ranked[i].first] = int(i + 2);
    words << ranked[i].first << '\n';
  }
  words.close();
  if (!words)
    throw std::runtime_error("cannot write MR vocabulary");
  nn::NamedTensors arrays;
  arrays.emplace_back("config", nn::from_vector(std::vector<int>{int(vocabulary.size() + 2), 64}));
  std::ofstream manifest(stage / "manifest.json");
  manifest
      << "{\"dataset\":\"Pang/Lee MR "
         "v1.0\",\"url\":\"https://www.cs.cornell.edu/people/pabo/movie-review-data/"
         "rt-polaritydata.tar.gz\","
         "\"md5\":\"50c1c2c047b4225e148e97aa7708c34e\",\"seed\":337,\"external_selection\":false,"
         "\"selection\":\"balanced duplicate-group holdout, std::mt19937\","
         "\"tokenizer\":\"ASCII lowercase words with apostrophes, first 64 "
         "tokens\",\"vocabulary\":\"fit only, frequency then lexicographic order, PAD=0 "
         "UNK=1\",\"source_ids\":{";
  bool first = true;
  for (const std::string split : {"fit", "validation", "test"}) {
    auto rows = selected[split];
    std::shuffle(rows.begin(), rows.end(), random);
    const auto       count = static_cast<std::int64_t>(rows.size());
    std::vector<int> tokens(rows.size() * 64), lengths, labels, ids;
    if (!first)
      manifest << ',';
    first = false;
    manifest << '"' << split << "\":[";
    for (std::size_t row = 0; row < rows.size(); ++row) {
      const auto& review = reviews[rows[row]];
      lengths.push_back(int(review.words.size()));
      labels.push_back(review.label);
      ids.push_back(review.source);
      if (row)
        manifest << ',';
      manifest << review.source;
      for (std::size_t time = 0; time < review.words.size(); ++time) {
        const auto found        = vocabulary.find(review.words[time]);
        tokens[row * 64 + time] = found == vocabulary.end() ? 1 : found->second;
      }
    }
    manifest << ']';
    arrays.emplace_back(split + "_tokens", nn::from_vector(tokens, {count, 64}));
    arrays.emplace_back(split + "_lengths", nn::from_vector(lengths));
    arrays.emplace_back(split + "_labels", nn::from_vector(labels));
    arrays.emplace_back(split + "_source_ids", nn::from_vector(ids));
  }
  manifest << "}}\n";
  manifest.close();
  if (!manifest)
    throw std::runtime_error("cannot write MR manifest");
  nn::save_npz((stage / "sentence.npz").string(), arrays);
}
} // namespace detail
/// @brief Ensure the original deterministic sentence tensors, vocabulary and provenance.
/// @param directory Sentence cache directory containing the pinned corpus.
inline void EnsureSentences(const std::string& directory) {
  namespace fs        = std::filesystem;
  const fs::path root = directory;
  if (fs::is_regular_file(root / "sentence.npz") && fs::is_regular_file(root / "vocabulary.txt") &&
      fs::is_regular_file(root / "manifest.json"))
    return;
  datamunge::datasets::detail::Temporary stage(root);
  datamunge::datasets::extract_archive(
      (root / "rt-polaritydata.tar.gz").string(),
      stage.path.string(),
      {"rt-polaritydata/rt-polarity.neg", "rt-polaritydata/rt-polarity.pos"});
  detail::WriteSentenceData(stage.path, stage.path);
  for (const auto& name : {"sentence.npz", "manifest.json", "vocabulary.txt"})
    fs::rename(stage.path / name, root / name);
}
} // namespace datamunge::datasets::detail::sentence
