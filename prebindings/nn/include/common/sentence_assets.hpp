// Pang/Lee MR v1.0 preparation for Kim's sentence CNN. All languages use this
// C++ implementation; vocabulary statistics use fitting sentences only.
#pragma once

#include <sstream>

#include "common/dataset_assets.hpp"

namespace dataset_assets {
#ifndef SWIG
namespace detail {

inline std::string TarText(const std::vector<unsigned char>& bytes, const std::string& member) {
  for (std::size_t at = 0; at + 512 <= bytes.size();) {
    const auto* header = bytes.data() + at;
    if (!header[0])
      break;
    const std::string name(
        reinterpret_cast<const char*>(header), std::find(header, header + 100, 0) - header);
    std::size_t size = 0;
    for (int i = 124; i < 136; ++i) {
      if (header[i] == 0 || header[i] == ' ')
        continue;
      if (header[i] < '0' || header[i] > '7' || size > bytes.size() / 8)
        throw std::runtime_error("invalid MR tar member size");
      size = size * 8 + (header[i] - '0');
    }
    at += 512;
    if (size > bytes.size() - at)
      throw std::runtime_error("truncated MR tar archive");
    if (name == member) {
      if (header[156] != 0 && header[156] != '0')
        throw std::runtime_error("MR corpus member must be a regular file");
      return {reinterpret_cast<const char*>(bytes.data() + at), size};
    }
    at += ((size + 511) / 512) * 512;
  }
  throw std::runtime_error("missing MR corpus member " + member);
}

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

struct Review {
  std::vector<std::string> words;
  int                      label, source;
};

inline void WriteSentenceData(const std::vector<unsigned char>& archive, const fs::path& stage) {
  std::vector<Review>                             reviews;
  std::map<std::string, std::vector<std::size_t>> groups;
  int                                             source = 0;
  for (int label = 0; label < 2; ++label) {
    std::istringstream input(
        TarText(archive, "rt-polaritydata/rt-polarity." + std::string(label ? "pos" : "neg")));
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
#endif

inline void EnsureSentences(const std::string& directory) {
  namespace fs    = std::filesystem;
  const auto root = fs::path(directory);
  if (detail::Present(root / "sentence.npz"))
    return;
  const auto        archive = root / "rt-polaritydata.tar.gz";
  const std::string hash    = "50c1c2c047b4225e148e97aa7708c34e";
  if (!detail::Present(archive)) {
    detail::Online(root);
    detail::Staging download(root);
    detail::Fetch(
        "sentence",
        "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz",
        download.path / "corpus.tar.gz",
        hash);
    detail::Publish(download.path / "corpus.tar.gz", archive);
  }
  if (datamunge::algorithms::md5_hex(detail::Read(archive)) != hash)
    throw std::runtime_error("MR corpus checksum mismatch in " + archive.string());
  detail::Staging stage(root);
  detail::WriteSentenceData(detail::Gunzip(archive), stage.path);
  for (const std::string file : {"vocabulary.txt", "manifest.json", "sentence.npz"})
    detail::Publish(stage.path / file, root / file);
}
} // namespace dataset_assets
