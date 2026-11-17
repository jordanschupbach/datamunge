/// @file language_assets.cpp
/// @brief Exact pinned upstream text and palette preparation for literate NN reports.
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include <algorithm>
#include <datamunge/algorithms/sha256.hpp>
#include <datamunge/datasets/download.hpp>
#include <datamunge/nn/io.hpp>
#include <map>
#include <random>
#include <set>

#include "support.hpp"
namespace datamunge::datasets::detail {
/// @brief Private original-recipe data assembly helpers, with no training state.
namespace language_assets {
namespace n = datamunge::nn; ///< Native tensor serialization.
/// @brief Reject a violated pinned source or recipe invariant.
/// @param valid Invariant condition.
/// @param message Error detail.
void Check(bool valid, const std::string& message) {
  if (!valid)
    throw std::runtime_error(message);
}
/// @brief Publish owned bytes after checking the output stream.
/// @param path Destination file.
/// @param text Exact bytes, including intentional newlines.
void Write(const fs::path& path, const std::string& text) {
  std::ofstream f(path, std::ios::binary);
  f.write(text.data(), text.size());
  if (!f)
    throw std::runtime_error("cannot write " + path.string());
}
/// @brief Fetch and revalidate one immutable recipe source.
/// @param root Cache directory.
/// @param name Local filename.
/// @param url Pinned upstream URL.
/// @param sha Expected SHA256.
/// @return Source file bytes.
std::string Fetch(
    const fs::path&    root,
    const std::string& name,
    const std::string& url,
    const std::string& sha) {
  fetch_file(url, (root / name).string(), sha, 0, root.filename().string());
  return read(root / name);
}
/// @brief Save native arrays with an explicit upstream preparation fingerprint.
/// @param root Cache destination.
/// @param filename Native NPZ basename.
/// @param arrays Owned completed arrays.
/// @param manifest Source preparation metadata; native serialization hash is added.
void Save(
    const fs::path&        root,
    const std::string&     filename,
    const n::NamedTensors& arrays,
    Json                   manifest) {
  n::save_npz((root / filename).string(), arrays);
  manifest["npz_sha256"]          = file_digest((root / filename).string());
  manifest["preparation_runtime"] = "native datamunge C++; original tflow_examples recipe";
  Write(root / "manifest.json", manifest.dump(2) + "\n");
}
/// @brief Preserve original tinyShakespeare boundaries, fit-only alphabet and disjoint windows.
/// @param root Character corpus cache directory.
void Character(const fs::path& root) {
  const std::string revision = "370cbcd448eb7daf32f21a6be560b70e0b33c4e3",
                    sha      = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
                    url      = "https://raw.githubusercontent.com/karpathy/char-rnn/" + revision +
                          "/data/tinyshakespeare/input.txt";
  auto raw = Fetch(root, "input.txt", url, sha);
  Check(raw.size() == 1115394, "original tinyShakespeare length");
  for (unsigned char c : raw)
    Check(c < 128, "original ASCII corpus required");
  size_t         a = raw.size() * 9 / 10, b = raw.size() * 95 / 100;
  std::set<char> alphabet(raw.begin(), raw.begin() + a);
  std::string    vocabulary(alphabet.begin(), alphabet.end());
  Check(vocabulary.size() == 65, "fit-only65-character alphabet");
  Write(root / "vocabulary.txt", vocabulary);
  std::map<char, int> lookup;
  for (int i = 0; i < 65; ++i)
    lookup[vocabulary[i]] = i;
  n::NamedTensors arrays;
  Json            partitions = Json::array();
  int             k          = 0;
  for (const std::string split : {"fit", "validation", "test"}) {
    size_t start = k == 0 ? 0 : k == 1 ? a : b, end = k == 0 ? a : k == 1 ? b : raw.size();
    int    count = k == 0 ? 128 : 32;
    std::vector<int> x, y, starts;
    for (int i = 0; i < count; ++i) {
      size_t at = start + i * 65;
      Check(at + 65 <= end, "disjoint original character window");
      starts.push_back(int(at));
      for (int t = 0; t < 64; ++t) {
        x.push_back(lookup.at(raw[at + t]));
        y.push_back(lookup.at(raw[at + t + 1]));
      }
    }
    arrays.push_back({split + "_input", n::from_vector(x, {count, 64})});
    arrays.push_back({split + "_target", n::from_vector(y, {count, 64})});
    arrays.push_back({split + "_start", n::from_vector(starts)});
    partitions.push_back(
        Json{
            {"split", split},
            {"partition_start", start},
            {"partition_end_exclusive", end},
            {"windows", count},
            {"selected_start", start},
            {"selected_end_exclusive", start + count * 65},
            {"window_stride", 65}});
    ++k;
  }
  std::vector<std::string> v;
  for (char c : vocabulary)
    v.emplace_back(1, c);
  Save(
      root,
      "character_rnn.npz",
      arrays,
      Json{
          {"url", url},
          {"revision", revision},
          {"sha256", sha},
          {"bytes", raw.size()},
          {"encoding", "ASCII/UTF8"},
          {"vocabulary", v},
          {"vocabulary_fitted_on_partition", Json::array({0, a})},
          {"partitions", partitions},
          {"window", 64},
          {"window_stride", 65},
          {"selection",
           "first128/32/32disjoint65characterchunkswithincontiguous90/5/5partitions"}});
}
/// @brief Split whitespace tokens after inserting the original newline EOS marker.
/// @param raw Source UTF8 PTB bytes.
/// @return Owned source token stream.
std::vector<std::string> Words(const std::string& raw) {
  std::string expanded;
  for (char c : raw)
    if (c == '\n')
      expanded += " <eos> ";
    else
      expanded += c;
  std::istringstream       input(expanded);
  std::vector<std::string> words;
  std::string              word;
  while (input >> word)
    words.push_back(word);
  return words;
}
/// @brief Preserve canonical PTB sources, full training vocabulary and stride17 windows.
/// @param root PTB cache directory.
void Ptb(const fs::path& root) {
  const std::string                  revision = "76870253cfca069477f06b7056af87f98490b6eb";
  std::map<std::string, std::string> hashes{
      {"train", "fcea919f6cf83f35d4d00c6cbf08040d13d4155226340912e2fef9c9c4102cbf"},
      {"valid", "c9fe6985fe0d4ccb578183407d7668fc6066c20700cb4cf87d8ff1cc34df1bf2"},
      {"test", "dd65dff31e70846b2a6030a87482edcd5d199130cdcfa1f3dccbb033728deee0"}};
  std::map<std::string, std::vector<std::string>> streams;
  Json                                            records = Json::object();
  for (auto& [name, sha] : hashes) {
    std::string url = "https://raw.githubusercontent.com/wojzaremba/lstm/" + revision +
                      "/data/ptb." + name + ".txt";
    auto raw      = Fetch(root, "ptb." + name + ".txt", url, sha);
    streams[name] = Words(raw);
    records[name] = Json{
        {"url", url}, {"sha256", sha}, {"bytes", raw.size()}, {"tokens", streams[name].size()}};
  }
  std::set<std::string> alphabet(streams["train"].begin(), streams["train"].end());
  alphabet.erase("<unk>");
  std::vector<std::string> vocabulary{"<unk>"};
  vocabulary.insert(vocabulary.end(), alphabet.begin(), alphabet.end());
  Check(vocabulary.size() == 10000, "original full training PTB vocabulary");
  std::map<std::string, int> lookup;
  std::string                text;
  for (int i = 0; i < 10000; ++i) {
    lookup[vocabulary[i]] = i;
    text += vocabulary[i] + "\n";
  }
  Write(root / "vocabulary.txt", text);
  n::NamedTensors arrays;
  int             k = 0;
  for (auto names :
       {std::pair{"fit", "train"}, std::pair{"validation", "valid"}, std::pair{"test", "test"}}) {
    auto [split, source]   = names;
    int              count = k == 0 ? 128 : 32;
    std::vector<int> x, y, starts;
    auto&            stream  = streams[source];
    int              unknown = 0;
    for (int i = 0; i < count; ++i) {
      starts.push_back(i * 17);
      for (int t = 0; t < 17; ++t) {
        auto it    = lookup.find(stream.at(i * 17 + t));
        int  token = it == lookup.end() ? 0 : it->second;
        if (t < 16)
          x.push_back(token);
        if (t > 0) {
          y.push_back(token);
          unknown += token == 0;
        }
      }
    }
    arrays.push_back({std::string(split) + "_input", n::from_vector(x, {count, 16})});
    arrays.push_back({std::string(split) + "_target", n::from_vector(y, {count, 16})});
    arrays.push_back({std::string(split) + "_start", n::from_vector(starts)});
    records[source]["selected_windows"]         = count;
    records[source]["window"]                   = 16;
    records[source]["stride"]                   = 17;
    records[source]["selected_tokens"]          = count * 17;
    records[source]["selected_unknown_targets"] = unknown;
    int total                                   = 0;
    for (auto& w : stream)
      total += !lookup.count(w);
    records[source]["raw_out_of_vocabulary_tokens"] = total;
    ++k;
  }
  Save(
      root,
      "ptb_word.npz",
      arrays,
      Json{
          {"revision", revision},
          {"source", "word-level PTB preprocessing distributed with wojzaremba/lstm"},
          {"vocabulary_fitted_on", "entire training source ONLY"},
          {"vocabulary", vocabulary},
          {"unknown_id", 0},
          {"newline_token", "<eos>"},
          {"records", records},
          {"selection",
           "first128/32/32disjoint17-tokenchunkswithinseparatecanonicaltrain/valid/testfiles"}});
}

/// @brief Decode only the integer-list payload of hash-verified official Keras object arrays.
/// @param file Original x_train.npy or x_test.npy extracted from the verified imdb.npz.
/// @return Exactly25000 owned nonempty integer-token rows; no pickle callable is executed.
std::vector<std::vector<int>> ImdbRows(const fs::path& file) {
  auto raw = read(file);
  Check(
      raw.size() > 128 && raw.size() < 40000000 && raw.substr(1, 5) == "NUMPY",
      "bounded original IMDB NPY required");
  size_t off = 10 + (unsigned char)raw[8] + 256 * (unsigned char)raw[9];
  Check(off < raw.size(), "IMDB NPY header");
  auto marker = raw.find(std::string("\x89]q\x06(", 5), off);
  Check(
      marker != std::string::npos && marker - off < 160,
      "original non-executable IMDB object preamble");
  size_t                        pos = marker + 5;
  std::vector<std::vector<int>> rows;
  std::vector<int>              current;
  bool                          active = false, stopped = false;
  /// @brief Read one bounded unsigned little-endian opcode argument without evaluating objects.
  auto number = [&](int bytes) {
    Check(pos + bytes <= raw.size(), "truncated IMDB opcode");
    uint32_t value = 0;
    for (int i = 0; i < bytes; ++i)
      value |= uint32_t((unsigned char)raw[pos++]) << (8 * i);
    return value;
  };
  while (pos < raw.size()) {
    unsigned char op = raw[pos++];
    if (op == ']') {
      if (active) {
        Check(!current.empty() && rows.size() < 25000, "bounded nonempty IMDB review");
        rows.push_back(std::move(current));
        current.clear();
      }
      active = true;
    } else if (op == 'K' || op == 'M' || op == 'J') {
      Check(active, "integer outside review");
      auto value = number(op == 'K' ? 1 : op == 'M' ? 2 : 4);
      Check(value > 0 && value < 100000 && current.size() < 10000, "bounded positive IMDB token");
      current.push_back(int(value));
    } else if (op == 'q')
      number(1);
    else if (op == 'r')
      number(4);
    else if (op == '(' || op == 'e' || op == 'a' || op == 't' || op == 'b') {
    } else if (op == '.') {
      stopped = true;
      break;
    } else
      throw std::runtime_error("unsupported executable/object opcode in pinned IMDB payload");
  }
  Check(
      stopped && pos == raw.size() && active && !current.empty(), "complete original IMDB payload");
  rows.push_back(std::move(current));
  Check(rows.size() == 25000, "official25000-review partition");
  return rows;
}
/// @brief Produce exact selected-fit-only Keras IMDB vocabulary and balanced short prefixes.
/// @param root IMDB cache directory.
void Imdb(const fs::path& root) {
  const std::string archive_sha =
                        "69664113be75683a8fe16e3ed0ab59fda8886cb3cd7ada244f7d9544e4676b9f",
                    index_sha = "027313fe9a320280f3bf0fac5baf58c4a683cf39a7ed9c3d3d3912bde4284cf6",
                    base      = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/";
  fetch_file(base + "imdb.npz", (root / "imdb.npz").string(), archive_sha, 0, "imdb_lstm");
  auto mapping =
      Json::parse(Fetch(root, "imdb_word_index.json", base + "imdb_word_index.json", index_sha));
  std::map<int, std::string> reverse;
  for (auto it = mapping.begin(); it != mapping.end(); ++it)
    reverse[it.value().get<int>()] = it.key();
  Temporary stage(root);
  extract_archive(
      (root / "imdb.npz").string(),
      stage.path.string(),
      {"x_train.npy", "x_test.npy", "y_train.npy", "y_test.npy"},
      100000000);
  auto train = ImdbRows(stage.path / "x_train.npy"), test = ImdbRows(stage.path / "x_test.npy");
  auto yt = n::load_npy((stage.path / "y_train.npy").string()).to_vector<int>(),
       ye = n::load_npy((stage.path / "y_test.npy").string()).to_vector<int>();
  std::array<std::vector<int>, 2> ti, ei;
  for (int i = 0; i < 25000; ++i) {
    Check(yt[i] == 0 || yt[i] == 1, "binary official IMDB label");
    Check(ye[i] == 0 || ye[i] == 1, "binary official IMDB label");
    ti[yt[i]].push_back(i);
    ei[ye[i]].push_back(i);
  }
  std::map<std::string, int> counts;
  for (int i = 0; i < 32; ++i)
    for (int label = 0; label < 2; ++label)
      for (int token : train[ti[label][i]])
        ++counts[reverse.at(token)];
  std::vector<std::pair<std::string, int>> order(counts.begin(), counts.end());
  std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
    return a.second != b.second ? a.second > b.second : a.first < b.first;
  });
  std::vector<std::string> vocabulary{"<pad>", "<unk>"};
  for (int i = 0; i < 254; ++i)
    vocabulary.push_back(order.at(i).first);
  std::map<std::string, int> lookup;
  std::string                text;
  for (int i = 0; i < 256; ++i) {
    lookup[vocabulary[i]] = i;
    text += vocabulary[i] + "\n";
  }
  Write(root / "vocabulary.txt", text);
  n::NamedTensors       arrays;
  Json                  records = Json::array();
  std::set<std::string> seen;
  int                   split_index = 0;
  for (const std::string split : {"fit", "validation", "test"}) {
    bool  testing   = split_index == 2;
    auto& tokens    = testing ? test : train;
    auto& indices   = testing ? ei : ti;
    int   per_class = split_index == 0 ? 32 : 16, start = split_index == 1 ? 32 : 0,
        count = 2 * per_class;
    std::vector<int> x(count * 48, 0), lengths, labels, ids;
    int              b = 0;
    for (int i = 0; i < per_class; ++i)
      for (int label = 0; label < 2; ++label) {
        int         row      = indices[label][start + i];
        auto&       original = tokens[row];
        std::string bytes;
        for (int value : original)
          for (int shift = 0; shift < 32; shift += 8)
            bytes.push_back(char((uint32_t(value) >> shift) & 255));
        std::string digest = datamunge::algorithms::sha256_hex(bytes);
        Check(seen.insert(digest).second, "duplicate original IMDB review across splits");
        int length = std::min<int>(48, original.size()), unknown = 0;
        for (int t = 0; t < length; ++t) {
          auto it       = lookup.find(reverse.at(original[t]));
          int  value    = it == lookup.end() ? 1 : it->second;
          x[b * 48 + t] = value;
          unknown += value == 1;
        }
        int id = row + (testing ? 100000 : 0);
        lengths.push_back(length);
        labels.push_back(label);
        ids.push_back(id);
        records.push_back(
            Json{
                {"split", split},
                {"source_partition", testing ? "test" : "train"},
                {"source_row", row},
                {"source_id", id},
                {"label", label},
                {"raw_token_sha256", digest},
                {"raw_tokens", original.size()},
                {"model_tokens", length},
                {"unknown_tokens", unknown}});
        ++b;
      }
    arrays.push_back({split + "_input", n::from_vector(x, {count, 48})});
    arrays.push_back({split + "_lengths", n::from_vector(lengths)});
    arrays.push_back({split + "_labels", n::from_vector(labels)});
    arrays.push_back({split + "_source_ids", n::from_vector(ids)});
    ++split_index;
  }
  Save(
      root,
      "imdb_lstm.npz",
      arrays,
      Json{
          {"source",
           "Keras pretokenized IMDB; original upstream preprocessing already removed some unseen "
           "test words"},
          {"hashes", Json{{"imdb.npz", archive_sha}, {"imdb_word_index.json", index_sha}}},
          {"vocabulary", vocabulary},
          {"vocabulary_fit",
           "only64selectedfittingreviews;frequencydescending/lexicalties;no global index cutoff"},
          {"maximum_tokens", 48},
          {"pad_id", 0},
          {"unknown_id", 1},
          {"records", records}});
}
/// @brief One complete original CoNLL sentence with source document identity.
struct ConllSentence {
  int                                     document; ///< Original document index.
  std::vector<std::array<std::string, 4>> rows;     ///< Word, POS, chunk and raw IOB1 fields.
};
/// @brief Parse original sentence/document separators without truncating sentences.
/// @param raw Hash-verified CoNLL bytes.
/// @return Owning complete source sentences in order.
std::vector<ConllSentence> Conll(const std::string& raw) {
  std::istringstream         input(raw);
  std::string                line;
  int                        document = -1;
  std::vector<ConllSentence> result;
  ConllSentence              current{-1, {}};
  while (std::getline(input, line)) {
    std::istringstream       fields(line);
    std::vector<std::string> v;
    std::string              value;
    while (fields >> value)
      v.push_back(value);
    if (v.empty() || v[0] == "-DOCSTART-") {
      if (!current.rows.empty()) {
        result.push_back(std::move(current));
        current = {document, {}};
      }
      if (!v.empty())
        ++document;
      current.document = document;
      continue;
    }
    Check(v.size() == 4, "four-column CoNLL source");
    current.rows.push_back({v[0], v[1], v[2], v[3]});
  }
  if (!current.rows.empty())
    result.push_back(std::move(current));
  return result;
}
/// @brief Preserve original CoNLL POS ontology, fit-only word vocabulary and IOB2 labels.
/// @param root Tagging cache directory.
void Tagging(const fs::path& root) {
  const std::string        revision = "c0ea07c81ec3512b5047dcd804b0ac2b10a3bdfc";
  std::vector<std::string> names{"eng.train", "eng.testa", "eng.testb"},
      hashes{
          "eac0822f501353f265383ddf1edb0dcd42a1b6eefd3e383d7074db98c38eb41c",
          "aedc4ccbd1ae9795feaa0e86fe72d195b0258abed9d705b630d1c889276fac1e",
          "0632f32c090ef31c223135b7fee3187a7cf8e65c4015c7866bb7783428cdd9a6"};
  std::vector<std::vector<ConllSentence>> sources;
  Json                                    provenance;
  for (int k = 0; k < 3; ++k) {
    auto url = "https://raw.githubusercontent.com/synalp/NER/" + revision + "/corpus/CoNLL-2003/" +
               names[k];
    auto raw = Fetch(root, names[k], url, hashes[k]);
    sources.push_back(Conll(raw));
    provenance[names[k]] = Json{
        {"url", url},
        {"sha256", hashes[k]},
        {"bytes", raw.size()},
        {"sentences", sources.back().size()}};
  }
  std::set<std::string> pos_set;
  for (auto& sentence : sources[0])
    for (auto& row : sentence.rows)
      pos_set.insert(row[1]);
  Check(pos_set.size() == 45, "training-only45-POS ontology");
  std::vector<std::string> pos(pos_set.begin(), pos_set.end()),
      ner{"O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"};
  std::vector<std::vector<int>> selected(3);
  for (int k = 0; k < 3; ++k) {
    int count = k == 0 ? 128 : 32;
    for (size_t i = 0; i < sources[k].size() && int(selected[k].size()) < count; ++i)
      if (sources[k][i].rows.size() >= 2 && sources[k][i].rows.size() <= 24)
        selected[k].push_back(i);
    Check(int(selected[k].size()) == count, "original short complete sentence counts");
  }
  std::map<std::string, int> counts;
  for (int i : selected[0])
    for (auto& row : sources[0][i].rows)
      ++counts[row[0]];
  std::vector<std::pair<std::string, int>> order(counts.begin(), counts.end());
  std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
    return a.second != b.second ? a.second > b.second : a.first < b.first;
  });
  std::vector<std::string> vocabulary{"<pad>", "<unk>"};
  for (int i = 0; i < 510; ++i)
    vocabulary.push_back(order.at(i).first);
  std::map<std::string, int> word_map, pos_map, ner_map;
  std::string                text;
  for (int i = 0; i < 512; ++i) {
    word_map[vocabulary[i]] = i;
    text += vocabulary[i] + "\n";
  }
  Write(root / "vocabulary.txt", text);
  for (int i = 0; i < 45; ++i)
    pos_map[pos[i]] = i;
  for (int i = 0; i < 9; ++i)
    ner_map[ner[i]] = i;
  n::NamedTensors          arrays;
  Json                     records = Json::array();
  std::vector<std::string> splits{"fit", "validation", "test"};
  for (int k = 0; k < 3; ++k) {
    int              count = selected[k].size();
    std::vector<int> x(count * 24, 0), pt(x.size(), 0), nt(x.size(), 0), lengths, ids;
    for (int b = 0; b < count; ++b) {
      int         index    = selected[k][b];
      auto&       sentence = sources[k][index];
      int         length = sentence.rows.size(), unknown = 0;
      std::string previous = "O";
      Json        words = Json::array(), ptags = Json::array(), original = Json::array(),
           converted = Json::array();
      for (int t = 0; t < length; ++t) {
        auto& row     = sentence.rows[t];
        auto  found   = word_map.find(row[0]);
        x[b * 24 + t] = found == word_map.end() ? 1 : found->second;
        unknown += x[b * 24 + t] == 1;
        pt[b * 24 + t]     = pos_map.at(row[1]);
        auto        tag    = row[3];
        std::string mapped = tag;
        if (tag != "O") {
          Check(tag.size() > 2 && (tag[0] == 'B' || tag[0] == 'I'), "source IOB1 tag");
          mapped =
              ((tag[0] == 'B' || previous == "O" || previous.substr(2) != tag.substr(2)) ? "B-"
                                                                                         : "I-") +
              tag.substr(2);
        }
        nt[b * 24 + t] = ner_map.at(mapped);
        previous       = tag;
        words.push_back(row[0]);
        ptags.push_back(row[1]);
        original.push_back(tag);
        converted.push_back(mapped);
      }
      int id = k * 100000 + index;
      lengths.push_back(length);
      ids.push_back(id);
      records.push_back(
          Json{
              {"split", splits[k]},
              {"source_file", names[k]},
              {"sentence_index", index},
              {"document_index", sentence.document},
              {"source_id", id},
              {"length", length},
              {"words", words},
              {"pos", ptags},
              {"raw_ner", original},
              {"ner_iob2", converted},
              {"unknown_tokens", unknown}});
    }
    for (auto field : {std::pair{"input", &x}, std::pair{"pos", &pt}, std::pair{"ner", &nt}})
      arrays.push_back({splits[k] + "_" + field.first, n::from_vector(*field.second, {count, 24})});
    arrays.push_back({splits[k] + "_lengths", n::from_vector(lengths)});
    arrays.push_back({splits[k] + "_source_ids", n::from_vector(ids)});
  }
  Save(
      root,
      "bilstm_tagging.npz",
      arrays,
      Json{
          {"repository", "synalp/NER data mirror"},
          {"revision", revision},
          {"provenance", provenance},
          {"vocabulary", vocabulary},
          {"vocabulary_fit", "only128selectedfittingcompletecase-sensitivesentences"},
          {"pos_tags", pos},
          {"pos_tagset_fit", "fulltrainingfileontologyONLY"},
          {"ner_tags", ner},
          {"ner_conversion", "IOB1toIOB2: I at sentence start or after O/different type becomes B"},
          {"selection", "first128/32/32complete2..24tokensentencesincanonicaltrain/dev/test"},
          {"maximum_length", 24},
          {"records", records}});
}

/// @brief Prepare original CIFAR source rows, integer area codes and fit-only K16 palette.
/// @param root ImageGPT cache directory.
void ImagePalette(const fs::path& root) {
  fs::path cifar = datamunge::datasets::fetch("cifar10");
  fetch_file(
      "https://storage.googleapis.com/mirror.tensorflow.org/www.cs.toronto.edu/~kriz/"
      "cifar-10-binary.tar.gz",
      (cifar / "cifar-10-binary.tar.gz").string(),
      "c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd",
      170052171,
      "cifar10");
  Check(
      file_digest((cifar / "cifar-10-binary.tar.gz").string()) ==
          "c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd",
      "pinned original CIFAR archive");
  Temporary verified(root);
  extract_archive(
      (cifar / "cifar-10-binary.tar.gz").string(),
      verified.path.string(),
      {"cifar-10-batches-bin/data_batch_1.bin", "cifar-10-batches-bin/test_batch.bin"},
      70000000);
  for (const std::string file : {"data_batch_1.bin", "test_batch.bin"})
    Check(
        file_digest((cifar / file).string()) ==
            file_digest((verified.path / "cifar-10-batches-bin" / file).string()),
        "cached CIFAR bytes must equal verified original archive member");
  n::NamedTensors                 arrays;
  std::array<std::vector<int>, 3> codes;
  std::vector<std::string>        splits{"fit", "validation", "test"};
  for (int k = 0; k < 3; ++k) {
    int  count = k == 0 ? 128 : 32, start = k == 0 ? 0 : k == 1 ? 128 : 64;
    auto raw = read(cifar / (k == 2 ? "test_batch.bin" : "data_batch_1.bin"));
    Check(raw.size() == 10000 * 3073, "original CIFAR binary records");
    std::vector<int>   original(count * 32 * 32 * 3), labels, ids;
    std::vector<float> rgb(count * 64 * 3);
    auto&              small = codes[k];
    small.resize(count * 64 * 3);
    for (int b = 0; b < count; ++b) {
      size_t base = size_t(start + b) * 3073;
      labels.push_back((unsigned char)raw[base]);
      ids.push_back(start + b + (k == 2 ? 50000 : 0));
      for (int y = 0; y < 32; ++y)
        for (int x = 0; x < 32; ++x)
          for (int c = 0; c < 3; ++c)
            original[((b * 32 + y) * 32 + x) * 3 + c] =
                (unsigned char)raw[base + 1 + c * 1024 + y * 32 + x];
      for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
          for (int c = 0; c < 3; ++c) {
            int sum = 0;
            for (int yy = 0; yy < 4; ++yy)
              for (int xx = 0; xx < 4; ++xx)
                sum += original[((b * 32 + y * 4 + yy) * 32 + x * 4 + xx) * 3 + c];
            int i    = ((b * 8 + y) * 8 + x) * 3 + c;
            small[i] = (sum + 8) / 16;
            rgb[i]   = float(double(small[i]) / 255.);
          }
    }
    arrays.push_back({splits[k] + "_original", n::from_vector(original, {count, 32, 32, 3})});
    arrays.push_back({splits[k] + "_rgb_codes", n::from_vector(small, {count, 8, 8, 3})});
    arrays.push_back({splits[k] + "_rgb", n::from_vector(rgb, {count, 8, 8, 3})});
    arrays.push_back({splits[k] + "_source_ids", n::from_vector(ids)});
    arrays.push_back({splits[k] + "_labels", n::from_vector(labels)});
  }
  const int           count = 128 * 64;
  std::vector<double> points(count * 3);
  for (size_t i = 0; i < points.size(); ++i)
    points[i] = double(codes[0][i]) / 255.;
  std::mt19937         rng(9113);
  std::vector<int>     seeds{int(rng() & 8191)};
  std::vector<double>  nearest(count, INFINITY), centers(16 * 3), history, objectives;
  std::vector<int64_t> all_counts;
  while (seeds.size() < 16) {
    int last = seeds.back();
    for (int i = 0; i < count; ++i) {
      double distance = 0;
      for (int c = 0; c < 3; ++c) {
        double d = points[i * 3 + c] - points[last * 3 + c];
        distance += d * d;
      }
      nearest[i] = std::min(nearest[i], distance);
    }
    seeds.push_back(std::max_element(nearest.begin(), nearest.end()) - nearest.begin());
  }
  for (int k = 0; k < 16; ++k)
    for (int c = 0; c < 3; ++c)
      centers[k * 3 + c] = points[seeds[k] * 3 + c];
  history = centers;
  for (int iteration = 0; iteration < 10; ++iteration) {
    std::vector<int>     labels(count);
    std::vector<int64_t> counts(16, 0);
    std::vector<double>  minimum(count), sums(48, 0);
    for (int i = 0; i < count; ++i) {
      double best  = INFINITY;
      int    label = 0;
      for (int k = 0; k < 16; ++k) {
        double distance = 0;
        for (int c = 0; c < 3; ++c) {
          double d = points[i * 3 + c] - centers[k * 3 + c];
          distance += d * d;
        }
        if (distance < best) {
          best  = distance;
          label = k;
        }
      }
      labels[i]  = label;
      minimum[i] = best;
      ++counts[label];
      for (int c = 0; c < 3; ++c)
        sums[label * 3 + c] += points[i * 3 + c];
    }
    double objective = 0;
    for (double value : minimum)
      objective += value;
    objectives.push_back(objective / count);
    all_counts.insert(all_counts.end(), counts.begin(), counts.end());
    int farthest = std::max_element(minimum.begin(), minimum.end()) - minimum.begin();
    for (int k = 0; k < 16; ++k)
      for (int c = 0; c < 3; ++c)
        centers[k * 3 + c] = counts[k] ? sums[k * 3 + c] / counts[k] : points[farthest * 3 + c];
    history.insert(history.end(), centers.begin(), centers.end());
  }
  std::vector<float> palette(centers.begin(), centers.end());
  arrays.push_back({"palette", n::from_vector(palette, {16, 3})});
  arrays.push_back({"palette_history", n::from_vector(history, {11, 16, 3})});
  arrays.push_back({"palette_objective", n::from_vector(objectives)});
  arrays.push_back({"palette_counts", n::from_vector(all_counts, {10, 16})});
  arrays.push_back({"palette_seed_indices", n::from_vector(seeds)});
  for (int k = 0; k < 3; ++k) {
    int              rows = k == 0 ? 128 : 32;
    std::vector<int> target(rows * 64), input(rows * 64);
    for (int i = 0; i < rows * 64; ++i) {
      double best  = INFINITY;
      int    label = 0;
      for (int j = 0; j < 16; ++j) {
        double distance = 0;
        for (int c = 0; c < 3; ++c) {
          double d = double(codes[k][i * 3 + c]) / 255. - double(palette[j * 3 + c]);
          distance += d * d;
        }
        if (distance < best) {
          best  = distance;
          label = j;
        }
      }
      target[i] = label;
    }
    for (int b = 0; b < rows; ++b) {
      input[b * 64] = 16;
      for (int t = 1; t < 64; ++t)
        input[b * 64 + t] = target[b * 64 + t - 1];
    }
    arrays.push_back({splits[k] + "_target", n::from_vector(target, {rows, 64})});
    arrays.push_back({splits[k] + "_input", n::from_vector(input, {rows, 64})});
  }
  Save(
      root,
      "image_gpt.npz",
      arrays,
      Json{
          {"cifar_archive_sha256",
           "c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd"},
          {"fit_rows", {0, 127}},
          {"validation_rows", {128, 159}},
          {"test_rows", {64, 95}},
          {"test_source_id_offset", 50000},
          {"resize", "integer4x4area/sum+8//16/uint8"},
          {"palette_fit", "ONLYfit128images8192RGBpixels/farthest-firstseed9113/Lloyd10iterations"},
          {"palette_colors", 16},
          {"palette_float", "F64fitthenF32paletteBEFOREnearestassignment"},
          {"vocabulary_inputs", 17},
          {"vocabulary_outputs", 16},
          {"bos_id", 16},
          {"token_order", "rasterWHOLEcolorcode/inputBOS+pixels0..62/targetpixels0..63"},
          {"storage_note",
           "native tensors store source byte-valued arrays as Int32 without changing values"},
          {"limitation",
           "CPU8x8/K16/singleblockgenerativestudy,notpaper512paletteorfullImageNetpretraining/"
           "classification"}});
}
} // namespace language_assets
/// @brief Prepare the exact recovered upstream text/pixel dataset protocol in native C++.
/// @param name Dataset family registered by prepare_example_data.
/// @param root Owned output/cache directory.
void prepare_language_assets(const std::string& name, const fs::path& root) {
  fs::create_directories(root);
  if (name == "character_rnn")
    language_assets::Character(root);
  else if (name == "ptb_word")
    language_assets::Ptb(root);
  else if (name == "image_gpt")
    language_assets::ImagePalette(root);
  else if (name == "imdb_lstm")
    language_assets::Imdb(root);
  else if (name == "bilstm_tagging")
    language_assets::Tagging(root);
  else
    throw std::invalid_argument("unimplemented recovered dataset family: " + name);
}
} // namespace datamunge::datasets::detail
#endif
