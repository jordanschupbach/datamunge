/// @file reconstructed_assets.cpp
/// @brief Explicit version-one reconstructions of missing NN teaching-data recipes.
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include <algorithm>
#include <cmath>
#include <datamunge/datasets/download.hpp>
#include <datamunge/nn/datasets.hpp>
#include <datamunge/nn/io.hpp>
#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/safetensors.hpp>
#include <datamunge/nn/training.hpp>
#include <map>
#include <numeric>
#include <random>
#include <set>

#include "support.hpp"
namespace datamunge::datasets::detail {
/// @brief Private deterministic recipe assembly, with no optimizer or framework runtime.
namespace reconstructed {
namespace n = datamunge::nn;                    ///< Native owning arrays and serialization.
using A     = std::map<std::string, n::Tensor>; ///< Named recipe fields.

/// @brief Reject invalid source content.
/// @param ok Required invariant.
/// @param message Failure description.
void Check(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}
/// @brief Save a reconstructed recipe with its explicit limitations and content identity.
/// @param root Destination family directory.
/// @param arrays Owning native arrays.
/// @param protocol Complete reconstruction description.
void Save(const fs::path& root, const A& arrays, const std::string& protocol) {
  fs::create_directories(root);
  n::save_npz((root / "data.npz").string(), {arrays.begin(), arrays.end()});
  std::ofstream(root/"manifest.json")<<Json{{"recipe","datamunge-nn-reconstruction-v1"},{"original_recipe_recovered",false},{"protocol",protocol},{"limitation","Reconstructed preparation, not original TensorFlow data/checkpoint parity. Model equations, seeds and training budgets are preserved."},{"npz_sha256",file_digest((root/"data.npz").string())}}.dump(2)<<'\n';
}
/// @brief Load a prepared native NPZ into an owning name map.
/// @param file Readable archive.
/// @return Owned tensor handles.
A Load(const fs::path& file) {
  auto values = n::load_npz(file.string());
  return {values.begin(), values.end()};
}
/// @brief Publish an authenticated source download.
/// @param root Owned cache directory.
/// @param name Source basename.
/// @param url Upstream original URL.
/// @param sha Expected SHA256.
/// @return Cached authenticated path.
fs::path Fetch(
    const fs::path&    root,
    const std::string& name,
    const std::string& url,
    const std::string& sha) {
  fs::create_directories(root);
  auto p = root / name;
  fetch_file(url, p.string(), sha, 0, "nn-completion-assets");
  return p;
}
/// @brief Save standard integer source identity values.
/// @param arrays Accumulated fields.
/// @param prefix Split prefix.
/// @param rows Row count.
/// @param offset Disjoint original pool offset.
void Ids(A& arrays, const std::string& prefix, int rows, int offset) {
  std::vector<int> ids(rows);
  std::iota(ids.begin(), ids.end(), offset);
  arrays[prefix + "_source_ids"] = n::from_vector(ids);
}
/// @brief Reconstruct fixed small CIFAR pools without touching the held-out source files.
/// @param root Family directory.
void Vision(const fs::path& root) {
  auto source = datamunge::datasets::fetch("cifar10");
  auto train = n::load_cifar10(source, "train"), test = n::load_cifar10(source, "test");
  A    a;
  for (int s = 0; s < 3; ++s) {
    std::string               p    = s == 0 ? "fit" : s == 1 ? "validation" : "test";
    int                       rows = s ? 32 : 128, start = s == 1 ? 128 : s == 2 ? 64 : 0;
    std::vector<std::int64_t> ids(rows);
    std::iota(ids.begin(), ids.end(), start);
    auto& d          = s == 2 ? test : train;
    a[p + "_images"] = n::take_rows(d.images, ids);
    a[p + "_labels"] = n::cast(n::take_rows(d.labels, ids), n::DType::Int32);
    Ids(a, p, rows, start + (s == 2 ? 50000 : 0));
  }
  Save(
      root,
      a,
      "CIFAR-10 authenticated canonical files; train rows0..127 fit,128..159 validation; official "
      "test64..95; native RGB Float32 /255. Reconstructed partition, not recovered original "
      "manifest.");
}
/// @brief Reconstruct chronological windows with fitting-only statistics from pinned yearly
/// observations.
/// @param root Family directory.
void Forecast(const fs::path& root) {
  auto file = Fetch(
      root,
      "sunspots.csv",
      "https://raw.githubusercontent.com/statsmodels/statsmodels/v0.14.6/statsmodels/datasets/"
      "sunspots/sunspots.csv",
      "f67889b1d9002cd5227f0e0ef54e35b419cdd85a31279adef6f73fb41e5c0a9b");
  std::ifstream in(file);
  std::string   line;
  std::getline(in, line);
  std::vector<float> values;
  while (std::getline(in, line)) {
    auto pos = line.find(',');
    if (pos != std::string::npos)
      values.push_back(std::stof(line.substr(pos + 1)));
  }
  Check(values.size() == 309, "309 pinned sunspot observations");
  double mean = 0, sd = 0;
  for (int i = 0; i < 222; ++i)
    mean += values[i] / 222.;
  for (int i = 0; i < 222; ++i)
    sd += (values[i] - mean) * (values[i] - mean) / 222.;
  sd = std::sqrt(sd);
  A a{{"normalization", n::from_vector(std::vector<float>{float(mean), float(sd)})}};
  for (int s = 0; s < 3; ++s) {
    int                rows = s ? 32 : 208, start = s == 0 ? 12 : s == 1 ? 222 : 256;
    std::string        p = s == 0 ? "fit" : s == 1 ? "validation" : "test";
    std::vector<float> x, y;
    std::vector<int>   origins;
    for (int r = 0; r < rows; ++r) {
      int origin = start + r;
      origins.push_back(1700 + origin);
      x.insert(x.end(), values.begin() + origin - 12, values.begin() + origin);
      y.insert(y.end(), values.begin() + origin, values.begin() + origin + 3);
    }
    a[p + "_input"]   = n::from_vector(x, {rows, 12, 1});
    a[p + "_target"]  = n::from_vector(y, {rows, 3});
    a[p + "_origins"] = n::from_vector(origins);
  }
  Save(
      root,
      a,
      "Pinned statsmodels0.14.6 yearly1700..2008 sunspots.208 fit origins1712..1919; "
      "validation1922..1953; test1956..1987;12-year past and3-year target; no target overlap "
      "across partitions; mean/populationSD only first222 years. Reconstructed boundaries.");
}
/// @brief Reconstruct procedural sequence tasks with disjoint deterministic streams.
/// @param root Family directory.
/// @param pointer True generates coordinate targets; otherwise token reversal/addition/copy.
/// @param copy True uses extended-vocabulary copying.
void Sequence(const fs::path& root, bool pointer, bool copy) {
  for (std::string task : pointer ? std::vector<std::string>{"sorting", "convex_hull"}
                          : copy  ? std::vector<std::string>{"copy"}
                                  : std::vector<std::string>{"reversal", "addition"}) {
    A a;
    for (int s = 0; s < 3; ++s) {
      int          rows = s ? 32 : 128;
      std::string  p    = s == 0 ? "fit" : s == 1 ? "validation" : "test";
      std::mt19937 rng(44317 + 1009 * s + (task == "addition" || task == "convex_hull" ? 97 : 0));
      std::vector<int>   x(rows * 6), teacher(rows * 6), target(rows * 6), il(rows), ol(rows);
      std::vector<float> points(rows * 10);
      for (int r = 0; r < rows; ++r) {
        int len = 2 + int(rng() % 4);
        il[r]   = len;
        std::vector<int> out;
        if (pointer) {
          for (int j = 0; j < len; ++j) {
            points[r * 10 + j * 2]     = float(rng() % 10001) / 10000;
            points[r * 10 + j * 2 + 1] = float(rng() % 10001) / 10000;
          }
          out.resize(len);
          std::iota(out.begin(), out.end(), 0);
          if (task == "sorting")
            std::stable_sort(out.begin(), out.end(), [&](int l, int k) {
              return points[r * 10 + 2 * l] < points[r * 10 + 2 * k];
            });
          else {
            auto cross = [&](int a, int b, int c) {
              return (points[r * 10 + 2 * b] - points[r * 10 + 2 * a]) *
                         (points[r * 10 + 2 * c + 1] - points[r * 10 + 2 * a + 1]) -
                     (points[r * 10 + 2 * b + 1] - points[r * 10 + 2 * a + 1]) *
                         (points[r * 10 + 2 * c] - points[r * 10 + 2 * a]);
            };
            std::sort(out.begin(), out.end(), [&](int l, int k) {
              return points[r * 10 + 2 * l] < points[r * 10 + 2 * k];
            });
            std::vector<int> h;
            for (int k : out) {
              while (h.size() > 1 && cross(h[h.size() - 2], h.back(), k) <= 0)
                h.pop_back();
              h.push_back(k);
            }
            size_t lower = h.size();
            for (int j = len - 2; j >= 0; --j) {
              int k = out[j];
              while (h.size() > lower && cross(h[h.size() - 2], h.back(), k) <= 0)
                h.pop_back();
              h.push_back(k);
            }
            if (h.size() > 1)
              h.pop_back();
            out = h;
          }
          out.push_back(5);
        } else {
          for (int j = 0; j < len; ++j) {
            x[r * 6 + j] = copy ? 3 + int(rng() % 17) : 3 + int(rng() % 10);
            out.push_back(x[r * 6 + j]);
          }
          if (task == "reversal")
            std::reverse(out.begin(), out.end());
          if (task == "addition") {
            int a = x[r * 6] - 3, b = x[r * 6 + 1] - 3;
            x[r * 6]     = a + 3;
            x[r * 6 + 1] = 13;
            x[r * 6 + 2] = b + 3;
            il[r]        = 3;
            out.clear();
            if (a + b >= 10)
              out.push_back(4);
            out.push_back((a + b) % 10 + 3);
          }
          out.push_back(2);
        }
        ol[r]          = out.size();
        teacher[r * 6] = pointer ? -1 : 1;
        for (size_t j = 0; j < out.size(); ++j) {
          target[r * 6 + j] = out[j];
          if (j + 1 < out.size())
            teacher[r * 6 + j + 1] = out[j];
        }
      }
      a[p + "_input"] =
          pointer ? n::from_vector(points, {rows, 5, 2}) : n::from_vector(x, {rows, 6});
      a[p + "_teacher"]        = n::from_vector(teacher, {rows, 6});
      a[p + "_target"]         = n::from_vector(target, {rows, 6});
      a[p + "_input_lengths"]  = n::from_vector(il);
      a[p + "_output_lengths"] = n::from_vector(ol);
      Ids(a, p, rows, s * 10000);
    }
    Save(
        root / task,
        a,
        "Deterministic reconstructed procedural " + task +
            " task. mt19937 seeds44317+1009*split+97 for addition/hull;128/32/32 rows. Targets "
            "derive solely from each source, with BOS/EOS/STOP and masked padding. No external "
            "corpus or TensorFlow parity claimed.");
  }
}
/// @brief Split ASCII tokens while retaining apostrophes and common punctuation as bytes.
/// @param text Source sentence.
/// @return Lowercase whitespace-delimited words.
std::vector<std::string> Words(std::string text) {
  for (char& c : text) {
    unsigned char u = c;
    if (u > 127)
      continue;
    else if (c >= 'A' && c <= 'Z')
      c += 32;
    else if (!std::isalnum(u) && c != '\'')
      c = ' ';
  }
  std::istringstream       in(text);
  std::vector<std::string> out;
  std::string              w;
  while (in >> w)
    out.push_back(w);
  return out;
}
/// @brief Reconstruct short English/French examples from authenticated tutorial source rows.
/// @param root Family directory.
void Translation(const fs::path& root) {
  auto zip = Fetch(
      root,
      "translation.zip",
      "https://download.pytorch.org/tutorial/data.zip",
      "fb317e80248faeb62dc25ef3390ae24ca34b94e276bbc5141fd8862c2200bff5");
  extract_archive(zip.string(), root.string(), {"data/eng-fra.txt"});
  fs::copy_file(
      root / "data/eng-fra.txt", root / "eng-fra.txt", fs::copy_options::overwrite_existing);
  std::ifstream in(root / "eng-fra.txt");
  std::string   line;
  std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> pairs;
  std::vector<int>                                                           source;
  std::set<std::string> seen,seen_source,seen_target;
  int                                                                        id = 0;
  while (std::getline(in, line) && pairs.size() < 192) {
    auto tab = line.find('\t');
    if (tab == std::string::npos)
      continue;
    auto x = Words(line.substr(0, tab)), y = Words(line.substr(tab + 1));
    if (!x.empty() && !y.empty() && x.size() <= 5 && y.size() <= 5 && !seen.count(line)) {
      std::string source_key, target_key;
      for (const auto& w : x)
        source_key += w + " ";
      for (const auto& w : y)
        target_key += w + " ";
      if (seen_source.count(source_key) || seen_target.count(target_key)) {
        ++id;
        continue;
      }
      seen.insert(line);
      seen_source.insert(source_key);
      seen_target.insert(target_key);
      pairs.push_back({x, y});
      source.push_back(id);
    }
    ++id;
  }
  Check(pairs.size() == 192, "192 short translation pairs");
  std::map<std::string, int> freq;
  for (int r = 0; r < 128; ++r)
    for (auto* part : {&pairs[r].first, &pairs[r].second})
      for (auto& w : *part)
        ++freq[w];
  std::vector<std::pair<std::string, int>> rank(freq.begin(), freq.end());
  std::sort(rank.begin(), rank.end(), [](auto& a, auto& b) {
    return a.second != b.second ? a.second > b.second : a.first < b.first;
  });
  std::vector<std::string> vocab{"[PAD]", "[BOS]", "[EOS]", "[UNK]"};
  for (auto& [word, count] : rank)
    if (vocab.size() < 256)
      vocab.push_back(word);
  std::map<std::string, int> lookup;
  std::ofstream              vocabulary(root / "vocabulary.txt");
  for (size_t i = 0; i < 256; ++i) {
    std::string word = i < vocab.size() ? vocab[i] : "[UNUSED" + std::to_string(i) + "]";
    lookup[word]     = i;
    vocabulary << word << '\n';
  }
  A a;
  for (int s = 0; s < 3; ++s) {
    int              rows = s ? 32 : 128, start = s == 0 ? 0 : s == 1 ? 128 : 160;
    std::string      p = s == 0 ? "fit" : s == 1 ? "validation" : "test";
    std::vector<int> x(rows * 6), teacher(rows * 6), target(rows * 6), il, ol, ids;
    for (int r = 0; r < rows; ++r) {
      auto& pair = pairs[start + r];
      auto  code = [&](const std::string& w) {
        auto it = lookup.find(w);
        return it == lookup.end() ? 3 : it->second;
      };
      il.push_back(pair.first.size());
      ol.push_back(pair.second.size() + 1);
      ids.push_back(source[start + r]);
      for (size_t j = 0; j < pair.first.size(); ++j)
        x[r * 6 + j] = code(pair.first[j]);
      teacher[r * 6] = 1;
      for (size_t j = 0; j < pair.second.size(); ++j) {
        target[r * 6 + j]      = code(pair.second[j]);
        teacher[r * 6 + j + 1] = code(pair.second[j]);
      }
      target[r * 6 + pair.second.size()] = 2;
    }
    a[p + "_input"]          = n::from_vector(x, {rows, 6});
    a[p + "_teacher"]        = n::from_vector(teacher, {rows, 6});
    a[p + "_target"]         = n::from_vector(target, {rows, 6});
    a[p + "_input_lengths"]  = n::from_vector(il);
    a[p + "_output_lengths"] = n::from_vector(ol);
    a[p + "_source_ids"]     = n::from_vector(ids);
  }
  std::ofstream(root / "attribution.tsv")
      << "source\tPyTorch tutorial data/eng-fra.txt; Tatoeba sentence contributors\n";
  std::ofstream(root / "_about.txt")
      << "Reconstructed short-sentence subset of authenticated PyTorch tutorial English/French "
         "data; fit-only shared vocabulary; ASCII case folding preserves UTF8 word bytes.\n";
  Save(
      root,
      a,
      "First192 unique nonempty English/French pairs with each side<=5 ASCII-tokenized words; "
      "first128 fit,next32 validation,last32 test;256 shared fit-only vocabulary frequency/lexical "
      "order, PAD0 BOS1 EOS2 UNK3. Original preparation unavailable; reconstructed subset, no "
      "translation benchmark claim.");
}
/// @brief Reconstruct character-level masked-language and segment-memory pools from original pinned
/// corpus partitions.
/// @param root Shared recipe root.
void Language(const fs::path& root) {
  fs::path       source = prepare_example_data("character_rnn");
  std::string    text = read(source / "input.txt"), alphabet = read(source / "vocabulary.txt");
  std::set<char> letters;
  for (size_t i = 0; i < text.size() * 9 / 10; ++i)
    letters.insert(text[i]);
  Check(letters.size() == 65, "original65-character training alphabet");
  std::map<char, int> codes;
  for (char c : letters)
    codes[c] = codes.size();
  std::vector<int> corpus;
  for (char c : text)
    corpus.push_back(codes.at(c));
  auto word = [&](const fs::path& dir) {
    fs::create_directories(dir);
    std::ofstream(dir / "vocabulary.json")
        << Json{{"base", "original fit-only sorted65-character alphabet"}, {"special_ids", {{"PAD", 0}, {"UNK", 1}, {"CLS", 2}, {"SEP", 3}, {"MASK", 4}}}, {"character_offset", 5}, {"reconstruction", true}}
               .dump(2)
        << '\n';
    std::ofstream(dir / "vocabulary.txt") << alphabet;
  };
  A mlm, roberta, sw;
  for (int s = 0; s < 3; ++s) {
    int         rows   = s ? 32 : 128;
    std::string p      = s == 0 ? "fit" : s == 1 ? "validation" : "test";
    size_t      offset = s == 0 ? 0 : s == 1 ? text.size() * 9 / 10 : text.size() * 95 / 100;
    auto        make   = [&](unsigned seed) {
      std::mt19937     rng(seed);
      A                a;
      std::vector<int> x(rows * 32), y(rows * 32), segments(rows * 32), lengths(rows), starts(rows),
          actions(rows * 32);
      std::vector<float> selected(rows * 32);
      for (int r = 0; r < rows; ++r) {
        int len    = 24 + r % 7;
        lengths[r] = len;
        starts[r]  = offset + r * 65;
        x[r * 32] = y[r * 32] = 2;
        for (int t = 1; t < len - 1; ++t) {
          int i       = r * 32 + t;
          y[i]        = corpus[offset + r * 65 + t] + 5;
          x[i]        = y[i];
          segments[i] = t > len / 2;
          if (t == 1 || rng() % 100 < 15) {
            selected[i] = 1;
            int choice  = rng() % 10;
            if (choice < 8) {
              x[i]       = 4;
              actions[i] = 1;
            } else if (choice == 8) {
              x[i]       = 5 + rng() % 65;
              actions[i] = 2;
            } else
              actions[i] = 3;
          }
        }
        x[r * 32 + len - 1] = y[r * 32 + len - 1] = 3;
      }
      a["input"]        = n::from_vector(x, {rows, 32});
      a["target"]       = n::from_vector(y, {rows, 32});
      a["segments"]     = n::from_vector(segments, {rows, 32});
      a["lengths"]      = n::from_vector(lengths);
      a["source_start"] = n::from_vector(starts);
      a["actions"]      = n::from_vector(actions, {rows, 32});
      a["selected"]     = n::from_vector(selected, {rows, 32});
      return a;
    };
    auto a = make(46401 + 1009 * s);
    if (!s)
      for (const auto* k : {"target", "segments", "lengths", "source_start"})
        roberta["fit_" + std::string(k)] = a.at(k);
    for (auto& [k, v] : a)
      mlm[p + "_" + k] = v;
    if (!s) {
      for (int epoch = 1; epoch <= 100; ++epoch)
        for (auto& [k, v] : make(46701 + epoch))
          roberta["epoch" + std::to_string(epoch) + "_" + k] = v;
    } else
      for (auto& [k, v] : a)
        roberta[p + "_" + k] = v;
    std::vector<int> x(rows * 16), y(rows * 16), lengths(rows, 16), starts(rows);
    for (int r = 0; r < rows; ++r) {
      starts[r] = offset + r * 17;
      for (int t = 0; t < 16; ++t) {
        x[r * 16 + t] = corpus[offset + r * 17 + t];
        y[r * 16 + t] = corpus[offset + r * 17 + t + 1];
      }
    }
    sw[p + "_input"]   = n::from_vector(x, {rows, 16});
    sw[p + "_target"]  = n::from_vector(y, {rows, 16});
    sw[p + "_lengths"] = n::from_vector(lengths);
    sw[p + "_start"]   = n::from_vector(starts);
  }
  word(root / "bert_mlm");
  Save(
      root / "bert_mlm",
      mlm,
      "Original hash-verified tinyShakespeare90/5/5 raw partitions. Reconstructed24..30-position "
      "character sentence-pair surrogate,65 fit characters+5 special tokens;32 context;128/32/32; "
      "source starts offset+r*65;80/10/10 corruption, selection probability.15, guaranteed first "
      "token, seeds46401+1009*split. This is character-level teaching MLM, not WordPiece or "
      "original missing corpus preparation.");
  word(root / "roberta_tweaks");
  Save(
      root / "roberta_tweaks",
      roberta,
      "Same character/pool protocol as reconstructed bert_mlm; fit corruption independently "
      "seeded46701+epoch,100 stored epochs; held-out corruption fixed. Static/dynamic controls "
      "reuse identical source starts.");
  word(root / "switch_transformer");
  Save(
      root / "switch_transformer",
      sw,
      "Original tinyShakespeare90/5/5 partitions,65 original fit-only characters; "
      "reconstructed128/32/32 disjoint17-character windows,16 next-token decisions, starts "
      "partition_start+r*17.");
  for (std::string family : {"transformer_xl", "xlnet"}) {
    A a;
    for (int s = 0; s < 3; ++s) {
      int              steps = s ? 4 : 16;
      std::string      p     = s == 0 ? "fit" : s == 1 ? "validation" : "test";
      size_t           base  = s == 0 ? 0 : s == 1 ? text.size() * 9 / 10 : text.size() * 95 / 100;
      std::vector<int> x(steps * 4 * 16), y(x.size()), starts(4);
      for (int stream = 0; stream < 4; ++stream) {
        starts[stream] = base + stream * (steps * 16 + 1);
        for (int k = 0; k < steps; ++k)
          for (int t = 0; t < 16; ++t) {
            int    i        = (k * 4 + stream) * 16 + t;
            size_t position = starts[stream] + k * 16 + t;
            x[i]            = corpus[position];
            y[i]            = corpus[position + 1];
          }
      }
      a[p + "_input"]  = n::from_vector(x, {steps, 4, 16});
      a[p + "_target"] = n::from_vector(y, {steps, 4, 16});
      a[p + "_start"]  = n::from_vector(starts);
      if (family == "xlnet")
        for (int epoch = 0; epoch <= (s ? 0 : 100); ++epoch) {
          std::mt19937     rng(47201 + s * 1009 + epoch);
          std::vector<int> rank(x.size());
          for (int row = 0; row < steps * 4; ++row) {
            std::vector<int> perm(16);
            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), rng);
            for (int t = 0; t < 16; ++t)
              rank[row * 16 + t] = perm[t];
          }
          a[p + "_rank" + std::to_string(epoch)] = n::from_vector(rank, {steps, 4, 16});
        }
    }
    word(root / family);
    Save(
        root / family,
        a,
        "Reconstructed four independent chronological character streams in original "
        "tinyShakespeare90/5/5 partitions; fit16segments,validation/test4segments,16 "
        "tokens/segment. XL target next character; XLNet predicts current character using "
        "target-excluding random factorization ranks seeded47201+1009*split+epoch. No stream "
        "crosses a raw partition.");
  }
}
/// @brief Reconstruct real-MNIST CTC rows and disjoint procedural pitch-class melodies.
/// @param root Shared recipe root.
void CtcMidi(const fs::path& root) {
  auto home  = datamunge::datasets::fetch("mnist");
  auto train = n::load_mnist(home, "train"), test = n::load_mnist(home, "test");
  A    ctc, midi;
  for (int s = 0; s < 3; ++s) {
    int                rows = s ? 32 : 128, start = s == 1 ? 1000 : 0;
    std::string        p      = s == 0 ? "fit" : s == 1 ? "validation" : "test";
    auto&              images = s == 2 ? test : train;
    auto               pixels = images.images.to_vector<float>();
    auto               labels = images.labels.to_vector<int>();
    std::vector<float> x(rows * 8 * 49);
    std::vector<int>   y(rows * 3), lengths(rows), label_lengths(rows), ids(rows), notes(rows * 16),
        input(rows * 16);
    std::mt19937 rng(44901 + 1009 * s);
    for (int r = 0; r < rows; ++r) {
      int count        = 1 + r % 3;
      lengths[r]       = 2 * count + 1;
      label_lengths[r] = count;
      ids[r]           = start + 3 * r + (s == 2 ? 60000 : 0);
      for (int j = 0; j < count; ++j) {
        int index    = start + 3 * r + j;
        y[r * 3 + j] = labels[index];
        for (int yy = 0; yy < 7; ++yy)
          for (int xx = 0; xx < 7; ++xx) {
            float value = 0;
            for (int dy = 0; dy < 4; ++dy)
              for (int dx = 0; dx < 4; ++dx)
                value += pixels[index * 784 + (yy * 4 + dy) * 28 + xx * 4 + dx] / 16.f;
            for (int repeat = 0; repeat < 2; ++repeat)
              x[(r * 8 + 2 * j + repeat) * 49 + yy * 7 + xx] = value;
          }
      }
      int pitch = rng() % 12;
      for (int t = 0; t < 16; ++t) {
        static constexpr int scale[]{0, 2, 4, 5, 7, 9, 11};
        notes[r * 16 + t] = (pitch + scale[(t / 2 + r) % 7] + (rng() % 5 == 0 ? 2 : 0)) % 12;
        input[r * 16 + t] = t ? notes[r * 16 + t - 1] : 12;
      }
    }
    ctc[p + "_input"]         = n::from_vector(x, {rows, 8, 49});
    ctc[p + "_labels"]        = n::from_vector(y, {rows, 3});
    ctc[p + "_lengths"]       = n::from_vector(lengths);
    ctc[p + "_label_lengths"] = n::from_vector(label_lengths);
    ctc[p + "_source_ids"]    = n::from_vector(ids);
    midi[p + "_input"]        = n::from_vector(input, {rows, 16});
    midi[p + "_target"]       = n::from_vector(notes, {rows, 16});
    Ids(midi, p, rows, 10000 * s);
  }
  Save(
      root / "ctc_handwriting",
      ctc,
      "Authenticated MNIST;128/32/32 sequences,1..3 source digits,integer4x4 average to7x7; each "
      "digit repeated twice then blank frame,8 padded frames; first train0..383 "
      "fit,train1000..1095 validation,official test0..95. No source image reused across splits. "
      "Reconstructed frame composition.");
  Save(
      root / "midi_lstm",
      midi,
      "Reconstructed procedural diatonic pitch-class melodies,16 notes,128/32/32,mt19937 "
      "seed44901+1009*split;start token12+shiftednotes,output0..11. Distinct melody draws; this "
      "recipe directly creates notes and does not claim the missing MIDI serialization roundtrip.");
}
/// @brief Reconstruct real pen-coordinate prompts from pinned UCI PenDigits source rows.
/// @param root Family directory.
void Handwriting(const fs::path& root) {
  auto zip = Fetch(
      root,
      "pendigits.zip",
      "https://archive.ics.uci.edu/static/public/81/"
      "pen+based+recognition+of+handwritten+digits.zip",
      "1e02bea023613c2b11c9492f6f34caf975420455934f3527d270cee9a1f03b64");
  extract_archive(zip.string(), root.string(), {"pendigits.tra", "pendigits.tes"});
  A a;
  for (int s = 0; s < 3; ++s) {
    std::ifstream                   in(root / (s == 2 ? "pendigits.tes" : "pendigits.tra"));
    std::vector<std::vector<float>> raw;
    std::string                     line;
    while (std::getline(in, line)) {
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream row(line);
      std::vector<float> v;
      float              value;
      while (row >> value)
        v.push_back(value);
      if (v.size() == 17)
        raw.push_back(v);
    }
    int                count = s ? 40 : 140, start = s == 1 ? 420 : 0;
    std::string        p = s == 0 ? "fit" : s == 1 ? "validation" : "test";
    std::vector<float> x(count * 33 * 3), y(x.size());
    std::vector<int>   labels(count * 3), ids(count);
    for (int r = 0; r < count; ++r) {
      float previous_x = 0, previous_y = 0;
      ids[r] = start + 3 * r + (s == 2 ? 10000 : 0);
      for (int symbol = 0; symbol < 3; ++symbol) {
        const auto& v          = raw.at(start + 3 * r + symbol);
        labels[r * 3 + symbol] = int(v[16]);
        for (int t = 0; t < 11; ++t) {
          double pos   = t * 7. / 10.;
          int    i     = std::min(int(pos), 6);
          double alpha = pos - i;
          float  xx    = (v[2 * i] * (1 - alpha) + v[2 * i + 2] * alpha) / 100.f + symbol * 1.2f,
                yy     = (v[2 * i + 1] * (1 - alpha) + v[2 * i + 3] * alpha) / 100.f;
          int step = symbol * 11 + t, index = (r * 33 + step) * 3;
          y[index]     = xx - previous_x;
          y[index + 1] = yy - previous_y;
          y[index + 2] = t == 10;
          previous_x   = xx;
          previous_y   = yy;
        }
      }
      for (int t = 1; t < 33; ++t)
        for (int c = 0; c < 3; ++c)
          x[(r * 33 + t) * 3 + c] = y[(r * 33 + t - 1) * 3 + c];
    }
    a[p + "_input"]      = n::from_vector(x, {count, 33, 3});
    a[p + "_target"]     = n::from_vector(y, {count, 33, 3});
    a[p + "_labels"]     = n::from_vector(labels, {count, 3});
    a[p + "_source_ids"] = n::from_vector(ids);
  }
  std::ofstream(root / "ATTRIBUTION.txt")
      << "UCI Pen-Based Recognition of Handwritten Digits; Alpaydin and Alimoglu(1996), "
         "doi:10.24432/C5MG6K. Reconstructed3-digit trajectories, not the missing "
         "original21-symbol corpus.\n";
  Save(
      root,
      a,
      "Reconstructed3-digit prompts from original UCI PenDigits16coordinates+label. Train first420 "
      "source rows fit140 sequences,train420..539 validation40,official testfirst120 rows test40. "
      "Linear interpolation8points->11per digit, x offset1.2 per symbol, coordinate "
      "scale100,pen-up after eachdigit;33 next-offset steps. Only digit IDs0..9 of21-symbol model "
      "used; original source alphabet/protocol unavailable.");
}
/// @brief Reconstruct character-tokenized sentiment, aligned tags, synthetic QA and paired-text
/// controls.
/// @param root Shared recipe root.
void TextTasks(const fs::path& root) {
  fs::path textroot = prepare_example_data("text"), chars = prepare_example_data("character_rnn");
  auto     original = read(chars / "input.txt");
  std::set<char> alphabet;
  for (size_t i = 0; i < original.size() * 9 / 10; ++i)
    alphabet.insert(original[i]);
  std::map<char, int> codes;
  for (char c : alphabet)
    codes[c] = codes.size() + 5;
  auto encode = [&](const std::string& s, int max) {
    std::vector<int> out;
    for (char c : s) {
      if (int(out.size()) >= max)
        break;
      auto it = codes.find(c);
      out.push_back(it == codes.end() ? 1 : it->second);
    }
    return out;
  };
  auto                                       meta = json(textroot / "sentence/manifest.json");
  std::map<int, std::pair<int, std::string>> reviews;
  std::ifstream                              in(textroot / "sentence/reviews.tsv");
  std::string                                line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    auto first = line.find('\t'), second = line.find('\t', first + 1);
    Check(second != std::string::npos, "review ledger columns");
    reviews[std::stoi(line.substr(0, first))] = {
        std::stoi(line.substr(first + 1, second - first - 1)), line.substr(second + 1)};
  }
  A classification, lora, instruction, pairs, denoise, sentiment;
  for (int s = 0; s < 3; ++s) {
    int         rows = s ? 16 : 64;
    std::string p    = s == 0 ? "fit" : s == 1 ? "validation" : "test";
    auto        ids  = meta["source_ids"][p].get<std::vector<int>>();
    Check(int(ids.size()) >= rows, "enough original disjoint review rows");
    std::vector<int> x(rows * 128), seg(x.size()), lengths(rows), labels(rows), source_ids(rows),
        ib(rows * 64, 65), it(rows * 64, 65), ip(rows * 64, 65), il(rows), pl(rows);
    std::vector<float> weights(rows * 128), selected(rows * 64);
    std::vector<int> tx(rows * 48), tt(rows * 48), td(rows * 48), tl(rows), to(rows), dx(rows * 48),
        dt(rows * 48), dd(rows * 48), dl(rows), dout(rows);
    for (int r = 0; r < rows; ++r) {
      auto& review  = reviews.at(ids[r]);
      source_ids[r] = ids[r];
      labels[r]     = review.first;
      auto tokens   = encode(review.second, 124);
      x[r * 128]    = 2;
      for (size_t j = 0; j < tokens.size(); ++j)
        x[r * 128 + j + 1] = tokens[j];
      lengths[r]                  = tokens.size() + 2;
      x[r * 128 + lengths[r] - 1] = 3;
      for (int t = 0; t < lengths[r]; ++t)
        weights[r * 128 + t] = 1;
      std::string      prompt = "sentiment: " + review.second.substr(0, 20) + " answer:";
      auto             pc     = encode(prompt, 40);
      std::vector<int> raw;
      for (char c : prompt.substr(0, 40)) {
        auto it = codes.find(c);
        raw.push_back(it == codes.end() ? 0 : it->second - 5);
      }
      pl[r] = raw.size();
      for (size_t t = 0; t < raw.size(); ++t)
        ip[r * 64 + t] = raw[t];
      std::string      answer = review.first ? "positive" : "negative";
      std::vector<int> completion;
      for (char c : answer) {
        auto it = codes.find(c);
        completion.push_back(it == codes.end() ? 0 : it->second - 5);
      }
      completion.push_back(66);
      std::vector<int> sequence = raw;
      sequence.insert(sequence.end(), completion.begin(), completion.end());
      il[r] = sequence.size() - 1;
      for (int t = 0; t < il[r]; ++t) {
        ib[r * 64 + t]       = sequence[t];
        it[r * 64 + t]       = sequence[t + 1];
        selected[r * 64 + t] = t >= pl[r] - 1;
      }
      auto source = encode("sentiment: " + review.second, 46), answer_tokens = encode(answer, 44);
      tl[r] = source.size();
      to[r] = answer_tokens.size() + 1;
      for (size_t t = 0; t < source.size(); ++t)
        tx[r * 48 + t] = source[t];
      for (size_t t = 0; t < answer_tokens.size(); ++t) {
        tt[r * 48 + t]     = answer_tokens[t];
        td[r * 48 + t + 1] = answer_tokens[t];
      }
      tt[r * 48 + answer_tokens.size()] = 1;
      auto clean                        = encode(review.second, 44);
      int  begin                        = std::min(5, int(clean.size() / 2)),
          removed                       = std::min(5, int(clean.size()) - begin);
      std::vector<int> masked(clean.begin(), clean.begin() + begin);
      masked.push_back(70);
      masked.insert(masked.end(), clean.begin() + begin + removed, clean.end());
      std::vector<int> target{70};
      target.insert(target.end(), clean.begin() + begin, clean.begin() + begin + removed);
      target.push_back(71);
      target.push_back(1);
      dl[r]   = masked.size();
      dout[r] = target.size();
      for (size_t t = 0; t < masked.size(); ++t)
        dx[r * 48 + t] = masked[t];
      for (size_t t = 0; t < target.size(); ++t) {
        dt[r * 48 + t] = target[t];
        if (t + 1 < target.size())
          dd[r * 48 + t + 1] = target[t];
      }
    }
    classification[p + "_input"]      = n::from_vector(x, {rows, 128});
    classification[p + "_segments"]   = n::from_vector(seg, {rows, 128});
    classification[p + "_lengths"]    = n::from_vector(lengths);
    classification[p + "_labels"]     = n::from_vector(labels);
    classification[p + "_weights"]    = n::from_vector(weights, {rows, 128});
    classification[p + "_source_ids"] = n::from_vector(source_ids);
    for (auto k : {"input", "lengths", "labels", "source_ids"})
      lora[p + "_" + k] = classification.at(p + "_" + k);
    instruction[p + "_input"]          = n::from_vector(ib, {rows, 64});
    instruction[p + "_target"]         = n::from_vector(it, {rows, 64});
    instruction[p + "_selected"]       = n::from_vector(selected, {rows, 64});
    instruction[p + "_lengths"]        = n::from_vector(il);
    instruction[p + "_prompt"]         = n::from_vector(ip, {rows, 64});
    instruction[p + "_prompt_lengths"] = n::from_vector(pl);
    instruction[p + "_source_ids"]     = n::from_vector(source_ids);
    for (auto mode : {"sbert", "dual"}) {
      std::vector<int>   right = x;
      std::vector<float> score(rows, 1);
      for (int r = 0; r < rows; ++r)
        if (std::string(mode) == "sbert" && r % 2) {
          int other = (r + 1) % rows;
          for (int t = 0; t < 128; ++t)
            right[r * 128 + t] = x[other * 128 + t];
          score[r] = labels[r] == labels[other] ? 0.5f : 0.f;
        }
      std::vector<int> rl = lengths;
      if (std::string(mode) == "sbert")
        for (int r = 1; r < rows; r += 2)
          rl[r] = lengths[(r + 1) % rows];
      std::string key               = std::string(mode) + "_" + p;
      pairs[key + "_input_a"]       = n::from_vector(x, {rows, 128});
      pairs[key + "_input_b"]       = n::from_vector(right, {rows, 128});
      pairs[key + "_lengths_left"]  = n::from_vector(lengths);
      pairs[key + "_lengths_right"] = n::from_vector(rl);
      pairs[key + "_score"]         = n::from_vector(score);
      pairs[key + "_source_id"]     = n::from_vector(source_ids);
    }
    for (int type = 0; type < 2; ++type) {
      A& a                     = type ? sentiment : denoise;
      a[p + "_input"]          = n::from_vector(type ? tx : dx, {rows, 48});
      a[p + "_target"]         = n::from_vector(type ? tt : dt, {rows, 48});
      a[p + "_teacher"]        = n::from_vector(type ? td : dd, {rows, 48});
      a[p + "_input_lengths"]  = n::from_vector(type ? tl : dl);
      a[p + "_output_lengths"] = n::from_vector(type ? to : dout);
      a[p + "_source_ids"]     = n::from_vector(source_ids);
    }
  }
  std::string protocol =
      "Reconstructed character-level tasks from authenticated Pang/Lee MR v1.0 original "
      "duplicate-group-disjoint native fit/validation/test pools;64/16/16 rows,first source IDs in "
      "original manifest.65 tinyShakespeare fit characters+special IDs,unknown1. Sentiment uses "
      "actual source labels; no WordPiece or original missing BERT task recipe parity.";
  Save(root / "bert_finetune/classification", classification, protocol);
  Save(root / "lora_adapters", lora, protocol);
  Save(
      root / "instruction_tuning",
      instruction,
      protocol +
          " Original65 GPT character IDs, PAD65 EOS66;prompt<=40,completion positive/negative,CE "
          "only completion,own-token generation.");
  Save(
      root / "sentence_embeddings",
      pairs,
      protocol +
          " Explicit self-retrieval dual pairs;regression scores1 for identical review,.5 for same "
          "sentiment and0 for different sentiment. These labels are constructed controls, not "
          "human similarity annotations or retrieval benchmarks.");
  Save(
      root / "t5_text/denoise",
      denoise,
      protocol +
          " Single5-character removed span, sentinel70 then71, PAD/BOS0 EOS1;48 "
          "context;teacher-forced restoration.");
  Save(
      root / "t5_text/sentiment",
      sentiment,
      protocol +
          " Input sentiment prefix+review,target positive/negative+EOS1;full transfer from locally "
          "trained span-denoising checkpoint.");
  fs::path                   conll = prepare_example_data("bilstm_tagging");
  A                          ner, qa;
  std::map<std::string, int> tag_ids{
      {"O", 0},
      {"B-PER", 1},
      {"I-PER", 2},
      {"B-ORG", 3},
      {"I-ORG", 4},
      {"B-LOC", 5},
      {"I-LOC", 6},
      {"B-MISC", 7},
      {"I-MISC", 8}};
  for (int s = 0; s < 3; ++s) {
    int           rows = s ? 16 : 64;
    std::string   p    = s == 0 ? "fit" : s == 1 ? "validation" : "test";
    std::ifstream f(conll / (s == 0 ? "eng.train" : s == 1 ? "eng.testa" : "eng.testb"));
    std::vector<std::vector<std::pair<std::string, std::string>>> sentences;
    std::vector<std::pair<std::string, std::string>>              sentence;
    while (std::getline(f, line) && int(sentences.size()) < rows) {
      if (line.empty()) {
        if (sentence.size() >= 2 && sentence.size() <= 24)
          sentences.push_back(sentence);
        sentence.clear();
        continue;
      }
      if (line.rfind("-DOCSTART-", 0) == 0)
        continue;
      std::istringstream row(line);
      std::string        w, pos, chunk, tag;
      if (row >> w >> pos >> chunk >> tag)
        sentence.push_back({w, tag});
    }
    Check(int(sentences.size()) == rows, "enough disjoint CoNLL sentences");
    std::vector<int> x(rows * 128), seg(x.size()), len(rows), labels(x.size()), ids(rows),
        qx(rows * 256), qs(qx.size()), ql(rows), qy(rows * 2);
    std::vector<float> w(x.size()), qw(qx.size());
    for (int r = 0; r < rows; ++r) {
      int         t = 1;
      std::string context;
      for (auto& [word, tag] : sentences[r]) {
        auto c = encode(word, 128);
        if (t + int(c.size()) + 1 >= 128)
          break;
        int label           = tag_ids.at(tag);
        w[r * 128 + t]      = 1;
        labels[r * 128 + t] = label;
        for (int code : c)
          x[r * 128 + t++] = code;
        if (t < 126)
          x[r * 128 + t++] = codes.count(' ') ? codes.at(' ') : 1;
        context += word + ' ';
      }
      x[r * 128]       = 2;
      x[r * 128 + t++] = 3;
      len[r]           = t;
      ids[r]           = s * 10000 + r;
      auto question = encode("find " + sentences[r][0].first, 80), c = encode(context, 170);
      int  pos            = 0;
      qx[r * 256 + pos++] = 2;
      for (int code : question)
        qx[r * 256 + pos++] = code;
      qx[r * 256 + pos++] = 3;
      int start           = pos;
      for (int code : c) {
        qx[r * 256 + pos] = code;
        qs[r * 256 + pos] = 1;
        qw[r * 256 + pos] = 1;
        ++pos;
      }
      qx[r * 256 + pos++] = 3;
      ql[r]               = pos;
      qy[r * 2]           = start;
      qy[r * 2 + 1]       = start + int(encode(sentences[r][0].first, 170).size()) - 1;
    }
    ner[p + "_input"]      = n::from_vector(x, {rows, 128});
    ner[p + "_segments"]   = n::from_vector(seg, {rows, 128});
    ner[p + "_lengths"]    = n::from_vector(len);
    ner[p + "_labels"]     = n::from_vector(labels, {rows, 128});
    ner[p + "_weights"]    = n::from_vector(w, {rows, 128});
    ner[p + "_source_ids"] = n::from_vector(ids);
    qa[p + "_input"]       = n::from_vector(qx, {rows, 256});
    qa[p + "_segments"]    = n::from_vector(qs, {rows, 256});
    qa[p + "_lengths"]     = n::from_vector(ql);
    qa[p + "_labels"]      = n::from_vector(qy, {rows, 2});
    qa[p + "_weights"]     = n::from_vector(qw, {rows, 256});
    qa[p + "_source_ids"]  = n::from_vector(ids);
  }
  Save(
      root / "bert_finetune/ner",
      ner,
      "Reconstructed first64/16/16 complete2..24-word sentences from authenticated original CoNLL "
      "train/testa/testb files;character encoding,weights on first character of each word;original "
      "IOB1 tags mapped to9 of13 output IDs. Not WordPiece or original missing NER preparation.");
  Save(
      root / "bert_finetune/qa",
      qa,
      "Reconstructed extractive lookup control from disjoint real CoNLL sentences;question find "
      "FIRSTWORD,context source sentence,answer first word span;context-only weights;64/16/16. "
      "This is a generated span-location task, not SQuAD or natural-question benchmark.");
  for (const auto& family :
       {"lora_adapters",
        "instruction_tuning",
        "sentence_embeddings",
        "t5_text/denoise",
        "t5_text/sentiment"}) {
    auto dir = root / family;
    std::ofstream(dir / "vocabulary.json")
        << Json{{"base", "65 sorted tinyShakespeare fit characters; task-specific special IDs"}, {"reconstructed", true}}
               .dump(2)
        << '\n';
  }
}
/// @brief Restore actual published Small BERT arrays from authenticated nonexecuting SafeTensors.
/// @param root Family directory.
void PublishedBert(const fs::path& root) {
  std::string base =
      "https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/"
      "30b0a37ccaaa32f332884b96992754e246e48c5f/";
  auto safe = Fetch(
      root,
      "model.safetensors",
      base + "model.safetensors",
      "7fb69ad9f6866d8983183c930e33828f326470bf6ad8bbb2ad4ed957a92e9414");
  Fetch(
      root,
      "vocab.txt",
      base + "vocab.txt",
      "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3");
  auto fields = n::load_safetensors(safe.string());
  A    weights;
  for (auto& [key, value] : fields) {
    if (key.rfind("cls.seq_relationship.", 0) == 0)
      continue;
    std::string name;
    if (key == "bert.embeddings.word_embeddings.weight")
      name = "word_embeddings/embeddings";
    else if (key == "bert.embeddings.position_embeddings.weight")
      name = "position_embedding/embeddings";
    else if (key == "bert.embeddings.token_type_embeddings.weight")
      name = "type_embeddings/embeddings";
    else if (key.rfind("bert.embeddings.LayerNorm.", 0) == 0)
      name = "embeddings/layer_norm/" + std::string(key.ends_with("weight") ? "gamma" : "beta");
    else if (key.rfind("bert.pooler.dense.", 0) == 0)
      name = "pooler_transform/" + std::string(key.ends_with("weight") ? "kernel" : "bias");
    else if (key.rfind("bert.encoder.layer.", 0) == 0) {
      int         layer  = key.at(19) - '0';
      std::string suffix = key.substr(21),
                  prefix = "transformer/layer_" + std::to_string(layer) + "/";
      if (suffix.rfind("attention.self.", 0) == 0) {
        auto part = suffix.substr(15);
        auto dot  = part.find('.');
        name      = prefix + "self_attention/" + part.substr(0, dot) + "/" +
               (part.ends_with("weight") ? "kernel" : "bias");
      } else if (suffix.rfind("attention.output.dense.", 0) == 0)
        name = prefix + "self_attention/attention_output/" +
               (suffix.ends_with("weight") ? "kernel" : "bias");
      else if (suffix.rfind("attention.output.LayerNorm.", 0) == 0)
        name =
            prefix + "self_attention_layer_norm/" + (suffix.ends_with("weight") ? "gamma" : "beta");
      else if (suffix.rfind("intermediate.dense.", 0) == 0)
        name = prefix + "intermediate/" + (suffix.ends_with("weight") ? "kernel" : "bias");
      else if (suffix.rfind("output.dense.", 0) == 0)
        name = prefix + "output/" + (suffix.ends_with("weight") ? "kernel" : "bias");
      else if (suffix.rfind("output.LayerNorm.", 0) == 0)
        name = prefix + "output_layer_norm/" + (suffix.ends_with("weight") ? "gamma" : "beta");
    } else
      name = "unused_mlm/" + key;
    Check(!name.empty(), "published BERT tensor mapping");
    bool transpose = key.ends_with(".weight") && key.find("embeddings") == std::string::npos &&
                     key.find("LayerNorm") == std::string::npos;
    if (transpose)
      value = n::transpose(value, {1, 0}).contiguous();
    weights[name] = value;
  }
  Check(weights.size() == 44, "published44array inventory");
  std::int64_t count = 0;
  for (auto& [k, v] : weights)
    count += v.numel();
  Check(count == 4433210, "published4433210parameters");
  n::save_npz((root / "pretrained_arrays.npz").string(), {weights.begin(), weights.end()});
  std::ofstream(root/"source_manifest.json")<<Json{{"source","Google published bert_uncased_L-2_H-128_A-2"},{"revision","30b0a37ccaaa32f332884b96992754e246e48c5f"},{"safetensors_sha256",file_digest(safe.string())},{"weights",44},{"parameters",count},{"mapping","Linear matrices transposed from out,in to in,out; two NSP tensors excluded, unused MLM transform/bias retained"},{"external_TF_recordings_available",false}}.dump(2)<<'\n';
}
} // namespace reconstructed
/// @brief Prepare the explicitly versioned missing NN recipes without executing external scripts.
/// @param root Owned nn-completion-assets cache directory; children use original family names.
void prepare_reconstructed_assets(const fs::path& root) {
  fs::create_directories(root);
  reconstructed::Vision(root / "vision_transformer");
  reconstructed::Forecast(root / "lstm_forecast");
  reconstructed::Sequence(root / "seq2seq", false, false);
  reconstructed::Sequence(root / "pointer_network", true, false);
  reconstructed::Sequence(root / "pointer_generator", false, true);
  reconstructed::Translation(root / "translation");
  reconstructed::Language(root);
  reconstructed::CtcMidi(root);
  reconstructed::Handwriting(root / "handwriting");
  reconstructed::TextTasks(root);
  reconstructed::PublishedBert(root / "pretrained_bert_serve");
  std::ofstream(root/"manifest.json")<<Json{{"recipe","datamunge-nn-reconstruction-v1"},{"original_recipes_recovered",false},{"families",{"vision_transformer","lstm_forecast","seq2seq","pointer_network","pointer_generator","translation","bert_mlm","roberta_tweaks","switch_transformer","transformer_xl","xlnet","ctc_handwriting","midi_lstm","handwriting","bert_finetune","lora_adapters","instruction_tuning","sentence_embeddings","t5_text"}},{"published_checkpoint","Google Small BERT real authenticated weights; separate from locally trained teaching models"}}.dump(2)<<'\n';
}
} // namespace datamunge::datasets::detail
#endif
