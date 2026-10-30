/// @file basic.cpp
/// @brief Existing deterministic local memory and MNIST source selections in the canonical library.
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include <datamunge/datasets/download.hpp>
#include <datamunge/nn/nn.hpp>
#include <map>
#include <random>
#include <set>

#include "support.hpp"
namespace datamunge::datasets::detail {
namespace nn = datamunge::nn;
/// @brief Generate deterministic fitting/validation/test copy or QA tensor records.
/// @param copy_task True selects binary copy, false selects fact/query answering.
/// @return All split observations, targets, lengths and source IDs.
static nn::NamedTensors MakeMemory(bool copy_task) {
  std::mt19937     generator(copy_task ? 45101 : 45301);
  nn::NamedTensors records;
  for (int split = 0; split < 3; ++split) {
    const std::string prefix = split == 0 ? "fit_" : split == 1 ? "validation_" : "test_";
    const int         count  = split == 0 ? 128 : 32;
    std::vector<int>  lengths(count), ids(count), facts(count * 12), queries(count), answers(count);
    std::vector<float> input(count * 36), target(count * 27);
    for (int row = 0; row < count; ++row) {
      ids[row]         = split * 10000 + row;
      const int length = 1 + generator() % (copy_task ? 4 : 6);
      lengths[row]     = length;
      if (copy_task) {
        for (int step = 0; step < length; ++step)
          for (int bit = 0; bit < 3; ++bit) {
            const float value                               = float(generator() & 1);
            input[(row * 9 + step) * 4 + bit]               = value;
            target[(row * 9 + length + 1 + step) * 3 + bit] = value;
          }
        input[(row * 9 + length) * 4 + 3] = 1;
      } else {
        int entities[7] = {1, 2, 3, 4, 5, 6, 7};
        for (int i = 6; i > 0; --i)
          std::swap(entities[i], entities[generator() % (i + 1)]);
        const int selected = generator() % length;
        for (int slot = 0; slot < length; ++slot) {
          const int answer                = generator() % 4;
          facts[(row * 6 + slot) * 2]     = entities[slot];
          facts[(row * 6 + slot) * 2 + 1] = 8 + answer;
          if (slot == selected) {
            queries[row] = entities[slot];
            answers[row] = answer;
          }
        }
      }
    }
    if (copy_task) {
      records.emplace_back(prefix + "input", nn::from_vector(input, {count, 9, 4}));
      records.emplace_back(prefix + "target", nn::from_vector(target, {count, 9, 3}));
    } else {
      records.emplace_back(prefix + "facts", nn::from_vector(facts, {count, 6, 2}));
      records.emplace_back(prefix + "query", nn::from_vector(queries, {count}));
      records.emplace_back(prefix + "target", nn::from_vector(answers, {count}));
    }
    records.emplace_back(prefix + "lengths", nn::from_vector(lengths, {count}));
    records.emplace_back(prefix + "source_ids", nn::from_vector(ids, {count}));
  }
  return records;
}

/// @brief Select balanced, byte-distinct original MNIST pools using seed1931.
/// @param directory Saliency cache with the original MNIST cache as its sibling.
static void Saliency(const std::string& directory) {
  namespace fs = std::filesystem;
  namespace nn = datamunge::nn;
  if (fs::is_regular_file(fs::path(directory) / "saliency_data.npz")) {
    if (!fs::is_regular_file(fs::path(directory) / "manifest.json"))
      save(fs::path(directory) / "manifest.json",
           {{"source", "caller-supplied saliency_data.npz"}, {"generation_verified", false}});
    return;
  }
  const auto mnist = fs::path(directory).parent_path() / "mnist";
  datamunge::datasets::fetch("mnist", mnist.string());
  const auto train = nn::load_mnist(mnist.string(), "train"),
             test  = nn::load_mnist(mnist.string(), "test");
  if (train.images.shape() != nn::Shape{60000, 28, 28, 1} ||
      test.images.shape() != nn::Shape{10000, 28, 28, 1})
    throw std::runtime_error("official MNIST dimensions required to prepare autoencoder pools");
  // read_idx retains exact original uint8 pixels (represented as int32).
  const auto locate = [&](const std::string& file) {
    return fs::is_regular_file(mnist / file) ? mnist / file : mnist / (file + ".gz");
  };
  const auto training =
      nn::reshape(nn::read_idx(locate("train-images-idx3-ubyte").string()), {60000, 28, 28, 1});
  const auto testing =
      nn::reshape(nn::read_idx(locate("t10k-images-idx3-ubyte").string()), {10000, 28, 28, 1});
  const auto   labels = train.labels.to_vector<int>(), test_labels = test.labels.to_vector<int>();
  const auto   pixels = training.to_vector<int>(), test_pixels = testing.to_vector<int>();
  std::mt19937 random(1931);
  std::set<std::string>                            seen;
  std::map<std::string, std::vector<std::int64_t>> rows;
  for (int digit = 0; digit < 10; ++digit) {
    for (bool held_out : {false, true}) {
      const auto&               y = held_out ? test_labels : labels;
      const auto&               x = held_out ? test_pixels : pixels;
      std::vector<std::int64_t> candidates;
      for (std::size_t i = 0; i < y.size(); ++i)
        if (y[i] == digit)
          candidates.push_back(i);
      std::shuffle(candidates.begin(), candidates.end(), random);
      std::size_t at = 0;
      for (const auto& [split, count] :
           (held_out
                ? std::vector<std::pair<std::string, int>>{{"test", 20}}
                : std::vector<std::pair<std::string, int>>{{"fit", 100}, {"validation", 20}})) {
        for (int i = 0; i < count; ++i) {
          bool added = false;
          while (at < candidates.size()) {
            const auto  row = candidates[at++];
            std::string bytes(784, '\0');
            for (int p = 0; p < 784; ++p)
              bytes[p] = static_cast<char>(x[row * 784 + p]);
            if (seen.insert(bytes).second) {
              rows[split].push_back(row);
              added = true;
              break;
            }
          }
          if (!added)
            throw std::runtime_error("insufficient distinct MNIST images");
        }
      }
    }
  }
  nn::NamedTensors records;
  for (const std::string split : {"fit", "validation", "test"}) {
    const auto& selected = rows[split];
    auto        ids      = selected;
    if (split == "test")
      for (auto& id : ids)
        id += 60000;
    records.emplace_back(
        split + "_images", nn::take_rows(split == "test" ? testing : training, selected));
    records.emplace_back(
        split + "_labels",
        nn::cast(
            nn::take_rows(split == "test" ? test.labels : train.labels, selected),
            nn::DType::Int32));
    records.emplace_back(split + "_source_ids", nn::cast(nn::from_vector(ids), nn::DType::Int32));
  }
  fs::create_directories(directory);
  Temporary stage(directory);
  nn::save_npz((stage.path / "saliency_data.npz").string(), records);
  std::ofstream manifest(stage.path / "manifest.json");
  manifest << "{\"dataset\":\"MNIST\",\"generator\":\"datamunge C++ balanced distinct images "
              "v1\",\"seed\":1931,"
              "\"selection\":\"std::mt19937, independent local "
              "selection\",\"test_source_offset\":60000,\"source_ids\":{";
  bool first = true;
  for (const std::string split : {"fit", "validation", "test"}) {
    if (!first)
      manifest << ',';
    first = false;
    manifest << '"' << split << "\":[";
    for (std::size_t i = 0; i < rows[split].size(); ++i) {
      if (i)
        manifest << ',';
      manifest << rows[split][i] + (split == "test" ? 60000 : 0);
    }
    manifest << ']';
  }
  manifest << "}}\n";
  manifest.close();
  if (!manifest)
    throw std::runtime_error("cannot write dataset manifest");
  fs::rename(stage.path / "manifest.json", fs::path(directory) / "manifest.json");
  fs::rename(stage.path / "saliency_data.npz", fs::path(directory) / "saliency_data.npz");
}

/// @brief Prepare local copy/QA tasks or original MNIST plus its balanced teaching pools.
/// @param name "memory" or "mnist-assets".
/// @param root Parent dataset cache.
void prepare_basic(const std::string& name, const fs::path& root) {
  if (name == "mnist-assets") {
    datamunge::datasets::fetch("mnist", (root / "mnist").string());
    Saliency((root / "saliency").string());
    return;
  }
  for (bool copy : {false, true}) {
    const auto dir = root / (copy ? "ntm" : "memnet");
    fs::create_directories(dir);
    Temporary stage(dir);
    if (!fs::is_regular_file(dir / "data.npz")) {
      nn::save_npz((stage.path / "data.npz").string(), MakeMemory(copy));
      fs::rename(stage.path / "data.npz", dir / "data.npz");
      save(
          dir / "manifest.json",
          {{"generator", "datamunge local C++ memory task v1"},
           {"external_selection", false},
           {"seed", copy ? 45101 : 45301},
           {"fit", 128},
           {"validation", 32},
           {"test", 32},
           {"task",
            copy ? "binary copy with delimiter and recall-only targets"
                 : "unambiguous fact/query question answering"}});
    } else if (!fs::is_regular_file(dir / "manifest.json")) {
      save(
          dir / "manifest.json",
          {{"source", "caller-supplied data.npz"}, {"generation_verified", false}});
    }
  }
}
} // namespace datamunge::datasets::detail

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
