// Example dataset preparation. C++ owns downloads and derived data; SWIG ports
// call these same functions. The core nn::load_* functions remain local readers.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <datamunge/algorithms/md5.hpp>
#include <datamunge/image/detail/crc32.hpp>
#include <datamunge/image/detail/zlib_codec.hpp>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SWIG
#ifdef _WIN32
#include <process.h>
#else
#include <spawn.h>
#include <sys/wait.h>

#include <cerrno>
extern char** environ;
#endif
#endif

namespace dataset_assets {
#ifndef SWIG
namespace detail {
namespace fs = std::filesystem;
namespace nn = datamunge::nn;

inline bool Present(const fs::path& file) {
  return fs::is_regular_file(file);
}

inline std::string Read(const fs::path& file) {
  std::ifstream input(file, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot read dataset file " + file.string());
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

// Each invocation has its own staging directory. A failed process cannot leave
// a partial file at the final path or block later attempts with a stale lock.
struct Staging {
  fs::path path;
  explicit Staging(const fs::path& directory) {
    fs::create_directories(directory);
    static std::atomic<unsigned long long> serial{0};
    for (int attempt = 0; attempt < 100; ++attempt) {
      path = directory /
             (".prepare-" +
              std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
              "-" + std::to_string(serial++));
      if (fs::create_directory(path))
        return;
    }
    throw std::runtime_error("cannot create dataset staging directory in " + directory.string());
  }
  ~Staging() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

inline void Publish(const fs::path& from, const fs::path& to) {
  // Preserve supplied assets, including small caller-owned test datasets.
  if (Present(to))
    return;
  std::error_code error;
  fs::rename(from, to, error);
  if (error && !Present(to))
    throw fs::filesystem_error("publish dataset", from, to, error);
}

inline void Online(const fs::path& directory) {
  const char* offline = std::getenv("DATAMUNGE_DATA_OFFLINE");
  if (offline && std::string(offline) == "1")
    throw std::runtime_error(
        "dataset files missing in " + directory.string() +
        "; automatic download disabled by DATAMUNGE_DATA_OFFLINE=1");
}

// Pass an argument vector directly to curl: data= paths never enter a shell.
// curl and its CA certificates are included in the NN development shells.
inline void Curl(const std::string& url, const fs::path& file) {
  std::vector<std::string> args{
      "curl",
      "--fail",
      "--location",
      "--silent",
      "--show-error",
      "--retry",
      "2",
      "--connect-timeout",
      "20",
      "--max-time",
      "600",
      "--proto",
      "=https,http,file",
      "--proto-redir",
      "=https,http",
      "--output",
      file.string(),
      "--url",
      url};
  std::vector<char*> argv;
  for (auto& arg : args)
    argv.push_back(arg.data());
  argv.push_back(nullptr);
#ifdef _WIN32
  const auto status = _spawnvp(_P_WAIT, "curl", argv.data());
  if (status != 0)
    throw std::runtime_error(
        "dataset download failed (curl exit " + std::to_string(status) + "): " + url);
#else
  pid_t     pid;
  const int started = posix_spawnp(&pid, "curl", nullptr, nullptr, argv.data(), environ);
  if (started)
    throw std::runtime_error(
        "cannot start curl for automatic dataset download; install curl or use the Nix dev shell");
  int status = 0;
  while (waitpid(pid, &status, 0) == -1) {
    if (errno != EINTR)
      throw std::runtime_error("cannot wait for dataset download");
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    throw std::runtime_error(
        "dataset download failed: " + url + "; retry with a network connection or populate " +
        file.parent_path().parent_path().string());
#endif
}

inline void Fetch(
    const std::string& dataset,
    const std::string& url,
    const fs::path&    file,
    const std::string& md5) {
  std::string source = url;
  // Optional mirror retains dataset/filename layout and the original checksum.
  if (const char* mirror = std::getenv("DATAMUNGE_DATA_MIRROR"); mirror && *mirror) {
    source = std::string(mirror);
    if (source.back() != '/')
      source += '/';
    source += dataset + '/' + url.substr(url.find_last_of('/') + 1);
  }
  std::cout << "Preparing " << dataset << ": downloading " << source << std::endl;
  Curl(source, file);
  if (datamunge::algorithms::md5_hex(Read(file)) != md5)
    throw std::runtime_error(
        "dataset checksum mismatch for " + source +
        "; incomplete/corrupt download was not installed");
}

inline std::vector<unsigned char> Gunzip(const fs::path& file) {
  const std::string bytes = Read(file);
  const auto*       b     = reinterpret_cast<const unsigned char*>(bytes.data());
  const auto        fail  = [] { return std::runtime_error("invalid dataset gzip archive"); };
  if (bytes.size() < 18 || b[0] != 31 || b[1] != 139 || b[2] != 8 || (b[3] & 0xe0))
    throw fail();
  const auto little = [&](std::size_t at) {
    return std::uint32_t(b[at]) | (std::uint32_t(b[at + 1]) << 8) |
           (std::uint32_t(b[at + 2]) << 16) | (std::uint32_t(b[at + 3]) << 24);
  };
  std::size_t at = 10, end = bytes.size() - 8;
  if (b[3] & 4) {
    if (at + 2 > end)
      throw fail();
    const std::size_t extra = b[at] | (std::size_t(b[at + 1]) << 8);
    at += 2 + extra;
  }
  for (unsigned flag : {8u, 16u})
    if (b[3] & flag) {
      while (at < end && b[at])
        ++at;
      ++at;
    }
  if (b[3] & 2)
    at += 2;
  const auto size = little(end + 4);
  if (at > end || size > 512u * 1024u * 1024u)
    throw fail();
  auto output = datamunge::image::detail::deflate_decompress(b + at, end - at, size);
  if (datamunge::image::detail::crc32_update(0, output.data(), output.size()) != little(end))
    throw fail();
  return output;
}

// Extract only explicitly allowed regular files; archive paths never become
// output paths. Links, traversal entries and unrequested files cannot escape.
inline void ExtractCifar(
    const fs::path&                 archive,
    const fs::path&                 stage,
    const std::string&              folder,
    const std::vector<std::string>& files) {
  auto                  bytes = Gunzip(archive);
  std::set<std::string> found;
  for (std::size_t at = 0; at + 512 <= bytes.size();) {
    const auto* header = bytes.data() + at;
    if (!header[0])
      break;
    std::string name(
        reinterpret_cast<const char*>(header), std::find(header, header + 100, 0) - header);
    std::size_t size = 0;
    for (int i = 124; i < 136; ++i) {
      if (header[i] == 0 || header[i] == ' ')
        continue;
      if (header[i] < '0' || header[i] > '7' || size > bytes.size() / 8)
        throw std::runtime_error("invalid dataset tar size");
      size = size * 8 + (header[i] - '0');
    }
    at += 512;
    if (size > bytes.size() - at)
      throw std::runtime_error("truncated dataset tar archive");
    for (const auto& file : files)
      if (name == folder + '/' + file) {
        if (header[156] != 0 && header[156] != '0')
          throw std::runtime_error("dataset archive member is not a regular file");
        if (!found.insert(file).second)
          throw std::runtime_error("duplicate dataset archive member");
        const std::size_t record = folder == "cifar-10-batches-bin" ? 3073 : 3074;
        const std::size_t rows   = file == "test_batch.bin" || file == "test.bin"
                                       ? 10000
                                       : (file == "train.bin" ? 50000 : 10000);
        if (size != record * rows)
          throw std::runtime_error("unexpected CIFAR dataset file dimensions");
        std::ofstream out(stage / file, std::ios::binary);
        out.write(
            reinterpret_cast<const char*>(bytes.data() + at), static_cast<std::streamsize>(size));
        if (!out)
          throw std::runtime_error("cannot write extracted dataset file");
      }
    at += ((size + 511) / 512) * 512;
  }
  if (found.size() != files.size())
    throw std::runtime_error("dataset archive is missing required members");
}
} // namespace detail
#endif

/// Shared default, independent of another project's checkout. Paths are resolved
/// without side effects; data=PATH overrides are prepared only when actually read.
inline std::string DataRoot() {
  if (const char* root = std::getenv("DATAMUNGE_DATA_DIR"); root && *root)
    return root;
  if (const char* cache = std::getenv("XDG_CACHE_HOME"); cache && *cache)
    return (std::filesystem::path(cache) / "datamunge" / "datasets").string();
  if (const char* home = std::getenv("HOME"); home && *home)
    return (std::filesystem::path(home) / ".cache" / "datamunge" / "datasets").string();
#ifdef _WIN32
  if (const char* home = std::getenv("LOCALAPPDATA"); home && *home)
    return (std::filesystem::path(home) / "datamunge" / "datasets").string();
#endif
  return (std::filesystem::path("build") / "datasets").string();
}

/// Download only absent MNIST/Fashion-MNIST files. Existing plain or gzip IDX
/// files are retained. Downloaded archives are verified before atomic publish.
inline void
EnsureMnist(const std::string& directory, bool fashion = false, const std::string& split = "") {
  namespace fs           = std::filesystem;
  const std::string name = fashion ? "fashion_mnist" : "mnist";
  if (!split.empty() && split != "train" && split != "test")
    throw std::invalid_argument("MNIST split must be train or test");
  const std::string base =
      fashion
          ? "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
          : "https://ossci-datasets.s3.amazonaws.com/mnist/";
  const std::vector<std::pair<std::string, std::string>> files = fashion
    ? std::vector<std::pair<std::string,std::string>>{{"train-images-idx3-ubyte", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"},
      {"train-labels-idx1-ubyte", "25c81989df183df01b3e8a0aad5dffbe"}, {"t10k-images-idx3-ubyte", "bef4ecab320f06d8554ea6380940ec79"},
      {"t10k-labels-idx1-ubyte", "bb300cfdad3c16e7a12a480ee83cd310"}}
    : std::vector<std::pair<std::string,std::string>>{{"train-images-idx3-ubyte", "f68b3c2dcbeaaa9fbdd348bbdeb94873"},
      {"train-labels-idx1-ubyte", "d53e105ee54ea40749a09fcbcd1e9432"}, {"t10k-images-idx3-ubyte", "9fb629c4189551a2d022fa330f9573f3"},
      {"t10k-labels-idx1-ubyte", "ec29112dd5afa0611ce80d1b7f02629c"}};
  for (const auto& [file, hash] : files) {
    if (!split.empty() && file.rfind(split == "train" ? "train-" : "t10k-", 0) != 0)
      continue;
    if (detail::Present(fs::path(directory) / file) ||
        detail::Present(fs::path(directory) / (file + ".gz")))
      continue;
    detail::Online(directory);
    detail::Staging stage(directory);
    detail::Fetch(name, base + file + ".gz", stage.path / file, hash);
    // Keep the verified gzip; nn::read_idx detects its magic bytes.
    detail::Publish(stage.path / file, fs::path(directory) / (file + ".gz"));
  }
}

inline void
EnsureCifar(const std::string& directory, bool cifar100 = false, const std::string& split = "") {
  namespace fs                         = std::filesystem;
  const std::vector<std::string> files = cifar100
                                             ? std::vector<std::string>{"train.bin", "test.bin"}
                                             : std::vector<std::string>{
                                                   "data_batch_1.bin",
                                                   "data_batch_2.bin",
                                                   "data_batch_3.bin",
                                                   "data_batch_4.bin",
                                                   "data_batch_5.bin",
                                                   "test_batch.bin"};
  if (!split.empty() && split != "train" && split != "test")
    throw std::invalid_argument("CIFAR split must be train or test");
  if (std::all_of(files.begin(), files.end(), [&](const auto& file) {
        const bool test = file == "test.bin" || file == "test_batch.bin";
        return (!split.empty() && test != (split == "test")) ||
               detail::Present(fs::path(directory) / file);
      }))
    return;
  detail::Online(directory);
  detail::Staging   stage(directory);
  const std::string name     = cifar100 ? "cifar100" : "cifar10";
  const std::string archive  = cifar100 ? "cifar-100-binary.tar.gz" : "cifar-10-binary.tar.gz";
  const std::string original = "https://www.cs.toronto.edu/~kriz/" + archive;
  const std::string hash =
      cifar100 ? "03b5dce01913d631647c71ecec9e9cb8" : "c32a1d4ab5d03f1284b67883e8d87530";
  // TensorFlow mirrors the identical CIFAR-10 archive. The author's server
  // can be too slow for an interactive example's first run.
  const std::string source =
      cifar100 ? original
               : "https://storage.googleapis.com/mirror.tensorflow.org/www.cs.toronto.edu/~kriz/" +
                     archive;
  try {
    detail::Fetch(name, source, stage.path / "archive.tar.gz", hash);
  } catch (const std::exception&) {
    const char* mirror = std::getenv("DATAMUNGE_DATA_MIRROR");
    if (cifar100 || (mirror && *mirror))
      throw;
    std::cout << "CIFAR mirror unavailable; trying the dataset author's server" << std::endl;
    detail::Fetch(name, original, stage.path / "archive.tar.gz", hash);
  }
  detail::ExtractCifar(
      stage.path / "archive.tar.gz",
      stage.path,
      cifar100 ? "cifar-100-binary" : "cifar-10-batches-bin",
      files);
  for (const auto& file : files)
    detail::Publish(stage.path / file, fs::path(directory) / file);
}

/// Balanced, disjoint real MNIST pools for the autoencoder examples. Selection
/// uses std::mt19937(1931), independently of external repositories' selections.
inline void EnsureSaliency(const std::string& directory) {
  namespace fs = std::filesystem;
  namespace nn = datamunge::nn;
  if (detail::Present(fs::path(directory) / "saliency_data.npz"))
    return;
  const auto mnist = fs::path(directory).parent_path() / "mnist";
  EnsureMnist(mnist.string());
  const auto train = nn::load_mnist(mnist.string(), "train"),
             test  = nn::load_mnist(mnist.string(), "test");
  if (train.images.shape() != nn::Shape{60000, 28, 28, 1} ||
      test.images.shape() != nn::Shape{10000, 28, 28, 1})
    throw std::runtime_error("official MNIST dimensions required to prepare autoencoder pools");
  // read_idx retains exact original uint8 pixels (represented as int32).
  const auto locate = [&](const std::string& file) {
    return detail::Present(mnist / file) ? mnist / file : mnist / (file + ".gz");
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
  detail::Staging stage(directory);
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
  detail::Publish(stage.path / "manifest.json", fs::path(directory) / "manifest.json");
  detail::Publish(stage.path / "saliency_data.npz", fs::path(directory) / "saliency_data.npz");
}

inline datamunge::nn::ImageDataset LoadMnist(
    const std::string& directory,
    const std::string& split) {
  if (split != "train" && split != "test")
    throw std::invalid_argument("MNIST split must be train or test");
  EnsureMnist(directory, false, split);
  return datamunge::nn::load_mnist(directory, split);
}
inline datamunge::nn::ImageDataset LoadCifar10(
    const std::string& directory,
    const std::string& split) {
  if (split != "train" && split != "test")
    throw std::invalid_argument("CIFAR split must be train or test");
  EnsureCifar(directory, false, split);
  return datamunge::nn::load_cifar10(directory, split);
}
inline datamunge::nn::ImageDataset
LoadCifar100(const std::string& directory, const std::string& split, bool fine = true) {
  if (split != "train" && split != "test")
    throw std::invalid_argument("CIFAR split must be train or test");
  EnsureCifar(directory, true, split);
  return datamunge::nn::load_cifar100(directory, split, fine);
}
} // namespace dataset_assets
