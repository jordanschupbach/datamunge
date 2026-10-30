/// @file download.cpp
/// @brief Pinned dataset sources, curl downloads with checksum checks, and gzip/tar unpacking.

#include <datamunge/datasets/download.hpp>

#include <datamunge/algorithms/md5.hpp>
#include <datamunge/image/detail/crc32.hpp>
#include <datamunge/image/detail/zlib_codec.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <spawn.h>
#include <sys/wait.h>

#include <cerrno>
/// @brief The process environment, passed on to the curl child process.
extern char** environ;
#endif

namespace datamunge::datasets {

/// @brief Download, staging and archive helpers behind fetch() and download_file().
namespace {

namespace fs = std::filesystem;

/// @brief One file of a downloadable dataset: where to get it and its MD5 checksum.
struct Remote {
  std::string url;  ///< Original download URL.
  std::string md5;  ///< Lowercase hex MD5 of the downloaded bytes.
};

/// @brief Pinned sources. Tabular CSVs come from fixed commits so their checksums are stable.
const Remote kDiamonds{
    "https://raw.githubusercontent.com/tidyverse/ggplot2/4c678917393c807090f8c1168ac8d6483390a529/"
    "data-raw/diamonds.csv",
    "4d3d1d4bbad5e0806dbaec425cf90196"};

/// @brief Géron's copy of the 1990 California census housing data, with ocean_proximity.
const Remote kCaliforniaHousing{
    "https://raw.githubusercontent.com/ageron/handson-ml2/aa555aa16b9d0af6c84e67c7757f3317d05acb12/"
    "datasets/housing/housing.csv",
    "d1c47305887e2252bf1ccbd74ff159a6"};

/// @brief Norvig's Sudoku collections (file name, MD5): 50 easy grids (from Project Euler
///        problem 96), 95 hard ones, and 11 of the hardest he found.
const std::vector<std::pair<std::string, std::string>> kSudokuFiles = {
    {"easy50.txt", "a0111712232d314f72683d55ccfcc092"},
    {"top95.txt", "3dd7cba557d55fc0df8cbf9817e06d15"},
    {"hardest.txt", "2d25727dd1d99775c1a5339cf64000ce"}};

/// @brief The four IDX files of MNIST-format datasets, in train-images, train-labels,
///        test-images, test-labels order.
const std::vector<std::string> kIdxFiles = {"train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                                            "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"};

/// @brief MD5 checksums of the gzipped MNIST files, in kIdxFiles order.
const std::vector<std::string> kMnistMd5 = {"f68b3c2dcbeaaa9fbdd348bbdeb94873", "d53e105ee54ea40749a09fcbcd1e9432",
                                            "9fb629c4189551a2d022fa330f9573f3", "ec29112dd5afa0611ce80d1b7f02629c"};

/// @brief MD5 checksums of the gzipped Fashion-MNIST files, in kIdxFiles order.
const std::vector<std::string> kFashionMd5 = {"8d4fb7e6c68d591d4c3dfef9ec88bf0d", "25c81989df183df01b3e8a0aad5dffbe",
                                              "bef4ecab320f06d8554ea6380940ec79", "bb300cfdad3c16e7a12a480ee83cd310"};

/// @brief Reads a whole file into memory.
std::string read_file(const fs::path& file) {
  std::ifstream input(file, std::ios::binary);
  if (!input) throw std::runtime_error("cannot read dataset file " + file.string());
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

/// @brief A private temporary directory, removed with its contents on destruction, so a
///        failed or interrupted preparation never leaves partial files at final paths.
struct Staging {
  fs::path path;  ///< The temporary directory.

  /// @brief Creates a fresh directory inside @p directory (created if needed).
  explicit Staging(const fs::path& directory) {
    fs::create_directories(directory);
    static std::atomic<unsigned long long> serial{0};
    for (int attempt = 0; attempt < 100; ++attempt) {
      path = directory / (".prepare-" +
                          std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
                          "-" + std::to_string(serial++));
      if (fs::create_directory(path)) return;
    }
    throw std::runtime_error("cannot create dataset staging directory in " + directory.string());
  }
  /// @brief Not copyable: each staging directory has exactly one owner.
  Staging(const Staging&) = delete;
  /// @brief Not copy-assignable; see the deleted copy constructor.
  Staging& operator=(const Staging&) = delete;
  /// @brief Removes the directory, ignoring errors.
  ~Staging() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

/// @brief Moves a finished file into place unless another process got there first.
void publish(const fs::path& from, const fs::path& to) {
  if (fs::is_regular_file(to)) return;
  std::error_code error;
  fs::rename(from, to, error);
  if (error && !fs::is_regular_file(to)) throw fs::filesystem_error("publish dataset", from, to, error);
}

/// @brief Throws if DATAMUNGE_DATA_OFFLINE=1, naming what is missing.
void require_online(const fs::path& missing) {
  const char* offline = std::getenv("DATAMUNGE_DATA_OFFLINE");
  if (offline && std::string(offline) == "1")
    throw std::runtime_error("dataset file " + missing.string() +
                             " is missing and downloads are disabled by DATAMUNGE_DATA_OFFLINE=1");
}

/// @brief Runs curl with an argument vector (never a shell) to save @p url as @p file.
void curl(const std::string& url, const fs::path& file) {
  std::vector<std::string> args{"curl", "--fail", "--location", "--silent", "--show-error", "--retry", "2",
                                "--connect-timeout", "20", "--max-time", "1800", "--proto", "=https,http,file",
                                "--proto-redir", "=https,http", "--output", file.string(), "--url", url};
  std::vector<char*> argv;
  for (auto& arg : args) argv.push_back(arg.data());
  argv.push_back(nullptr);
#ifdef _WIN32
  const auto status = _spawnvp(_P_WAIT, "curl", argv.data());
  if (status != 0) throw std::runtime_error("dataset download failed (curl exit " + std::to_string(status) + "): " + url);
#else
  pid_t pid;
  if (posix_spawnp(&pid, "curl", nullptr, nullptr, argv.data(), environ) != 0)
    throw std::runtime_error("cannot start curl to download datasets; install curl and retry");
  int status = 0;
  while (waitpid(pid, &status, 0) == -1)
    if (errno != EINTR) throw std::runtime_error("cannot wait for dataset download");
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    throw std::runtime_error("dataset download failed: " + url + " (check the network connection, or place the "
                             "file at " + file.filename().string() + " in the dataset directory yourself)");
#endif
}

/// @brief Downloads to @p file and verifies its checksum (empty @p md5 skips the check).
void download_verified(const std::string& url, const fs::path& file, const std::string& md5) {
  curl(url, file);
  if (!md5.empty() && algorithms::md5_hex(read_file(file)) != md5)
    throw std::runtime_error("checksum mismatch for " + url + "; the download was discarded");
}

/// @brief The URL to use for @p remote: DATAMUNGE_DATA_MIRROR/<dataset>/<file> if set.
std::string source_url(const std::string& dataset, const Remote& remote) {
  const char* mirror = std::getenv("DATAMUNGE_DATA_MIRROR");
  if (!mirror || !*mirror) return remote.url;
  std::string url = mirror;
  if (url.back() != '/') url += '/';
  return url + dataset + '/' + remote.url.substr(remote.url.find_last_of('/') + 1);
}

/// @brief Downloads one remote file to @p target (via staging) unless it already exists.
void ensure_file(const std::string& dataset, const Remote& remote, const fs::path& target) {
  if (fs::is_regular_file(target)) return;
  require_online(target);
  Staging stage(target.parent_path());
  const std::string url = source_url(dataset, remote);
  std::clog << "datamunge: downloading " << dataset << " from " << url << std::endl;
  download_verified(url, stage.path / target.filename(), remote.md5);
  publish(stage.path / target.filename(), target);
}

/// @brief Decompresses a single-member gzip file in memory, checking its CRC.
std::vector<unsigned char> gunzip(const fs::path& file) {
  const std::string bytes = read_file(file);
  const auto* b = reinterpret_cast<const unsigned char*>(bytes.data());
  const auto fail = [&] { return std::runtime_error("invalid gzip file " + file.string()); };
  if (bytes.size() < 18 || b[0] != 31 || b[1] != 139 || b[2] != 8 || (b[3] & 0xe0)) throw fail();
  const auto little = [&](std::size_t at) {
    return std::uint32_t(b[at]) | (std::uint32_t(b[at + 1]) << 8) | (std::uint32_t(b[at + 2]) << 16) |
           (std::uint32_t(b[at + 3]) << 24);
  };
  std::size_t at = 10;
  const std::size_t end = bytes.size() - 8;
  if (b[3] & 4) {
    if (at + 2 > end) throw fail();
    at += 2 + (b[at] | (std::size_t(b[at + 1]) << 8));
  }
  for (unsigned flag : {8u, 16u})
    if (b[3] & flag) {
      while (at < end && b[at]) ++at;
      ++at;
    }
  if (b[3] & 2) at += 2;
  // ISIZE is the size mod 2^32; every dataset here is far below 4 GiB.
  const auto size = little(end + 4);
  if (at > end) throw fail();
  auto output = image::detail::deflate_decompress(b + at, end - at, size);
  if (image::detail::crc32_update(0, output.data(), output.size()) != little(end)) throw fail();
  return output;
}

/// @brief Writes @p bytes to @p file.
void write_file(const fs::path& file, const unsigned char* bytes, std::size_t size) {
  std::ofstream out(file, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes), static_cast<std::streamsize>(size));
  if (!out) throw std::runtime_error("cannot write dataset file " + file.string());
}

/// @brief Extracts the named regular-file members of an in-memory tar archive into @p stage.
///        Member paths never become output paths, so archive entries cannot escape @p stage.
void extract_tar(const std::vector<unsigned char>& bytes, const fs::path& stage, const std::string& folder,
                 const std::vector<std::string>& files) {
  std::set<std::string> found;
  for (std::size_t at = 0; at + 512 <= bytes.size();) {
    const auto* header = bytes.data() + at;
    if (!header[0]) break;
    const std::string name(reinterpret_cast<const char*>(header), std::find(header, header + 100, 0) - header);
    std::size_t size = 0;
    for (int i = 124; i < 136; ++i) {
      if (header[i] == 0 || header[i] == ' ') continue;
      if (header[i] < '0' || header[i] > '7' || size > bytes.size() / 8) throw std::runtime_error("invalid tar size");
      size = size * 8 + (header[i] - '0');
    }
    at += 512;
    if (size > bytes.size() - at) throw std::runtime_error("truncated tar archive");
    for (const auto& file : files)
      if (name == folder + '/' + file) {
        if (header[156] != 0 && header[156] != '0') throw std::runtime_error("tar member is not a regular file");
        if (!found.insert(file).second) throw std::runtime_error("duplicate tar member " + name);
        write_file(stage / file, bytes.data() + at, size);
      }
    at += ((size + 511) / 512) * 512;
  }
  if (found.size() != files.size()) throw std::runtime_error("dataset archive is missing expected files");
}

/// @brief Ensures uncompressed MNIST-format IDX files exist in @p dir. A `.gz` file already
///        there (as the NN examples cache them) is decompressed instead of downloaded again.
void ensure_idx(const std::string& dataset, const std::string& base, const std::vector<std::string>& md5,
                const fs::path& dir) {
  for (std::size_t i = 0; i < kIdxFiles.size(); ++i) {
    const fs::path plain = dir / kIdxFiles[i];
    if (fs::is_regular_file(plain)) continue;
    const fs::path gz = dir / (kIdxFiles[i] + ".gz");
    ensure_file(dataset, Remote{base + kIdxFiles[i] + ".gz", md5[i]}, gz);
    Staging stage(dir);
    const auto bytes = gunzip(gz);
    write_file(stage.path / kIdxFiles[i], bytes.data(), bytes.size());
    publish(stage.path / kIdxFiles[i], plain);
  }
}

/// @brief Ensures the CIFAR binary batch files exist in @p dir, downloading and unpacking the
///        archive only if one of them is missing.
void ensure_cifar(bool cifar100, const fs::path& dir) {
  const std::vector<std::string> files =
      cifar100 ? std::vector<std::string>{"train.bin", "test.bin"}
               : std::vector<std::string>{"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
                                          "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin"};
  if (std::all_of(files.begin(), files.end(), [&](const auto& f) { return fs::is_regular_file(dir / f); })) return;
  const std::string dataset = cifar100 ? "cifar100" : "cifar10";
  const std::string archive = cifar100 ? "cifar-100-binary.tar.gz" : "cifar-10-binary.tar.gz";
  const Remote remote{"https://www.cs.toronto.edu/~kriz/" + archive,
                      cifar100 ? "03b5dce01913d631647c71ecec9e9cb8" : "c32a1d4ab5d03f1284b67883e8d87530"};
  require_online(dir / files.front());
  Staging stage(dir);
  // TensorFlow mirrors the identical CIFAR-10 archive; the author's server is often slow.
  const char* mirror = std::getenv("DATAMUNGE_DATA_MIRROR");
  const bool custom_mirror = mirror && *mirror;
  const std::string first =
      custom_mirror || cifar100
          ? source_url(dataset, remote)
          : "https://storage.googleapis.com/mirror.tensorflow.org/www.cs.toronto.edu/~kriz/" + archive;
  std::clog << "datamunge: downloading " << dataset << " (~160 MB) from " << first << std::endl;
  try {
    download_verified(first, stage.path / archive, remote.md5);
  } catch (const std::runtime_error&) {
    if (custom_mirror || cifar100) throw;
    std::clog << "datamunge: mirror failed, trying " << remote.url << std::endl;
    download_verified(remote.url, stage.path / archive, remote.md5);
  }
  extract_tar(gunzip(stage.path / archive), stage.path, cifar100 ? "cifar-100-binary" : "cifar-10-batches-bin", files);
  for (const auto& file : files) publish(stage.path / file, dir / file);
}

} // namespace

std::string data_home() {
  if (const char* root = std::getenv("DATAMUNGE_DATA_DIR"); root && *root) return root;
  if (const char* cache = std::getenv("XDG_CACHE_HOME"); cache && *cache)
    return (fs::path(cache) / "datamunge" / "datasets").string();
  if (const char* home = std::getenv("HOME"); home && *home)
    return (fs::path(home) / ".cache" / "datamunge" / "datasets").string();
#ifdef _WIN32
  if (const char* local = std::getenv("LOCALAPPDATA"); local && *local)
    return (fs::path(local) / "datamunge" / "datasets").string();
#endif
  return (fs::path("build") / "datasets").string();
}

std::vector<std::string> downloadable() {
  return {"california_housing", "cifar10", "cifar100", "diamonds", "fashion_mnist", "mnist", "sudoku"};
}

std::string fetch(const std::string& name, const std::string& directory) {
  const fs::path dir = directory.empty() ? fs::path(data_home()) / name : fs::path(directory);
  if (name == "diamonds" || name == "california_housing") {
    const fs::path file = dir / (name + ".csv");
    ensure_file(name, name == "diamonds" ? kDiamonds : kCaliforniaHousing, file);
    return file.string();
  }
  if (name == "sudoku") {
    for (const auto& [file, md5] : kSudokuFiles)
      ensure_file(name, Remote{"https://norvig.com/" + file, md5}, dir / file);
    return dir.string();
  }
  if (name == "mnist") {
    ensure_idx(name, "https://ossci-datasets.s3.amazonaws.com/mnist/", kMnistMd5, dir);
  } else if (name == "fashion_mnist") {
    ensure_idx(name, "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
               kFashionMd5, dir);
  } else if (name == "cifar10" || name == "cifar100") {
    ensure_cifar(name == "cifar100", dir);
  } else {
    throw std::invalid_argument("unknown downloadable dataset \"" + name +
                                "\"; expected california_housing, cifar10, cifar100, diamonds, fashion_mnist, mnist or sudoku");
  }
  return dir.string();
}

void download_file(const std::string& url, const std::string& destination, const std::string& md5) {
  const fs::path target(destination);
  if (fs::is_regular_file(target)) return;
  require_online(target);
  const fs::path parent = target.has_parent_path() ? target.parent_path() : fs::path(".");
  Staging stage(parent);
  download_verified(url, stage.path / target.filename(), md5);
  publish(stage.path / target.filename(), target);
}

} // namespace datamunge::datasets
