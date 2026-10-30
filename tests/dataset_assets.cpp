/// @file dataset_assets.cpp
/// @brief Offline native transfer, archive, NPZ and deterministic preparation regression tests.
#include <gtest/gtest.h>

#include <atomic>
#include <datamunge/image/detail/crc32.hpp>
#include <dm/dm.hpp>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <thread>

#include "../src/datamunge/datasets/preparation/support.hpp"
namespace {
namespace fs = std::filesystem;
namespace d  = dm::datasets;
namespace n  = dm::nn;
struct Workspace : testing::Test {
  fs::path root;
  void     SetUp() override {
    static std::atomic<unsigned> serial{0};
    root = fs::temp_directory_path() /
           ("dm-data-test-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
            std::to_string(serial++));
    fs::create_directories(root);
  }
  void TearDown() override {
    std::error_code ec;
    fs::remove_all(root, ec);
  }
  void write(const fs::path& p, const std::string& s) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f << s;
    ASSERT_TRUE(f);
  }
  std::string read(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
  }
  std::string zip(const std::string& name, const std::string& data) {
    std::string out;
    const auto  u16 = [&](unsigned n) {
      out += char(n);
      out += char(n >> 8);
    };
    const auto u32 = [&](std::uint32_t n) {
      u16(n & 65535);
      u16(n >> 16);
    };
    const auto crc = dm::image::detail::crc32_update(
        0, reinterpret_cast<const unsigned char*>(data.data()), data.size());
    u32(0x04034b50);
    u16(20);
    u16(0);
    u16(0);
    u16(0);
    u16(0);
    u32(crc);
    u32(data.size());
    u32(data.size());
    u16(name.size());
    u16(0);
    out += name;
    out += data;
    const auto offset = out.size();
    u32(0x02014b50);
    u16(20);
    u16(20);
    u16(0);
    u16(0);
    u16(0);
    u16(0);
    u32(crc);
    u32(data.size());
    u32(data.size());
    u16(name.size());
    u16(0);
    u16(0);
    u16(0);
    u16(0);
    u32(0);
    u32(0);
    out += name;
    const auto size = out.size() - offset;
    u32(0x06054b50);
    u16(0);
    u16(0);
    u16(1);
    u16(1);
    u32(size);
    u32(offset);
    u16(0);
    return out;
  }
};
TEST_F(Workspace, StreamingHashesAndVerifiedFileTransfer) {
  write(root / "original.bin", "abc");
  const auto sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
  EXPECT_EQ(d::file_digest((root / "original.bin").string()), sha);
  EXPECT_EQ(
      d::file_digest((root / "original.bin").string(), "md5"), "900150983cd24fb0d6963f7d28e17f72");
  EXPECT_THROW(d::file_digest((root / "original.bin").string(), "sha1"), std::invalid_argument);
  d::fetch_file("file://" + (root / "original.bin").string(), (root / "copy.bin").string(), sha, 3);
  EXPECT_EQ(read(root / "copy.bin"), "abc");
  write(root / "copy.bin", "xyz");
  EXPECT_THROW(
      d::fetch_file(
          "file://" + (root / "original.bin").string(), (root / "copy.bin").string(), sha, 3),
      std::runtime_error);
  EXPECT_EQ(read(root / "copy.bin"), "xyz");
  EXPECT_THROW(
      d::fetch_file(
          "file://" + (root / "original.bin").string(),
          (root / "bad.bin").string(),
          std::string(64, '0')),
      std::runtime_error);
  EXPECT_FALSE(fs::exists(root / "bad.bin"));
}
// Environment changes are scoped to these serial tests, never to callers.
struct Environment {
  std::string name, previous;
  bool        present;
  Environment(std::string key, const std::string& value) : name(std::move(key)) {
    const char* old = std::getenv(name.c_str());
    present         = old != nullptr;
    if (old)
      previous = old;
    set(value);
  }
  void set(const std::string& value) {
#ifdef _WIN32
    _putenv_s(name.c_str(), value.c_str());
#else
    setenv(name.c_str(), value.c_str(), 1);
#endif
  }
  ~Environment() {
    if (present)
      set(previous);
    else {
#ifdef _WIN32
      _putenv_s(name.c_str(), "");
#else
      unsetenv(name.c_str());
#endif
    }
  }
};
TEST_F(Workspace, OfflineCacheHitAndMissNeverLaunchDownload) {
  write(root / "cached.bin", "abc");
  Environment       offline("DATAMUNGE_DATA_OFFLINE", "1");
  const std::string digest = "900150983cd24fb0d6963f7d28e17f72";
  EXPECT_NO_THROW(
      d::fetch_file("https://invalid.example/cached", (root / "cached.bin").string(), digest));
  EXPECT_THROW(
      d::fetch_file("https://invalid.example/missing", (root / "missing.bin").string(), digest),
      std::runtime_error);
  EXPECT_FALSE(fs::exists(root / "missing.bin"));
}
TEST_F(Workspace, FileMirrorPreservesChecksumsAndHandlesDestinationSpaces) {
  write(root / "mirror/pinned/data.bin", "abc");
  Environment mirror("DATAMUNGE_DATA_MIRROR", "file://" + (root / "mirror").string());
  Environment online("DATAMUNGE_DATA_OFFLINE", "0");
  const auto  target = root / "directory with spaces/data.bin";
  d::fetch_file(
      "https://invalid.example/data.bin",
      target.string(),
      "900150983cd24fb0d6963f7d28e17f72",
      3,
      "pinned");
  EXPECT_EQ(read(target), "abc");
}
TEST_F(Workspace, BoundedArchiveExtractionAndMissingMembers) {
  write(root / "source.zip", zip("folder/data.txt", "abc"));
  d::extract_archive(
      (root / "source.zip").string(), (root / "out").string(), {"folder/data.txt"}, 3);
  EXPECT_EQ(read(root / "out/folder/data.txt"), "abc");
  EXPECT_THROW(
      d::extract_archive((root / "source.zip").string(), (root / "limited").string(), {}, 2),
      std::runtime_error);
  EXPECT_FALSE(fs::exists(root / "limited/folder/data.txt"));
  EXPECT_THROW(
      d::extract_archive((root / "source.zip").string(), (root / "missing").string(), {"absent"}),
      std::runtime_error);
  EXPECT_THROW(
      d::extract_archive(
          (root / "source.zip").string(),
          (root / "duplicate").string(),
          {"folder/data.txt", "folder/data.txt"}),
      std::invalid_argument);
}
TEST_F(Workspace, RejectArchiveTraversalSymlinksAndCrcDamage) {
  for (const auto& name : {"../escaped", "/absolute", "C:/drive", "..\\escaped"}) {
    write(root / "unsafe.zip", zip(name, "bad"));
    EXPECT_THROW(
        d::extract_archive((root / "unsafe.zip").string(), (root / "out").string()),
        std::invalid_argument);
  }
  write(root / "source.zip", zip("folder/data.txt", "abc"));
  fs::create_directories(root / "outside");
  fs::create_directories(root / "out");
  fs::create_directory_symlink(root / "outside", root / "out/folder");
  EXPECT_THROW(
      d::extract_archive((root / "source.zip").string(), (root / "out").string()),
      std::invalid_argument);
  EXPECT_FALSE(fs::exists(root / "outside/data.txt"));
  auto damaged    = zip("file.txt", "abc");
  damaged[30 + 8] = 'z';
  write(root / "bad.zip", damaged);
  EXPECT_THROW(
      d::extract_archive((root / "bad.zip").string(), (root / "bad-out").string()),
      std::runtime_error);
  EXPECT_FALSE(fs::exists(root / "bad-out/file.txt"));
}
TEST_F(Workspace, NpzIngressRequiresNoNumpyOrPython) {
  const auto expected = n::from_vector(std::vector<float>{1, 2, 3, 4}, {2, 2});
  n::save_npz((root / "arrays.npz").string(), {{"nested/value", expected}});
  const auto arrays = n::load_npz((root / "arrays.npz").string());
  ASSERT_EQ(arrays.size(), 1u);
  EXPECT_EQ(arrays[0].second.to_vector<float>(), expected.to_vector<float>());
  d::extract_archive(
      (root / "arrays.npz").string(), (root / "expanded").string(), {"nested/value.npy"});
  EXPECT_EQ(
      n::load_npy((root / "expanded/nested/value.npy").string()).to_vector<float>(),
      expected.to_vector<float>());
}
TEST_F(Workspace, SeededMetadataMatchesOriginalPublishedSelections) {
  std::vector<int> values(12);
  std::iota(values.begin(), values.end(), 0);
  d::detail::MetadataRandom(271).shuffle(values);
  EXPECT_EQ(values, (std::vector<int>{1, 6, 0, 3, 8, 4, 11, 2, 9, 5, 10, 7}));
}
TEST_F(Workspace, CallerSuppliedMemoryArraysKeepTheirBytesAndProvenance) {
  fs::create_directories(root / "ntm");
  n::save_npz((root / "ntm/data.npz").string(), {{"custom", n::tensor({1., 2.})}});
  const auto supplied = read(root / "ntm/data.npz");
  d::prepare_example_data("memory", root.string());
  EXPECT_EQ(read(root / "ntm/data.npz"), supplied);
  EXPECT_NE(read(root / "ntm/manifest.json").find("caller-supplied"), std::string::npos);
  EXPECT_EQ(read(root / "ntm/manifest.json").find("generator"), std::string::npos);
}
TEST_F(Workspace, DefaultCacheMatchesTheSharedExampleLayout) {
  Environment cache("DATAMUNGE_DATA_DIR", root.string());
  EXPECT_EQ(d::prepare_example_data("memory"), fs::absolute(root).string());
  EXPECT_TRUE(fs::is_regular_file(root / "ntm/data.npz"));
  EXPECT_TRUE(fs::is_regular_file(root / "memnet/data.npz"));
  EXPECT_FALSE(fs::exists(root / "memory"));
}
TEST_F(Workspace, NativeMemoryPreparationIsReusableAndRepairsMissingOutput) {
  d::prepare_example_data("memory", root.string());
  const auto before = read(root / "ntm/data.npz");
  ASSERT_FALSE(before.empty());
  std::thread a([&] { d::prepare_example_data("memory", root.string()); }),
      b([&] { d::prepare_example_data("memory", root.string()); });
  a.join();
  b.join();
  EXPECT_EQ(read(root / "ntm/data.npz"), before);
  fs::remove(root / "ntm/data.npz");
  d::prepare_example_data("memory", root.string());
  EXPECT_EQ(read(root / "ntm/data.npz"), before);
  EXPECT_THROW(d::prepare_example_data("unknown", root.string()), std::invalid_argument);
}
} // namespace
