/// @file dataset_assets.cpp
/// @brief Offline native transfer, archive, NPZ and deterministic preparation regression tests.
#include <gtest/gtest.h>

#include <atomic>
#include <array>
#include <cmath>
#include <map>
#include <set>
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

TEST_F(Workspace, KnownFactorLatticeIsExhaustiveBalancedAndReproducible) {
  const auto records=n::models::KnownFactorLattice();
  ASSERT_EQ(records.size(),9u);
  std::set<int> all_ids;std::set<std::vector<float>> unique_images;
  for(int split=0;split<3;++split){
    const int count=split==0?210:42;
    const auto& images=records[3*split].second;const auto& factors=records[3*split+1].second;
    const auto& ids=records[3*split+2].second;
    EXPECT_EQ(images.shape(),(n::Shape{count,28,28,1}));EXPECT_EQ(images.dtype(),n::DType::Float32);
    EXPECT_EQ(factors.dtype(),n::DType::Int32);EXPECT_EQ(ids.dtype(),n::DType::Int32);
    EXPECT_FALSE(images.requires_grad());
    const auto pixels=images.to_vector<float>();const auto source=ids.to_vector<int>(),f=factors.to_vector<int>();
    int counts[4][7]={};
    for(int row=0;row<count;++row){
      const int id=source[row];ASSERT_TRUE(all_ids.insert(id).second);
      if(row)EXPECT_LT(source[row-1],id);
      const int y=id%7,x=(id/7)%7,size=(id/49)%3,shape=id/147;
      EXPECT_EQ((std::vector<int>{f[4*row],f[4*row+1],f[4*row+2],f[4*row+3]}),(std::vector<int>{shape,size,x,y}));
      const int residue=(x+2*y+size+3*shape)%7;
      EXPECT_TRUE(split==2 ? residue==0 : split==1 ? residue==1 : residue>=2);
      for(int j=0;j<4;++j)++counts[j][f[4*row+j]];
      unique_images.insert(std::vector<float>(pixels.begin()+row*784,pixels.begin()+(row+1)*784));
      for(int py=0;py<28;++py)for(int px=0;px<28;++px){
        const double horizontal=std::abs(px-(3*x+5)),vertical=std::abs(py-(3*y+5));
        const bool expected=shape ? std::max(horizontal,vertical)<=size+2 : std::hypot(horizontal,vertical)<=size+2;
        EXPECT_EQ(pixels[row*784+py*28+px],float(expected));
      }
    }
    for(int j=0;j<4;++j)for(int level=0;level<(j==0?2:j==1?3:7);++level)EXPECT_EQ(counts[j][level],count/(j==0?2:j==1?3:7));
  }
  EXPECT_EQ(all_ids.size(),294u);EXPECT_EQ(unique_images.size(),294u);
  auto repeat=n::models::KnownFactorLattice();
  for(std::size_t i=0;i<records.size();++i)EXPECT_TRUE(n::equal_bits(records[i].second,repeat[i].second));
}
TEST_F(Workspace, KnownFactorPreparationIsOfflineReusedAndRepairsMissingOutputs) {
  const auto result=d::prepare_example_data("beta-vae-assets",root.string());EXPECT_EQ(result,fs::absolute(root).string());
  const auto npz=root/"beta_vae/factors.npz",manifest=root/"beta_vae/manifest.json";
  ASSERT_TRUE(fs::is_regular_file(npz));ASSERT_TRUE(fs::is_regular_file(manifest));
  const auto expected=n::models::KnownFactorLattice(),actual=n::load_npz(npz.string());ASSERT_EQ(actual.size(),9u);
  for(std::size_t i=0;i<actual.size();++i){EXPECT_EQ(actual[i].first,expected[i].first);EXPECT_EQ(actual[i].second.dtype(),n::DType::Int32);EXPECT_TRUE(n::equal_bits(actual[i].second,n::cast(expected[i].second,n::DType::Int32)));}
  const auto bytes=read(npz),metadata=read(manifest);
  d::prepare_example_data("beta-vae-assets",root.string());EXPECT_EQ(read(npz),bytes);EXPECT_EQ(read(manifest),metadata);
  fs::remove(npz);d::prepare_example_data("beta-vae-assets",root.string());
  const auto regenerated=n::load_npz(npz.string());for(std::size_t i=0;i<actual.size();++i)EXPECT_TRUE(n::equal_bits(actual[i].second,regenerated[i].second));
  const auto provenance=nlohmann::json::parse(read(manifest));EXPECT_TRUE(provenance["generation_verified"].get<bool>());EXPECT_FALSE(provenance["external_source"].get<bool>());
  EXPECT_EQ(provenance["fit"],210);EXPECT_EQ(provenance["validation"],42);EXPECT_EQ(provenance["test"],42);
}

TEST_F(Workspace, SentenceGrammarIsExhaustiveBalancedAndTeacherForced) {
  d::prepare_example_data("sequence-vae-assets", root.string());
  const auto records = n::load_npz((root / "sequence_vae/sentences.npz").string());
  ASSERT_EQ(records.size(), 16u);
  std::map<std::string, n::Tensor> arrays(records.begin(), records.end());
  ASSERT_EQ(arrays.at("config").shape(), (n::Shape{2}));
  EXPECT_EQ(arrays.at("config").to_vector<int>(), (std::vector<int>{19, 7}));
  for (const auto& [name, tensor] : records) {
    EXPECT_EQ(tensor.dtype(), n::DType::Int32) << name;
    EXPECT_FALSE(tensor.requires_grad()) << name;
  }
  EXPECT_EQ(read(root / "sequence_vae/vocabulary.txt"),
            "[pad]\n[bos]\n[eos]\na\nthe\nred\ngreen\nblue\ncircle\nsquare\ntriangle\n"
            "moves\nrolls\nleft\nright\nup\ndown\nquickly\nslowly\n");
  std::set<int> all_ids;
  std::set<std::vector<int>> all_sentences;
  const std::array<int, 6> cardinalities{2, 3, 3, 2, 4, 3};
  for (const std::string split : {"fit", "validation", "test"}) {
    const int count = split == "fit" ? 288 : 72;
    ASSERT_EQ(arrays.at(split + "_tokens").shape(), (n::Shape{count, 7}));
    ASSERT_EQ(arrays.at(split + "_decoder_inputs").shape(), (n::Shape{count, 7}));
    ASSERT_EQ(arrays.at(split + "_lengths").shape(), (n::Shape{count}));
    ASSERT_EQ(arrays.at(split + "_factors").shape(), (n::Shape{count, 6}));
    ASSERT_EQ(arrays.at(split + "_source_ids").shape(), (n::Shape{count}));
    const auto targets = arrays.at(split + "_tokens").to_vector<int>();
    const auto inputs = arrays.at(split + "_decoder_inputs").to_vector<int>();
    const auto lengths = arrays.at(split + "_lengths").to_vector<int>();
    const auto factors = arrays.at(split + "_factors").to_vector<int>();
    const auto sources = arrays.at(split + "_source_ids").to_vector<int>();
    int marginals[6][4]{}, short_count = 0, token_count = 0;
    for (int row = 0; row < count; ++row) {
      SCOPED_TRACE(split + " row " + std::to_string(row));
      int id = 0;
      for (int j = 0; j < 6; ++j) {
        const int f = factors[row * 6 + j];
        ASSERT_GE(f, 0); ASSERT_LT(f, cardinalities[j]);
        ++marginals[j][f];
        id = id * cardinalities[j] + f;
      }
      EXPECT_EQ(id, sources[row]);
      EXPECT_TRUE(all_ids.insert(id).second);
      if (row) EXPECT_LT(sources[row - 1], sources[row]);
      const auto f = factors.begin() + row * 6;
      const int residue = 2 * ((f[1] + f[2] + f[5]) % 3) + (f[0] + f[3] + f[4]) % 2;
      EXPECT_TRUE(split == "test" ? residue == 0 : split == "validation" ? residue == 1 : residue >= 2);
      std::vector<int> sentence(targets.begin() + row * 7, targets.begin() + (row + 1) * 7);
      EXPECT_TRUE(all_sentences.insert(sentence).second);
      const std::array<int, 5> offsets{3, 5, 8, 11, 13};
      for (int j = 0; j < 5; ++j) EXPECT_EQ(sentence[j], offsets[j] + f[j]);
      const int eos = f[5] ? 6 : 5;
      if (f[5]) EXPECT_EQ(sentence[5], 16 + f[5]);
      EXPECT_EQ(sentence[eos], 2);
      if (eos == 5) EXPECT_EQ(sentence[6], 0);
      EXPECT_EQ(lengths[row], eos + 1);
      short_count += eos == 5;
      token_count += lengths[row];
      EXPECT_EQ(inputs[row * 7], 1);
      for (int t = 1; t < 7; ++t) EXPECT_EQ(inputs[row * 7 + t], sentence[t - 1]);
    }
    for (int j = 0; j < 6; ++j)
      for (int value = 0; value < cardinalities[j]; ++value)
        EXPECT_EQ(marginals[j][value], count / cardinalities[j]);
    EXPECT_EQ(short_count, count / 3);
    EXPECT_EQ(token_count, count * 20 / 3);
  }
  EXPECT_EQ(all_ids.size(), 432u); EXPECT_EQ(all_sentences.size(), 432u);
  EXPECT_EQ(*all_ids.begin(), 0); EXPECT_EQ(*all_ids.rbegin(), 431);
}

TEST_F(Workspace, SentenceAssetsUseSharedCacheAndRepairEveryMissingOutput) {
  Environment cache("DATAMUNGE_DATA_DIR", root.string());
  Environment offline("DATAMUNGE_DATA_OFFLINE", "1");
  EXPECT_EQ(d::prepare_example_data("sequence-vae-assets"), fs::absolute(root).string());
  EXPECT_FALSE(fs::exists(root / "sequence-vae-assets"));
  const auto directory = root / "sequence_vae";
  const auto archive = directory / "sentences.npz", manifest = directory / "manifest.json";
  const auto bytes = read(archive), vocabulary = read(directory / "vocabulary.txt"), metadata = read(manifest);
  const auto expected = n::load_npz(archive.string());
  const auto provenance = nlohmann::json::parse(metadata);
  EXPECT_TRUE(provenance["generation_verified"].get<bool>());
  EXPECT_FALSE(provenance["external_source"].get<bool>());
  EXPECT_EQ(provenance["cardinalities"], (std::vector<int>{2, 3, 3, 2, 4, 3}));
  EXPECT_EQ(provenance["fit"], 288); EXPECT_EQ(provenance["validation"], 72); EXPECT_EQ(provenance["test"], 72);
  std::thread a([&] { d::prepare_example_data("sequence-vae-assets"); }),
              b([&] { d::prepare_example_data("sequence-vae-assets"); });
  a.join(); b.join();
  EXPECT_EQ(read(archive), bytes); EXPECT_EQ(read(manifest), metadata);
  fs::remove(directory / "vocabulary.txt");
  d::prepare_example_data("sequence-vae-assets");
  EXPECT_EQ(read(directory / "vocabulary.txt"), vocabulary);
  EXPECT_EQ(read(archive), bytes); EXPECT_EQ(read(manifest), metadata);
  fs::remove(manifest);
  d::prepare_example_data("sequence-vae-assets");
  EXPECT_EQ(read(archive), bytes);
  EXPECT_FALSE(nlohmann::json::parse(read(manifest))["generation_verified"].get<bool>());
  fs::remove(archive);
  d::prepare_example_data("sequence-vae-assets");
  const auto regenerated = n::load_npz(archive.string());
  ASSERT_EQ(regenerated.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(regenerated[i].first, expected[i].first);
    EXPECT_TRUE(n::equal_bits(regenerated[i].second, expected[i].second));
  }
  EXPECT_EQ(read(manifest), metadata);
  EXPECT_EQ(read(directory / "vocabulary.txt"), vocabulary);
  for (const auto& entry : fs::directory_iterator(directory))
    EXPECT_FALSE(entry.path().filename().string().starts_with(".native-"));
}

TEST_F(Workspace, SuppliedSentenceArchiveAndVocabularyKeepTheirBytesAndUnverifiedProvenance) {
  const auto directory = root / "sequence_vae";
  fs::create_directories(directory);
  n::save_npz((directory / "sentences.npz").string(), {{"custom", n::tensor({7., 8.})}});
  write(directory / "vocabulary.txt", "custom vocabulary\n");
  const auto supplied = read(directory / "sentences.npz");
  d::prepare_example_data("sequence-vae-assets", root.string());
  EXPECT_EQ(read(directory / "sentences.npz"), supplied);
  EXPECT_EQ(read(directory / "vocabulary.txt"), "custom vocabulary\n");
  const auto manifest = nlohmann::json::parse(read(directory / "manifest.json"));
  EXPECT_FALSE(manifest["generation_verified"].get<bool>());
  EXPECT_FALSE(manifest.contains("generator"));
  EXPECT_EQ(manifest["source"], "caller-supplied sentences.npz");
  fs::remove(directory / "vocabulary.txt");
  d::prepare_example_data("sequence-vae-assets", root.string());
  EXPECT_EQ(read(directory / "sentences.npz"), supplied);
  EXPECT_EQ(nlohmann::json::parse(read(directory / "manifest.json")), manifest);
  EXPECT_TRUE(read(directory / "vocabulary.txt").starts_with("[pad]\n[bos]\n[eos]\n"));
}

TEST_F(Workspace, InceptionMetricRejectsCorruptSourceBeforeAnyExtractionOrDownload) {
  write(root / "inception-2015-12-05.tgz", "invalid original archive");
  EXPECT_THROW(d::prepare_example_data("inception-metric", root.string()), std::runtime_error);
  EXPECT_FALSE(fs::exists(root / "classify_image_graph_def.pb"));
  EXPECT_FALSE(fs::exists(root / ".native-inception-metric-v1.json"));
  EXPECT_EQ(read(root / "inception-2015-12-05.tgz"), "invalid original archive");
}

TEST_F(Workspace, InceptionMetricRehashesCompletedGraphInsteadOfTrustingMarkerSizes) {
  write(root / "classify_image_graph_def.pb", "altered graph");
  write(root / "LICENSE", "license");
  write(root / "manifest.json", "{}");
  nlohmann::json sizes;
  for (const auto& name : {"classify_image_graph_def.pb", "LICENSE", "manifest.json"})
    sizes[name] = fs::file_size(root / name);
  write(root / ".native-inception-metric-v1.json", sizes.dump());
  EXPECT_THROW(d::prepare_example_data("inception-metric", root.string()), std::runtime_error);
  EXPECT_EQ(read(root / "classify_image_graph_def.pb"), "altered graph");
}
