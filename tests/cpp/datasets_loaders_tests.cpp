// datamunge::datasets: built-in tables, CSV parsing/streaming, cached downloads and the
// disk-backed ImageLoader. Everything here runs offline: downloads use file:// URLs and the
// image tests write small synthetic MNIST/CIFAR files.
#include <datamunge/algorithms/md5.hpp>
#include <datamunge/datasets/datasets.hpp>

#include <gtest/gtest.h>

#include <algorithm>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

namespace ds = datamunge::datasets;
namespace fs = std::filesystem;
using Frame = datamunge::dstruct::DataFrame;

namespace {

/// @brief A fresh, empty directory removed at the end of the test.
struct TempDir {
  fs::path path;
  TempDir() {
    path = fs::temp_directory_path() /
           ("dm-datasets-test-" + std::to_string(std::rand()) + "-" +
            std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    fs::remove_all(path);
    fs::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

/// @brief Sets an environment variable for one scope, restoring the old value afterwards.
struct EnvGuard {
  std::string name;
  std::string old;
  bool had = false;
  EnvGuard(std::string n, const char* value) : name(std::move(n)) {
    if (const char* v = std::getenv(name.c_str())) {
      had = true;
      old = v;
    }
    if (value)
      setenv(name.c_str(), value, 1);
    else
      unsetenv(name.c_str());
  }
  ~EnvGuard() {
    if (had)
      setenv(name.c_str(), old.c_str(), 1);
    else
      unsetenv(name.c_str());
  }
};

void write_text(const fs::path& file, const std::string& text) {
  std::ofstream out(file, std::ios::binary);
  out << text;
}

std::string read_text(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void put_be32(std::ofstream& out, std::uint32_t v) {
  const unsigned char b[4] = {static_cast<unsigned char>(v >> 24), static_cast<unsigned char>(v >> 16),
                              static_cast<unsigned char>(v >> 8), static_cast<unsigned char>(v)};
  out.write(reinterpret_cast<const char*>(b), 4);
}

/// @brief Writes an n-image 2x3 IDX dataset where pixel p of image i is 10*i + p and the
///        label of image i is i % 10.
void write_idx(const fs::path& dir, const std::string& prefix, int n) {
  std::ofstream img(dir / (prefix + "-images-idx3-ubyte"), std::ios::binary);
  put_be32(img, 0x803);
  put_be32(img, n);
  put_be32(img, 2);
  put_be32(img, 3);
  for (int i = 0; i < n; ++i)
    for (int p = 0; p < 6; ++p) img.put(static_cast<char>(10 * i + p));
  std::ofstream lab(dir / (prefix + "-labels-idx1-ubyte"), std::ios::binary);
  put_be32(lab, 0x801);
  put_be32(lab, n);
  for (int i = 0; i < n; ++i) lab.put(static_cast<char>(i % 10));
}

/// @brief Writes a CIFAR file of @p n records. Record r has label (base + r) % 10 (CIFAR-100:
///        coarse = that, fine = that + 50) and pixel value channel*100 + position%100.
void write_cifar(const fs::path& file, int n, int base, bool c100) {
  std::ofstream out(file, std::ios::binary);
  for (int r = 0; r < n; ++r) {
    out.put(static_cast<char>((base + r) % 10));
    if (c100) out.put(static_cast<char>((base + r) % 10 + 50));
    for (int c = 0; c < 3; ++c)
      for (int p = 0; p < 1024; ++p) out.put(static_cast<char>(c * 100 + p % 100));
  }
}

}  // namespace

// ---------------------------------------------------------------------------- built-in tables

TEST(Datasets, BuiltInShapesTypesAndMissingValues) {
  struct Expect {
    const char* name;
    std::size_t rows, cols;
  };
  for (const auto& e : std::vector<Expect>{{"wine", 178, 14},          {"titanic", 891, 12},
                                           {"eight_schools", 8, 3},    {"faithful", 272, 2},
                                           {"mtcars", 32, 12},         {"airquality", 153, 6},
                                           {"breast_cancer", 569, 31}, {"diabetes", 442, 11},
                                           {"digits", 1797, 65},       {"iris", 150, 5},
                                           {"penguins", 344, 8},       {"tooth_growth", 60, 3},
                                           {"volcano", 5307, 3},        {"orthodont", 108, 4}}) {
    const auto frame = ds::load(e.name);
    EXPECT_EQ(frame.nrows(), e.rows) << e.name;
    EXPECT_EQ(frame.ncols(), e.cols) << e.name;
  }
  const auto titanic = ds::titanic();
  EXPECT_EQ(titanic.null_count("Age"), 177u);
  EXPECT_EQ(titanic.null_count("Cabin"), 687u);
  EXPECT_EQ(titanic.null_count("Embarked"), 2u);
  EXPECT_EQ(titanic.column_type("Name"), Frame::ColumnType::String);
  EXPECT_EQ(titanic.column_type("Survived"), Frame::ColumnType::Numeric);
  EXPECT_EQ(titanic.string_at("Name", 1), "Cumings, Mrs. John Bradley (Florence Briggs Thayer)");

  const auto volcano = ds::volcano();
  double top = 0.0;
  for (std::size_t i = 0; i < volcano.nrows(); ++i) top = std::max(top, volcano.double_at("elevation", i));
  EXPECT_EQ(top, 195.0);                              // R: max(volcano)
  EXPECT_EQ(volcano.double_at("elevation", 0), 100.0);  // R: volcano[1, 1]
  EXPECT_EQ(volcano.double_at("x", 86), 860.0);
  EXPECT_EQ(volcano.double_at("y", 87), 10.0);

  const auto air = ds::airquality();
  EXPECT_EQ(air.null_count("Ozone"), 37u);
  EXPECT_EQ(air.null_count("Solar.R"), 7u);
  EXPECT_EQ(air.drop_nulls().nrows(), 111u);
}

TEST(Datasets, BuiltInValuesMatchTheirSources) {
  const auto wine = ds::wine();
  EXPECT_DOUBLE_EQ(wine.double_at("alcohol", 0), 14.23);
  EXPECT_DOUBLE_EQ(wine.double_at("proline", 0), 1065);
  EXPECT_EQ(wine.string_at("cultivar", 0), "class_0");
  EXPECT_EQ(wine.string_at("cultivar", 177), "class_2");

  const auto schools = ds::eight_schools();
  EXPECT_EQ(schools.get_column("effect").values(), (std::vector<double>{28, 8, -3, 7, -1, 1, 18, 12}));
  EXPECT_EQ(schools.get_column("std_error").values(), (std::vector<double>{15, 10, 16, 11, 9, 11, 10, 18}));

  const auto faithful = ds::faithful();
  EXPECT_DOUBLE_EQ(faithful.double_at("eruptions", 0), 3.6);
  EXPECT_DOUBLE_EQ(faithful.double_at("waiting", 271), 74);

  const auto cars = ds::mtcars();
  EXPECT_EQ(cars.string_at("model", 0), "Mazda RX4");
  EXPECT_DOUBLE_EQ(cars.double_at("mpg", 0), 21);

  const auto cancer = ds::breast_cancer();
  EXPECT_DOUBLE_EQ(cancer.double_at("mean_radius", 0), 17.99);
  std::size_t malignant = 0;
  for (std::size_t i = 0; i < cancer.nrows(); ++i) malignant += cancer.string_at("diagnosis", i) == "malignant";
  EXPECT_EQ(malignant, 212u);

  const auto diabetes = ds::diabetes();
  EXPECT_DOUBLE_EQ(diabetes.double_at("progression", 0), 151);

  const auto digits = ds::digits();
  double max_pixel = 0;
  for (int p = 0; p < 64; ++p)
    for (double v : digits.get_column("pixel_" + std::to_string(p)).values()) max_pixel = std::max(max_pixel, v);
  EXPECT_DOUBLE_EQ(max_pixel, 16);
  EXPECT_DOUBLE_EQ(digits.double_at("digit", 0), 0);
}

TEST(Datasets, CatalogListsEveryDatasetAndLoadRejectsOthers) {
  const auto catalog = ds::catalog();
  EXPECT_EQ(catalog.nrows(), 23u);
  std::set<std::string> storage;
  for (std::size_t i = 0; i < catalog.nrows(); ++i) {
    storage.insert(catalog.string_at("storage", i));
    const std::string name = catalog.string_at("name", i);
    if (catalog.string_at("storage", i) == "built-in") EXPECT_NO_THROW((void)ds::load(name)) << name;
  }
  EXPECT_EQ(storage, (std::set<std::string>{"built-in", "download", "stream"}));
  EXPECT_THROW((void)ds::load("nope"), std::invalid_argument);
  EXPECT_THROW((void)ds::load("mnist"), std::invalid_argument);
}

// ---------------------------------------------------------------------------- CSV

TEST(DatasetsCsv, InfersTypesAndMissingValues) {
  const auto frame = ds::parse_csv("x,label,empty\n1.5,a,\nNA,\"b, c\",\n-2e3,\"say \"\"hi\"\"\",\n");
  ASSERT_EQ(frame.nrows(), 3u);
  EXPECT_EQ(frame.column_type("x"), Frame::ColumnType::Numeric);
  EXPECT_EQ(frame.column_type("label"), Frame::ColumnType::String);
  EXPECT_EQ(frame.column_type("empty"), Frame::ColumnType::String);
  EXPECT_DOUBLE_EQ(frame.double_at("x", 2), -2000);
  EXPECT_EQ(frame.null_count("x"), 1u);
  EXPECT_EQ(frame.string_at("label", 1), "b, c");
  EXPECT_EQ(frame.string_at("label", 2), "say \"hi\"");
  EXPECT_EQ(frame.null_count("empty"), 3u);
}

TEST(DatasetsCsv, HandlesQuotedNewlinesCrlfDelimitersAndNoHeader) {
  const auto quoted = ds::parse_csv("note,n\r\n\"two\nlines\",1\r\n\"x\",2\r\n");
  EXPECT_EQ(quoted.string_at("note", 0), "two\nlines");
  EXPECT_DOUBLE_EQ(quoted.double_at("n", 1), 2);

  ds::CsvOptions options;
  options.delimiter = ';';
  options.header = false;
  const auto bare = ds::parse_csv("1;2\n3;4\n", options);
  EXPECT_EQ(bare.columns(), (std::vector<std::string>{"V1", "V2"}));
  EXPECT_DOUBLE_EQ(bare.double_at("V2", 1), 4);

  ds::CsvOptions strings;
  strings.string_columns = {"zip"};
  const auto zips = ds::parse_csv("zip\n02134\n", strings);
  EXPECT_EQ(zips.string_at("zip", 0), "02134");
}

TEST(DatasetsCsv, RejectsMalformedInput) {
  EXPECT_THROW((void)ds::parse_csv("a,b\n1\n"), std::runtime_error);
  EXPECT_THROW((void)ds::parse_csv("a\n\"open\n"), std::runtime_error);
  EXPECT_THROW((void)ds::parse_csv("a,a\n1,2\n"), std::runtime_error);
  EXPECT_THROW((void)ds::read_csv("/definitely/not/here.csv"), std::runtime_error);
  EXPECT_EQ(ds::parse_csv("").nrows(), 0u);
}

TEST(DatasetsCsv, ReaderStreamsFixedSizeChunksAndRewinds) {
  TempDir dir;
  std::string text = "i,name\n";
  for (int i = 0; i < 25; ++i) text += std::to_string(i) + ",row" + std::to_string(i) + "\n";
  write_text(dir.path / "rows.csv", text);

  ds::CsvReader reader((dir.path / "rows.csv").string(), 10);
  EXPECT_EQ(reader.column_names(), (std::vector<std::string>{"i", "name"}));
  EXPECT_EQ(reader.numeric_columns(), (std::vector<bool>{true, false}));
  std::vector<std::size_t> sizes;
  double sum = 0;
  while (reader.has_next()) {
    const auto chunk = reader.next();
    sizes.push_back(chunk.nrows());
    for (double v : chunk.get_column("i").values()) sum += v;
  }
  EXPECT_EQ(sizes, (std::vector<std::size_t>{10, 10, 5}));
  EXPECT_DOUBLE_EQ(sum, 300);
  EXPECT_EQ(reader.rows_read(), 25);
  EXPECT_THROW((void)reader.next(), std::out_of_range);

  reader.reset();
  reader.set_batch_size(4);
  const auto first = reader.next();
  EXPECT_EQ(first.nrows(), 4u);
  EXPECT_EQ(first.string_at("name", 0), "row0");
  EXPECT_THROW(reader.set_batch_size(0), std::invalid_argument);
  EXPECT_THROW(ds::CsvReader((dir.path / "rows.csv").string(), 0), std::invalid_argument);
}

TEST(DatasetsCsv, ReaderTypesFromLeadingRowsAndReportsLaterMismatches) {
  TempDir dir;
  write_text(dir.path / "late.csv", "v\n1\n2\n3\noops\n");
  ds::CsvOptions options;
  options.infer_rows = 2;
  ds::CsvReader reader((dir.path / "late.csv").string(), 2, options);
  EXPECT_EQ(reader.next().nrows(), 2u);
  EXPECT_THROW((void)reader.next(), std::runtime_error);

  options.string_columns = {"v"};
  ds::CsvReader as_text((dir.path / "late.csv").string(), 10, options);
  EXPECT_EQ(as_text.next().string_at("v", 3), "oops");
}

// ---------------------------------------------------------------------------- downloads

TEST(DatasetsDownload, DownloadsVerifiesAndKeepsExistingFiles) {
  if (std::system("curl --version > /dev/null 2>&1") != 0) GTEST_SKIP() << "curl is not installed";
  TempDir dir;
  write_text(dir.path / "source.txt", "hello datasets\n");
  const std::string url = "file://" + (dir.path / "source.txt").string();
  const std::string md5 = datamunge::algorithms::md5_hex("hello datasets\n");

  const auto target = dir.path / "sub" / "copy.txt";
  ds::download_file(url, target.string(), md5);
  EXPECT_EQ(read_text(target), "hello datasets\n");

  // An existing destination is kept as is.
  write_text(target, "local edit");
  ds::download_file(url, target.string(), md5);
  EXPECT_EQ(read_text(target), "local edit");

  // A checksum mismatch throws and leaves nothing behind.
  const auto bad = dir.path / "bad.txt";
  EXPECT_THROW(ds::download_file(url, bad.string(), "00000000000000000000000000000000"), std::runtime_error);
  EXPECT_FALSE(fs::exists(bad));
  for (const auto& entry : fs::directory_iterator(dir.path))
    EXPECT_EQ(entry.path().filename().string().rfind(".prepare-", 0), std::string::npos);
}

TEST(DatasetsDownload, OfflineModeAndDataHome) {
  TempDir dir;
  {
    EnvGuard offline("DATAMUNGE_DATA_OFFLINE", "1");
    EnvGuard root("DATAMUNGE_DATA_DIR", dir.path.string().c_str());
    EXPECT_EQ(ds::data_home(), dir.path.string());
    EXPECT_THROW((void)ds::fetch("diamonds"), std::runtime_error);
    EXPECT_THROW(ds::download_file("file:///nowhere", (dir.path / "x").string()), std::runtime_error);
    // Files already present are used without any download.
    fs::create_directories(dir.path / "diamonds");
    write_text(dir.path / "diamonds" / "diamonds.csv", "carat,price\n0.23,326\n");
    EXPECT_EQ(ds::diamonds().nrows(), 1u);
  }
  EXPECT_THROW((void)ds::fetch("not-a-dataset"), std::invalid_argument);
  EXPECT_EQ(ds::downloadable().size(), 7u);
}

// ---------------------------------------------------------------------------- ImageLoader

TEST(DatasetsImageLoader, ReadsMnistStyleFilesInEveryLayout) {
  TempDir dir;
  write_idx(dir.path, "train", 7);
  write_idx(dir.path, "t10k", 3);
  ds::ImageLoader loader("mnist", "train", dir.path.string());
  EXPECT_EQ(loader.size(), 7);
  EXPECT_EQ(loader.height(), 2);
  EXPECT_EQ(loader.width(), 3);
  EXPECT_EQ(loader.channels(), 1);
  EXPECT_EQ(loader.class_names().size(), 10u);

  loader.set_batch_size(3).set_normalize(false);
  EXPECT_EQ(loader.num_batches(), 3);
  const auto first = loader.next();
  EXPECT_EQ(first.shape, (std::vector<int>{3, 2, 3, 1}));
  EXPECT_EQ(first.labels, (std::vector<int>{0, 1, 2}));
  EXPECT_EQ(first.indices, (std::vector<int>{0, 1, 2}));
  EXPECT_EQ(first.image(1), (std::vector<double>{10, 11, 12, 13, 14, 15}));

  loader.set_layout("flat").set_normalize(true);
  const auto flat = loader.batch(2);
  EXPECT_EQ(flat.shape, (std::vector<int>{1, 6}));
  EXPECT_DOUBLE_EQ(flat.pixels[0], 60.0 / 255.0);

  const auto frame = ds::to_dataframe(loader.take({4, 0}));
  EXPECT_EQ(frame.nrows(), 2u);
  EXPECT_EQ(frame.ncols(), 8u);
  EXPECT_DOUBLE_EQ(frame.double_at("label", 0), 4);
  EXPECT_DOUBLE_EQ(frame.double_at("index", 1), 0);

  ds::ImageLoader test("mnist", "test", dir.path.string());
  EXPECT_EQ(test.size(), 3);
  EXPECT_THROW(ds::ImageLoader("mnist", "valid", dir.path.string()), std::invalid_argument);
  EXPECT_THROW(ds::ImageLoader("imagenet", "train", dir.path.string()), std::invalid_argument);
}

TEST(DatasetsImageLoader, EpochsShuffleDropLastAndLimit) {
  TempDir dir;
  write_idx(dir.path, "train", 10);
  write_idx(dir.path, "t10k", 1);
  ds::ImageLoader loader("mnist", "train", dir.path.string());
  loader.set_batch_size(4).set_shuffle(true).set_seed(7);

  const auto epoch_order = [&] {
    std::vector<int> seen;
    while (loader.has_next())
      for (int i : loader.next().indices) seen.push_back(i);
    return seen;
  };
  const auto e0 = epoch_order();
  EXPECT_EQ(std::set<int>(e0.begin(), e0.end()).size(), 10u);
  EXPECT_THROW((void)loader.next(), std::out_of_range);
  loader.reset();
  EXPECT_EQ(loader.epoch(), 1);
  const auto e1 = epoch_order();
  EXPECT_NE(e0, e1);

  // The same seed reproduces the same epochs.
  ds::ImageLoader again("mnist", "train", dir.path.string());
  again.set_batch_size(4).set_shuffle(true).set_seed(7);
  std::vector<int> r0;
  while (again.has_next())
    for (int i : again.next().indices) r0.push_back(i);
  EXPECT_EQ(r0, e0);

  loader.set_shuffle(false).set_drop_last(true);
  EXPECT_EQ(loader.num_batches(), 2);
  loader.set_limit(5).set_drop_last(false);
  EXPECT_EQ(loader.size(), 5);
  EXPECT_EQ(loader.split_size(), 10);
  EXPECT_EQ(loader.num_batches(), 2);
  EXPECT_EQ(loader.load_all().indices, (std::vector<int>{0, 1, 2, 3, 4}));
  EXPECT_THROW((void)loader.take({5}), std::out_of_range);
  EXPECT_THROW((void)loader.batch(2), std::out_of_range);
  EXPECT_THROW(loader.set_layout("hwcn"), std::invalid_argument);
  EXPECT_THROW(loader.set_batch_size(0), std::invalid_argument);
}

TEST(DatasetsImageLoader, ReadsCifarAcrossFilesAndConvertsChannelOrder) {
  TempDir dir;
  for (int b = 1; b <= 5; ++b) write_cifar(dir.path / ("data_batch_" + std::to_string(b) + ".bin"), 2, 2 * (b - 1), false);
  write_cifar(dir.path / "test_batch.bin", 2, 0, false);
  ds::ImageLoader loader("cifar10", "train", dir.path.string());
  EXPECT_EQ(loader.size(), 10);
  EXPECT_EQ(loader.class_names()[3], "cat");
  loader.set_normalize(false);

  const auto hwc = loader.take({9});
  EXPECT_EQ(hwc.labels, (std::vector<int>{9}));
  EXPECT_EQ(hwc.shape, (std::vector<int>{1, 32, 32, 3}));
  // Pixel 5, channels R, G, B.
  EXPECT_DOUBLE_EQ(hwc.pixels[5 * 3 + 0], 5);
  EXPECT_DOUBLE_EQ(hwc.pixels[5 * 3 + 1], 105);
  EXPECT_DOUBLE_EQ(hwc.pixels[5 * 3 + 2], 205);

  loader.set_layout("nchw");
  const auto chw = loader.take({9});
  EXPECT_EQ(chw.shape, (std::vector<int>{1, 3, 32, 32}));
  EXPECT_DOUBLE_EQ(chw.pixels[1024 + 5], 105);

  TempDir c100;
  write_cifar(c100.path / "train.bin", 3, 0, true);
  write_cifar(c100.path / "test.bin", 1, 0, true);
  ds::ImageLoader fine("cifar100", "train", c100.path.string());
  EXPECT_EQ(fine.num_classes(), 100);
  EXPECT_EQ(fine.class_names().size(), 100u);
  EXPECT_EQ(fine.take({2}).labels, (std::vector<int>{52}));
  fine.set_coarse_labels(true);
  EXPECT_EQ(fine.num_classes(), 20);
  EXPECT_EQ(fine.take({2}).labels, (std::vector<int>{2}));
  EXPECT_EQ(fine.class_names().size(), 20u);
}
