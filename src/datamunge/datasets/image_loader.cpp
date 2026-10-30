/// @file image_loader.cpp
/// @brief ImageLoader: IDX/CIFAR file layout, epoch ordering, and per-batch disk reads.

#include <datamunge/datasets/download.hpp>
#include <datamunge/datasets/image_loader.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::datasets {

/// @brief Label-name tables and binary-reading helpers for ImageLoader.
namespace {

namespace fs = std::filesystem;

/// @brief Fashion-MNIST label names, by label.
const std::vector<std::string> kFashionNames = {"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                                                "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot"};

/// @brief CIFAR-10 label names, by label (batches.meta.txt).
const std::vector<std::string> kCifar10Names = {"airplane", "automobile", "bird",  "cat",  "deer",
                                                "dog",      "frog",       "horse", "ship", "truck"};

/// @brief CIFAR-100 fine label names, by label (fine_label_names.txt).
const std::vector<std::string> kCifar100Fine = {
    "apple",        "aquarium_fish", "baby",       "bear",         "beaver",      "bed",        "bee",
    "beetle",       "bicycle",       "bottle",     "bowl",         "boy",         "bridge",     "bus",
    "butterfly",    "camel",         "can",        "castle",       "caterpillar", "cattle",     "chair",
    "chimpanzee",   "clock",         "cloud",      "cockroach",    "couch",       "crab",       "crocodile",
    "cup",          "dinosaur",      "dolphin",    "elephant",     "flatfish",    "forest",     "fox",
    "girl",         "hamster",       "house",      "kangaroo",     "keyboard",    "lamp",       "lawn_mower",
    "leopard",      "lion",          "lizard",     "lobster",      "man",         "maple_tree", "motorcycle",
    "mountain",     "mouse",         "mushroom",   "oak_tree",     "orange",      "orchid",     "otter",
    "palm_tree",    "pear",          "pickup_truck", "pine_tree",  "plain",       "plate",      "poppy",
    "porcupine",    "possum",        "rabbit",     "raccoon",      "ray",         "road",       "rocket",
    "rose",         "sea",           "seal",       "shark",        "shrew",       "skunk",      "skyscraper",
    "snail",        "snake",         "spider",     "squirrel",     "streetcar",   "sunflower",  "sweet_pepper",
    "table",        "tank",          "telephone",  "television",   "tiger",       "tractor",    "train",
    "trout",        "tulip",         "turtle",     "wardrobe",     "whale",       "willow_tree", "wolf",
    "woman",        "worm"};

/// @brief CIFAR-100 coarse (superclass) label names, by label (coarse_label_names.txt).
const std::vector<std::string> kCifar100Coarse = {
    "aquatic_mammals",       "fish",
    "flowers",               "food_containers",
    "fruit_and_vegetables",  "household_electrical_devices",
    "household_furniture",   "insects",
    "large_carnivores",      "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes", "large_omnivores_and_herbivores",
    "medium_mammals",        "non-insect_invertebrates",
    "people",                "reptiles",
    "small_mammals",         "trees",
    "vehicles_1",            "vehicles_2"};

/// @brief Reads a big-endian 32-bit integer from an IDX header.
std::uint32_t read_be32(std::istream& in) {
  unsigned char b[4];
  in.read(reinterpret_cast<char*>(b), 4);
  if (!in) throw std::runtime_error("truncated IDX header");
  return (std::uint32_t(b[0]) << 24) | (std::uint32_t(b[1]) << 16) | (std::uint32_t(b[2]) << 8) | b[3];
}

/// @brief Opens a dataset file for binary reading, with a helpful error when it is missing.
std::ifstream open_binary(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open dataset file " + path);
  return in;
}

} // namespace

int ImageBatch::size() const noexcept { return shape.empty() ? 0 : shape[0]; }

int ImageBatch::features() const noexcept {
  int n = 1;
  for (std::size_t i = 1; i < shape.size(); ++i) n *= shape[i];
  return shape.empty() ? 0 : n;
}

std::vector<double> ImageBatch::image(int i) const {
  if (i < 0 || i >= size()) throw std::out_of_range("ImageBatch::image: index out of range");
  const auto f = static_cast<std::size_t>(features());
  return {pixels.begin() + static_cast<std::ptrdiff_t>(f * i), pixels.begin() + static_cast<std::ptrdiff_t>(f * (i + 1))};
}

ImageLoader::ImageLoader(const std::string& dataset, const std::string& split, const std::string& directory)
    : dataset_(dataset), split_(split) {
  if (split != "train" && split != "test")
    throw std::invalid_argument("ImageLoader split must be \"train\" or \"test\", got \"" + split + "\"");
  const bool idx = dataset == "mnist" || dataset == "fashion_mnist";
  if (!idx && dataset != "cifar10" && dataset != "cifar100")
    throw std::invalid_argument("ImageLoader dataset must be mnist, fashion_mnist, cifar10 or cifar100, got \"" +
                                dataset + "\"");
  directory_ = fetch(dataset, directory);
  const fs::path dir(directory_);
  if (idx) {
    const std::string prefix = split == "train" ? "train" : "t10k";
    const std::string images = (dir / (prefix + "-images-idx3-ubyte")).string();
    const std::string labels = (dir / (prefix + "-labels-idx1-ubyte")).string();
    auto in = open_binary(images);
    if (read_be32(in) != 0x00000803) throw std::runtime_error(images + " is not an IDX image file");
    split_size_ = static_cast<int>(read_be32(in));
    height_ = static_cast<int>(read_be32(in));
    width_ = static_cast<int>(read_be32(in));
    channels_ = 1;
    header_bytes_ = 16;
    const auto expected = 16ull + 1ull * split_size_ * height_ * width_;
    if (fs::file_size(images) != expected) throw std::runtime_error(images + " has the wrong size for its header");
    auto lin = open_binary(labels);
    if (read_be32(lin) != 0x00000801) throw std::runtime_error(labels + " is not an IDX label file");
    if (static_cast<int>(read_be32(lin)) != split_size_)
      throw std::runtime_error(labels + " and " + images + " disagree on the number of examples");
    labels_.resize(static_cast<std::size_t>(split_size_));
    lin.read(reinterpret_cast<char*>(labels_.data()), static_cast<std::streamsize>(labels_.size()));
    if (!lin) throw std::runtime_error("truncated IDX label file " + labels);
    files_ = {images};
    file_rows_ = {split_size_};
  } else {
    const bool c100 = dataset == "cifar100";
    height_ = width_ = 32;
    channels_ = 3;
    label_bytes_ = c100 ? 2 : 1;
    std::vector<std::string> names;
    if (c100)
      names = {split == "train" ? "train.bin" : "test.bin"};
    else if (split == "test")
      names = {"test_batch.bin"};
    else
      names = {"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin"};
    const auto record = static_cast<std::uintmax_t>(label_bytes_ + 3072);
    for (const auto& name : names) {
      const std::string path = (dir / name).string();
      const auto bytes = fs::file_size(path);
      if (bytes == 0 || bytes % record != 0) throw std::runtime_error(path + " is not a CIFAR binary batch file");
      files_.push_back(path);
      file_rows_.push_back(static_cast<int>(bytes / record));
      split_size_ += file_rows_.back();
    }
  }
  reorder();
}

ImageLoader& ImageLoader::set_batch_size(int size) {
  if (size <= 0) throw std::invalid_argument("ImageLoader batch size must be positive");
  batch_size_ = size;
  position_ = 0;
  return *this;
}

ImageLoader& ImageLoader::set_shuffle(bool shuffle) {
  shuffle_ = shuffle;
  reorder();
  return *this;
}

ImageLoader& ImageLoader::set_seed(int seed) {
  seed_ = seed;
  reorder();
  return *this;
}

ImageLoader& ImageLoader::set_drop_last(bool drop_last) {
  drop_last_ = drop_last;
  position_ = 0;
  return *this;
}

ImageLoader& ImageLoader::set_normalize(bool normalize) {
  normalize_ = normalize;
  position_ = 0;
  return *this;
}

ImageLoader& ImageLoader::set_layout(const std::string& layout) {
  if (layout != "nhwc" && layout != "nchw" && layout != "flat")
    throw std::invalid_argument("ImageLoader layout must be \"nhwc\", \"nchw\" or \"flat\", got \"" + layout + "\"");
  layout_ = layout;
  position_ = 0;
  return *this;
}

ImageLoader& ImageLoader::set_limit(int count) {
  if (count < 0) throw std::invalid_argument("ImageLoader limit must not be negative");
  limit_ = count;
  reorder();
  return *this;
}

ImageLoader& ImageLoader::set_coarse_labels(bool coarse) {
  coarse_ = coarse;
  position_ = 0;
  return *this;
}

const std::string& ImageLoader::dataset() const noexcept { return dataset_; }
const std::string& ImageLoader::split() const noexcept { return split_; }
const std::string& ImageLoader::directory() const noexcept { return directory_; }
int ImageLoader::batch_size() const noexcept { return batch_size_; }
bool ImageLoader::shuffle() const noexcept { return shuffle_; }
int ImageLoader::seed() const noexcept { return seed_; }
bool ImageLoader::drop_last() const noexcept { return drop_last_; }
bool ImageLoader::normalize() const noexcept { return normalize_; }
const std::string& ImageLoader::layout() const noexcept { return layout_; }
bool ImageLoader::coarse_labels() const noexcept { return coarse_; }

int ImageLoader::size() const noexcept { return limit_ > 0 && limit_ < split_size_ ? limit_ : split_size_; }
int ImageLoader::split_size() const noexcept { return split_size_; }

int ImageLoader::num_batches() const noexcept {
  return drop_last_ ? size() / batch_size_ : (size() + batch_size_ - 1) / batch_size_;
}

int ImageLoader::height() const noexcept { return height_; }
int ImageLoader::width() const noexcept { return width_; }
int ImageLoader::channels() const noexcept { return channels_; }

int ImageLoader::num_classes() const noexcept {
  if (dataset_ == "cifar100") return coarse_ ? 20 : 100;
  return 10;
}

std::vector<std::string> ImageLoader::class_names() const {
  if (dataset_ == "fashion_mnist") return kFashionNames;
  if (dataset_ == "cifar10") return kCifar10Names;
  if (dataset_ == "cifar100") return coarse_ ? kCifar100Coarse : kCifar100Fine;
  return {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
}

int ImageLoader::epoch() const noexcept { return epoch_; }
int ImageLoader::position() const noexcept { return position_; }

bool ImageLoader::has_next() const noexcept { return position_ < num_batches(); }

ImageBatch ImageLoader::next() {
  if (!has_next()) throw std::out_of_range("ImageLoader::next: epoch finished; call reset() to start the next one");
  return batch(position_++);
}

void ImageLoader::reset() {
  ++epoch_;
  reorder();
}

void ImageLoader::reorder() {
  position_ = 0;
  order_.resize(static_cast<std::size_t>(size()));
  std::iota(order_.begin(), order_.end(), 0);
  if (shuffle_) {
    std::seed_seq seq{static_cast<std::uint32_t>(seed_), static_cast<std::uint32_t>(epoch_)};
    std::mt19937_64 rng(seq);
    std::shuffle(order_.begin(), order_.end(), rng);
  }
}

ImageBatch ImageLoader::batch(int index) const {
  if (index < 0 || index >= num_batches()) throw std::out_of_range("ImageLoader::batch: index out of range");
  const auto begin = static_cast<std::size_t>(index) * static_cast<std::size_t>(batch_size_);
  const auto end = std::min(order_.size(), begin + static_cast<std::size_t>(batch_size_));
  return take(std::vector<int>(order_.begin() + static_cast<std::ptrdiff_t>(begin),
                               order_.begin() + static_cast<std::ptrdiff_t>(end)));
}

ImageBatch ImageLoader::load_all() const {
  std::vector<int> all(static_cast<std::size_t>(size()));
  std::iota(all.begin(), all.end(), 0);
  return take(all);
}

ImageBatch ImageLoader::take(const std::vector<int>& indices) const {
  const int n = static_cast<int>(indices.size());
  const int plane = height_ * width_;
  const int features = plane * channels_;
  ImageBatch out;
  out.indices = indices;
  out.labels.resize(indices.size());
  out.pixels.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(features));
  if (layout_ == "nhwc")
    out.shape = {n, height_, width_, channels_};
  else if (layout_ == "nchw")
    out.shape = {n, channels_, height_, width_};
  else
    out.shape = {n, features};
  const bool planar_out = layout_ == "nchw";
  const double scale = normalize_ ? 1.0 / 255.0 : 1.0;
  std::vector<std::ifstream> streams;
  for (const auto& file : files_) streams.push_back(open_binary(file));
  std::vector<unsigned char> record(static_cast<std::size_t>(label_bytes_ + features));
  for (int k = 0; k < n; ++k) {
    const int example = indices[static_cast<std::size_t>(k)];
    if (example < 0 || example >= size()) throw std::out_of_range("ImageLoader::take: index out of range");
    // Locate the file holding this example and its position inside it.
    std::size_t file = 0;
    int local = example;
    while (local >= file_rows_[file]) local -= file_rows_[file++];
    auto& in = streams[file];
    in.seekg(header_bytes_ + static_cast<long long>(local) * static_cast<long long>(record.size()));
    in.read(reinterpret_cast<char*>(record.data()), static_cast<std::streamsize>(record.size()));
    if (!in) throw std::runtime_error("truncated dataset file " + files_[file]);
    if (label_bytes_ == 0)
      out.labels[static_cast<std::size_t>(k)] = labels_[static_cast<std::size_t>(example)];
    else  // CIFAR-100 records are <coarse><fine><pixels>; CIFAR-10 are <label><pixels>.
      out.labels[static_cast<std::size_t>(k)] = record[label_bytes_ == 2 && !coarse_ ? 1 : 0];
    const unsigned char* px = record.data() + label_bytes_;
    double* dst = out.pixels.data() + static_cast<std::size_t>(k) * static_cast<std::size_t>(features);
    // On disk: MNIST is HW (one channel); CIFAR is CHW (all red, then green, then blue).
    const bool planar_in = channels_ > 1;
    for (int c = 0; c < channels_; ++c)
      for (int p = 0; p < plane; ++p) {
        const int from = planar_in ? c * plane + p : p * channels_ + c;
        const int to = planar_out ? c * plane + p : p * channels_ + c;
        dst[to] = px[from] * scale;
      }
  }
  return out;
}

} // namespace datamunge::datasets
