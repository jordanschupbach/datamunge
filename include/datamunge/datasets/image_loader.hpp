#pragma once

/// @file image_loader.hpp
/// @brief Streams MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100 from disk in batches.
///
/// An ImageLoader never loads a whole dataset. It keeps only the file layout (and, for MNIST,
/// the 60 KB of labels) and reads each batch's images from disk when asked, so memory use is
/// set by the batch size. The files are downloaded and cached on first use (see fetch()).
///
/// Change how batches come back with the chainable setters: batch size, shuffling (a new
/// order every epoch, reproducible from the seed), dropping a short last batch, raw 0..255 or
/// [0, 1] pixel values, the memory layout, a cap on the number of examples, and coarse or fine
/// CIFAR-100 labels.
///
/// @code
/// dm::datasets::ImageLoader train("mnist", "train");
/// train.set_batch_size(128).set_shuffle(true).set_seed(42).set_layout("flat");
/// for (int epoch = 0; epoch < 3; ++epoch) {
///   while (train.has_next()) {
///     const auto batch = train.next();   // batch.pixels: 128 x 784 doubles in [0, 1]
///     // ... fit on batch.pixels / batch.labels ...
///   }
///   train.reset();                       // next epoch, reshuffled
/// }
/// @endcode

#include <string>
#include <vector>

namespace datamunge::datasets {

/// @brief One batch of images and labels returned by ImageLoader.
///
/// `pixels` holds size() images back to back, each laid out as `shape` describes:
/// - "nhwc": shape {n, height, width, channels}, channels interleaved per pixel;
/// - "nchw": shape {n, channels, height, width}, one full plane per channel;
/// - "flat": shape {n, height * width * channels}, in "nhwc" order.
struct ImageBatch {
  std::vector<double> pixels;  ///< Image values, row-major per `shape`.
  std::vector<int> labels;     ///< Class index of each image (see ImageLoader::class_names()).
  std::vector<int> indices;    ///< Position of each image within its split, 0-based.
  std::vector<int> shape;      ///< Dimensions of `pixels`; shape[0] is the number of images.

  /// @brief Number of images in the batch.
  [[nodiscard]] int size() const noexcept;

  /// @brief Values per image (height * width * channels).
  [[nodiscard]] int features() const noexcept;

  /// @brief A copy of image @p i's features() values.
  /// @throws std::out_of_range if @p i is not in [0, size()).
  [[nodiscard]] std::vector<double> image(int i) const;
};

/// @brief A configurable, disk-backed batch iterator over a standard image dataset.
///
/// Datasets: "mnist" and "fashion_mnist" (28x28 grayscale, 10 classes; 60000 train / 10000
/// test), "cifar10" (32x32 RGB, 10 classes; 50000 / 10000) and "cifar100" (32x32 RGB, 100 fine
/// or 20 coarse classes; 50000 / 10000).
///
/// Iteration works in epochs. has_next()/next() walk the current epoch's order; reset() starts
/// the next epoch (with a fresh shuffle when shuffling is on). Every setter returns the loader
/// for chaining and rewinds the current epoch, so configure before iterating.
///
/// Reading is const and opens files per call, so batch()/take() may be called from several
/// threads at once; next()/reset() and the setters mutate and must not race.
///
/// @par Example
/// @code
/// dm::datasets::ImageLoader test("cifar10", "test");
/// test.set_batch_size(500).set_layout("nchw").set_normalize(false);
/// const auto first = test.next();          // shape {500, 3, 32, 32}, values 0..255
/// std::print("{}\n", test.class_names()[first.labels[0]]);   // "cat"
/// @endcode
class ImageLoader {
 public:
  /// @brief Opens a dataset split, downloading the files first if they are missing.
  /// @param dataset "mnist", "fashion_mnist", "cifar10" or "cifar100".
  /// @param split "train" or "test".
  /// @param directory Where the dataset's files are (or should be downloaded); empty means
  ///        `data_home() + "/" + dataset`.
  /// @throws std::invalid_argument for an unknown dataset or split.
  /// @throws std::runtime_error if the files are missing and cannot be downloaded, or are
  ///         not valid files of that dataset.
  explicit ImageLoader(const std::string& dataset, const std::string& split = "train",
                       const std::string& directory = "");

  /// @brief Images per batch (default 32).
  /// @throws std::invalid_argument if @p size is not positive.
  ImageLoader& set_batch_size(int size);

  /// @brief Turns shuffling on or off (default off). Epoch e's order is a permutation drawn
  ///        from (seed(), e), so the same seed always gives the same sequence of epochs.
  ImageLoader& set_shuffle(bool shuffle);

  /// @brief Sets the shuffle seed (default 0); see set_shuffle().
  ImageLoader& set_seed(int seed);

  /// @brief When true, a final batch smaller than batch_size() is skipped (default false).
  ImageLoader& set_drop_last(bool drop_last);

  /// @brief true scales pixels to [0, 1] (default); false keeps the raw 0..255 values.
  ImageLoader& set_normalize(bool normalize);

  /// @brief Memory layout of ImageBatch::pixels: "nhwc" (default), "nchw" or "flat".
  /// @throws std::invalid_argument for any other value.
  ImageLoader& set_layout(const std::string& layout);

  /// @brief Uses only the first @p count examples of the split (0, the default, means all).
  ///        Handy for quick experiments. Shuffling then permutes just those examples.
  /// @throws std::invalid_argument if @p count is negative.
  ImageLoader& set_limit(int count);

  /// @brief CIFAR-100 only: true labels images with the 20 superclasses, false (default)
  ///        with the 100 fine classes. Ignored by the other datasets.
  ImageLoader& set_coarse_labels(bool coarse);

  /// @brief The dataset name given to the constructor.
  [[nodiscard]] const std::string& dataset() const noexcept;
  /// @brief "train" or "test".
  [[nodiscard]] const std::string& split() const noexcept;
  /// @brief The directory the files are read from.
  [[nodiscard]] const std::string& directory() const noexcept;
  /// @brief Images per batch.
  [[nodiscard]] int batch_size() const noexcept;
  /// @brief Whether each epoch is shuffled.
  [[nodiscard]] bool shuffle() const noexcept;
  /// @brief The shuffle seed.
  [[nodiscard]] int seed() const noexcept;
  /// @brief Whether a short final batch is skipped.
  [[nodiscard]] bool drop_last() const noexcept;
  /// @brief Whether pixels are scaled to [0, 1].
  [[nodiscard]] bool normalize() const noexcept;
  /// @brief "nhwc", "nchw" or "flat".
  [[nodiscard]] const std::string& layout() const noexcept;
  /// @brief Whether CIFAR-100 uses coarse labels.
  [[nodiscard]] bool coarse_labels() const noexcept;

  /// @brief Examples per epoch: the split size, or the limit if one is set and smaller.
  [[nodiscard]] int size() const noexcept;
  /// @brief Examples in the split on disk, ignoring any limit.
  [[nodiscard]] int split_size() const noexcept;
  /// @brief Batches per epoch, honoring drop_last().
  [[nodiscard]] int num_batches() const noexcept;
  /// @brief Image height in pixels.
  [[nodiscard]] int height() const noexcept;
  /// @brief Image width in pixels.
  [[nodiscard]] int width() const noexcept;
  /// @brief Channels per pixel: 1 for MNIST-style, 3 (RGB) for CIFAR.
  [[nodiscard]] int channels() const noexcept;
  /// @brief Number of label classes (CIFAR-100: 100, or 20 with coarse labels).
  [[nodiscard]] int num_classes() const noexcept;
  /// @brief Label names, indexed by label (e.g. "airplane" for CIFAR-10 label 0).
  [[nodiscard]] std::vector<std::string> class_names() const;

  /// @brief The current epoch number, starting at 0 and advanced by reset().
  [[nodiscard]] int epoch() const noexcept;
  /// @brief Index of the batch next() will return in this epoch.
  [[nodiscard]] int position() const noexcept;

  /// @brief true while the current epoch has batches left.
  [[nodiscard]] bool has_next() const noexcept;
  /// @brief Reads the next batch of the current epoch.
  /// @throws std::out_of_range when the epoch is finished (call reset() for the next one).
  [[nodiscard]] ImageBatch next();
  /// @brief Starts the next epoch: the position returns to 0 and, if shuffling, a new
  ///        order is drawn.
  void reset();

  /// @brief Reads batch @p index of the current epoch without moving the position.
  /// @throws std::out_of_range if @p index is not in [0, num_batches()).
  [[nodiscard]] ImageBatch batch(int index) const;
  /// @brief Reads the listed examples (split positions), in the order given.
  /// @throws std::out_of_range if an index is not in [0, size()).
  [[nodiscard]] ImageBatch take(const std::vector<int>& indices) const;
  /// @brief Reads every example of the epoch (all of size()) into one batch, in split order.
  ///        Mind memory: all 60000 MNIST images take ~376 MB as doubles; see set_limit().
  [[nodiscard]] ImageBatch load_all() const;

 private:
  /// @brief Recomputes the current epoch's example order.
  void reorder();

  std::string dataset_;                 ///< Dataset name.
  std::string split_;                   ///< "train" or "test".
  std::string directory_;               ///< Directory holding the files.
  std::vector<std::string> files_;      ///< Image files, in split order.
  std::vector<int> file_rows_;          ///< Examples in each of files_.
  std::vector<unsigned char> labels_;   ///< MNIST-style labels (CIFAR labels live in records).
  int split_size_ = 0;                  ///< Examples on disk.
  int height_ = 0;                      ///< Image height.
  int width_ = 0;                       ///< Image width.
  int channels_ = 0;                    ///< Channels per pixel.
  long long header_bytes_ = 0;          ///< Bytes before the first image (IDX header).
  int label_bytes_ = 0;                 ///< Label bytes before each CIFAR record's pixels.
  int batch_size_ = 32;                 ///< Images per batch.
  bool shuffle_ = false;                ///< Whether to shuffle each epoch.
  int seed_ = 0;                        ///< Shuffle seed.
  bool drop_last_ = false;              ///< Skip a short final batch.
  bool normalize_ = true;               ///< Scale pixels to [0, 1].
  std::string layout_ = "nhwc";         ///< Output layout.
  int limit_ = 0;                       ///< Example cap (0 = none).
  bool coarse_ = false;                 ///< CIFAR-100 coarse labels.
  int epoch_ = 0;                       ///< Current epoch.
  int position_ = 0;                    ///< Next batch index.
  std::vector<int> order_;              ///< This epoch's example order.
};

} // namespace datamunge::datasets
