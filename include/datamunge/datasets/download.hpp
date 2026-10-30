#pragma once

/// @file download.hpp
/// @brief Downloads and caches the datasets too large to compile into the library.
///
/// Files are stored under data_home() (one subdirectory per dataset) and downloaded at most
/// once: later calls find them on disk. Each download is checked against a pinned MD5
/// checksum and moved into place only once complete, so an interrupted download never leaves
/// a corrupt file behind. Downloads run the `curl` program, which must be on `PATH`.
///
/// Environment variables (the same ones the NN examples use, so caches are shared):
/// - `DATAMUNGE_DATA_DIR` -- the cache root (see data_home());
/// - `DATAMUNGE_DATA_OFFLINE=1` -- never download; a missing file throws instead;
/// - `DATAMUNGE_DATA_MIRROR` -- a base URL used instead of the original hosts, laid out as
///   `<mirror>/<dataset>/<file name>`; checksums still apply.
///
/// @code
/// const std::string csv = dm::datasets::fetch("diamonds");   // downloads once, ~2.7 MB
/// const std::string dir = dm::datasets::fetch("mnist");      // directory of IDX files
/// @endcode

#include <string>
#include <vector>

namespace datamunge::datasets {

/// @brief The cache root where downloaded datasets are stored.
///
/// The first of: `$DATAMUNGE_DATA_DIR`; `$XDG_CACHE_HOME/datamunge/datasets`;
/// `$HOME/.cache/datamunge/datasets`; on Windows `%LOCALAPPDATA%/datamunge/datasets`;
/// otherwise `build/datasets` relative to the working directory. Nothing is created.
///
/// @return The directory path (it may not exist yet).
///
/// @par Example
/// @code
/// std::print("{}\n", dm::datasets::data_home());   // e.g. /home/me/.cache/datamunge/datasets
/// @endcode
/// @see fetch()
std::string data_home();

/// @brief The names fetch() accepts, sorted: "california_housing", "cifar10", "cifar100",
///        "diamonds", "fashion_mnist", "mnist", "sudoku".
///
/// @par Example
/// @code
/// for (const auto& name : dm::datasets::downloadable()) std::print("{}\n", name);
/// @endcode
std::vector<std::string> downloadable();

/// @brief Makes sure a downloadable dataset is on disk and returns where it is.
///
/// For the tabular datasets ("diamonds", "california_housing") this is the CSV file. For the
/// image datasets it is a directory: "mnist" and "fashion_mnist" hold the four uncompressed
/// IDX files (`train-images-idx3-ubyte`, ...); "cifar10" holds `data_batch_1..5.bin` and
/// `test_batch.bin`; "cifar100" holds `train.bin` and `test.bin` (the binary versions);
/// "sudoku" holds Norvig's `easy50.txt`, `top95.txt` and `hardest.txt`.
/// Approximate download sizes: diamonds 2.7 MB, california_housing 1.4 MB, MNIST 11 MB,
/// Fashion-MNIST 30 MB, CIFAR-10 163 MB, CIFAR-100 161 MB, sudoku 14 KB.
///
/// @param name One of downloadable().
/// @param directory Where the files live; empty means `data_home() + "/" + name`. Files
///        already present there are used as they are.
/// @return The local path of the CSV file or the dataset directory.
/// @throws std::invalid_argument for an unknown name.
/// @throws std::runtime_error if files are missing and downloading is disabled or fails, or
///         a download does not match its checksum.
///
/// @par Example
/// @code
/// const auto housing = dm::datasets::read_csv(dm::datasets::fetch("california_housing"));
/// @endcode
/// @see data_home(), downloadable(), download_file()
std::string fetch(const std::string& name, const std::string& directory = "");

/// @brief Downloads @p url to @p destination with curl, unless the destination already exists.
///
/// The file is downloaded into a temporary directory next to @p destination, checked, and only
/// then renamed into place. Parent directories are created as needed. `DATAMUNGE_DATA_OFFLINE`
/// is honored; `DATAMUNGE_DATA_MIRROR` is not (fetch() applies it).
///
/// @param url An `https://`, `http://` or `file://` URL.
/// @param destination The final file path.
/// @param md5 Expected lowercase hex MD5 of the downloaded bytes; empty skips the check.
/// @throws std::runtime_error if downloading is disabled or fails, or the checksum differs.
///
/// @par Example
/// @code
/// dm::datasets::download_file("https://example.org/data.csv", "/tmp/data.csv");
/// @endcode
void download_file(const std::string& url, const std::string& destination, const std::string& md5 = "");

} // namespace datamunge::datasets
