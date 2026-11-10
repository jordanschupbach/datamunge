#pragma once
/// @file assets.hpp
/// @brief Native verified downloads, bounded archive extraction and report dataset preparation.
#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::datasets {
/// @brief Compute a file's SHA-256 or MD5 using bounded streaming memory.
/// @param path Readable regular file, including an empty file.
/// @param algorithm "sha256" (default) or "md5"; other algorithms are rejected.
/// @return Lowercase hexadecimal digest, with no mutation of the source.
/// @throws std::invalid_argument For an unsupported algorithm.
/// @throws std::runtime_error For unavailable native adapters or read/hash errors.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <fstream>
/// std::ofstream("build/empty.bin").close();
/// const auto digest = dm::datasets::file_digest("build/empty.bin");
/// @endcode
/// @par References
/// - OpenSSL EVP streaming digests, https://docs.openssl.org/3.0/man3/EVP_DigestInit/
std::string file_digest(const std::string& path, const std::string& algorithm = "sha256");

/// @brief Fetch a source file atomically and verify existing and newly downloaded bytes.
///
/// Uses curl without a shell. Honors DATAMUNGE_DATA_OFFLINE=1 and, when
/// dataset is provided, `DATAMUNGE_DATA_MIRROR/<dataset>/<destination basename>`.
/// A corrupt existing file throws; callers may remove it and retry explicitly.
/// Parallel callers publish only complete verified files; no partial file is reused.
/// @param url Original HTTP(S) or file URL.
/// @param destination Local file path; parent directories are created.
/// @param digest Required lowercase SHA-256 (64 characters) or MD5 (32 characters).
/// @param bytes Expected file size, or zero to omit the size check.
/// @param dataset Optional mirror subdirectory, empty by default.
/// @throws std::invalid_argument For a malformed digest or mirror dataset name.
/// @throws std::runtime_error For offline cache misses, transfer or verification failures.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <fstream>
/// std::ofstream("/tmp/dm-source.bin").close();
/// dm::datasets::fetch_file("file:///tmp/dm-source.bin", "/tmp/dm-copy.bin",
///   "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
/// @endcode
/// @see download_file(), file_digest()
void fetch_file(
    const std::string& url,
    const std::string& destination,
    const std::string& digest,
    std::uint64_t      bytes   = 0,
    const std::string& dataset = "");

/// @brief Extract selected regular ZIP or tar members, including gzip-compressed tar.
///
/// Rejects absolute paths, traversal, symlinks, hardlinks and special
/// files, including traversal after ZIP path separator normalization. Existing
/// destination symlinks are rejected. Members are streamed into
/// temporary files and renamed only after successful decoding. All requested
/// names must occur exactly once. An empty selection extracts all regular files.
/// Completed files may remain if a subsequent member fails; incomplete files do not.
/// @param archive Local archive file, detected by format rather than extension.
/// @param directory Destination directory, created if needed.
/// @param members Exact archive member names, empty by default.
/// @param max_bytes Maximum cumulative expanded bytes; defaults to 8 GiB.
/// @throws std::invalid_argument For unsafe paths, duplicate or nonregular entries.
/// @throws std::runtime_error For missing members, expansion limit or archive/I/O failures.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::save_npz("build/source.npz", {{"values", dm::nn::tensor({1., 2.})}});
/// dm::datasets::extract_archive("build/source.npz", "build/source", {"values.npy"});
/// @endcode
/// @par References
/// - libarchive streaming readers, https://www.libarchive.org/
void extract_archive(
    const std::string&              archive,
    const std::string&              directory,
    const std::vector<std::string>& members   = {},
    std::uint64_t                   max_bytes = 8ULL * 1024 * 1024 * 1024);

/// @brief Prepare original-source datasets or explicit local teaching fixtures through the native library.
///
/// Accepted names: audio, depth, flowers, medical, mesh, multilabel, ocr,
/// optical_flow, pose, signals, style, text, two_stream, video, voxel,
/// memory, mnist-assets, beta-vae-assets, sequence-vae-assets and inception-metric.
/// Uses the original pinned sources and deterministic record selections of the
/// executable reports. Produces NPZ arrays plus provenance JSON/TSV files.
/// Existing completed native preparations are reused; missing outputs are rebuilt.
/// Archive downloads honor the same offline/mirror controls as fetch_file().
/// Video uses ffmpeg on PATH for codec decoding only; curl is needed for downloads.
/// Calls for the same cache are serialized. Errors propagate, never generating
/// substitute data. beta-vae-assets explicitly generates the original 294 binary
/// circle/square teaching images with known factors, compositional splits and
/// ascending source IDs; it is procedural data, not a downloaded benchmark.
/// sequence-vae-assets enumerates the repository's 432 unique six-factor
/// sentences: 288 fit, 72 validation, 72 test, ordered by source ID. Its
/// sequence_vae/ directory contains sentences.npz (sixteen Int32 arrays),
/// vocabulary.txt (PAD0/BOS1/EOS2 and sixteen words) and manifest.json.
/// Targets have seven steps with EOS-inclusive lengths six or seven; decoder
/// inputs are BOS followed by the previous target. Held-out combinations use
/// residue=2*((colour+shape+adverb)%3)+(article+verb+direction)%2:
/// test0, validation1, fit2..5. This is local procedural teaching data.
/// Existing supplied sentence archives are preserved, without schema validation;
/// a missing vocabulary is supplied and a missing manifest records unverified
/// caller provenance. Reports must validate supplied arrays before using them.
/// Independent framework reference fixtures are test artifacts,
/// not downloaded or regenerated by this data API.
/// @param name Dataset adapter name from the list above.
/// @param directory Explicit dataset cache directory. Empty selects the conventional
/// cache below data_home(): e.g. flowers102, modelnet, style_transfer or action_video.
/// For signals, text, memory, mnist-assets, beta-vae-assets and sequence-vae-assets,
/// the directory is their shared parent;
/// an empty value selects data_home() itself.
/// inception-metric instead selects data_home()/inception_metric and downloads
/// the original inception-2015-12-05.tgz from TensorFlow's official archive.
/// Retains the source archive, unmodified95,673,916-byte classify_image_graph_def.pb,
/// original LICENSE and provenance manifest. Both archive and graph have pinned
/// SHA-256/size checks; completed graph caches are rehashed on every call.
/// No weights/features are synthesized or converted; no TensorFlow runtime is
/// used. Native FrozenGraph executes the supplied ExpandDims:0 Float32 RGB
/// input in0..255 to the original2048 features and1008 classes. Callers choose
/// whether to include the graph's classifier bias when computing probabilities.
/// @return Absolute prepared cache directory.
/// @throws std::invalid_argument For an unknown adapter.
/// @throws std::runtime_error For unavailable adapters, download, format or preparation errors.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto root = dm::datasets::prepare_example_data("beta-vae-assets", "build/local-datasets");
/// const auto arrays = dm::nn::load_npz(root + "/beta_vae/factors.npz");
/// @endcode
/// @code
/// #include <dm/dm.hpp>
/// const auto root = dm::datasets::prepare_example_data("sequence-vae-assets", "build/local-datasets");
/// const auto arrays = dm::nn::load_npz(root + "/sequence_vae/sentences.npz");
/// @endcode
/// @see fetch_file(), extract_archive(), datamunge::nn::load_npz()
std::string prepare_example_data(const std::string& name, const std::string& directory = "");
} // namespace datamunge::datasets
