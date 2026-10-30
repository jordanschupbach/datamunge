/// @file assets.cpp
/// @brief Native bounded hashing and archive ingress with atomic verified transfers.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <datamunge/datasets/assets.hpp>
#include <datamunge/datasets/download.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include <archive.h>
#include <archive_entry.h>
#include <openssl/evp.h>

#include "preparation/support.hpp"
#endif

namespace datamunge::datasets {
namespace fs = std::filesystem;
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
/// @brief Archive path and destination validation helpers.
namespace {
/// @brief Reject platform-dependent or escaping archive paths.
/// @param name Exact archive path.
void safe_path(const std::string& name) {
  const fs::path p(name);
  if (name.empty() || p.is_absolute() || name.find_first_of("\\:") != std::string::npos)
    throw std::invalid_argument("unsafe archive path: " + name);
  for (const auto& c : p)
    if (c == "..")
      throw std::invalid_argument("archive traversal: " + name);
}
/// @brief Reject symlinks along an extraction destination, including its ancestors.
/// @param path Absolute target path.
void no_symlinks(const fs::path& path) {
  fs::path at;
  for (const auto& c : fs::absolute(path)) {
    at /= c;
    if (fs::is_symlink(fs::symlink_status(at)))
      throw std::invalid_argument("symlink in extraction path: " + at.string());
  }
}
} // namespace
#endif
std::string file_digest(const std::string& path, const std::string& algorithm) {
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
  if (algorithm != "sha256" && algorithm != "md5")
    throw std::invalid_argument("expected sha256 or md5");
  std::ifstream in(path, std::ios::binary);
  if (!in || !fs::is_regular_file(path))
    throw std::runtime_error("cannot hash " + path);
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!ctx ||
      EVP_DigestInit_ex(ctx.get(), algorithm == "sha256" ? EVP_sha256() : EVP_md5(), nullptr) != 1)
    throw std::runtime_error("digest initialization failed");
  std::array<char, 65536> block;
  while (in.read(block.data(), block.size()) || in.gcount())
    if (EVP_DigestUpdate(ctx.get(), block.data(), std::size_t(in.gcount())) != 1)
      throw std::runtime_error("digest update failed");
  if (!in.eof())
    throw std::runtime_error("digest read failed: " + path);
  unsigned char bytes[EVP_MAX_MD_SIZE];
  unsigned int  size = 0;
  if (EVP_DigestFinal_ex(ctx.get(), bytes, &size) != 1)
    throw std::runtime_error("digest finalization failed");
  std::string out;
  const char* hex = "0123456789abcdef";
  for (unsigned int i = 0; i < size; ++i) {
    out += hex[bytes[i] >> 4];
    out += hex[bytes[i] & 15];
  }
  return out;
#else
  throw std::runtime_error("native data adapters require DATAMUNGE_ENABLE_DATA_ASSETS=ON");
#endif
}
void fetch_file(
    const std::string& url,
    const std::string& destination,
    const std::string& digest,
    std::uint64_t      bytes,
    const std::string& dataset) {
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
  if ((digest.size() != 32 && digest.size() != 64) ||
      digest.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::invalid_argument("expected lowercase SHA256 or MD5 digest");
  if (!dataset.empty() &&
      (dataset == "." || dataset == ".." || dataset.find_first_of("/\\:") != std::string::npos))
    throw std::invalid_argument("invalid mirror dataset name");
  const auto verify = [&](const fs::path& path) {
    if ((bytes && fs::file_size(path) != bytes) ||
        file_digest(path.string(), digest.size() == 64 ? "sha256" : "md5") != digest)
      throw std::runtime_error("source fingerprint mismatch: " + path.string());
  };
  if (fs::exists(destination)) {
    verify(destination);
    return;
  }
  if (const char* off = std::getenv("DATAMUNGE_DATA_OFFLINE"); off && std::string(off) == "1")
    throw std::runtime_error("offline dataset cache miss: " + destination);
  const auto target = fs::absolute(destination);
  fs::create_directories(target.parent_path());
  detail::Temporary stage(target.parent_path());
  std::string       source = url;
  if (const char* mirror = std::getenv("DATAMUNGE_DATA_MIRROR");
      mirror && *mirror && !dataset.empty())
    source = std::string(mirror) + "/" + dataset + "/" + target.filename().string();
  download_file(source, (stage.path / "download").string());
  verify(stage.path / "download");
  if (fs::exists(target)) {
    verify(target);
    return;
  }
  fs::rename(stage.path / "download", target);
#else
  throw std::runtime_error("native data adapters require DATAMUNGE_ENABLE_DATA_ASSETS=ON");
#endif
}
void extract_archive(
    const std::string&              source,
    const std::string&              directory,
    const std::vector<std::string>& members,
    std::uint64_t                   max_bytes) {
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
  std::set<std::string> wanted(members.begin(), members.end()), seen;
  if (wanted.size() != members.size())
    throw std::invalid_argument("duplicate requested member");
  for (const auto& s : wanted)
    safe_path(s);
  const auto root = fs::absolute(directory);
  no_symlinks(root);
  fs::create_directories(root);
  detail::Temporary                                      stage(root);
  std::unique_ptr<archive, decltype(&archive_read_free)> reader(
      archive_read_new(), archive_read_free);
  if (!reader)
    throw std::runtime_error("archive allocation failed");
  archive_read_support_filter_all(reader.get());
  archive_read_support_format_all(reader.get());
  if (archive_read_open_filename(reader.get(), source.c_str(), 65536) != ARCHIVE_OK)
    throw std::runtime_error("cannot open archive: " + source);
  archive_entry* entry;
  std::uint64_t  total = 0;
  int            status;
  while ((status = archive_read_next_header(reader.get(), &entry)) != ARCHIVE_EOF) {
    if (status != ARCHIVE_OK)
      throw std::runtime_error("invalid archive header: " + source);
    const char* raw = archive_entry_pathname(entry);
    if (!raw)
      throw std::invalid_argument("archive entry has no name");
    const std::string name(raw);
    safe_path(name);
    if (!wanted.empty() && !wanted.count(name)) {
      archive_read_data_skip(reader.get());
      continue;
    }
    if (archive_entry_symlink(entry) || archive_entry_hardlink(entry))
      throw std::invalid_argument("archive link: " + name);
    if (archive_entry_filetype(entry) == AE_IFDIR)
      continue;
    if (archive_entry_filetype(entry) != AE_IFREG || !seen.insert(name).second)
      throw std::invalid_argument("duplicate or nonregular archive member: " + name);
    const auto target = root / name;
    no_symlinks(target);
    fs::create_directories(target.parent_path());
    const auto              temp = stage.path / "member";
    std::ofstream           out(temp, std::ios::binary);
    std::array<char, 65536> block;
    la_ssize_t              n;
    while ((n = archive_read_data(reader.get(), block.data(), block.size())) > 0) {
      if (std::uint64_t(n) > max_bytes - total)
        throw std::runtime_error("archive expansion limit exceeded");
      total += n;
      out.write(block.data(), n);
    }
    if (n < 0)
      throw std::runtime_error("archive decode/checksum failed: " + name);
    out.close();
    if (!out)
      throw std::runtime_error("archive output failed: " + target.string());
    fs::rename(temp, target);
  }
  for (const auto& name : wanted)
    if (!seen.count(name))
      throw std::runtime_error("missing archive member: " + name);
#else
  throw std::runtime_error("native data adapters require DATAMUNGE_ENABLE_DATA_ASSETS=ON");
#endif
}
#ifndef DATAMUNGE_ENABLE_DATA_ASSETS
std::string prepare_example_data(const std::string&, const std::string&) {
  throw std::runtime_error("native data adapters require DATAMUNGE_ENABLE_DATA_ASSETS=ON");
}
#endif
} // namespace datamunge::datasets
