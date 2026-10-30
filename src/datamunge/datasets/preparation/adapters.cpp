/// @file adapters.cpp
/// @brief Original-source acquisition and deterministic metadata adapters for executable reports.
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include <archive.h>
#include <archive_entry.h>

#include <bit>
#include <cmath>
#include <cstring>
#include <datamunge/datasets/download.hpp>
#include <datamunge/image/detail/zlib_codec.hpp>
#include <datamunge/nn/audio.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <regex>
#include <set>

#include "preparation.hpp"
#include "protocols.hpp"
#include "support.hpp"
#ifdef _WIN32
#include <process.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
/// @brief Environment inherited only by directly spawned codec processes.
extern char** environ;
#endif

namespace datamunge::datasets::detail {
void run(const std::vector<std::string>& args) {
  if (args.empty())
    throw std::invalid_argument("empty process arguments");
  std::vector<char*> argv;
  for (const auto& a : args)
    argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
#ifdef _WIN32
  const auto code = _spawnvp(_P_WAIT, argv[0], argv.data());
  if (code != 0)
    throw std::runtime_error("native codec failed: " + args[0]);
#else
  pid_t     pid;
  const int error = posix_spawnp(&pid, argv[0], nullptr, nullptr, argv.data(), environ);
  if (error)
    throw std::runtime_error("cannot launch " + args[0] + ": " + std::strerror(error));
  int status;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR)
      throw std::runtime_error("cannot wait for codec");
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    throw std::runtime_error("native codec failed: " + args[0]);
#endif
}
/// @brief Private source ingress and tensor preparation helpers.
namespace {
/// @brief Serialize processes preparing the same cache; flock releases after crashes.
struct CacheLock {
#ifdef _WIN32
  HANDLE handle = INVALID_HANDLE_VALUE; ///< Owned Windows lock handle.
#else
  int fd = -1; ///< Owned lock descriptor.
#endif
  /// @brief Acquire the cache's native preparation lock.
  /// @param root Existing cache root.
  explicit CacheLock(const fs::path& root) {
#ifdef _WIN32
    handle = CreateFileW(
        (root / ".native-data.lock").c_str(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr,
        OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (handle == INVALID_HANDLE_VALUE)
      throw std::runtime_error("cannot open dataset lock");
    OVERLAPPED region{};
    if (!LockFileEx(handle, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &region)) {
      CloseHandle(handle);
      handle = INVALID_HANDLE_VALUE;
      throw std::runtime_error("cannot lock dataset");
    }
#else
    fd = ::open((root / ".native-data.lock").c_str(), O_CREAT | O_RDWR, 0600);
    if (fd < 0)
      throw std::runtime_error("cannot open dataset lock");
    while (flock(fd, LOCK_EX) < 0)
      if (errno != EINTR) {
        ::close(fd);
        fd = -1;
        throw std::runtime_error("cannot lock dataset");
      }
#endif
  }
  /// @brief Cache lock ownership is unique.
  /// @param other Unused source because copying is prohibited.
  CacheLock(const CacheLock& other) = delete;
  /// @brief Release the process lock.
  ~CacheLock() {
#ifdef _WIN32
    if (handle != INVALID_HANDLE_VALUE)
      CloseHandle(handle);
#else
    if (fd >= 0)
      ::close(fd);
#endif
  }
};
/// @brief Invoke a compiled tensor preparer with exact owning argument strings.
/// @param fn Native adapter entrypoint.
/// @param paths Ordered directory and optional mode arguments.
void call(int (*fn)(int, char**), std::vector<std::string> paths) {
  paths.insert(paths.begin(), "native-data");
  std::vector<char*> args;
  for (auto& p : paths)
    args.push_back(p.data());
  args.push_back(nullptr);
  if (fn(int(paths.size()), args.data()) != 0)
    throw std::runtime_error("native tensor preparation failed in " + paths[1]);
}
/// @brief Find regular archive member names without expanding their contents.
/// @param path Archive path.
/// @return Sorted exact member names.
std::vector<std::string> listing(const fs::path& path) {
  std::unique_ptr<archive, decltype(&archive_read_free)> a(archive_read_new(), archive_read_free);
  archive_read_support_filter_all(a.get());
  archive_read_support_format_all(a.get());
  if (archive_read_open_filename(a.get(), path.string().c_str(), 65536) != ARCHIVE_OK)
    throw std::runtime_error("cannot inventory " + path.string());
  archive_entry*           e;
  std::vector<std::string> names;
  int                      rc;
  while ((rc = archive_read_next_header(a.get(), &e)) != ARCHIVE_EOF) {
    if (rc != ARCHIVE_OK)
      throw std::runtime_error("invalid archive inventory");
    if (archive_entry_filetype(e) == AE_IFREG) {
      if (!archive_entry_pathname(e))
        throw std::runtime_error("unnamed member");
      names.emplace_back(archive_entry_pathname(e));
    }
    archive_read_data_skip(a.get());
  }
  std::sort(names.begin(), names.end());
  return names;
}
/// @brief Fetch one pinned source while retaining its original metadata in the manifest.
/// @param root Cache directory.
/// @param dataset Mirror subdirectory.
/// @param name Local source filename.
/// @param url Original URL.
/// @param hash Pinned source digest.
/// @param size Expected size, zero if not specified upstream.
void source(
    const fs::path&    root,
    const std::string& dataset,
    const std::string& name,
    const std::string& url,
    const std::string& hash,
    std::uint64_t      size = 0) {
  fetch_file(url, (root / name).string(), hash, size, dataset);
}
/// @brief Start a manifest with preserved source protocol text.
/// @param name Former adapter protocol key.
/// @return Owned static protocol metadata.
Json manifest(const std::string& name) {
  return protocols().at(name).at("manifest");
}
/// @brief Fetch each source from a former script's pinned SOURCES mapping.
/// @param root Destination cache.
/// @param protocol Protocol key.
void sources(const fs::path& root, const std::string& protocol) {
  for (auto it = protocols().at(protocol).at("constants").at("SOURCES").begin();
       it != protocols().at(protocol).at("constants").at("SOURCES").end();
       ++it) {
    const auto& s = it.value();
    if (s.size() == 2)
      source(root, protocol, it.key(), s[0], s[1]);
    else
      source(root, protocol, it.key(), s[0], s[2], s[1]);
  }
}
/// @brief Convert a source protocol map into a JSON source inventory.
/// @param protocol Adapter protocol key.
/// @return Source URLs, checksums and optional sizes.
Json source_manifest(const std::string& protocol) {
  Json out = Json::object();
  for (auto it = protocols().at(protocol).at("constants").at("SOURCES").begin();
       it != protocols().at(protocol).at("constants").at("SOURCES").end();
       ++it) {
    const auto& s = it.value();
    out[it.key()] = {{"url", s[0]}, {"sha256", s.size() == 2 ? s[1] : s[2]}};
    if (s.size() > 2)
      out[it.key()]["bytes"] = s[1];
    if (s.size() > 3)
      out[it.key()]["member"] = s[3];
  }
  return out;
}
/// @brief Format a source identity with minimum decimal width.
/// @param value Nonnegative source ID.
/// @param width Zero-padding width.
/// @return Decimal ID string.
std::string padded(int value, int width) {
  std::ostringstream s;
  s << std::setw(width) << std::setfill('0') << value;
  return s.str();
}
/// @brief Read newline-delimited metadata, retaining original byte strings.
/// @param data Original bytes.
/// @return Lines excluding newline separators.
std::vector<std::string> lines(const std::string& data) {
  std::istringstream       in(data);
  std::vector<std::string> out;
  std::string              s;
  while (std::getline(in, s)) {
    if (!s.empty() && s.back() == '\r')
      s.pop_back();
    out.push_back(s);
  }
  return out;
}
/// @brief Split a metadata line at an exact delimiter.
/// @param text Input string.
/// @param delimiter Separator byte.
/// @return Fields, including empty fields.
std::vector<std::string> split(const std::string& text, char delimiter) {
  std::vector<std::string> out;
  std::size_t              at = 0, end;
  while ((end = text.find(delimiter, at)) != std::string::npos) {
    out.push_back(text.substr(at, end - at));
    at = end + 1;
  }
  out.push_back(text.substr(at));
  return out;
}
/// @brief ASCII lowercase used by the original annotation policies.
/// @param s Owned metadata string.
/// @return ASCII-normalized string, with all other bytes retained.
std::string lower(std::string s) {
  for (char& c : s)
    if (c >= 'A' && c <= 'Z')
      c += 32;
  return s;
}
/// @brief Check an ingress invariant without synthesizing missing metadata.
/// @param condition Required invariant.
/// @param message Failure explanation.
void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

/// @brief Prepare original compressed MedMNIST arrays through native NPZ ingress.
/// @param root Dataset cache.
void medical(const fs::path& root) {
  auto j = manifest("medical");
  source(root, "medical_volume", "nodulemnist3d.npz", j["url"], j["sha256"], j["bytes"]);
  call(prepare_medical, {root.string()});
  save(root / "manifest.json", j);
}

/// @brief Preserve COCO128 image identities, category sets and seed271 partitions.
/// @param root Dataset cache.
void multilabel(const fs::path& root) {
  const auto c = protocols()["multilabel"]["constants"];
  source(root, "multilabel", "coco128.zip", c["SOURCE"], c["SHA256"]);
  extract_archive((root / "coco128.zip").string(), root.string());
  const auto                   names = listing(root / "coco128.zip");
  std::map<int, std::set<int>> cats;
  Json                         missing = Json::array(), orphan = Json::array();
  for (const auto& name : names)
    if (name.starts_with("coco128/images/train2017/") && name.ends_with(".jpg")) {
      const auto stem = fs::path(name).stem().string();
      require(
          stem.size() == 12 && stem.find_first_not_of("0123456789") == std::string::npos,
          "invalid COCO identity");
      int id = std::stoi(stem);
      require(!cats.count(id), "duplicate COCO image identity");
      auto&      categories = cats[id];
      const auto label      = root / "coco128/labels/train2017" / (stem + ".txt");
      if (!fs::exists(label)) {
        missing.push_back(id);
        continue;
      }
      for (const auto& line : lines(read(label))) {
        std::istringstream row(line);
        int                category;
        double             box[4];
        if (!(row >> category))
          continue;
        require(
            bool(row >> box[0] >> box[1] >> box[2] >> box[3]) && category >= 0 && category < 80,
            "invalid COCO label");
        std::string extra;
        require(!(row >> extra), "extra COCO label fields");
        categories.insert(category);
      }
    }
  require(cats.size() == 128, "expected128 original COCO images");
  for (const auto& name : names)
    if (name.starts_with("coco128/labels/train2017/") && name.ends_with(".txt")) {
      const auto stem = fs::path(name).stem().string();
      if (stem.find_first_not_of("0123456789") == std::string::npos && !cats.count(std::stoi(stem)))
        orphan.push_back(std::stoi(stem));
    }
  std::vector<int> ids;
  for (const auto& [id, _] : cats)
    ids.push_back(id);
  MetadataRandom(271).shuffle(ids);
  auto j                             = manifest("multilabel");
  j["rows"]                          = Json::array();
  j["missing_annotation_ids"]        = missing;
  j["annotation_ids_without_images"] = orphan;
  std::ostringstream rows;
  rows << std::setprecision(17);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const auto       s  = i < 80 ? "fit" : i < 104 ? "validation" : "test";
    int              id = ids[i];
    std::vector<int> labels(cats[id].begin(), cats[id].end());
    rows << s << '\t' << id << '\t' << labels.size();
    for (int l : labels)
      rows << '\t' << l;
    rows << '\n';
    j["rows"].push_back({{"split", s}, {"id", id}, {"categories", labels}});
  }
  write(root / "prepared_rows.tsv", rows.str());
  call(prepare_multilabel, {root.string()});
  save(root / "manifest.json", j);
}

/// @brief Decode original ModelNet OFF topology and preserve its lexical object selection.
/// @param root Dataset cache.
void mesh(const fs::path& root) {
  const auto        c   = protocols()["meshes"]["constants"];
  const std::string url = "https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip";
  source(root, "modelnet", "ModelNet10.zip", url, c["SHA"], 473402300);
  const auto               names = listing(root / "ModelNet10.zip");
  std::vector<std::string> selected;
  Json                     records = Json::array();
  std::ostringstream       rows;
  rows << std::setprecision(17);
  for (int label = 0; label < 3; ++label) {
    const std::string        kind = c["CLASSES"][label];
    std::vector<std::string> train, test;
    for (const auto& name : names)
      if (name.ends_with(".off")) {
        if (name.starts_with("ModelNet10/" + kind + "/train/"))
          train.push_back(name);
        if (name.starts_with("ModelNet10/" + kind + "/test/"))
          test.push_back(name);
      }
    require(train.size() >= 10 && test.size() >= 4, "missing original ModelNet objects");
    for (int i = 0; i < 14; ++i) {
      const auto name = i < 10 ? train[i] : test[i - 10];
      const auto s    = i < 8 ? "fit" : i < 10 ? "validation" : "test";
      selected.push_back(name);
      records.push_back(
          {{"source_id", records.size()},
           {"split", s},
           {"label", label},
           {"class", kind},
           {"member", name}});
    }
  }
  Temporary stage(root);
  extract_archive((root / "ModelNet10.zip").string(), stage.path.string(), selected);
  for (auto& r : records) {
    const std::string member = r["member"];
    const auto        bytes  = read(stage.path / member);
    auto              text   = bytes;
    require(text.starts_with("OFF"), "expected original OFF mesh");
    text.insert(3, " ");
    std::istringstream in(text);
    std::string        magic;
    int                vertices, faces, edges;
    require(
        bool(in >> magic >> vertices >> faces >> edges) && magic == "OFF" && vertices > 0 &&
            faces > 0,
        "invalid OFF header");
    std::vector<std::array<std::string, 3>> coords(vertices);
    for (auto& xyz : coords)
      require(bool(in >> xyz[0] >> xyz[1] >> xyz[2]), "truncated OFF coordinates");
    std::vector<std::array<int, 3>> triangles;
    for (int f = 0; f < faces; ++f) {
      int count;
      require(bool(in >> count) && count >= 3 && count <= vertices, "invalid OFF polygon");
      std::vector<int> ids(count);
      for (auto& id : ids)
        require(bool(in >> id) && id >= 0 && id < vertices, "invalid OFF index");
      for (int k = 1; k < count - 1; ++k)
        triangles.push_back({ids[0], ids[k], ids[k + 1]});
    }
    int                id      = r["source_id"];
    const auto         ingress = "meshes/" + std::to_string(id) + ".tsv";
    std::ostringstream out;
    out << vertices << ' ' << triangles.size() << '\n';
    for (const auto& v : coords)
      out << v[0] << ' ' << v[1] << ' ' << v[2] << '\n';
    for (const auto& t : triangles)
      out << t[0] << ' ' << t[1] << ' ' << t[2] << '\n';
    write(root / ingress, out.str());
    write(root / "meshes" / fs::path(member).filename(), bytes);
    r["sha256"]        = file_digest((stage.path / member).string());
    r["vertices"]      = vertices;
    r["faces"]         = faces;
    r["fan_triangles"] = triangles.size();
    r["ingress"]       = ingress;
    rows << r["split"].get<std::string>() << '\t' << id << '\t' << r["label"].get<int>() << '\t'
         << ingress << '\n';
  }
  write(root / "mesh_rows.tsv", rows.str());
  call(prepare_mesh, {root.string()});
  auto j       = manifest("meshes");
  j["records"] = records;
  save(root / "manifest.json", j);
}

/// @brief Extract genuine FSDD audio with the original speaker and take boundaries.
/// @param root Dataset cache.
void audio(const fs::path& root) {
  auto       j = manifest("audio");
  const auto c = protocols()["audio"]["constants"];
  source(root, "wavenet", "fsdd-v1.0.10.zip", j["url"], c["SHA"], c["SIZE"]);
  const std::vector<std::string> speakers = {
      "george", "jackson", "lucas", "nicolas", "theo", "yweweler"};
  std::vector<std::string> members;
  Json                     records = Json::array();
  for (int s = 0; s < 6; ++s)
    for (int digit = 0; digit < 3; ++digit)
      for (int take = 0; take < ((s == 0 || s == 1 || s == 5) ? 2 : 1); ++take) {
        const auto name = std::to_string(digit) + "_" + speakers[s] + "_" + std::to_string(take) +
                          ".wav",
                   member = "free-spoken-digit-dataset-1.0.10/recordings/" + name;
        members.push_back(member);
        records.push_back(
            {{"split",
              s < 4    ? "fit"
              : s == 4 ? "validation"
                       : "test"},
             {"source_id", records.size()},
             {"speaker_id", s},
             {"speaker", speakers[s]},
             {"digit", digit},
             {"take", take},
             {"member", member},
             {"path", "recordings/" + name}});
      }
  Temporary stage(root);
  extract_archive((root / "fsdd-v1.0.10.zip").string(), stage.path.string(), members);
  std::ostringstream rows;
  rows << std::setprecision(17);
  for (auto& r : records) {
    const std::string member = r["member"], path = r["path"];
    write(root / path, read(stage.path / member));
    const auto wav = datamunge::nn::read_wav((root / path).string());
    require(
        wav.sample_rate == 8000 && wav.samples.size(1) == 1 && wav.samples.size(0) >= 1025,
        "invalid FSDD WAV geometry");
    r["source_frames"] = wav.samples.size(0);
    r["sha256"]        = file_digest((root / path).string());
    rows << r["split"].get<std::string>() << '\t' << r["source_id"].get<int>() << '\t'
         << r["speaker_id"].get<int>() << '\t' << path << '\n';
  }
  write(root / "source_files.tsv", rows.str());
  call(prepare_audio, {root.string()});
  j["records"] = records;
  save(root / "manifest.json", j);
}

/// @brief Preserve UCR/UEA source text geometry and Python-compatible seed407 row splits.
/// @param base Parent of the individual series caches.
void signals(const fs::path& base) {
  const auto specs = protocols()["signals"]["constants"]["SOURCES"];
  for (auto it = specs.begin(); it != specs.end(); ++it) {
    const auto        name    = it.key();
    const auto        s       = it.value();
    const std::string folder  = s[0];
    const auto        archive = base / folder / (name + ".zip"),
               root           = name == "GunPoint" ? base / folder : base / folder / name;
    fs::create_directories(root);
    const int time = s[3], channels = s[4], training = s[5];
    source(
        archive.parent_path(),
        folder,
        name + ".zip",
        "https://www.timeseriesclassification.com/aeon-toolkit/" + name + ".zip",
        s[2],
        s[1]);
    Temporary stage(root);
    extract_archive(archive.string(), stage.path.string(), {name + "_TRAIN.ts", name + "_TEST.ts"});
    Json                                  classes = Json::array();
    std::vector<std::vector<std::string>> values;
    std::vector<int>                      labels;
    for (const std::string part : {"TRAIN", "TEST"}) {
      bool active = false;
      int  count  = 0;
      for (auto line : lines(read(stage.path / (name + "_" + part + ".ts")))) {
        if (line.starts_with("\xef\xbb\xbf"))
          line.erase(0, 3);
        const auto first = line.find_first_not_of(" \t");
        if (first == std::string::npos)
          continue;
        line            = line.substr(first);
        const auto last = line.find_last_not_of(" \t");
        line.resize(last + 1);
        if (lower(line).starts_with("@classlabel")) {
          std::istringstream input(line);
          std::string        field;
          input >> field >> field;
          Json names = Json::array();
          while (input >> field)
            names.push_back(field);
          if (classes.empty())
            classes = names;
          else
            require(classes == names, "train/test class order differs");
        }
        if (lower(line) == "@data") {
          active = true;
          continue;
        }
        if (!active || line[0] == '#')
          continue;
        const auto parts = split(line, ':');
        require(int(parts.size()) == channels + 1, "signal channel geometry differs");
        std::vector<std::vector<std::string>> channel;
        for (int c = 0; c < channels; ++c) {
          channel.push_back(split(parts[c], ','));
          require(int(channel.back().size()) == time, "signal time geometry differs");
        }
        int label = -1;
        for (std::size_t l = 0; l < classes.size(); ++l)
          if (classes[l] == parts.back())
            label = int(l);
        require(label >= 0, "unknown original signal class");
        labels.push_back(label);
        std::vector<std::string> row;
        for (int t = 0; t < time; ++t)
          for (int c = 0; c < channels; ++c)
            row.push_back(channel[c][t]);
        values.push_back(std::move(row));
        ++count;
      }
      if (part == "TRAIN")
        require(count == training, "official signal training count differs");
    }
    require(
        int(values.size()) == (name == "GunPoint" ? 200 : training * 2),
        "official signal total count differs");
    MetadataRandom   rng(407);
    std::vector<int> fit, val, test;
    for (int l = 0; l < int(classes.size()); ++l) {
      std::vector<int> group;
      for (int i = 0; i < training; ++i)
        if (labels[i] == l)
          group.push_back(i);
      rng.shuffle(group);
      int held = std::max(1, int((group.size() + 2) / 5));
      val.insert(val.end(), group.begin(), group.begin() + held);
      fit.insert(fit.end(), group.begin() + held, group.end());
    }
    rng.shuffle(fit);
    rng.shuffle(val);
    for (int i = training; i < int(values.size()); ++i)
      test.push_back(i);
    Json               partitions = {{"fit", fit}, {"validation", val}, {"test", test}};
    std::ostringstream rows;
    rows << std::setprecision(17);
    for (auto p = partitions.begin(); p != partitions.end(); ++p)
      for (int id : p.value().get<std::vector<int>>()) {
        rows << p.key() << '\t' << id << '\t' << labels[id];
        for (const auto& value : values[id])
          rows << '\t' << value;
        rows << '\n';
      }
    write(root / "raw_signals.tsv", rows.str());
    write(
        root / "config.tsv",
        std::to_string(time) + "\t" + std::to_string(channels) + "\t" +
            std::to_string(classes.size()) + "\t" + std::to_string(training) + "\n");
    call(prepare_signals, {root.string(), name == "GunPoint" ? "time_series.npz" : "signals.npz"});
    auto j              = manifest("signals");
    j["name"]           = name;
    j["url"]            = "https://www.timeseriesclassification.com/aeon-toolkit/" + name + ".zip";
    j["archive_bytes"]  = s[1];
    j["archive_sha256"] = s[2];
    j["class_names"]    = classes;
    j["dimensions"]     = {time, channels};
    j["source_rows"]    = partitions;
    j["source_members"] = {name + "_TEST.ts", name + "_TRAIN.ts"};
    save(root / "manifest.json", j);
  }
}

/// @brief Decode little-endian MAT-v5 integer label and split arrays, including zlib elements.
/// @param path Original verified MATLAB metadata.
/// @return Matrix names mapped to integral values.
std::map<std::string, std::vector<int>> mat_integers(const fs::path& path) {
  const auto bytes = read(path);
  require(
      bytes.size() >= 128 && bytes.substr(126, 2) == "IM",
      "expected little-endian MAT-v5 metadata");
  using Element       = std::pair<unsigned, std::string>;
  const auto elements = [](const std::string& data, std::size_t at) {
    std::vector<Element> out;
    const auto           u32 = [&](std::size_t p) {
      require(p + 4 <= data.size(), "truncated MAT word");
      std::uint32_t n = 0;
      for (int i = 0; i < 4; ++i)
        n |= std::uint32_t((unsigned char)data[p + i]) << (8 * i);
      return n;
    };
    while (at + 8 <= data.size()) {
      const auto tag = u32(at), size = u32(at + 4);
      if (tag >> 16) {
        require((tag >> 16) <= 4, "invalid compact MAT tag");
        out.emplace_back(tag & 65535, data.substr(at + 4, tag >> 16));
        at += 8;
      } else {
        require(size <= data.size() - at - 8, "truncated MAT element");
        out.emplace_back(tag, data.substr(at + 8, size));
        at = tag == 15 ? at + 8 + size : at + 8 + ((std::size_t(size) + 7) / 8) * 8;
      }
    }
    require(at == data.size(), "trailing MAT metadata bytes");
    return out;
  };
  auto                                    pending = elements(bytes, 128);
  std::map<std::string, std::vector<int>> out;
  for (std::size_t at = 0; at < pending.size(); ++at) {
    auto [kind, data] = std::move(pending[at]);
    if (kind == 15) {
      auto raw = datamunge::image::detail::zlib_decompress(
          std::vector<unsigned char>(data.begin(), data.end()));
      auto next = elements(std::string(raw.begin(), raw.end()), 0);
      pending.insert(pending.end(), next.begin(), next.end());
      continue;
    }
    require(kind == 14, "unexpected MAT element");
    auto fields = elements(data, 0);
    require(fields.size() == 4 && fields[2].first == 1, "expected simple numeric MAT matrix");
    const auto                    type  = fields[3].first;
    const std::map<unsigned, int> sizes = {
        {1, 1}, {2, 1}, {3, 2}, {4, 2}, {5, 4}, {6, 4}, {7, 4}, {9, 8}, {12, 8}, {13, 8}};
    require(sizes.count(type), "unsupported MAT scalar");
    const int   width   = sizes.at(type);
    const auto& payload = fields[3].second;
    require(payload.size() % width == 0, "unaligned MAT payload");
    std::vector<int> values;
    for (std::size_t p = 0; p < payload.size(); p += width) {
      std::uint64_t bits = 0;
      for (int k = 0; k < width; ++k)
        bits |= std::uint64_t((unsigned char)payload[p + k]) << (8 * k);
      double value;
      if (type == 7)
        value = std::bit_cast<float>(std::uint32_t(bits));
      else if (type == 9)
        value = std::bit_cast<double>(bits);
      else if (type == 1)
        value = std::int8_t(bits);
      else if (type == 3)
        value = std::int16_t(bits);
      else if (type == 5)
        value = std::int32_t(bits);
      else if (type == 12)
        value = std::int64_t(bits);
      else
        value = double(bits);
      require(
          std::isfinite(value) && value >= std::numeric_limits<int>::min() &&
              value <= std::numeric_limits<int>::max() && value == std::floor(value),
          "noninteger MAT metadata");
      values.push_back(int(value));
    }
    require(out.emplace(fields[2].second, std::move(values)).second, "duplicate MAT matrix");
  }
  return out;
}

/// @brief Select the same six official Flowers102 classes from disjoint official partitions.
/// @param root Dataset cache.
void flowers(const fs::path& root) {
  const auto        c      = protocols()["flowers"]["constants"];
  const std::string base   = c["BASE"];
  const std::string mirror = "https://torch-cdn.mlverse.org/datasets/oxford_flowers102/";
  for (const std::string name : {"imagelabels.mat", "setid.mat", "102flowers.tgz"}) {
    try {
      source(root, "flowers102", name, mirror + name, c["FILES"][name]);
    } catch (const std::runtime_error&) {
      const char* configured_mirror  = std::getenv("DATAMUNGE_DATA_MIRROR");
      const char* configured_offline = std::getenv("DATAMUNGE_DATA_OFFLINE");
      if (fs::exists(root / name) || (configured_mirror && *configured_mirror) ||
          (configured_offline && std::string(configured_offline) == "1"))
        throw;
      source(root, "flowers102", name, base + name, c["FILES"][name]);
    }
  }
  const auto labels = mat_integers(root / "imagelabels.mat").at("labels");
  const auto parts  = mat_integers(root / "setid.mat");
  require(labels.size() == 8189, "official flower class count differs");
  std::set<int> all;
  for (const auto& key : {"trnid", "valid", "tstid"})
    for (int id : parts.at(key))
      require(
          id >= 1 && id <= 8189 && all.insert(id).second,
          "official flower IDs must partition sources");
  require(all.size() == 8189, "incomplete flower partitions");
  Json               records = Json::array();
  std::ostringstream rows;
  rows << std::setprecision(17);
  std::vector<std::string> members;
  for (const auto& [name, key] : std::vector<std::pair<std::string, std::string>>{
           {"fit", "trnid"}, {"validation", "valid"}, {"test", "tstid"}})
    for (int category = 1; category <= 6; ++category) {
      std::vector<int> ids;
      for (int id : parts.at(key))
        if (labels.at(id - 1) == category)
          ids.push_back(id);
      std::sort(ids.begin(), ids.end());
      require(ids.size() >= 8, "insufficient official flower category rows");
      for (int k = 0; k < 8; ++k) {
        int id = ids[k];
        rows << name << '\t' << id << '\t' << category - 1 << '\n';
        records.push_back({{"split", name}, {"id", id}, {"label", category - 1}});
        members.push_back("jpg/image_" + padded(id, 5) + ".jpg");
      }
    }
  write(root / "prepared_rows.tsv", rows.str());
  extract_archive((root / "102flowers.tgz").string(), root.string(), members);
  call(prepare_flowers, {root.string()});
  auto j               = manifest("flowers");
  j["download_mirror"] = mirror;
  j["rows"]            = records;
  save(root / "manifest.json", j);
}

/// @brief Preserve original pose images, person crops and visibility/mirroring provenance.
/// @param root Dataset cache.
void pose(const fs::path& root) {
  const auto c = protocols()["pose"]["constants"];
  source(root, "human_pose", "coco8-pose.zip", c["URL"], c["SHA256"], c["SIZE"]);
  extract_archive((root / "coco8-pose.zip").string(), root.string());
  const auto         names   = listing(root / "coco8-pose.zip");
  Json               records = Json::array();
  std::ostringstream rows;
  rows << std::setprecision(17);
  for (const std::string part : {"train", "val"})
    for (const auto& name : names)
      if (name.starts_with("coco8-pose/images/" + part + "/") && name.ends_with(".jpg")) {
        const auto        stem  = fs::path(name).stem().string();
        int               id    = std::stoi(stem);
        const std::string split = part == "val" ? "test"
                                  : id == 86    ? "validation"
                                                : "fit",
                          label = "coco8-pose/labels/" + part + "/" + stem + ".txt";
        rows << split << '\t' << id << '\t' << name << '\t' << label << '\n';
        records.push_back({{"split", split}, {"image_id", id}, {"image", name}, {"label", label}});
      }
  require(records.size() == 8, "expected8 original pose images");
  write(root / "source_files.tsv", rows.str());
  call(prepare_pose, {root.string()});
  auto j      = manifest("pose");
  j["images"] = records;
  j["crops"]  = Json::array();
  std::istringstream in(read(root / "crops.tsv"));
  std::string        split;
  int                image, person, l, t, r, b, mirror;
  while (in >> split >> image >> person >> l >> t >> r >> b >> mirror)
    j["crops"].push_back(
        {{"split", split},
         {"image_id", image},
         {"person_id", person},
         {"bounds_xyxy", {l, t, r, b}},
         {"mirrored", bool(mirror)}});
  require(in.eof(), "invalid pose crop manifest");
  save(root / "manifest.json", j);
}

/// @brief Extract published Middlebury images and flow, preserving scene-family partitions.
/// @param root Dataset cache.
void optical_flow(const fs::path& root) {
  sources(root, "optical_flow");
  const auto               scenes = protocols()["optical_flow"]["constants"]["SCENES"];
  std::vector<std::string> rgb, gt;
  for (const auto& s : scenes) {
    const std::string scene = s[0];
    for (const std::string frame : {"frame10.png", "frame11.png"})
      rgb.push_back("other-data/" + scene + "/" + frame);
    gt.push_back("other-gt-flow/" + scene + "/flow10.flo");
  }
  Temporary stage(root);
  extract_archive((root / "other-color-twoframes.zip").string(), stage.path.string(), rgb);
  extract_archive((root / "other-gt-flow.zip").string(), stage.path.string(), gt);
  std::ostringstream rows;
  rows << std::setprecision(17);
  for (const auto& s : scenes) {
    const std::string scene = s[0];
    for (const auto& f : {"frame10.png", "frame11.png"})
      write(root / "scenes" / scene / f, read(stage.path / "other-data" / scene / f));
    write(
        root / "scenes" / scene / "flow10.flo",
        read(stage.path / "other-gt-flow" / scene / "flow10.flo"));
    rows << s[1].get<std::string>() << '\t' << scene << '\t' << s[2].get<int>() << '\t'
         << s[3].get<int>() << '\n';
  }
  write(root / "scene_rows.tsv", rows.str());
  call(prepare_optical_flow, {root.string()});
  auto j       = manifest("optical_flow");
  j["sources"] = source_manifest("optical_flow");
  j["records"] = Json::array();
  std::istringstream in(read(root / "crop_rows.tsv"));
  std::string        split, scene;
  int                id, family, y, x, h, w;
  while (in >> split >> scene >> id >> family >> y >> x >> h >> w)
    j["records"].push_back(
        {{"split", split},
         {"scene", scene},
         {"scene_id", id},
         {"family_id", family},
         {"y", y},
         {"x", x},
         {"source_height", h},
         {"source_width", w},
         {"crop_side", 64}});
  require(in.eof(), "invalid flow crop metadata");
  save(root / "manifest.json", j);
}

/// @brief Prepare full frozen VGG convolution weights from original Keras HDF5 files.
/// @param root Dataset cache.
void style(const fs::path& root) {
  sources(root, "style");
  call(prepare_style, {root.string()});
  auto j       = manifest("style");
  j["sources"] = source_manifest("style");
  save(root / "manifest.json", j);
}

/// @brief Select original NYU canonical rooms with the historical seed137 shuffle.
/// @param root Dataset cache.
void depth(const fs::path& root) {
  const auto c = protocols()["depth"]["constants"];
  source(root, "monocular_depth", "nyu_depth_v2_labeled.mat", c["SOURCE"], c["SHA256"], c["SIZE"]);
  call(prepare_depth, {root.string(), "--inventory"});
  Json rooms = Json::array();
  for (const auto& line : lines(read(root / "room_rows.tsv"))) {
    auto f = split(line, '\t');
    require(f.size() == 2, "invalid room inventory");
    rooms.push_back({{"room", f[0]}, {"room_id", rooms.size()}, {"source_row", std::stoi(f[1])}});
  }
  require(rooms.size() == 464, "expected464 canonical rooms");
  auto records = rooms.get<std::vector<Json>>();
  MetadataRandom(137).shuffle(records);
  records.resize(56);
  std::ostringstream rows;
  rows << std::setprecision(17);
  for (int i = 0; i < 56; ++i) {
    records[i]["split"] = i < 32 ? "fit" : i < 40 ? "validation" : "test";
    const auto& r       = records[i];
    rows << r["split"].get<std::string>() << '\t' << r["room"].get<std::string>() << '\t'
         << r["room_id"].get<int>() << '\t' << r["source_row"].get<int>() << '\n';
  }
  write(root / "selected_rows.tsv", rows.str());
  call(prepare_depth, {root.string()});
  auto j       = manifest("depth");
  j["records"] = records;
  save(root / "manifest.json", j);
}

/// @brief Prepare human COCO-Text words with photograph-disjoint caps and native crops.
/// @param root Dataset cache.
/// @param coco Verified original COCO128 cache.
void ocr(const fs::path& root, const fs::path& coco) {
  const auto c = protocols()["ocr"]["constants"];
  source(root, "ocr", "cocotext.v2.zip", c["SOURCE"], c["SHA256"]);
  Temporary stage(root);
  extract_archive((root / "cocotext.v2.zip").string(), stage.path.string(), {"cocotext.v2.json"});
  const auto    annotations = json(stage.path / "cocotext.v2.json");
  std::set<int> available;
  for (const auto& p : fs::directory_iterator(coco / "coco128/images/train2017"))
    if (p.path().extension() == ".jpg")
      available.insert(std::stoi(p.path().stem().string()));
  std::map<int, std::vector<Json>> groups;
  int                              total = 0;
  for (const auto& a : annotations["anns"]) {
    const auto text = lower(a.value("utf8_string", std::string{}));
    int        id   = a["image_id"];
    if (available.count(id) && a.value("legibility", std::string{}) == "legible" &&
        text.size() >= 1 && text.size() <= 12 &&
        text.find_first_not_of("0123456789abcdefghijklmnopqrstuvwxyz") == std::string::npos) {
      groups[id].push_back(a);
      ++total;
    }
  }
  Json               records = Json::array(), discarded = Json::array();
  auto               at = groups.begin();
  std::ostringstream rows;
  rows << std::setprecision(17);
  for (const auto& [name, cap] :
       std::vector<std::pair<std::string, int>>{{"fit", 43}, {"validation", 7}, {"test", 6}}) {
    int selected = 0;
    while (selected < cap) {
      require(at != groups.end(), "insufficient original COCO-Text words");
      const int id    = at->first;
      auto      words = at->second;
      std::sort(words.begin(), words.end(), [](const Json& a, const Json& b) {
        return a["id"].get<int>() < b["id"].get<int>();
      });
      for (const auto& a : words) {
        if (selected == cap) {
          discarded.push_back(a["id"]);
          continue;
        }
        const auto im   = annotations["imgs"].at(std::to_string(id));
        const auto text = lower(a["utf8_string"]);
        Json       r    = {
            {"split", name},
            {"image_id", id},
            {"annotation_id", a["id"]},
            {"text", text},
            {"original_text", a["utf8_string"]},
            {"bbox", a["bbox"]},
            {"image_width", im["width"]},
            {"image_height", im["height"]},
            {"annotation_source_split", im["set"]}};
        records.push_back(r);
        rows << name << '\t' << id << '\t' << a["id"].get<int>() << '\t' << text;
        for (const auto& b : a["bbox"])
          rows << '\t' << b.get<double>();
        rows << '\t' << im["width"].get<int>() << '\t' << im["height"].get<int>() << '\n';
        ++selected;
      }
      ++at;
    }
  }
  write(root / "prepared_rows.tsv", rows.str());
  call(prepare_ocr, {root.string(), (coco / "coco128/images/train2017").string()});
  auto j                        = manifest("ocr");
  j["available_words"]          = total;
  j["available_photographs"]    = groups.size();
  j["discarded_annotation_ids"] = discarded;
  j["records"]                  = records;
  save(root / "manifest.json", j);
}

/// @brief Normalize released human answer strings using the declared article/number policy.
/// @param input Original annotation text.
/// @return Space-collapsed ASCII lowercase normalized answer.
std::string answer(const std::string& input) {
  static const std::map<std::string, std::string> numbers = {
      {"none", "0"},
      {"zero", "0"},
      {"one", "1"},
      {"two", "2"},
      {"three", "3"},
      {"four", "4"},
      {"five", "5"},
      {"six", "6"},
      {"seven", "7"},
      {"eight", "8"},
      {"nine", "9"},
      {"ten", "10"}};
  std::istringstream in(lower(input));
  std::string        w, out;
  while (in >> w) {
    if (w == "a" || w == "an" || w == "the")
      continue;
    if (!out.empty())
      out += ' ';
    out += numbers.count(w) ? numbers.at(w) : w;
  }
  return out;
}
/// @brief Rank fitting-only token counts with lexical tie breaking.
/// @param counts Fitting frequency map.
/// @param initial Reserved vocabulary entries.
/// @param limit Maximum final vocabulary size.
/// @return Ordered vocabulary including reserved entries.
std::vector<std::string> vocabulary(
    const std::map<std::string, int>& counts,
    std::vector<std::string>          initial,
    std::size_t                       limit) {
  std::vector<std::pair<std::string, int>> ranked(counts.begin(), counts.end());
  std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
    return a.second != b.second ? a.second > b.second : a.first < b.first;
  });
  for (const auto& [w, _] : ranked) {
    if (initial.size() == limit)
      break;
    initial.push_back(w);
  }
  return initial;
}
/// @brief Align genuine VQA-v2 human questions and all ten answers to source COCO images.
/// @param root Dataset cache.
/// @param coco Verified original COCO128 cache.
void vqa(const fs::path& root, const fs::path& coco) {
  sources(root, "vqa");
  std::map<int, fs::path> images;
  for (const auto& p : fs::directory_iterator(coco / "coco128/images/train2017"))
    if (p.path().extension() == ".jpg")
      images[std::stoi(p.path().stem().string())] = p.path();
  require(images.size() == 128, "expected128 original COCO photos");
  std::map<std::int64_t, Json> questions, annotations;
  Temporary                    stage(root);
  for (const auto& [name, values] : protocols()["vqa"]["constants"]["SOURCES"].items()) {
    const std::string member = values[3], key = values[4];
    extract_archive((root / name).string(), stage.path.string(), {member});
    const auto document = json(stage.path / member);
    auto&      target   = key == "questions" ? questions : annotations;
    for (const auto& r : document[key])
      if (images.count(r["image_id"].get<int>()))
        require(
            target.emplace(r["question_id"].get<std::int64_t>(), r).second,
            "duplicate VQA question identity");
  }
  require(questions.size() == annotations.size(), "VQA question/answer identities differ");
  std::set<int> distinct;
  for (const auto& [_, q] : questions)
    distinct.insert(q["image_id"].get<int>());
  std::vector<int> ids(distinct.begin(), distinct.end());
  require(ids.size() == 85, "expected85 matched original VQA images");
  MetadataRandom(271).shuffle(ids);
  Json                       pools = Json::object();
  std::map<int, std::string> split_of;
  for (int i = 0; i < 85; ++i) {
    const std::string s = i < 55 ? "fit" : i < 70 ? "validation" : "test";
    pools[s].push_back(ids[i]);
    split_of[ids[i]] = s;
  }
  const std::regex           token("[a-z0-9]+(?:'[a-z0-9]+)?");
  Json                       rows = Json::array();
  std::map<std::string, int> tc, ac;
  for (const auto& [qid, q] : questions) {
    const auto& a = annotations.at(qid);
    require(
        a["image_id"] == q["image_id"] && a["answers"].size() == 10, "unaligned VQA human answers");
    const std::string text = q["question"], normalized = lower(text),
                      s = split_of[q["image_id"].get<int>()];
    std::vector<std::string> tokens, answers;
    for (auto it = std::sregex_iterator(normalized.begin(), normalized.end(), token);
         it != std::sregex_iterator() && tokens.size() < 24;
         ++it)
      tokens.push_back(it->str());
    require(!tokens.empty(), "empty question tokens");
    for (const auto& item : a["answers"])
      answers.push_back(answer(item["answer"]));
    if (s == "fit") {
      for (const auto& w : tokens)
        ++tc[w];
      for (const auto& w : answers)
        ++ac[w];
    }
    rows.push_back(
        {{"split", s},
         {"image_id", q["image_id"]},
         {"question_id", qid},
         {"question", text},
         {"tokens", tokens},
         {"answers", answers},
         {"answer_type", a["answer_type"]}});
  }
  const auto tv = vocabulary(tc, {"<PAD>", "<UNK>"}, 512), av = vocabulary(ac, {"<UNK>"}, 64);
  std::map<std::string, int> ti, ai;
  std::ostringstream         tsv, asv, questions_tsv, images_tsv, records;
  for (int i = 0; i < int(tv.size()); ++i) {
    ti[tv[i]] = i;
    tsv << i << '\t' << tv[i] << '\n';
  }
  for (int i = 0; i < int(av.size()); ++i) {
    ai[av[i]] = i;
    asv << i << '\t' << av[i] << '\n';
  }
  for (int id : distinct)
    images_tsv << id << '\t' << images[id].string() << '\n';
  const std::map<std::string, int> types = {{"yes/no", 0}, {"number", 1}, {"other", 2}};
  for (const auto& r : rows) {
    std::string q = r["question"];
    std::replace(q.begin(), q.end(), '\t', ' ');
    std::replace(q.begin(), q.end(), '\n', ' ');
    questions_tsv << r["question_id"].get<std::int64_t>() << '\t' << r["image_id"].get<int>()
                  << '\t' << q << '\n';
    records << r["split"].get<std::string>() << '\t' << r["image_id"].get<int>() << '\t'
            << r["question_id"].get<std::int64_t>() << '\t' << types.at(r["answer_type"]) << '\t'
            << r["tokens"].size();
    for (const std::string w : r["tokens"])
      records << '\t' << (ti.count(w) ? ti[w] : 1);
    for (std::size_t i = r["tokens"].size(); i < 24; ++i)
      records << "\t0";
    for (const std::string w : r["answers"])
      records << '\t' << (ai.count(w) ? ai[w] : 0);
    records << '\n';
  }
  write(
      root / "config.tsv", std::to_string(tv.size()) + "\t" + std::to_string(av.size()) + "\t24\n");
  write(root / "token_vocabulary.tsv", tsv.str());
  write(root / "answer_vocabulary.tsv", asv.str());
  write(root / "questions.tsv", questions_tsv.str());
  write(root / "source_images.tsv", images_tsv.str());
  write(root / "records.tsv", records.str());
  call(prepare_vqa, {root.string()});
  auto j            = manifest("vqa");
  j["sources"]      = source_manifest("vqa");
  j["image_pools"]  = pools;
  j["records"]      = rows;
  j["vocabularies"] = {
      {"tokens", tv.size()},
      {"answers", av.size()},
      {"selection",
       "Fitting-only frequency descending,lexical ties;max512 token classes including "
       "PAD0/UNK1,max64 answer classes including UNK0;right-pad/truncate questions to24"}};
  save(root / "manifest.json", j);
}

/// @brief Transcode original Latin1 metadata into UTF-8 for auditable JSON/TSV text.
/// @param bytes Original corpus byte string.
/// @return UTF-8 encoding of the same characters.
std::string latin1(const std::string& bytes) {
  std::string out;
  for (unsigned char c : bytes) {
    if (c < 128)
      out += char(c);
    else {
      out += char(0xc0 | (c >> 6));
      out += char(0x80 | (c & 63));
    }
  }
  return out;
}
/// @brief Preserve aligned original MR word/character observations and source text provenance.
/// @param base Parent of sentence and character caches.
void text(const fs::path& base) {
  const auto c = protocols()["text"]["constants"];
  source(
      base / "sentence",
      "sentence",
      "rt-polaritydata.tar.gz",
      "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz",
      c["SHA"],
      487770);
  call(prepare_text, {base.string()});
  Temporary stage(base);
  extract_archive(
      (base / "sentence/rt-polaritydata.tar.gz").string(),
      stage.path.string(),
      {"rt-polaritydata/rt-polarity.neg", "rt-polaritydata/rt-polarity.pos"});
  std::vector<std::string> originals;
  for (const auto& name : {"neg", "pos"}) {
    auto source_lines =
        lines(read(stage.path / "rt-polaritydata" / (std::string("rt-polarity.") + name)));
    require(source_lines.size() == 5331, "original MR class count differs");
    originals.insert(originals.end(), source_lines.begin(), source_lines.end());
  }
  auto j                         = json(base / "sentence/manifest.json");
  j["archive_bytes"]             = 487770;
  j["archive_sha256"]            = c["SHA"];
  Json               source_rows = Json::object();
  std::ostringstream reviews;
  reviews << "source_id\tlabel\toriginal_text\n";
  std::map<std::string, std::string> owner;
  for (const auto& [split, ids] : j["source_ids"].items())
    for (int id : ids.get<std::vector<int>>()) {
      auto raw = originals.at(id);
      auto key = raw.substr(raw.size() > 256 ? raw.size() - 256 : 0);
      key      = lower(key);
      std::reverse(key.begin(), key.end());
      if (owner.count(key))
        require(owner[key] == split, "character truncation creates split leakage");
      owner[key]                      = split;
      source_rows[std::to_string(id)] = {{"label", id / 5331}, {"text", latin1(raw)}};
      std::replace(raw.begin(), raw.end(), '\t', ' ');
      std::replace(raw.begin(), raw.end(), '\r', ' ');
      reviews << id << '\t' << id / 5331 << '\t' << latin1(raw) << '\n';
    }
  j["original_source"] = source_rows;
  j["character_policy"] =
      "Same exact original source rows/labels as words;raw Latin1 bytes ASCII lowercase,reversed "
      "last256 bytes,authors69-slot alphabet duplicate hyphen uses last "
      "index;unknown/PAD=-1,structural onehot zero";
  write(base / "sentence/reviews.tsv", reviews.str());
  save(base / "sentence/manifest.json", j);
  save(base / "character/manifest.json", j);
}

/// @brief Decode exactly the selected original frame indices without synthetic frames.
/// @param input Original video.
/// @param root Prepared cache.
/// @param source Original video ID.
/// @param targets Sorted unique zero-based frame indices.
/// @return Frame index to relative PNG path mapping.
std::map<int, std::string>
frames(const fs::path& input, const fs::path& root, int source, const std::set<int>& targets) {
  const auto folder = root / "frames" / std::to_string(source);
  fs::create_directories(folder);
  std::map<int, std::string> out;
  int                        i       = 0;
  bool                       present = true;
  std::string                expr;
  for (int index : targets) {
    const auto p = folder / (padded(++i, 4) + ".png");
    present &= fs::is_regular_file(p);
    out[index] = fs::relative(p, root).generic_string();
    if (!expr.empty())
      expr += '+';
    expr += "eq(n\\," + std::to_string(index) + ")";
  }
  if (!present) {
    Temporary stage(folder);
    run(
        {"ffmpeg",
         "-v",
         "error",
         "-threads",
         "1",
         "-i",
         input.string(),
         "-vf",
         "select=" + expr,
         "-fps_mode",
         "vfr",
         "-threads",
         "1",
         "-start_number",
         "1",
         (stage.path / "%04d.png").string()});
    for (const auto& [_, p] : out) {
      const auto decoded = stage.path / fs::path(p).filename();
      require(fs::is_regular_file(decoded), "missing decoded source frame");
      fs::rename(decoded, root / p);
    }
  }
  return out;
}
} // namespace
} // namespace datamunge::datasets::detail
#include "kth_sources.hpp"
namespace datamunge::datasets::detail {
/// @brief Private source ingress and tensor preparation helpers.
namespace {
/// @brief Preserve KTH person/scenario source identities and original inclusive segment
/// annotations.
/// @param root Dataset cache.
void video(const fs::path& root) {
  const std::string
      rev  = protocols()["video"]["constants"]["REV"],
      base = "https://huggingface.co/datasets/AtharvaPote/KTH-Action-Recognition-mirror/resolve/" +
             rev + "/";
  for (const auto& s : kth_sources()) {
    const std::string p = s["path"];
    source(root, "action_video", p, base + p, s["sha256"], s["bytes"]);
  }
  const auto       annotations = read(root / "00sequences.txt");
  const std::regex line("(person[0-9]+_[a-z]+_d[0-9])\\s+frames\\s+([^\\n]+)"),
      segment("([0-9]+)-([0-9]+)");
  std::map<std::string, std::vector<std::pair<int, int>>> ranges;
  for (auto it = std::sregex_iterator(annotations.begin(), annotations.end(), line);
       it != std::sregex_iterator();
       ++it) {
    const auto text = (*it)[2].str();
    for (auto m = std::sregex_iterator(text.begin(), text.end(), segment);
         m != std::sregex_iterator();
         ++m)
      ranges[(*it)[1]].push_back({std::stoi((*m)[1]), std::stoi((*m)[2])});
  }
  Json               records = Json::array();
  std::ostringstream rows;
  rows << std::setprecision(17);
  const std::vector<std::string> actions = {"boxing", "handclapping", "handwaving"};
  for (int label = 0; label < 3; ++label)
    for (int person = 1; person <= 6; ++person) {
      const auto action = actions[label], key = "person" + padded(person, 2) + "_" + action + "_d1",
                 path = action + "/" + key + "_uncomp.avi";
      const int  id   = label * 6 + person - 1;
      const auto part = person <= 4 ? "fit" : person == 5 ? "validation" : "test";
      require(ranges[key].size() >= 2, "missing original KTH segments");
      std::vector<std::vector<int>> selections;
      std::set<int>                 targets;
      Json                          segments = Json::array();
      for (int s = 0; s < 2; ++s) {
        const auto [start, end] = ranges[key][s];
        require(start >= 1 && end - start >= 15, "invalid original KTH range");
        std::vector<int> indices;
        for (int i = 0; i < 16; ++i) {
          int at = start - 1 + (end - start) * i / 15;
          indices.push_back(at);
          targets.insert(at);
        }
        selections.push_back(indices);
        segments.push_back({start, end});
      }
      const auto paths      = frames(root / path, root, id, targets);
      Json       paths_json = Json::object();
      for (const auto& [frame, p] : paths)
        paths_json[std::to_string(frame)] = p;
      for (int s = 0; s < 2; ++s) {
        rows << part << '\t' << label << '\t' << person << '\t' << id << '\t' << s;
        for (int f : selections[s])
          rows << '\t' << paths.at(f);
        rows << '\n';
      }
      records.push_back(
          {{"source_id", id},
           {"person", person},
           {"label", label},
           {"action", action},
           {"path", path},
           {"sha256", file_digest((root / path).string())},
           {"segments_one_based_inclusive", segments},
           {"frame_paths", paths_json}});
    }
  write(root / "clip_rows.tsv", rows.str());
  call(prepare_video, {root.string()});
  auto j = manifest("video");
  j["mirror"] =
      "https://huggingface.co/datasets/AtharvaPote/KTH-Action-Recognition-mirror/tree/" + rev;
  j["annotation_sha256"] = file_digest((root / "00sequences.txt").string());
  j["records"]           = records;
  save(root / "manifest.json", j);
}

/// @brief Prepare eleven genuine adjacent frames per KTH segment with the declared tensor flow
/// solver.
/// @param root Destination cache.
/// @param videos Verified original KTH cache.
void two_stream(const fs::path& root, const fs::path& videos) {
  auto                                                     j       = json(videos / "manifest.json");
  Json                                                     records = Json::array();
  std::vector<std::tuple<int, int, int, int, std::string>> rows;
  for (const auto& r : j["records"]) {
    const int                     id = r["source_id"], person = r["person"], label = r["label"];
    const auto                    part = person <= 4 ? "fit" : person == 5 ? "validation" : "test";
    std::set<int>                 targets;
    std::vector<std::vector<int>> selections;
    for (const auto& segment : r["segments_one_based_inclusive"]) {
      int start = segment[0], end = segment[1], first = (start - 1 + end - 1) / 2 - 5;
      require(first >= start - 1 && first + 10 < end, "KTH segment too short for adjacent frames");
      std::vector<int> ids;
      for (int f = first; f <= first + 10; ++f) {
        ids.push_back(f);
        targets.insert(f);
      }
      selections.push_back(ids);
    }
    const auto paths = frames(videos / r["path"].get<std::string>(), root, id, targets);
    for (int s = 0; s < int(selections.size()); ++s) {
      std::ostringstream line;
      line << part << '\t' << label << '\t' << person << '\t' << id << '\t' << s;
      Json selected = Json::array();
      for (int f : selections[s]) {
        line << '\t' << paths.at(f);
        selected.push_back(paths.at(f));
      }
      line << '\n';
      rows.push_back({person <= 4 ? 0 : person == 5 ? 1 : 2, label, person, s, line.str()});
      records.push_back(
          {{"source_id", id},
           {"segment", s},
           {"person", person},
           {"label", label},
           {"split", part},
           {"zero_based_original_frames", selections[s]},
           {"original_avi_sha256", r["sha256"]},
           {"paths", selected}});
    }
  }
  std::sort(rows.begin(), rows.end());
  std::string metadata;
  for (const auto& row : rows)
    metadata += std::get<4>(row);
  write(root / "clip_rows.tsv", metadata);
  call(prepare_two_stream, {root.string()});
  j["records"] = records;
  j["frames"] =
      "Eleven consecutive genuine frames centered within each original annotated segment. Middle "
      "actual frame is RGB;ten forward adjacent pairs supply dense flow. FFmpeg codec-only decode "
      "at original resolution,thenFloat32tensor bilinear96px/255.";
  j["flow"] =
      "Declared single-scale Horn-Schunck quadratic objective solved by400explicit TensorSGD "
      "steps,not Farneback/TVL1/published flow or ground truth;all per-pixel derivatives/flow "
      "updates use tensors. Exact details in report/native preparation.";
  save(root / "manifest.json", j);
  write(
      root / "ATTRIBUTION.txt",
      "Original KTH non-commercial research action videos and pinned mirror provenance in "
      "manifest.json. Declared tensor variational flow, not ground truth.\n");
}

/// @brief Prepare sampled-surface voxel occupancy from the same genuine source meshes.
/// @param root Verified ModelNet cache.
void voxel(const fs::path& root) {
  call(prepare_voxel, {root.string()});
  auto j = json(root / "manifest.json");
  j["numeric"] =
      "Double tensor triangle areas,65536area-weighted faces/generator1193+ID,barycentric "
      "sqrt-uniform sampling. Center at full original vertex bounding-box midpoint,divide by "
      "maximum bounding-box half extent. Float32 normalized points are binned "
      "floor((xyz+1)*14.5)+1,clamp1..30,tensor segment sum>0. Sampled-SURFACE occupancy,NOT solid "
      "fill,triangle/voxel intersection or original published preprocessing. Original "
      "point_clouds.npz remains the separate1024point unit-ball representation.";
  j["sampling_audit"] =
      "voxel_sampling.npz retains every selected triangle,uniform pair,center and radius. "
      "voxnet.npz keeps24/6/12occupancy/label/source arrays.";
  save(root / "voxel_manifest.json", j);
  write(
      root / "ATTRIBUTION.txt",
      "Original Princeton ModelNet10 OFF CAD meshes; "
      "https://3dvision.princeton.edu/projects/2014/3DShapeNets/ . Original archive SHA256 " +
          j["sha256"].get<std::string>() +
          "\nCustom chair/sofa/table42object sampled-surface occupancy;not the published "
          "benchmark.\n");
  const auto destination = root.parent_path() / "voxnet";
  fs::create_directories(destination);
  for (const auto& name :
       {"voxnet.npz", "voxel_sampling.npz", "test_contact.png", "ATTRIBUTION.txt"})
    fs::copy_file(root / name, destination / name, fs::copy_options::overwrite_existing);
  save(destination / "manifest.json", j);
}

/// @brief Required outputs used to validate a completed native preparation marker.
/// @param name Dataset adapter.
/// @return Relative output paths, including the provenance manifest.
std::vector<std::string> outputs(const std::string& name) {
  static const std::map<std::string, std::vector<std::string>> files = {
      {"memory", {"ntm/data.npz", "ntm/manifest.json", "memnet/data.npz", "memnet/manifest.json"}},
      {"mnist-assets", {"saliency/saliency_data.npz", "saliency/manifest.json"}},
      {"audio", {"wavenet_stack.npz", "manifest.json", "source_files.tsv"}},
      {"depth", {"pairs.npz", "manifest.json", "selected_rows.tsv"}},
      {"flowers", {"flowers.npz", "manifest.json", "prepared_rows.tsv"}},
      {"medical", {"medical_volume.npz", "manifest.json", "selected_rows.tsv"}},
      {"mesh", {"point_clouds.npz", "point_sampling.npz", "manifest.json", "mesh_rows.tsv"}},
      {"multilabel", {"multilabel.npz", "manifest.json", "prepared_rows.tsv"}},
      {"ocr", {"ocr.npz", "manifest.json", "prepared_rows.tsv"}},
      {"optical_flow", {"pairs.npz", "manifest.json", "crop_rows.tsv"}},
      {"pose", {"poses.npz", "manifest.json", "crops.tsv"}},
      {"signals",
       {"time_series/time_series.npz",
        "time_series/manifest.json",
        "ecg_sensor/ECG200/signals.npz",
        "ecg_sensor/ECG200/manifest.json",
        "ecg_sensor/BasicMotions/signals.npz",
        "ecg_sensor/BasicMotions/manifest.json"}},
      {"style", {"vgg16.npz", "vgg19.npz", "content.jpg", "style.jpg", "manifest.json"}},
      {"text",
       {"sentence/sentence.npz",
        "sentence/manifest.json",
        "sentence/reviews.tsv",
        "sentence/vocabulary.txt",
        "character/character.npz",
        "character/manifest.json"}},
      {"two_stream", {"two_stream.npz", "flow_audit.npz", "manifest.json"}},
      {"vqa", {"vqa.npz", "manifest.json", "records.tsv", "source_images.tsv"}},
      {"video", {"action_video.npz", "manifest.json", "clip_rows.tsv"}},
      {"voxel",
       {"voxnet.npz",
        "voxel_sampling.npz",
        "voxel_manifest.json",
        "../voxnet/voxnet.npz",
        "../voxnet/manifest.json"}}};
  const auto it = files.find(name);
  if (it == files.end())
    throw std::invalid_argument("unknown example dataset: " + name);
  return it->second;
}
} // namespace
} // namespace datamunge::datasets::detail

namespace datamunge::datasets {
std::string prepare_example_data(const std::string& name, const std::string& directory) {
  using namespace detail;
  const auto files  = outputs(name);
  fs::path   target = directory;
  if (target.empty()) {
    target = data_home();
    /// @brief Public adapter names mapped to the shared examples' conventional cache folders.
    static const std::map<std::string, std::string> folders = {
        {"audio", "wavenet"},
        {"depth", "monocular_depth"},
        {"flowers", "flowers102"},
        {"medical", "medical_volume"},
        {"mesh", "modelnet"},
        {"pose", "human_pose"},
        {"style", "style_transfer"},
        {"video", "action_video"},
        {"voxel", "modelnet"}};
    if (name != "signals" && name != "text" && name != "memory" && name != "mnist-assets") {
      const auto found = folders.find(name);
      target /= found == folders.end() ? name : found->second;
    }
  }
  const auto root = fs::absolute(target);
  // Dependencies are prepared before locking the target, avoiding lock inversion.
  if (name == "ocr" || name == "vqa")
    prepare_example_data("multilabel", (root.parent_path() / "multilabel").string());
  if (name == "two_stream")
    prepare_example_data("video", (root.parent_path() / "action_video").string());
  if (name == "voxel")
    prepare_example_data("mesh", root.string());
  /// @brief Serialize calls within this process as well as across cache lock files.
  static std::recursive_mutex mutex;
  const std::lock_guard       guard(mutex);
  fs::create_directories(root);
  CacheLock  lock(root);
  const auto marker   = root / (".native-" + name + (name == "mesh" ? "-v2.json" : "-v1.json"));
  bool       complete = fs::is_regular_file(marker);
  if (complete) {
    const auto stamp = json(marker);
    for (const auto& path : files) {
      const auto file = root / path;
      if (!fs::is_regular_file(file) || !stamp.contains(path) ||
          stamp[path] != fs::file_size(file)) {
        complete = false;
        break;
      }
    }
  }
  if (complete)
    return root.string();
  if (name == "memory" || name == "mnist-assets")
    prepare_basic(name, root);
  else if (name == "audio")
    audio(root);
  else if (name == "depth")
    depth(root);
  else if (name == "flowers")
    flowers(root);
  else if (name == "medical")
    medical(root);
  else if (name == "mesh")
    mesh(root);
  else if (name == "multilabel")
    multilabel(root);
  else if (name == "ocr")
    ocr(root, root.parent_path() / "multilabel");
  else if (name == "optical_flow")
    optical_flow(root);
  else if (name == "pose")
    pose(root);
  else if (name == "signals")
    signals(root);
  else if (name == "style")
    style(root);
  else if (name == "text")
    text(root);
  else if (name == "vqa")
    vqa(root, root.parent_path() / "multilabel");
  else if (name == "video")
    video(root);
  else if (name == "two_stream")
    two_stream(root, root.parent_path() / "action_video");
  else if (name == "voxel")
    voxel(root);
  Json stamp = Json::object();
  for (const auto& path : files) {
    require(fs::is_regular_file(root / path), "native preparation did not produce " + path);
    stamp[path] = fs::file_size(root / path);
  }
  save(marker, stamp);
  return root.string();
}
} // namespace datamunge::datasets

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
