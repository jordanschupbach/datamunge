// Where the nn example ports read datasets and write their outputs.
//
// Datasets: $DATAMUNGE_DATA_DIR, or ~/projects/tflow_examples/output/datasets
// (the tflow examples' download cache, filled by its scripts/fetch_datasets.py).
// Outputs (checkpoints, histories, fixtures): $DATAMUNGE_OUTPUT_DIR, or
// output/nn relative to the working directory (`just run-nn` runs from the
// repository root). Both are created on demand by the caller.
#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>
#include "common/runtime.hpp"

namespace paths {

inline std::filesystem::path DataRoot() {
  if (const char* dir = std::getenv("DATAMUNGE_DATA_DIR"); dir && *dir) return dir;
  const char* home = std::getenv("HOME");
  return std::filesystem::path(home ? home : ".") / "projects" / "tflow_examples" / "output" / "datasets";
}

/// DataRoot() / name, e.g. Dataset("mnist").
inline std::filesystem::path Dataset(const std::string& name) { return DataRoot() / name; }

inline std::filesystem::path OutputRoot() {
  if (const char* dir = std::getenv("DATAMUNGE_OUTPUT_DIR"); dir && *dir) return dir;
  return example_runtime::Full() ? "output/nn/full" : "output/nn/quick";
}

/// OutputRoot() / relative, created as a directory.
inline std::filesystem::path OutputDir(const std::string& relative) {
  auto dir = OutputRoot() / relative;
  std::filesystem::create_directories(dir);
  return dir;
}

}  // namespace paths
