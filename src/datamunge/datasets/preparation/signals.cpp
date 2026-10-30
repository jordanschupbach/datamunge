/// @file signals.cpp
/// @brief Native signals tensor preparation from verified source metadata.
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include <dm/dm.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include "preparation.hpp"
/// @brief Preserve raw source numeric ingress as Float32 tensors with aligned exact row metadata.
/// @param argc Program name,verified metadata directory and output NPZ basename.
/// @param argv Paths to original-source ingress and final aligned tensor file.
/// @return Zero when all three partitions and shape metadata are saved.
int datamunge::datasets::detail::prepare_signals(int argc, char** argv) {
  try {
    if (argc != 3)
      throw std::invalid_argument("expected raw metadata root and output basename");
    namespace n = dm::nn;
    n::NoGradGuard              guard;
    const std::filesystem::path root = argv[1];
    int                         time, channels, classes, train;
    std::ifstream               cfg(root / "config.tsv");
    if (!(cfg >> time >> channels >> classes >> train))
      throw std::invalid_argument("signal configuration required");
    std::ifstream                                 file(root / "raw_signals.tsv");
    std::map<std::string, std::vector<n::Tensor>> signals;
    std::map<std::string, std::vector<int>>       labels, ids;
    std::string                                   line;
    while (std::getline(file, line)) {
      std::istringstream row(line);
      std::string        split;
      int                id, label;
      if (!(row >> split >> id >> label))
        throw std::invalid_argument("signal record metadata required");
      std::vector<float> values(time * channels);
      for (auto& v : values)
        if (!(row >> v))
          throw std::invalid_argument("all unmodified source samples required");
      signals[split].push_back(n::from_vector(values, {time, channels}));
      labels[split].push_back(label);
      ids[split].push_back(id);
    }
    n::NamedTensors arrays{
        {"config", n::from_vector(std::vector<int>{time, channels, classes, train}, {4})}};
    for (const std::string split : {"fit", "validation", "test"}) {
      arrays.push_back({split + "_signals", n::stack(signals.at(split), 0)});
      arrays.push_back({split + "_labels", n::from_vector(labels.at(split))});
      arrays.push_back({split + "_source_ids", n::from_vector(ids.at(split))});
      std::cout << root.filename() << " " << split << ":" << signals.at(split).size()
                << " original source rows\n";
    }
    n::save_npz((root / argv[2]).string(), arrays);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
