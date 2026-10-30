/// @file text.cpp
/// @brief Native text tensor preparation from verified source metadata.
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include <datamunge/nn/character_cnn_models.hpp>

#include "preparation.hpp"
namespace charcnn = datamunge::nn::models;
#include <dm/dm.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "sentence_assets.hpp"
/// @brief Prepare aligned genuine word and raw-character observations from pinned MR text.
/// @param argc Program name and dataset-cache parent.
/// @param argv Cache parent containing sentence/rt-polaritydata.tar.gz.
/// @return Zero after existing deterministic word selection and aligned character metadata are
/// saved.
int datamunge::datasets::detail::prepare_text(int argc, char** argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("expected dataset-cache parent");
    namespace n                      = dm::nn;
    const std::filesystem::path root = argv[1];
    datamunge::datasets::detail::sentence::EnsureSentences((root / "sentence").string());
    datamunge::datasets::detail::Temporary stage(root);
    dm::datasets::extract_archive(
        (root / "sentence/rt-polaritydata.tar.gz").string(),
        stage.path.string(),
        {"rt-polaritydata/rt-polarity.neg", "rt-polaritydata/rt-polarity.pos"});
    std::vector<std::string> texts;
    for (const std::string kind : {"neg", "pos"}) {
      std::istringstream input(
          datamunge::datasets::detail::read(
              stage.path / "rt-polaritydata" / ("rt-polarity." + kind)));
      std::string line;
      while (std::getline(input, line))
        texts.push_back(line);
    }
    if (texts.size() != 10662)
      throw std::invalid_argument("original MR source lines required");
    std::map<std::string, n::Tensor> words;
    for (const auto& [name, value] : n::load_npz((root / "sentence/sentence.npz").string()))
      words.emplace(name, value);
    n::NamedTensors arrays{{"config", n::from_vector(std::vector<int>{69, 256}, {2})}};
    for (const std::string split : {"fit", "validation", "test"}) {
      const auto       ids = words.at(split + "_source_ids").to_vector<int>();
      std::vector<int> tokens, lengths;
      for (int id : ids) {
        const auto& text    = texts.at(id);
        const auto  encoded = charcnn::CharacterEncode(text, 256);
        tokens.insert(tokens.end(), encoded.begin(), encoded.end());
        lengths.push_back(std::min<int>(256, text.size()));
      }
      arrays.push_back({split + "_ids", n::from_vector(tokens, {std::int64_t(ids.size()), 256})});
      arrays.push_back({split + "_lengths", n::from_vector(lengths)});
      arrays.push_back({split + "_labels", words.at(split + "_labels")});
      arrays.push_back({split + "_source_ids", words.at(split + "_source_ids")});
      std::cout << "Original MR " << split << ":" << ids.size()
                << " aligned word/character observations\n";
    }
    std::filesystem::create_directories(root / "character");
    n::save_npz((root / "character/character.npz").string(), arrays);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
