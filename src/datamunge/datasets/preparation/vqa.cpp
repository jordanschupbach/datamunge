/// @file vqa.cpp
/// @brief Native vqa tensor preparation from verified source metadata.
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include <datamunge/image/jpeg.hpp>
#include <dm/dm.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include "preparation.hpp"

/// @brief Decode original COCO observations and form all human-answer numeric targets with tensors.
/// @param argc Program name and prepared metadata root.
/// @param argv Root containing source_images.tsv,records.tsv and config.tsv.
/// @return Zero after original RGB,tokens,ten-answer frequency/consensus and IDs are saved.
int datamunge::datasets::detail::prepare_vqa(int argc, char** argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("expected VQA metadata/cache root");
    namespace n = dm::nn;
    n::set_num_threads(2);
    n::NoGradGuard              guard;
    const std::filesystem::path root = argv[1];
    int                         vocabulary, classes, time;
    std::ifstream               config(root / "config.tsv");
    if (!(config >> vocabulary >> classes >> time) || vocabulary < 2 || vocabulary > 512 ||
        classes < 2 || classes > 64 || time != 24)
      throw std::invalid_argument("bounded fit-only VQA vocabulary metadata required");
    std::map<int, n::Tensor> images;
    std::ifstream            sources(root / "source_images.tsv");
    int                      image_id;
    std::string              path;
    std::string              source_line;
    while (std::getline(sources, source_line)) {
      const auto tab = source_line.find('\t');
      if (tab == std::string::npos || tab + 1 == source_line.size())
        throw std::invalid_argument("VQA source row requires an ID and a tab-delimited image path");
      std::size_t parsed = 0;
      image_id           = std::stoi(source_line.substr(0, tab), &parsed);
      if (parsed != tab)
        throw std::invalid_argument("invalid VQA source image ID");
      path = source_line.substr(tab + 1);
      dm::image::JpegDecodeOptions options;
      options.channels    = 3;
      const auto original = n::expand_dims(n::from_image(dm::image::read_jpeg(path, options)), 0);
      images.emplace(image_id, n::resize_bilinear(original / 255., {64, 64}).contiguous());
    }
    std::ifstream records(root / "records.tsv");
    std::string   line;
    std::map<std::string, std::map<std::string, std::vector<n::Tensor>>> parts;
    int                                                                  total = 0;
    while (std::getline(records, line)) {
      std::istringstream row(line);
      std::string        split;
      std::int64_t       question_id;
      int                type, length;
      if (!(row >> split >> image_id >> question_id >> type >> length))
        throw std::invalid_argument("malformed original VQA row metadata");
      std::vector<std::int32_t> tokens(time), answers(10);
      for (auto& token : tokens)
        if (!(row >> token))
          throw std::invalid_argument("VQA24 token IDs required");
      for (auto& answer : answers)
        if (!(row >> answer))
          throw std::invalid_argument("VQA ten original human answers required");
      const auto ids    = n::from_vector(answers, {1, 10});
      const auto count  = n::sum(n::one_hot(ids, classes), {1}).to(n::DType::Float64);
      const auto target = (count / 10.).to(n::DType::Float32), zero = n::zeros_like(count),
                 one            = n::ones_like(count);
      const auto leave_other    = n::minimum(one, n::maximum(count - 1., zero) / 3.),
                 leave_nonmatch = n::minimum(one, count / 3.);
      const auto consensus      = ((count * leave_other + (10. - count) * leave_nonmatch) / 10.)
                                 .to(n::DType::Float32),
                 columns = n::arange(0, classes).reshape({1, classes}),
                 score   = n::where(n::greater(columns, 0.), consensus, 0.);
      const auto hard    = n::argmax(count, 1).to(n::DType::Int32);
      auto&      p       = parts[split];
      p["rgb"].push_back(images.at(image_id));
      p["tokens"].push_back(n::from_vector(tokens, {1, time}));
      p["lengths"].push_back(n::from_vector(std::vector<std::int32_t>{length}, {1}));
      p["target"].push_back(target);
      p["score"].push_back(score);
      p["hard"].push_back(hard);
      p["image_ids"].push_back(n::from_vector(std::vector<std::int32_t>{image_id}, {1}));
      p["question_ids"].push_back(n::from_vector(std::vector<std::int64_t>{question_id}, {1}));
      p["types"].push_back(n::from_vector(std::vector<std::int32_t>{type}, {1}));
      p["prefix"].push_back(
          n::from_vector(std::vector<std::int32_t>{tokens[0], tokens[1]}, {1, 2}));
      ++total;
    }
    n::NamedTensors arrays{
        {"config", n::from_vector(std::vector<std::int32_t>{vocabulary, classes, time}, {3})}};
    for (const auto& [split, fields] : parts) {
      for (const auto& [name, rows] : fields)
        arrays.push_back({split + "_" + name, n::concat(rows, 0)});
      std::cout << split << ":" << fields.at("rgb").size() << " original human questions.\n";
    }
    n::save_npz((root / "vqa.npz").string(), arrays);
    std::cout << "Prepared" << total << " human questions over" << images.size()
              << " original COCO photos;all RGB/soft targets/official leave-one-out consensus "
                 "are tensors.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
