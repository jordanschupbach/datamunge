#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include "preparation.hpp"
/// @file ocr.cpp
/// @brief Tensor original-image word cropping from checked human annotation metadata.
#include <dm/dm.hpp>
#include <fstream>
#include <iostream>
#include <map>

/// @brief Prepare43/7/6 source-image-disjoint human word tensors.
/// @param argc Three arguments are required.
/// @param argv Program name,prepared metadata directory and original image directory.
/// @return Zero after saving and validating all56 aligned word crops.
int datamunge::datasets::detail::prepare_ocr(int argc, char** argv) {
  try {
    if (argc != 3)
      throw std::invalid_argument("expected OCR metadata/cache and original image directories");
    namespace n = dm::nn;
    namespace m = n::models;
    n::set_num_threads(2);
    const std::filesystem::path root = argv[1], images = argv[2];
    std::ifstream               input(root / "prepared_rows.tsv");
    if (!input)
      throw std::runtime_error("OCR row metadata missing");
    std::map<std::string, std::vector<m::OcrCropRecord>> records;
    std::string                                          split;
    m::OcrCropRecord                                     row;
    while (input >> split >> row.image_id >> row.annotation_id >> row.text >> row.bbox[0] >>
           row.bbox[1] >> row.bbox[2] >> row.bbox[3] >> row.expected_width >> row.expected_height) {
      if (split != "fit" && split != "validation" && split != "test")
        throw std::invalid_argument("OCR split metadata");
      records[split].push_back(row);
    }
    if (!input.eof())
      throw std::invalid_argument("truncated OCR row metadata");
    n::NamedTensors arrays;
    for (const std::string name : {"fit", "validation", "test"}) {
      const auto d = m::ReadOcrCrops(images, records.at(name));
      arrays.push_back({name + "_pixels", d.pixels});
      arrays.push_back({name + "_labels", d.labels});
      arrays.push_back({name + "_lengths", d.lengths});
      arrays.push_back({name + "_image_ids", d.image_ids});
      arrays.push_back({name + "_annotation_ids", d.annotation_ids});
    }
    n::save_npz((root / "ocr.npz").string(), arrays);
    const auto data = m::LoadOcr(root);
    if (data[0].pixels.size(0) != 43 || data[1].pixels.size(0) != 7 || data[2].pixels.size(0) != 6)
      throw std::runtime_error("wrong prepared OCR counts");
    std::cout << "Validated43/7/6 tensor word crops and source-image-disjoint human targets.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
