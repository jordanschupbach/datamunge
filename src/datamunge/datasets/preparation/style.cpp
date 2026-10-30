/// @file style.cpp
/// @brief Native style tensor preparation from verified source metadata.
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include <H5Cpp.h>

#include <dm/dm.hpp>
#include <filesystem>
#include <iostream>

#include "preparation.hpp"

/// @brief Decode official Keras HDF5 weight arrays and serialize unmodified tensors.
/// @param argc Program name plus source/cache directory.
/// @param argv Root containing pinned VGG16/VGG19 no-top HDF5 sources.
/// @return Zero after exact complete Float32 convolution inventories are written.
int datamunge::datasets::detail::prepare_style(int argc, char** argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("expected style source/cache directory");
    namespace n                      = dm::nn;
    const std::filesystem::path root = argv[1];
    for (int version : {16, 19}) {
      H5::H5File file(
          (root / ("vgg" + std::to_string(version) + "_notop.h5")).string(), H5F_ACC_RDONLY);
      const std::vector<int> depths =
          version == 16 ? std::vector<int>{2, 2, 3, 3, 3} : std::vector<int>{2, 2, 4, 4, 4};
      const std::vector<int> widths = {64, 128, 256, 512, 512};
      n::NamedTensors        tensors;
      int                    input = 3;
      std::int64_t           count = 0;
      for (int b = 0; b < 5; ++b)
        for (int l = 0; l < depths[b]; ++l) {
          const auto name = "block" + std::to_string(b + 1) + "_conv" + std::to_string(l + 1);
          for (const auto suffix : {std::string("W"), std::string("b")}) {
            const auto dataset = file.openDataSet("/" + name + "/" + name + "_" + suffix + "_1:0");
            const auto space   = dataset.getSpace();
            const int  rank    = space.getSimpleExtentNdims();
            const n::Shape expected =
                suffix == "W" ? n::Shape{3, 3, input, widths[b]} : n::Shape{widths[b]};
            if (rank != int(expected.size()) || dataset.getTypeClass() != H5T_FLOAT ||
                dataset.getDataType().getSize() != sizeof(float))
              throw std::invalid_argument("official VGG array rank/dtype mismatch");
            std::vector<hsize_t> dimensions(rank);
            space.getSimpleExtentDims(dimensions.data());
            if (n::Shape(dimensions.begin(), dimensions.end()) != expected)
              throw std::invalid_argument("official VGG shape mismatch");
            std::vector<float> values(n::shape_numel(expected));
            dataset.read(values.data(), H5::PredType::NATIVE_FLOAT);
            const auto tensor = n::from_vector(values, expected);
            if (!n::all(n::is_finite(tensor)).item<bool>())
              throw std::invalid_argument("nonfinite source VGG value");
            tensors.push_back({name + "_" + suffix, tensor});
            count += tensor.numel();
          }
          input = widths[b];
        }
      if (count != (version == 16 ? 14714688 : 20024384))
        throw std::invalid_argument("VGG convolution parameter census mismatch");
      n::save_npz((root / ("vgg" + std::to_string(version) + ".npz")).string(), tensors);
      std::cout << "Decoded full official VGG" << version << ":" << count
                << " unmodified Float32 values," << tensors.size() << " arrays.\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
