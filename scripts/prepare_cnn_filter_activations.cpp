#include <dm/dm.hpp>
#include <filesystem>
#include <print>
/// @brief Preserve the two original source JPEG observations as tensor64pxRGBinputs.
/// @param argc Program name plus pinned style-source root and output root.
/// @param argv Source/destination directory paths.
/// @return Zero after both genuine original inputs are prepared with tensor processing.
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      return 1;
    namespace n = dm::nn;
    namespace m = n::models;
    n::set_num_threads(2);
    const auto source = std::filesystem::path(argv[1]), root = std::filesystem::path(argv[2]);
    std::filesystem::create_directories(root);
    const auto rgb = n::concat({m::LoadStyleImage(source / "content.jpg", 64),
                                m::LoadStyleImage(source / "style.jpg", 64)},
                               0);
    n::save_npz((root / "inputs.npz").string(), {{"rgb", rgb}});
    n::save_image_grid(rgb, (root / "inputs.png").string(), 2, 0, 1, 2, .8, 4);
    std::print("Two original source JPEGs:center-square crop,Float32tensor64pxbilinear "
               "andRGB/255.\n");
  } catch (const std::exception &e) {
    std::print(stderr, "{}\n", e.what());
    return 1;
  }
}
