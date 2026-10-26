// CMD: just run-nn filter_activations
// Pretrained learned RGB filter grids and all-channel VGG19 activation mosaics.
// Five block taps,shared-layer versus per-channel scales,real attributed images.
//
// Port of tflow's filter_activations. Data: inputs.npz of tflow's
// `just filter-activation-assets` (paths::Dataset("filter_activations"), data=PATH)
// and the Keras VGG-19 weights vgg19.npz of `just style-assets`
// (paths::Dataset("style_transfer")/vgg19.npz, weights=PATH). The frozen network is
// style::FrozenVgg19 (common/perceptual.hpp) with max pools (common/visual_features.hpp).
// Outputs: output/nn/models/filter_activations/.
// Options:data=PATH,weights=PATH,out=PATH,threads=N.

#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "common/checks.hpp"
#include "common/paths.hpp"
#include "common/perceptual.hpp"
#include "common/saliency.hpp"
#include "common/visual_features.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using sal::Canvas;

namespace {

// Small numeric labels;row-major channels are also documented in the gallery.
void Number(Canvas& canvas, int y, int x, int value) {
  const std::array<std::string, 10> font = {"111101101101111", "010110010010111", "111001111100111", "111001111001111",
                                            "101101111001001", "111100111001111", "111100111101111", "111001001001001",
                                            "111101111101111", "111101111001111"};
  for (char c : std::to_string(value)) {
    for (int a = 0; a < 5; ++a)
      for (int b = 0; b < 3; ++b)
        if (font[c - '0'][a * 3 + b] == '1')
          for (int k = 0; k < 3; ++k) canvas.at(y + a, x + b, k) = 230;
    x += 4;
  }
}

std::uint8_t Byte(double value) { return sal::SaliencyByte(value); }

/// features [N, H, W, C] (host values) -> all-channel mosaic of `image`.
Canvas Mosaic(const std::vector<float>& x, const nn::Shape& shape, int image, bool shared, double maximum) {
  const int height = int(shape[1]), width = int(shape[2]), channels = int(shape[3]), tile = std::max(16, height),
            columns = channels == 64 ? 8 : 16, rows = (channels + columns - 1) / columns;
  auto   at = [&](int b, int y, int a, int c) { return double(x[((std::size_t(b) * height + y) * width + a) * channels + c]); };
  Canvas canvas(rows * (tile + 10) + 2, columns * (tile + 2) + 2, 24);
  for (int c = 0; c < channels; ++c) {
    double peak = 0;
    for (int y = 0; y < height; ++y)
      for (int a = 0; a < width; ++a) peak = std::max(peak, at(image, y, a, c));
    const double scale = shared ? maximum : peak;
    const int    oy = c / columns * (tile + 10) + 2, ox = c % columns * (tile + 2) + 2;
    Number(canvas, oy, ox, c);
    for (int y = 0; y < tile; ++y)
      for (int a = 0; a < tile; ++a) {
        const std::uint8_t v = scale > 0 ? Byte(255 * at(image, y * height / tile, a * width / tile, c) / scale) : 0;
        for (int k = 0; k < 3; ++k) canvas.at(oy + 8 + y, ox + a, k) = v;
      }
  }
  return canvas;
}

Canvas Filters(const Tensor& kernel_t, double* maximum) {
  check::Require(kernel_t.shape() == nn::Shape({3, 3, 3, 64}), "first learned3x3BGR filter shape");
  const auto kernel = sal::F(kernel_t);
  double     peak = 0;
  for (float v : kernel) peak = std::max(peak, std::abs(double(v)));
  check::Require(peak > 0, "nonzero learned filters");
  *maximum = peak;
  constexpr int tile = 36;
  Canvas        canvas(8 * (tile + 10) + 2, 8 * (tile + 2) + 2, 24);
  for (int c = 0; c < 64; ++c) {
    const int oy = c / 8 * (tile + 10) + 2, ox = c % 8 * (tile + 2) + 2;
    Number(canvas, oy, ox, c);
    for (int y = 0; y < tile; ++y)
      for (int x = 0; x < tile; ++x)
        for (int rgb = 0; rgb < 3; ++rgb)
          canvas.at(oy + 8 + y, ox + x, rgb) = Byte(127.5 + 127.5 * kernel[(((y / 12) * 3 + x / 12) * 3 + (2 - rgb)) * 64 + c] / peak);
  }
  return canvas;
}

void RenderingChecks() {
  std::vector<float> zero(256, 0.f);
  auto               map = Mosaic(zero, {1, 2, 2, 64}, 0, false, 0);
  check::Require(map.at(10, 2, 0) == 0, "allzero channels render black,notNaN");
  std::vector<float> two(256, 2.f);
  map = Mosaic(two, {1, 2, 2, 64}, 0, false, 2);
  check::Require(map.at(10, 2, 0) == 255, "constantpositive ReLUchannel renderswhite,notzero");
  map = Mosaic(two, {1, 2, 2, 64}, 0, true, 4);
  check::Require(map.at(10, 2, 0) == 128, "shared scale retains relative response size");
  std::cout << "render scale fixtures:zero/constantpositive/sharedhalf values passed\n";
}

void Run(const std::filesystem::path& data, const std::filesystem::path& weights, const std::filesystem::path& directory) {
  const auto   arrays = sal::LoadArrays(data / "inputs.npz");
  const Tensor rgb = arrays.at("rgb");
  check::Require(rgb.dtype() == nn::DType::Float32 && rgb.shape() == nn::Shape({2, 64, 64, 3}), "two real64x64RGB observations");
  const auto rgb_v = sal::F(rgb);
  for (float v : rgb_v) check::Require(std::isfinite(v) && v >= 0 && v <= 1, "finite RGBunit inputs");
  const auto              tensors = style::LoadTensorMap(weights.string());
  const style::FrozenVgg19 vgg(tensors);
  std::vector<Tensor>      values;
  visual::VisualFeatures   features;
  {
    nn::NoGradGuard guard;
    features = visual::VggVisualFeatures(vgg, rgb);  // actual full16conv pretrained VGG19 originalmaxpool forward
    values = {features.centered.contiguous(), features.first_pre.contiguous()};
    for (const auto& a : features.activation) values.push_back(a.contiguous());
  }
  std::filesystem::create_directories(directory);
  sal::Arrays export_arrays = {{"rgb", rgb},
                               {"centered", values[0]},
                               {"first_pre", values[1]},
                               {"first_kernel", tensors.at("block1_conv1_W")},
                               {"first_bias", tensors.at("block1_conv1_b")}};
  for (int b = 0; b < 2; ++b) {
    Canvas pixels(64, 64);
    for (int i = 0; i < 64 * 64 * 3; ++i) pixels.px[i] = Byte(255 * rgb_v[b * 64 * 64 * 3 + i]);
    sal::SaliencyPng(directory / ("input" + std::to_string(b) + ".png"), pixels);
  }
  // First convolution independent host reference includes all64channels/corners/center.
  const auto centered = sal::F(values[0]), first_pre = sal::F(values[1]), first_act = sal::F(values[2]);
  const auto kernel = sal::F(tensors.at("block1_conv1_W")), bias = sal::F(tensors.at("block1_conv1_b"));
  double     first_error = 0;
  int        probes = 0;
  for (int b = 0; b < 2; ++b)
    for (const auto& pos : std::vector<std::pair<int, int>>{{0, 0}, {0, 63}, {63, 0}, {63, 63}, {32, 32}})
      for (int c = 0; c < 64; ++c) {
        double expected = bias[c];
        for (int y = 0; y < 3; ++y)
          for (int x = 0; x < 3; ++x) {
            const int iy = pos.first + y - 1, ix = pos.second + x - 1;
            if (iy < 0 || ix < 0 || iy >= 64 || ix >= 64) continue;
            for (int k = 0; k < 3; ++k)
              expected += double(centered[((b * 64 + iy) * 64 + ix) * 3 + k]) * kernel[((y * 3 + x) * 3 + k) * 64 + c];
          }
        first_error = std::max(first_error, std::abs(expected - first_pre[((b * 64 + pos.first) * 64 + pos.second) * 64 + c]));
        ++probes;
      }
  check::Near(first_error, 0, .001, "all640 independent firstconv corner/center probes");
  for (std::size_t i = 0; i < first_pre.size(); ++i) check::Near(first_act[i], std::max(0.f, first_pre[i]), 0, "every firstconv ReLUvalue");
  const std::array<float, 3> means{103.939f, 116.779f, 123.68f};
  for (int b = 0; b < 2; ++b)
    for (int y = 0; y < 64; ++y)
      for (int x = 0; x < 64; ++x)
        for (int k = 0; k < 3; ++k)
          check::Near(centered[((b * 64 + y) * 64 + x) * 3 + k], rgb_v[((b * 64 + y) * 64 + x) * 3 + 2 - k] * 255.f - means[k], 1e-5,
                      "all RGB->BGR/ImageNetcenter values");
  std::ofstream stats(directory / "channels.tsv");
  stats << "layer\timage\tchannel\tmean\tmaximum\tpositive_fraction\n";
  std::ofstream gallery(directory / "index.html");
  gallery << "<!doctype html><meta charset='utf-8'><title>VGG19 filters and activations</title><style>body{font:16px "
             "system-ui;max-width:1200px;margin:30px "
             "auto;background:#eee;color:#222}img{max-width:100%;image-rendering:pixelated}section{margin:32px 0}table "
             "td{vertical-align:top}p{max-width:90ch}</style><h1>Learned filters and real-image activation maps</h1><p>Full frozen "
             "ImageNet VGG19,standard MAX pools,no training/classification head. Inputs:Yellow Labrador(user Elf,CC BY-SA3.0) and "
             "Kandinsky Composition7(public domain). Early/deep feature responses are not class explanations. <a "
             "href=https://commons.wikimedia.org/wiki/File:YellowLabradorLooking_new.jpg>Labrador source</a>,<a "
             "href=https://creativecommons.org/licenses/by-sa/3.0/>CC BY-SA3.0</a>;input crops resized/center-cropped. <a "
             "href=https://storage.googleapis.com/download.tensorflow.org/example_images/"
             "Vassily_Kandinsky%2C_1913_-_Composition_7.jpg>Kandinsky source</a>.</p><p>All channels are shown and numbered,row-major. "
             "Shared scale uses one maximum across both images and every channel OF THAT LAYER;different layers have different units. "
             "Per-channel scale uses each image/channel maximum and can make weak responses look strong. Zero channels are "
             "black;constantpositive channels remain visible.</p><img src='input0.png'><img src='input1.png'><h2>All64 learned first-layer "
             "filters</h2><p>3x3 BGR-input kernels displayed in RGB channel order,nearest enlargement;one symmetric scale across all "
             "filters,neutralgray=zero. These are learned weights,not optimized images or explanations.</p><img src='filters.png'>";
  double kernel_peak;
  sal::SaliencyPng(directory / "filters.png", Filters(tensors.at("block1_conv1_W"), &kernel_peak));
  std::vector<float> peaks(5);
  for (int l = 0; l < 5; ++l) {
    const Tensor& a = values[l + 2];
    const int     side = 64 >> l, channels = style::FrozenVgg19::Widths[std::size_t(l)];
    check::Require(a.shape() == nn::Shape({2, side, side, channels}), "allfive fullreference activation shapes");
    const auto av = sal::F(a);
    double     maximum = 0;
    for (float v : av) {
      check::Require(std::isfinite(v) && v >= 0, "finite postReLU feature responses");
      maximum = std::max(maximum, double(v));
    }
    check::Require(maximum > 0, "nontrivial pretrained layer responses");
    peaks[l] = float(maximum);
    export_arrays[features.names[l]] = a;
    gallery << "<section><h2>" << features.names[l] << " : " << side << "x" << side << "x" << channels << "</h2><p>Shared maximum "
            << maximum
            << ";every numbered channel retained,including zero response maps.</p><table><tr><th>Shared layer scale</th><th>Per-channel "
               "scale</th></tr>";
    for (int b = 0; b < 2; ++b) {
      const std::string stem = features.names[l] + "_image" + std::to_string(b);
      sal::SaliencyPng(directory / (stem + "_shared.png"), Mosaic(av, a.shape(), b, true, maximum));
      sal::SaliencyPng(directory / (stem + "_channel.png"), Mosaic(av, a.shape(), b, false, maximum));
      gallery << "<tr><td><p>Image" << b << " shared</p><img src='" << stem << "_shared.png'></td><td><p>Image" << b
              << " per-channel</p><img src='" << stem << "_channel.png'></td></tr>";
      for (int c = 0; c < channels; ++c) {
        double sum = 0, peak = 0, positive = 0;
        for (int y = 0; y < side; ++y)
          for (int x = 0; x < side; ++x) {
            const double v = av[((std::size_t(b) * side + y) * side + x) * channels + c];
            sum += v;
            peak = std::max(peak, v);
            positive += v > 0;
          }
        stats << features.names[l] << '\t' << b << '\t' << c << '\t' << sum / (side * side) << '\t' << peak << '\t'
              << positive / (side * side) << '\n';
      }
    }
    gallery << "</table></section>";
    std::cout << features.names[l] << ":" << side << "x" << side << "x" << channels << ",sharedpeak=" << maximum
              << ",allchannels saved forbothimages\n";
  }
  export_arrays["layer_peaks"] = sal::FromF(peaks, {5});
  export_arrays["kernel_peak"] = nn::scalar(float(kernel_peak));
  sal::SaveArrays(directory / "features.npz", export_arrays);
  {
    nn::NoGradGuard guard;
    const auto      single = visual::VggVisualFeatures(vgg, nn::slice_axis(rgb, 0, 0, 1).contiguous());  // singleimage/batch
    for (int l = 0; l < 5; ++l) {
      const auto s = sal::F(single.activation[std::size_t(l)]), v = sal::F(values[l + 2]);
      for (std::size_t i = 0; i < s.size(); ++i) check::Near(s[i], v[i], .005, "single/batch pretrained feature agreement");
    }
  }
  vgg.CheckFrozen();
  sal::CopyManifests(directory, {{data / "manifest.json", "data_manifest.json"}, {data / "ATTRIBUTION.txt", "ATTRIBUTION.txt"}});
  gallery << "<p>Raw tensors:features.npz;all channel statistics:channels.tsv. Shared units are only comparable within a layer. No "
             "object-class prediction or causal attribution claim.</p>";
  check::Require(bool(stats) && bool(gallery), "write gallery/channel statistics");
  std::cout << "all" << probes << " firstconv hostprobes error=" << first_error
            << ";allRGB/BGR/ReLUvalues,single/batch and immutable20024384 frozenparameters passed\nFull16convolutions;1472channels "
               "perimage;allmaps+shared/perchannel scales+64learnedfilters;no training or class-attribution claim.\nArtifacts:"
            << directory.string() << "/index.html\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    nn::set_num_threads(4);
    std::filesystem::path data = paths::Dataset("filter_activations"), weights = paths::Dataset("style_transfer") / "vgg19.npz",
                          out = paths::OutputRoot() / "models/filter_activations";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("weights=", 0) == 0)
        weights = a.substr(8);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(a.substr(8)));
      else
        throw std::invalid_argument("expected data=PATH weights=PATH out=PATH threads=N");
    }
    RenderingChecks();
    Run(data, weights, out);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
