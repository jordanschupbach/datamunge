// CMD: just run-nn latent_arithmetic
// Frozen THREEepoch undercompleteAE;0newtraining. Prepare:just autoencoder-assets;just run undercomplete_autoencoder.
//
// Port of tflow's latent_arithmetic (helpers: examples/nn/common/latent_arithmetic.hpp).
// Data: $DATAMUNGE_DATA_DIR/saliency (tflow's `just autoencoder-assets`). The frozen model is
// read from encoder=DIR: best.npz of our undercomplete_autoencoder port (default
// output/nn/models/undercomplete_autoencoder) or tflow's model_arrays.npz when pointed at
// its output/models/undercomplete_autoencoder (reproduces tflow's arrays). The TF symbolic
// gradients of the float64 interpolation become nn::grad reverse passes. Outputs
// (analysis.npz, model_arrays.npz, five mosaics, metrics.json, index.html) go to
// output/nn/models/latent_arithmetic. Options: data=PATH encoder=PATH out=PATH.

#include <fstream>
#include <iostream>
#include <map>

#include "common/latent_arithmetic.hpp"

namespace {

namespace nn = datamunge::nn;

void Run(const std::filesystem::path& data, const std::filesystem::path& encoder, const std::filesystem::path& root) {
  auto pools = ae::LoadSaliency(data);
  std::filesystem::create_directories(root);
  ae::UndercompleteModel model(false);
  ae::LoadFrozen(model.parameters, encoder, "frozen3epochAEforlatentpaths");
  const auto before = ae::Snapshot(model.parameters);
  auto       fit = model.Evaluate(pools[0].images), test = model.Evaluate(pools[2].images);
  auto       means       = ae::LatentClassMeans(fit[0], pools[0].labels),
       pixel_means = ae::LatentClassMeans(ae::Flatten(pools[0].images), pools[0].labels);
  const auto test_labels = pools[2].labels.to_vector<int>(), test_source = pools[2].source.to_vector<int>();
  const auto test_code = test[0].contiguous().to_vector<float>(), mean_values = means.to_vector<float>();
  std::vector<int>   rows(10), source_ids(10), target_ids(10), target_rows(10);
  std::vector<float> a(320), b(320), directions(320), random(320), alpha(11);
  const std::vector<float> levels{-1, 0, .5f, 1, 2};
  for (int k = 0; k < 11; ++k) alpha[k] = k / 10.f;
  auto       standard = ae::GaussianNoise({10, 32}, 1, 4309);
  const auto z        = standard.to_vector<double>();
  for (int digit = 0; digit < 10; ++digit) {
    int source = 0, target = 0;
    while (test_labels[source] != digit) ++source;
    while (test_labels[target] != (digit + 1) % 10) ++target;
    rows[digit]        = source;
    target_rows[digit] = target;
    source_ids[digit]  = test_source[source];
    target_ids[digit]  = test_source[target];
    double length = 0, norm = 0;
    for (int j = 0; j < 32; ++j) {
      a[digit * 32 + j]          = test_code[source * 32 + j];
      b[digit * 32 + j]          = test_code[target * 32 + j];
      directions[digit * 32 + j] = mean_values[((digit + 1) % 10) * 32 + j] - mean_values[digit * 32 + j];
      length += std::pow(double(directions[digit * 32 + j]), 2);
      norm += std::pow(z[digit * 32 + j], 2);
    }
    check::Require(length > 0 && norm > 0, "nonzero true/random directions");
    for (int j = 0; j < 32; ++j) random[digit * 32 + j] = float(z[digit * 32 + j] * std::sqrt(length / norm));
  }
  auto A = nn::from_vector(a, {10, 32}), B = nn::from_vector(b, {10, 32}), T = nn::from_vector(alpha, {11}),
       L = nn::from_vector(levels, {5});
  nn::Tensor interpolation, reverse, constant, arithmetic, random_path;
  {
    nn::NoGradGuard guard;
    interpolation = ae::LatentLerp(A, B, T).contiguous();
    reverse       = ae::LatentLerp(B, A, T).contiguous();
    constant      = ae::LatentLerp(A, A, T).contiguous();
    arithmetic    = ae::LatentDirection(A, nn::from_vector(directions, {10, 32}), L).contiguous();
    random_path   = ae::LatentDirection(A, nn::from_vector(random, {10, 32}), L).contiguous();
  }
  auto A64 = nn::cast(A, nn::DType::Float64).requires_grad_(), B64 = nn::cast(B, nn::DType::Float64).requires_grad_(),
       T64 = nn::cast(T, nn::DType::Float64).requires_grad_();
  auto gradients = nn::grad(nn::sum(ae::LatentLerp(A64, B64, T64)), {A64, B64, T64});  // actual latent interpolation derivatives
  const auto ip = interpolation.to_vector<float>(), rp = reverse.to_vector<float>(), cp = constant.to_vector<float>(),
             ap = arithmetic.to_vector<float>();
  const auto ga = gradients[0].to_vector<double>(), gb = gradients[1].to_vector<double>();
  double     reversal_error = 0;
  for (int digit = 0; digit < 10; ++digit)
    for (int j = 0; j < 32; ++j) {
      check::Near(ip[(digit * 11 + 0) * 32 + j], a[digit * 32 + j], 0, "left latent endpoint EXACT");
      check::Near(ip[(digit * 11 + 10) * 32 + j], b[digit * 32 + j], 0, "right latent endpoint EXACT");
      check::Near(ap[(digit * 5 + 1) * 32 + j], a[digit * 32 + j], 0, "zero arithmetic scale EXACTsource");
      check::Near(ga[digit * 32 + j], 5.5, 1e-6, "SUMlerp derivative toleft");
      check::Near(gb[digit * 32 + j], 5.5, 1e-6, "SUMlerp derivative toright");
      for (int k = 0; k < 11; ++k) {
        reversal_error = std::max(reversal_error, std::abs(double(ip[(digit * 11 + k) * 32 + j]) -
                                                           rp[(digit * 11 + 10 - k) * 32 + j]));
        check::Near(cp[(digit * 11 + k) * 32 + j], a[digit * 32 + j], 1e-6,
                    "samecode interpolation staysconstant withinF32rounding");
      }
    }
  check::Require(reversal_error < 3e-6, "reversepath symmetry withinF32alpha rounding");
  auto interpolated      = model.Decode(ae::LatentFlatten(interpolation)),
       arithmetic_images = model.Decode(ae::LatentFlatten(arithmetic)),
       random_images = model.Decode(ae::LatentFlatten(random_path)), endpoint_a = model.Decode(A),
       endpoint_b = model.Decode(B);
  const auto images = pools[2].images.contiguous().to_vector<float>(), pm = pixel_means.to_vector<float>(),
             iv = interpolated.to_vector<float>(), ea = endpoint_a.to_vector<float>(), eb = endpoint_b.to_vector<float>();
  std::vector<float> pixel_blends(110 * 784), decoder_blends(110 * 784), pixel_arithmetic(50 * 784);
  double             nonlinearity   = 0;
  int                negative_codes = 0, clipped_pixels = 0;
  for (int d = 0; d < 10; ++d) {
    const int row = rows[d], target = target_rows[d];
    for (int k = 0; k < 11; ++k)
      for (int j = 0; j < 784; ++j) {
        const float t = alpha[k];
        const auto  o = (d * 11 + k) * 784 + j;
        pixel_blends[o]   = (1 - t) * images[row * 784 + j] + t * images[target * 784 + j];
        decoder_blends[o] = (1 - t) * ea[d * 784 + j] + t * eb[d * 784 + j];
        nonlinearity      = std::max(nonlinearity, std::abs(double(iv[o]) - decoder_blends[o]));
        if (k == 0) check::Near(iv[o], ea[d * 784 + j], 2e-6, "leftdecodedendpointbatchparity");
        if (k == 10) check::Near(iv[o], eb[d * 784 + j], 2e-6, "rightdecodedendpointbatchparity");
      }
    for (int k = 0; k < 5; ++k)
      for (int j = 0; j < 784; ++j) {
        const float value = images[row * 784 + j] + levels[k] * (pm[((d + 1) % 10) * 784 + j] - pm[d * 784 + j]);
        clipped_pixels += value < 0 || value > 1;
        pixel_arithmetic[(d * 5 + k) * 784 + j] = std::clamp(value, 0.f, 1.f);
      }
  }
  for (float v : ap) negative_codes += v < 0;
  check::Require(nonlinearity > 1e-3, "sigmoiddecoder makes code-interpolationdifferent from pixelblend of endpoints");
  check::Require(before == ae::Snapshot(model.parameters), "all code-only interpolation/arithmetic calls immutable");
  auto ints = [](const std::vector<int>& v) { return nn::from_vector(v, {std::int64_t(v.size())}); };
  auto pixel_blend_t = nn::from_vector(pixel_blends, {110, 784}), pixel_arithmetic_t = nn::from_vector(pixel_arithmetic, {50, 784});
  std::map<std::string, nn::Tensor> arrays{{"fit_images", pools[0].images},
                                           {"fit_labels", pools[0].labels},
                                           {"fit_source_ids", pools[0].source},
                                           {"test_images", pools[2].images},
                                           {"test_labels", pools[2].labels},
                                           {"test_source_ids", pools[2].source},
                                           {"fit_code", fit[0]},
                                           {"test_code", test[0]},
                                           {"fit_class_code_means", means},
                                           {"fit_class_pixel_means", pixel_means},
                                           {"selected_test_rows", ints(rows)},
                                           {"source_ids", ints(source_ids)},
                                           {"target_ids", ints(target_ids)},
                                           {"a", A},
                                           {"b", B},
                                           {"directions", nn::from_vector(directions, {10, 32})},
                                           {"random_directions", nn::from_vector(random, {10, 32})},
                                           {"random_standard", standard},
                                           {"alpha", T},
                                           {"levels", L},
                                           {"interpolated_codes", interpolation},
                                           {"reverse_codes", reverse},
                                           {"constant_codes", constant},
                                           {"arithmetic_codes", arithmetic},
                                           {"random_codes", random_path},
                                           {"gradient_a", gradients[0]},
                                           {"gradient_b", gradients[1]},
                                           {"gradient_alpha", gradients[2]},
                                           {"interpolated_images", interpolated},
                                           {"arithmetic_images", arithmetic_images},
                                           {"random_images", random_images},
                                           {"pixel_blends", pixel_blend_t},
                                           {"decoder_endpoint_blends", nn::from_vector(decoder_blends, {110, 784})},
                                           {"pixel_arithmetic_clipped", pixel_arithmetic_t},
                                           {"endpoint_a", endpoint_a},
                                           {"endpoint_b", endpoint_b}};
  nn::save_npz((root / "analysis.npz").string(), nn::NamedTensors(arrays.begin(), arrays.end()));
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(model.parameters));
  if (std::filesystem::exists(data / "manifest.json"))
    std::filesystem::copy_file(data / "manifest.json", root / "data_manifest.json",
                               std::filesystem::copy_options::overwrite_existing);
  ae::Mosaic3(root / "latent_interpolations.png", interpolated, 11);
  ae::Mosaic3(root / "pixel_interpolations.png", pixel_blend_t, 11);
  ae::Mosaic3(root / "latent_arithmetic.png", arithmetic_images, 5);
  ae::Mosaic3(root / "random_arithmetic.png", random_images, 5);
  ae::Mosaic3(root / "pixel_arithmetic.png", pixel_arithmetic_t, 5);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"additional_training_updates\":0,\"parameters\":50992,\"pairs\":10,\"interpolation_steps\":11,\"arithmetic_"
          "steps\":5,\"decoder_nonlinearity_max_difference\":"
       << nonlinearity << ",\"reverse_code_max_error\":" << reversal_error
       << ",\"negative_arithmetic_code_entries\":" << negative_codes
       << ",\"pixel_arithmetic_clipped_entries\":" << clipped_pixels << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Latent interpolation and arithmetic</title><style>body{font:16px "
          "system-ui;margin:30px;max-width:1200px}img{max-width:100%;image-rendering:pixelated}</style><h1>Latent interpolation "
          "and arithmetic</h1><p>Frozen THREEepochMNISTAE784-32ReLU-784sigmoid,50992params,0newtraining. No generative prior or "
          "semantic-factor guarantee. Rowd selects FIRSToriginalheldout digitd and FIRSTdigit(d+1)%10,labels used only for "
          "fixedcase selection. Everypair shown unfiltered. Eleven t=0,.1,...1codes(1-t)za+t*zb. Endpoints exact,reversepath "
          "agrees withinF32rounding,samecode path staysconstant;true reverse-mode derivatives checked. Decoder-only inference,no "
          "images fed to decoder.</p><h2>Decoded latent paths,10unfiltered pairs</h2><img src='latent_interpolations.png'><h2>"
          "Originalpixel crossfade baseline,SAMEpairs</h2><img src='pixel_interpolations.png'><p>Smoothly varying probabilities "
          "do not demonstrate semantic morphing. Sigmoiddecoder means latent-interpolateddecode differs from blending "
          "decodedendpoints,maxdifference="
       << nonlinearity
       << ". Original endpoint images are not claimed to reconstruct exactly. Fixed0..1gray,not perimage contrast.</p><h2>"
          "Label-centroid latent arithmetic</h2><p>FittingONLY1000labels define class mean32Dcodes;direction=mean(nextclass)-"
          "mean(sourceclass). Apply to FIRSTheldoutsourceclasscase at levels[-1,0,.5,1,2]. This is SUPERVISEDposthoc direction "
          "construction,not discovered unsupervised disentanglement. ZEROscale exactlysourcecode. Extrapolated latent coordinates "
          "remain UNCLIPPED,including"
       << negative_codes
       << "negativeentries outside ReLUencoder range. No projection onto trained code distribution.</p><img "
          "src='latent_arithmetic.png'><h2>Random direction control,matchednorm perrow</h2><img "
          "src='random_arithmetic.png'><p>Gaussianseed4309directions normalized to each classdifference norm,not chosen by "
          "outputquality. If classarithmetic resembles random changes,it is not evidence of semantic transport.</p><h2>"
          "Fittingclass pixel-mean direction baseline</h2><img src='pixel_arithmetic.png'><p>Originalsourcepixels+level*("
          "fitpixelmean(next)-fitpixelmean(source)),clipped0..1for display;"
       << clipped_pixels
       << "components clipped. This baseline has different geometry;arrays retain source means/codes. Allfiveresultgrids "
          "unfiltered;blurry/ambiguous/failure outputs retained. No classifier/digit-fidelity metric or target-success claim. "
          "AEcoordinate counts are not codec bitrates.</p>";
  check::Require(bool(meta) && bool(html), "complete latent arithmetic artifacts");
  std::cout << "0newupdates,10pairs11steps/10directions5levels;nonlinearity=" << nonlinearity
            << ",negativecodes=" << negative_codes << ",reverseerror=" << reversal_error
            << ";endpoints/actualgradients/zeroScale/codeOnly/immutable PASS\nArtifacts:" << (root / "index.html").string()
            << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path data    = paths::Dataset("saliency"),
                          encoder = paths::OutputRoot() / "models/undercomplete_autoencoder",
                          out     = paths::OutputRoot() / "models/latent_arithmetic";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("encoder=", 0) == 0)
        encoder = a.substr(8);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else
        throw std::invalid_argument("expected data=PATH encoder=PATH out=PATH");
    }
    Run(data, encoder, out);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
