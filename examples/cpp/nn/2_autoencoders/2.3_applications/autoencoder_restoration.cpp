// CMD: just run-nn autoencoder_restoration
// Prepare:just autoencoder-assets;just run denoising_autoencoder;just run masked_autoencoder.
// All preceding training examples use THREEepochs by default;this application adds0updates.
//
// Port of tflow's autoencoder_restoration (helpers: examples/nn/common/autoencoder_restoration.hpp).
// Data: $DATAMUNGE_DATA_DIR/saliency. The three frozen models are read from models=DIR
// (default output/nn/models): DIR/denoising_autoencoder/{denoising,ordinary} and
// DIR/masked_autoencoder, as best.npz written by our training ports or, when DIR is tflow's
// output/models, its model_arrays.npz files (the restored best TF checkpoints), which
// reproduce tflow's numbers. Outputs (analysis.npz, denoising.tsv, {denoising,ordinary,masked}
// _arrays.npz, metrics.json, index.html, case_<k>.png) go to
// output/nn/models/autoencoder_restoration. Options: data=PATH models=DIR out=PATH.

#include <fstream>
#include <iostream>
#include <map>

#include "common/autoencoder_restoration.hpp"

namespace {

namespace nn = datamunge::nn;

void Run(const std::filesystem::path& data, const std::filesystem::path& models, const std::filesystem::path& root) {
  auto pools = ae::LoadSaliency(data);
  std::filesystem::create_directories(root);
  ae::DenoisingModel         denoising(false), ordinary(false);
  ae::MaskedAutoencoderModel masked(false);
  ae::LoadFrozen(denoising.parameters, models / "denoising_autoencoder/denoising",
                 "frozen corrupted-input clean-target3epoch model");
  ae::LoadFrozen(ordinary.parameters, models / "denoising_autoencoder/ordinary", "frozen matched ordinary3epoch control");
  ae::LoadFrozen(masked.parameters, models / "masked_autoencoder", "frozen3epoch visible-only transformer MAE");
  const auto db = ae::Snapshot(denoising.parameters), ob = ae::Snapshot(ordinary.parameters),
             mb = ae::Snapshot(masked.parameters);
  const auto clean = pools[2].images;
  std::map<std::string, nn::Tensor> arrays;
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    arrays.emplace(split + "_images", pools[k].images);
    arrays.emplace(split + "_labels", pools[k].labels);
    arrays.emplace(split + "_source_ids", pools[k].source);
  }
  const auto standard = ae::GaussianNoise(clean.shape(), 1, 4229);
  arrays.emplace("standard_noise", standard);
  const auto mean = ae::PixelMean(pools[0].images), mean_images = ae::Constant(mean, 200);
  arrays.emplace("fit_pixel_mean", mean);
  arrays.emplace("mean_images", mean_images);
  std::ofstream table(root / "denoising.tsv");
  table.precision(17);
  table << "condition\tsigma\tnoisy_input_mse\tdenoising_mse\tordinary_mse\tfit_mean_mse\tdenoising_global_psnr_db\n";
  const std::vector<std::string> conditions{"clean", "low", "trained_sigma", "high"};
  const float                    sigmas[] = {0, .15f, .35f, .6f};
  for (int k = 0; k < 4; ++k) {
    auto input = ae::CorruptForDenoising(clean, standard, sigmas[k]);
    auto d = denoising.Predict(input), o = ordinary.Predict(input);
    if (!k) check::Require(ae::Bytes(input) == ae::Bytes(clean), "sigma0exactcleanidentity");
    arrays.emplace(conditions[k] + "_input", input);
    arrays.emplace(conditions[k] + "_denoising", d[1]);
    arrays.emplace(conditions[k] + "_ordinary", o[1]);
    const double dm = ae::Mse(d[1], clean), raw = ae::Mse(input, clean), om = ae::Mse(o[1], clean);
    table << conditions[k] << '\t' << sigmas[k] << '\t' << raw << '\t' << dm << '\t' << om << '\t'
          << ae::Mse(mean_images, clean) << '\t' << 10 * std::log10(1 / dm) << '\n';
    std::cout << conditions[k] << ":inputMSE=" << raw << ",denoised=" << dm << ",ordinary=" << om << std::endl;
  }
  const auto selection = ae::MaeSelection(pools[2].source, 0, 4243), observed = ae::VisibleImage(clean, selection, 0),
             changed   = ae::VisibleImage(clean, selection, 1);
  const auto inferred = ae::RestoreVisible(masked, observed, selection), probe = ae::RestoreVisible(masked, changed, selection),
             clean_probe = ae::RestoreVisible(masked, clean, selection);
  check::Require(ae::Bytes(inferred[0]) == ae::Bytes(probe[0]) && ae::Bytes(inferred[1]) == ae::Bytes(probe[1]) &&
                     ae::Bytes(inferred[1]) == ae::Bytes(clean_probe[1]),
                 "actual MAEgraph predictions BITEXACTindependent of ALLhiddenpixel values");
  const auto hidden_prediction = ae::MaeImage(inferred[1]),
             composite         = ae::InpaintComposite(observed, hidden_prediction, selection),
             mean_fill         = ae::InpaintComposite(observed, mean_images, selection);
  const auto d_masked = denoising.Predict(observed)[1], d_fill = ae::InpaintComposite(observed, d_masked, selection);
  {
    const auto c = composite.to_vector<float>(), x = clean.contiguous().to_vector<float>(),
               s = selection.to_vector<float>();
    for (int b = 0; b < 200; ++b)
      for (int p = 0; p < 16; ++p)
        if (!ae::MaeHidden(s, b, p))
          for (int i = 0; i < 49; ++i) {
            const int index = b * 784 + ae::MaePixel(p, i);
            check::Near(c[index], x[index], 0, "observed pixels copied BITEXACT");
          }
  }
  for (const auto& [key, value] : std::map<std::string, nn::Tensor>{{"selection", selection},
                                                                    {"observed", observed},
                                                                    {"hidden_changed", changed},
                                                                    {"mae_code", inferred[0]},
                                                                    {"mae_patch_prediction", inferred[1]},
                                                                    {"mae_hidden_prediction", hidden_prediction},
                                                                    {"mae_composite", composite},
                                                                    {"mean_composite", mean_fill},
                                                                    {"denoising_masked_prediction", d_masked},
                                                                    {"denoising_composite", d_fill}})
    arrays.emplace(key, value);
  check::Require(db == ae::Snapshot(denoising.parameters) && ob == ae::Snapshot(ordinary.parameters) &&
                     mb == ae::Snapshot(masked.parameters),
                 "all restoration calls leave THREE models immutable");
  nn::save_npz((root / "analysis.npz").string(), nn::NamedTensors(arrays.begin(), arrays.end()));
  nn::save_npz((root / "denoising_arrays.npz").string(), ae::Arrays(denoising.parameters));
  nn::save_npz((root / "ordinary_arrays.npz").string(), ae::Arrays(ordinary.parameters));
  nn::save_npz((root / "masked_arrays.npz").string(), ae::Arrays(masked.parameters));
  if (std::filesystem::exists(data / "manifest.json"))
    std::filesystem::copy_file(data / "manifest.json", root / "data_manifest.json",
                               std::filesystem::copy_options::overwrite_existing);
  const double hidden_mse = ae::HiddenRestorationMse(composite, clean, selection),
               mean_mse   = ae::HiddenRestorationMse(mean_fill, clean, selection),
               d_mse      = ae::HiddenRestorationMse(d_fill, clean, selection),
               zero_mse   = ae::HiddenRestorationMse(observed, clean, selection);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"additional_training_updates\":0,\"denoising_parameters\":" << denoising.parameters.count()
       << ",\"ordinary_parameters\":" << ordinary.parameters.count() << ",\"masked_parameters\":" << masked.parameters.count()
       << ",\"test_records\":200,\"visible_patches\":4,\"hidden_patches\":12,\"hidden_pixels_per_case\":588,\"mae_hidden_mse\":"
       << hidden_mse << ",\"fit_mean_hidden_mse\":" << mean_mse << ",\"denoising_hidden_mse\":" << d_mse
       << ",\"zero_hidden_mse\":" << zero_mse << ",\"mae_composite_full_mse\":" << ae::Mse(composite, clean)
       << ",\"hidden_values_prediction_max_effect\":0,\"observed_composite_max_error\":0}\n";
  check::Require(denoising.parameters.count() == 50992 && masked.parameters.count() == 13649, "50992/13649 inventories");
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Autoencoder denoising and inpainting application</title><style>body{"
          "font:16px system-ui;margin:30px;max-width:1300px}img{max-width:100%;image-rendering:pixelated}</style><h1>Image "
          "denoising and inpainting</h1><p>Reuse THREEepoch frozen denoisingAE+matchedordinaryAE(50992paramsEACH),"
          "maskedtransformerMAE13649params. ZEROnewoptimizer updates. Real MNIST1000fit200validation200test,clean targets used "
          "ONLY for evaluation here. Native inference feeds corrupted image ONLY to DAE;MAE feeds observedpixels+mask ONLY and "
          "fetches code/prediction without hidden-target loss. Changing ALLhidden values0to1orclean leaves actual MAE outputs "
          "BITEXACT;composite copies visible pixels BITEXACT.</p><p>Denoising:NEWfixedGaussianseed4229,addsigma0/.15/.35/.6"
          "thenF32cast+clip0..1,same standard noise across severities. .35matches original trainingnoise;other severities are "
          "sensitivity controls,not selected hyperparameters. Compare rawnoisyinput,DAE,matchedordinaryAE,fit-onlypixelmean. "
          "See <a href='denoising.tsv'>all four measured conditions</a>. PSNR=10log10(1/GLOBALMSE),not average perimage PSNR. "
          "Low-noise identity can beat a blurry learned model;failures retained.</p><p>Inpainting:NEWdeterministic "
          "source-keyedmaskseed4243,visible4of16nonoverlap7x7patches,hidden75%. Fixedmasklocations supplied,no hidden pixels. "
          "RawMAE prediction fills hidden sites;visible original pixels copied. Hidden-only denominator588percase;"
          "fullcompositeMSE=.75hiddenMSE. HiddenMSE MAE="
       << hidden_mse << ",fit-onlymean=" << mean_mse << ",DAEonmaskedimage=" << d_mse << ",zerofill=" << zero_mse
       << ". DAE was trained on Gaussian corruption,not missingpatches;cross-task result is an explicit mismatch baseline,not "
          "a fair architecture ranking. No photorealism or semantic completion claim.</p><p>Ten evenlyspacedunfilteredheldout"
          "cases,columns clean/noisy(.35)/DAE/ordinary/maskedinput/MAEcomposite/fitmeanfill/DAEfill. Fixed0..1gray,not perimage "
          "normalization. Outputs are probabilities,not pixel draws. Entire dataset scores retained;no label-based quality "
          "filtering.</p>";
  const auto source = pools[2].source.to_vector<int>(), labels = pools[2].labels.to_vector<int>();
  for (int k = 0; k < 10; ++k) {
    const int b = k * 199 / 9;
    ae::CaseStrip3(root / ("case_" + std::to_string(k) + ".png"),
                   {clean, arrays.at("trained_sigma_input"), arrays.at("trained_sigma_denoising"),
                    arrays.at("trained_sigma_ordinary"), observed, composite, mean_fill, d_fill},
                   b);
    html << "<p>Source" << source[b] << ",digit" << labels[b] << "<br><img src='case_" << k << ".png'></p>";
  }
  check::Require(bool(meta) && bool(html) && bool(table), "complete restoration artifacts");
  std::cout << "hidden-only MSE MAE=" << hidden_mse << ",mean=" << mean_mse << ",DAEmismatch=" << d_mse << ",zero=" << zero_mse
            << ";targetFree/hiddenPerturbationBITEXACT/visibleCopy/immutable3models PASS\nArtifacts:"
            << (root / "index.html").string() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path data = paths::Dataset("saliency"), models = paths::OutputRoot() / "models",
                          out  = paths::OutputRoot() / "models/autoencoder_restoration";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("models=", 0) == 0)
        models = a.substr(7);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else
        throw std::invalid_argument("expected data=PATH models=DIR out=PATH");
    }
    Run(data, models, out);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
