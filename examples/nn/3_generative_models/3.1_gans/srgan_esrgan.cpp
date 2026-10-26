// CMD: just run-nn srgan_esrgan
// SRGAN (Ledig et al. 2017) and ESRGAN (Wang et al. 2018): compact paired x2
// RGB super-resolution on CIFAR-10 with a frozen pretrained VGG19 first-layer
// content loss and the bicubic baseline.
//
// Port of tflow's srgan_esrgan. Data: CIFAR-10 binaries from
// $DATAMUNGE_DATA_DIR/cifar10 (data_batch_1.bin rows 0..127 fit, 128..159
// validation; test_batch.bin rows 64..95 test). HR 32x32 -> LR 16x16 by
// antialiased Keys bicubic (a = -.5, half-pixel, symmetric edges), bicubic x2
// back up. Content features: VGG19 block1_conv1 (1792 frozen values) from
// features=PATH: the full $DATAMUNGE_DATA_DIR/style_transfer/vgg19.npz (tflow
// `just style-transfer-assets`) if present, otherwise the two arrays tflow
// recorded in output/models/srgan_esrgan/frozen_features.npz (identical values).
//
// Two declared CPU teaching adaptations as tflow: width 4 and a bicubic IMAGE
// residual shortcut with a small N(0, .001) output kernel. SRGAN: two BN/PReLU
// residual blocks, global feature skip, CRD pixel shuffle x2 + PReLU (1591
// parameters + 40 BN statistics); D 1673 (conv downsampling, global pool, raw
// score), logistic GAN; content = post-ReLU feature MSE / 255, G = content +
// .001 adversarial. ESRGAN: RRDB of THREE 5-layer dense blocks (growth 2), .2
// residual scaling, no BN, nearest x2 + conv / HR conv (3511 parameters); D
// 1672 without the score bias; relativistic average GAN; content =
// pre-activation feature L1, G = content + .005 adversarial + .01 L1 pixel.
// Schedule (both): epoch 1 = pixel-loss warm-up of G only (D untouched), then
// the Adam states (.0001, beta1 .9) are RESET and every later epoch runs a D
// step then a G step per batch; 8-row batches, 16 per epoch, BN moving
// averages (momentum .9, unbiased variance) updated from the moments captured
// with each G gradient. Three default epochs, fixed final checkpoint.
//
// Data and batch order are tflow's exactly; weights use datamunge's
// initializer unless init=DIR names tflow's output/models/srgan_esrgan (then
// srgan/ and esrgan/initial_arrays.npz are replayed and the runs follow TF's
// trajectories up to float rounding). tflow's per-step .npz dumps (first_*,
// epoch*_arrays/optimizer_state/preview, gan_start_*), its independent audit
// and index.html are not written. Outputs (data.npz, frozen_features.npz,
// relativistic_algebra.npz and per model srgan/ esrgan/: history, batches,
// initial/final checkpoint .npz, final_first_forward.npz,
// final_gradients_group*.npz, outputs.npz, target_canary.npz, comparison.png,
// metrics.json) go to output/nn/models/srgan_esrgan.
// Options: epochs=3 data=PATH features=PATH out=PATH init=DIR.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "common/adversarial_super_resolution.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using gan::Arrays;

namespace {

double Value(const Arrays& a, const std::string& key) { return a.at(key).item(); }

/// Unequal-batch RaGAN fixture: closed-form partials and common logit-shift invariance.
void RaAlgebra(const std::filesystem::path& root) {
  Tensor real = nn::tensor({-2, 0, 3}, {3, 1}), fake = nn::tensor({-1, 2}, {2, 1});
  real.requires_grad_();
  fake.requires_grad_();
  const auto loss = srgan::RelativisticAverageSR(real, fake);
  nn::GradOptions o;
  o.retain_graph  = true;
  const auto dg   = nn::grad(loss.discriminator, {real, fake}, o);
  const auto gg   = nn::grad(loss.generator, {real, fake}, o);
  const auto shifted = srgan::RelativisticAverageSR(real.detach() + 5.0, fake.detach() + 5.0);
  check::Near(loss.discriminator.item(), shifted.discriminator.item(), 3e-7, "Ra D common logit shift invariance");
  check::Near(loss.generator.item(), shifted.generator.item(), 3e-7, "Ra G common logit shift invariance");
  gan::SaveArrays((root / "relativistic_algebra.npz").string(), {{"real", real.detach()},
                                                                 {"fake", fake.detach()},
                                                                 {"d_loss", loss.discriminator.detach()},
                                                                 {"g_loss", loss.generator.detach()},
                                                                 {"relative_real", loss.real_relative.detach()},
                                                                 {"relative_fake", loss.fake_relative.detach()},
                                                                 {"d_real_gradient", dg[0]},
                                                                 {"d_fake_gradient", dg[1]},
                                                                 {"g_real_gradient", gg[0]},
                                                                 {"g_fake_gradient", gg[1]}});
  std::cout << "RaGAN algebra: D=" << loss.discriminator.item() << ", G=" << loss.generator.item()
            << ", dG/dreal=" << gg[0].values_str() << '\n';
}

/// 8 test crops (20x20 centre at 4x): bicubic | prediction | HR | 8 |Y error| (red).
void Preview(const sr::SRData& data, const Tensor& prediction, const std::filesystem::path& path) {
  const auto         p = prediction.contiguous().to_vector<float>(), hr = data.hr.contiguous().to_vector<float>(),
             bi = data.bicubic.contiguous().to_vector<float>();
  std::vector<float> image(640 * 320 * 3, 0.f);
  auto at = [](const std::vector<float>& v, int n, int y, int x, int c) { return double(v[((n * 32 + y) * 32 + x) * 3 + c]); };
  for (int n = 0; n < 8; ++n)
    for (int y = 0; y < 20; ++y)
      for (int x = 0; x < 20; ++x) {
        const int    yy = y + 6, xx = x + 6;
        const double truth_y = sr::SRLuma(at(hr, n, yy, xx, 0), at(hr, n, yy, xx, 1), at(hr, n, yy, xx, 2));
        const double estimated_y = sr::SRLuma(at(p, n, yy, xx, 0), at(p, n, yy, xx, 1), at(p, n, yy, xx, 2));
        for (int col = 0; col < 4; ++col)
          for (int c = 0; c < 3; ++c) {
            const double value = col == 0   ? at(bi, n, yy, xx, c)
                                 : col == 1 ? at(p, n, yy, xx, c)
                                 : col == 2 ? at(hr, n, yy, xx, c)
                                 : c == 0   ? 8 * std::abs(estimated_y - truth_y)
                                            : 0;
            const float byte = float(std::floor(255 * std::clamp(value, 0., 1.) + .5));
            for (int dy = 0; dy < 4; ++dy)
              for (int dx = 0; dx < 4; ++dx) image[((n * 80 + y * 4 + dy) * 320 + col * 80 + x * 4 + dx) * 3 + c] = byte;
          }
      }
  nn::save_png(nn::from_vector(image, {640, 320, 3}), path.string(), 0, 255);
}

void Train(const std::filesystem::path& root, const std::vector<sr::SRData>& pools, const Arrays& features, int epochs,
           bool enhanced, const std::string& init) {
  const auto dir = root / (enhanced ? "esrgan" : "srgan");
  const char* label = enhanced ? "ESRGAN" : "SRGAN";
  std::filesystem::create_directories(dir);
  srgan::AdversarialSRModel model(enhanced, features);
  if (!init.empty()) model.Load((std::filesystem::path(init) / (enhanced ? "esrgan" : "srgan") / "initial_arrays.npz").string());
  check::Require(model.Count(0) == (enhanced ? 3511 : 1591) && model.Count(1) == (enhanced ? 1672 : 1673) &&
                     model.Count(2) == (enhanced ? 0 : 40),
                 "compact SRGAN/ESRGAN parameter census");
  const auto initial = model.AllArrays();
  gan::SaveArrays((dir / "initial_arrays.npz").string(), initial);
  gan::SaveArrays((dir / "initial_outputs.npz").string(), model.Evaluate(pools[2].lr, pools[2].hr, false));
  model.CheckFrozen();
  nn::BatchOrder order(128, 8, 7103);
  int            gsteps = 0, dsteps = 0;
  std::ofstream  history(dir / "history.tsv"), batches(dir / "batches.tsv");
  history.precision(17);
  history << "epoch\tadversarial_phase\tg_loss\tgan_loss\tpixel_loss\tcontent_loss\tadversarial_loss\td_loss\n";
  batches << "epoch\tstep\trow\n";
  Tensor     first_lr, first_hr;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    const bool gan = epoch >= 2;
    if (epoch == 2) {
      const auto before = model.AllArrays();
      model.ResetOptimizers();
      check::Require(gan::BitEqual(before, model.AllArrays()), "warmup-to-GAN optimizer RESET never changes weights/BN");
    }
    order.begin_epoch();
    int epoch_step = 0;
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const Tensor x = sr::SRBatch(pools[0].lr, rows), y = sr::SRBatch(pools[0].hr, rows);
      if (gan && epoch_step == 0 && epoch == 2) first_lr = x, first_hr = y;
      for (int group : gan ? std::vector<int>{1, 0} : std::vector<int>{0}) {
        const auto captured = model.Capture(group, x, y, gan);
        const auto before   = model.AllArrays();
        model.Apply(group, captured);
        const auto after = model.AllArrays();
        for (const auto& [key, value] : before) {
          const int owner = key[0] - '0';
          if (owner != group && !(group == 0 && owner == 2))
            check::Require(gan::BitEqual(value, after.at(key)), "SR phase isolation / frozen D during warmup / G during D");
        }
        (group ? dsteps : gsteps)++;
      }
      for (auto row : rows) batches << epoch << '\t' << gsteps - 1 << '\t' << row << '\n';
      ++epoch_step;
    }
    check::Require(epoch_step == 16, "16 cheap fit batches / epoch");
    if (!gan) {
      const auto after = model.AllArrays();
      for (const auto& [key, value] : initial)
        if (key[0] == '1') check::Require(gan::BitEqual(value, after.at(key)), "critic COMPLETELY untrained during reconstruction warmup");
    }
    model.CheckFrozen();
    const auto v = model.Evaluate(pools[1].lr, pools[1].hr, gan);
    history << epoch << '\t' << gan;
    for (auto key : {"g_loss", "gan_loss", "pixel_loss", "content_loss", "adversarial_loss", "d_loss"})
      history << '\t' << Value(v, key);
    history << '\n';
    std::cout << label << " epoch " << epoch << ": " << (gan ? "GAN" : "reconstruction-warmup")
              << ", pixel/content/adv=" << Value(v, "pixel_loss") << '/' << Value(v, "content_loss") << '/'
              << Value(v, "adversarial_loss") << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  const auto   checkpoint = (dir / "final.npz").string();
  model.Save(checkpoint);
  gan::SaveArrays((dir / "final_first_forward.npz").string(), model.Evaluate(first_lr, first_hr, true));
  for (int group = 0; group < 2; ++group)
    gan::SaveArrays((dir / ("final_gradients_group" + std::to_string(group) + ".npz")).string(),
                    model.Capture(group, first_lr, first_hr, true));
  srgan::AdversarialSRModel deployed(enhanced, features, false, {1, 2, 3});
  deployed.Load(checkpoint);
  check::Require(gan::BitEqual(model.AllArrays(), deployed.AllArrays()), "all SR weights + BN restore BIT EXACT");
  const auto   before     = deployed.AllArrays();
  const auto   output     = deployed.Evaluate(pools[2].lr, pools[2].hr, true);
  const Tensor prediction = deployed.Predict(pools[2].lr);
  check::Require(gan::BitEqual(prediction, output.at("prediction")), "LR ONLY prediction without HR/phase exact");
  const double parity = gan::MaxAbsDiff(deployed.Predict(sr::SRBatch(pools[2].lr, {0})), sr::SRBatch(prediction, {0}));
  check::Require(parity < 3e-6, "LR single/batch parity within FP32");
  const auto canary = deployed.Evaluate(pools[2].lr, nn::zeros(pools[2].hr.shape()), true);
  check::Require(gan::BitEqual(canary.at("prediction"), prediction) &&
                     gan::BitEqual(canary.at("prediction_training"), output.at("prediction_training")),
                 "changing ALL HR targets leaves BOTH G paths BIT EXACT");
  gan::SaveArrays((dir / "target_canary.npz").string(), {{"prediction", canary.at("prediction")},
                                                         {"prediction_training", canary.at("prediction_training")},
                                                         {"content_loss", canary.at("content_loss")},
                                                         {"pixel_loss", canary.at("pixel_loss")}});
  deployed.CheckFrozen();
  check::Require(gan::BitEqual(before, deployed.AllArrays()), "immutable LR inference/evaluation/FROZEN pretrained features");
  gan::SaveArrays((dir / "outputs.npz").string(), output);
  const auto score = sr::SRScore(prediction, pools[2]);
  Preview(pools[2], prediction, dir / "comparison.png");
  std::ofstream meta(dir / "metrics.json");
  meta.precision(17);
  meta << "{\"enhanced\":" << (enhanced ? "true" : "false") << ",\"generator_parameters\":" << model.Count(0)
       << ",\"discriminator_parameters\":" << model.Count(1) << ",\"bn_state_values\":" << model.Count(2)
       << ",\"frozen_vgg_values\":1792,\"epochs\":" << epochs << ",\"warmup_epochs\":1,\"generator_updates\":" << gsteps
       << ",\"discriminator_updates\":" << dsteps << ",\"selected_epoch_fixed_final\":" << epochs
       << ",\"test_mean_psnr\":" << score.psnr << ",\"test_mean_ssim\":" << score.ssim
       << ",\"test_pooled_psnr\":" << sr::SRPSNR(score.mse) << ",\"bicubic_mean_psnr\":" << score.baseline_psnr
       << ",\"bicubic_mean_ssim\":" << score.baseline_ssim << ",\"single_batch_max_error\":" << parity;
  for (auto key : {"g_loss", "gan_loss", "pixel_loss", "content_loss", "adversarial_loss", "d_loss"})
    meta << ",\"test_" << key << "\":" << Value(output, key);
  meta << ",\"training_seconds\":" << seconds << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "complete SRGAN records");
  std::cout << label << ": all 32 test mean PSNR/SSIM=" << score.psnr << '/' << score.ssim
            << ", bicubic=" << score.baseline_psnr << '/' << score.baseline_ssim
            << "; frozen VGG/warmup+GAN/restore/LR only/target canary/immutable PASS (training " << seconds << " s)"
            << std::endl;
}

void Run(const std::filesystem::path& data, const std::filesystem::path& feature_path, const std::filesystem::path& root,
         int epochs, const std::string& init) {
  std::filesystem::create_directories(root);
  const auto training = data::LoadCifar({data / "data_batch_1.bin"}, data::Cifar::k10),
             testing  = data::LoadCifar({data / "test_batch.bin"}, data::Cifar::k10);
  check::Require(training.size() == 10000 && testing.size() == 10000, "canonical CIFAR binary sizes");
  std::vector<sr::SRData> pools;
  Arrays                  records;
  for (int k = 0; k < 3; ++k) {
    const int                 n = k == 0 ? 128 : 32, start = k == 0 ? 0 : k == 1 ? 128 : 64;
    std::vector<std::int64_t> rows;
    for (int i = 0; i < n; ++i) rows.push_back(start + i);
    const auto& source = k == 2 ? testing : training;
    pools.push_back(sr::SRPairs(source, rows));
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    records.emplace(split + "_hr", pools.back().hr);
    records.emplace(split + "_lr", pools.back().lr);
    records.emplace(split + "_bicubic", pools.back().bicubic);
    std::vector<std::int32_t> ids;
    for (int i = 0; i < n; ++i) ids.push_back(start + i + (k == 2 ? 50000 : 0));
    records.emplace(split + "_source_ids", nn::from_vector(ids, {n}));
    records.emplace(split + "_labels", nn::take_rows(source.labels, rows));
  }
  gan::SaveArrays((root / "data.npz").string(), records);
  check::Require(std::filesystem::exists(feature_path), feature_path.string() + " missing; pass features=PATH");
  Arrays full;
  for (auto& [k, v] : nn::load_npz(feature_path.string())) full.emplace(k, v);
  if (full.size() != 2) check::Require(full.size() == 32, "original VGG19 convolution inventory");
  Arrays features;
  for (auto name : {"block1_conv1_W", "block1_conv1_b"}) features.emplace(name, nn::cast(full.at(name), nn::DType::Float32));
  gan::RequireFinite(features, "finite pretrained VGG filter");
  gan::SaveArrays((root / "frozen_features.npz").string(), features);
  RaAlgebra(root);
  Train(root, pools, features, epochs, false, init);
  Train(root, pools, features, epochs, true, init);
  std::cout << "Artifacts: " << root.string() << "\nAll srgan_esrgan checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("cifar10"), features = paths::Dataset("style_transfer") / "vgg19.npz",
                          out = paths::OutputRoot() / "models" / "srgan_esrgan";
    if (!std::filesystem::exists(features))
      features = paths::DataRoot().parent_path() / "models" / "srgan_esrgan" / "frozen_features.npz";
    std::string init;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0)
        epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("features=", 0) == 0)
        features = a.substr(9);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("init=", 0) == 0)
        init = a.substr(5);
      else
        throw std::invalid_argument("expected epochs=N data=PATH features=PATH out=PATH init=DIR");
    }
    check::Require(epochs >= 3 && epochs <= 100, "three TOTAL epochs minimum (one warmup, two gan)");
    Run(data, features, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
