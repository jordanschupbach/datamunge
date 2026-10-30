
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_medical_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.10_3d_and_video_convolutions/medical_volume_classification.cpp"
#undef main
#include "../fixtures/nn_medical_reference.hpp"
TEST(MedicalVolume, Every646OddEvenResidualKernelProjectionBiasInputHostDerivative) {
  NumericalChecks();
}
TEST(MedicalVolume, ActualAllStagesAnd20LearnedGradients) {
  nn::set_num_threads(2);
  GraphChecks();
  namespace m = dm::nn::models;
  const auto data = m::LoadMedical("build/cnn-datasets/medical_volume");
  nn::ParameterStore p(1049), state(1051);
  const auto fit = data[0].voxels.to(nn::DType::Float64);
  state.set("normalization/mean", nn::mean(fit).to(nn::DType::Float32));
  state.set("normalization/stddev", nn::sqrt(nn::variance(fit)).to(nn::DType::Float32));
  const auto out = m::MedicalVolume(p, state, nn::slice_axis(data[0].voxels, 0, 0, 4));
  for (const auto &g : nn::grad(nn::sparse_softmax_cross_entropy(
                                    out.logits, nn::slice_axis(data[0].labels, 0, 0, 4)),
                                p.parameters())) {
    EXPECT_TRUE(nn::all(nn::is_finite(g)).item<bool>());
    EXPECT_GT(nn::sum(nn::square(g)).item(), 0);
  }
}
TEST(MedicalVolume, Every3161088OriginalVoxelAndReleasedLabelSourceIdentity) {
  namespace m = dm::nn::models;
  const auto root = std::filesystem::path("build/cnn-datasets/medical_volume");
  const auto pools = m::LoadMedical(root);
  std::map<std::string, nn::Tensor> original;
  for (const auto &[key, value] : nn::load_npz((root / "nodulemnist3d.npz").string()))
    original[key] = value;
  std::ifstream rows(root / "selected_rows.tsv");
  std::string split;
  int row, id, label;
  std::array<int, 3> offset{};
  int total = 0;
  while (rows >> split >> row >> id >> label) {
    const int pool = split == "train" ? 0 : split == "val" ? 1 : 2;
    const auto &d = pools[pool];
    const int selected = offset[pool]++;
    EXPECT_EQ(d.labels.to_vector<int>()[selected], label);
    EXPECT_EQ(d.source.to_vector<int>()[selected], id);
    EXPECT_EQ(original.at(split + "_labels").reshape({-1}).to_vector<int>()[row], label);
    const auto raw = nn::select(original.at(split + "_images"), 0, row).to_vector<int>();
    const auto actual = nn::select(d.voxels, 0, selected).to_vector<float>();
    for (int pixel = 0; pixel < 21952; ++pixel) {
      EXPECT_EQ(actual[pixel], float(raw[pixel]) / 255.f);
      ++total;
    }
  }
  EXPECT_EQ(total, 3161088);
  EXPECT_EQ(offset, (std::array<int, 3>{64, 16, 64}));
}
TEST(MedicalVolume, IndependentEveryFittingMomentBlockFeatureAndNearestScore) {
  namespace m = dm::nn::models;
  const auto pools = m::LoadMedical("build/cnn-datasets/medical_volume");
  const auto raw = pools[0].voxels.to_vector<float>();
  double mean = 0;
  for (float x : raw)
    mean += x;
  mean /= raw.size();
  double var = 0;
  for (float x : raw)
    var += (x - mean) * (x - mean);
  var /= raw.size();
  EXPECT_NEAR(nn::mean(pools[0].voxels.to(nn::DType::Float64)).item(), mean, 1e-12);
  EXPECT_NEAR(nn::variance(pools[0].voxels.to(nn::DType::Float64)).item(), var, 1e-12);
  const auto features = m::MedicalFeatures(pools[0].voxels).to_vector<double>();
  for (int row = 0; row < 64; ++row) {
    std::array<double, 64> expected{};
    for (int z = 0; z < 28; ++z)
      for (int y = 0; y < 28; ++y)
        for (int x = 0; x < 28; ++x)
          expected[(z / 7 * 4 + y / 7) * 4 + x / 7] +=
              raw[((row * 28 + z) * 28 + y) * 28 + x] / 343.f;
    for (int feature = 0; feature < 64; ++feature)
      EXPECT_EQ(features[row * 64 + feature], expected[feature]);
  }
  EXPECT_EQ(m::MedicalBaseline(pools[0], pools[2]).to_vector<float>(),
            medical_reference::MedicalBaseline(pools[0], pools[2]).to_vector<float>());
}
TEST(MedicalVolume, IndependentIntensityOnlyRestoreAndNormalizationContracts) {
  namespace m = dm::nn::models;
  const auto d = m::LoadMedical("build/cnn-datasets/medical_volume");
  nn::ParameterStore p(1049), s(1051), fresh(99), state(100);
  const auto x = nn::slice_axis(d[2].voxels, 0, 0, 2);
  m::MedicalVolume(p, s, x);
  const auto fit = d[0].voxels.to(nn::DType::Float64);
  s.set("normalization/mean", nn::mean(fit).to(nn::DType::Float32));
  s.set("normalization/stddev", nn::sqrt(nn::variance(fit)).to(nn::DType::Float32));
  const auto z = m::MedicalVolume(p, s, x).logits;
  m::MedicalVolume(fresh, state, x);
  for (const auto &name : p.names())
    fresh.set(name, p.at(name).clone());
  for (const auto &name : s.names())
    state.set(name, s.at(name).clone());
  const auto before = fresh.flatten(), moments = state.flatten();
  EXPECT_EQ(nn::max(nn::abs(z - m::MedicalVolume(fresh, state, x).logits)).item(), 0);
  EXPECT_EQ(before, fresh.flatten());
  EXPECT_EQ(moments, state.flatten());
  EXPECT_THROW(m::MedicalVolume(p, s, x, 4, true, nullptr), std::invalid_argument);
  s.set("normalization/stddev", nn::zeros({}));
  EXPECT_THROW(m::MedicalVolume(p, s, x), std::invalid_argument);
}
