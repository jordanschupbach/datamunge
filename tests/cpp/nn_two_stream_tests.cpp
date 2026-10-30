
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_two_stream_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.10_3d_and_video_convolutions/two_stream.cpp"
#undef main
TEST(TwoStream, AllFlowCenterClipStackValuesAndAdjoints) {
  FlowStackChecks();
}
TEST(TwoStream, EveryProbabilityFusionHostValueAndGradient) {
  FusionChecks();
}
TEST(TwoStream, ExplicitPoolBorderValuesAndDerivatives) {
  PoolChecks();
}
TEST(TwoStream, BothCompleteCompactBranchesAndAnalyticalFullCensus) {
  GraphChecks();
}
TEST(TwoStream, EveryVariationalFlowLeafDerivativeIndependentHostFiniteDifference) {
  namespace n = dm::nn;
  namespace m = n::models;
  std::vector<double> a(12), b(12);
  for (int i = 0; i < 12; ++i) {
    a[i] = .1 * std::sin(i);
    b[i] = .1 * std::cos(i);
  }
  auto u = n::from_vector(a, {1, 3, 4, 1}).requires_grad_(),
       v = n::from_vector(b, {1, 3, 4, 1}).requires_grad_();
  const auto ix = n::full({1, 3, 4, 1}, .2, n::DType::Float64),
             iy = n::full({1, 3, 4, 1}, -.3, n::DType::Float64),
             it = n::full({1, 3, 4, 1}, .07, n::DType::Float64);
  const auto gs = n::grad(m::TwoStreamFlowEnergy(u, v, {ix, iy, it}, .05), {u, v});
  auto oracle = [](const std::vector<double> &u, const std::vector<double> &v) {
    double e = 0;
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 4; ++x) {
        const int i = y * 4 + x;
        e += .5 * std::pow(.2 * u[i] - .3 * v[i] + .07, 2);
        for (const auto *f : {&u, &v}) {
          if (x < 3)
            e += .5 * .0025 * std::pow((*f)[i + 1] - (*f)[i], 2);
          if (y < 2)
            e += .5 * .0025 * std::pow((*f)[i + 4] - (*f)[i], 2);
        }
      }
    return e;
  };
  EXPECT_NEAR(m::TwoStreamFlowEnergy(u, v, {ix, iy, it}).item(), oracle(a, b), 1e-14);
  for (int f = 0; f < 2; ++f)
    for (int i = 0; i < 12; ++i) {
      auto ap = a, am = a, bp = b, bm = b;
      if (f) {
        bp[i] += 1e-6;
        bm[i] -= 1e-6;
      } else {
        ap[i] += 1e-6;
        am[i] -= 1e-6;
      }
      EXPECT_NEAR(gs[f].to_vector<double>()[i], (oracle(ap, bp) - oracle(am, bm)) / 2e-6,
                  1e-10);
    }
}
TEST(TwoStream, EveryClassicalDerivativeValueIncludingSymmetricBorders) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto a = n::arange(0, 12).reshape({1, 3, 4, 1}) * .01, b = a + .02;
  const auto d = m::TwoStreamFlowDerivatives(a, b);
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 4; ++x) {
      EXPECT_NEAR(d[0].at({0, y, x, 0}), x < 3 ? .01 : 0, 1e-7);
      EXPECT_NEAR(d[1].at({0, y, x, 0}), y < 2 ? .04 : 0, 1e-7);
      EXPECT_NEAR(d[2].at({0, y, x, 0}), .02, 1e-7);
    }
}
TEST(TwoStream, InvalidFlowGeometryAndTrainingDropout) {
  namespace n = dm::nn;
  namespace m = n::models;
  EXPECT_THROW(m::TwoStreamFlowStack(n::zeros({1, 9, 96, 96, 2}), 96), std::invalid_argument);
  n::ParameterStore p;
  EXPECT_THROW(m::TwoStream(p, n::zeros({1, 96, 96, 3}), n::zeros({1, 10, 96, 96, 2}),
                            {96, 16, 64, 32, 3}, true),
               std::invalid_argument);
}

#include "../fixtures/nn_two_stream_reference.hpp"
#include <datamunge/image/read_image.hpp>
TEST(TwoStream, EveryOriginalResizedFramePixelAndMiddleFrameIdentity) {
  namespace n = dm::nn;
  const auto root = std::filesystem::path("build/cnn-datasets/two_stream");
  std::map<std::string, n::Tensor> audit;
  for (const auto &[key, value] : n::load_npz((root / "flow_audit.npz").string()))
    audit.emplace(key, value);
  std::ifstream rows(root / "clip_rows.tsv");
  std::string split, path;
  int label, person, video, segment, total = 0;
  double maximum=0;
  while (rows >> split >> label >> person >> video >> segment) {
    const auto clip = audit.at(std::to_string(video) + "/" + std::to_string(segment) + "/rgb")
                          .to_vector<float>();
    for (int frame = 0; frame < 11; ++frame) {
      rows >> path;
      const auto image = n::from_image(dm::image::read_image((root / path).string()));
      const auto raw = image.to_vector<float>();
      int height = image.size(0), width = image.size(1);
      for (int y = 0; y < 96; ++y)
        for (int x = 0; x < 96; ++x)
          for (int c = 0; c < 3; ++c) {
            double sy=float(y)*(float(height)/96.f),sx=float(x)*(float(width)/96.f);
            int y0 = int(sy), x0 = int(sx), y1 = std::min(y0 + 1, height - 1),
                x1 = std::min(x0 + 1, width - 1);
            double dy = sy - y0, dx = sx - x0;
            auto at = [&](int yy, int xx) {
              return double(raw[(yy * width + xx) * 3 + c]);
            };
            float expected =
                float(at(y0, x0) * (1 - dx) * (1 - dy) + at(y0, x1) * dx * (1 - dy) +
                      at(y1, x0) * (1 - dx) * dy + at(y1, x1) * dx * dy) /
                255.f;
            maximum=std::max(maximum,std::abs(double(clip[((frame*96+y)*96+x)*3+c])-expected));
            ++total;
          }
    }
  }
  EXPECT_EQ(total,10948608);EXPECT_LE(maximum,2e-7);std::cout<<"All10948608 original resized pixels,double-weight oracle with sourceFloat32coordinates max "<<maximum<<std::endl;
}
TEST(TwoStream, EveryOriginalHostModalityDescriptorAndMemoryNeighbor) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto pools = m::LoadTwoStream("build/cnn-datasets/two_stream");
  for (bool motion : {false, true}) {
    for (const auto &d : pools)
      EXPECT_EQ(m::TwoStreamGridFeatures(d, motion).to_vector<float>(),
                two_stream_reference::GridFeatures(d, motion).to_vector<float>());
    EXPECT_EQ(m::TwoStreamNearest(pools[0], pools[2], motion).to_vector<float>(),
              two_stream_reference::Nearest(pools[0], pools[2], motion).to_vector<float>());
  }
}
