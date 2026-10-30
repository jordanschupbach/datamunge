// CMD: just run-nn point_geometry
// Supporting PointNet++ CPU FPS/radius/grouping checks; classifier demonstrations are separate.
// Two batches, ties, strict-radius boundary, padding, duplicate gradient accumulation,
// frozen-discrete XYZ derivatives, translation and explicit invalid-input errors.
//
// Port of tflow's point_geometry. TF -> datamunge: tflow's custom CPU kernels
// become canonical tensor distance/ranking primitives with detached integer
// choices and differentiable gathers. Invalid contracts throw exceptions.
// The complete hand-checkable source fixture needs no external dataset.

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/point_geometry.hpp"

namespace nn = datamunge::nn;
namespace pg = point_geometry;
using nn::Tensor;

int main() {
  try {
    const float points[2][6][3] = {{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}, {.25f, .25f, 0}},
                                   {{10, 0, 0}, {11, 0, 0}, {13, 0, 0}, {12, 0, 0}, {10.25f, 0, 0}, {9, 0, 0}}};
    std::vector<float> xv(36), fv(24), coeff(64), rc(96);
    for (int b = 0; b < 2; ++b)
      for (int j = 0; j < 6; ++j) {
        for (int c = 0; c < 3; ++c) xv[static_cast<std::size_t>((b * 6 + j) * 3 + c)] = points[b][j][c];
        for (int c = 0; c < 2; ++c) fv[static_cast<std::size_t>((b * 6 + j) * 2 + c)] = static_cast<float>(100 * b + 10 * j + c);
      }
    for (int i = 0; i < 64; ++i) coeff[static_cast<std::size_t>(i)] = static_cast<float>(.1 * std::cos(.31 * i));
    for (int i = 0; i < 96; ++i) rc[static_cast<std::size_t>(i)] = static_cast<float>(.1 * std::sin(.29 * i));

    // The tflow graph: FPS(4) centers, radius-1 ball of 4, gathered features and centroid-relative XYZ.
    struct Run {
      Tensor fps, indices, counts, gathered, feature_gradient, relative, coordinate_gradient;
    };
    const auto run = [&](const std::vector<float>& x_values) {
      Tensor xyz = nn::from_vector(x_values, {2, 6, 3}), features = nn::from_vector(fv, {2, 6, 2});
      xyz.requires_grad_(true);
      features.requires_grad_(true);
      Run          r;
      r.fps                  = pg::PointFps(xyz.detach(), 4);
      const Tensor centers   = pg::PointGather(xyz, r.fps);
      const auto   ball      = pg::PointBall(xyz.detach(), centers.detach(), 1.f, 4);
      r.indices              = ball.indices;
      r.counts               = ball.counts;
      r.gathered             = pg::PointGather(features, ball.indices);
      r.relative             = pg::PointGather(xyz, ball.indices) - nn::expand_dims(centers, 2);
      const Tensor loss      = nn::sum(r.gathered * nn::from_vector(coeff, {2, 4, 4, 2}));
      const Tensor coordinate_loss = nn::sum(r.relative * nn::from_vector(rc, {2, 4, 4, 3}));
      r.feature_gradient     = nn::grad(loss, {features})[0];
      r.coordinate_gradient  = nn::grad(coordinate_loss, {xyz})[0];
      return r;
    };
    const Run out = run(xv);
    const int expected_fps[2][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}},
              expected_ball[2][4][4] = {{{0, 5, 0, 0}, {1, 5, 1, 1}, {2, 5, 2, 2}, {3, 3, 3, 3}},
                                        {{0, 4, 0, 0}, {2, 2, 2, 2}, {1, 4, 1, 1}, {3, 3, 3, 3}}},
              expected_counts[2][4] = {{2, 2, 2, 1}, {2, 1, 2, 1}};
    const auto fps = out.fps.to_vector<int>(), ball = out.indices.to_vector<int>(), counts = out.counts.to_vector<int>();
    const auto gathered = out.gathered.to_vector<float>(), relative = out.relative.to_vector<float>();
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < 4; ++i) {
        check::Require(fps[static_cast<std::size_t>(b * 4 + i)] == expected_fps[b][i],
                       "first-index FPS start and lowest-index distance ties");
        check::Require(counts[static_cast<std::size_t>(b * 4 + i)] == expected_counts[b][i],
                       "strict radius and capped distinct-neighbor count");
        for (int k = 0; k < 4; ++k) {
          const int index = expected_ball[b][i][k];
          check::Require(ball[static_cast<std::size_t>((b * 4 + i) * 4 + k)] == index,
                         "first-hit padding,input-order radius neighbors,batch isolation");
          for (int c = 0; c < 2; ++c)
            check::Near(gathered[static_cast<std::size_t>(((b * 4 + i) * 4 + k) * 2 + c)], 100 * b + 10 * index + c, 0,
                        "all64 gathered feature values");
          for (int c = 0; c < 3; ++c)
            check::Near(relative[static_cast<std::size_t>(((b * 4 + i) * 4 + k) * 3 + c)],
                        points[b][index][c] - points[b][expected_fps[b][i]][c], 0, "all96 centroid-relative XYZ values");
        }
      }
    const auto host = [&](const std::vector<double>& values, bool coordinates) {
      double    objective = 0;
      const int channels  = coordinates ? 3 : 2;
      for (int b = 0; b < 2; ++b)
        for (int i = 0; i < 4; ++i)
          for (int k = 0; k < 4; ++k)
            for (int c = 0; c < channels; ++c) {
              double v = values[static_cast<std::size_t>((b * 6 + expected_ball[b][i][k]) * channels + c)];
              if (coordinates) v -= values[static_cast<std::size_t>((b * 6 + expected_fps[b][i]) * channels + c)];
              const std::size_t at = static_cast<std::size_t>(((b * 4 + i) * 4 + k) * channels + c);
              objective += v * (coordinates ? rc[at] : coeff[at]);
            }
      return objective;
    };
    double worst = 0;
    int    covered = 0;
    for (bool coordinates : {false, true}) {
      const auto&         original = coordinates ? xv : fv;
      const auto          actual   = (coordinates ? out.coordinate_gradient : out.feature_gradient).to_vector<float>();
      std::vector<double> values(original.begin(), original.end());
      for (std::size_t i = 0; i < values.size(); ++i) {
        auto plus = values, minus = values;
        plus[i] += 1e-5;
        minus[i] -= 1e-5;
        const double fd = (host(plus, coordinates) - host(minus, coordinates)) / 2e-5;
        worst           = std::max(worst, std::abs(fd - actual[i]));
        ++covered;
      }
    }
    check::Require(covered == 60, "all24features+36XYZ gradient coordinates");
    check::Near(worst, 0, 3e-7, "ALL60 hostFD through grouped gather/relative centroids with frozen discrete indices");

    std::vector<float> translated(xv.size());
    for (std::size_t i = 0; i < xv.size(); ++i) translated[i] = xv[i] + 2;
    const Run  moved = run(translated);
    const auto moved_ball = moved.indices.to_vector<int>();
    const auto moved_relative = moved.relative.to_vector<float>();
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k) {
          check::Require(moved_ball[static_cast<std::size_t>((b * 4 + i) * 4 + k)] == expected_ball[b][i][k],
                         "translation preserves radius memberships");
          for (int c = 0; c < 3; ++c)
            check::Near(moved_relative[static_cast<std::size_t>(((b * 4 + i) * 4 + k) * 3 + c)],
                        points[b][expected_ball[b][i][k]][c] - points[b][expected_fps[b][i]][c], 0,
                        "translation cancels in local coordinates");
        }
    const auto duplicate = pg::PointFps(nn::zeros({2, 6, 3}), 4).to_vector<int>();
    for (int v : duplicate)
      check::Require(v == 0, "collapsed clouds repeat reference seed rather than inventing coordinates");
    const std::string empty = check::Throws(
        [&] { pg::PointBall(nn::from_vector(xv, {2, 6, 3}), nn::full({2, 1, 3}, 100.0), .1f, 4); },
        "empty neighborhood gives explicit error,never undefined indices");
    check::Require(empty.find("empty radius") != std::string::npos,
                   "empty neighborhood gives explicit error,never undefined indices");
    const std::string few = check::Throws([&] { pg::PointFps(nn::zeros({2, 3, 3}), 4); }, "FPS sample count validated at runtime");
    check::Require(few.find("N>=count") != std::string::npos, "FPS sample count validated at runtime");
    std::cout << "PointNet++ nativeCPU geometry:two-batchFPS/ties/strict-radius/padding/all64features/"
                 "all96relativecoordinates passed;ALL60 scatter/centroid hostFD error="
              << worst
              << ";translation/duplicates/empty and undersized inputs checked. Supporting primitives only;classifier "
                 "training and whole-model audits are separate.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
