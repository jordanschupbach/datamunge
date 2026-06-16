#include <gtest/gtest.h>
#include <octra/octra_c.h>

#include <array>
#include <octra/octra.hpp>
#include <vector>

namespace {

class TimesTwoCallback final : public octra::Callback {
 public:
  double call(double x) override { return x * 2.0; }
};

double times_two_c_cb(double x, void* userdata) {
  const double factor = userdata ? *static_cast<const double*>(userdata) : 2.0;
  return x * factor;
}

} // namespace

TEST(octra_cpp, call_with_callback_null_is_identity) {
  EXPECT_DOUBLE_EQ(octra::call_with_callback(3.5, nullptr), 3.5);
}

TEST(octra_hpp, callback_default_impl_is_identity) {
  octra::Callback cb;
  EXPECT_DOUBLE_EQ(cb.call(3.5), 3.5);
}

TEST(octra_cpp, call_with_callback_invokes_callback) {
  TimesTwoCallback cb;
  EXPECT_DOUBLE_EQ(octra::call_with_callback(3.5, &cb), 7.0);
}

TEST(octra_cpp, map_dvector_with_callback_null_is_identity) {
  const std::vector<double> in{1.0, 2.0, 3.0};
  const auto                out = octra::map_dvector_with_callback(in, nullptr);
  EXPECT_EQ(out, in);
}

TEST(octra_cpp, map_dvector_with_callback_invokes_callback) {
  TimesTwoCallback          cb;
  const std::vector<double> in{1.0, 2.0, 3.0};
  const auto                out = octra::map_dvector_with_callback(in, &cb);
  ASSERT_EQ(out.size(), in.size());
  EXPECT_DOUBLE_EQ(out[0], 2.0);
  EXPECT_DOUBLE_EQ(out[1], 4.0);
  EXPECT_DOUBLE_EQ(out[2], 6.0);
}

TEST(octra_cpp, make_and_sum_dvector) {
  const auto v = octra::make_dvector(1.0, 2.0, 3.5);
  ASSERT_EQ(v.size(), 3u);
  EXPECT_DOUBLE_EQ(v[0], 1.0);
  EXPECT_DOUBLE_EQ(v[1], 2.0);
  EXPECT_DOUBLE_EQ(v[2], 3.5);
  EXPECT_DOUBLE_EQ(octra::sum_dvector(v), 6.5);
}

TEST(octra_cpp, make_and_sum_dpair) {
  const auto p = octra::make_dpair(1.25, 2.75);
  EXPECT_DOUBLE_EQ(p.first, 1.25);
  EXPECT_DOUBLE_EQ(p.second, 2.75);
  EXPECT_DOUBLE_EQ(octra::sum_dpair(p), 4.0);
}

TEST(octra_c, make_and_sum_dvector) {
  std::array<double, 3> out{};
  octra_make_dvector(1.0, 2.0, 3.5, out.data());
  EXPECT_DOUBLE_EQ(out[0], 1.0);
  EXPECT_DOUBLE_EQ(out[1], 2.0);
  EXPECT_DOUBLE_EQ(out[2], 3.5);

  EXPECT_DOUBLE_EQ(octra_sum_dvector(out.data(), out.size()), 6.5);
}

TEST(octra_c, hello_is_callable) {
  ASSERT_NO_THROW(octra_hello());
}

TEST(octra_c, sum_dvector_empty) {
  EXPECT_DOUBLE_EQ(octra_sum_dvector(nullptr, 0), 0.0);
}

TEST(octra_c, make_and_sum_dpair) {
  const octra_dpair p = octra_make_dpair(1.25, 2.75);
  EXPECT_DOUBLE_EQ(p.first, 1.25);
  EXPECT_DOUBLE_EQ(p.second, 2.75);
  EXPECT_DOUBLE_EQ(octra_sum_dpair(p), 4.0);
}

TEST(octra_c, call_double_cb_null_is_identity) {
  EXPECT_DOUBLE_EQ(octra_call_double_cb(3.5, nullptr, nullptr), 3.5);
}

TEST(octra_c, call_double_cb_invokes_callback) {
  const double factor = 2.0;
  EXPECT_DOUBLE_EQ(octra_call_double_cb(3.5, &times_two_c_cb, (void*)&factor), 7.0);
}

TEST(octra_c, map_dvector_cb_handles_null_buffers) {
  const double in[3]  = {1.0, 2.0, 3.0};
  double       out[3] = {0.0, 0.0, 0.0};

  octra_map_dvector_cb(nullptr, 3, out, &times_two_c_cb, nullptr);
  EXPECT_DOUBLE_EQ(out[0], 0.0);
  EXPECT_DOUBLE_EQ(out[1], 0.0);
  EXPECT_DOUBLE_EQ(out[2], 0.0);

  octra_map_dvector_cb(in, 3, nullptr, &times_two_c_cb, nullptr);
  EXPECT_DOUBLE_EQ(out[0], 0.0);
  EXPECT_DOUBLE_EQ(out[1], 0.0);
  EXPECT_DOUBLE_EQ(out[2], 0.0);
}

TEST(octra_c, map_dvector_cb_null_is_identity) {
  const double in[3] = {1.0, 2.0, 3.0};
  double       out[3]{};
  octra_map_dvector_cb(in, 3, out, nullptr, nullptr);
  EXPECT_DOUBLE_EQ(out[0], 1.0);
  EXPECT_DOUBLE_EQ(out[1], 2.0);
  EXPECT_DOUBLE_EQ(out[2], 3.0);
}

TEST(octra_c, map_dvector_cb_invokes_callback) {
  const double in[3] = {1.0, 2.0, 3.0};
  double       out[3]{};
  const double factor = 2.0;
  octra_map_dvector_cb(in, 3, out, &times_two_c_cb, (void*)&factor);
  EXPECT_DOUBLE_EQ(out[0], 2.0);
  EXPECT_DOUBLE_EQ(out[1], 4.0);
  EXPECT_DOUBLE_EQ(out[2], 6.0);
}
