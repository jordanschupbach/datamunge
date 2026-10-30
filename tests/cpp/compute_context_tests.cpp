#include <gtest/gtest.h>

#include <atomic>
#include <datamunge/compute/context.hpp>
#include <limits>
#include <thread>

namespace c = datamunge::compute;
TEST(ComputeContext, ExplicitCapabilitiesAndNoGpuFallback) {
  EXPECT_TRUE(c::probe().available);
  EXPECT_FALSE(c::probe({c::Backend::CPU, 1}).available);
  EXPECT_THROW((c::Context({c::Backend::CPU, 1})), std::invalid_argument);
  EXPECT_THROW((c::Context({}, 0)), std::invalid_argument);
  EXPECT_THROW(c::backend_name(static_cast<c::Backend>(99)),
               std::invalid_argument);
  for (auto backend : {c::Backend::CUDA, c::Backend::Vulkan}) {
    const auto info = c::probe({backend, 0});
    if (!info.available) {
      EXPECT_FALSE(info.diagnostic.empty());
      EXPECT_THROW((c::Context({backend, 0})), std::runtime_error);
    }
  }
}
TEST(ComputeContext, ViewsRetainOwnersAndCountAllocationsOnce) {
  c::Buffer retained;
  {
    c::Context context;
    {
      auto bytes = context.allocate(64);
      retained = bytes.slice(8, 16);
      EXPECT_EQ(context.live_bytes(), 64);
    }
    EXPECT_EQ(context.live_bytes(), 64);
    retained.host_data()[0] = 42;
    EXPECT_THROW(retained.slice(17, 0), std::out_of_range);
    EXPECT_THROW(retained.slice(1, std::numeric_limits<std::size_t>::max()),
                 std::out_of_range);
  }
  EXPECT_EQ(retained.host_data()[0], 42);
  c::Buffer empty;
  EXPECT_THROW(empty.device(), std::logic_error);
  c::Context context;
  auto zero = context.allocate(0);
  EXPECT_TRUE(zero.defined());
  EXPECT_NE(zero.host_data(), nullptr);
  EXPECT_EQ(context.live_bytes(), 0);
}
TEST(ComputeContext, BoundedCopyOverlapAndForeignOwnership) {
  c::Context context, other;
  auto bytes = context.allocate(6);
  for (unsigned i = 0; i < 6; ++i) bytes.host_data()[i] = i;
  auto src = bytes.slice(0, 4), dst = bytes.slice(2, 4);
  context.copy(dst, src, 4);
  EXPECT_EQ(std::vector<std::uint8_t>(bytes.host_data(), bytes.host_data() + 6),
            (std::vector<std::uint8_t>{0, 1, 0, 1, 2, 3}));
  EXPECT_THROW(context.copy(dst, src, 5), std::out_of_range);
  EXPECT_THROW(other.fill(bytes), std::invalid_argument);
  context.fill(dst, 9);
  EXPECT_EQ(bytes.host_data()[2], 9);
  EXPECT_EQ(bytes.host_data()[1], 1);
}
TEST(ComputeContext, WorkersVisitEachIndexOnceAndRecoverAfterException) {
  const c::Context context({}, 4);
  std::vector<unsigned> values(10001, 0);
  for (int pass = 0; pass < 4; ++pass)
    context.parallel_for(values.size(), [&](std::size_t i) { ++values[i]; });
  for (auto v : values) EXPECT_EQ(v, 4);
  EXPECT_THROW(
      context.parallel_for(1000,
                           [](std::size_t i) {
                             if (i == 17)
                               throw std::runtime_error("worker failure");
                           }),
      std::runtime_error);
  context.parallel_for(values.size(),
                       [&](std::size_t i) { values[i] = int(i); });
  EXPECT_EQ(values.back(), 10000);
  context.parallel_for(0, [](std::size_t) { FAIL(); });
  EXPECT_THROW(context.parallel_for(1, {}), std::invalid_argument);
  EXPECT_THROW(
      context.parallel_for(1, [&](std::size_t) { context.synchronize(); }),
      std::logic_error);
  context.synchronize();
}
TEST(ComputeContext, CopiedContextsSerializeIndependentSubmitters) {
  c::Context context({}, 3);
  auto alias = context;
  std::atomic<int> calls{0};
  std::thread first(
      [&] { context.parallel_for(2000, [&](std::size_t) { ++calls; }); });
  std::thread second(
      [&] { alias.parallel_for(2000, [&](std::size_t) { ++calls; }); });
  first.join();
  second.join();
  EXPECT_EQ(calls, 4000);
}

TEST(ComputeContext, IndirectContextReentryFailsInsteadOfDeadlocking) {
  c::Context first, second;
  EXPECT_THROW(first.parallel_for(1,
                                  [&](std::size_t) {
                                    second.parallel_for(1, [&](std::size_t) {
                                      first.synchronize();
                                    });
                                  }),
               std::logic_error);
  first.synchronize();
  second.synchronize();
}
