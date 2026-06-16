#include <gtest/gtest.h>

#include <datamunge/dstruct/dstruct.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

using datamunge::dstruct::ThreadSafeVector;

TEST(ThreadSafeVector, SupportsBasicMutationAndSnapshot) {
  ThreadSafeVector<int> values{1, 2};
  values.push_back(3);
  values.emplace_back(4);
  values.set(1, 20);
  values.append(std::vector<int>{5, 6});

  const auto snapshot = values.snapshot();
  EXPECT_EQ(snapshot, (std::vector<int>{1, 20, 3, 4, 5, 6}));
  EXPECT_EQ(values.size(), 6u);
  EXPECT_FALSE(values.empty());
}

TEST(ThreadSafeVector, IsGenericAcrossElementTypes) {
  ThreadSafeVector<std::string> values{"alpha", "beta"};
  values.push_back("gamma");
  values.set(1, std::string("delta"));

  EXPECT_EQ(values.snapshot(), (std::vector<std::string>{"alpha", "delta", "gamma"}));
}

TEST(ThreadSafeVector, AtAndSetCheckBounds) {
  ThreadSafeVector<int> values{7, 8, 9};

  EXPECT_EQ(values.at(0), 7);
  EXPECT_THROW(static_cast<void>(values.at(4)), std::out_of_range);
  EXPECT_THROW(values.set(4, 10), std::out_of_range);
}

TEST(ThreadSafeVector, ApplyTransformsValuesInParallelAndPreservesOrder) {
  ThreadSafeVector<int> values;
  for (int i = 0; i < 32; ++i) {
    values.push_back(i);
  }

  const auto squared = datamunge::dstruct::apply(values, [](const int value) { return value * value; }, 4, 5);
  ASSERT_EQ(squared.size(), 32u);
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(squared[static_cast<std::size_t>(i)], i * i);
  }
}

TEST(ThreadSafeVector, ApplySupportsDifferentOutputTypes) {
  ThreadSafeVector<std::string> values{"a", "abcd", "xy"};

  const auto lengths =
      datamunge::dstruct::apply(values, [](const std::string& value) { return value.size(); }, 3, 1);

  EXPECT_EQ(lengths, (std::vector<std::size_t>{1u, 4u, 2u}));
}

TEST(ThreadSafeVector, PopBackReturnsValuesAndSignalsEmpty) {
  ThreadSafeVector<int> values{11, 12};

  const auto first = values.pop_back();
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first.value(), 12);

  const auto second = values.pop_back();
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second.value(), 11);

  EXPECT_FALSE(values.pop_back().has_value());
  EXPECT_TRUE(values.empty());
}

TEST(ThreadSafeVector, ConcurrentPushBackKeepsAllValues) {
  ThreadSafeVector<int> values;
  constexpr int thread_count = 8;
  constexpr int values_per_thread = 1000;

  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (int t = 0; t < thread_count; ++t) {
    threads.emplace_back([&values, t] {
      const int base = t * values_per_thread;
      for (int i = 0; i < values_per_thread; ++i) {
        values.push_back(base + i);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  auto snapshot = values.snapshot();
  ASSERT_EQ(snapshot.size(), static_cast<std::size_t>(thread_count * values_per_thread));
  std::sort(snapshot.begin(), snapshot.end());
  EXPECT_EQ(snapshot.front(), 0);
  EXPECT_EQ(snapshot.back(), thread_count * values_per_thread - 1);
  for (std::size_t i = 0; i < snapshot.size(); ++i) {
    EXPECT_EQ(snapshot[i], static_cast<int>(i));
  }
}

TEST(ThreadSafeVector, SnapshotIsSafeWhileWritersAreActive) {
  ThreadSafeVector<int> values;
  constexpr int thread_count = 4;
  constexpr int values_per_thread = 500;

  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (int t = 0; t < thread_count; ++t) {
    threads.emplace_back([&values, t] {
      for (int i = 0; i < values_per_thread; ++i) {
        values.push_back(t + i);
      }
    });
  }

  std::size_t observed_size = 0;
  while (observed_size < static_cast<std::size_t>(thread_count * values_per_thread)) {
    observed_size = values.snapshot().size();
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(values.size(), static_cast<std::size_t>(thread_count * values_per_thread));
}
