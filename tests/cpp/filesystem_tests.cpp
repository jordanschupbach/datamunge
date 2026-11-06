/// @file filesystem_tests.cpp
/// @brief Public mkdir helper: recursive creation, repeat calls and filesystem errors.
#include <gtest/gtest.h>

#include <chrono>
#include <dm/dm.hpp>
#include <fstream>

namespace {
/// @brief Own a unique temporary directory and remove only that owned tree.
class MkdirTest : public testing::Test {
 protected:
  std::filesystem::path root; ///< Directory assigned only after successful creation.

  void SetUp() override {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    for (int attempt = 0; attempt < 100; ++attempt) {
      auto candidate = std::filesystem::temp_directory_path() /
                       ("datamunge-mkdir-" + std::to_string(stamp) + "-" + std::to_string(attempt));
      if (std::filesystem::create_directory(candidate)) {
        root = std::move(candidate);
        break;
      }
    }
    ASSERT_FALSE(root.empty());
  }

  void TearDown() override {
    if (!root.empty()) {
      std::error_code ignored;
      std::filesystem::remove_all(root, ignored);
    }
  }
};
} // namespace

TEST_F(MkdirTest, CreatesMissingParentsAndAcceptsStringPaths) {
  const auto target = root / "parent folder" / "child";
  EXPECT_TRUE(dm::mkdir(target.string()));
  EXPECT_TRUE(std::filesystem::is_directory(target));
  EXPECT_TRUE(std::filesystem::is_directory(target.parent_path()));
}

TEST_F(MkdirTest, ExistingDirectoryIsANoopAndPreservesContents) {
  const auto target = root / "existing";
  ASSERT_TRUE(dm::mkdir(target));
  std::ofstream(target / "keep.txt") << "keep";
  EXPECT_FALSE(dm::mkdir(target));
  EXPECT_TRUE(std::filesystem::is_regular_file(target / "keep.txt"));
}

TEST_F(MkdirTest, ConflictingFilesRaiseAndRemainUnchanged) {
  const auto file = root / "file";
  std::ofstream(file) << "keep";
  EXPECT_THROW(dm::mkdir(file), std::filesystem::filesystem_error);
  EXPECT_THROW(dm::mkdir(file / "child"), std::filesystem::filesystem_error);
  EXPECT_EQ(std::filesystem::file_size(file), 4u);
}

TEST_F(MkdirTest, EmptyPathRaisesWithoutChangingWorkingDirectory) {
  const auto before = std::filesystem::current_path();
  EXPECT_THROW(dm::mkdir(""), std::filesystem::filesystem_error);
  EXPECT_EQ(std::filesystem::current_path(), before);
}
