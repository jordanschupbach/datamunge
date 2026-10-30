#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <format>
#include <memory>
#include <print>
#include <string>
#include <vector>

TEST(DataFrameFormat, UsesTheStandardTextPreviewAndStringFormatting) {
    auto frame = dm::datasets::iris().head(2);
    const auto preview = frame.to_string();
    EXPECT_EQ(std::format("{}", frame), preview);
    EXPECT_EQ(std::format("{:>{}}", frame, preview.size() + 3), "   " + preview);
    EXPECT_EQ(std::format("[{}]", dm::dstruct::DataFrame{}), "[DataFrame[0 x 0]]");
}

TEST(DataFrameFormat, PrintsToAFileWithTheStandardPrintOverload) {
    dm::dstruct::DataFrame frame;
    frame.add_column("value", std::vector<double>{1, 2});
    std::unique_ptr<std::FILE, decltype(&std::fclose)> file(std::tmpfile(), &std::fclose);
    ASSERT_NE(file, nullptr);
    std::print(file.get(), "{}\n", frame);
    std::rewind(file.get());
    char buffer[1024];
    std::string printed;
    while (const auto count = std::fread(buffer, 1, sizeof(buffer), file.get()))
        printed.append(buffer, count);
    EXPECT_EQ(printed, frame.to_string() + '\n');
}

TEST(LibraryFormat, VectorsOfLibraryTypesPrintElementsUnquoted) {
    // The shared formatter base must not expose set_debug_format(), or the
    // standard range formatter would quote every element.
    const std::vector<dm::dstruct::DataFrame> frames(2);
    EXPECT_EQ(std::format("{}", frames), "[DataFrame[0 x 0], DataFrame[0 x 0]]");
}

TEST(LibraryFormat, DirListingsPrintTheirText) {
    const auto listing = dm::dir<dm::geometry::Point2D>();
    EXPECT_EQ(std::format("{}", listing), listing.to_string());
}
