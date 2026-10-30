// dm::dir(): Python-style listings of a library object's public members.
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace {
bool contains(const std::vector<std::string>& names, const std::string& name) {
    return std::find(names.begin(), names.end(), name) != names.end();
}
}  // namespace

TEST(Dir, ListsAModelsPublicMethods) {
    const auto model = dm::stats::LM(dm::formula("len ~ dose"), dm::datasets::tooth_growth());
    const auto listing = dm::dir(model);
    ASSERT_TRUE(listing.found());
    EXPECT_EQ(listing.type_name(), "stats::LM");
    const auto names = listing.names();
    EXPECT_TRUE(contains(names, "coefficients"));
    EXPECT_TRUE(contains(names, "summary"));
    EXPECT_FALSE(contains(names, "formula_"));  // private members are not listed
    EXPECT_TRUE(std::is_sorted(names.begin(), names.end()));
    // The formula-first constructor is listed with its signature.
    const auto ctors = listing.of_kind(dm::DirMember::Kind::Constructor);
    EXPECT_TRUE(std::any_of(ctors.begin(), ctors.end(), [](const dm::DirMember& m) {
        return m.signature.find("const Formula& formula") != std::string::npos;
    }));
}

TEST(Dir, ListsFieldsAndWorksWithoutAnObject) {
    const auto listing = dm::dir<dm::geometry::Point2D>();
    ASSERT_TRUE(listing.found());
    const auto fields = listing.of_kind(dm::DirMember::Kind::Field);
    ASSERT_EQ(fields.size(), 2U);
    EXPECT_EQ(fields[0].name, "x");
    EXPECT_EQ(fields[1].name, "y");
}

TEST(Dir, ResolvesTemplatesAliasesAndInheritedMembers) {
    dm::linalg::DenseMatrix<double> matrix(2, 2);
    EXPECT_EQ(dm::dir(matrix).type_name(), "linalg::DenseMatrix<double>");
    EXPECT_TRUE(dm::dir(matrix).found());
    dm::Plot plot;  // an alias of graphics::RPlot
    EXPECT_EQ(dm::dir(plot).type_name(), "graphics::RPlot");
    // dm::Histogram derives from graphics::RPlot: inherited methods are listed too.
    EXPECT_TRUE(contains(dm::dir<dm::Histogram>().names(), "set_title"));
}

TEST(Dir, ReportsTypesOutsideTheLibrary) {
    const auto listing = dm::dir(std::vector<int>{});
    EXPECT_FALSE(listing.found());
    EXPECT_TRUE(listing.names().empty());
    EXPECT_NE(listing.to_string().find("no member listing"), std::string::npos);
}

TEST(Dir, TextGroupsMembersWithDocs) {
    const std::string text = dm::dir<dm::DirListing>().to_string();
    EXPECT_EQ(text.rfind("DirListing", 0), 0U);
    EXPECT_NE(text.find("methods ("), std::string::npos);
    EXPECT_NE(text.find("names() const -> std::vector<std::string>"), std::string::npos);
    EXPECT_NE(text.find("// The member names, sorted and de-duplicated"), std::string::npos);
}
