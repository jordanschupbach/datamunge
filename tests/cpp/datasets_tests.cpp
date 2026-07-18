#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>

TEST(Datasets, IrisShapeAndBalance) {
    const auto df = datamunge::datasets::iris();
    EXPECT_EQ(df.nrows(), 150u);
    EXPECT_EQ(df.ncols(), 5u);

    const auto counts = df.value_counts("Species");
    ASSERT_EQ(counts.size(), 3u);
    EXPECT_EQ(counts.at("setosa"), 50u);
    EXPECT_EQ(counts.at("versicolor"), 50u);
    EXPECT_EQ(counts.at("virginica"), 50u);

    EXPECT_DOUBLE_EQ(df.double_at("Sepal.Length", 0), 5.1);
    EXPECT_DOUBLE_EQ(df.double_at("Petal.Width", 149), 1.8);
    EXPECT_EQ(df.string_at("Species", 149), "virginica");
}

TEST(Datasets, PenguinsShapeNullsAndBalance) {
    const auto df = datamunge::datasets::penguins();
    EXPECT_EQ(df.nrows(), 344u);
    EXPECT_EQ(df.ncols(), 8u);

    EXPECT_EQ(df.null_count("bill_length_mm"), 2u);
    EXPECT_EQ(df.null_count("bill_depth_mm"), 2u);
    EXPECT_EQ(df.null_count("flipper_length_mm"), 2u);
    EXPECT_EQ(df.null_count("body_mass_g"), 2u);
    EXPECT_EQ(df.null_count("sex"), 11u);
    EXPECT_EQ(df.null_count("species"), 0u);
    EXPECT_EQ(df.null_count("island"), 0u);

    const auto counts = df.value_counts("species");
    ASSERT_EQ(counts.size(), 3u);
    EXPECT_EQ(counts.at("Adelie"), 152u);
    EXPECT_EQ(counts.at("Chinstrap"), 68u);
    EXPECT_EQ(counts.at("Gentoo"), 124u);

    EXPECT_TRUE(df.is_null("bill_length_mm", 3));
    EXPECT_EQ(df.string_at("species", 0), "Adelie");
    EXPECT_EQ(df.string_at("island", 0), "Torgersen");
    EXPECT_DOUBLE_EQ(df.double_at("bill_length_mm", 0), 39.1);
}
