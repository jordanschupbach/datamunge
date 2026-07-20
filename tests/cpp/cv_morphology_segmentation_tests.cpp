#include <gtest/gtest.h>

#include <datamunge/cv/morphology.hpp>
#include <datamunge/cv/segmentation.hpp>
#include <datamunge/image/draw.hpp>
#include <datamunge/image/filters.hpp>

using namespace datamunge::cv;
using namespace datamunge::image;

TEST(Erode, RemovesAnIsolatedForegroundPixel) {
    Image dot(10, 10, ImageMode::RGB, Pixel{0, 0, 0, 255});
    dot.set_pixel(5, 5, Pixel{255, 255, 255, 255});
    const Image eroded = erode(dot);
    EXPECT_EQ(eroded.get_pixel(5, 5).r, 0);
}

TEST(Dilate, GrowsAnIsolatedForegroundPixelIntoItsNeighborhood) {
    Image dot(10, 10, ImageMode::RGB, Pixel{0, 0, 0, 255});
    dot.set_pixel(5, 5, Pixel{255, 255, 255, 255});
    const Image dilated = dilate(dot);
    EXPECT_EQ(dilated.get_pixel(5, 5).r, 255);
    EXPECT_EQ(dilated.get_pixel(4, 5).r, 255);
    EXPECT_EQ(dilated.get_pixel(6, 6).r, 255);
    EXPECT_EQ(dilated.get_pixel(2, 2).r, 0);
}

TEST(MorphologicalOpen, RemovesNoiseButPreservesLargeRegions) {
    Image block(20, 20, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_rectangle(block, 5, 5, 14, 14, Pixel{255, 255, 255, 255}, true);
    block.set_pixel(0, 0, Pixel{255, 255, 255, 255});
    const Image opened = morphological_open(block);
    EXPECT_EQ(opened.get_pixel(0, 0).r, 0);
    EXPECT_EQ(opened.get_pixel(9, 9).r, 255);
}

TEST(MorphologicalClose, FillsASmallHoleInASolidRegion) {
    Image block(20, 20, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_rectangle(block, 5, 5, 14, 14, Pixel{255, 255, 255, 255}, true);
    block.set_pixel(9, 9, Pixel{0, 0, 0, 255});
    const Image closed = morphological_close(block);
    EXPECT_EQ(closed.get_pixel(9, 9).r, 255);
}

TEST(ConnectedComponents, LabelsTwoSeparateBlobsDistinctly) {
    Image blobs(20, 10, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_rectangle(blobs, 1, 1, 3, 3, Pixel{255, 255, 255, 255}, true);
    draw_rectangle(blobs, 10, 1, 12, 3, Pixel{255, 255, 255, 255}, true);
    const LabelMap labels = connected_components(blobs);
    ASSERT_EQ(labels.count, 2);
    const int label_a = labels.labels[1 * 20 + 2];
    const int label_b = labels.labels[1 * 20 + 11];
    EXPECT_NE(label_a, 0);
    EXPECT_NE(label_b, 0);
    EXPECT_NE(label_a, label_b);
    EXPECT_EQ(labels.labels[0], 0); // background
}

TEST(ConnectedComponents, EightConnectivityMergesDiagonalTouchingBlobs) {
    Image diag(10, 10, ImageMode::RGB, Pixel{0, 0, 0, 255});
    diag.set_pixel(3, 3, Pixel{255, 255, 255, 255});
    diag.set_pixel(4, 4, Pixel{255, 255, 255, 255});
    EXPECT_EQ(connected_components(diag, true).count, 1);
    EXPECT_EQ(connected_components(diag, false).count, 2);
}

TEST(OtsuThreshold, ProducesAValidSeparatingThresholdForABimodalImage) {
    Image bimodal(20, 20, ImageMode::RGB, Pixel{0, 0, 0, 255});
    for (int y = 0; y < 20; ++y)
        for (int x = 10; x < 20; ++x) bimodal.set_pixel(x, y, Pixel{255, 255, 255, 255});
    const int t = otsu_threshold(bimodal);
    // Any threshold in (0, 255] correctly separates a strictly-{0,255}-valued image; the
    // meaningful check is that threshold(bimodal, t) reproduces the same two regions.
    const Image thresholded = threshold(bimodal, static_cast<std::uint8_t>(t));
    EXPECT_EQ(thresholded.get_pixel(0, 0).r, 0);
    EXPECT_EQ(thresholded.get_pixel(15, 0).r, 255);
}

TEST(KMeansSegment, SeparatesTwoWellSeparatedSolidColorRegions) {
    Image two_color(20, 20, ImageMode::RGB, Pixel{10, 10, 10, 255});
    for (int y = 0; y < 20; ++y)
        for (int x = 10; x < 20; ++x) two_color.set_pixel(x, y, Pixel{240, 240, 240, 255});
    const Image segmented = kmeans_segment(two_color, 2);
    const Pixel c1 = segmented.get_pixel(0, 0);
    const Pixel c2 = segmented.get_pixel(15, 0);
    const bool separated = (c1.r < 30 && c2.r > 220) || (c2.r < 30 && c1.r > 220);
    EXPECT_TRUE(separated) << "c1.r=" << int(c1.r) << " c2.r=" << int(c2.r);
}

TEST(KMeansSegment, RejectsNonPositiveK) {
    const Image img(10, 10, ImageMode::RGB);
    EXPECT_THROW(kmeans_segment(img, 0), std::invalid_argument);
}

TEST(KMeansSegment, RejectsKLargerThanPixelCount) {
    const Image img(2, 2, ImageMode::RGB);
    EXPECT_THROW(kmeans_segment(img, 10), std::invalid_argument);
}
