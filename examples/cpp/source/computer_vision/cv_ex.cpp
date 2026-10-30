#include <datamunge/cv/cv.hpp>
#include <datamunge/image/imaging.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace datamunge::cv;
using namespace datamunge::image;
using datamunge::linalg::Tensor;

namespace {
Image make_checkerboard(int size, int cell) {
    Image img(size, size, ImageMode::RGB, Pixel{0, 0, 0, 255});
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
            if (((x / cell) + (y / cell)) % 2 == 0) img.set_pixel(x, y, Pixel{255, 255, 255, 255});
    return img;
}
} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "=================== Corner and blob detection ===================\n";
    const Image checker = make_checkerboard(80, 20);
    const auto harris = harris_corners(checker);
    const auto shi_tomasi = shi_tomasi_corners(checker);
    std::cout << "Harris found " << harris.size() << " corners, Shi-Tomasi found " << shi_tomasi.size() << " (checkerboard intersections)\n";

    Image square(80, 80, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_rectangle(square, 20, 20, 59, 59, Pixel{255, 255, 255, 255}, true);
    const auto fast = fast_corners(square);
    std::cout << "FAST found " << fast.size() << " corners on a filled square (its 4 true corners)\n";

    Image disk(80, 80, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_circle(disk, 40, 40, 8, Pixel{255, 255, 255, 255}, true);
    // num_scales/sigma0 tuned so the scale-space search actually spans this disk's radius --
    // the defaults are tuned for smaller blobs than this one.
    const auto blobs = dog_blobs(disk, 5, 1.0, 2.0);
    std::cout << "Difference-of-Gaussians found " << blobs.size() << " blob(s) on a solid disk\n\n";

    std::cout << "=================== Hough line and circle transforms ===================\n";
    Image lines_img(100, 100, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_line(lines_img, 10, 30, 90, 30, Pixel{255, 255, 255, 255});
    draw_line(lines_img, 20, 10, 20, 90, Pixel{255, 255, 255, 255});
    const auto hough_lines_result = hough_lines(lines_img);
    std::cout << "Hough transform found " << hough_lines_result.size() << " line(s); strongest has " << hough_lines_result.front().votes
              << " votes\n";

    Image circle_img(100, 100, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_circle(circle_img, 50, 50, 25, Pixel{255, 255, 255, 255}, false);
    const auto hough_circles_result = hough_circles(circle_img, 20, 30);
    if (!hough_circles_result.empty())
        std::cout << "Hough circle transform found a circle near (" << hough_circles_result.front().cx << ", "
                  << hough_circles_result.front().cy << ") with radius " << hough_circles_result.front().radius << "\n\n";

    std::cout << "=================== Morphology and segmentation ===================\n";
    Image blobs_img(40, 20, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_rectangle(blobs_img, 2, 2, 8, 8, Pixel{255, 255, 255, 255}, true);
    draw_rectangle(blobs_img, 20, 2, 26, 8, Pixel{255, 255, 255, 255}, true);
    const auto labels = connected_components(blobs_img);
    std::cout << "connected_components found " << labels.count << " components\n";
    const Image opened = morphological_open(blobs_img);
    const Image closed = morphological_close(blobs_img);
    std::cout << "applied morphological_open and morphological_close\n";

    const int otsu_level = otsu_threshold(checker);
    std::cout << "Otsu automatic threshold on the checkerboard: " << otsu_level << "\n";

    Image two_color(40, 20, ImageMode::RGB, Pixel{20, 20, 20, 255});
    for (int y = 0; y < 20; ++y)
        for (int x = 20; x < 40; ++x) two_color.set_pixel(x, y, Pixel{230, 230, 230, 255});
    const Image segmented = kmeans_segment(two_color, 2);
    std::cout << "kmeans_segment cluster colors: " << int(segmented.get_pixel(0, 0).r) << " and " << int(segmented.get_pixel(30, 0).r)
              << "\n\n";

    std::cout << "=================== Pyramids and optical flow ===================\n";
    const auto gpyr = gaussian_pyramid(checker, 4);
    std::cout << "Gaussian pyramid levels: ";
    for (const auto& level : gpyr) std::cout << level.width() << "x" << level.height() << " ";
    std::cout << "\n";
    const auto lpyr = laplacian_pyramid(checker, 4);
    std::cout << "Laplacian pyramid has " << lpyr.size() << " levels\n";

    // Lucas-Kanade needs genuine 2D texture: a pattern that varies in BOTH x and y (a pattern
    // varying in only one direction is the textbook "aperture problem" -- correctly flagged as
    // invalid, not something worth demonstrating here).
    const auto pattern_at = [](int x, int y) {
        const double v = 128.0 + 60.0 * std::sin(x / 6.0) + 60.0 * std::sin(y / 7.0);
        return static_cast<std::uint8_t>(std::max(0.0, std::min(255.0, v)));
    };
    Image frame1(80, 80, ImageMode::RGB, Pixel{0, 0, 0, 255});
    for (int y = 0; y < 80; ++y)
        for (int x = 0; x < 80; ++x) {
            const std::uint8_t v = pattern_at(x, y);
            frame1.set_pixel(x, y, Pixel{v, v, v, 255});
        }
    Image frame2(80, 80, ImageMode::RGB, Pixel{0, 0, 0, 255});
    for (int y = 0; y < 80; ++y)
        for (int x = 0; x < 80; ++x) {
            const std::uint8_t v = pattern_at(x - 3, y - 1); // the pattern shifted right 3, down 1
            frame2.set_pixel(x, y, Pixel{v, v, v, 255});
        }
    const auto flow = lucas_kanade_optical_flow(frame1, frame2, {{40, 40}}, 10);
    std::cout << "Lucas-Kanade optical flow at (40,40): valid=" << std::boolalpha << flow[0].valid << " dx=" << flow[0].dx
              << " dy=" << flow[0].dy << " (a 3px rightward shift)\n\n";

    std::cout << "=================== Basic NN inference scaffolding (on linalg::Tensor) ===================\n";
    const Tensor img_tensor = image_to_tensor(checker);
    std::cout << "image_to_tensor produced shape [" << img_tensor.shape()[0] << ", " << img_tensor.shape()[1] << ", "
              << img_tensor.shape()[2] << "]\n";

    const Tensor kernel = Tensor::from_values({1, 3, 3, 3}, std::vector<double>(27, 1.0 / 27.0)); // a 3x3 box-blur kernel
    const Tensor bias = Tensor::zeros({1});
    const Tensor conv_out = conv2d(img_tensor, kernel, bias, /*stride=*/2, /*padding=*/1);
    std::cout << "conv2d output shape: [" << conv_out.shape()[0] << ", " << conv_out.shape()[1] << ", " << conv_out.shape()[2] << "]\n";

    const Tensor pooled = max_pool2d(conv_out, 2);
    std::cout << "max_pool2d output shape: [" << pooled.shape()[0] << ", " << pooled.shape()[1] << ", " << pooled.shape()[2] << "]\n";

    const Tensor activated = relu(pooled);
    std::cout << "relu output sum: " << activated.sum() << "\n";

    const Tensor logits = Tensor::from_values({3}, {1.0, 0.5, -0.5});
    const Tensor probs = softmax(logits);
    std::cout << "softmax([1.0, 0.5, -0.5]) sums to " << (probs.at_flat(0) + probs.at_flat(1) + probs.at_flat(2)) << "\n";

    return 0;
}
