#include <datamunge/image/imaging.hpp>

#include <iostream>

using namespace datamunge::image;

namespace {
void print_pixel(const char* label, const Image& img, int x, int y) {
    const Pixel p = img.get_pixel(x, y);
    std::cout << label << " (" << x << "," << y << "): r=" << int(p.r) << " g=" << int(p.g) << " b=" << int(p.b)
              << " a=" << int(p.a) << "\n";
}
} // namespace

int main() {
    std::cout << "=================== Image core: construct, draw, inspect ===================\n";
    Image canvas(120, 90, ImageMode::RGB, Pixel{30, 30, 30, 255});
    draw_rectangle(canvas, 10, 10, 50, 50, Pixel{200, 60, 60, 255}, /*filled=*/true);
    draw_circle(canvas, 85, 30, 20, Pixel{60, 140, 220, 255}, /*filled=*/true);
    draw_line(canvas, 0, 89, 119, 0, Pixel{255, 255, 0, 255});
    draw_polygon(canvas, {{20, 60}, {60, 60}, {40, 88}}, Pixel{60, 220, 100, 255}, /*filled=*/true);
    std::cout << "canvas: " << canvas.width() << "x" << canvas.height() << ", mode=" << canvas.channels() << " channels\n";
    print_pixel("red square interior", canvas, 30, 30);
    print_pixel("blue circle interior", canvas, 85, 30);
    print_pixel("green triangle interior", canvas, 40, 75);
    std::cout << "\n";

    std::cout << "=================== Color conversion ===================\n";
    const Image gray = to_grayscale(canvas);
    print_pixel("grayscale of the red square", gray, 30, 30);
    const HSV red_hsv = rgb_to_hsv(200, 60, 60);
    std::cout << "the red square's HSV: h=" << red_hsv.h << " s=" << red_hsv.s << " v=" << red_hsv.v << "\n";
    const auto planes = split_channels(canvas);
    std::cout << "split_channels produced " << planes.size() << " single-channel planes\n\n";

    std::cout << "=================== Geometric transforms ===================\n";
    const Image resized = resize(canvas, 60, 45, ResampleFilter::Bilinear);
    std::cout << "resized to " << resized.width() << "x" << resized.height() << "\n";
    const Image cropped = crop(canvas, 10, 10, 40, 40);
    std::cout << "cropped the red square out: " << cropped.width() << "x" << cropped.height() << "\n";
    const Image rotated = rotate90(canvas, 90);
    std::cout << "rotated 90 degrees: " << rotated.width() << "x" << rotated.height() << " (dimensions swapped)\n";
    const Image flipped = flip_horizontal(canvas);
    print_pixel("flipped canvas, where the blue circle used to be on the right (now the left side)", flipped, 34, 30);
    std::cout << "\n";

    std::cout << "=================== Filters ===================\n";
    const Image blurred = gaussian_blur(canvas, 3.0);
    const Image sharpened = sharpen(canvas, 1.0);
    const Image edges = sobel_edges(canvas);
    const Image brighter = adjust_brightness(canvas, 40);
    const Image higher_contrast = adjust_contrast(canvas, 1.5);
    const Image bw = threshold(canvas, 128);
    std::cout << "applied gaussian_blur, sharpen, sobel_edges, adjust_brightness, adjust_contrast, threshold\n";
    print_pixel("brightened red square", brighter, 30, 30);
    print_pixel("thresholded red square (now pure black/white)", bw, 30, 30);
    std::cout << "\n";

    std::cout << "=================== File I/O: PPM, PGM, PBM, BMP, PNG ===================\n";
    write_ppm(canvas, "image_ex_output.ppm");
    write_pgm(gray, "image_ex_output.pgm");
    write_pbm(canvas, "image_ex_output.pbm");
    write_bmp(canvas, "image_ex_output.bmp");
    write_png(to_rgba(canvas), "image_ex_output.png");
    std::cout << "wrote image_ex_output.{ppm,pgm,pbm,bmp,png}\n";

    const Image reloaded_png = read_png("image_ex_output.png");
    std::cout << "read back image_ex_output.png: " << reloaded_png.width() << "x" << reloaded_png.height() << ", pixel (30,30) matches: "
              << std::boolalpha << (reloaded_png.get_pixel(30, 30).r == canvas.get_pixel(30, 30).r) << "\n";

    return 0;
}
