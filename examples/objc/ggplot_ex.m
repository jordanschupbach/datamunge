// CMD: just TARGET=ggplot_ex run-objc
// Demonstrates DMGgPlot -- the ggplot2-style grammar-of-graphics library.
#import "DM.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// Prints a formatted line to stdout (C stdio, which the C++ library's output shares).
static void println(NSString *format, ...) NS_FORMAT_FUNCTION(1, 2);
static void println(NSString *format, ...) {
    va_list args;
    va_start(args, format);
    NSString *line = [[NSString alloc] initWithFormat:format arguments:args];
    va_end(args);
    printf("%s\n", line.UTF8String);
}

int main(void) {
    @autoreleasepool {
        @try {
            DMDataFrame *iris = [DMData iris];

            // ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
            [[[[[[DMGgPlot alloc] initWithData:iris xColumn:@"Sepal.Length" yColumn:@"Sepal.Width" colorColumn:@"Species"]
                geomPoint]
                labs:@"Iris Sepal Dimensions" x:@"Sepal Length" y:@"Sepal Width"]
                themeMinimal]
                saveSvg:@"objc_ggplot_point.svg"];

            // + geom_smooth(): an lm() fit line, reusing stats::LM internally.
            [[[[[[DMGgPlot alloc] initWithData:iris xColumn:@"Sepal.Length" yColumn:@"Petal.Length"]
                geomPoint:[[DMRgb alloc] initWithRed:156 green:163 blue:175] size:2.5]
                geomSmooth]
                labs:@"Petal Length vs Sepal Length With a Linear Fit" x:@"Sepal Length" y:@"Petal Length"]
                saveSvg:@"objc_ggplot_smooth.svg"];

            // geom_bar(): counts a discrete column (stat = "count").
            [[[[[[DMGgPlot alloc] initWithData:iris xColumn:@"Species"]
                geomBar]
                labs:@"Observations per Species" x:@"Species" y:@"Count"]
                themeBw]
                saveSvg:@"objc_ggplot_bar.svg"];

            // geom_boxplot(): grouped by a discrete x column.
            [[[[[DMGgPlot alloc] initWithData:iris xColumn:@"Species" yColumn:@"Petal.Width"]
                geomBoxplot]
                labs:@"Petal Width by Species" x:@"Species" y:@"Petal Width"]
                saveSvg:@"objc_ggplot_boxplot.svg"];

            // geom_histogram() + geom_density(): distribution of a single numeric column.
            [[[[[DMGgPlot alloc] initWithData:iris xColumn:@"Sepal.Length"]
                geomHistogram:20]
                labs:@"Distribution of Sepal Length" x:@"Sepal Length" y:@"Count"]
                saveSvg:@"objc_ggplot_histogram.svg"];

            [[[[[DMGgPlot alloc] initWithData:iris xColumn:@"Sepal.Length"]
                geomDensity]
                labs:@"Density of Sepal Length" x:@"Sepal Length" y:@"Density"]
                saveSvg:@"objc_ggplot_density.svg"];

            // facet_wrap(): one panel per Species, composed via RLayout under the hood.
            [[[[[[DMGgPlot alloc] initWithData:iris xColumn:@"Petal.Length" yColumn:@"Petal.Width"]
                geomPoint]
                facetWrap:@"Species"]
                labs:@"Petal Dimensions" x:@"Petal Length" y:@"Petal Width"]
                saveSvg:@"objc_ggplot_facet.svg"];

            // theme_classic(): another built-in theme.
            [[[[[DMGgPlot alloc] initWithData:iris xColumn:@"Sepal.Length" yColumn:@"Sepal.Width" colorColumn:@"Species"]
                geomPoint]
                themeClassic]
                saveSvg:@"objc_ggplot_classic.svg"];

            println(@"Wrote 7 SVGs to objc_ggplot_*.svg");
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
