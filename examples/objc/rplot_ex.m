// CMD: just TARGET=rplot_ex run-objc
// R base-graphics-style plotting with RPlot / RLayout: plot, hist, barplot, boxplot, pie, qqnorm, layouts.
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
            // plot(x, y, type = "p") then abline() layered on afterward.
            DMRPlot *scatter = [DMRPlot plot:@[@1.0, @2.0, @3.0, @4.0, @5.0] y:@[@2.1, @3.9, @6.2, @7.8, @10.1] type:@"p" label:@"observed"];
            [scatter abline:0.0 slope:2.0 color:[[DMRgb alloc] initWithRed:220 green:38 blue:38] strokeWidth:1.5];
            [[[scatter title:@"plot() + abline()"] xLabel:@"x"] yLabel:@"y"];
            [scatter saveSvg:@"objc_rplot_scatter.svg"];

            // hist(): equal-width binning over the data range.
            DMRPlot *histogram = [DMRPlot hist:@[@1.0, @2, @2, @3, @3, @3, @4, @4, @4, @4, @5, @5, @5, @6, @6, @7] bins:6 label:@"counts"];
            [histogram saveSvg:@"objc_rplot_hist.svg"];

            // barplot(): categorical positions with x tick labels.
            [[[DMRPlot barplot:@[@23.0, @41.0, @12.0] names:@[@"A", @"B", @"C"]] title:@"barplot()"] saveSvg:@"objc_rplot_barplot.svg"];

            // boxplot(): Tukey five-number summary per group.
            [[[DMRPlot boxplot:@[@[@2.0, @4, @4, @4, @5, @5, @7, @9], @[@1.0, @2, @2, @2, @3, @3, @3, @3, @4, @20]]
                         names:@[@"low variance", @"has outlier"]] title:@"boxplot()"] saveSvg:@"objc_rplot_boxplot.svg"];

            // pie(): wedge areas proportional to value, axes hidden automatically.
            [[[DMRPlot pie:@[@35.0, @25.0, @20.0, @20.0] names:@[@"Q1", @"Q2", @"Q3", @"Q4"]] title:@"pie()"] saveSvg:@"objc_rplot_pie.svg"];

            // curve() is omitted to stay in step with the Lua example (Lua cannot supply the callback).

            // qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
            NSArray<NSNumber *> *residuals = @[@-2.1, @-1.3, @-0.8, @-0.4, @-0.1, @0.2, @0.5, @0.9, @1.4, @2.3];
            [[[DMRPlot qqnorm:residuals] qqline:residuals] saveSvg:@"objc_rplot_qqnorm.svg"];

            // par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
            [[[[DMRLayout create:1 cols:2] add:scatter] add:histogram] saveSvg:@"objc_rplot_layout.svg"];

            println(@"Wrote 7 SVGs to objc_rplot_*.svg");
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
