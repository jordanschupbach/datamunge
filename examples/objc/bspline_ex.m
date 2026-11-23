// CMD: just TARGET=bspline_ex run-objc
// Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
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

/// A double through a C printf format, e.g. cfmt("%.6f", x) for C#'s {x:F6}.
static NSString *cfmt(const char *format, double x) {
    char buf[64];
    snprintf(buf, sizeof buf, format, x);
    return @(buf);
}

int main(void) {
    @autoreleasepool {
        @try {

            // Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
            NSMutableArray<NSNumber *> *x = [NSMutableArray array], *y = [NSMutableArray array], *z = [NSMutableArray array];
            for (int iy = 0; iy < 8; iy++)
                for (int ix = 0; ix < 8; ix++) {
                    double xv = ix / 7.0, yv = iy / 7.0;
                    [x addObject:@(xv)];
                    [y addObject:@(yv)];
                    [z addObject:@(sin(xv) + yv * yv)];
                }
            DMDataFrame *data = [[DMDataFrame alloc] init];
            [data addNumericColumn:@"x" values:x];
            [data addNumericColumn:@"y" values:y];
            [data addNumericColumn:@"z" values:z];

            DMLm *surface = [[DMLm alloc] initWithData:data formula:@"z ~ bs(x, y)"];
            println(@"Fitted z ~ bs(x, y)");
            [surface printSummary];

            DMDataFrame *newPoints = [[DMDataFrame alloc] init];
            [newPoints addNumericColumn:@"x" values:@[@0.25, @0.75]];
            [newPoints addNumericColumn:@"y" values:@[@0.50, @0.25]];
            NSArray<NSNumber *> *pred = [surface predict:newPoints];
            println(@"Predictions: [%@, %@]", cfmt("%.6f", pred[0].doubleValue), cfmt("%.6f", pred[1].doubleValue));
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
