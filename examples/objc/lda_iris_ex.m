// CMD: just TARGET=lda_iris_ex run-objc
// Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.
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
            println(@"iris: %zu rows x %zu cols\n", iris.nrows, iris.ncols);

            DMLda *model = [[DMLda alloc] initWithData:iris formula:@"Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"];
            [model printSummary];

            println(@"\nConfusion matrix (rows = actual, cols = predicted):\n%@", [[model confusionMatrix] toString]);

            DMDataFrame *newdata = [[DMDataFrame alloc] init];
            [newdata addNumericColumn:@"Sepal.Length" values:@[@5.1, @6.0, @6.5, @6.2]];
            [newdata addNumericColumn:@"Sepal.Width" values:@[@3.5, @2.7, @3.0, @2.8]];
            [newdata addNumericColumn:@"Petal.Length" values:@[@1.4, @4.5, @5.5, @4.8]];
            [newdata addNumericColumn:@"Petal.Width" values:@[@0.2, @1.5, @2.0, @1.8]];
            println(@"\nPredictions for new flowers:\n%@", [[model predictFrame:newdata] toString]);

            [model saveDiscriminantPlot:@"lda_iris_discriminants.svg"];
            println(@"\nSaved discriminant plot as lda_iris_discriminants.svg");
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
