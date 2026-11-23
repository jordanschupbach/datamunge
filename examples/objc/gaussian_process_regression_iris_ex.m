// CMD: just TARGET=gaussian_process_regression_iris_ex run-objc
// Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals.
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
            NSString *formula = @"Petal.Length ~ Petal.Width";

            DMDataFrame *iris = [DMData iris];
            println(@"iris: %zu rows x %zu cols", iris.nrows, iris.ncols);
            println(@"formula: %@\n", formula);

            // Both the length scale and the noise ratio are auto-selected by maximizing the exact log
            // marginal likelihood.
            DMGaussianProcessRegression *model = [[DMGaussianProcessRegression alloc] initWithData:iris formula:formula];
            [model printSummary];

            [[model plotFit:iris] save:@"gpr_iris_fit.svg"];
            [[model plotLengthScaleProfile] save:@"gpr_iris_length_scale_profile.svg"];
            println(@"\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg");

            // Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
            // interval at every point -- including far outside the training data, where it should widen
            // substantially as the model's uncertainty grows.
            DMDataFrame *query = [[DMDataFrame alloc] init];
            [query addNumericColumn:@"Petal.Width" values:@[@0.2, @1.3, @2.5, @10.0]];
            DMDataFrame *detail = [model predictFrame:query intervalKind:@"confidence"];
            println(@"\nPredictions with 95%% confidence intervals:");
            println(@"%@", [detail toString]);
            println(@"(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n interval is than the in-range predictions.)");
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
