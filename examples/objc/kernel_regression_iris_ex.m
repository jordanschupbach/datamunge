// CMD: just TARGET=kernel_regression_iris_ex run-objc
// Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
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

/// A double as C#'s default ToString() prints it: the shortest round-trip digits, in E notation
/// when the decimal exponent is below -4 or at least 15 (6, 0.1, 1E-05, 1E+15).
static NSString *fmt(double x) {
    if (isnan(x)) return @"NaN";
    if (isinf(x)) return x < 0 ? @"-∞" : @"∞";
    if (x == 0) return signbit(x) ? @"-0" : @"0";
    char buf[48];
    int precision = 1;
    for (; precision < 17; precision++) {
        snprintf(buf, sizeof buf, "%.*e", precision - 1, x);
        if (strtod(buf, NULL) == x) break;
    }
    snprintf(buf, sizeof buf, "%.*e", precision - 1, x);
    const char *s = buf;
    BOOL negative = *s == '-';
    if (negative) s++;
    char digits[32];
    int n = 0;
    for (; *s && *s != 'e'; s++)
        if (*s != '.') digits[n++] = *s;
    while (n > 1 && digits[n - 1] == '0') n--;
    digits[n] = 0;
    int e = atoi(s + 1);
    NSMutableString *out = [NSMutableString stringWithString:negative ? @"-" : @""];
    if (e < -4 || e >= 15) {
        [out appendFormat:@"%c", digits[0]];
        if (n > 1) [out appendFormat:@".%s", digits + 1];
        [out appendFormat:@"E%c%02d", e < 0 ? '-' : '+', abs(e)];
    } else if (e < 0) {
        [out appendString:@"0."];
        for (int i = 0; i < -e - 1; i++) [out appendString:@"0"];
        [out appendFormat:@"%s", digits];
    } else if (e + 1 >= n) {
        [out appendFormat:@"%s", digits];
        for (int i = 0; i < e + 1 - n; i++) [out appendString:@"0"];
    } else {
        [out appendFormat:@"%.*s.%s", e + 1, digits, digits + e + 1];
    }
    return out;
}

int main(void) {
    @autoreleasepool {
        @try {
            NSString *formula = @"Petal.Length ~ Petal.Width";

            DMDataFrame *iris = [DMData iris];
            println(@"iris: %zu rows x %zu cols", iris.nrows, iris.ncols);
            println(@"formula: %@\n", formula);

            DMKernelRegression *model = [[DMKernelRegression alloc] initWithData:iris formula:formula];  // bandwidth auto-selected via leave-one-out CV
            [model printSummary];

            [[model plotFit:iris] save:@"kernel_regression_iris_fit.svg"];
            [[model plotCvCurve] save:@"kernel_regression_iris_cv.svg"];
            println(@"\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg");

            // A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
            // bias-variance story. initWithData:formula:kernel:bandwidth:nBandwidth:standardize:
            DMKernelRegression *small = [[DMKernelRegression alloc] initWithData:iris formula:formula kernel:@"gaussian" bandwidth:0.05];
            println(@"\nbandwidth=0.05 (too small): LOO R-squared=%@  LOO RMSE=%@", fmt([small rSquared]), fmt([small rmse]));
            [[small plotFit:iris] save:@"kernel_regression_iris_fit_small_bandwidth.svg"];

            DMKernelRegression *large = [[DMKernelRegression alloc] initWithData:iris formula:formula kernel:@"gaussian" bandwidth:5.0];
            println(@"bandwidth=5.0 (too large):  LOO R-squared=%@  LOO RMSE=%@", fmt([large rSquared]), fmt([large rmse]));
            [[large plotFit:iris] save:@"kernel_regression_iris_fit_large_bandwidth.svg"];

            println(@"bandwidth=%@ (CV-selected): LOO R-squared=%@  LOO RMSE=%@", fmt([model bandwidth]), fmt([model rSquared]),
                    fmt([model rmse]));
            println(@"\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg");
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
