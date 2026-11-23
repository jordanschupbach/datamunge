// CMD: just TARGET=svm_iris_ex run-objc
// Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
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
            NSString *formula = @"Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width";
            DMDataFrame *iris = [DMData iris];
            println(@"iris: %zu rows x %zu cols\n", iris.nrows, iris.ncols);

            println(@"=== RBF kernel (default) ===");
            DMSvm *rbfModel = [[DMSvm alloc] initWithData:iris formula:formula];
            [rbfModel printSummary];
            println(@"\nConfusion matrix (rows = actual, cols = predicted):\n%@", [[rbfModel confusionMatrix] toString]);

            println(@"\n=== Linear kernel, for comparison ===");
            DMSvm *linearModel = [[DMSvm alloc] initWithData:iris formula:formula kernel:@"linear"];
            println(@"Training accuracy: %@%%", fmt([linearModel trainingAccuracy] * 100));
            println(@"Support vectors: %zu", [linearModel numSupportVectors]);

            DMDataFrame *newdata = [[DMDataFrame alloc] init];
            [newdata addNumericColumn:@"Sepal.Length" values:@[@5.1, @6.0, @6.5, @6.2]];
            [newdata addNumericColumn:@"Sepal.Width" values:@[@3.5, @2.7, @3.0, @2.8]];
            [newdata addNumericColumn:@"Petal.Length" values:@[@1.4, @4.5, @5.5, @4.8]];
            [newdata addNumericColumn:@"Petal.Width" values:@[@0.2, @1.5, @2.0, @1.8]];
            println(@"\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n%@",
                    [[rbfModel predictFrame:newdata] toString]);
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
