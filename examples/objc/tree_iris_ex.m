// CMD: just TARGET=tree_iris_ex run-objc
// Decision tree classifier on iris: summary, confusion matrix, misclassified rows and plots.
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
            DMDataFrame *iris = [DMData iris];
            println(@"iris: %zu rows x %zu cols\n", iris.nrows, iris.ncols);

            DMDecisionTreeClassifier *model = [[DMDecisionTreeClassifier alloc] initWithData:iris formula:@"Species ~ Petal.Length + Petal.Width"];
            [model printSummary];

            println(@"\nConfusion matrix (rows = actual, cols = predicted):");
            println(@"%@", [[model confusionMatrix] toString]);

            println(@"\nMisclassified rows:");
            NSArray<NSString *> *predictions = [model predict:iris];
            size_t n = iris.nrows;
            int misclassified = 0;
            for (size_t i = 0; i < n; i++) {
                NSString *actual = [iris stringAt:@"Species" rowIndex:i], *predicted = predictions[i];
                if ([predicted isEqualToString:actual]) continue;
                misclassified++;
                println(@"  row %zu: Petal.Length=%@ Petal.Width=%@  actual=%@  predicted=%@", i,
                        fmt([iris numericAt:@"Petal.Length" rowIndex:i]), fmt([iris numericAt:@"Petal.Width" rowIndex:i]),
                        actual, predicted);
            }
            println(@"%d of %zu misclassified (%@%%)", misclassified, n, fmt(100.0 * misclassified / n));

            [[model plotClassification:iris xFeature:@"Petal.Length" yFeature:@"Petal.Width"] save:@"tree_iris_classification.svg"];
            [[model plotDecisionRegions:@"Petal.Length" yFeature:@"Petal.Width"] save:@"tree_iris_decision_regions.svg"];
            println(@"\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg");

            // A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
            DMDecisionTreeClassifier *shallow = [[DMDecisionTreeClassifier alloc] initWithData:iris formula:@"Species ~ Petal.Length + Petal.Width" maxDepth:2];
            println(@"\nDepth-2 tree training accuracy: %@%% (%zu leaves)", fmt([shallow trainingAccuracy] * 100), [shallow leafCount]);
            [[shallow plotDecisionRegions:@"Petal.Length" yFeature:@"Petal.Width"] save:@"tree_iris_decision_regions_depth2.svg"];
            println(@"Saved tree_iris_decision_regions_depth2.svg");
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
