// CMD: just TARGET=naive_bayes_penguins_ex run-objc
// Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
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
            // Note: DMNaiveBayesClassifier drops incomplete rows for its own formula columns internally
            // when fitting, matching every other formula-based model in this library.
            DMDataFrame *penguins = [DMData penguins];
            println(@"penguins: %zu rows x %zu cols\n", [penguins nrows], [penguins ncols]);

            // A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
            // predictors in one formula -- each modeled independently given the class, per the naive
            // Bayes assumption.
            DMNaiveBayesClassifier *model = [[DMNaiveBayesClassifier alloc] initWithData:penguins
                formula:@"species ~ bill_length_mm + bill_depth_mm + island + sex"];
            [model printSummary];

            println(@"\nConfusion matrix (rows = actual, cols = predicted):\n%@", [[model confusionMatrix] toString]);

            println(@"\nMisclassified rows:");
            NSArray<NSString *> *predictions = [model predict:penguins];
            int misclassified = 0;
            NSArray<NSString *> *columns = @[@"species", @"bill_length_mm", @"bill_depth_mm", @"island", @"sex"];
            for (size_t i = 0; i < [penguins nrows]; i++) {
                BOOL incomplete = NO;
                for (NSString *c in columns)
                    if ([penguins isNull:c rowIndex:i]) incomplete = YES;
                if (incomplete) continue;
                NSString *actual = [penguins stringAt:@"species" rowIndex:i], *predicted = predictions[i];
                if ([predicted isEqualToString:actual]) continue;
                misclassified++;
                println(@"  row %zu: bill_length=%@ bill_depth=%@ island=%@ sex=%@  actual=%@  predicted=%@", i,
                        fmt([penguins numericAt:@"bill_length_mm" rowIndex:i]), fmt([penguins numericAt:@"bill_depth_mm" rowIndex:i]),
                        [penguins stringAt:@"island" rowIndex:i], [penguins stringAt:@"sex" rowIndex:i], actual, predicted);
            }
            println(@"%d misclassified (of %zu rows, some incomplete)", misclassified, [penguins nrows]);

            // -plotDecisionRegions:yFeature: requires exactly two NUMERIC predictors, so build a separate
            // two-predictor model (bill measurements alone) just for visualization.
            DMNaiveBayesClassifier *billOnly = [[DMNaiveBayesClassifier alloc] initWithData:penguins
                formula:@"species ~ bill_length_mm + bill_depth_mm"];
            println(@"\nbill-measurements-only model training accuracy: %@%%", fmt(100 * [billOnly trainingAccuracy]));
            [[billOnly plotClassification:penguins xFeature:@"bill_length_mm" yFeature:@"bill_depth_mm"]
                save:@"naive_bayes_penguins_classification.svg"];
            [[billOnly plotDecisionRegions:@"bill_length_mm" yFeature:@"bill_depth_mm"] save:@"naive_bayes_penguins_decision_regions.svg"];
            println(@"Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg");
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
