// CMD: just TARGET=glm_iris_ex run-objc
// Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
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
            size_t nrows = [iris nrows];

            // Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
            // the other two on these predictors, which sends logistic regression's coefficients toward
            // +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
            // binary split for a demo.
            NSMutableArray<NSNumber *> *isVirginica = [NSMutableArray array], *petalLength = [NSMutableArray array],
                                       *petalWidth = [NSMutableArray array];
            for (size_t i = 0; i < nrows; i++) {
                NSString *sp = [iris stringAt:@"Species" rowIndex:i];
                if (!([sp isEqualToString:@"versicolor"] || [sp isEqualToString:@"virginica"])) continue;
                [isVirginica addObject:@([sp isEqualToString:@"virginica"] ? 1.0 : 0.0)];
                [petalLength addObject:@([iris numericAt:@"Petal.Length" rowIndex:i])];
                [petalWidth addObject:@([iris numericAt:@"Petal.Width" rowIndex:i])];
            }

            DMDataFrame *sub = [[DMDataFrame alloc] init];
            [sub addNumericColumn:@"Petal.Length" values:petalLength];
            [sub addNumericColumn:@"Petal.Width" values:petalWidth];
            [sub addNumericColumn:@"is_virginica" values:isVirginica];

            println(@"=================== Logistic regression (binomial, logit link) ===================");
            DMGlm *logit = [[DMGlm alloc] initWithData:sub formula:@"is_virginica ~ Petal.Length + Petal.Width" family:@"binomial"];
            [logit printSummary];

            NSArray<NSNumber *> *fitted = [logit fittedValues];
            NSUInteger n = isVirginica.count;
            int correct = 0;
            for (NSUInteger i = 0; i < n; i++)
                if ((fitted[i].doubleValue >= 0.5) == (isVirginica[i].doubleValue >= 0.5)) correct++;
            println(@"\nResubstitution accuracy at 0.5 threshold: %@%%", fmt(100.0 * correct / n));

            [logit saveDiagnosticPlots:@"glm_logistic_iris"];
            println(@"\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg");

            // Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
            // classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
            {
                const int gridN = 100;
                double widthSum = 0, plMin = INFINITY, plMax = -INFINITY;
                for (NSNumber *w in petalWidth) widthSum += w.doubleValue;
                for (NSNumber *l in petalLength) {
                    plMin = MIN(plMin, l.doubleValue);
                    plMax = MAX(plMax, l.doubleValue);
                }
                double widthMean = widthSum / petalWidth.count;
                plMin -= 0.3;
                plMax += 0.3;
                NSMutableArray<NSNumber *> *lengths = [NSMutableArray array], *widths = [NSMutableArray array];
                for (int i = 0; i < gridN; i++) {
                    [lengths addObject:@(plMin + (plMax - plMin) * i / (gridN - 1))];
                    [widths addObject:@(widthMean)];
                }
                DMDataFrame *grid = [[DMDataFrame alloc] init];
                [grid addNumericColumn:@"Petal.Length" values:lengths];
                [grid addNumericColumn:@"Petal.Width" values:widths];
                DMDataFrame *curveFrame = [logit predictFrame:grid intervalKind:@"confidence"];
                println(@"\nPredicted-probability curve (first 5 rows):");
                println(@"%@", [curveFrame toString:5]);
            }

            // Poisson regression, for contrast: same IRLS engine, different family/link.
            println(@"\n=================== Poisson regression (log link) ===================");
            NSMutableArray<NSNumber *> *sepalWidth = [NSMutableArray array], *allPetalLength = [NSMutableArray array],
                                       *count = [NSMutableArray array];
            for (size_t i = 0; i < nrows; i++) {
                [sepalWidth addObject:@([iris numericAt:@"Sepal.Width" rowIndex:i])];
                [allPetalLength addObject:@([iris numericAt:@"Petal.Length" rowIndex:i])];
                [count addObject:@(floor([iris numericAt:@"Sepal.Length" rowIndex:i] + 0.5))];
            }
            DMDataFrame *countData = [[DMDataFrame alloc] init];
            [countData addNumericColumn:@"Sepal.Width" values:sepalWidth];
            [countData addNumericColumn:@"Petal.Length" values:allPetalLength];
            [countData addNumericColumn:@"count" values:count];
            [[[DMGlm alloc] initWithData:countData formula:@"count ~ Sepal.Width + Petal.Length" family:@"poisson"] printSummary];
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
