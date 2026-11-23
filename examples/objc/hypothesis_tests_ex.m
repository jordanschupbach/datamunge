// CMD: just TARGET=hypothesis_tests_ex run-objc
// Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
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

static void printResult(NSString *label, DMHypothesisTestResult *r) {
    NSString *alternative = r.alternative == DMAlternativeLess      ? @"less"
                            : r.alternative == DMAlternativeGreater ? @"greater"
                                                                    : @"two.sided";
    NSMutableString *line = [NSMutableString stringWithFormat:@"%@: statistic=%@", label, fmt(r.statistic)];
    if (r.parameter1 != 0) [line appendFormat:@", df1=%@", fmt(r.parameter1)];
    if (r.parameter2 != 0) [line appendFormat:@", df2=%@", fmt(r.parameter2)];
    [line appendFormat:@", p=%@ (%@)", fmt(r.pValue), alternative];
    if (r.hasConfInt) [line appendFormat:@", CI=[%@, %@]", fmt(r.confIntLower), fmt(r.confIntUpper)];
    println(@"%@ -- %@", line, r.method);
}

static NSArray<NSNumber *> *columnForSpecies(DMDataFrame *df, NSString *column, NSString *species) {
    NSMutableArray<NSNumber *> *values = [NSMutableArray array];
    for (size_t i = 0; i < [df nrows]; i++)
        if ([[df stringAt:@"Species" rowIndex:i] isEqualToString:species])
            [values addObject:@([df numericAt:column rowIndex:i])];
    return values;
}

static NSArray<NSNumber *> *column(DMDataFrame *df, NSString *column) {
    NSMutableArray<NSNumber *> *values = [NSMutableArray array];
    for (size_t i = 0; i < [df nrows]; i++)
        [values addObject:@([df numericAt:column rowIndex:i])];
    return values;
}

static size_t countLong(NSArray<NSNumber *> *v, double threshold) {
    size_t count = 0;
    for (NSNumber *x in v)
        if (x.doubleValue > threshold) count++;
    return count;
}

int main(void) {
    @autoreleasepool {
        @try {
            DMDataFrame *iris = [DMData iris];

            NSArray<NSNumber *> *setosaPetal = columnForSpecies(iris, @"Petal.Length", @"setosa");
            NSArray<NSNumber *> *versicolorPetal = columnForSpecies(iris, @"Petal.Length", @"versicolor");
            NSArray<NSNumber *> *virginicaPetal = columnForSpecies(iris, @"Petal.Length", @"virginica");

            println(@"=================== t-tests: petal length, setosa vs. versicolor ===================");
            printResult(@"Welch two-sample t-test", [DM tTestTwoSample:setosaPetal y:versicolorPetal]);
            printResult(@"Wilcoxon rank-sum test", [DM wilcoxonRankSumTest:setosaPetal y:versicolorPetal]);

            println(@"\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================");
            NSArray<NSNumber *> *allPetal =
                [[setosaPetal arrayByAddingObjectsFromArray:versicolorPetal] arrayByAddingObjectsFromArray:virginicaPetal];
            NSArray<NSNumber *> *sizes = @[@(setosaPetal.count), @(versicolorPetal.count), @(virginicaPetal.count)];
            printResult(@"One-way ANOVA", [DM oneWayAnova:allPetal groupSizes:sizes]);
            printResult(@"Kruskal-Wallis", [DM kruskalWallisTest:allPetal groupSizes:sizes]);

            println(@"\n=================== Correlation: sepal length vs. petal length ===================");
            NSArray<NSNumber *> *sepalLength = column(iris, @"Sepal.Length");
            NSArray<NSNumber *> *petalLength = column(iris, @"Petal.Length");
            printResult(@"Pearson correlation", [DM pearsonCorrelationTest:sepalLength y:petalLength]);
            printResult(@"Spearman correlation", [DM spearmanCorrelationTest:sepalLength y:petalLength]);

            println(@"\n=================== F-test: petal length variance, setosa vs. virginica ===================");
            printResult(@"F test", [DM fTestVariance:setosaPetal y:virginicaPetal]);

            println(@"\n=================== Normality: is sepal length normally distributed within setosa? ===================");
            NSArray<NSNumber *> *setosaSepal = columnForSpecies(iris, @"Sepal.Length", @"setosa");
            printResult(@"Shapiro-Francia", [DM shapiroFranciaTest:setosaSepal]);
            printResult(@"KS vs. fitted normal", [DM ksTestOneSampleNormal:setosaSepal mean:5.006 sd:0.3525]);

            println(@"\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================");
            double medianAll = [[allPetal sortedArrayUsingSelector:@selector(compare:)][allPetal.count / 2] doubleValue];
            {
                size_t setosaLong = countLong(setosaPetal, medianAll);
                size_t setosaShort = setosaPetal.count - setosaLong;
                size_t versicolorLong = countLong(versicolorPetal, medianAll);
                size_t versicolorShort = versicolorPetal.count - versicolorLong;
                println(@"table: setosa=[%zu,%zu] versicolor=[%zu,%zu]", setosaLong, setosaShort, versicolorLong, versicolorShort);
                NSArray<NSNumber *> *table = @[@((double)setosaLong), @((double)setosaShort), @((double)versicolorLong),
                                               @((double)versicolorShort)];
                printResult(@"Chi-squared independence", [DM chiSquaredTestIndependence:table nrows:2 ncols:2]);
                printResult(@"Fisher's exact test",
                            [DM fisherExactTest2x2:setosaLong b:setosaShort c:versicolorLong d:versicolorShort]);
            }

            println(@"\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================");
            {
                size_t longCount = countLong(allPetal, medianAll);
                size_t n = allPetal.count;
                printResult(@"One-sample proportion test (vs 0.5)", [DM proportionTestOneSample:longCount n:n p:0.5]);
                printResult(@"Exact binomial test (vs 0.5)", [DM binomialTest:longCount n:n p:0.5]);
            }
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
