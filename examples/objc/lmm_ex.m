// CMD: just TARGET=lmm_ex run-objc
// Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
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

/// System.Random(seed) from .NET (the legacy subtractive generator a seeded Random uses).
typedef struct {
    int seeds[56];
    int inext, inextp;
} DotNetRandom;

static void DotNetRandomInit(DotNetRandom *r, int seed) {
    const int mbig = 2147483647;
    int mj = 161803398 - (seed == INT_MIN ? mbig : abs(seed));
    r->seeds[55] = mj;
    int mk = 1;
    for (int i = 1; i < 55; i++) {
        int ii = (21 * i) % 55;
        r->seeds[ii] = mk;
        mk = mj - mk;
        if (mk < 0) mk += mbig;
        mj = r->seeds[ii];
    }
    for (int k = 1; k < 5; k++)
        for (int i = 1; i < 56; i++) {
            r->seeds[i] -= r->seeds[1 + (i + 30) % 55];
            if (r->seeds[i] < 0) r->seeds[i] += mbig;
        }
    r->inext = 0;
    r->inextp = 21;
}

/// Random.NextDouble(): [0, 1).
static double DotNetRandomNextDouble(DotNetRandom *r) {
    const int mbig = 2147483647;
    r->inext = r->inext + 1 >= 56 ? 1 : r->inext + 1;
    r->inextp = r->inextp + 1 >= 56 ? 1 : r->inextp + 1;
    int value = r->seeds[r->inext] - r->seeds[r->inextp];
    if (value == mbig) value--;
    if (value < 0) value += mbig;
    r->seeds[r->inext] = value;
    return value * (1.0 / mbig);
}

/// Random.Next(minValue, maxValue): [minValue, maxValue).
static int DotNetRandomNext(DotNetRandom *r, int minValue, int maxValue) {
    return (int)(DotNetRandomNextDouble(r) * (double)(maxValue - minValue)) + minValue;
}

static DotNetRandom rng;
static double uniform(void) { return 1 - DotNetRandomNextDouble(&rng); }  // (0, 1]: avoids log(0) in the Box-Muller transform.
static double gauss(void) {
    double u1 = uniform(), u2 = uniform();  // C# evaluates the operands left to right
    return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
}

int main(void) {
    @autoreleasepool {
        @try {
            DotNetRandomInit(&rng, 2024);

            println(@"=================== Random intercept on a real dataset (penguins) ===================");
            [[[DMLmm alloc] initWithData:[DMData penguins] formula:@"body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)"]
                printSummary];

            println(@"\n=================== Random intercept + slope on a simulated multi-school dataset ===================");
            // 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
            // and the return on an extra hour of study vary by school -- the textbook case for a
            // random-intercept-and-slope model.
            enum { nSchools = 30 };
            double schoolIntercept[nSchools], schoolSlope[nSchools];
            for (int s = 0; s < nSchools; s++) schoolIntercept[s] = gauss() * 6;
            for (int s = 0; s < nSchools; s++) schoolSlope[s] = gauss() * 1.2;
            const double trueIntercept = 60.0;
            const double trueSlope = 3.0;

            NSMutableArray<NSNumber *> *school = [NSMutableArray array], *studyHours = [NSMutableArray array],
                                       *score = [NSMutableArray array];
            for (int s = 0; s < nSchools; s++) {
                int nStudents = 15 + DotNetRandomNext(&rng, 0, 21);
                for (int student = 0; student < nStudents; student++) {
                    double hours = DotNetRandomNextDouble(&rng) * 10;
                    [school addObject:@((double)s)];
                    [studyHours addObject:@(hours)];
                    [score addObject:@(trueIntercept + schoolIntercept[s] + (trueSlope + schoolSlope[s]) * hours + gauss() * 4)];
                }
            }
            DMDataFrame *df = [[DMDataFrame alloc] init];
            [df addNumericColumn:@"school" values:school];
            [df addNumericColumn:@"study_hours" values:studyHours];
            [df addNumericColumn:@"score" values:score];

            DMLmm *model = [[DMLmm alloc] initWithData:df formula:@"score ~ study_hours + (1 + study_hours | school)"];
            [model printSummary];

            println(@"\nTrue generating values: intercept=%@, slope=%@, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0",
                    fmt(trueIntercept), fmt(trueSlope));

            println(@"\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---");
            NSArray<NSString *> *labels = [model groupLabels];
            for (int idx = 0; idx < 3; idx++) {
                NSArray<NSNumber *> *re = [model randomEffectsForGroup:idx];
                println(@"school %@: intercept shift=%@, slope shift=%@", labels[idx], fmt(re[0].doubleValue), fmt(re[1].doubleValue));
            }

            println(@"\n--- Prediction: population-level vs. school-adjusted ---");
            DMDataFrame *population = [[DMDataFrame alloc] init];
            DMDataFrame *school0 = [[DMDataFrame alloc] init];
            [population addNumericColumn:@"study_hours" values:@[@5.0]];
            [school0 addNumericColumn:@"study_hours" values:@[@5.0]];
            [school0 addNumericColumn:@"school" values:@[@0.0]];
            println(@"5 study hours, unseen school:      %@ (fixed effects only)", fmt([model predict:population][0].doubleValue));
            println(@"5 study hours, school 0 (known):    %@ (fixed effects + school 0's BLUP)",
                    fmt([model predict:school0][0].doubleValue));
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
