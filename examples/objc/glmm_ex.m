// CMD: just TARGET=glmm_ex run-objc
// Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
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
static double uniform(void) { return DotNetRandomNextDouble(&rng); }
static double gauss(void) {
    double u1 = 1 - uniform(), u2 = uniform();  // 1 - u keeps log() away from 0
    return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
}

int main(void) {
    @autoreleasepool {
        @try {
            DotNetRandomInit(&rng, 4242);

            println(@"=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================");
            // Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
            // confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
            // is not tied to location, so this is a well-behaved fit.
            {
                DMDataFrame *penguins = [DMData penguins];
                NSMutableArray<NSNumber *> *isMale = [NSMutableArray array], *bodyMass = [NSMutableArray array];
                NSMutableArray<NSString *> *island = [NSMutableArray array];
                for (size_t i = 0; i < [penguins nrows]; i++) {
                    if ([penguins isNull:@"sex" rowIndex:i] || [penguins isNull:@"body_mass_g" rowIndex:i] ||
                        [penguins isNull:@"island" rowIndex:i])
                        continue;
                    [isMale addObject:@([[penguins stringAt:@"sex" rowIndex:i] isEqualToString:@"male"] ? 1.0 : 0.0)];
                    [bodyMass addObject:@([penguins numericAt:@"body_mass_g" rowIndex:i])];
                    [island addObject:[penguins stringAt:@"island" rowIndex:i]];
                }
                DMDataFrame *sexDf = [[DMDataFrame alloc] init];
                [sexDf addNumericColumn:@"is_male" values:isMale];
                [sexDf addNumericColumn:@"body_mass_g" values:bodyMass];
                [sexDf addStringColumn:@"island" values:island];
                [[[DMGlmm alloc] initWithData:sexDf formula:@"is_male ~ body_mass_g + (1 | island)" family:@"binomial"]
                    printSummary];
            }

            println(@"\n=================== Poisson mixed model on simulated multi-site count data ===================");
            // 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
            // the baseline traffic and the promo's effectiveness vary by store.
            enum { nStores = 25 };
            double storeEffect[nStores];
            for (int s = 0; s < nStores; s++) storeEffect[s] = gauss() * 0.4;
            const double trueIntercept = 2.0;
            const double trueSlope = 0.3;

            NSMutableArray<NSNumber *> *store = [NSMutableArray array], *promo = [NSMutableArray array],
                                       *visits = [NSMutableArray array];
            for (int s = 0; s < nStores; s++) {
                int nDays = 15 + DotNetRandomNext(&rng, 0, 11);
                for (int day = 0; day < nDays; day++) {
                    double promoIntensity = uniform() * 3;
                    double lam = exp(trueIntercept + storeEffect[s] + trueSlope * promoIntensity);
                    // Knuth's Poisson sampler.
                    double threshold = exp(-lam), p = 1.0;
                    int k = 0;
                    do {
                        k++;
                        p *= uniform();
                    } while (p > threshold);
                    [store addObject:@((double)s)];
                    [promo addObject:@(promoIntensity)];
                    [visits addObject:@((double)(k - 1))];
                }
            }
            DMDataFrame *df = [[DMDataFrame alloc] init];
            [df addNumericColumn:@"store" values:store];
            [df addNumericColumn:@"promo" values:promo];
            [df addNumericColumn:@"visits" values:visits];
            DMGlmm *storeModel = [[DMGlmm alloc] initWithData:df formula:@"visits ~ promo + (1 | store)" family:@"poisson"];

            [storeModel printSummary];

            println(@"\nTrue generating values: intercept=%@, slope=%@, random-intercept SD (log scale)=0.4",
                    fmt(trueIntercept), fmt(trueSlope));

            println(@"\n--- BLUPs for a few stores ---");
            NSArray<NSString *> *labels = [storeModel groupLabels];
            for (size_t idx = 0; idx < 3; idx++)
                println(@"store %@: intercept shift=%@", labels[idx],
                        fmt([storeModel randomEffectsForGroup:idx][0].doubleValue));

            println(@"\n--- Prediction: population-level vs. store-adjusted ---");
            DMDataFrame *population = [[DMDataFrame alloc] init];
            DMDataFrame *store0 = [[DMDataFrame alloc] init];
            [population addNumericColumn:@"promo" values:@[@1.5]];
            [store0 addNumericColumn:@"promo" values:@[@1.5]];
            [store0 addNumericColumn:@"store" values:@[@0.0]];
            println(@"promo=1.5, unseen store:   %@ expected visits (fixed effects only)",
                    fmt([storeModel predict:population][0].doubleValue));
            println(@"promo=1.5, store 0 (known): %@ expected visits (fixed effects + store 0's BLUP)",
                    fmt([storeModel predict:store0][0].doubleValue));
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
