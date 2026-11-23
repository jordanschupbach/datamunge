// CMD: just TARGET=timeseries_ex run-objc
// Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
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


static DotNetRandom rng;
static void seed(int n) { DotNetRandomInit(&rng, n); }
static double uniform(void) { return 1 - DotNetRandomNextDouble(&rng); }  // (0, 1]: avoids log(0) in the Box-Muller transform.
static double gauss(void) { return sqrt(-2 * log(uniform())) * cos(2 * M_PI * uniform()); }
static NSString *joined(NSArray<NSNumber *> *v) {
    NSMutableArray<NSString *> *parts = [NSMutableArray array];
    for (NSNumber *x in v) [parts addObject:fmt(x.doubleValue)];
    return [parts componentsJoinedByString:@", "];
}

int main(void) {
    @autoreleasepool {
        @try {

            println(@"=================== ARIMA(1,1,1) on a simulated random walk with drift ===================");
            seed(42);
            NSMutableArray<NSNumber *> *y = [NSMutableArray array];
            {
                double level = 100.0, prevShock = 0.0;
                for (int i = 0; i < 150; i++) {
                    double shock = gauss();
                    level += 0.3 + 0.4 * prevShock + shock;  // integrated ARMA(0,1) increments with drift
                    [y addObject:@(level)];
                    prevShock = shock;
                }
            }

            DMArimaOptions *options = [[DMArimaOptions alloc] init];
            options.p = 1;
            options.d = 1;
            options.q = 1;
            options.dePopulationSize = 80;
            options.deMaxGenerations = 400;
            DMArima *model = [[DMArima alloc] initWithY:y options:options];

            println(@"AR coefficient: %@", fmt([model arCoefficients][0].doubleValue));
            println(@"MA coefficient: %@", fmt([model maCoefficients][0].doubleValue));
            println(@"sigma^2: %@, AIC: %@, BIC: %@", fmt([model sigma2]), fmt([model aic]), fmt([model bic]));

            NSArray *intervals = [model forecastWithIntervals:6];  // a std::pair: [values, std. errors]
            println(@"6-step forecast: %@", joined(intervals[0]));
            println(@"forecast std. errors: %@", joined(intervals[1]));

            println(@"\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================");
            seed(7);
            NSMutableArray<NSNumber *> *s = [NSMutableArray array];
            {
                double prev = 0.0;
                for (int i = 0; i < 120; i++) {
                    prev = 0.5 * prev + gauss();
                    [s addObject:@(20 + 0.2 * i + 5 * sin(2 * M_PI * i / 12) + prev)];
                }
            }

            DMArimaOptions *options2 = [[DMArimaOptions alloc] init];
            options2.p = 1;
            options2.seasonalP = 1;
            options2.seasonalD = 1;
            options2.seasonalPeriod = 12;
            options2.dePopulationSize = 100;
            options2.deMaxGenerations = 500;
            DMArima *model2 = [[DMArima alloc] initWithY:s options:options2];
            println(@"AR coefficient: %@, seasonal AR coefficient: %@", fmt([model2 arCoefficients][0].doubleValue),
                    fmt([model2 seasonalArCoefficients][0].doubleValue));
            println(@"12-step forecast: %@", joined([model2 forecast:12]));

            println(@"\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================");
            const double seasonalShape[12] = {0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6};
            seed(11);
            NSMutableArray<NSNumber *> *y2 = [NSMutableArray array];
            for (int i = 0; i < 48; i++) [y2 addObject:@((100 + 2 * i) * seasonalShape[i % 12] + gauss() * 3)];

            DMExponentialSmoothingOptions *esOptions = [[DMExponentialSmoothingOptions alloc] init];
            esOptions.trend = DMTrendTypeAdditive;
            esOptions.seasonal = DMSeasonalTypeMultiplicative;
            esOptions.seasonalPeriod = 12;
            esOptions.dePopulationSize = 60;
            esOptions.deMaxGenerations = 300;
            DMExponentialSmoothing *esModel = [[DMExponentialSmoothing alloc] initWithY:y2 options:esOptions];
            println(@"alpha=%@ beta=%@ gamma=%@", fmt([esModel alpha]), fmt([esModel beta]), fmt([esModel gamma]));
            println(@"sigma^2: %@, AIC: %@", fmt([esModel sigma2]), fmt([esModel aic]));
            println(@"12-month forecast: %@", joined([esModel forecast:12]));

            println(@"\n=================== Simple exponential smoothing vs. Holt's linear trend ===================");
            NSArray<NSNumber *> *flat = @[@50.2, @49.8, @50.5, @49.6, @50.1, @50.3, @49.9, @50.0, @50.4, @49.7];

            DMExponentialSmoothing *ses =
                [[DMExponentialSmoothing alloc] initWithY:flat options:[[DMExponentialSmoothingOptions alloc] init]];
            println(@"SES alpha= %@", fmt([ses alpha]));
            println(@"SES 5-step forecast: %@", joined([ses forecast:5]));

            DMExponentialSmoothingOptions *holtOptions = [[DMExponentialSmoothingOptions alloc] init];
            holtOptions.trend = DMTrendTypeAdditive;
            DMExponentialSmoothing *holt = [[DMExponentialSmoothing alloc] initWithY:flat options:holtOptions];
            println(@"Holt alpha=%@ beta=%@", fmt([holt alpha]), fmt([holt beta]));
            println(@"Holt 5-step forecast: %@", joined([holt forecast:5]));
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
