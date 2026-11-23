// CMD: just TARGET=optim_advanced_ex run-objc
// Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton,
// AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Objective-C.
//
// Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
// EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
// ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
// implement the objective. From Objective-C, subclass the interface class and override its virtual
// methods; the optimizer then calls the overrides. Coordinates are updated in place through
// an NSArray ** parameter (the C++ parameter is a non-const std::vector<double>&).
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

/// A double through a C printf format, e.g. cfmt("%.6f", x) for C#'s {x:F6}.
static NSString *cfmt(const char *format, double x) {
    char buf[64];
    snprintf(buf, sizeof buf, format, x);
    return @(buf);
}

/// C#'s {x:G<digits>}: %g with an uppercase exponent.
static NSString *fmtG(double x, int digits) {
    char buf[64];
    snprintf(buf, sizeof buf, "%.*g", digits, x);
    for (char *c = buf; *c; c++)
        if (*c == 'e') *c = 'E';
    return @(buf);
}

@interface Quadratic : DMHessianFunction
@end
@implementation Quadratic
- (double)evaluate:(NSArray<NSNumber *> *)coordinates {
    double x = coordinates[0].doubleValue, y = coordinates[1].doubleValue;
    return pow(x - 1, 2) + 10 * pow(y + 2, 2);
}
- (NSArray<NSNumber *> *)gradient:(NSArray<NSNumber *> *)coordinates {
    double x = coordinates[0].doubleValue, y = coordinates[1].doubleValue;
    return @[@(2 * (x - 1)), @(20 * (y + 2))];
}
- (NSArray<NSArray<NSNumber *> *> *)hessian:(NSArray<NSNumber *> *)coordinates {
    return @[@[@2.0, @0.0], @[@0.0, @20.0]];
}
@end

@interface Exponential : DMResidualFunction
@property (nonatomic, copy) NSArray<NSNumber *> *ts;
@end
@implementation Exponential
- (NSArray<NSNumber *> *)residuals:(NSArray<NSNumber *> *)p {
    double a = p[0].doubleValue, b = p[1].doubleValue;
    NSMutableArray<NSNumber *> *out = [NSMutableArray array];
    for (NSNumber *t in self.ts) [out addObject:@(a * exp(b * t.doubleValue) - 2 * exp(0.5 * t.doubleValue))];
    return out;
}
- (NSArray<NSArray<NSNumber *> *> *)jacobian:(NSArray<NSNumber *> *)p {
    double a = p[0].doubleValue, b = p[1].doubleValue;
    NSMutableArray<NSArray<NSNumber *> *> *out = [NSMutableArray array];
    for (NSNumber *n in self.ts) {
        double t = n.doubleValue;
        [out addObject:@[@(exp(b * t)), @(a * t * exp(b * t))]];
    }
    return out;
}
@end

@interface Incomplete : DMHessianFunction
@end
@implementation Incomplete
- (double)evaluate:(NSArray<NSNumber *> *)coordinates { return 0.0; }
@end

int main(void) {
    @autoreleasepool {
        @try {
            println(@"=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ===================");
            // Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
            // then check the acquisition function favors an unexplored region.
            {
                NSArray<NSNumber *> *xs = @[@0.0, @0.5, @1.0, @1.5, @2.0, @3.0];
                NSMutableArray<NSNumber *> *ys = [NSMutableArray array];
                NSMutableArray<NSArray<NSNumber *> *> *points = [NSMutableArray array];
                double minY = INFINITY;
                for (NSNumber *x in xs) {
                    double y = (x.doubleValue - 1) * (x.doubleValue - 1);
                    [ys addObject:@(y)];
                    [points addObject:@[x]];
                    minY = fmin(minY, y);
                }
                DMRbfGaussianProcessSurrogate *surrogate = [[DMRbfGaussianProcessSurrogate alloc] initWithLengthScale:1.0 noise:1e-6];
                [surrogate fit:points values:ys];
                double incumbent = fmin(0.0, minY);  // best observed value so far
                println(@"incumbent (best observed y): %@", fmt(incumbent));
                for (NSNumber *x in @[@0.25, @1.0, @2.5])
                    println(@"acquisition(x=%@) = %@", cfmt("%.2f", x.doubleValue),
                            cfmt("%.6f", [surrogate acquisition:@[x] incumbent:incumbent]));
            }

            println(@"\n=================== Options structs + optimizer construction (all 10 new classes) ===================");
            // Every new Options struct and optimizer class constructs cleanly, proving
            // the binding for the whole batch is wired correctly.
            DMProximalGradientOptions *pgOptions = [[DMProximalGradientOptions alloc] init];
            pgOptions.stepSize = 0.05;
            DMProximalGradient *pg = [[DMProximalGradient alloc] initWithOptions:pgOptions];
            println(@"ProximalGradient: %@", pg);
            DMFista *fista = [[DMFista alloc] initWithOptions:[[DMFistaOptions alloc] init]];
            println(@"FISTA: %@", fista);
            DMLevenbergMarquardtOptions *lmOptions = [[DMLevenbergMarquardtOptions alloc] init];
            lmOptions.maxIterations = 200;
            DMLevenbergMarquardt *lm = [[DMLevenbergMarquardt alloc] initWithOptions:lmOptions];
            println(@"LevenbergMarquardt: %@", lm);
            DMNewton *newton = [[DMNewton alloc] initWithOptions:[[DMNewtonOptions alloc] init]];
            println(@"Newton: %@", newton);
            DMTrustRegionNewton *trn = [[DMTrustRegionNewton alloc] initWithOptions:[[DMTrustRegionNewtonOptions alloc] init]];
            println(@"TrustRegionNewton: %@", trn);
            DMAugmentedLagrangian *al = [[DMAugmentedLagrangian alloc] initWithOptions:[[DMAugmentedLagrangianOptions alloc] init]];
            println(@"AugmentedLagrangian: %@", al);
            DMSqp *sqp = [[DMSqp alloc] initWithOptions:[[DMSqpOptions alloc] init]];
            println(@"SQP: %@", sqp);
            DMInteriorPoint *ip = [[DMInteriorPoint alloc] initWithOptions:[[DMInteriorPointOptions alloc] init]];
            println(@"InteriorPoint: %@", ip);
            DMBayesianOptimizationOptions *boOptions = [[DMBayesianOptimizationOptions alloc] init];
            boOptions.initialSamples = 8;
            DMBayesianOptimization *bo = [[DMBayesianOptimization alloc] initWithOptions:boOptions];
            println(@"BayesianOptimization: %@", bo);

            println(@"\n=================== Objective-C objectives (subclassing the C++ interfaces) ===================");
            // Newton's method on f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
            {
                NSArray<NSNumber *> *start = @[@5.0, @5.0];
                double value = [newton optimize:[[Quadratic alloc] init] coordinates:&start];
                println(@"Newton: f = %@ at (%@, %@) (exact minimum 0 at (1, -2))", fmtG(value, 6),
                        cfmt("%.6f", start[0].doubleValue), cfmt("%.6f", start[1].doubleValue));
            }

            // Levenberg-Marquardt fitting y = a * exp(b t) to exact data from a = 2, b = 0.5.
            {
                NSArray<NSNumber *> *parameters = @[@1.0, @0.1];
                Exponential *exponential = [[Exponential alloc] init];
                exponential.ts = @[@0.0, @0.5, @1.0, @1.5, @2.0];
                double sse = [lm optimize:exponential coordinates:&parameters];
                println(@"LevenbergMarquardt: residual = %@, a = %@, b = %@ (exact: 2, 0.5)", fmtG(sse, 3),
                        cfmt("%.6f", parameters[0].doubleValue), cfmt("%.6f", parameters[1].doubleValue));
            }

            // An interface method a subclass leaves out raises when the optimizer calls it.
            {
                NSException *err = nil;
                @try {
                    NSArray<NSNumber *> *origin = @[@0.0, @0.0];
                    [newton optimize:[[Incomplete alloc] init] coordinates:&origin];
                } @catch (NSException *e) {
                    err = e;
                }
                println(@"missing gradient(): ok=%s, error mentions it: %s", err == nil ? "True" : "False",
                        err != nil && [err.reason rangeOfString:@"gradient"].location != NSNotFound ? "True" : "False");
            }
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
