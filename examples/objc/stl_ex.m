// CMD: just TARGET=stl_ex run-objc
// C++ std::vector and std::pair values in Objective-C: vectors are NSArray<NSNumber *>, pairs are NSArrays of two.
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
            const int n = 100;
            NSMutableArray<NSNumber *> *v = [NSMutableArray array];
            for (int i = 0; i < n; i++) [v addObject:@(i * 1.5)];
            for (NSNumber *x in v) println(@"%@", fmt(x.doubleValue));
            NSMutableArray<NSNumber *> *v2 = [NSMutableArray array];
            for (int i = 0; i < n; i++) [v2 addObject:@((int)floor(i * 1.5))];
            for (NSNumber *x in v2) println(@"%d", x.intValue);

            int p[2] = {3, 4};
            println(@"p: (%d, %d)", p[0], p[1]);

            int p2[2] = {10, 20};
            println(@"p2: (%d, %d)", p2[0], p2[1]);

            // A C++ std::pair<double,double> comes back as an NSArray of two.
            NSArray *p3 = [DM makeDpair:1.25 b:2.75];
            println(@"make_dpair(1.25, 2.75): (%@, %@), sum_dpair = %@", fmt([p3[0] doubleValue]),
                    fmt([p3[1] doubleValue]), fmt([DM sumDpair:p3]));
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
