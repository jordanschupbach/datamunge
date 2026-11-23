// CMD: just TARGET=dataframe_ex run-objc
// DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
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
            DMDataFrame *sales = [[DMDataFrame alloc] init];
            [sales addStringColumn:@"region" values:@[@"west", @"west", @"east", @"south", @"south", @"south"]];
            [sales addStringColumn:@"product" values:@[@"widget", @"widget", @"widget", @"gizmo", @"gizmo", @"gizmo"]];
            [sales addNumericColumn:@"sales" values:@[@10.0, @10.0, @14.0, @8.0, @0.0, @11.0] validMask:@[@1, @1, @1, @1, @0, @1]];
            [sales addStringColumn:@"quarter" values:@[@"Q1", @"Q1", @"Q1", @"Q2", @"Q2", @""] validMask:@[@1, @1, @1, @1, @1, @0]];

            println(@"raw data\n%@\n", [sales toString]);

            DMDataFrame *cleaned = [sales dropDuplicates:@[@"region", @"product", @"sales", @"quarter"]];
            [cleaned fillNullString:@"quarter" value:@"unknown"];
            [cleaned fillNullNumeric:@"sales" value:0.0];
            println(@"after drop_duplicates + fill_null\n%@\n", [cleaned toString]);

            DMDataFrame *selected = [[cleaned select:@[@"region", @"sales", @"quarter"]] sortBy:@"sales" ascending:NO];
            println(@"selected + sorted\n%@\n", [selected toString]);

            DMDataFrame *grouped = [[cleaned groupBySum:@[@"region"] valueColumns:@[@"sales"]] sortBy:@"sales" ascending:NO];
            println(@"group_by_sum(region)\n%@\n", [grouped toString]);

            DMDataFrame *targets = [[DMDataFrame alloc] init];
            [targets addStringColumn:@"region" values:@[@"west", @"east", @"south"]];
            [targets addNumericColumn:@"target" values:@[@18.0, @12.0, @25.0]];
            println(@"joined with targets\n%@\n", [[grouped join:targets leftKey:@"region" rightKey:@"region" joinType:@"left"] toString]);

            NSArray<NSNumber *> *shape = [cleaned shape];
            println(@"shape = (%@, %@)", shape[0], shape[1]);
            println(@"sales count = %zu", [cleaned numericCount:@"sales"]);
            println(@"sales nulls = %zu", [cleaned numericNullCount:@"sales"]);
            println(@"sales sum = %@", fmt([cleaned numericSum:@"sales"]));
            println(@"sales mean = %@", fmt([cleaned numericMean:@"sales"]));
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
