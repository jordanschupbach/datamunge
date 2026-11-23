// CMD: just TARGET=tensor_ex run-objc
// Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
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

static NSString *shapeString(DMTensor *tensor) {
    return [NSString stringWithFormat:@"[%@]", [[tensor shape] componentsJoinedByString:@", "]];
}

int main(void) {
    @autoreleasepool {
        @try {
            println(@"=================== Construction ===================");
            println(@"zeros([2,3]): %@", [[DMTensor zeros:@[@2, @3]] toString]);
            println(@"eye(3): %@", [[DMTensor eye:3] toString]);

            DMTensor *r = [[DMTensor arange:0.0 stop:12.0 step:1.0] reshape:@[@3, @4]];
            println(@"arange(0,12).reshape([3,4]): %@", [r toString]);

            println(@"\n=================== Shape ops ===================");
            println(@"transpose -> shape %@", shapeString([r transpose]));
            println(@"slice(axis=1, start=1, stop=3): %@", [[r slice:1 start:1 stop:3] toString]);

            println(@"\n=================== Broadcasting arithmetic ===================");
            {
                DMTensor *col = [DMTensor fromValues:@[@3, @1] values:@[@1.0, @2.0, @3.0]];
                DMTensor *row = [DMTensor fromValues:@[@1, @4] values:@[@10.0, @20.0, @30.0, @40.0]];
                println(@"(3,1) + (1,4) -> %@", [[col add:row] toString]);
            }

            println(@"\n=================== Reductions ===================");
            println(@"r.sum() = %@, r.mean() = %@", fmt([r sum]), fmt([r mean]));
            println(@"column means (axis=0): %@", [[r meanAxis:0] toString]);

            println(@"\n=================== Linear algebra ===================");
            {
                DMTensor *a = [DMTensor fromValues:@[@2, @3] values:@[@1.0, @2.0, @3.0, @4.0, @5.0, @6.0]];
                DMTensor *b = [DMTensor fromValues:@[@3, @2] values:@[@7.0, @8.0, @9.0, @10.0, @11.0, @12.0]];
                println(@"matmul(2x3, 3x2) -> %@", [[a matmul:b] toString]);
            }
            {
                DMTensor *v1 = [DMTensor fromValues:@[@3] values:@[@1.0, @2.0, @3.0]];
                DMTensor *v2 = [DMTensor fromValues:@[@3] values:@[@4.0, @5.0, @6.0]];
                println(@"dot([1,2,3], [4,5,6]) = %@", fmt([v1 dot:v2]));
                println(@"outer(v1, v2) -> %@", [[v1 outer:v2] toString]);
            }

            println(@"\n=================== Comparisons & masks ===================");
            {
                DMTensor *mask = [r greaterEqual:[DMTensor full:@[@3, @4] value:6.0]];
                println(@"r >= 6 -> %@", [mask toString]);
                println(@"count(r >= 6) = %@", fmt([mask sum]));
            }

            println(@"\n=================== A real dataset as a Tensor ===================");
            {
                DMDataFrame *iris = [DMData iris];
                size_t n = [iris nrows];
                NSArray<NSString *> *columns = @[@"Sepal.Length", @"Sepal.Width", @"Petal.Length", @"Petal.Width"];
                NSMutableArray<NSNumber *> *flat = [NSMutableArray array];
                for (size_t i = 0; i < n; i++)
                    for (NSString *column in columns) [flat addObject:@([iris numericAt:column rowIndex:i])];
                DMTensor *x = [DMTensor fromValues:@[@(n), @4] values:flat];
                println(@"iris feature tensor shape: %@", shapeString(x));

                DMTensor *featureMeans = [x meanAxis:0];
                DMTensor *centered = [x subtract:[featureMeans reshape:@[@1, @4]]];
                DMTensor *scatter = [[centered transpose] matmul:centered];
                println(@"feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): %@", [featureMeans toString]);
                println(@"(X-mean)^T (X-mean) [4x4 scatter matrix]: %@", [scatter toString:16]);

                NSMutableArray<NSString *> *species = [NSMutableArray array];
                for (size_t i = 0; i < n; i++) [species addObject:[iris stringAt:@"Species" rowIndex:i]];
                DMTensor *speciesTensor = [DMTensor fromStringValues:@[@(n)] values:species];
                DMTensor *setosaMask = [speciesTensor equal:[DMTensor fromStringValues:@[@1] values:@[@"setosa"]]];
                println(@"setosa count = %@ (of %zu rows)", fmt([setosaMask sum]), n);
            }
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
