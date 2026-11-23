// CMD: just TARGET=autodiff_ex run-objc
// Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
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

// Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
// (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
// bound. This port demonstrates the same ideas with the bound named-method API.
// -[DMVar index] is the Var's position in the adjoint vector -[DMTape backward:] returns.

static DMDual *rosenbrockDual(DMDual *x0, DMDual *x1) {
    DMDual *a = [[[DMDual alloc] initWithValue:1.0 derivative:0.0] subtract:x0];
    DMDual *b = [x1 subtract:[x0 multiply:x0]];
    return [[a multiply:a] add:[[b multiply:b] multiplyScalar:100.0]];
}

static double rosenbrockValue(double x0, double x1) { return pow(1.0 - x0, 2) + 100.0 * pow(x1 - x0 * x0, 2); }

// Build the Rosenbrock graph on tape; returns the output Var.
static DMVar *rosenbrockVar(DMTape *tape, DMVar *x0, DMVar *x1) {
    DMVar *a = [[[DMVar alloc] initWithTape:tape value:1.0] subtract:x0];
    DMVar *b = [x1 subtract:[x0 multiply:x0]];
    return [[a multiply:a] add:[[b multiply:b] multiplyScalar:100.0]];
}

static DMHyperDual *rosenbrockHyperDual(DMHyperDual *x0, DMHyperDual *x1) {
    DMHyperDual *a = [[[DMHyperDual alloc] initWithValue:1.0 eps1:0.0 eps2:0.0 eps1eps2:0.0] subtract:x0];
    DMHyperDual *b = [x1 subtract:[x0 multiply:x0]];
    return [[a multiply:a] add:[[b multiply:b] multiplyScalar:100.0]];
}

static double adjoint(NSArray<NSNumber *> *adjoints, DMVar *v) { return adjoints[[v index]].doubleValue; }

static DMDual *dual(double value, double derivative) {
    return [[DMDual alloc] initWithValue:value derivative:derivative];
}

int main(void) {
    @autoreleasepool {
        @try {
            println(@"=================== Forward mode: scalar derivative ===================");
            {
                DMDual *x0 = dual(2.0, 1.0);  // seed derivative = 1 to read df/dx directly
                DMDual *f = [[[x0 multiply:x0] multiply:x0] subtract:[x0 multiplyScalar:2.0]];  // x^3 - 2x
                println(@"f(x) = x^3 - 2x, f'(2) = %@ (exact: 10)", fmt([f derivative]));
            }

            println(@"\n=================== Reverse mode: build a graph by hand ===================");
            {
                DMTape *tape = [[DMTape alloc] init];
                DMVar *a = [[DMVar alloc] initWithTape:tape value:2.0];
                DMVar *b = [[DMVar alloc] initWithTape:tape value:3.0];
                DMVar *y = [[a multiply:b] add:[a sin]];
                println(@"y = a*b + sin(a) at a=2, b=3 -> y = %@", fmt([y value]));
                NSArray<NSNumber *> *adjoints = [tape backward:y];
                println(@"dy/da = %@ (exact: b + cos(a))", fmt(adjoint(adjoints, a)));
                println(@"dy/db = %@ (exact: a)", fmt(adjoint(adjoints, b)));
            }

            println(@"\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================");
            {
                double px = 0.0, py = 0.0;
                // Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
                double gx = [rosenbrockDual(dual(px, 1.0), dual(py, 0.0)) derivative];
                double gy = [rosenbrockDual(dual(px, 0.0), dual(py, 1.0)) derivative];
                // Reverse mode: one Tape/Var pass computes every partial at once.
                DMTape *tape = [[DMTape alloc] init];
                DMVar *vx = [[DMVar alloc] initWithTape:tape value:px];
                DMVar *vy = [[DMVar alloc] initWithTape:tape value:py];
                NSArray<NSNumber *> *grad = [tape backward:rosenbrockVar(tape, vx, vy)];
                println(@"f(0,0) = %@", fmt(rosenbrockValue(px, py)));
                println(@"gradient (forward mode): [%@, %@]", fmt(gx), fmt(gy));
                println(@"gradient (reverse mode): [%@, %@]", fmt(adjoint(grad, vx)), fmt(adjoint(grad, vy)));
            }

            println(@"\n=================== Jacobian of a vector-valued function ===================");
            // f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
            {
                double jac[3][2];
                double seeds[2][2] = {{1.0, 0.0}, {0.0, 1.0}};
                for (int col = 0; col < 2; col++) {
                    DMDual *x = dual(2.0, seeds[col][0]), *y = dual(3.0, seeds[col][1]);
                    NSArray<DMDual *> *outputs = @[[x multiply:x], [x multiply:y], [[y multiply:y] multiply:y]];
                    for (NSUInteger row = 0; row < outputs.count; row++)
                        jac[row][col] = [outputs[row] derivative];
                }
                println(@"f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:");
                for (int row = 0; row < 3; row++)
                    println(@"  [%@, %@]", fmt(jac[row][0]), fmt(jac[row][1]));
            }

            println(@"\n=================== Hessian via second-order forward mode (HyperDual) ===================");
            // One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
            {
                double mx = 1.0, my = 1.0;  // the Rosenbrock function's minimum
                double seeds[2][2] = {{1.0, 0.0}, {0.0, 1.0}};
                double h[2][2];
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 2; j++) {
                        double *e1 = seeds[i], *e2 = seeds[j];
                        h[i][j] = [rosenbrockHyperDual(
                            [[DMHyperDual alloc] initWithValue:mx eps1:e1[0] eps2:e2[0] eps1eps2:0.0],
                            [[DMHyperDual alloc] initWithValue:my eps1:e1[1] eps2:e2[1] eps1eps2:0.0]) eps1eps2];
                    }
                println(@"Hessian of the Rosenbrock function at its minimum (1,1):");
                for (int i = 0; i < 2; i++)
                    println(@"  [%@, %@]", fmt(h[i][0]), fmt(h[i][1]));
            }

            println(@"\n=================== Gradient descent driven by reverse-mode gradients ===================");
            {
                double point[2] = {-1.2, 1.0};  // the classic Rosenbrock start
                const double learningRate = 0.001;
                const int nSteps = 2000;
                for (int step = 0; step < nSteps; step++) {
                    @autoreleasepool {
                        DMTape *tape = [[DMTape alloc] init];
                        DMVar *vx = [[DMVar alloc] initWithTape:tape value:point[0]];
                        DMVar *vy = [[DMVar alloc] initWithTape:tape value:point[1]];
                        DMVar *vf = rosenbrockVar(tape, vx, vy);
                        NSArray<NSNumber *> *grad = [tape backward:vf];
                        double loss = [vf value];
                        point[0] -= learningRate * adjoint(grad, vx);
                        point[1] -= learningRate * adjoint(grad, vy);
                        if (step == 0 || step == nSteps - 1)
                            println(@"step %d: loss = %@, x = [%@, %@]", step, fmt(loss), fmt(point[0]), fmt(point[1]));
                    }
                }
                println(@"(true minimum is at [1, 1] with loss 0)");
            }
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
