// CMD: just TARGET=ode_ex run-objc
// ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
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

// Approximate C's %g: six significant digits, no padding (C#'s G6).
static NSString *fmtG6(NSNumber *x) { return fmtG(x.doubleValue, 6); }

static NSArray<NSNumber *> *lastState(DMOdeSolution *sol) { return [sol stateAt:[sol size] - 1]; }

// The time vector, and the requested state components (0-based), one value per step.
static NSArray<NSNumber *> *solutionTimes(DMOdeSolution *sol) {
    NSMutableArray<NSNumber *> *times = [NSMutableArray array];
    for (size_t i = 0; i < [sol size]; i++) [times addObject:@([sol timeAt:i])];
    return times;
}

static NSArray<NSArray<NSNumber *> *> *solutionComponents(DMOdeSolution *sol, NSArray<NSNumber *> *components) {
    NSMutableArray<NSMutableArray<NSNumber *> *> *series = [NSMutableArray array];
    for (NSUInteger c = 0; c < components.count; c++) [series addObject:[NSMutableArray array]];
    for (size_t i = 0; i < [sol size]; i++) {
        NSArray<NSNumber *> *state = [sol stateAt:i];
        for (NSUInteger c = 0; c < components.count; c++) [series[c] addObject:state[components[c].unsignedIntegerValue]];
    }
    return series;
}

static DMOdeSolver *rk4Solver(double step) {
    DMOdeOptions *options = [[DMOdeOptions alloc] init];
    options.method = DMStepMethodRk4;
    options.stepSize = step;
    return [[DMOdeSolver alloc] initWithOptions:options];
}

static NSString *joined(NSArray<NSNumber *> *values) {
    NSMutableArray<NSString *> *parts = [NSMutableArray array];
    for (NSNumber *v in values) [parts addObject:fmtG6(v)];
    return [parts componentsJoinedByString:@", "];
}

// NOTE: like the Lua version this example uses the built-in named systems via -solveBuiltin:....

int main(void) {
    @autoreleasepool {
        @try {
            println(@"=================== Every named built-in system (no director needed) ===================");
            {
                DMOdeSolver *solver = [[DMOdeSolver alloc] init];
                NSArray *systems = @[
                    @[@"exponential_decay", @[@1.0], @[@1.0]],
                    @[@"logistic_growth", @[@1.0, @1.0], @[@0.5]],
                    @[@"harmonic_oscillator", @[@1.0], @[@1.0, @0.0]],
                    @[@"van_der_pol", @[@1.0], @[@2.0, @0.0]],
                    @[@"lorenz", @[@10.0, @28.0, @(8.0 / 3)], @[@1.0, @1.0, @1.0]],
                ];
                for (NSArray *system in systems) {
                    NSString *name = system[0];
                    DMOdeSolution *sol = [solver solveBuiltin:name params:system[1] y0:system[2] tStart:0.0 tEnd:1.0];
                    println(@"%-22s steps=%-6zu final state=[%@]", name.UTF8String, sol.stepsTaken, joined(lastState(sol)));
                }
            }

            println(@"\n=================== Harmonic oscillator (energy conservation) ===================");
            {
                DMOdeSolution *sol = [rk4Solver(0.01) solveBuiltin:@"harmonic_oscillator" params:@[@1.0] y0:@[@1.0, @0.0]
                                                            tStart:0.0 tEnd:20.0];
                NSArray<NSNumber *> *last = lastState(sol);
                double x = last[0].doubleValue, v = last[1].doubleValue;
                println(@"x(20) = %@ (cos(20) = %@)", cfmt("%.6f", x), cfmt("%.6f", cos(20.0)));
                println(@"energy x^2+v^2 = %@ (should stay near 1.0)", cfmt("%.6f", x * x + v * v));
                NSArray<NSNumber *> *ts = solutionTimes(sol);
                NSArray<NSArray<NSNumber *> *> *series = solutionComponents(sol, @[@0, @1]);
                DMRPlot *p = [DMRPlot plot:ts y:series[0] type:@"l" label:@"x(t)"];
                [p lines:ts y:series[1] label:@"v(t)"];
                [[[p title:@"Harmonic Oscillator"] xLabel:@"t"] yLabel:@"state"];
                [p saveSvg:@"ode_harmonic_oscillator_objc.svg"];
                println(@"wrote ode_harmonic_oscillator_objc.svg");
            }

            println(@"\n=================== Lorenz attractor (phase plane) ===================");
            {
                DMOdeSolution *sol = [rk4Solver(0.005) solveBuiltin:@"lorenz" params:@[@10.0, @28.0, @(8.0 / 3)]
                                                                 y0:@[@1.0, @1.0, @1.0] tStart:0.0 tEnd:25.0];
                println(@"steps_taken = %zu", sol.stepsTaken);
                NSArray<NSArray<NSNumber *> *> *series = solutionComponents(sol, @[@0, @2]);
                DMRPlot *p = [DMRPlot plot:series[0] y:series[1] type:@"l" label:@"trajectory"];
                [[[p title:@"Lorenz Attractor (x-z phase plane)"] xLabel:@"x"] yLabel:@"z"];
                [p saveSvg:@"ode_lorenz_phase_plane_objc.svg"];
                println(@"wrote ode_lorenz_phase_plane_objc.svg");
            }
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
