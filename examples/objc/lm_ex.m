// CMD: just TARGET=lm_ex run-objc
// Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions.
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

int main(void) {
    @autoreleasepool {
        @try {
            DMDataFrame *cars = [[DMDataFrame alloc] init];
            [cars addNumericColumn:@"hp" values:@[@110.0, @110.0, @93.0, @110.0, @175.0, @105.0, @245.0, @62.0, @95.0, @123.0]];
            [cars addNumericColumn:@"wt" values:@[@2.62, @2.875, @2.32, @3.215, @3.44, @3.46, @3.57, @3.19, @3.15, @3.44]];
            [cars addStringColumn:@"transmission" values:@[@"manual", @"manual", @"manual", @"automatic", @"automatic",
                                                           @"automatic", @"automatic", @"automatic", @"automatic", @"automatic"]];
            [cars addNumericColumn:@"mpg" values:@[@21.0, @21.0, @22.8, @21.4, @18.7, @18.1, @14.3, @24.4, @22.8, @19.2]];

            println(@"Fitting: mpg ~ hp + wt + transmission\n");
            DMLm *model = [[DMLm alloc] initWithData:cars formula:@"mpg ~ hp + wt + transmission"];
            [model printSummary];

            println(@"\nSequential ANOVA:\n%@", [[model anova] toString]);

            DMDataFrame *newcars = [[DMDataFrame alloc] init];
            [newcars addNumericColumn:@"hp" values:@[@150.0, @90.0]];
            [newcars addNumericColumn:@"wt" values:@[@3.0, @2.5]];
            [newcars addStringColumn:@"transmission" values:@[@"manual", @"automatic"]];
            println(@"\nPredictions with 95%% confidence intervals:\n%@",
                    [[model predictFrame:newcars intervalKind:@"confidence"] toString]);

            [model saveDiagnosticPlots:@"lm_ex_diagnostics"];
            println(@"\nSaved diagnostic plots as lm_ex_diagnostics_*.svg");
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
