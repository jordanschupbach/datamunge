// CMD: just TARGET=elastic_net_iris_ex run-objc
// Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
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
            NSString *formula = @"Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width";

            DMDataFrame *iris = [DMData iris];
            println(@"iris: %zu rows x %zu cols", iris.nrows, iris.ncols);
            println(@"formula: %@\n", formula);

            println(@"=================== Ridge ===================");
            DMRidge *ridge = [[DMRidge alloc] initWithData:iris formula:formula];
            [ridge printSummary];

            println(@"\n=================== Lasso ===================");
            DMLasso *lasso = [[DMLasso alloc] initWithData:iris formula:formula];
            [lasso printSummary];

            println(@"\n================= Elastic Net =================");
            // initWithData:formula:alpha:lambda:nLambda:cvFolds:standardize:seed: (alpha = 0.5)
            DMElasticNet *elastic = [[DMElasticNet alloc] initWithData:iris formula:formula alpha:0.5];
            [elastic printSummary];

            println(@"\nSaved figures showing how each model's coefficients respond to the regularization strength, "
                    @"and the cross-validation curve used to pick it:");

            NSArray<NSArray *> *figures = @[
                @[[ridge plotCoefficientPath], [ridge plotCvCurve], @"ridge", @"ridge:      "],
                @[[lasso plotCoefficientPath], [lasso plotCvCurve], @"lasso", @"lasso:      "],
                @[[elastic plotCoefficientPath], [elastic plotCvCurve], @"elasticnet", @"elastic net:"],
            ];
            for (NSArray *figure in figures) {
                DMRPlot *path = figure[0], *cv = figure[1];
                NSString *name = figure[2], *label = figure[3];
                [path save:[NSString stringWithFormat:@"elastic_net_%@_path.svg", name]];
                [cv save:[NSString stringWithFormat:@"elastic_net_%@_cv.svg", name]];
                println(@"  %@ elastic_net_%@_path.svg, elastic_net_%@_cv.svg", label, name, name);
            }
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
