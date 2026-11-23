// CMD: just TARGET=pca_ex run-objc
// PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
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

int main(void) {
    @autoreleasepool {
        @try {
            DMDataFrame *iris = [DMData iris];
            NSArray<NSString *> *featureColumns = @[@"Sepal.Length", @"Sepal.Width", @"Petal.Length", @"Petal.Width"];

            // species labels, for the grouped (colored-by-species) scores plot
            NSMutableArray<NSString *> *species = [NSMutableArray array];
            for (size_t i = 0; i < iris.nrows; i++) [species addObject:[iris stringAt:@"Species" rowIndex:i]];

            println(@"=================== PCA on iris (scaled) ===================");
            DMPca *pca = [[DMPca alloc] initWithData:iris featureColumns:featureColumns center:YES scale:YES];
            [pca printSummary];

            println(@"\nPC1 loadings (which original features drive it):");
            NSArray<NSString *> *names = [pca featureNames];
            NSArray<NSNumber *> *loadings = [pca componentLoadings:0];
            for (NSUInteger i = 0; i < MIN(names.count, loadings.count); i++)
                println(@"  %@: %@", names[i], cfmt("%.4f", loadings[i].doubleValue));

            println(@"\nTraining scores as a DataFrame:\n%@", [[pca scoresFrame] toString]);

            [[pca plotScoresGrouped:species] saveSvg:@"pca_iris_scores.svg"];
            println(@"\nScores scatter (colored by species) saved as pca_iris_scores.svg");

            [[pca plotScree] saveSvg:@"pca_iris_scree.svg"];
            println(@"Scree plot saved as pca_iris_scree.svg");

            println(@"\n=================== PCA on iris (unscaled) ===================");
            DMPca *unscaled = [[DMPca alloc] initWithData:iris featureColumns:featureColumns center:YES scale:NO];
            println(@"PC1 explains %@%% of variance (vs. %@%% scaled) -- Sepal.Length's larger raw variance dominates the unscaled covariance matrix.",
                    cfmt("%.2f", 100 * [unscaled explainedVarianceRatio][0].doubleValue),
                    cfmt("%.2f", 100 * [pca explainedVarianceRatio][0].doubleValue));
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
