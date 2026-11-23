// CMD: just TARGET=mds_ex run-objc
// Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
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
            NSMutableArray<NSString *> *species = [NSMutableArray array];
            for (size_t i = 0; i < iris.nrows; i++) [species addObject:[iris stringAt:@"Species" rowIndex:i]];

            println(@"=================== Classical MDS on iris (euclidean) ===================");
            DMMds *mds = [[DMMds alloc] initWithData:iris featureColumns:featureColumns nComponents:2 metric:@"euclidean"];
            [mds printSummary];

            println(@"\nEmbedding as a DataFrame:\n%@", [[mds embeddingFrame] toString]);

            [[mds plotEmbeddingGrouped:species] saveSvg:@"mds_iris_embedding.svg"];
            println(@"\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg");

            println(@"\n=================== Classical MDS on iris (manhattan) ===================");
            DMMds *manhattanMds = [[DMMds alloc] initWithData:iris featureColumns:featureColumns nComponents:2 metric:@"manhattan"];
            println(@"Goodness of fit: euclidean=%@, manhattan=%@", cfmt("%.4f", [mds goodnessOfFit]),
                    cfmt("%.4f", [manhattanMds goodnessOfFit]));
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
