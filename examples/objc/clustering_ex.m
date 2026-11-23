// CMD: just TARGET=clustering_ex run-objc
// Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
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

static NSArray<NSString *> *species;

// For each cluster, print its size and its majority species (negative labels are noise).
static void reportPurity(NSArray<NSNumber *> *clusterLabels, int nClusters) {
    NSMutableArray<NSMutableDictionary<NSString *, NSNumber *> *> *votes = [NSMutableArray array];
    NSMutableArray<NSMutableArray<NSString *> *> *order = [NSMutableArray array];  // first-seen order, as C#'s Dictionary
    for (int c = 0; c < nClusters; c++) {
        [votes addObject:[NSMutableDictionary dictionary]];
        [order addObject:[NSMutableArray array]];
    }
    for (NSUInteger i = 0; i < clusterLabels.count && i < species.count; i++) {
        long label = clusterLabels[i].longValue;
        if (label < 0) continue;
        NSString *sp = species[i];
        if (!votes[label][sp]) [order[label] addObject:sp];
        votes[label][sp] = @(votes[label][sp].intValue + 1);
    }
    for (int c = 0; c < nClusters; c++) {
        int total = 0, best = -1;
        NSString *bestSpecies = nil;
        for (NSString *sp in order[c]) {
            int count = votes[c][sp].intValue;
            total += count;
            if (count > best) { best = count; bestSpecies = sp; }
        }
        if (total == 0) continue;
        println(@"  cluster %d: %d points, majority %@ (%d/%d)", c, total, bestSpecies, best, total);
    }
}

int main(void) {
    @autoreleasepool {
        @try {
            DMDataFrame *iris = [DMData iris];
            NSMutableArray<NSString *> *names = [NSMutableArray array];
            for (size_t i = 0; i < [iris nrows]; i++) [names addObject:[iris stringAt:@"Species" rowIndex:i]];
            species = names;
            NSArray<NSString *> *featureColumns = @[@"Sepal.Length", @"Sepal.Width", @"Petal.Length", @"Petal.Width"];

            println(@"=================== K-means (k=3) on iris ===================");
            DMKMeans *kmeans = [[DMKMeans alloc] initWithData:iris featureColumns:featureColumns nClusters:3];
            println(@"Inertia: %@, iterations (best run): %zu", fmt([kmeans inertia]), [kmeans iterationsUsed]);
            reportPurity([kmeans labels], 3);

            println(@"\n=================== Agglomerative clustering on iris ===================");
            for (NSString *linkage in @[@"ward", @"complete", @"average"]) {
                DMAgglomerativeClustering *model = [[DMAgglomerativeClustering alloc] initWithData:iris featureColumns:featureColumns
                                                                                         nClusters:3 linkage:linkage];
                println(@"--- linkage=%@ ---", linkage);
                reportPurity([model labels], 3);
            }

            println(@"\n--- Re-cutting the ward dendrogram at k=2 without refitting ---");
            DMAgglomerativeClustering *wardModel = [[DMAgglomerativeClustering alloc] initWithData:iris featureColumns:featureColumns
                                                                                         nClusters:3 linkage:@"ward"];
            reportPurity([wardModel cut:2], 2);

            println(@"\n=================== DBSCAN on iris ===================");
            DMDbscan *dbscan = [[DMDbscan alloc] initWithData:iris featureColumns:featureColumns eps:0.6 minSamples:5];
            println(@"Clusters found: %zu, noise points: %zu (of %zu)", [dbscan nClusters], [dbscan nNoise], [dbscan observations]);
            reportPurity([dbscan labels], (int)[dbscan nClusters]);
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
