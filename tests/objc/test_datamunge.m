// Smoke tests of the Rosetta-generated Objective-C binding (just test-objc).
#import "DM.h"
#include <stdio.h>
#include <stdlib.h>

static int checks = 0;

static void check(BOOL ok, const char *message) {
    checks++;
    if (!ok) {
        printf("check failed: %s\n", message);
        exit(1);
    }
}

// An Objective-C subclass of DMCallback overriding its virtual -call: (a director).
@interface Doubler : DMCallback
@end
@implementation Doubler
- (double)call:(double)x { return 2 * x; }
@end

@interface Failing : DMCallback
@end
@implementation Failing
- (double)call:(double)x {
    @throw [NSException exceptionWithName:@"FromObjC" reason:@"from Objective-C" userInfo:nil];
}
@end

static double sum(NSArray<NSNumber *> *values) { return [DM sumDvector:values]; }

int main(void) {
    @autoreleasepool {
        // Smoke test: call into the native library.
        [DM hello];

        // std::vector<double> and std::pair<double,double> round trips.
        NSArray<NSNumber *> *v = [DM makeDvector:1 b:2 c:3];
        check(v.count == 3 && v[0].doubleValue == 1 && v[2].doubleValue == 3, "std::vector<double> is an NSArray");
        check(sum(v) == 6, "sumDvector");
        check(sum(@[@0.5, @1.5]) == 2, "array literals convert to std::vector");
        NSArray *p = [DM makeDpair:1.25 b:2.75];
        check(p.count == 2 && [p[0] doubleValue] == 1.25 && [p[1] doubleValue] == 2.75, "std::pair is an NSArray of two");
        check([DM sumDpair:p] == 4, "sumDpair");

        // The C++ Callback class, and a subclass overriding its virtual method.
        DMCallback *native = [[DMCallback alloc] init];
        check([DM callWithCallback:3 cb:native] == 3, "native Callback");
        check(sum([DM mapDvectorWithCallback:@[@1, @2, @3] cb:native]) == 6, "native map");
        Doubler *doubler = [[Doubler alloc] init];
        check([DM callWithCallback:3 cb:doubler] == 6, "the Objective-C override runs");
        check(sum([DM mapDvectorWithCallback:@[@1, @2, @3] cb:doubler]) == 12, "override from C++ loops");

        // An Objective-C exception inside an override is raised again after C++ returns.
        BOOL raised = NO;
        @try {
            [DM callWithCallback:1 cb:[[Failing alloc] init]];
        } @catch (NSException *e) {
            raised = [e.name isEqualToString:@"FromObjC"];
        }
        check(raised, "the Objective-C exception propagates");

        // C++ exceptions raise DMError with their kind.
        raised = NO;
        @try {
            [DM runif:1 upper:0 n:3];
        } @catch (DMError *e) {
            raised = e.kind == DMErrorKindInvalidArgument;
        }
        check(raised, "C++ exceptions raise DMError");

        // Classes, methods and the policy namespace DMData.
        DMDataFrame *frame = [[DMDataFrame alloc] init];
        [frame addNumericColumn:@"x" values:@[@1, @2, @3]];
        check(frame.nrows == 3, "DMDataFrame");
        NSArray<NSNumber *> *seq = [DM seql:0 to:1 n:5];
        check(seq.count == 5 && seq[1].doubleValue == 0.25 && seq[4].doubleValue == 1, "seql");
        check([DMData iris].nrows == 150, "the DMData namespace");

        printf("test_datamunge.m: all %d checks passed\n", checks);
    }
    return 0;
}
