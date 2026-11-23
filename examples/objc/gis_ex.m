// CMD: just TARGET=gis_ex run-objc
// ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
// Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
// writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
// layout +[DMShapeLayer read:] expects, then reads them back through the public API.
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

typedef struct {
    double x, y;
} Point;

/// A ring of points packed in NSData; a shape is an NSArray of rings.
static NSData *ring(const Point *points, size_t n) { return [NSData dataWithBytes:points length:n * sizeof(Point)]; }
static size_t ringLength(NSData *r) { return r.length / sizeof(Point); }
static const Point *ringPoints(NSData *r) { return (const Point *)r.bytes; }

// Little- and big-endian packers that append to a byte buffer.
static void packI4Le(NSMutableData *buf, int32_t value) {
    uint32_t v = (uint32_t)value;
    uint8_t b[4] = {v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff};
    [buf appendBytes:b length:4];
}
static void packI4Be(NSMutableData *buf, int32_t value) {
    uint32_t v = (uint32_t)value;
    uint8_t b[4] = {(v >> 24) & 0xff, (v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff};
    [buf appendBytes:b length:4];
}
static void packI2Le(NSMutableData *buf, int16_t value) {
    uint16_t v = (uint16_t)value;
    uint8_t b[2] = {v & 0xff, (v >> 8) & 0xff};
    [buf appendBytes:b length:2];
}
static void packDoubleLe(NSMutableData *buf, double x) {
    uint64_t v;
    memcpy(&v, &x, 8);
    uint8_t b[8];
    for (int i = 0; i < 8; i++) b[i] = (v >> (8 * i)) & 0xff;
    [buf appendBytes:b length:8];
}
static void packAscii(NSMutableData *buf, NSString *s) { [buf appendData:[s dataUsingEncoding:NSASCIIStringEncoding]]; }
static void packZeros(NSMutableData *buf, NSUInteger n) { [buf increaseLengthBy:n]; }
static void packByte(NSMutableData *buf, uint8_t b) { [buf appendBytes:&b length:1]; }

// Bounding box (minx, miny, maxx, maxy) of every point in rings.
static void ringBounds(NSArray<NSData *> *rings, double out[4]) {
    out[0] = out[1] = INFINITY;
    out[2] = out[3] = -INFINITY;
    for (NSData *r in rings)
        for (size_t i = 0; i < ringLength(r); i++) {
            Point p = ringPoints(r)[i];
            out[0] = p.x < out[0] ? p.x : out[0];
            out[1] = p.y < out[1] ? p.y : out[1];
            out[2] = p.x > out[2] ? p.x : out[2];
            out[3] = p.y > out[3] ? p.y : out[3];
        }
}

// Bytes of one Polygon record's contents (without the record header).
static NSData *polygonRecord(NSArray<NSData *> *rings) {
    NSMutableData *output = [NSMutableData data];
    packI4Le(output, 5);  // shape type: Polygon
    double bounds[4];
    ringBounds(rings, bounds);
    for (int i = 0; i < 4; i++) packDoubleLe(output, bounds[i]);
    packI4Le(output, (int32_t)rings.count);
    int32_t total = 0;
    for (NSData *r in rings) total += (int32_t)ringLength(r);
    packI4Le(output, total);
    int32_t start = 0;
    for (NSData *r in rings) {
        packI4Le(output, start);
        start += (int32_t)ringLength(r);
    }
    for (NSData *r in rings)
        for (size_t i = 0; i < ringLength(r); i++) {
            packDoubleLe(output, ringPoints(r)[i].x);
            packDoubleLe(output, ringPoints(r)[i].y);
        }
    return output;
}

static void writeCountiesShp(NSString *path, NSArray<NSArray<NSData *> *> *shapes) {
    NSMutableArray<NSData *> *contents = [NSMutableArray array];
    NSMutableArray<NSData *> *allRings = [NSMutableArray array];
    int32_t totalWords = 0;
    for (NSArray<NSData *> *shape in shapes) {
        NSData *c = polygonRecord(shape);
        [contents addObject:c];
        [allRings addObjectsFromArray:shape];
        totalWords += 4 + (int32_t)c.length / 2;
    }
    NSMutableData *f = [NSMutableData data];
    packI4Be(f, 9994);
    for (int i = 0; i < 5; i++) packI4Be(f, 0);
    packI4Be(f, 50 + totalWords);
    packI4Le(f, 1000);
    packI4Le(f, 5);  // Polygon
    double bounds[4];
    ringBounds(allRings, bounds);
    for (int i = 0; i < 4; i++) packDoubleLe(f, bounds[i]);
    for (int i = 0; i < 4; i++) packDoubleLe(f, 0.0);
    for (NSUInteger i = 0; i < contents.count; i++) {
        packI4Be(f, (int32_t)i + 1);
        packI4Be(f, (int32_t)contents[i].length / 2);
        [f appendData:contents[i]];
    }
    [f writeToFile:path atomically:NO];
}

static NSString *ljust(NSString *s, NSUInteger n, NSString *ch) {
    return s.length >= n ? [s substringToIndex:n] : [s stringByPaddingToLength:n withString:ch startingAtIndex:0];
}

static void writeField(NSMutableData *f, NSString *name, NSString *fieldType, uint8_t length) {
    packAscii(f, ljust(name, 11, @"\0"));
    packAscii(f, fieldType);
    packZeros(f, 4);
    packByte(f, length);
    packZeros(f, 15);
}

static void writeCountiesDbf(NSString *path, NSArray<NSString *> *names, NSArray<NSString *> *populations) {
    const int headerSize = 32 + 2 * 32 + 1;
    const int recordSize = 1 + 12 + 8;
    NSMutableData *f = [NSMutableData dataWithBytes:(uint8_t[]){0x03, 0, 0, 0} length:4];
    packI4Le(f, (int32_t)names.count);
    packI2Le(f, headerSize);
    packI2Le(f, recordSize);
    packZeros(f, 20);
    writeField(f, @"NAME", @"C", 12);
    writeField(f, @"POP", @"N", 8);
    packByte(f, 0x0D);
    for (NSUInteger i = 0; i < names.count; i++) {
        packAscii(f, @" ");
        packAscii(f, ljust(names[i], 12, @" "));
        packAscii(f, ljust(populations[i], 8, @" "));
    }
    [f writeToFile:path atomically:NO];
}

// Roughly C's %g: integral doubles print without a fractional part.
static NSString *fmtIntegral(double x) {
    return x == floor(x) ? [NSString stringWithFormat:@"%ld", (long)x] : cfmt("%.6f", x);
}

int main(void) {
    @autoreleasepool {
        @try {
            NSString *basePath = @"datamunge_gis_ex_counties_objc";

            // Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
            NSArray<NSData *> *plain = @[ring((Point[]){{10, 0}, {10, 10}, {20, 10}, {20, 0}, {10, 0}}, 5)];
            NSArray<NSData *> *withLake = @[
                ring((Point[]){{0, 0}, {0, 10}, {10, 10}, {10, 0}, {0, 0}}, 5),
                ring((Point[]){{3, 3}, {4, 3}, {4, 4}, {3, 4}, {3, 3}}, 5),
            ];

            writeCountiesShp([basePath stringByAppendingString:@".shp"], @[withLake, plain]);
            writeCountiesDbf([basePath stringByAppendingString:@".dbf"], @[@"Lakeside", @"Plainview"], @[@"48231", @"19876"]);

            DMShapeLayer *counties = [DMShapeLayer read:basePath];

            println(@"shapes = %zu, shape_type = %@", [counties size], [counties shapeType]);
            NSMutableArray<NSString *> *bounds = [NSMutableArray array];
            for (NSNumber *b in [counties bounds]) [bounds addObject:fmtIntegral(b.doubleValue)];
            println(@"bounds = [%@]\n", [bounds componentsJoinedByString:@", "]);

            DMDataFrame *attrs = [counties attributes];
            println(@"attributes\n%@\n", [attrs toString]);

            for (size_t i = 0; i < [counties size]; i++)
                println(@"%@: %zu ring(s)", [attrs stringAt:@"NAME" rowIndex:i], [counties numParts:i]);

            NSString *svgPath = @"datamunge_gis_ex_map_objc.svg";
            [[counties plot] saveSvg:svgPath];
            println(@"\nmap saved to %@", svgPath);
        } @catch (NSException *e) {
            printf("%s: %s\n", e.name.UTF8String, e.reason.UTF8String);
            return 1;
        }
    }
    return 0;
}
