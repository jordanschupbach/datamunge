// CMD: just TARGET=gis_ex run-dart
// ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
// Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
// writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
// layout ShapeLayer.read() expects, then reads them back through the public API.
import 'dart:convert' show ascii;
import 'dart:io' show File;
import 'dart:math' as math;
import 'dart:typed_data' show ByteData, Endian;

import 'package:datamunge/datamunge.dart';

typedef Point = ({double x, double y});
typedef Ring = List<Point>;

const basePath = 'datamunge_gis_ex_counties_dart';

/// Roughly C's %g: integral doubles print without a fractional part.
String fmtG(double x) => x == x.floorToDouble() ? '${x.toInt()}' : x.toStringAsFixed(6);

void main() {
  // Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
  final List<Ring> plain = [
    [(x: 10, y: 0), (x: 10, y: 10), (x: 20, y: 10), (x: 20, y: 0), (x: 10, y: 0)],
  ];
  final List<Ring> withLake = [
    [(x: 0, y: 0), (x: 0, y: 10), (x: 10, y: 10), (x: 10, y: 0), (x: 0, y: 0)],
    [(x: 3, y: 3), (x: 4, y: 3), (x: 4, y: 4), (x: 3, y: 4), (x: 3, y: 3)],
  ];

  writeCountiesShp('$basePath.shp', [withLake, plain]);
  writeCountiesDbf('$basePath.dbf', ['Lakeside', 'Plainview'], ['48231', '19876']);

  final counties = ShapeLayer.read(basePath)!;

  print('shapes = ${counties.size()}, shape_type = ${counties.shapeType()}');
  print('bounds = [${counties.bounds().map(fmtG).join(', ')}]\n');

  final attrs = counties.attributes()!;
  print('attributes\n${attrs.toString()}\n');

  for (var i = 0; i < counties.size(); i++) {
    print('${attrs.stringAt('NAME', i)}: ${counties.numParts(i)} ring(s)');
  }

  const svgPath = 'datamunge_gis_ex_map_dart.svg';
  counties.plot().saveSvg(svgPath);
  print('\nmap saved to $svgPath');
}

// Little- and big-endian packers that append to a byte buffer.
void packI4Le(List<int> buf, int value) =>
    buf.addAll((ByteData(4)..setInt32(0, value, Endian.little)).buffer.asUint8List());
void packI4Be(List<int> buf, int value) =>
    buf.addAll((ByteData(4)..setInt32(0, value, Endian.big)).buffer.asUint8List());
void packI2Le(List<int> buf, int value) =>
    buf.addAll((ByteData(2)..setInt16(0, value, Endian.little)).buffer.asUint8List());
void packDoubleLe(List<int> buf, double x) =>
    buf.addAll((ByteData(8)..setFloat64(0, x, Endian.little)).buffer.asUint8List());
void packAscii(List<int> buf, String s) => buf.addAll(ascii.encode(s));
void packZeros(List<int> buf, int n) => buf.addAll(List.filled(n, 0));

/// Bounding box (minx, miny, maxx, maxy) of every point in rings.
List<double> ringBounds(Iterable<Ring> rings) {
  final points = rings.expand((r) => r).toList();
  return [
    points.map((p) => p.x).reduce(math.min),
    points.map((p) => p.y).reduce(math.min),
    points.map((p) => p.x).reduce(math.max),
    points.map((p) => p.y).reduce(math.max),
  ];
}

/// Bytes of one Polygon record's contents (without the record header).
List<int> polygonRecord(List<Ring> rings) {
  final output = <int>[];
  packI4Le(output, 5); // shape type: Polygon
  for (final b in ringBounds(rings)) {
    packDoubleLe(output, b);
  }
  packI4Le(output, rings.length);
  packI4Le(output, rings.fold(0, (n, r) => n + r.length));
  var start = 0;
  for (final ring in rings) {
    packI4Le(output, start);
    start += ring.length;
  }
  for (final (:x, :y) in rings.expand((r) => r)) {
    packDoubleLe(output, x);
    packDoubleLe(output, y);
  }
  return output;
}

void writeCountiesShp(String path, List<List<Ring>> shapes) {
  final contents = shapes.map(polygonRecord).toList();
  final totalWords = contents.fold(0, (n, c) => n + 4 + c.length ~/ 2);
  final f = <int>[];
  packI4Be(f, 9994);
  for (var i = 0; i < 5; i++) {
    packI4Be(f, 0);
  }
  packI4Be(f, 50 + totalWords);
  packI4Le(f, 1000);
  packI4Le(f, 5); // Polygon
  for (final b in ringBounds(shapes.expand((s) => s))) {
    packDoubleLe(f, b);
  }
  for (var i = 0; i < 4; i++) {
    packDoubleLe(f, 0.0);
  }
  for (var i = 0; i < contents.length; i++) {
    packI4Be(f, i + 1);
    packI4Be(f, contents[i].length ~/ 2);
    f.addAll(contents[i]);
  }
  File(path).writeAsBytesSync(f);
}

String ljust(String s, int n, String ch) => s.length >= n ? s.substring(0, n) : s.padRight(n, ch);

void writeCountiesDbf(String path, List<String> names, List<String> populations) {
  const headerSize = 32 + 2 * 32 + 1;
  const recordSize = 1 + 12 + 8;
  final f = <int>[0x03, 0, 0, 0];
  packI4Le(f, names.length);
  packI2Le(f, headerSize);
  packI2Le(f, recordSize);
  packZeros(f, 20);
  void writeField(String name, String fieldType, int length) {
    packAscii(f, ljust(name, 11, '\x00'));
    packAscii(f, fieldType);
    packZeros(f, 4);
    f.add(length);
    packZeros(f, 15);
  }

  writeField('NAME', 'C', 12);
  writeField('POP', 'N', 8);
  f.add(0x0D);
  for (var i = 0; i < names.length && i < populations.length; i++) {
    packAscii(f, ' ');
    packAscii(f, ljust(names[i], 12, ' '));
    packAscii(f, ljust(populations[i], 8, ' '));
  }
  File(path).writeAsBytesSync(f);
}
