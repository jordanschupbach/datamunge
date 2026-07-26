module app;

// Demonstrates datamunge.ShapeLayer: reading a shapefile (.shp geometry + .dbf attributes) and
// drawing it as a map. Real .shp/.dbf files are large binary bundles that don't belong in this
// repo, so this example first writes a tiny synthetic shapefile by hand (two "counties": one
// plain square, one with a lake-shaped hole) using the same ESRI byte layout ShapeLayer.read()
// expects, then reads it back through the public API. D's std.bitmanip reproduces Python's
// struct.pack byte layout (little-endian records, big-endian .shp header integers).

import std.stdio : writeln, writefln, File;
import std.bitmanip : nativeToLittleEndian, nativeToBigEndian;
import std.algorithm : min, max;
import std.format : format;
import datamunge;

ubyte[] leI(int v) { return nativeToLittleEndian(v).dup; }
ubyte[] beI(int v) { return nativeToBigEndian(v).dup; }
ubyte[] leD(double v) { return nativeToLittleEndian(v).dup; }
ubyte[] leU32(uint v) { return nativeToLittleEndian(v).dup; }
ubyte[] leU16(ushort v) { return nativeToLittleEndian(v).dup; }

// Left-justify string s to n bytes, padding on the right with the given fill byte; truncate if longer.
ubyte[] ljust(string s, size_t n, ubyte fill) {
  ubyte[] b = cast(ubyte[]) s.dup;
  if (b.length >= n) return b[0 .. n].dup;
  ubyte[] pad;
  pad.length = n - b.length;
  pad[] = fill;
  return b ~ pad;
}

ubyte[] zeros(size_t n) {
  ubyte[] b;
  b.length = n;
  return b;
}

ubyte[] polygon_record(double[2][][] rings) {
  double[2][] all_points;
  foreach (ring; rings) foreach (p; ring) all_points ~= p;
  double minx = all_points[0][0], miny = all_points[0][1];
  double maxx = minx, maxy = miny;
  foreach (p; all_points) {
    minx = min(minx, p[0]); maxx = max(maxx, p[0]);
    miny = min(miny, p[1]); maxy = max(maxy, p[1]);
  }
  ubyte[] outb = leI(5); // shape type: Polygon
  outb ~= leD(minx) ~ leD(miny) ~ leD(maxx) ~ leD(maxy);
  outb ~= leI(cast(int) rings.length);
  outb ~= leI(cast(int) all_points.length);
  int start = 0;
  foreach (ring; rings) {
    outb ~= leI(start);
    start += cast(int) ring.length;
  }
  foreach (ring; rings) {
    foreach (p; ring) {
      outb ~= leD(p[0]) ~ leD(p[1]);
    }
  }
  return outb;
}

void write_counties_shp(string path, double[2][][][] shapes) {
  ubyte[][] contents;
  foreach (rings; shapes) contents ~= polygon_record(rings);
  int total_words = 0;
  foreach (c; contents) total_words += 4 + cast(int)(c.length / 2);

  double minx = shapes[0][0][0][0], miny = shapes[0][0][0][1];
  double maxx = minx, maxy = miny;
  foreach (rings; shapes) foreach (ring; rings) foreach (p; ring) {
    minx = min(minx, p[0]); maxx = max(maxx, p[0]);
    miny = min(miny, p[1]); maxy = max(maxy, p[1]);
  }

  ubyte[] buf;
  buf ~= beI(9994);
  foreach (_; 0 .. 5) buf ~= beI(0);
  buf ~= beI(50 + total_words);
  buf ~= leI(1000);
  buf ~= leI(5); // Polygon
  buf ~= leD(minx) ~ leD(miny) ~ leD(maxx) ~ leD(maxy);
  buf ~= leD(0.0) ~ leD(0.0) ~ leD(0.0) ~ leD(0.0);
  foreach (i, content; contents) {
    buf ~= beI(cast(int)(i + 1));
    buf ~= beI(cast(int)(content.length / 2));
    buf ~= content;
  }

  auto f = File(path, "wb");
  f.rawWrite(buf);
  f.close();
}

void write_counties_dbf(string path, string[] names, string[] populations) {
  int header_size = 32 + 2 * 32 + 1;
  int record_size = 1 + 12 + 8;

  ubyte[] buf;
  buf ~= [cast(ubyte) 0x03, 0, 0, 0];
  buf ~= leU32(cast(uint) names.length);
  buf ~= leU16(cast(ushort) header_size);
  buf ~= leU16(cast(ushort) record_size);
  buf ~= zeros(20);

  void write_field(string name, string field_type, ubyte length) {
    buf ~= ljust(name, 11, 0);
    buf ~= cast(ubyte[]) field_type.dup;
    buf ~= zeros(4);
    buf ~= length;
    buf ~= zeros(15);
  }

  write_field("NAME", "C", 12);
  write_field("POP", "N", 8);
  buf ~= cast(ubyte) 0x0D;

  foreach (i, name; names) {
    buf ~= cast(ubyte) ' ';
    buf ~= ljust(name, 12, cast(ubyte) ' ');
    buf ~= ljust(populations[i], 8, cast(ubyte) ' ');
  }

  auto f = File(path, "wb");
  f.rawWrite(buf);
  f.close();
}

void main() {
  string base = "datamunge_gis_ex_counties_d";

  // Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
  double[2][][] plain = [
    [[10.0, 0.0], [10.0, 10.0], [20.0, 10.0], [20.0, 0.0], [10.0, 0.0]],
  ];
  double[2][][] with_lake = [
    [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0], [0.0, 0.0]],
    [[3.0, 3.0], [4.0, 3.0], [4.0, 4.0], [3.0, 4.0], [3.0, 3.0]],
  ];

  write_counties_shp(base ~ ".shp", [with_lake, plain]);
  write_counties_dbf(base ~ ".dbf", ["Lakeside", "Plainview"], ["48231", "19876"]);

  auto counties = ShapeLayer.read(base);

  writefln("shapes = %d, shape_type = %s", counties.size(), counties.shape_type());
  auto bounds = counties.bounds();
  string bt;
  for (size_t i = 0; i < bounds.size(); i++) bt ~= (i == 0 ? "" : ", ") ~ format("%g", bounds[i]);
  writefln("bounds = [%s]", bt);
  writeln();

  auto attributes = counties.attributes();
  writeln("attributes");
  writeln(attributes.to_string());
  writeln();

  for (size_t i = 0; i < counties.size(); i++) {
    auto name = attributes.string_at("NAME", i);
    writefln("%s: %d ring(s)", name, counties.num_parts(i));
  }

  string svg_path = "datamunge_gis_ex_map_d.svg";
  counties.plot().save_svg(svg_path);
  writefln("\nmap saved to %s", svg_path);
}
