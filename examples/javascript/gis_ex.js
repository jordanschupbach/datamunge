// CMD: just TARGET=gis_ex run-javascript
// Demonstrates datamunge.ShapeLayer: reading a shapefile (.shp geometry + .dbf attributes) and
// drawing it as a map. Real .shp/.dbf files are large binary bundles that don't belong in this
// repo, so this example first writes a tiny synthetic shapefile by hand (two "counties": one
// plain square, one with a lake-shaped hole) using the same ESRI byte layout ShapeLayer.read()
// expects, then reads it back through the public API. Node's Buffer read/write helpers mirror
// Python's struct.pack for the binary layout.

const fs = require("fs");
const datamunge = require("../../index.js");

// Bounding box [minx, miny, maxx, maxy] of every point in rings.
function ringBounds(rings) {
  const points = rings.flat();
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function int32LE(v) {
  const b = Buffer.alloc(4);
  b.writeInt32LE(v);
  return b;
}

function int32BE(v) {
  const b = Buffer.alloc(4);
  b.writeInt32BE(v);
  return b;
}

function doublesLE(values) {
  const b = Buffer.alloc(8 * values.length);
  values.forEach((v, i) => b.writeDoubleLE(v, 8 * i));
  return b;
}

// Bytes of one Polygon record's contents (without the record header).
function polygonRecord(rings) {
  const parts = [int32LE(5)]; // shape type: Polygon
  parts.push(doublesLE(ringBounds(rings)));
  parts.push(int32LE(rings.length));
  parts.push(int32LE(rings.reduce((n, ring) => n + ring.length, 0)));
  let start = 0;
  for (const ring of rings) {
    parts.push(int32LE(start));
    start += ring.length;
  }
  for (const ring of rings) {
    for (const [x, y] of ring) parts.push(doublesLE([x, y]));
  }
  return Buffer.concat(parts);
}

function writeCountiesShp(path, shapes) {
  const contents = shapes.map(polygonRecord);
  const totalWords = contents.reduce((n, c) => n + 4 + c.length / 2, 0);
  const parts = [int32BE(9994)];
  for (let i = 0; i < 5; i++) parts.push(int32BE(0));
  parts.push(int32BE(50 + totalWords));
  parts.push(int32LE(1000));
  parts.push(int32LE(5)); // Polygon
  parts.push(doublesLE(ringBounds(shapes.flat())));
  parts.push(doublesLE([0.0, 0.0, 0.0, 0.0]));
  contents.forEach((content, i) => {
    parts.push(int32BE(i + 1));
    parts.push(int32BE(content.length / 2));
    parts.push(content);
  });
  fs.writeFileSync(path, Buffer.concat(parts));
}

// ASCII bytes of s truncated or padded with fill to exactly n bytes.
function fixedAscii(s, n, fill) {
  const b = Buffer.alloc(n, fill);
  b.write(s.slice(0, n), "ascii");
  return b;
}

function writeCountiesDbf(path, names, populations) {
  const headerSize = 32 + 2 * 32 + 1;
  const recordSize = 1 + 12 + 8;
  const header = Buffer.alloc(32);
  header.writeUInt8(0x03, 0);
  header.writeUInt32LE(names.length, 4);
  header.writeUInt16LE(headerSize, 8);
  header.writeUInt16LE(recordSize, 10);
  const parts = [header];

  const field = (name, fieldType, length) => {
    const b = Buffer.alloc(32);
    b.write(name, 0, "ascii");
    b.write(fieldType, 11, "ascii");
    b.writeUInt8(length, 16);
    return b;
  };
  parts.push(field("NAME", "C", 12));
  parts.push(field("POP", "N", 8));
  parts.push(Buffer.from([0x0d]));

  names.forEach((name, i) => {
    parts.push(Buffer.from(" ", "ascii"));
    parts.push(fixedAscii(name, 12, " "));
    parts.push(fixedAscii(populations[i], 8, " "));
  });
  fs.writeFileSync(path, Buffer.concat(parts));
}

const base = "datamunge_gis_ex_counties_javascript";

// Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
const plain = [[[10, 0], [10, 10], [20, 10], [20, 0], [10, 0]]];
const withLake = [
  [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]],
  [[3, 3], [4, 3], [4, 4], [3, 4], [3, 3]],
];

writeCountiesShp(`${base}.shp`, [withLake, plain]);
writeCountiesDbf(`${base}.dbf`, ["Lakeside", "Plainview"], ["48231", "19876"]);

const counties = datamunge.ShapeLayer.read(base);

console.log(`shapes = ${counties.size()}, shape_type = ${counties.shape_type()}`);
console.log(`bounds = [${Array.from(counties.bounds()).join(", ")}]`);
console.log();

const attributes = counties.attributes();
console.log("attributes");
console.log(attributes.to_string());
console.log();

for (let i = 0; i < counties.size(); i++) {
  const name = attributes.string_at("NAME", i);
  console.log(`${name}: ${counties.num_parts(i)} ring(s)`);
}

const svgPath = "datamunge_gis_ex_map_javascript.svg";
counties.plot().save_svg(svgPath);
console.log(`\nmap saved to ${svgPath}`);
