// CMD: just TARGET=gis_ex run-swift
// ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
import Datamunge
#if canImport(Glibc)
import Glibc
#else
import Darwin
#endif

/// A Double as C#'s default ToString() prints it: the shortest round-trip digits, in E notation
/// when the decimal exponent is below -4 or at least 15 (6, 0.1, 1E-05, 1E+15).
func fmt(_ x: Double) -> String {
    if x.isNaN { return "NaN" }
    if x.isInfinite { return x < 0 ? "-∞" : "∞" }
    var s = "\(x)"
    let negative = s.hasPrefix("-")
    if negative { s.removeFirst() }
    var mantissa = Substring(s)
    var exponent = 0
    if let e = s.firstIndex(of: "e") {
        mantissa = s[..<e]
        exponent = Int(s[s.index(after: e)...])!
    }
    let parts = mantissa.split(separator: ".", omittingEmptySubsequences: false)
    var digits = String(parts[0]) + (parts.count > 1 ? String(parts[1]) : "")
    var point = parts[0].count + exponent
    while digits.hasPrefix("0") { digits.removeFirst(); point -= 1 }
    while digits.hasSuffix("0") { digits.removeLast() }
    let sign = negative ? "-" : ""
    if digits.isEmpty { return sign + "0" }
    let e = point - 1
    if e < -4 || e >= 15 {
        let rest = digits.dropFirst()
        let magnitude = abs(e) < 10 ? "0\(abs(e))" : "\(abs(e))"
        return sign + String(digits.first!) + (rest.isEmpty ? "" : "." + rest) + "E" + (e < 0 ? "-" : "+") + magnitude
    }
    if point <= 0 { return sign + "0." + String(repeating: "0", count: -point) + digits }
    if point >= digits.count { return sign + digits + String(repeating: "0", count: point - digits.count) }
    return sign + digits.prefix(point) + "." + digits.dropFirst(point)
}

/// A Double through a C printf format, e.g. cfmt("%.6f", x) for C#'s {x:F6}.
func cfmt(_ format: String, _ x: Double) -> String {
    var buffer = [CChar](repeating: 0, count: 64)
    _ = withVaList([x]) { vsnprintf(&buffer, buffer.count, format, $0) }
    return String(cString: buffer)
}

// Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
// writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
// layout ShapeLayer.read() expects, then reads them back through the public API.

typealias Ring = [(x: Double, y: Double)]

// Little- and big-endian packers that append to a byte buffer.
func packI4Le(_ buf: inout [UInt8], _ value: Int32) { withUnsafeBytes(of: value.littleEndian) { buf += $0 } }
func packI4Be(_ buf: inout [UInt8], _ value: Int32) { withUnsafeBytes(of: value.bigEndian) { buf += $0 } }
func packI2Le(_ buf: inout [UInt8], _ value: Int16) { withUnsafeBytes(of: value.littleEndian) { buf += $0 } }
func packDoubleLe(_ buf: inout [UInt8], _ x: Double) { withUnsafeBytes(of: x.bitPattern.littleEndian) { buf += $0 } }
func packAscii(_ buf: inout [UInt8], _ s: String) { buf += Array(s.utf8) }
func packZeros(_ buf: inout [UInt8], _ n: Int) { buf += [UInt8](repeating: 0, count: n) }

func writeAllBytes(_ path: String, _ bytes: [UInt8]) throws {
    struct WriteError: Error, CustomStringConvertible { let description: String }
    guard let file = fopen(path, "wb") else { throw WriteError(description: "cannot open \(path)") }
    defer { fclose(file) }
    guard fwrite(bytes, 1, bytes.count, file) == bytes.count else { throw WriteError(description: "cannot write \(path)") }
}

// Bounding box (minx, miny, maxx, maxy) of every point in rings.
func ringBounds(_ rings: [Ring]) -> [Double] {
    let points = rings.flatMap { $0 }
    return [points.map(\.x).min()!, points.map(\.y).min()!, points.map(\.x).max()!, points.map(\.y).max()!]
}

// Bytes of one Polygon record's contents (without the record header).
func polygonRecord(_ rings: [Ring]) -> [UInt8] {
    var output: [UInt8] = []
    packI4Le(&output, 5)  // shape type: Polygon
    for b in ringBounds(rings) { packDoubleLe(&output, b) }
    packI4Le(&output, Int32(rings.count))
    packI4Le(&output, Int32(rings.reduce(0) { $0 + $1.count }))
    var start = 0
    for ring in rings {
        packI4Le(&output, Int32(start))
        start += ring.count
    }
    for (x, y) in rings.flatMap({ $0 }) {
        packDoubleLe(&output, x)
        packDoubleLe(&output, y)
    }
    return output
}

func writeCountiesShp(_ path: String, _ shapes: [[Ring]]) throws {
    let contents = shapes.map(polygonRecord)
    let totalWords = contents.reduce(0) { $0 + 4 + $1.count / 2 }
    var f: [UInt8] = []
    packI4Be(&f, 9994)
    for _ in 0..<5 { packI4Be(&f, 0) }
    packI4Be(&f, Int32(50 + totalWords))
    packI4Le(&f, 1000)
    packI4Le(&f, 5)  // Polygon
    for b in ringBounds(shapes.flatMap { $0 }) { packDoubleLe(&f, b) }
    for _ in 0..<4 { packDoubleLe(&f, 0.0) }
    for (i, content) in contents.enumerated() {
        packI4Be(&f, Int32(i + 1))
        packI4Be(&f, Int32(content.count / 2))
        f += content
    }
    try writeAllBytes(path, f)
}

func ljust(_ s: String, _ n: Int, _ ch: Character) -> String {
    s.count >= n ? String(s.prefix(n)) : s + String(repeating: ch, count: n - s.count)
}

func writeCountiesDbf(_ path: String, _ names: [String], _ populations: [String]) throws {
    let headerSize: Int16 = 32 + 2 * 32 + 1
    let recordSize: Int16 = 1 + 12 + 8
    var f: [UInt8] = [0x03, 0, 0, 0]
    packI4Le(&f, Int32(names.count))
    packI2Le(&f, headerSize)
    packI2Le(&f, recordSize)
    packZeros(&f, 20)
    func writeField(_ name: String, _ fieldType: String, _ length: Int) {
        packAscii(&f, ljust(name, 11, "\0"))
        packAscii(&f, fieldType)
        packZeros(&f, 4)
        f.append(UInt8(length))
        packZeros(&f, 15)
    }
    writeField("NAME", "C", 12)
    writeField("POP", "N", 8)
    f.append(0x0D)
    for (name, pop) in zip(names, populations) {
        packAscii(&f, " ")
        packAscii(&f, ljust(name, 12, " "))
        packAscii(&f, ljust(pop, 8, " "))
    }
    try writeAllBytes(path, f)
}

// Roughly C's %g: integral doubles print without a fractional part.
func fmtG(_ x: Double) -> String { x == x.rounded(.down) ? String(Int64(x)) : cfmt("%.6f", x) }

let basePath = "datamunge_gis_ex_counties_swift"

// Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
let plain: [Ring] = [[(10, 0), (10, 10), (20, 10), (20, 0), (10, 0)]]
let withLake: [Ring] = [
    [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
    [(3, 3), (4, 3), (4, 4), (3, 4), (3, 3)],
]

do {
    try writeCountiesShp("\(basePath).shp", [withLake, plain])
    try writeCountiesDbf("\(basePath).dbf", ["Lakeside", "Plainview"], ["48231", "19876"])

    let counties = try ShapeLayer.read(basePath)!

    print("shapes = \(try counties.size()), shape_type = \(try counties.shapeType())")
    print("bounds = [\(try counties.bounds().map(fmtG).joined(separator: ", "))]\n")

    let attrs = try counties.attributes()!
    print("attributes\n\(attrs)\n")

    for i in 0..<(try counties.size()) {
        print("\(try attrs.stringAt("NAME", i)): \(try counties.numParts(i)) ring(s)")
    }

    let svgPath = "datamunge_gis_ex_map_swift.svg"
    try counties.plot().saveSvg(svgPath)
    print("\nmap saved to \(svgPath)")
} catch {
    print(error)
    exit(1)
}
