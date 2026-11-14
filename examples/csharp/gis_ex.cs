// CMD: just TARGET=gis_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
// Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
// writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
// layout ShapeLayer.Read() expects, then reads them back through the public API.
using System.Buffers.Binary;
using System.Text;
using Datamunge;

const string basePath = "datamunge_gis_ex_counties_csharp";

// Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
(double X, double Y)[][] plain = [[(10, 0), (10, 10), (20, 10), (20, 0), (10, 0)]];
(double X, double Y)[][] withLake =
[
    [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
    [(3, 3), (4, 3), (4, 4), (3, 4), (3, 3)],
];

WriteCountiesShp($"{basePath}.shp", [withLake, plain]);
WriteCountiesDbf($"{basePath}.dbf", ["Lakeside", "Plainview"], ["48231", "19876"]);

var counties = ShapeLayer.Read(basePath)!;

Console.WriteLine($"shapes = {counties.Size()}, shape_type = {counties.ShapeType()}");
// Roughly C's %g: integral doubles print without a fractional part.
static string FmtG(double x) => x == Math.Floor(x) ? ((long)x).ToString() : x.ToString("F6");
Console.WriteLine($"bounds = [{string.Join(", ", counties.Bounds().Select(FmtG))}]\n");

var attrs = counties.Attributes()!;
Console.WriteLine($"attributes\n{attrs}\n");

for (ulong i = 0; i < counties.Size(); i++)
    Console.WriteLine($"{attrs.StringAt("NAME", i)}: {counties.NumParts(i)} ring(s)");

const string svgPath = "datamunge_gis_ex_map_csharp.svg";
counties.Plot().SaveSvg(svgPath);
Console.WriteLine($"\nmap saved to {svgPath}");

// Little- and big-endian packers that append to a byte buffer.
static void PackI4Le(List<byte> buf, int value)
{
    var bytes = new byte[4];
    BinaryPrimitives.WriteInt32LittleEndian(bytes, value);
    buf.AddRange(bytes);
}
static void PackI4Be(List<byte> buf, int value)
{
    var bytes = new byte[4];
    BinaryPrimitives.WriteInt32BigEndian(bytes, value);
    buf.AddRange(bytes);
}
static void PackI2Le(List<byte> buf, short value)
{
    var bytes = new byte[2];
    BinaryPrimitives.WriteInt16LittleEndian(bytes, value);
    buf.AddRange(bytes);
}
static void PackDoubleLe(List<byte> buf, double x)
{
    var bytes = new byte[8];
    BinaryPrimitives.WriteDoubleLittleEndian(bytes, x);
    buf.AddRange(bytes);
}
static void PackAscii(List<byte> buf, string s) => buf.AddRange(Encoding.ASCII.GetBytes(s));
static void PackZeros(List<byte> buf, int n) => buf.AddRange(new byte[n]);

// Bounding box (minx, miny, maxx, maxy) of every point in rings.
static double[] RingBounds(IEnumerable<(double X, double Y)[]> rings)
{
    var points = rings.SelectMany(r => r).ToList();
    return [points.Min(p => p.X), points.Min(p => p.Y), points.Max(p => p.X), points.Max(p => p.Y)];
}

// Bytes of one Polygon record's contents (without the record header).
static List<byte> PolygonRecord((double X, double Y)[][] rings)
{
    List<byte> output = [];
    PackI4Le(output, 5);  // shape type: Polygon
    foreach (var b in RingBounds(rings)) PackDoubleLe(output, b);
    PackI4Le(output, rings.Length);
    PackI4Le(output, rings.Sum(r => r.Length));
    var start = 0;
    foreach (var ring in rings)
    {
        PackI4Le(output, start);
        start += ring.Length;
    }
    foreach (var (x, y) in rings.SelectMany(r => r))
    {
        PackDoubleLe(output, x);
        PackDoubleLe(output, y);
    }
    return output;
}

static void WriteCountiesShp(string path, (double X, double Y)[][][] shapes)
{
    var contents = shapes.Select(PolygonRecord).ToList();
    var totalWords = contents.Sum(c => 4 + c.Count / 2);
    List<byte> f = [];
    PackI4Be(f, 9994);
    for (var i = 0; i < 5; i++) PackI4Be(f, 0);
    PackI4Be(f, 50 + totalWords);
    PackI4Le(f, 1000);
    PackI4Le(f, 5);  // Polygon
    foreach (var b in RingBounds(shapes.SelectMany(s => s))) PackDoubleLe(f, b);
    for (var i = 0; i < 4; i++) PackDoubleLe(f, 0.0);
    for (var i = 0; i < contents.Count; i++)
    {
        PackI4Be(f, i + 1);
        PackI4Be(f, contents[i].Count / 2);
        f.AddRange(contents[i]);
    }
    File.WriteAllBytes(path, [.. f]);
}

static string Ljust(string s, int n, char ch) => s.Length >= n ? s[..n] : s.PadRight(n, ch);

static void WriteCountiesDbf(string path, string[] names, string[] populations)
{
    const int headerSize = 32 + 2 * 32 + 1;
    const int recordSize = 1 + 12 + 8;
    List<byte> f = [0x03, 0, 0, 0];
    PackI4Le(f, names.Length);
    PackI2Le(f, headerSize);
    PackI2Le(f, recordSize);
    PackZeros(f, 20);
    void WriteField(string name, string fieldType, int length)
    {
        PackAscii(f, Ljust(name, 11, '\0'));
        PackAscii(f, fieldType);
        PackZeros(f, 4);
        f.Add((byte)length);
        PackZeros(f, 15);
    }
    WriteField("NAME", "C", 12);
    WriteField("POP", "N", 8);
    f.Add(0x0D);
    foreach (var (name, pop) in names.Zip(populations))
    {
        PackAscii(f, " ");
        PackAscii(f, Ljust(name, 12, ' '));
        PackAscii(f, Ljust(pop, 8, ' '));
    }
    File.WriteAllBytes(path, [.. f]);
}
