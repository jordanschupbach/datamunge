' CMD: just TARGET=gis_ex run-vb
' ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
' Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
' writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
' layout ShapeLayer.Read() expects, then reads them back through the public API.
Option Strict On
Imports System
Imports System.Buffers.Binary
Imports System.Collections.Generic
Imports System.IO
Imports System.Linq
Imports System.Text
Imports Datamunge

Module Example
    Const basePath As String = "datamunge_gis_ex_counties_vb"
    Const svgPath As String = "datamunge_gis_ex_map_vb.svg"

    Sub Main()
        ' Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
        Dim plain As (X As Double, Y As Double)()() = {
            New(X As Double, Y As Double)() {(10, 0), (10, 10), (20, 10), (20, 0), (10, 0)}
        }
        Dim withLake As (X As Double, Y As Double)()() = {
            New(X As Double, Y As Double)() {(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)},
            New(X As Double, Y As Double)() {(3, 3), (4, 3), (4, 4), (3, 4), (3, 3)}
        }

        WriteCountiesShp($"{basePath}.shp", {withLake, plain})
        WriteCountiesDbf($"{basePath}.dbf", {"Lakeside", "Plainview"}, {"48231", "19876"})

        Dim counties = ShapeLayer.Read(basePath)

        Console.WriteLine($"shapes = {counties.Size()}, shape_type = {counties.ShapeType()}")
        Console.WriteLine($"bounds = [{String.Join(", ", counties.Bounds().Select(AddressOf FmtG))}]{vbLf}")

        Dim attrs = counties.Attributes()
        Console.WriteLine($"attributes{vbLf}{attrs}{vbLf}")

        For i As ULong = 0UL To counties.Size() - 1UL
            Console.WriteLine($"{attrs.StringAt("NAME", i)}: {counties.NumParts(i)} ring(s)")
        Next

        counties.Plot().SaveSvg(svgPath)
        Console.WriteLine($"{vbLf}map saved to {svgPath}")
    End Sub

    ' Roughly C's %g: integral doubles print without a fractional part.
    Private Function FmtG(x As Double) As String
        Return If(x = Math.Floor(x), CLng(x).ToString(), x.ToString("F6"))
    End Function

    ' Little- and big-endian packers that append to a byte buffer.
    Private Sub PackI4Le(buf As List(Of Byte), value As Integer)
        Dim bytes(3) As Byte
        BinaryPrimitives.WriteInt32LittleEndian(bytes, value)
        buf.AddRange(bytes)
    End Sub

    Private Sub PackI4Be(buf As List(Of Byte), value As Integer)
        Dim bytes(3) As Byte
        BinaryPrimitives.WriteInt32BigEndian(bytes, value)
        buf.AddRange(bytes)
    End Sub

    Private Sub PackI2Le(buf As List(Of Byte), value As Short)
        Dim bytes(1) As Byte
        BinaryPrimitives.WriteInt16LittleEndian(bytes, value)
        buf.AddRange(bytes)
    End Sub

    Private Sub PackDoubleLe(buf As List(Of Byte), x As Double)
        Dim bytes(7) As Byte
        BinaryPrimitives.WriteDoubleLittleEndian(bytes, x)
        buf.AddRange(bytes)
    End Sub

    Private Sub PackAscii(buf As List(Of Byte), s As String)
        buf.AddRange(Encoding.ASCII.GetBytes(s))
    End Sub

    Private Sub PackZeros(buf As List(Of Byte), n As Integer)
        buf.AddRange(New Byte(n - 1) {})
    End Sub

    ' Bounding box (minx, miny, maxx, maxy) of every point in rings.
    Private Function RingBounds(rings As IEnumerable(Of (X As Double, Y As Double)())) As Double()
        Dim points = rings.SelectMany(Function(r) r).ToList()
        Return {points.Min(Function(p) p.X), points.Min(Function(p) p.Y), points.Max(Function(p) p.X), points.Max(Function(p) p.Y)}
    End Function

    ' Bytes of one Polygon record's contents (without the record header).
    Private Function PolygonRecord(rings As (X As Double, Y As Double)()()) As List(Of Byte)
        Dim output As New List(Of Byte)
        PackI4Le(output, 5)  ' shape type: Polygon
        For Each b In RingBounds(rings) : PackDoubleLe(output, b) : Next
        PackI4Le(output, rings.Length)
        PackI4Le(output, rings.Sum(Function(r) r.Length))
        Dim start = 0
        For Each ring In rings
            PackI4Le(output, start)
            start += ring.Length
        Next
        For Each point In rings.SelectMany(Function(r) r)
            PackDoubleLe(output, point.X)
            PackDoubleLe(output, point.Y)
        Next
        Return output
    End Function

    Private Sub WriteCountiesShp(path As String, shapes As (X As Double, Y As Double)()()())
        Dim contents = shapes.Select(AddressOf PolygonRecord).ToList()
        Dim totalWords = contents.Sum(Function(c) 4 + c.Count \ 2)
        Dim f As New List(Of Byte)
        PackI4Be(f, 9994)
        For i = 0 To 4 : PackI4Be(f, 0) : Next
        PackI4Be(f, 50 + totalWords)
        PackI4Le(f, 1000)
        PackI4Le(f, 5)  ' Polygon
        For Each b In RingBounds(shapes.SelectMany(Function(s) s)) : PackDoubleLe(f, b) : Next
        For i = 0 To 3 : PackDoubleLe(f, 0.0) : Next
        For i = 0 To contents.Count - 1
            PackI4Be(f, i + 1)
            PackI4Be(f, contents(i).Count \ 2)
            f.AddRange(contents(i))
        Next
        File.WriteAllBytes(path, f.ToArray())
    End Sub

    Private Function Ljust(s As String, n As Integer, ch As Char) As String
        Return If(s.Length >= n, s.Substring(0, n), s.PadRight(n, ch))
    End Function

    Private Sub WriteField(f As List(Of Byte), name As String, fieldType As String, length As Integer)
        PackAscii(f, Ljust(name, 11, ChrW(0)))
        PackAscii(f, fieldType)
        PackZeros(f, 4)
        f.Add(CByte(length))
        PackZeros(f, 15)
    End Sub

    Private Sub WriteCountiesDbf(path As String, names As String(), populations As String())
        Const headerSize As Short = 32 + 2 * 32 + 1
        Const recordSize As Short = 1 + 12 + 8
        Dim f As New List(Of Byte) From {&H3, 0, 0, 0}
        PackI4Le(f, names.Length)
        PackI2Le(f, headerSize)
        PackI2Le(f, recordSize)
        PackZeros(f, 20)
        WriteField(f, "NAME", "C", 12)
        WriteField(f, "POP", "N", 8)
        f.Add(&HD)
        For Each pair In names.Zip(populations)
            PackAscii(f, " ")
            PackAscii(f, Ljust(pair.Item1, 12, " "c))
            PackAscii(f, Ljust(pair.Item2, 8, " "c))
        Next
        File.WriteAllBytes(path, f.ToArray())
    End Sub
End Module
