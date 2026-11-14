// CMD: just TARGET=gis_ex run-go

//go:build ignore

// ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
package main

import (
	"bytes"
	"datamunge"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

// Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
// writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
// layout ShapeLayerRead expects, then reads them back through the public API.

func check(err error) {
	if err != nil {
		panic(err)
	}
}

// A shape is a list of rings; a ring is a list of {x, y} points.
type shape [][][2]float64

// bounds is the bounding box {minx, miny, maxx, maxy} of every point in the given shapes.
func bounds(shapes []shape) [4]float64 {
	b := [4]float64{math.Inf(1), math.Inf(1), math.Inf(-1), math.Inf(-1)}
	for _, s := range shapes {
		for _, ring := range s {
			for _, p := range ring {
				b[0] = math.Min(b[0], p[0])
				b[1] = math.Min(b[1], p[1])
				b[2] = math.Max(b[2], p[0])
				b[3] = math.Max(b[3], p[1])
			}
		}
	}
	return b
}

// put appends the binary encoding of each value to buf in the given byte order.
func put(buf *bytes.Buffer, order binary.ByteOrder, values ...any) {
	for _, v := range values {
		check(binary.Write(buf, order, v))
	}
}

// polygonRecord is the bytes of one Polygon record's contents (without the record header).
func polygonRecord(s shape) []byte {
	var buf bytes.Buffer
	le := binary.LittleEndian
	total := 0
	for _, ring := range s {
		total += len(ring)
	}
	// Shape type 5 = Polygon, then bounds, part/point counts, part starts and the points.
	put(&buf, le, int32(5), bounds([]shape{s}), int32(len(s)), int32(total))
	start := 0
	for _, ring := range s {
		put(&buf, le, int32(start))
		start += len(ring)
	}
	for _, ring := range s {
		for _, p := range ring {
			put(&buf, le, p)
		}
	}
	return buf.Bytes()
}

func writeCountiesShp(path string, shapes []shape) {
	var contents [][]byte
	totalWords := 0
	for _, s := range shapes {
		c := polygonRecord(s)
		contents = append(contents, c)
		totalWords += 4 + len(c)/2
	}
	var out bytes.Buffer
	be, le := binary.BigEndian, binary.LittleEndian
	// big-endian file code and length, then little-endian version, Polygon type and bounds
	put(&out, be, int32(9994), [5]int32{}, int32(50+totalWords))
	put(&out, le, int32(1000), int32(5), bounds(shapes), [4]float64{})
	for i, c := range contents {
		put(&out, be, int32(i+1), int32(len(c)/2))
		out.Write(c)
	}
	check(os.WriteFile(path, out.Bytes(), 0o644))
}

// dbfField is one 32-byte dBASE field descriptor.
func dbfField(name string, typ byte, length int) []byte {
	b := make([]byte, 32)
	copy(b[:11], name)
	b[11] = typ
	b[16] = byte(length)
	return b
}

func writeCountiesDbf(path string, names, populations []string) {
	headerSize := 32 + 2*32 + 1
	recordSize := 1 + 12 + 8
	var out bytes.Buffer
	le := binary.LittleEndian
	put(&out, le, [4]byte{3, 0, 0, 0}, int32(len(names)), int16(headerSize), int16(recordSize), [20]byte{})
	out.Write(dbfField("NAME", 'C', 12))
	out.Write(dbfField("POP", 'N', 8))
	out.WriteByte('\r')
	for i := range names {
		fmt.Fprintf(&out, " %-12.12s%-8.8s", names[i], populations[i])
	}
	check(os.WriteFile(path, out.Bytes(), 0o644))
}

// fmtG is roughly C's %g: integral doubles print without a fractional part.
func fmtG(x float64) string {
	if x == math.Floor(x) {
		return strconv.FormatInt(int64(x), 10)
	}
	return fmt.Sprintf("%.6f", x)
}

func main() {
	basePath := "datamunge_gis_ex_counties_go"

	// Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
	plain := shape{{{10, 0}, {10, 10}, {20, 10}, {20, 0}, {10, 0}}}
	withLake := shape{
		{{0, 0}, {0, 10}, {10, 10}, {10, 0}, {0, 0}},
		{{3, 3}, {4, 3}, {4, 4}, {3, 4}, {3, 3}},
	}

	writeCountiesShp(basePath+".shp", []shape{withLake, plain})
	writeCountiesDbf(basePath+".dbf", []string{"Lakeside", "Plainview"}, []string{"48231", "19876"})

	counties := datamunge.Must(datamunge.ShapeLayerRead(basePath))
	defer counties.Close()

	fmt.Printf("shapes = %d, shape_type = %s\n", datamunge.Must(counties.Size()), datamunge.Must(counties.ShapeType()))
	var b []string
	for _, x := range datamunge.Must(counties.Bounds()) {
		b = append(b, fmtG(x))
	}
	fmt.Printf("bounds = [%s]\n\n", strings.Join(b, ", "))

	attrs := datamunge.Must(counties.Attributes())
	defer attrs.Close()
	fmt.Printf("attributes\n%v\n\n", attrs)

	for i := 0; i < datamunge.Must(counties.Size()); i++ {
		fmt.Printf("%s: %d ring(s)\n", datamunge.Must(attrs.StringAt("NAME", i)), datamunge.Must(counties.NumParts(i)))
	}

	svgPath := "datamunge_gis_ex_map_go.svg"
	check(datamunge.Must(counties.Plot()).SaveSvg(svgPath))
	fmt.Println("\nmap saved to " + svgPath)
}
