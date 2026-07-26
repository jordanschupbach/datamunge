// Demonstrates datamunge.ShapeLayer: reading a shapefile (.shp geometry + .dbf attributes) and
// drawing it as a map. Real .shp/.dbf files are large binary bundles that don't belong in this
// repo, so this example first writes a tiny synthetic shapefile by hand (two "counties": one
// plain square, one with a lake-shaped hole) using the same ESRI byte layout ShapeLayer.read()
// expects, then reads it back through the public API. Go's encoding/binary reproduces the byte
// layout Python's struct.pack produces.
package main

import (
	"bytes"
	"datamunge"
	"encoding/binary"
	"fmt"
	"os"
	"strings"
)

// padRight clips to n bytes if longer, otherwise pads on the right with pad up to n bytes.
func padRight(b []byte, n int, pad byte) []byte {
	if len(b) >= n {
		return b[:n]
	}
	out := make([]byte, n)
	copy(out, b)
	for i := len(b); i < n; i++ {
		out[i] = pad
	}
	return out
}

func bounds(points [][2]float64) (minx, miny, maxx, maxy float64) {
	minx, miny = points[0][0], points[0][1]
	maxx, maxy = points[0][0], points[0][1]
	for _, p := range points {
		if p[0] < minx {
			minx = p[0]
		}
		if p[0] > maxx {
			maxx = p[0]
		}
		if p[1] < miny {
			miny = p[1]
		}
		if p[1] > maxy {
			maxy = p[1]
		}
	}
	return
}

func polygonRecord(rings [][][2]float64) []byte {
	var allPoints [][2]float64
	for _, ring := range rings {
		allPoints = append(allPoints, ring...)
	}
	minx, miny, maxx, maxy := bounds(allPoints)

	buf := new(bytes.Buffer)
	binary.Write(buf, binary.LittleEndian, int32(5)) // shape type: Polygon
	binary.Write(buf, binary.LittleEndian, minx)
	binary.Write(buf, binary.LittleEndian, miny)
	binary.Write(buf, binary.LittleEndian, maxx)
	binary.Write(buf, binary.LittleEndian, maxy)
	binary.Write(buf, binary.LittleEndian, int32(len(rings)))
	binary.Write(buf, binary.LittleEndian, int32(len(allPoints)))
	start := int32(0)
	for _, ring := range rings {
		binary.Write(buf, binary.LittleEndian, start)
		start += int32(len(ring))
	}
	for _, ring := range rings {
		for _, p := range ring {
			binary.Write(buf, binary.LittleEndian, p[0])
			binary.Write(buf, binary.LittleEndian, p[1])
		}
	}
	return buf.Bytes()
}

func writeCountiesShp(path string, shapes [][][][2]float64) {
	contents := make([][]byte, len(shapes))
	for i, rings := range shapes {
		contents[i] = polygonRecord(rings)
	}
	totalWords := 0
	for _, c := range contents {
		totalWords += 4 + len(c)/2
	}
	var allPoints [][2]float64
	for _, rings := range shapes {
		for _, ring := range rings {
			allPoints = append(allPoints, ring...)
		}
	}
	minx, miny, maxx, maxy := bounds(allPoints)

	buf := new(bytes.Buffer)
	binary.Write(buf, binary.BigEndian, int32(9994))
	for k := 0; k < 5; k++ {
		binary.Write(buf, binary.BigEndian, int32(0))
	}
	binary.Write(buf, binary.BigEndian, int32(50+totalWords))
	binary.Write(buf, binary.LittleEndian, int32(1000))
	binary.Write(buf, binary.LittleEndian, int32(5)) // Polygon
	binary.Write(buf, binary.LittleEndian, minx)
	binary.Write(buf, binary.LittleEndian, miny)
	binary.Write(buf, binary.LittleEndian, maxx)
	binary.Write(buf, binary.LittleEndian, maxy)
	binary.Write(buf, binary.LittleEndian, float64(0))
	binary.Write(buf, binary.LittleEndian, float64(0))
	binary.Write(buf, binary.LittleEndian, float64(0))
	binary.Write(buf, binary.LittleEndian, float64(0))
	for i, content := range contents {
		binary.Write(buf, binary.BigEndian, int32(i+1))
		binary.Write(buf, binary.BigEndian, int32(len(content)/2))
		buf.Write(content)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		panic(err)
	}
}

func writeCountiesDbf(path string, names, populations []string) {
	headerSize := 32 + 2*32 + 1
	recordSize := 1 + 12 + 8

	buf := new(bytes.Buffer)
	buf.Write([]byte{0x03, 0, 0, 0})
	binary.Write(buf, binary.LittleEndian, uint32(len(names)))
	binary.Write(buf, binary.LittleEndian, uint16(headerSize))
	binary.Write(buf, binary.LittleEndian, uint16(recordSize))
	buf.Write(make([]byte, 20))

	writeField := func(name string, fieldType byte, length byte) {
		buf.Write(padRight([]byte(name), 11, 0))
		buf.WriteByte(fieldType)
		buf.Write(make([]byte, 4))
		buf.WriteByte(length)
		buf.Write(make([]byte, 15))
	}

	writeField("NAME", 'C', 12)
	writeField("POP", 'N', 8)
	buf.WriteByte(0x0D)

	for i := range names {
		buf.WriteByte(' ')
		buf.Write(padRight([]byte(names[i]), 12, ' '))
		buf.Write(padRight([]byte(populations[i]), 8, ' '))
	}
	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		panic(err)
	}
}

func main() {
	base := "datamunge_gis_ex_counties_go"

	// Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
	plain := [][][2]float64{{{10, 0}, {10, 10}, {20, 10}, {20, 0}, {10, 0}}}
	withLake := [][][2]float64{
		{{0, 0}, {0, 10}, {10, 10}, {10, 0}, {0, 0}},
		{{3, 3}, {4, 3}, {4, 4}, {3, 4}, {3, 3}},
	}

	writeCountiesShp(base+".shp", [][][][2]float64{withLake, plain})
	writeCountiesDbf(base+".dbf", []string{"Lakeside", "Plainview"}, []string{"48231", "19876"})

	counties := datamunge.ShapeLayerRead(base)

	fmt.Printf("shapes = %d, shape_type = %s\n", counties.Size(), counties.Shape_type())
	b := counties.Bounds()
	parts := make([]string, b.Size())
	for i := 0; i < int(b.Size()); i++ {
		parts[i] = fmt.Sprintf("%g", b.Get(i))
	}
	fmt.Printf("bounds = [%s]\n", strings.Join(parts, ", "))
	fmt.Println()

	attributes := counties.Attributes()
	fmt.Println("attributes")
	fmt.Println(attributes.To_string())
	fmt.Println()

	for i := int64(0); i < counties.Size(); i++ {
		name := attributes.String_at("NAME", i)
		fmt.Printf("%s: %d ring(s)\n", name, counties.Num_parts(i))
	}

	svgPath := "datamunge_gis_ex_map_go.svg"
	counties.Plot().Save_svg(svgPath)
	fmt.Printf("\nmap saved to %s\n", svgPath)
}
