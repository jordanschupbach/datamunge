⍝ CMD: just TARGET=gis_ex run-apl
⍝ ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
⍝ Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
⍝ writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
⍝ layout dm∆ShapeLayer∆read expects, then reads them back through the public API.
NL←⎕UCS 10
BASE_PATH←'datamunge_gis_ex_counties_apl'
SVG_PATH←'datamunge_gis_ex_map_apl.svg'
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
⍝ NUM X: X with up to 10 significant digits ('-' for '¯').
∇Z←NUM X
 Z←⍕X
 Z[(Z='¯')/⍳⍴Z]←'-'
∇

⍝ Byte packers: each returns a vector of byte values (0-255).
∇Z←I4LE N
 Z←256|⌊((2*32)|N)÷256*0 1 2 3
∇
∇Z←I4BE N
 Z←⌽I4LE N
∇
∇Z←I2LE N
 Z←256|⌊((2*16)|N)÷256*0 1
∇
⍝ F8LE X: the IEEE-754 double X, little-endian (normal numbers and 0; exact for the small
⍝ coordinates used here). The 64-bit pattern exceeds APL's exact integers, so it is assembled per byte.
∇Z←F8LE X;S;A;E;F
 Z←8⍴0
 →(X=0)/0
 S←X<0
 A←|X
 E←⌊2⍟A
 E←E-A<2*E
 E←E+A≥2*E+1
 F←(2*52)×(A÷2*E)-1          ⍝ the 52 fraction bits, an integer
 E←E+1023                     ⍝ the biased 11-bit exponent
 Z←(256|⌊F÷256*¯1+⍳6),(16|⌊F÷2*48)+16×16|E
 Z←Z,(⌊E÷16)+128×S
∇
⍝ PATH WRITE_BYTES BYTES: writes the byte vector to the file PATH.
∇PATH WRITE_BYTES BYTES;H;N
 H←'w' ⎕FIO[3] PATH
 N←BYTES ⎕FIO[7] H
 N←⎕FIO[4] H
∇

⍝ A ring is a vector of (x y) points; a shape is a vector of rings.
⍝ (GNU APL is APL2: ↑ is first and ⊃ is disclose, so ⊃↑,/RINGS is the matrix of all points.)
⍝ RING_BOUNDS RINGS: (minx miny maxx maxy) of every point in RINGS.
∇Z←RING_BOUNDS RINGS;P
 P←⊃↑,/RINGS
 Z←(⌊/P[;1]),(⌊/P[;2]),(⌈/P[;1]),⌈/P[;2]
∇
⍝ POLYGON_RECORD RINGS: bytes of one Polygon record's contents (without the record header).
∇Z←POLYGON_RECORD RINGS;P
 P←⊃↑,/RINGS
 Z←(I4LE 5),(∊F8LE¨RING_BOUNDS RINGS)     ⍝ shape type: Polygon, then its bounding box
 Z←Z,(I4LE ≢RINGS),(I4LE ↑⍴P)               ⍝ number of parts and of points
 Z←Z,(∊I4LE¨¯1↓0,+\≢¨RINGS)                 ⍝ the index of each part's first point
 Z←Z,∊F8LE¨,P                                ⍝ x y of every point
∇
∇PATH WRITE_COUNTIES_SHP SHAPES;CONTENTS;F;I
 CONTENTS←POLYGON_RECORD¨SHAPES
 F←(I4BE 9994),(∊I4BE¨5⍴0),(I4BE 50++/4+(≢¨CONTENTS)÷2)
 F←F,(I4LE 1000),(I4LE 5),(∊F8LE¨RING_BOUNDS ↑,/SHAPES),∊F8LE¨4⍴0
 I←0
LOOP:→((≢CONTENTS)<I←I+1)/DONE
 F←F,(I4BE I),(I4BE (≢⊃CONTENTS[I])÷2),⊃CONTENTS[I]
 →LOOP
DONE:PATH WRITE_BYTES F
∇
⍝ N LJUST (TEXT PAD): TEXT padded with PAD (or truncated) to N characters.
∇Z←N LJUST B;TEXT;PAD
 (TEXT PAD)←B
 Z←N↑TEXT,N⍴PAD
∇
∇Z←FIELD B;NAME;TYPE;LEN
 (NAME TYPE LEN)←B
 Z←(⎕UCS 11 LJUST NAME (⎕UCS 0)),(⎕UCS TYPE),(4⍴0),LEN,15⍴0
∇
∇PATH WRITE_COUNTIES_DBF B;NAMES;POPS;F;I
 (NAMES POPS)←B
 F←3 0 0 0,(I4LE ≢NAMES),(I2LE 32+(2×32)+1),(I2LE 1+12+8),20⍴0
 F←F,(FIELD 'NAME' 'C' 12),(FIELD 'POP' 'N' 8),13
 I←0
LOOP:→((≢NAMES)<I←I+1)/DONE
 F←F,⎕UCS ' ',(12 LJUST (⊃NAMES[I]) ' '),8 LJUST (⊃POPS[I]) ' '
 →LOOP
DONE:PATH WRITE_BYTES F
∇

⍝ Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
PLAIN←,⊂(10 0) (10 10) (20 10) (20 0) (10 0)
WITH_LAKE←((0 0) (0 10) (10 10) (10 0) (0 0)) ((3 3) (4 3) (4 4) (3 4) (3 3))

(BASE_PATH,'.shp') WRITE_COUNTIES_SHP WITH_LAKE PLAIN
(BASE_PATH,'.dbf') WRITE_COUNTIES_DBF ('Lakeside' 'Plainview') ('48231' '19876')

COUNTIES←dm∆ShapeLayer∆read BASE_PATH
PUT 'shapes = ',(⍕dm∆ShapeLayer∆size COUNTIES),', shape_type = ',dm∆ShapeLayer∆shape_type COUNTIES
PUT 'bounds = [',(¯2↓∊(NUM¨dm∆ShapeLayer∆bounds COUNTIES),¨⊂', '),']',NL

⍝ ShapeLayer::attributes returns a new DataFrame, which the caller owns.
ATTRIBUTES←dm∆ShapeLayer∆attributes COUNTIES
PUT 'attributes',NL,(ATTRIBUTES dm∆DataFrame∆to_string ⍬),NL
∇SHOW_RINGS I
 PUT (ATTRIBUTES dm∆DataFrame∆string_at 'NAME' I),': ',(⍕COUNTIES dm∆ShapeLayer∆num_parts I),' ring(s)'
∇
SHOW_RINGS 0
SHOW_RINGS 1

MAP←COUNTIES dm∆ShapeLayer∆plot ⍬
MAP dm∆Plot∆save_svg SVG_PATH
PUT NL,'map saved to ',SVG_PATH
dm∆Delete COUNTIES ATTRIBUTES MAP
