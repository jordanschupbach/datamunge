--  CMD: just TARGET=gis_ex run-ada
--  ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
--  Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
--  writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
--  layout Shape_Layer_Read expects, then reads them back through the public API.
with Ada.Containers.Vectors;
with Ada.Streams.Stream_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Unchecked_Conversion;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;
with Interfaces; use Interfaces;

procedure Gis_Ex is
   use Ada.Streams;

   type Pt is record
      X, Y : Long_Float;
   end record;

   package Ring_Vectors is new Ada.Containers.Vectors (Positive, Pt);
   subtype Ring is Ring_Vectors.Vector;
   package Shape_Vectors is new Ada.Containers.Vectors (Positive, Ring, Ring_Vectors."=");
   subtype Shape is Shape_Vectors.Vector;
   package Shape_Lists is new Ada.Containers.Vectors (Positive, Shape, Shape_Vectors."=");
   package Byte_Vectors is new Ada.Containers.Vectors (Positive, Stream_Element);
   subtype Bytes is Byte_Vectors.Vector;

   Base_Path : constant String := "datamunge_gis_ex_counties_ada";
   Svg_Path  : constant String := "datamunge_gis_ex_map_ada.svg";

   --  Roughly C's %g: integral doubles print without a fractional part.
   function Fmt_G (X : Long_Float) return String is
     ((if X = Long_Float'Floor (X) then Img (Long_Long_Integer (X)) else F (X, 6)));

   --  Little- and big-endian packers that append to a byte buffer.
   procedure Pack_Le (Buf : in out Bytes; Value : Unsigned_64; Size : Positive) is
   begin
      for I in 0 .. Size - 1 loop
         Buf.Append (Stream_Element (Shift_Right (Value, 8 * I) and 16#FF#));
      end loop;
   end Pack_Le;

   procedure Pack_Be (Buf : in out Bytes; Value : Unsigned_64; Size : Positive) is
   begin
      for I in reverse 0 .. Size - 1 loop
         Buf.Append (Stream_Element (Shift_Right (Value, 8 * I) and 16#FF#));
      end loop;
   end Pack_Be;

   function U32 is new Ada.Unchecked_Conversion (Integer_32, Unsigned_32);
   function U16 is new Ada.Unchecked_Conversion (Integer_16, Unsigned_16);
   function Bits is new Ada.Unchecked_Conversion (Long_Float, Unsigned_64);

   procedure Pack_I4_Le (Buf : in out Bytes; Value : Integer) is
   begin
      Pack_Le (Buf, Unsigned_64 (U32 (Integer_32 (Value))), 4);
   end Pack_I4_Le;

   procedure Pack_I4_Be (Buf : in out Bytes; Value : Integer) is
   begin
      Pack_Be (Buf, Unsigned_64 (U32 (Integer_32 (Value))), 4);
   end Pack_I4_Be;

   procedure Pack_I2_Le (Buf : in out Bytes; Value : Integer) is
   begin
      Pack_Le (Buf, Unsigned_64 (U16 (Integer_16 (Value))), 2);
   end Pack_I2_Le;

   procedure Pack_Double_Le (Buf : in out Bytes; X : Long_Float) is
   begin
      Pack_Le (Buf, Bits (X), 8);
   end Pack_Double_Le;

   procedure Pack_Ascii (Buf : in out Bytes; S : String) is
   begin
      for C of S loop
         Buf.Append (Stream_Element (Character'Pos (C)));
      end loop;
   end Pack_Ascii;

   procedure Pack_Zeros (Buf : in out Bytes; N : Natural) is
   begin
      Buf.Append (0, Ada.Containers.Count_Type (N));
   end Pack_Zeros;

   procedure Write_All_Bytes (Path : String; Buf : Bytes) is
      File : Ada.Streams.Stream_IO.File_Type;
      Data : Stream_Element_Array (1 .. Stream_Element_Offset (Buf.Length));
   begin
      for I in Data'Range loop
         Data (I) := Buf (Positive (I));
      end loop;
      Ada.Streams.Stream_IO.Create (File, Ada.Streams.Stream_IO.Out_File, Path);
      Ada.Streams.Stream_IO.Write (File, Data);
      Ada.Streams.Stream_IO.Close (File);
   end Write_All_Bytes;

   type Box is array (1 .. 4) of Long_Float;

   --  Bounding box (minx, miny, maxx, maxy) of every point in rings.
   function Ring_Bounds (Rings : Shape) return Box is
      Result : Box := [0.0, 0.0, 0.0, 0.0];
      First  : Boolean := True;
   begin
      for R of Rings loop
         for P of R loop
            if First then
               Result := [P.X, P.Y, P.X, P.Y];
               First := False;
            else
               Result := [Long_Float'Min (Result (1), P.X), Long_Float'Min (Result (2), P.Y),
                          Long_Float'Max (Result (3), P.X), Long_Float'Max (Result (4), P.Y)];
            end if;
         end loop;
      end loop;
      return Result;
   end Ring_Bounds;

   --  Bytes of one Polygon record's contents (without the record header).
   function Polygon_Record (Rings : Shape) return Bytes is
      Output : Bytes;
      Start, Total : Natural := 0;
   begin
      Pack_I4_Le (Output, 5);  --  shape type: Polygon
      for B of Ring_Bounds (Rings) loop
         Pack_Double_Le (Output, B);
      end loop;
      Pack_I4_Le (Output, Natural (Rings.Length));
      for R of Rings loop
         Total := Total + Natural (R.Length);
      end loop;
      Pack_I4_Le (Output, Total);
      for R of Rings loop
         Pack_I4_Le (Output, Start);
         Start := Start + Natural (R.Length);
      end loop;
      for R of Rings loop
         for P of R loop
            Pack_Double_Le (Output, P.X);
            Pack_Double_Le (Output, P.Y);
         end loop;
      end loop;
      return Output;
   end Polygon_Record;

   procedure Write_Counties_Shp (Path : String; Shapes : Shape_Lists.Vector) is
      package Bytes_Lists is new Ada.Containers.Vectors (Positive, Bytes, Byte_Vectors."=");
      Contents    : Bytes_Lists.Vector;
      All_Rings   : Shape;
      F           : Bytes;
      Total_Words : Natural := 0;
   begin
      for S of Shapes loop
         Contents.Append (Polygon_Record (S));
         Total_Words := Total_Words + 4 + Natural (Contents.Last_Element.Length) / 2;
         All_Rings.Append_Vector (S);
      end loop;
      Pack_I4_Be (F, 9994);
      for I in 1 .. 5 loop
         Pack_I4_Be (F, 0);
      end loop;
      Pack_I4_Be (F, 50 + Total_Words);
      Pack_I4_Le (F, 1000);
      Pack_I4_Le (F, 5);  --  Polygon
      for B of Ring_Bounds (All_Rings) loop
         Pack_Double_Le (F, B);
      end loop;
      for I in 1 .. 4 loop
         Pack_Double_Le (F, 0.0);
      end loop;
      for I in Contents.First_Index .. Contents.Last_Index loop
         Pack_I4_Be (F, I);
         Pack_I4_Be (F, Natural (Contents (I).Length) / 2);
         F.Append_Vector (Contents (I));
      end loop;
      Write_All_Bytes (Path, F);
   end Write_Counties_Shp;

   function Ljust (S : String; N : Natural; Ch : Character) return String is
     ((if S'Length >= N then S (S'First .. S'First + N - 1) else S & [1 .. N - S'Length => Ch]));

   procedure Write_Counties_Dbf (Path : String; Names, Populations : String_Vector) is
      Header_Size : constant := 32 + 2 * 32 + 1;
      Record_Size : constant := 1 + 12 + 8;
      F : Bytes := [16#03#, 0, 0, 0];

      procedure Write_Field (Name, Field_Type : String; Length : Natural) is
      begin
         Pack_Ascii (F, Ljust (Name, 11, ASCII.NUL));
         Pack_Ascii (F, Field_Type);
         Pack_Zeros (F, 4);
         F.Append (Stream_Element (Length));
         Pack_Zeros (F, 15);
      end Write_Field;
   begin
      Pack_I4_Le (F, Natural (Names.Length));
      Pack_I2_Le (F, Header_Size);
      Pack_I2_Le (F, Record_Size);
      Pack_Zeros (F, 20);
      Write_Field ("NAME", "C", 12);
      Write_Field ("POP", "N", 8);
      F.Append (16#0D#);
      for I in 1 .. Natural'Min (Names.Last_Index, Populations.Last_Index) loop
         Pack_Ascii (F, " ");
         Pack_Ascii (F, Ljust (Names (I), 12, ' '));
         Pack_Ascii (F, Ljust (Populations (I), 8, ' '));
      end loop;
      Write_All_Bytes (Path, F);
   end Write_Counties_Dbf;

   --  Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
   Plain     : constant Shape := [[(10.0, 0.0), (10.0, 10.0), (20.0, 10.0), (20.0, 0.0), (10.0, 0.0)]];
   With_Lake : constant Shape :=
     [[(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0), (0.0, 0.0)],
      [(3.0, 3.0), (4.0, 3.0), (4.0, 4.0), (3.0, 4.0), (3.0, 3.0)]];

   Counties : Shape_Layer;
   Attrs    : Data_Frame;
   Map      : R_Plot;
   Bounds   : Unbounded_String;
begin
   Write_Counties_Shp (Base_Path & ".shp", [With_Lake, Plain]);
   Write_Counties_Dbf (Base_Path & ".dbf", ["Lakeside", "Plainview"], ["48231", "19876"]);

   Counties := Shape_Layer_Read (Base_Path);

   Put_Line ("shapes = " & Img (Counties.Size) & ", shape_type = " & Counties.Shape_Type);
   for B of Counties.Bounds loop
      if Length (Bounds) > 0 then
         Append (Bounds, ", ");
      end if;
      Append (Bounds, Fmt_G (B));
   end loop;
   Put_Line ("bounds = [" & To_String (Bounds) & "]" & ASCII.LF);

   Attrs := Counties.Attributes;
   Put_Line ("attributes" & ASCII.LF & Attrs.Image & ASCII.LF);

   for I in 0 .. Counties.Size - 1 loop
      Put_Line (Attrs.String_At ("NAME", I) & ": " & Img (Counties.Num_Parts (I)) & " ring(s)");
   end loop;

   --  ShapeLayer::plot is Plot_2 in Ada (the name Plot is the type of datamunge::graphics::Plot).
   Map := Counties.Plot_2;
   Map.Save_Svg (Svg_Path);
   Free (Map);
   Put_Line (ASCII.LF & "map saved to " & Svg_Path);

   Free (Attrs);
   Free (Counties);
end Gis_Ex;
