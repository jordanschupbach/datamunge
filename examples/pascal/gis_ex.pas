// CMD: just TARGET=gis_ex run-pascal
// ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
// Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
// writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
// layout TShapeLayer.Read expects, then reads them back through the public API.
program gis_ex;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, datamunge;

type
  TPt = record
    X, Y: Double;
  end;
  TRing = array of TPt;
  TShape = array of TRing;

const
  BasePath = 'datamunge_gis_ex_counties_pascal';
  SvgPath = 'datamunge_gis_ex_map_pascal.svg';

function Pt(AX, AY: Double): TPt;
begin
  Result.X := AX;
  Result.Y := AY;
end;

// Roughly C's %g: integral doubles print without a fractional part.
function FmtG(AX: Double): String;
var
  FS: TFormatSettings;
begin
  if AX = Floor(AX) then Exit(IntToStr(Trunc(AX)));
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  Result := FloatToStrF(AX, ffFixed, 15, 6, FS);
end;

// Little- and big-endian packers that append to a byte buffer.
procedure PackBytes(var Buf: TBytes; const Src; N: Integer);
var
  L: Integer;
begin
  L := Length(Buf);
  SetLength(Buf, L + N);
  if N > 0 then Move(Src, Buf[L], N);
end;

procedure PackI4Le(var Buf: TBytes; Value: LongInt);
begin
  Value := NtoLE(Value);
  PackBytes(Buf, Value, 4);
end;

procedure PackI4Be(var Buf: TBytes; Value: LongInt);
begin
  Value := NtoBE(Value);
  PackBytes(Buf, Value, 4);
end;

procedure PackI2Le(var Buf: TBytes; Value: SmallInt);
begin
  Value := NtoLE(Value);
  PackBytes(Buf, Value, 2);
end;

procedure PackDoubleLe(var Buf: TBytes; X: Double);
var
  Bits: QWord;
begin
  Move(X, Bits, 8);
  Bits := NtoLE(Bits);
  PackBytes(Buf, Bits, 8);
end;

procedure PackAscii(var Buf: TBytes; const S: String);
begin
  if S <> '' then PackBytes(Buf, S[1], Length(S));
end;

procedure PackZeros(var Buf: TBytes; N: Integer);
var
  L: Integer;
begin
  L := Length(Buf);
  SetLength(Buf, L + N);
  FillChar(Buf[L], N, 0);
end;

procedure PackByte(var Buf: TBytes; B: Byte);
begin
  PackBytes(Buf, B, 1);
end;

procedure WriteAllBytes(const APath: String; const Buf: TBytes);
var
  H: THandle;
begin
  H := FileCreate(APath);
  if H = THandle(-1) then raise EInOutError.Create('cannot create ' + APath);
  try
    if (Length(Buf) > 0) and (FileWrite(H, Buf[0], Length(Buf)) <> Length(Buf)) then
      raise EInOutError.Create('cannot write ' + APath);
  finally
    FileClose(H);
  end;
end;

// Bounding box (minx, miny, maxx, maxy) of every point in rings.
function RingBounds(const Rings: array of TRing): TArrayOfDouble;
var
  R, P: Integer;
  First: Boolean;
begin
  Result := [0, 0, 0, 0];
  First := True;
  for R := 0 to High(Rings) do
    for P := 0 to High(Rings[R]) do
      with Rings[R][P] do
        if First then
        begin
          Result := [X, Y, X, Y];
          First := False;
        end
        else
        begin
          Result[0] := Min(Result[0], X);
          Result[1] := Min(Result[1], Y);
          Result[2] := Max(Result[2], X);
          Result[3] := Max(Result[3], Y);
        end;
end;

// Bytes of one Polygon record's contents (without the record header).
function PolygonRecord(const Rings: TShape): TBytes;
var
  B: Double;
  R, P, Start, Total: Integer;
begin
  Result := nil;
  PackI4Le(Result, 5);  // shape type: Polygon
  for B in RingBounds(Rings) do PackDoubleLe(Result, B);
  PackI4Le(Result, Length(Rings));
  Total := 0;
  for R := 0 to High(Rings) do Inc(Total, Length(Rings[R]));
  PackI4Le(Result, Total);
  Start := 0;
  for R := 0 to High(Rings) do
  begin
    PackI4Le(Result, Start);
    Inc(Start, Length(Rings[R]));
  end;
  for R := 0 to High(Rings) do
    for P := 0 to High(Rings[R]) do
    begin
      PackDoubleLe(Result, Rings[R][P].X);
      PackDoubleLe(Result, Rings[R][P].Y);
    end;
end;

procedure WriteCountiesShp(const APath: String; const Shapes: array of TShape);
var
  Contents: array of TBytes;
  AllRings: array of TRing;
  F: TBytes;
  B: Double;
  I, R, TotalWords: Integer;
begin
  SetLength(Contents, Length(Shapes));
  AllRings := nil;
  TotalWords := 0;
  for I := 0 to High(Shapes) do
  begin
    Contents[I] := PolygonRecord(Shapes[I]);
    Inc(TotalWords, 4 + Length(Contents[I]) div 2);
    for R := 0 to High(Shapes[I]) do
      Insert(Shapes[I][R], AllRings, Length(AllRings));
  end;
  F := nil;
  PackI4Be(F, 9994);
  for I := 0 to 4 do PackI4Be(F, 0);
  PackI4Be(F, 50 + TotalWords);
  PackI4Le(F, 1000);
  PackI4Le(F, 5);  // Polygon
  for B in RingBounds(AllRings) do PackDoubleLe(F, B);
  for I := 0 to 3 do PackDoubleLe(F, 0.0);
  for I := 0 to High(Contents) do
  begin
    PackI4Be(F, I + 1);
    PackI4Be(F, Length(Contents[I]) div 2);
    PackBytes(F, Contents[I][0], Length(Contents[I]));
  end;
  WriteAllBytes(APath, F);
end;

function Ljust(const S: String; N: Integer; Ch: Char): String;
begin
  if Length(S) >= N then Result := Copy(S, 1, N)
  else Result := S + StringOfChar(Ch, N - Length(S));
end;

procedure WriteCountiesDbf(const APath: String; const Names, Populations: array of String);
const
  HeaderSize = 32 + 2 * 32 + 1;
  RecordSize = 1 + 12 + 8;
var
  F: TBytes;
  I: Integer;

  procedure WriteField(const AName, AFieldType: String; ALength: Integer);
  begin
    PackAscii(F, Ljust(AName, 11, #0));
    PackAscii(F, AFieldType);
    PackZeros(F, 4);
    PackByte(F, ALength);
    PackZeros(F, 15);
  end;

begin
  F := TBytes.Create($03, 0, 0, 0);
  PackI4Le(F, Length(Names));
  PackI2Le(F, HeaderSize);
  PackI2Le(F, RecordSize);
  PackZeros(F, 20);
  WriteField('NAME', 'C', 12);
  WriteField('POP', 'N', 8);
  PackByte(F, $0D);
  for I := 0 to Min(High(Names), High(Populations)) do
  begin
    PackAscii(F, ' ');
    PackAscii(F, Ljust(Names[I], 12, ' '));
    PackAscii(F, Ljust(Populations[I], 8, ' '));
  end;
  WriteAllBytes(APath, F);
end;

var
  Plain, WithLake: TShape;
  Counties: TShapeLayer;
  Attrs: TDataFrame;
  Map: TRPlot;
  Bounds: TArrayOfDouble;
  S: String;
  I: Int64;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;

  // Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
  Plain := [[Pt(10, 0), Pt(10, 10), Pt(20, 10), Pt(20, 0), Pt(10, 0)]];
  WithLake := [[Pt(0, 0), Pt(0, 10), Pt(10, 10), Pt(10, 0), Pt(0, 0)],
               [Pt(3, 3), Pt(4, 3), Pt(4, 4), Pt(3, 4), Pt(3, 3)]];

  WriteCountiesShp(BasePath + '.shp', [WithLake, Plain]);
  WriteCountiesDbf(BasePath + '.dbf', ['Lakeside', 'Plainview'], ['48231', '19876']);

  Counties := TShapeLayer.Read(BasePath);
  Attrs := nil;
  try
    WriteLn('shapes = ', Counties.Size, ', shape_type = ', Counties.ShapeType);
    Bounds := Counties.Bounds;
    S := '';
    for I := 0 to High(Bounds) do
    begin
      if I > 0 then S := S + ', ';
      S := S + FmtG(Bounds[I]);
    end;
    WriteLn('bounds = [', S, ']'#10);

    // ShapeLayer::attributes returns a new DataFrame*, which the caller owns.
    Attrs := Counties.Attributes;
    WriteLn('attributes'#10, Attrs.ToString, #10);

    for I := 0 to Counties.Size - 1 do
      WriteLn(Attrs.StringAt('NAME', I), ': ', Counties.NumParts(I), ' ring(s)');

    Map := Counties.Plot;
    try
      Map.SaveSvg(SvgPath);
    finally
      Map.Free;
    end;
    WriteLn(#10'map saved to ', SvgPath);
  finally
    Attrs.Free;
    Counties.Free;
  end;
end.
