// CMD: just TARGET=bspline_ex run-pascal
// Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
program bspline_ex;

{$mode objfpc}{$H+}

uses
  SysUtils, datamunge;

// C#'s F6 format: six decimals with a '.' separator.
function FmtF6(AX: Double): String;
var
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  Result := FloatToStrF(AX, ffFixed, 15, 6, FS);
end;

var
  X, Y, Z, Pred: TArrayOfDouble;
  IX, IY, K: Integer;
  XV, YV, SinX: Double;
  Data, NewPoints: TDataFrame;
  Surface: TLm;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;
  // (Double(7.0): FPC types the literal 7.0 as Single, which would make IX / 7.0 a Single.)
  // Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
  SetLength(X, 64);
  SetLength(Y, 64);
  SetLength(Z, 64);
  K := 0;
  for IY := 0 to 7 do
    for IX := 0 to 7 do
    begin
      XV := IX / Double(7.0);
      YV := IY / Double(7.0);
      X[K] := XV;
      Y[K] := YV;
      SinX := Sin(XV);  // Sin returns Extended: round it to Double first, as C# does
      Z[K] := SinX + YV * YV;
      Inc(K);
    end;
  Data := TDataFrame.Create;
  NewPoints := TDataFrame.Create;
  Surface := nil;
  try
    Data.AddNumericColumn('x', X);
    Data.AddNumericColumn('y', Y);
    Data.AddNumericColumn('z', Z);

    Surface := TLm.Create(Data, 'z ~ bs(x, y)');
    WriteLn('Fitted z ~ bs(x, y)');
    Surface.PrintSummary;

    NewPoints.AddNumericColumn('x', [0.25, 0.75]);
    NewPoints.AddNumericColumn('y', [0.50, 0.25]);
    Pred := Surface.Predict(NewPoints);
    WriteLn('Predictions: [', FmtF6(Pred[0]), ', ', FmtF6(Pred[1]), ']');
  finally
    Surface.Free;
    NewPoints.Free;
    Data.Free;
  end;
end.
