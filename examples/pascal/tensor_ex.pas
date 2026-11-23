// CMD: just TARGET=tensor_ex run-pascal
// Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
program tensor_ex;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, datamunge;

// C#'s default double formatting: the shortest string that round-trips ("6", "0.1", "1E-05").
function Fmt(AX: Double): String;
var
  FS: TFormatSettings;
  S, Digits: String;
  P, E, I, Prec: Integer;
  D: Double;
  X: Extended;
begin
  if IsNan(AX) then Exit('NaN');
  if IsInfinite(AX) then
    if AX > 0 then Exit('∞') else Exit('-∞');
  if AX = 0 then
    if (PQWord(@AX)^ shr 63) <> 0 then Exit('-0') else Exit('0');
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  for Prec := 15 to 17 do
  begin
    S := FloatToStrF(AX, ffExponent, Prec, 4, FS);
    X := StrToFloat(S, FS);
    if Abs(X) <= MaxDouble then D := X else D := 0;
    if (D = AX) or (Prec = 17) then Break;
  end;
  P := Pos('E', S);
  E := StrToInt(Copy(S, P + 1, Length(S)));
  Digits := '';
  for I := 1 to P - 1 do
    if S[I] in ['0'..'9'] then Digits := Digits + S[I];
  while (Length(Digits) > 1) and (Digits[Length(Digits)] = '0') do
    Delete(Digits, Length(Digits), 1);
  if AX < 0 then Result := '-' else Result := '';
  if (E >= 15) or (E < -4) then
  begin
    Result := Result + Digits[1];
    if Length(Digits) > 1 then Result := Result + '.' + Copy(Digits, 2, Length(Digits));
    if E < 0 then Result := Result + 'E-' else Result := Result + 'E+';
    Result := Result + Format('%.2d', [Abs(E)]);
  end
  else if E < 0 then
    Result := Result + '0.' + StringOfChar('0', -E - 1) + Digits
  else if Length(Digits) <= E + 1 then
    Result := Result + Digits + StringOfChar('0', E + 1 - Length(Digits))
  else
    Result := Result + Copy(Digits, 1, E + 1) + '.' + Copy(Digits, E + 2, Length(Digits));
end;

function ShapeString(ATensor: TTensor): String;
var
  Dims: TArrayOfInt64;
  I: Integer;
begin
  Dims := ATensor.Shape;
  Result := '[';
  for I := 0 to High(Dims) do
  begin
    if I > 0 then Result := Result + ', ';
    Result := Result + IntToStr(Dims[I]);
  end;
  Result := Result + ']';
end;

// The tensor as a string; frees the tensor (every Tensor operation returns a new one).
function Take(ATensor: TTensor): String;
begin
  try
    Result := ATensor.ToString;
  finally
    ATensor.Free;
  end;
end;

var
  R, Arange, Col, Row, A, B, V1, V2, Full, Mask, X, FeatureMeans, Means1x4, Centered, CenteredT,
    Scatter, SpeciesTensor, Setosa, SetosaMask: TTensor;
  Iris: TDataFrame;
  N, I: Int64;
  Column: String;
  Flat: TArrayOfDouble;
  Species: TArrayOfString;
const
  Columns: array[0..3] of String = ('Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width');
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;

  WriteLn('=================== Construction ===================');
  WriteLn('zeros([2,3]): ', Take(TTensor.Zeros([2, 3])));
  WriteLn('eye(3): ', Take(TTensor.Eye(3)));

  // A Tensor* a method returns is not marked owned in the binding policy, so its Pascal object
  // belongs to the receiver: R must not outlive Arange.
  Arange := TTensor.Arange(0.0, 12.0, 1.0);
  R := Arange.Reshape([3, 4]);
  try
    WriteLn('arange(0,12).reshape([3,4]): ', R.ToString);

    WriteLn(#10'=================== Shape ops ===================');
    A := R.Transpose;
    WriteLn('transpose -> shape ', ShapeString(A));
    A.Free;
    WriteLn('slice(axis=1, start=1, stop=3): ', Take(R.Slice(1, 1, 3)));

    WriteLn(#10'=================== Broadcasting arithmetic ===================');
    Col := TTensor.FromValues([3, 1], [1.0, 2.0, 3.0]);
    Row := TTensor.FromValues([1, 4], [10.0, 20.0, 30.0, 40.0]);
    WriteLn('(3,1) + (1,4) -> ', Take(Col.Add(Row)));
    Row.Free;
    Col.Free;

    WriteLn(#10'=================== Reductions ===================');
    WriteLn('r.sum() = ', Fmt(R.Sum), ', r.mean() = ', Fmt(R.Mean));
    WriteLn('column means (axis=0): ', Take(R.MeanAxis(0)));

    WriteLn(#10'=================== Linear algebra ===================');
    A := TTensor.FromValues([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    B := TTensor.FromValues([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    WriteLn('matmul(2x3, 3x2) -> ', Take(A.Matmul(B)));
    B.Free;
    A.Free;
    V1 := TTensor.FromValues([3], [1.0, 2.0, 3.0]);
    V2 := TTensor.FromValues([3], [4.0, 5.0, 6.0]);
    WriteLn('dot([1,2,3], [4,5,6]) = ', Fmt(V1.Dot(V2)));
    WriteLn('outer(v1, v2) -> ', Take(V1.Outer(V2)));
    V2.Free;
    V1.Free;

    WriteLn(#10'=================== Comparisons & masks ===================');
    Full := TTensor.Full([3, 4], 6.0);
    Mask := R.GreaterEqual(Full);
    WriteLn('r >= 6 -> ', Mask.ToString);
    WriteLn('count(r >= 6) = ', Fmt(Mask.Sum));
    Mask.Free;
    Full.Free;
  finally
    R.Free;
    Arange.Free;
  end;

  WriteLn(#10'=================== A real dataset as a Tensor ===================');
  Iris := Data.Iris;
  try
    N := Iris.Nrows;
    Flat := nil;
    for I := 0 to N - 1 do
      for Column in Columns do
        Flat := Concat(Flat, [Iris.NumericAt(Column, I)]);
    X := TTensor.FromValues([N, 4], Flat);
    WriteLn('iris feature tensor shape: ', ShapeString(X));

    FeatureMeans := X.MeanAxis(0);
    Means1x4 := FeatureMeans.Reshape([1, 4]);
    Centered := X.Subtract(Means1x4);
    CenteredT := Centered.Transpose;
    Scatter := CenteredT.Matmul(Centered);
    WriteLn('feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): ', FeatureMeans.ToString);
    WriteLn('(X-mean)^T (X-mean) [4x4 scatter matrix]: ', Scatter.ToString(16));
    Scatter.Free;
    CenteredT.Free;
    Centered.Free;
    Means1x4.Free;
    FeatureMeans.Free;
    X.Free;

    SetLength(Species, N);
    for I := 0 to N - 1 do
      Species[I] := Iris.StringAt('Species', I);
    SpeciesTensor := TTensor.FromStringValues([N], Species);
    Setosa := TTensor.FromStringValues([1], ['setosa']);
    SetosaMask := SpeciesTensor.Equal(Setosa);
    WriteLn('setosa count = ', Fmt(SetosaMask.Sum), ' (of ', N, ' rows)');
    SetosaMask.Free;
    Setosa.Free;
    SpeciesTensor.Free;
  finally
    Iris.Free;
  end;
end.
