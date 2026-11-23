// CMD: just TARGET=stl_ex run-pascal
// C++ std::vector and std::pair values in Pascal: vectors are dynamic arrays, pairs are records.
program stl_ex;

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

const
  N = 100;

type
  TIntPair = record
    First, Second: Integer;
  end;

var
  V: TArrayOfDouble;
  V2: array of Integer;
  P, P2: TIntPair;
  P3: TPairOfDoubleAndDouble;
  I: Integer;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;
  SetLength(V, N);
  for I := 0 to N - 1 do V[I] := I * 1.5;
  for I := 0 to N - 1 do WriteLn(Fmt(V[I]));
  SetLength(V2, N);
  for I := 0 to N - 1 do V2[I] := Floor(I * 1.5);
  for I := 0 to N - 1 do WriteLn(V2[I]);

  P.First := 3; P.Second := 4;
  WriteLn('p: (', P.First, ', ', P.Second, ')');

  P2.First := 10; P2.Second := 20;
  WriteLn('p2: (', P2.First, ', ', P2.Second, ')');

  // A C++ std::pair<double,double> comes back as a TPairOfDoubleAndDouble record.
  P3 := MakeDpair(1.25, 2.75);
  WriteLn('make_dpair(1.25, 2.75): (', Fmt(P3.First), ', ', Fmt(P3.Second), '), sum_dpair = ', Fmt(SumDpair(P3)));
end.
