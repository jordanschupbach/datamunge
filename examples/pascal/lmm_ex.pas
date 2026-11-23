// CMD: just TARGET=lmm_ex run-pascal
// Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
program lmm_ex;

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

type
  // .NET's System.Random(seed) (the seeded Knuth subtractive generator), so the simulated data
  // match the C# example exactly.
  TNetRandom = class
  private
    FSeedArray: array[0..55] of Int32;
    FNext, FNextP: Integer;
    function InternalSample: Int32;
  public
    constructor Create(ASeed: Int32);
    function NextDouble: Double;
    function Next(AMinValue, AMaxValue: Int32): Int32;
  end;

constructor TNetRandom.Create(ASeed: Int32);
const
  MBig = High(Int32);
  MSeed = 161803398;
var
  Mj, Mk, I, Ii, K: Int32;
begin
  inherited Create;
  if ASeed = Low(Int32) then Mj := MSeed - MBig else Mj := MSeed - Abs(ASeed);
  FSeedArray[55] := Mj;
  Mk := 1;
  for I := 1 to 54 do
  begin
    Ii := (21 * I) mod 55;
    FSeedArray[Ii] := Mk;
    Mk := Mj - Mk;
    if Mk < 0 then Inc(Mk, MBig);
    Mj := FSeedArray[Ii];
  end;
  for K := 1 to 4 do
    for I := 1 to 55 do
    begin
      Dec(FSeedArray[I], FSeedArray[1 + (I + 30) mod 55]);
      if FSeedArray[I] < 0 then Inc(FSeedArray[I], MBig);
    end;
  FNext := 0;
  FNextP := 21;
end;

function TNetRandom.InternalSample: Int32;
begin
  Inc(FNext);
  if FNext >= 56 then FNext := 1;
  Inc(FNextP);
  if FNextP >= 56 then FNextP := 1;
  Result := FSeedArray[FNext] - FSeedArray[FNextP];
  if Result = High(Int32) then Dec(Result);
  if Result < 0 then Inc(Result, High(Int32));
  FSeedArray[FNext] := Result;
end;

function TNetRandom.NextDouble: Double;
var
  Scale: Double;
begin
  Scale := 1;
  Scale := Scale / High(Int32);
  Result := InternalSample * Scale;
end;

function TNetRandom.Next(AMinValue, AMaxValue: Int32): Int32;
begin
  Result := Trunc(NextDouble * (Int64(AMaxValue) - AMinValue)) + AMinValue;
end;

// The C library's log and cos, which .NET's Math.Log and Math.Cos call on Linux.
function CLog(AX: Double): Double; cdecl; external 'm' name 'log';
function CCos(AX: Double): Double; cdecl; external 'm' name 'cos';

var
  Rng: TNetRandom;

// (0, 1]: avoids log(0) in the Box-Muller transform.
function Uniform: Double;
begin
  Result := 1 - Rng.NextDouble;
end;

function Gauss: Double;
var
  U1, U2, R, Theta: Double;
begin
  U1 := Uniform;
  R := CLog(U1);
  R := Sqrt(-2 * R);
  U2 := Uniform;
  Theta := 2 * Double(Pi) * U2;
  Result := R * CCos(Theta);
end;

const
  NSchools = 30;
  TrueIntercept: Double = 60.0;
  TrueSlope: Double = 3.0;

var
  Penguins, Df, Population, School0: TDataFrame;
  PenguinModel, Model: TLmm;
  SchoolIntercept, SchoolSlope: array[0..NSchools - 1] of Double;
  School, StudyHours, Score, Re: TArrayOfDouble;
  Labels: TArrayOfString;
  S, Student, NStudents, N, Idx: Integer;
  Hours: Double;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;
  Rng := TNetRandom.Create(2024);
  Df := TDataFrame.Create;
  Population := TDataFrame.Create;
  School0 := TDataFrame.Create;
  Model := nil;
  try
    WriteLn('=================== Random intercept on a real dataset (penguins) ===================');
    Penguins := Data.Penguins;
    PenguinModel := TLmm.Create(Penguins, 'body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)');
    PenguinModel.PrintSummary;
    PenguinModel.Free;
    Penguins.Free;

    WriteLn(#10'=================== Random intercept + slope on a simulated multi-school dataset ===================');
    // 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
    // and the return on an extra hour of study vary by school -- the textbook case for a
    // random-intercept-and-slope model.
    for S := 0 to NSchools - 1 do
      SchoolIntercept[S] := Gauss * 6;
    for S := 0 to NSchools - 1 do
      SchoolSlope[S] := Gauss * Double(1.2);

    School := nil;
    StudyHours := nil;
    Score := nil;
    for S := 0 to NSchools - 1 do
    begin
      NStudents := 15 + Rng.Next(0, 21);
      for Student := 0 to NStudents - 1 do
      begin
        Hours := Rng.NextDouble * 10;
        N := Length(School);
        SetLength(School, N + 1);
        SetLength(StudyHours, N + 1);
        SetLength(Score, N + 1);
        School[N] := S;
        StudyHours[N] := Hours;
        Score[N] := TrueIntercept + SchoolIntercept[S] + (TrueSlope + SchoolSlope[S]) * Hours + Gauss * 4;
      end;
    end;
    Df.AddNumericColumn('school', School);
    Df.AddNumericColumn('study_hours', StudyHours);
    Df.AddNumericColumn('score', Score);

    Model := TLmm.Create(Df, 'score ~ study_hours + (1 + study_hours | school)');
    Model.PrintSummary;

    WriteLn(#10'True generating values: intercept=', Fmt(TrueIntercept), ', slope=', Fmt(TrueSlope),
            ', random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0');

    WriteLn(#10'--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---');
    Labels := Model.GroupLabels;
    for Idx := 0 to 2 do
    begin
      Re := Model.RandomEffectsForGroup(Idx);
      WriteLn('school ', Labels[Idx], ': intercept shift=', Fmt(Re[0]), ', slope shift=', Fmt(Re[1]));
    end;

    WriteLn(#10'--- Prediction: population-level vs. school-adjusted ---');
    Population.AddNumericColumn('study_hours', [5.0]);
    School0.AddNumericColumn('study_hours', [5.0]);
    School0.AddNumericColumn('school', [0.0]);
    WriteLn('5 study hours, unseen school:      ', Fmt(Model.Predict(Population)[0]), ' (fixed effects only)');
    WriteLn('5 study hours, school 0 (known):    ', Fmt(Model.Predict(School0)[0]), ' (fixed effects + school 0''s BLUP)');
  finally
    Model.Free;
    School0.Free;
    Population.Free;
    Df.Free;
    Rng.Free;
  end;
end.
