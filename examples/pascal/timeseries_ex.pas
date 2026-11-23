// CMD: just TARGET=timeseries_ex run-pascal
// Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
program timeseries_ex;

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

// The C library's double-precision math (what .NET calls), so the simulated data matches the
// C# example bit for bit; FPC's own Sin/Cos/Ln compute in 80-bit extended precision.
function CCos(AX: Double): Double; cdecl; external 'm' name 'cos';
function CSin(AX: Double): Double; cdecl; external 'm' name 'sin';
function CLog(AX: Double): Double; cdecl; external 'm' name 'log';
function CSqrt(AX: Double): Double; cdecl; external 'm' name 'sqrt';

// FPC evaluates untyped real constants such as 0.3 in extended precision, so the constants of
// the simulations are typed Double, as in C#.
const
  TwoPi: Double = 2 * 3.141592653589793;  // 2 * Math.PI in C#
  Drift: Double = 0.3;
  MaWeight: Double = 0.4;
  ArWeight: Double = 0.5;
  Slope: Double = 0.2;

// .NET's seeded System.Random (Knuth's subtractive generator), so the data matches C#'s new Random(n).
var
  SeedArray: array[0..55] of LongInt;
  INext, INextP: Integer;

procedure Seed(ASeed: LongInt);
const
  MBig = High(LongInt);
  MSeed = 161803398;
var
  Mj, Mk, I, Ii, K, N: LongInt;
begin
  if ASeed = Low(LongInt) then Mj := MSeed - MBig else Mj := MSeed - Abs(ASeed);
  SeedArray[55] := Mj;
  Mk := 1;
  for I := 1 to 54 do
  begin
    Ii := (21 * I) mod 55;
    SeedArray[Ii] := Mk;
    Mk := Mj - Mk;
    if Mk < 0 then Inc(Mk, MBig);
    Mj := SeedArray[Ii];
  end;
  for K := 1 to 4 do
    for I := 1 to 55 do
    begin
      N := I + 30;
      if N >= 55 then Dec(N, 55);
      SeedArray[I] := SeedArray[I] - SeedArray[1 + N];
      if SeedArray[I] < 0 then Inc(SeedArray[I], MBig);
    end;
  INext := 0;
  INextP := 21;
end;

function NextDouble: Double;
const
  MBig = High(LongInt);
  InvMBig: Double = 1 / Double(High(LongInt));  // 1.0 / MBig would be folded in Single precision
var
  Ret: LongInt;
begin
  Inc(INext);
  if INext >= 56 then INext := 1;
  Inc(INextP);
  if INextP >= 56 then INextP := 1;
  Ret := SeedArray[INext] - SeedArray[INextP];
  if Ret = MBig then Dec(Ret);
  if Ret < 0 then Inc(Ret, MBig);
  SeedArray[INext] := Ret;
  Result := Ret * InvMBig;
end;

// (0, 1]: avoids log(0) in the Box-Muller transform.
function Uniform: Double;
begin
  Result := 1 - NextDouble;
end;

function Gauss: Double;
var
  R, C: Double;
begin
  R := CSqrt(-2 * CLog(Uniform));
  C := CCos(TwoPi * Uniform);
  Result := R * C;
end;

function Joined(const AValues: TArrayOfDouble): String;
var
  I: Integer;
begin
  Result := '';
  for I := 0 to High(AValues) do
  begin
    if I > 0 then Result := Result + ', ';
    Result := Result + Fmt(AValues[I]);
  end;
end;

const
  SeasonalShape: array[0..11] of Double = (0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6);
var
  Y, S, Y2: TArrayOfDouble;
  Level, PrevShock, Shock, Prev, Term: Double;
  I: Integer;
  Options: TArimaOptions;
  Model, Model2: TArima;
  Intervals: TPairOfArrayOfDoubleAndArrayOfDouble;
  EsOptions: TExponentialSmoothingOptions;
  EsModel, Ses, Holt: TExponentialSmoothing;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;

  WriteLn('=================== ARIMA(1,1,1) on a simulated random walk with drift ===================');
  Seed(42);
  SetLength(Y, 150);
  Level := 100.0;
  PrevShock := 0.0;
  for I := 0 to 149 do
  begin
    Shock := Gauss;
    Level := Level + (Drift + MaWeight * PrevShock + Shock);  // integrated ARMA(0,1) increments with drift
    Y[I] := Level;
    PrevShock := Shock;
  end;

  Options := TArimaOptions.Create;
  try
    Options.P := 1;
    Options.D := 1;
    Options.Q := 1;
    Options.DePopulationSize := 80;
    Options.DeMaxGenerations := 400;
    Model := TArima.Create(Y, Options);
  finally
    Options.Free;
  end;
  try
    WriteLn('AR coefficient: ', Fmt(Model.ArCoefficients[0]));
    WriteLn('MA coefficient: ', Fmt(Model.MaCoefficients[0]));
    WriteLn('sigma^2: ', Fmt(Model.Sigma2), ', AIC: ', Fmt(Model.Aic), ', BIC: ', Fmt(Model.Bic));

    Intervals := Model.ForecastWithIntervals(6);
    WriteLn('6-step forecast: ', Joined(Intervals.First));
    WriteLn('forecast std. errors: ', Joined(Intervals.Second));
  finally
    Model.Free;
  end;

  WriteLn(#10'=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================');
  Seed(7);
  SetLength(S, 120);
  Prev := 0.0;
  for I := 0 to 119 do
  begin
    Prev := ArWeight * Prev + Gauss;
    Term := 5 * CSin(TwoPi * I / 12);
    S[I] := 20 + Slope * I + Term + Prev;
  end;

  Options := TArimaOptions.Create;
  try
    Options.P := 1;
    Options.SeasonalP := 1;
    Options.SeasonalD := 1;
    Options.SeasonalPeriod := 12;
    Options.DePopulationSize := 100;
    Options.DeMaxGenerations := 500;
    Model2 := TArima.Create(S, Options);
  finally
    Options.Free;
  end;
  try
    WriteLn('AR coefficient: ', Fmt(Model2.ArCoefficients[0]),
            ', seasonal AR coefficient: ', Fmt(Model2.SeasonalArCoefficients[0]));
    WriteLn('12-step forecast: ', Joined(Model2.Forecast(12)));
  finally
    Model2.Free;
  end;

  WriteLn(#10'=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================');
  Seed(11);
  SetLength(Y2, 48);
  for I := 0 to 47 do
    Y2[I] := (100 + 2 * I) * SeasonalShape[I mod 12] + Gauss * 3;

  EsOptions := TExponentialSmoothingOptions.Create;
  try
    EsOptions.Trend := TTrendType.Additive;
    EsOptions.Seasonal := TSeasonalType.Multiplicative;
    EsOptions.SeasonalPeriod := 12;
    EsOptions.DePopulationSize := 60;
    EsOptions.DeMaxGenerations := 300;
    EsModel := TExponentialSmoothing.Create(Y2, EsOptions);
  finally
    EsOptions.Free;
  end;
  try
    WriteLn('alpha=', Fmt(EsModel.Alpha), ' beta=', Fmt(EsModel.Beta), ' gamma=', Fmt(EsModel.Gamma));
    WriteLn('sigma^2: ', Fmt(EsModel.Sigma2), ', AIC: ', Fmt(EsModel.Aic));
    WriteLn('12-month forecast: ', Joined(EsModel.Forecast(12)));
  finally
    EsModel.Free;
  end;

  WriteLn(#10'=================== Simple exponential smoothing vs. Holt''s linear trend ===================');
  EsOptions := TExponentialSmoothingOptions.Create;
  Ses := nil;
  Holt := nil;
  try
    Ses := TExponentialSmoothing.Create([50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7], EsOptions);
    WriteLn('SES alpha= ', Fmt(Ses.Alpha));
    WriteLn('SES 5-step forecast: ', Joined(Ses.Forecast(5)));

    EsOptions.Trend := TTrendType.Additive;
    Holt := TExponentialSmoothing.Create([50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7], EsOptions);
    WriteLn('Holt alpha=', Fmt(Holt.Alpha), ' beta=', Fmt(Holt.Beta));
    WriteLn('Holt 5-step forecast: ', Joined(Holt.Forecast(5)));
  finally
    Holt.Free;
    Ses.Free;
    EsOptions.Free;
  end;
end.
