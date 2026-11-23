// CMD: just TARGET=hypothesis_tests_ex run-pascal
// Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
program hypothesis_tests_ex;

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

// Prints a test result, and frees it (each test function returns a new result object).
procedure PrintResult(const ALabel: String; R: THypothesisTestResult);
var
  Alternative: String;
begin
  try
    case R.Alternative of
      TAlternative.Less: Alternative := 'less';
      TAlternative.Greater: Alternative := 'greater';
    else
      Alternative := 'two.sided';
    end;
    Write(ALabel, ': statistic=', Fmt(R.Statistic));
    if R.Parameter1 <> 0 then Write(', df1=', Fmt(R.Parameter1));
    if R.Parameter2 <> 0 then Write(', df2=', Fmt(R.Parameter2));
    Write(', p=', Fmt(R.PValue), ' (', Alternative, ')');
    if R.HasConfInt then Write(', CI=[', Fmt(R.ConfIntLower), ', ', Fmt(R.ConfIntUpper), ']');
    WriteLn(' -- ', R.Method);
  finally
    R.Free;
  end;
end;

function ColumnForSpecies(ADf: TDataFrame; const AColumn, ASpecies: String): TArrayOfDouble;
var
  I: Int64;
begin
  Result := nil;
  for I := 0 to ADf.Nrows - 1 do
    if ADf.StringAt('Species', I) = ASpecies then
    begin
      SetLength(Result, Length(Result) + 1);
      Result[High(Result)] := ADf.NumericAt(AColumn, I);
    end;
end;

function Column(ADf: TDataFrame; const AColumn: String): TArrayOfDouble;
var
  I: Int64;
begin
  Result := nil;
  SetLength(Result, ADf.Nrows);
  for I := 0 to ADf.Nrows - 1 do
    Result[I] := ADf.NumericAt(AColumn, I);
end;

// A sorted copy of AV (insertion sort).
function Sorted(const AV: TArrayOfDouble): TArrayOfDouble;
var
  I, J: Integer;
  X: Double;
begin
  Result := Copy(AV);
  for I := 1 to High(Result) do
  begin
    X := Result[I];
    J := I - 1;
    while (J >= 0) and (Result[J] > X) do
    begin
      Result[J + 1] := Result[J];
      Dec(J);
    end;
    Result[J + 1] := X;
  end;
end;

var
  MedianAll: Double;

function CountLong(const AV: TArrayOfDouble): Int64;
var
  X: Double;
begin
  Result := 0;
  for X in AV do
    if X > MedianAll then Inc(Result);
end;

var
  Iris: TDataFrame;
  SetosaPetal, VersicolorPetal, VirginicaPetal, AllPetal, SortedPetal: TArrayOfDouble;
  SepalLength, PetalLength, SetosaSepal: TArrayOfDouble;
  Sizes: TArrayOfInt64;
  SetosaLong, SetosaShort, VersicolorLong, VersicolorShort, LongCount, N: Int64;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;
  Iris := Data.Iris;
  try
    SetosaPetal := ColumnForSpecies(Iris, 'Petal.Length', 'setosa');
    VersicolorPetal := ColumnForSpecies(Iris, 'Petal.Length', 'versicolor');
    VirginicaPetal := ColumnForSpecies(Iris, 'Petal.Length', 'virginica');

    WriteLn('=================== t-tests: petal length, setosa vs. versicolor ===================');
    PrintResult('Welch two-sample t-test', TTestTwoSample(SetosaPetal, VersicolorPetal));
    PrintResult('Wilcoxon rank-sum test', WilcoxonRankSumTest(SetosaPetal, VersicolorPetal));

    WriteLn(#10'=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================');
    AllPetal := Concat(SetosaPetal, VersicolorPetal, VirginicaPetal);
    Sizes := [Length(SetosaPetal), Length(VersicolorPetal), Length(VirginicaPetal)];
    PrintResult('One-way ANOVA', OneWayAnova(AllPetal, Sizes));
    PrintResult('Kruskal-Wallis', KruskalWallisTest(AllPetal, Sizes));

    WriteLn(#10'=================== Correlation: sepal length vs. petal length ===================');
    SepalLength := Column(Iris, 'Sepal.Length');
    PetalLength := Column(Iris, 'Petal.Length');
    PrintResult('Pearson correlation', PearsonCorrelationTest(SepalLength, PetalLength));
    PrintResult('Spearman correlation', SpearmanCorrelationTest(SepalLength, PetalLength));

    WriteLn(#10'=================== F-test: petal length variance, setosa vs. virginica ===================');
    PrintResult('F test', FTestVariance(SetosaPetal, VirginicaPetal));

    WriteLn(#10'=================== Normality: is sepal length normally distributed within setosa? ===================');
    SetosaSepal := ColumnForSpecies(Iris, 'Sepal.Length', 'setosa');
    PrintResult('Shapiro-Francia', ShapiroFranciaTest(SetosaSepal));
    PrintResult('KS vs. fitted normal', KsTestOneSampleNormal(SetosaSepal, Double(5.006), Double(0.3525)));

    WriteLn(#10'=================== Chi-squared / Fisher: is petal length "long" independent of species? ===================');
    SortedPetal := Sorted(AllPetal);
    MedianAll := SortedPetal[Length(AllPetal) div 2];
    SetosaLong := CountLong(SetosaPetal);
    SetosaShort := Length(SetosaPetal) - SetosaLong;
    VersicolorLong := CountLong(VersicolorPetal);
    VersicolorShort := Length(VersicolorPetal) - VersicolorLong;
    WriteLn('table: setosa=[', SetosaLong, ',', SetosaShort, '] versicolor=[', VersicolorLong, ',', VersicolorShort, ']');
    PrintResult('Chi-squared independence',
                ChiSquaredTestIndependence([SetosaLong, SetosaShort, VersicolorLong, VersicolorShort], 2, 2));
    PrintResult('Fisher''s exact test', FisherExactTest2x2(SetosaLong, SetosaShort, VersicolorLong, VersicolorShort));

    WriteLn(#10'=================== Proportion / binomial: fraction of "long" petals overall ===================');
    LongCount := CountLong(AllPetal);
    N := Length(AllPetal);
    PrintResult('One-sample proportion test (vs 0.5)', ProportionTestOneSample(LongCount, N, 0.5));
    PrintResult('Exact binomial test (vs 0.5)', BinomialTest(LongCount, N, 0.5));
  finally
    Iris.Free;
  end;
end.
