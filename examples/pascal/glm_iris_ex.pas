// CMD: just TARGET=glm_iris_ex run-pascal
// Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
program glm_iris_ex;

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
  GridN = 100;

var
  Iris, Sub, Grid, CurveFrame, CountData: TDataFrame;
  Logit, Poisson: TGlm;
  IsVirginica, PetalLength, PetalWidth, Fitted, GridLength, GridWidth: TArrayOfDouble;
  SepalWidthAll, PetalLengthAll, CountAll: TArrayOfDouble;
  Species: String;
  NRows, I, N, Correct: Integer;
  WidthMean, PlMin, PlMax: Double;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;
  Iris := Data.Iris;
  Sub := TDataFrame.Create;
  Grid := TDataFrame.Create;
  CountData := TDataFrame.Create;
  Logit := nil;
  try
    NRows := Iris.Nrows;
    IsVirginica := nil;
    PetalLength := nil;
    PetalWidth := nil;

    // Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
    // the other two on these predictors, which sends logistic regression's coefficients toward
    // +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
    // binary split for a demo.
    for I := 0 to NRows - 1 do
    begin
      Species := Iris.StringAt('Species', I);
      if (Species <> 'versicolor') and (Species <> 'virginica') then Continue;
      N := Length(IsVirginica);
      SetLength(IsVirginica, N + 1);
      SetLength(PetalLength, N + 1);
      SetLength(PetalWidth, N + 1);
      IsVirginica[N] := IfThen(Species = 'virginica', 1.0, 0.0);
      PetalLength[N] := Iris.NumericAt('Petal.Length', I);
      PetalWidth[N] := Iris.NumericAt('Petal.Width', I);
    end;

    Sub.AddNumericColumn('Petal.Length', PetalLength);
    Sub.AddNumericColumn('Petal.Width', PetalWidth);
    Sub.AddNumericColumn('is_virginica', IsVirginica);

    WriteLn('=================== Logistic regression (binomial, logit link) ===================');
    Logit := TGlm.Create(Sub, 'is_virginica ~ Petal.Length + Petal.Width', 'binomial');
    Logit.PrintSummary;

    Fitted := Logit.FittedValues;
    N := Length(IsVirginica);
    Correct := 0;
    for I := 0 to N - 1 do
      if (Fitted[I] >= 0.5) = (IsVirginica[I] >= 0.5) then Inc(Correct);
    WriteLn(#10'Resubstitution accuracy at 0.5 threshold: ', Fmt(Double(100.0) * Correct / N), '%');

    Logit.SaveDiagnosticPlots('glm_logistic_iris');
    WriteLn(#10'Saved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg');

    // Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
    // classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
    WidthMean := Mean(PetalWidth);
    PlMin := MinValue(PetalLength) - 0.3;
    PlMax := MaxValue(PetalLength) + 0.3;
    SetLength(GridLength, GridN);
    SetLength(GridWidth, GridN);
    for I := 0 to GridN - 1 do
    begin
      GridLength[I] := PlMin + (PlMax - PlMin) * I / (GridN - 1);
      GridWidth[I] := WidthMean;
    end;
    Grid.AddNumericColumn('Petal.Length', GridLength);
    Grid.AddNumericColumn('Petal.Width', GridWidth);
    CurveFrame := Logit.PredictFrame(Grid, 'confidence');
    WriteLn(#10'Predicted-probability curve (first 5 rows):');
    WriteLn(CurveFrame.ToString(5));
    CurveFrame.Free;

    // Poisson regression, for contrast: same IRLS engine, different family/link.
    WriteLn(#10'=================== Poisson regression (log link) ===================');
    SetLength(SepalWidthAll, NRows);
    SetLength(PetalLengthAll, NRows);
    SetLength(CountAll, NRows);
    for I := 0 to NRows - 1 do
    begin
      SepalWidthAll[I] := Iris.NumericAt('Sepal.Width', I);
      PetalLengthAll[I] := Iris.NumericAt('Petal.Length', I);
      CountAll[I] := Floor(Iris.NumericAt('Sepal.Length', I) + 0.5);
    end;
    CountData.AddNumericColumn('Sepal.Width', SepalWidthAll);
    CountData.AddNumericColumn('Petal.Length', PetalLengthAll);
    CountData.AddNumericColumn('count', CountAll);
    Poisson := TGlm.Create(CountData, 'count ~ Sepal.Width + Petal.Length', 'poisson');
    Poisson.PrintSummary;
    Poisson.Free;
  finally
    Logit.Free;
    CountData.Free;
    Grid.Free;
    Sub.Free;
    Iris.Free;
  end;
end.
