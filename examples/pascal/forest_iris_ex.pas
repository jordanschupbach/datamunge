// CMD: just TARGET=forest_iris_ex run-pascal
// Random forest classifier on iris: summary, confusion matrix, misclassifications, OOB accuracy.
program forest_iris_ex;

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

// Saves a plot a model returned by value, then frees it.
procedure SaveAndFree(APlot: TPlot; const APath: String);
begin
  try
    APlot.Save(APath);
  finally
    APlot.Free;
  end;
end;

var
  Iris, Confusion: TDataFrame;
  Model, SmallForest: TRandomForestClassifier;
  Predictions: TArrayOfString;
  Actual: String;
  N, I: Int64;
  Misclassified: Integer;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;
  Iris := Data.Iris;
  Model := nil;
  SmallForest := nil;
  try
    WriteLn('iris: ', Iris.Nrows, ' rows x ', Iris.Ncols, ' cols'#10);

    Model := TRandomForestClassifier.Create(Iris, 'Species ~ Petal.Length + Petal.Width');
    Model.PrintSummary;

    WriteLn(#10'Confusion matrix (rows = actual, cols = predicted):');
    Confusion := Model.ConfusionMatrix;
    WriteLn(Confusion.ToString);
    Confusion.Free;

    WriteLn(#10'Misclassified rows:');
    Predictions := Model.Predict(Iris);
    N := Iris.Nrows;
    Misclassified := 0;
    for I := 0 to N - 1 do
    begin
      Actual := Iris.StringAt('Species', I);
      if Predictions[I] = Actual then Continue;
      Inc(Misclassified);
      WriteLn('  row ', I, ': Petal.Length=', Fmt(Iris.NumericAt('Petal.Length', I)),
              ' Petal.Width=', Fmt(Iris.NumericAt('Petal.Width', I)),
              '  actual=', Actual, '  predicted=', Predictions[I]);
    end;
    WriteLn(Misclassified, ' of ', N, ' misclassified (', Fmt(Double(100.0) * Misclassified / N), '%)');

    SaveAndFree(Model.PlotClassification(Iris, 'Petal.Length', 'Petal.Width'), 'forest_iris_classification.svg');
    SaveAndFree(Model.PlotDecisionRegions('Petal.Length', 'Petal.Width'), 'forest_iris_decision_regions.svg');
    WriteLn(#10'Saved forest_iris_classification.svg and forest_iris_decision_regions.svg');

    // A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
    // are added -- the defining random forest effect a single decision tree cannot demonstrate.
    // TRandomForestClassifier.Create(data, formula, nTrees, maxDepth, minSamplesSplit, minSamplesLeaf,
    //                                maxFeatures, criterion, bootstrap, sampleFraction, seed)
    SmallForest := TRandomForestClassifier.Create(Iris, 'Species ~ Petal.Length + Petal.Width', 5);
    WriteLn(#10'5-tree forest:   training accuracy=', Fmt(SmallForest.TrainingAccuracy * 100),
            '%  OOB accuracy=', Fmt(SmallForest.OobAccuracy * 100), '%');
    WriteLn('100-tree forest: training accuracy=', Fmt(Model.TrainingAccuracy * 100),
            '%  OOB accuracy=', Fmt(Model.OobAccuracy * 100), '%');
    SaveAndFree(SmallForest.PlotDecisionRegions('Petal.Length', 'Petal.Width'), 'forest_iris_decision_regions_5trees.svg');
    WriteLn('Saved forest_iris_decision_regions_5trees.svg');
  finally
    SmallForest.Free;
    Model.Free;
    Iris.Free;
  end;
end.
