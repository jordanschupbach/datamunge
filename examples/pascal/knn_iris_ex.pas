// CMD: just TARGET=knn_iris_ex run-pascal
// k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
program knn_iris_ex;

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
  Ks: array[0..1] of Int64 = (1, 25);

var
  Iris, Table: TDataFrame;
  Model, Knn: TKnnClassifier;
  Fitted: TArrayOfString;
  Plot: TRPlot;
  N, I: Int64;
  Misclassified, J: Integer;
  Actual, Predicted, FileName: String;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;
  Iris := Data.Iris;
  Model := nil;
  try
    WriteLn('iris: ', Iris.Nrows, ' rows x ', Iris.Ncols, ' cols'#10);

    Model := TKnnClassifier.Create(Iris, 'Species ~ Petal.Length + Petal.Width');
    Model.PrintSummary;

    WriteLn(#10'Confusion matrix (leave-one-out; rows = actual, cols = predicted):');
    Table := Model.ConfusionMatrix;
    WriteLn(Table.ToString);
    Table.Free;

    WriteLn(#10'Leave-one-out misclassified rows:');
    Fitted := Model.Predict(Iris);
    N := Iris.Nrows;
    Misclassified := 0;
    for I := 0 to N - 1 do
    begin
      Actual := Iris.StringAt('Species', I);
      Predicted := Fitted[I];
      if Predicted = Actual then Continue;
      Inc(Misclassified);
      WriteLn('  row ', I, ': Petal.Length=', Fmt(Iris.NumericAt('Petal.Length', I)),
              ' Petal.Width=', Fmt(Iris.NumericAt('Petal.Width', I)),
              '  actual=', Actual, '  predicted=', Predicted);
    end;
    WriteLn(Misclassified, ' of ', N, ' misclassified (', Fmt(Double(100.0) * Misclassified / N), '%)');

    Plot := Model.PlotDecisionRegions('Petal.Length', 'Petal.Width');
    Plot.Save('knn_iris_decision_regions_k5.svg');
    Plot.Free;
    WriteLn(#10'Saved knn_iris_decision_regions_k5.svg');

    // k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
    // every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
    // underfit boundary). TKnnClassifier.Create(data, formula, k, metric, weighted, standardize)
    for J := 0 to High(Ks) do
    begin
      Knn := TKnnClassifier.Create(Iris, 'Species ~ Petal.Length + Petal.Width', Ks[J]);
      try
        FileName := Format('knn_iris_decision_regions_k%d.svg', [Ks[J]]);
        WriteLn(#10'k=', Format('%-2d', [Ks[J]]), ' leave-one-out accuracy: ',
                Fmt(Knn.TrainingAccuracy * 100), '%');
        Plot := Knn.PlotDecisionRegions('Petal.Length', 'Petal.Width');
        Plot.Save(FileName);
        Plot.Free;
        WriteLn('Saved ', FileName);
      finally
        Knn.Free;
      end;
    end;
  finally
    Model.Free;
    Iris.Free;
  end;
end.
