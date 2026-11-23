// CMD: just TARGET=naive_bayes_penguins_ex run-pascal
// Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
program naive_bayes_penguins_ex;

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
  Columns: array[0..4] of String = ('species', 'bill_length_mm', 'bill_depth_mm', 'island', 'sex');

function AnyNull(AFrame: TDataFrame; ARow: Int64): Boolean;
var
  Column: String;
begin
  for Column in Columns do
    if AFrame.IsNull(Column, ARow) then Exit(True);
  Result := False;
end;

var
  Penguins, Confusion: TDataFrame;
  Model, BillOnly: TNaiveBayesClassifier;
  Predictions: TArrayOfString;
  Plot: TRPlot;
  Misclassified: Integer;
  I: Int64;
  Actual, Predicted: String;
  Accuracy: Double;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;

  // Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
  // when fitting, matching every other formula-based model in this library.
  Penguins := Data.Penguins;
  Model := nil;
  BillOnly := nil;
  try
    WriteLn('penguins: ', Penguins.Nrows, ' rows x ', Penguins.Ncols, ' cols'#10);

    // A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
    // predictors in one formula -- each modeled independently given the class, per the naive
    // Bayes assumption.
    Model := TNaiveBayesClassifier.Create(Penguins, 'species ~ bill_length_mm + bill_depth_mm + island + sex');
    Model.PrintSummary;

    Confusion := Model.ConfusionMatrix;
    WriteLn(#10'Confusion matrix (rows = actual, cols = predicted):'#10, Confusion.ToString);
    Confusion.Free;

    WriteLn(#10'Misclassified rows:');
    Predictions := Model.Predict(Penguins);
    Misclassified := 0;
    for I := 0 to Penguins.Nrows - 1 do
    begin
      if AnyNull(Penguins, I) then Continue;
      Actual := Penguins.StringAt('species', I);
      Predicted := Predictions[I];
      if Predicted = Actual then Continue;
      Inc(Misclassified);
      WriteLn('  row ', I, ': bill_length=', Fmt(Penguins.NumericAt('bill_length_mm', I)),
              ' bill_depth=', Fmt(Penguins.NumericAt('bill_depth_mm', I)),
              ' island=', Penguins.StringAt('island', I), ' sex=', Penguins.StringAt('sex', I),
              '  actual=', Actual, '  predicted=', Predicted);
    end;
    WriteLn(Misclassified, ' misclassified (of ', Penguins.Nrows, ' rows, some incomplete)');

    // PlotDecisionRegions requires exactly two NUMERIC predictors, so build a separate
    // two-predictor model (bill measurements alone) just for visualization.
    BillOnly := TNaiveBayesClassifier.Create(Penguins, 'species ~ bill_length_mm + bill_depth_mm');
    Accuracy := 100 * BillOnly.TrainingAccuracy;
    WriteLn(#10'bill-measurements-only model training accuracy: ', Fmt(Accuracy), '%');
    Plot := BillOnly.PlotClassification(Penguins, 'bill_length_mm', 'bill_depth_mm');
    Plot.Save('naive_bayes_penguins_classification.svg');
    Plot.Free;
    Plot := BillOnly.PlotDecisionRegions('bill_length_mm', 'bill_depth_mm');
    Plot.Save('naive_bayes_penguins_decision_regions.svg');
    Plot.Free;
    WriteLn('Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg');
  finally
    BillOnly.Free;
    Model.Free;
    Penguins.Free;
  end;
end.
