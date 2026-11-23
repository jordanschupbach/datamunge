// CMD: just TARGET=svm_iris_ex run-pascal
// Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
program svm_iris_ex;

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
  Formula = 'Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width';

var
  Iris, NewData, Table: TDataFrame;
  RbfModel, LinearModel: TSvm;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;
  Iris := Data.Iris;
  RbfModel := nil;
  LinearModel := nil;
  NewData := TDataFrame.Create;
  try
    WriteLn('iris: ', Iris.Nrows, ' rows x ', Iris.Ncols, ' cols'#10);

    WriteLn('=== RBF kernel (default) ===');
    RbfModel := TSvm.Create(Iris, Formula);
    RbfModel.PrintSummary;
    Table := RbfModel.ConfusionMatrix;
    WriteLn(#10'Confusion matrix (rows = actual, cols = predicted):'#10, Table.ToString);
    Table.Free;

    WriteLn(#10'=== Linear kernel, for comparison ===');
    LinearModel := TSvm.Create(Iris, Formula, 'linear');
    WriteLn('Training accuracy: ', Fmt(LinearModel.TrainingAccuracy * 100), '%');
    WriteLn('Support vectors: ', LinearModel.NumSupportVectors);

    NewData.AddNumericColumn('Sepal.Length', [5.1, 6.0, 6.5, 6.2]);
    NewData.AddNumericColumn('Sepal.Width', [3.5, 2.7, 3.0, 2.8]);
    NewData.AddNumericColumn('Petal.Length', [1.4, 4.5, 5.5, 4.8]);
    NewData.AddNumericColumn('Petal.Width', [0.2, 1.5, 2.0, 1.8]);
    Table := RbfModel.PredictFrame(NewData);
    WriteLn(#10'RBF predictions for new flowers (votes out of 3 one-vs-one pairs):'#10, Table.ToString);
    Table.Free;
  finally
    NewData.Free;
    LinearModel.Free;
    RbfModel.Free;
    Iris.Free;
  end;
end.
