// CMD: just TARGET=kernel_regression_iris_ex run-pascal
// Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
program kernel_regression_iris_ex;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, datamunge;

const
  Formula = 'Petal.Length ~ Petal.Width';

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
  Iris: TDataFrame;
  Model, Small, Large: TKernelRegression;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;
  Iris := Data.Iris;
  Model := nil;
  Small := nil;
  Large := nil;
  try
    WriteLn('iris: ', Iris.Nrows, ' rows x ', Iris.Ncols, ' cols');
    WriteLn('formula: ', Formula, #10);

    Model := TKernelRegression.Create(Iris, Formula);  // bandwidth auto-selected via leave-one-out CV
    Model.PrintSummary;

    SaveAndFree(Model.PlotFit(Iris), 'kernel_regression_iris_fit.svg');
    SaveAndFree(Model.PlotCvCurve, 'kernel_regression_iris_cv.svg');
    WriteLn(#10'Saved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg');

    // A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
    // bias-variance story. TKernelRegression.Create(data, formula, kernel, bandwidth, nBandwidth, standardize)
    Small := TKernelRegression.Create(Iris, Formula, 'gaussian', 0.05);
    WriteLn(#10'bandwidth=0.05 (too small): LOO R-squared=', Fmt(Small.RSquared), '  LOO RMSE=', Fmt(Small.Rmse));
    SaveAndFree(Small.PlotFit(Iris), 'kernel_regression_iris_fit_small_bandwidth.svg');

    Large := TKernelRegression.Create(Iris, Formula, 'gaussian', 5.0);
    WriteLn('bandwidth=5.0 (too large):  LOO R-squared=', Fmt(Large.RSquared), '  LOO RMSE=', Fmt(Large.Rmse));
    SaveAndFree(Large.PlotFit(Iris), 'kernel_regression_iris_fit_large_bandwidth.svg');

    WriteLn('bandwidth=', Fmt(Model.Bandwidth), ' (CV-selected): LOO R-squared=', Fmt(Model.RSquared),
            '  LOO RMSE=', Fmt(Model.Rmse));
    WriteLn(#10'Saved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg');
  finally
    Large.Free;
    Small.Free;
    Model.Free;
    Iris.Free;
  end;
end.
