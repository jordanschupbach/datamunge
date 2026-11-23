// CMD: just TARGET=lm_ex run-pascal
// Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions.
program lm_ex;

{$mode objfpc}{$H+}

uses
  SysUtils, datamunge;

var
  Cars, NewCars, Table: TDataFrame;
  Model: TLm;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;
  Cars := TDataFrame.Create;
  NewCars := TDataFrame.Create;
  Model := nil;
  try
    Cars.AddNumericColumn('hp', [110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0]);
    Cars.AddNumericColumn('wt', [2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44]);
    Cars.AddStringColumn('transmission', ['manual', 'manual', 'manual', 'automatic', 'automatic',
                                          'automatic', 'automatic', 'automatic', 'automatic', 'automatic']);
    Cars.AddNumericColumn('mpg', [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2]);

    WriteLn('Fitting: mpg ~ hp + wt + transmission'#10);
    Model := TLm.Create(Cars, 'mpg ~ hp + wt + transmission');
    Model.PrintSummary;

    Table := Model.Anova;
    WriteLn(#10'Sequential ANOVA:'#10, Table.ToString);
    Table.Free;

    NewCars.AddNumericColumn('hp', [150.0, 90.0]);
    NewCars.AddNumericColumn('wt', [3.0, 2.5]);
    NewCars.AddStringColumn('transmission', ['manual', 'automatic']);
    Table := Model.PredictFrame(NewCars, 'confidence');
    WriteLn(#10'Predictions with 95% confidence intervals:'#10, Table.ToString);
    Table.Free;

    Model.SaveDiagnosticPlots('lm_ex_diagnostics');
    WriteLn(#10'Saved diagnostic plots as lm_ex_diagnostics_*.svg');
  finally
    Model.Free;
    NewCars.Free;
    Cars.Free;
  end;
end.
