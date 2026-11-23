// CMD: just TARGET=gaussian_process_regression_iris_ex run-pascal
// Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals.
program gaussian_process_regression_iris_ex;

{$mode objfpc}{$H+}

uses
  SysUtils, datamunge;

const
  Formula = 'Petal.Length ~ Petal.Width';

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
  Iris, Query, Detail: TDataFrame;
  Model: TGaussianProcessRegression;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;
  Iris := Data.Iris;
  Query := TDataFrame.Create;
  Model := nil;
  try
    WriteLn('iris: ', Iris.Nrows, ' rows x ', Iris.Ncols, ' cols');
    WriteLn('formula: ', Formula, #10);

    // Both the length scale and the noise ratio are auto-selected by maximizing the exact log
    // marginal likelihood.
    Model := TGaussianProcessRegression.Create(Iris, Formula);
    Model.PrintSummary;

    SaveAndFree(Model.PlotFit(Iris), 'gpr_iris_fit.svg');
    SaveAndFree(Model.PlotLengthScaleProfile, 'gpr_iris_length_scale_profile.svg');
    WriteLn(#10'Saved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg');

    // Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
    // interval at every point -- including far outside the training data, where it should widen
    // substantially as the model's uncertainty grows.
    Query.AddNumericColumn('Petal.Width', [0.2, 1.3, 2.5, 10.0]);
    Detail := Model.PredictFrame(Query, 'confidence');
    WriteLn(#10'Predictions with 95% confidence intervals:');
    WriteLn(Detail.ToString);
    Detail.Free;
    WriteLn('(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its'#10 +
            ' interval is than the in-range predictions.)');
  finally
    Model.Free;
    Query.Free;
    Iris.Free;
  end;
end.
