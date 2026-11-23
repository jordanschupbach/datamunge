// CMD: just TARGET=elastic_net_iris_ex run-pascal
// Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
program elastic_net_iris_ex;

{$mode objfpc}{$H+}

uses
  SysUtils, datamunge;

const
  Formula = 'Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width';

// Saves a plot a model returned by value, then frees it.
procedure SaveAndFree(APlot: TPlot; const APath: String);
begin
  try
    APlot.Save(APath);
  finally
    APlot.Free;
  end;
end;

procedure SaveFigures(APath, ACv: TRPlot; const AName, ALabel: String);
begin
  SaveAndFree(APath, 'elastic_net_' + AName + '_path.svg');
  SaveAndFree(ACv, 'elastic_net_' + AName + '_cv.svg');
  WriteLn('  ', ALabel, ' elastic_net_', AName, '_path.svg, elastic_net_', AName, '_cv.svg');
end;

var
  Iris: TDataFrame;
  Ridge: TRidge;
  Lasso: TLasso;
  Elastic: TElasticNet;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;
  Iris := Data.Iris;
  Ridge := nil;
  Lasso := nil;
  Elastic := nil;
  try
    WriteLn('iris: ', Iris.Nrows, ' rows x ', Iris.Ncols, ' cols');
    WriteLn('formula: ', Formula, #10);

    WriteLn('=================== Ridge ===================');
    Ridge := TRidge.Create(Iris, Formula);
    Ridge.PrintSummary;

    WriteLn(#10'=================== Lasso ===================');
    Lasso := TLasso.Create(Iris, Formula);
    Lasso.PrintSummary;

    WriteLn(#10'================= Elastic Net =================');
    // TElasticNet.Create(data, formula, alpha=0.5, lambda, nLambda, cvFolds, standardize, seed)
    Elastic := TElasticNet.Create(Iris, Formula, 0.5);
    Elastic.PrintSummary;

    WriteLn(#10'Saved figures showing how each model''s coefficients respond to the regularization strength, ' +
            'and the cross-validation curve used to pick it:');
    SaveFigures(Ridge.PlotCoefficientPath, Ridge.PlotCvCurve, 'ridge', 'ridge:      ');
    SaveFigures(Lasso.PlotCoefficientPath, Lasso.PlotCvCurve, 'lasso', 'lasso:      ');
    SaveFigures(Elastic.PlotCoefficientPath, Elastic.PlotCvCurve, 'elasticnet', 'elastic net:');
  finally
    Elastic.Free;
    Lasso.Free;
    Ridge.Free;
    Iris.Free;
  end;
end.
