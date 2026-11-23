// CMD: just TARGET=lda_iris_ex run-pascal
// Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.
program lda_iris_ex;

{$mode objfpc}{$H+}

uses
  SysUtils, datamunge;

var
  Iris, NewData, Table: TDataFrame;
  Model: TLda;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;
  Iris := Data.Iris;
  NewData := TDataFrame.Create;
  Model := nil;
  try
    WriteLn('iris: ', Iris.Nrows, ' rows x ', Iris.Ncols, ' cols'#10);

    Model := TLda.Create(Iris, 'Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width');
    Model.PrintSummary;

    Table := Model.ConfusionMatrix;
    WriteLn(#10'Confusion matrix (rows = actual, cols = predicted):'#10, Table.ToString);
    Table.Free;

    NewData.AddNumericColumn('Sepal.Length', [5.1, 6.0, 6.5, 6.2]);
    NewData.AddNumericColumn('Sepal.Width', [3.5, 2.7, 3.0, 2.8]);
    NewData.AddNumericColumn('Petal.Length', [1.4, 4.5, 5.5, 4.8]);
    NewData.AddNumericColumn('Petal.Width', [0.2, 1.5, 2.0, 1.8]);
    Table := Model.PredictFrame(NewData);
    WriteLn(#10'Predictions for new flowers:'#10, Table.ToString);
    Table.Free;

    Model.SaveDiscriminantPlot('lda_iris_discriminants.svg');
    WriteLn(#10'Saved discriminant plot as lda_iris_discriminants.svg');
  finally
    Model.Free;
    NewData.Free;
    Iris.Free;
  end;
end.
