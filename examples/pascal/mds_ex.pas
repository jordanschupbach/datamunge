// CMD: just TARGET=mds_ex run-pascal
// Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
program mds_ex;

{$mode objfpc}{$H+}

uses
  SysUtils, datamunge;

// C#'s "F<n>" format: fixed point with ADecimals decimals.
function FmtF(AX: Double; ADecimals: Integer): String;
var
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  Result := FloatToStrF(AX, ffFixed, 15, ADecimals, FS);
end;

const
  FeatureColumns: array[0..3] of String = ('Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width');

var
  Iris, Table: TDataFrame;
  Species: TArrayOfString;
  Mds, ManhattanMds: TMds;
  Plot: TRPlot;
  I: Integer;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;
  Iris := Data.Iris;
  Mds := nil;
  ManhattanMds := nil;
  try
    SetLength(Species, Iris.Nrows);
    for I := 0 to High(Species) do
      Species[I] := Iris.StringAt('Species', I);

    WriteLn('=================== Classical MDS on iris (euclidean) ===================');
    Mds := TMds.Create(Iris, FeatureColumns, 2, 'euclidean');
    Mds.PrintSummary;

    Table := Mds.EmbeddingFrame;
    WriteLn(#10'Embedding as a DataFrame:'#10, Table.ToString);
    Table.Free;

    Plot := Mds.PlotEmbeddingGrouped(Species);
    Plot.SaveSvg('mds_iris_embedding.svg');
    Plot.Free;
    WriteLn(#10'Embedding scatter (colored by species) saved as mds_iris_embedding.svg');

    WriteLn(#10'=================== Classical MDS on iris (manhattan) ===================');
    ManhattanMds := TMds.Create(Iris, FeatureColumns, 2, 'manhattan');
    WriteLn('Goodness of fit: euclidean=', FmtF(Mds.GoodnessOfFit, 4),
            ', manhattan=', FmtF(ManhattanMds.GoodnessOfFit, 4));
  finally
    ManhattanMds.Free;
    Mds.Free;
    Iris.Free;
  end;
end.
