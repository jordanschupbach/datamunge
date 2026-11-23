// CMD: just TARGET=pca_ex run-pascal
// PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
program pca_ex;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, datamunge;

// C#'s "F<n>" format: fixed point with ADecimals decimals.
function FmtF(AX: Double; ADecimals: Integer): String;
var
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  Result := FloatToStrF(AX, ffFixed, 15, ADecimals, FS);
end;

var
  Iris, Scores: TDataFrame;
  Species, Names: TArrayOfString;
  Loadings: TArrayOfDouble;
  Pca, Unscaled: TPca;
  Plot: TRPlot;
  I: Integer;
const
  FeatureColumns: array[0..3] of String = ('Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width');
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;

  Iris := Data.Iris;
  Pca := nil;
  Unscaled := nil;
  try
    // species labels, for the grouped (colored-by-species) scores plot
    SetLength(Species, Iris.Nrows);
    for I := 0 to High(Species) do
      Species[I] := Iris.StringAt('Species', I);

    WriteLn('=================== PCA on iris (scaled) ===================');
    Pca := TPca.Create(Iris, FeatureColumns, True, True);
    Pca.PrintSummary;

    WriteLn(#10'PC1 loadings (which original features drive it):');
    Names := Pca.FeatureNames;
    Loadings := Pca.ComponentLoadings(0);
    for I := 0 to Min(High(Names), High(Loadings)) do
      WriteLn('  ', Names[I], ': ', FmtF(Loadings[I], 4));

    Scores := Pca.ScoresFrame;
    WriteLn(#10'Training scores as a DataFrame:'#10, Scores.ToString);
    Scores.Free;

    Plot := Pca.PlotScoresGrouped(Species);
    Plot.SaveSvg('pca_iris_scores.svg');
    Plot.Free;
    WriteLn(#10'Scores scatter (colored by species) saved as pca_iris_scores.svg');

    Plot := Pca.PlotScree;
    Plot.SaveSvg('pca_iris_scree.svg');
    Plot.Free;
    WriteLn('Scree plot saved as pca_iris_scree.svg');

    WriteLn(#10'=================== PCA on iris (unscaled) ===================');
    Unscaled := TPca.Create(Iris, FeatureColumns, True, False);
    WriteLn('PC1 explains ', FmtF(100 * Unscaled.ExplainedVarianceRatio[0], 2), '% of variance (vs. ',
            FmtF(100 * Pca.ExplainedVarianceRatio[0], 2),
            '% scaled) -- Sepal.Length''s larger raw variance dominates the unscaled covariance matrix.');
  finally
    Unscaled.Free;
    Pca.Free;
    Iris.Free;
  end;
end.
