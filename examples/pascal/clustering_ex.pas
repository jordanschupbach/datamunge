// CMD: just TARGET=clustering_ex run-pascal
// Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
program clustering_ex;

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

var
  Iris: TDataFrame;
  Species: array of String;
  FeatureColumns: TArrayOfString;

// For each cluster, print its size and its majority species (negative labels are noise).
procedure ReportPurity(const ClusterLabels: TArrayOfInt64; NClusters: Integer);
var
  Names: array of array of String;   // per cluster, the species in first-seen order
  Counts: array of array of Integer;
  I, C, J, K, Total, Best: Integer;
  Label_: Int64;
begin
  SetLength(Names, NClusters);
  SetLength(Counts, NClusters);
  for I := 0 to Min(Length(ClusterLabels), Length(Species)) - 1 do
  begin
    Label_ := ClusterLabels[I];
    if Label_ < 0 then Continue;
    K := -1;
    for J := 0 to High(Names[Label_]) do
      if Names[Label_][J] = Species[I] then K := J;
    if K < 0 then
    begin
      K := Length(Names[Label_]);
      SetLength(Names[Label_], K + 1);
      SetLength(Counts[Label_], K + 1);
      Names[Label_][K] := Species[I];
      Counts[Label_][K] := 0;
    end;
    Inc(Counts[Label_][K]);
  end;
  for C := 0 to NClusters - 1 do
  begin
    Total := 0;
    Best := -1;
    for J := 0 to High(Counts[C]) do
    begin
      Inc(Total, Counts[C][J]);
      if (Best < 0) or (Counts[C][J] > Counts[C][Best]) then Best := J;
    end;
    if Total = 0 then Continue;
    WriteLn('  cluster ', C, ': ', Total, ' points, majority ', Names[C][Best],
            ' (', Counts[C][Best], '/', Total, ')');
  end;
end;

function ToInt64s(const A: TArrayOfInt32): TArrayOfInt64;
var
  I: Integer;
begin
  Result := nil;
  SetLength(Result, Length(A));
  for I := 0 to High(A) do Result[I] := A[I];
end;

const
  Linkages: array[0..2] of String = ('ward', 'complete', 'average');
var
  KMeans: TKMeans;
  Model, WardModel: TAgglomerativeClustering;
  Dbscan: TDbscan;
  I: Integer;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;
  Iris := Data.Iris;
  try
    SetLength(Species, Iris.Nrows);
    for I := 0 to High(Species) do Species[I] := Iris.StringAt('Species', I);
    FeatureColumns := ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'];

    WriteLn('=================== K-means (k=3) on iris ===================');
    KMeans := TKMeans.Create(Iris, FeatureColumns, 3);
    try
      WriteLn('Inertia: ', Fmt(KMeans.Inertia), ', iterations (best run): ', KMeans.IterationsUsed);
      ReportPurity(KMeans.Labels, 3);
    finally
      KMeans.Free;
    end;

    WriteLn(#10'=================== Agglomerative clustering on iris ===================');
    for I := 0 to High(Linkages) do
    begin
      Model := TAgglomerativeClustering.Create(Iris, FeatureColumns, 3, Linkages[I]);
      try
        WriteLn('--- linkage=', Linkages[I], ' ---');
        ReportPurity(Model.Labels, 3);
      finally
        Model.Free;
      end;
    end;

    WriteLn(#10'--- Re-cutting the ward dendrogram at k=2 without refitting ---');
    WardModel := TAgglomerativeClustering.Create(Iris, FeatureColumns, 3, 'ward');
    try
      ReportPurity(WardModel.Cut(2), 2);
    finally
      WardModel.Free;
    end;

    WriteLn(#10'=================== DBSCAN on iris ===================');
    Dbscan := TDbscan.Create(Iris, FeatureColumns, 0.6, 5);
    try
      WriteLn('Clusters found: ', Dbscan.NClusters, ', noise points: ', Dbscan.NNoise,
              ' (of ', Dbscan.Observations, ')');
      ReportPurity(ToInt64s(Dbscan.Labels), Dbscan.NClusters);
    finally
      Dbscan.Free;
    end;
  finally
    Iris.Free;
  end;
end.
