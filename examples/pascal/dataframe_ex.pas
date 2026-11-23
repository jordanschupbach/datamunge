// CMD: just TARGET=dataframe_ex run-pascal
// DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
program dataframe_ex;

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
  Sales, Cleaned, Picked, Selected, Summed, Grouped, Targets, Joined: TDataFrame;
  Shape: TArrayOfInt64;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;
  Sales := TDataFrame.Create;
  Targets := TDataFrame.Create;
  Cleaned := nil; Picked := nil; Selected := nil; Summed := nil; Grouped := nil; Joined := nil;
  try
    Sales.AddStringColumn('region', ['west', 'west', 'east', 'south', 'south', 'south']);
    Sales.AddStringColumn('product', ['widget', 'widget', 'widget', 'gizmo', 'gizmo', 'gizmo']);
    Sales.AddNumericColumn('sales', [10.0, 10.0, 14.0, 8.0, 0.0, 11.0], [1, 1, 1, 1, 0, 1]);
    Sales.AddStringColumn('quarter', ['Q1', 'Q1', 'Q1', 'Q2', 'Q2', ''], [1, 1, 1, 1, 1, 0]);

    WriteLn('raw data'#10, Sales.ToString, #10);

    Cleaned := Sales.DropDuplicates(['region', 'product', 'sales', 'quarter']);
    Cleaned.FillNullString('quarter', 'unknown');
    Cleaned.FillNullNumeric('sales', 0.0);
    WriteLn('after drop_duplicates + fill_null'#10, Cleaned.ToString, #10);

    // Each step returns a new frame the caller frees, so chains keep the intermediates.
    Picked := Cleaned.Select(['region', 'sales', 'quarter']);
    Selected := Picked.SortBy('sales', False);
    WriteLn('selected + sorted'#10, Selected.ToString, #10);

    Summed := Cleaned.GroupBySum(['region'], ['sales']);
    Grouped := Summed.SortBy('sales', False);
    WriteLn('group_by_sum(region)'#10, Grouped.ToString, #10);

    Targets.AddStringColumn('region', ['west', 'east', 'south']);
    Targets.AddNumericColumn('target', [18.0, 12.0, 25.0]);
    Joined := Grouped.Join(Targets, 'region', 'region', 'left');
    WriteLn('joined with targets'#10, Joined.ToString, #10);

    Shape := Cleaned.Shape;
    WriteLn('shape = (', Shape[0], ', ', Shape[1], ')');
    WriteLn('sales count = ', Cleaned.NumericCount('sales'));
    WriteLn('sales nulls = ', Cleaned.NumericNullCount('sales'));
    WriteLn('sales sum = ', Fmt(Cleaned.NumericSum('sales')));
    WriteLn('sales mean = ', Fmt(Cleaned.NumericMean('sales')));
  finally
    Joined.Free; Grouped.Free; Summed.Free; Selected.Free; Picked.Free; Cleaned.Free;
    Targets.Free;
    Sales.Free;
  end;
end.
