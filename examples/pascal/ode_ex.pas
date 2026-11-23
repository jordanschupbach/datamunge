// CMD: just TARGET=ode_ex run-pascal
// ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
program ode_ex;

{$mode objfpc}{$H+}

uses
  SysUtils, datamunge;

// C#'s "G<n>" format (approximating C's %g): at most APrecision significant digits, no padding.
function FmtG(AX: Double; APrecision: Integer = 6): String;
var
  FS: TFormatSettings;
  S, Digits: String;
  P, E, I: Integer;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  if AX = 0 then Exit('0');
  S := FloatToStrF(AX, ffExponent, APrecision, 4, FS);
  P := Pos('E', S);
  E := StrToInt(Copy(S, P + 1, Length(S)));
  Digits := '';
  for I := 1 to P - 1 do
    if S[I] in ['0'..'9'] then Digits := Digits + S[I];
  while (Length(Digits) > 1) and (Digits[Length(Digits)] = '0') do
    Delete(Digits, Length(Digits), 1);
  if AX < 0 then Result := '-' else Result := '';
  if (E >= APrecision) or (E < -4) then
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

// C#'s "F<n>" format: fixed point with ADecimals decimals.
function FmtF(AX: Double; ADecimals: Integer): String;
var
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  Result := FloatToStrF(AX, ffFixed, 15, ADecimals, FS);
end;

function LastState(ASol: TOdeSolution): TArrayOfDouble;
begin
  Result := ASol.StateAt(ASol.Size - 1);
end;

// The time vector, one value per step.
function SolutionTimes(ASol: TOdeSolution): TArrayOfDouble;
var
  I: Int64;
begin
  Result := nil;
  SetLength(Result, ASol.Size);
  for I := 0 to ASol.Size - 1 do
    Result[I] := ASol.TimeAt(I);
end;

// The requested state component (0-based), one value per step.
function SolutionComponent(ASol: TOdeSolution; AComponent: Integer): TArrayOfDouble;
var
  I: Int64;
begin
  Result := nil;
  SetLength(Result, ASol.Size);
  for I := 0 to ASol.Size - 1 do
    Result[I] := ASol.StateAt(I)[AComponent];
end;

function Rk4Solver(AStep: Double): TOdeSolver;
var
  Options: TOdeOptions;
begin
  Options := TOdeOptions.Create;
  try
    Options.Method := TStepMethod.Rk4;
    Options.StepSize := AStep;
    Result := TOdeSolver.Create(Options);
  finally
    Options.Free;
  end;
end;

// Prints one built-in system's step count and final state.
procedure RunBuiltin(ASolver: TOdeSolver; const AName: String; const AParams, AY0: TArrayOfDouble);
var
  Sol: TOdeSolution;
  State: TArrayOfDouble;
  Text: String;
  I: Integer;
begin
  Sol := ASolver.SolveBuiltin(AName, AParams, AY0, 0.0, 1.0);
  try
    State := LastState(Sol);
    Text := '';
    for I := 0 to High(State) do
    begin
      if I > 0 then Text := Text + ', ';
      Text := Text + FmtG(State[I]);
    end;
    WriteLn(Format('%-22s steps=%-6d final state=[%s]', [AName, Sol.StepsTaken, Text]));
  finally
    Sol.Free;
  end;
end;

const
  // 8.0 / 3 would be folded in Single precision (FPC types the literal 8.0 as Single).
  LorenzBeta: Double = 8 / Double(3);
var
  Solver: TOdeSolver;
  Sol: TOdeSolution;
  Last, Ts, Xs, Vs, Zs: TArrayOfDouble;
  X, V: Double;
  P: TRPlot;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;

  // NOTE: like the Lua version this example uses the built-in named systems via SolveBuiltin.

  WriteLn('=================== Every named built-in system (no director needed) ===================');
  Solver := TOdeSolver.Create;
  try
    RunBuiltin(Solver, 'exponential_decay', [1.0], [1.0]);
    RunBuiltin(Solver, 'logistic_growth', [1.0, 1.0], [0.5]);
    RunBuiltin(Solver, 'harmonic_oscillator', [1.0], [1.0, 0.0]);
    RunBuiltin(Solver, 'van_der_pol', [1.0], [2.0, 0.0]);
    RunBuiltin(Solver, 'lorenz', [10.0, 28.0, LorenzBeta], [1.0, 1.0, 1.0]);
  finally
    Solver.Free;
  end;

  WriteLn(#10'=================== Harmonic oscillator (energy conservation) ===================');
  Solver := Rk4Solver(0.01);
  Sol := Solver.SolveBuiltin('harmonic_oscillator', [1.0], [1.0, 0.0], 0.0, 20.0);
  try
    Last := LastState(Sol);
    X := Last[0];
    V := Last[1];
    WriteLn('x(20) = ', FmtF(X, 6), ' (cos(20) = ', FmtF(Cos(20.0), 6), ')');
    WriteLn('energy x^2+v^2 = ', FmtF(X * X + V * V, 6), ' (should stay near 1.0)');
    Ts := SolutionTimes(Sol);
    Xs := SolutionComponent(Sol, 0);
    Vs := SolutionComponent(Sol, 1);
    P := TRPlot.Plot(Ts, Xs, 'l', 'x(t)');
    try
      P.Lines(Ts, Vs, 'v(t)');
      P.Title2('Harmonic Oscillator').XLabel2('t').YLabel2('state');
      P.SaveSvg('ode_harmonic_oscillator_pascal.svg');
    finally
      P.Free;
    end;
    WriteLn('wrote ode_harmonic_oscillator_pascal.svg');
  finally
    Sol.Free;
    Solver.Free;
  end;

  WriteLn(#10'=================== Lorenz attractor (phase plane) ===================');
  Solver := Rk4Solver(0.005);
  Sol := Solver.SolveBuiltin('lorenz', [10.0, 28.0, LorenzBeta], [1.0, 1.0, 1.0], 0.0, 25.0);
  try
    WriteLn('steps_taken = ', Sol.StepsTaken);
    Xs := SolutionComponent(Sol, 0);
    Zs := SolutionComponent(Sol, 2);
    P := TRPlot.Plot(Xs, Zs, 'l', 'trajectory');
    try
      P.Title2('Lorenz Attractor (x-z phase plane)').XLabel2('x').YLabel2('z');
      P.SaveSvg('ode_lorenz_phase_plane_pascal.svg');
    finally
      P.Free;
    end;
    WriteLn('wrote ode_lorenz_phase_plane_pascal.svg');
  finally
    Sol.Free;
    Solver.Free;
  end;
end.
