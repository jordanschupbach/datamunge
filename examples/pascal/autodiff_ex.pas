// CMD: just TARGET=autodiff_ex run-pascal
// Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
program autodiff_ex;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, datamunge;

// Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
// (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
// bound. This port demonstrates the same ideas with the bound named-method API.
// TVar.Index is the Var's position in the adjoint vector TTape.Backward returns.

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

// Every Dual/Var/HyperDual operation returns a new object that must be freed. The examples keep
// them in an arena and free it at the end of each section.
var
  Arena: array of TNativeObject;

function Keep(AObject: TNativeObject): TNativeObject;
begin
  SetLength(Arena, Length(Arena) + 1);
  Arena[High(Arena)] := AObject;
  Result := AObject;
end;

function KD(AObject: TDual): TDual; begin Result := TDual(Keep(AObject)); end;
function KV(AObject: TVar): TVar; begin Result := TVar(Keep(AObject)); end;
function KH(AObject: THyperDual): THyperDual; begin Result := THyperDual(Keep(AObject)); end;

// Frees the arena's objects, newest first (so each Var goes before its Tape).
procedure FreeArena;
var
  I: Integer;
begin
  for I := High(Arena) downto 0 do Arena[I].Free;
  Arena := nil;
end;

function RosenbrockDual(X0, X1: TDual): TDual;
var
  A, B: TDual;
begin
  A := KD(KD(TDual.Create(1.0, 0.0)).Subtract(X0));
  B := KD(X1.Subtract(KD(X0.Multiply(X0))));
  Result := KD(KD(A.Multiply(A)).Add(KD(KD(B.Multiply(B)).MultiplyScalar(100.0))));
end;

function RosenbrockValue(X0, X1: Double): Double;
begin
  Result := Sqr(1.0 - X0) + 100.0 * Sqr(X1 - X0 * X0);
end;

// Build the Rosenbrock graph on tape; returns the output Var.
function RosenbrockVar(ATape: TTape; X0, X1: TVar): TVar;
var
  A, B: TVar;
begin
  A := KV(KV(TVar.Create(ATape, 1.0)).Subtract(X0));
  B := KV(X1.Subtract(KV(X0.Multiply(X0))));
  Result := KV(KV(A.Multiply(A)).Add(KV(KV(B.Multiply(B)).MultiplyScalar(100.0))));
end;

function RosenbrockHyperDual(X0, X1: THyperDual): THyperDual;
var
  A, B: THyperDual;
begin
  A := KH(KH(THyperDual.Create(1.0, 0.0, 0.0, 0.0)).Subtract(X0));
  B := KH(X1.Subtract(KH(X0.Multiply(X0))));
  Result := KH(KH(A.Multiply(A)).Add(KH(KH(B.Multiply(B)).MultiplyScalar(100.0))));
end;

function Adjoint(const AAdjoints: TArrayOfDouble; V: TVar): Double;
begin
  Result := AAdjoints[V.Index];
end;

// f(x,y) = [x^2, xy, y^3]
function VectorFn(X, Y: TDual): specialize TArray<TDual>;
begin
  Result := [KD(X.Multiply(X)), KD(X.Multiply(Y)), KD(KD(Y.Multiply(Y)).Multiply(Y))];
end;

const
  LearningRate: Double = 0.001;
  NSteps = 2000;
  SeedX: array[0..1] of Double = (1.0, 0.0);
  SeedY: array[0..1] of Double = (0.0, 1.0);
var
  X0, F, Gx, Gy: TDual;
  Tape: TTape;
  A, B, Y, Vx, Vy, Vf: TVar;
  Adj, Grad: TArrayOfDouble;
  Px, Py, Mx, My, Loss: Double;
  Jac: array[0..2, 0..1] of Double;
  H: array[0..1, 0..1] of Double;
  Outputs: specialize TArray<TDual>;
  Point: array[0..1] of Double;
  Row, Col, I, J, Step: Integer;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;

  WriteLn('=================== Forward mode: scalar derivative ===================');
  X0 := KD(TDual.Create(2.0, 1.0));  // seed derivative = 1 to read df/dx directly
  F := KD(KD(KD(X0.Multiply(X0)).Multiply(X0)).Subtract(KD(X0.MultiplyScalar(2.0))));  // x^3 - 2x
  WriteLn('f(x) = x^3 - 2x, f''(2) = ', Fmt(F.Derivative), ' (exact: 10)');
  FreeArena;

  WriteLn(#10'=================== Reverse mode: build a graph by hand ===================');
  Tape := TTape.Create;
  try
    A := KV(TVar.Create(Tape, 2.0));
    B := KV(TVar.Create(Tape, 3.0));
    Y := KV(KV(A.Multiply(B)).Add(KV(A.Sin)));
    WriteLn('y = a*b + sin(a) at a=2, b=3 -> y = ', Fmt(Y.Value));
    Adj := Tape.Backward(Y);
    WriteLn('dy/da = ', Fmt(Adjoint(Adj, A)), ' (exact: b + cos(a))');
    WriteLn('dy/db = ', Fmt(Adjoint(Adj, B)), ' (exact: a)');
    FreeArena;
  finally
    Tape.Free;
  end;

  WriteLn(#10'=================== Forward vs reverse mode agree on the Rosenbrock function ===================');
  Px := 0.0;
  Py := 0.0;
  // Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
  Gx := RosenbrockDual(KD(TDual.Create(Px, 1.0)), KD(TDual.Create(Py, 0.0)));
  Gy := RosenbrockDual(KD(TDual.Create(Px, 0.0)), KD(TDual.Create(Py, 1.0)));
  // Reverse mode: one Tape/Var pass computes every partial at once.
  Tape := TTape.Create;
  try
    Vx := KV(TVar.Create(Tape, Px));
    Vy := KV(TVar.Create(Tape, Py));
    Grad := Tape.Backward(RosenbrockVar(Tape, Vx, Vy));
    WriteLn('f(0,0) = ', Fmt(RosenbrockValue(Px, Py)));
    WriteLn('gradient (forward mode): [', Fmt(Gx.Derivative), ', ', Fmt(Gy.Derivative), ']');
    WriteLn('gradient (reverse mode): [', Fmt(Adjoint(Grad, Vx)), ', ', Fmt(Adjoint(Grad, Vy)), ']');
    FreeArena;
  finally
    Tape.Free;
  end;

  WriteLn(#10'=================== Jacobian of a vector-valued function ===================');
  // f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
  for Col := 0 to 1 do
  begin
    Outputs := VectorFn(KD(TDual.Create(2.0, SeedX[Col])), KD(TDual.Create(3.0, SeedY[Col])));
    for Row := 0 to High(Outputs) do
      Jac[Row, Col] := Outputs[Row].Derivative;
  end;
  FreeArena;
  WriteLn('f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:');
  for Row := 0 to 2 do
    WriteLn('  [', Fmt(Jac[Row, 0]), ', ', Fmt(Jac[Row, 1]), ']');

  WriteLn(#10'=================== Hessian via second-order forward mode (HyperDual) ===================');
  // One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
  Mx := 1.0;  // the Rosenbrock function's minimum
  My := 1.0;
  for I := 0 to 1 do
    for J := 0 to 1 do
      H[I, J] := RosenbrockHyperDual(KH(THyperDual.Create(Mx, SeedX[I], SeedX[J], 0.0)),
                                     KH(THyperDual.Create(My, SeedY[I], SeedY[J], 0.0))).Eps1eps2;
  FreeArena;
  WriteLn('Hessian of the Rosenbrock function at its minimum (1,1):');
  for I := 0 to 1 do
    WriteLn('  [', Fmt(H[I, 0]), ', ', Fmt(H[I, 1]), ']');

  WriteLn(#10'=================== Gradient descent driven by reverse-mode gradients ===================');
  Point[0] := -1.2;  // the classic Rosenbrock start
  Point[1] := 1.0;
  for Step := 0 to NSteps - 1 do
  begin
    Tape := TTape.Create;
    try
      Vx := KV(TVar.Create(Tape, Point[0]));
      Vy := KV(TVar.Create(Tape, Point[1]));
      Vf := RosenbrockVar(Tape, Vx, Vy);
      Grad := Tape.Backward(Vf);
      Loss := Vf.Value;
      Point[0] := Point[0] - LearningRate * Adjoint(Grad, Vx);
      Point[1] := Point[1] - LearningRate * Adjoint(Grad, Vy);
      if (Step = 0) or (Step = NSteps - 1) then
        WriteLn('step ', Step, ': loss = ', Fmt(Loss), ', x = [', Fmt(Point[0]), ', ', Fmt(Point[1]), ']');
      FreeArena;
    finally
      Tape.Free;
    end;
  end;
  WriteLn('(true minimum is at [1, 1] with loss 0)');
end.
