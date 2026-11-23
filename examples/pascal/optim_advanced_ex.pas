// CMD: just TARGET=optim_advanced_ex run-pascal
// Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton,
// AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Pascal.
//
// Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
// EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
// ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
// implement the objective. From Pascal, derive from the interface class and override its virtual
// methods; the optimizer then calls the Pascal overrides. Coordinates are updated in place through
// a var parameter (the C++ parameter is a non-const std::vector<double>&).
program optim_advanced_ex;

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

// C#'s "F<n>" format: fixed point with ADecimals decimals.
function FmtF(AX: Double; ADecimals: Integer): String;
var
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  Result := FloatToStrF(AX, ffFixed, 15, ADecimals, FS);
end;

// C#'s "G<n>" format: at most APrecision significant digits, scientific notation for exponents
// below -5 or from APrecision up.
function FmtG(AX: Double; APrecision: Integer): String;
var
  FS: TFormatSettings;
  S, Digits: String;
  P, E, I: Integer;
begin
  if (AX = 0) or IsNan(AX) or IsInfinite(AX) then Exit(Fmt(AX));
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  S := FloatToStrF(AX, ffExponent, APrecision, 4, FS);
  P := Pos('E', S);
  E := StrToInt(Copy(S, P + 1, Length(S)));
  Digits := '';
  for I := 1 to P - 1 do
    if S[I] in ['0'..'9'] then Digits := Digits + S[I];
  while (Length(Digits) > 1) and (Digits[Length(Digits)] = '0') do
    Delete(Digits, Length(Digits), 1);
  if AX < 0 then Result := '-' else Result := '';
  if (E >= APrecision) or (E < -5) then
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

type
  TQuadratic = class(THessianFunction)
  public
    function Evaluate(const ACoordinates: TArrayOfDouble): Double; override;
    function Gradient(const ACoordinates: TArrayOfDouble): TArrayOfDouble; override;
    function Hessian(const ACoordinates: TArrayOfDouble): TArrayOfArrayOfDouble; override;
  end;

  TExponential = class(TResidualFunction)
  private
    FTs: TArrayOfDouble;
  public
    constructor Create(const ATs: TArrayOfDouble);
    function Residuals(const ACoordinates: TArrayOfDouble): TArrayOfDouble; override;
    function Jacobian(const ACoordinates: TArrayOfDouble): TArrayOfArrayOfDouble; override;
  end;

  TIncomplete = class(THessianFunction)
  public
    function Evaluate(const ACoordinates: TArrayOfDouble): Double; override;
  end;

function TQuadratic.Evaluate(const ACoordinates: TArrayOfDouble): Double;
begin
  Result := Sqr(ACoordinates[0] - 1) + 10 * Sqr(ACoordinates[1] + 2);
end;

function TQuadratic.Gradient(const ACoordinates: TArrayOfDouble): TArrayOfDouble;
begin
  Result := [2 * (ACoordinates[0] - 1), 20 * (ACoordinates[1] + 2)];
end;

function TQuadratic.Hessian(const ACoordinates: TArrayOfDouble): TArrayOfArrayOfDouble;
begin
  Result := [[2.0, 0.0], [0.0, 20.0]];
end;

constructor TExponential.Create(const ATs: TArrayOfDouble);
begin
  inherited Create;
  FTs := ATs;
end;

function TExponential.Residuals(const ACoordinates: TArrayOfDouble): TArrayOfDouble;
var
  I: Integer;
  A, B: Double;
begin
  Result := nil;
  SetLength(Result, Length(FTs));
  for I := 0 to High(FTs) do
  begin
    A := Exp(ACoordinates[1] * FTs[I]);
    B := Exp(Double(0.5) * FTs[I]);
    Result[I] := ACoordinates[0] * A - 2 * B;
  end;
end;

function TExponential.Jacobian(const ACoordinates: TArrayOfDouble): TArrayOfArrayOfDouble;
var
  I: Integer;
  E: Double;
begin
  Result := nil;
  SetLength(Result, Length(FTs));
  for I := 0 to High(FTs) do
  begin
    E := Exp(ACoordinates[1] * FTs[I]);
    Result[I] := [E, ACoordinates[0] * FTs[I] * E];
  end;
end;

function TIncomplete.Evaluate(const ACoordinates: TArrayOfDouble): Double;
begin
  Result := 0.0;
end;

const
  Xs: array[0..5] of Double = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0);
  Probes: array[0..2] of Double = (0.25, 1.0, 2.5);

var
  Surrogate: TRbfGaussianProcessSurrogate;
  Points: TArrayOfArrayOfDouble;
  Ys, Start, Parameters, Origin: TArrayOfDouble;
  Incumbent, Value, Sse: Double;
  I: Integer;
  PgOptions: TProximalGradientOptions;
  FistaOptions: TFistaOptions;
  LmOptions: TLevenbergMarquardtOptions;
  NewtonOptions: TNewtonOptions;
  TrnOptions: TTrustRegionNewtonOptions;
  AlOptions: TAugmentedLagrangianOptions;
  SqpOptions: TSqpOptions;
  IpOptions: TInteriorPointOptions;
  BoOptions: TBayesianOptimizationOptions;
  Pg: TProximalGradient;
  Fista: TFista;
  Lm: TLevenbergMarquardt;
  Newton: TNewton;
  Trn: TTrustRegionNewton;
  Al: TAugmentedLagrangian;
  Sqp: TSqp;
  Ip: TInteriorPoint;
  Bo: TBayesianOptimization;
  Quadratic: TQuadratic;
  Exponential: TExponential;
  Incomplete: TIncomplete;
  Err: Exception;
  ErrMessage: String;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;

  WriteLn('=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ===================');
  // Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
  // then check the acquisition function favors an unexplored region.
  SetLength(Ys, Length(Xs));
  SetLength(Points, Length(Xs));
  for I := 0 to High(Xs) do
  begin
    Ys[I] := (Xs[I] - 1) * (Xs[I] - 1);
    Points[I] := [Xs[I]];
  end;
  Surrogate := TRbfGaussianProcessSurrogate.Create(1.0, 1e-6);
  try
    Surrogate.Fit(Points, Ys);
    Incumbent := Min(0.0, MinValue(Ys));  // best observed value so far
    WriteLn('incumbent (best observed y): ', Fmt(Incumbent));
    for I := 0 to High(Probes) do
      WriteLn('acquisition(x=', FmtF(Probes[I], 2), ') = ', FmtF(Surrogate.Acquisition([Probes[I]], Incumbent), 6));
  finally
    Surrogate.Free;
  end;

  WriteLn(#10'=================== Options structs + optimizer construction (all 10 new classes) ===================');
  // Every new Options struct and optimizer class constructs cleanly, proving
  // the binding for the whole batch is wired correctly. (The optimizers copy their options.)
  PgOptions := TProximalGradientOptions.Create;
  PgOptions.StepSize := 0.05;
  FistaOptions := TFistaOptions.Create;
  LmOptions := TLevenbergMarquardtOptions.Create;
  LmOptions.MaxIterations := 200;
  NewtonOptions := TNewtonOptions.Create;
  TrnOptions := TTrustRegionNewtonOptions.Create;
  AlOptions := TAugmentedLagrangianOptions.Create;
  SqpOptions := TSqpOptions.Create;
  IpOptions := TInteriorPointOptions.Create;
  BoOptions := TBayesianOptimizationOptions.Create;
  BoOptions.InitialSamples := 8;
  Pg := TProximalGradient.Create(PgOptions);
  Fista := TFista.Create(FistaOptions);
  Lm := TLevenbergMarquardt.Create(LmOptions);
  Newton := TNewton.Create(NewtonOptions);
  Trn := TTrustRegionNewton.Create(TrnOptions);
  Al := TAugmentedLagrangian.Create(AlOptions);
  Sqp := TSqp.Create(SqpOptions);
  Ip := TInteriorPoint.Create(IpOptions);
  Bo := TBayesianOptimization.Create(BoOptions);
  Quadratic := nil;
  Exponential := nil;
  Incomplete := nil;
  try
    WriteLn('ProximalGradient: ', Pg.ToString);
    WriteLn('FISTA: ', Fista.ToString);
    WriteLn('LevenbergMarquardt: ', Lm.ToString);
    WriteLn('Newton: ', Newton.ToString);
    WriteLn('TrustRegionNewton: ', Trn.ToString);
    WriteLn('AugmentedLagrangian: ', Al.ToString);
    WriteLn('SQP: ', Sqp.ToString);
    WriteLn('InteriorPoint: ', Ip.ToString);
    WriteLn('BayesianOptimization: ', Bo.ToString);

    WriteLn(#10'=================== Pascal objectives (subclassing the C++ interfaces) ===================');
    // Newton's method on f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
    Start := [5.0, 5.0];
    Quadratic := TQuadratic.Create;
    Value := Newton.Optimize(Quadratic, Start);
    WriteLn('Newton: f = ', FmtG(Value, 6), ' at (', FmtF(Start[0], 6), ', ', FmtF(Start[1], 6),
            ') (exact minimum 0 at (1, -2))');

    // Levenberg-Marquardt fitting y = a * exp(b t) to exact data from a = 2, b = 0.5.
    Parameters := [1.0, 0.1];
    Exponential := TExponential.Create([0.0, 0.5, 1.0, 1.5, 2.0]);
    Sse := Lm.Optimize(Exponential, Parameters);
    WriteLn('LevenbergMarquardt: residual = ', FmtG(Sse, 3), ', a = ', FmtF(Parameters[0], 6),
            ', b = ', FmtF(Parameters[1], 6), ' (exact: 2, 0.5)');

    // An interface method a subclass leaves out raises when the optimizer calls it.
    Err := nil;
    ErrMessage := '';
    Incomplete := TIncomplete.Create;
    try
      Origin := [0.0, 0.0];
      Newton.Optimize(Incomplete, Origin);
    except
      on E: Exception do
      begin
        Err := E;
        ErrMessage := E.Message;
      end;
    end;
    WriteLn('missing gradient(): ok=', BoolToStr(Err = nil, 'True', 'False'),
            ', error mentions it: ', BoolToStr((Err <> nil) and (Pos('gradient', ErrMessage) > 0), 'True', 'False'));
  finally
    Incomplete.Free;
    Exponential.Free;
    Quadratic.Free;
    Bo.Free; Ip.Free; Sqp.Free; Al.Free; Trn.Free; Newton.Free; Lm.Free; Fista.Free; Pg.Free;
    BoOptions.Free; IpOptions.Free; SqpOptions.Free; AlOptions.Free; TrnOptions.Free;
    NewtonOptions.Free; LmOptions.Free; FistaOptions.Free; PgOptions.Free;
  end;
end.
