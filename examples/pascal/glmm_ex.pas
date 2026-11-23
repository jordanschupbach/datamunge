// CMD: just TARGET=glmm_ex run-pascal
// Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
program glmm_ex;

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

type
  // .NET's System.Random(seed) (the seeded Knuth subtractive generator), so the simulated data
  // match the C# example exactly.
  TNetRandom = class
  private
    FSeedArray: array[0..55] of Int32;
    FNext, FNextP: Integer;
    function InternalSample: Int32;
  public
    constructor Create(ASeed: Int32);
    function NextDouble: Double;
    function Next(AMinValue, AMaxValue: Int32): Int32;
  end;

constructor TNetRandom.Create(ASeed: Int32);
const
  MBig = High(Int32);
  MSeed = 161803398;
var
  Mj, Mk, I, Ii, K: Int32;
begin
  inherited Create;
  if ASeed = Low(Int32) then Mj := MSeed - MBig else Mj := MSeed - Abs(ASeed);
  FSeedArray[55] := Mj;
  Mk := 1;
  for I := 1 to 54 do
  begin
    Ii := (21 * I) mod 55;
    FSeedArray[Ii] := Mk;
    Mk := Mj - Mk;
    if Mk < 0 then Inc(Mk, MBig);
    Mj := FSeedArray[Ii];
  end;
  for K := 1 to 4 do
    for I := 1 to 55 do
    begin
      Dec(FSeedArray[I], FSeedArray[1 + (I + 30) mod 55]);
      if FSeedArray[I] < 0 then Inc(FSeedArray[I], MBig);
    end;
  FNext := 0;
  FNextP := 21;
end;

function TNetRandom.InternalSample: Int32;
begin
  Inc(FNext);
  if FNext >= 56 then FNext := 1;
  Inc(FNextP);
  if FNextP >= 56 then FNextP := 1;
  Result := FSeedArray[FNext] - FSeedArray[FNextP];
  if Result = High(Int32) then Dec(Result);
  if Result < 0 then Inc(Result, High(Int32));
  FSeedArray[FNext] := Result;
end;

function TNetRandom.NextDouble: Double;
var
  Scale: Double;
begin
  Scale := 1;
  Scale := Scale / High(Int32);
  Result := InternalSample * Scale;
end;

function TNetRandom.Next(AMinValue, AMaxValue: Int32): Int32;
begin
  Result := Trunc(NextDouble * (Int64(AMaxValue) - AMinValue)) + AMinValue;
end;

// The C library's log, cos and exp, which .NET's Math.Log, Math.Cos and Math.Exp call on Linux
// (FPC's own Ln, Cos and Exp compute in 80-bit extended precision and can round differently).
function CLog(AX: Double): Double; cdecl; external 'm' name 'log';
function CCos(AX: Double): Double; cdecl; external 'm' name 'cos';
function CExp(AX: Double): Double; cdecl; external 'm' name 'exp';

var
  Rng: TNetRandom;

function Uniform: Double;
begin
  Result := Rng.NextDouble;
end;

function Gauss: Double;
var
  U1, U2: Double;
begin
  U1 := 1 - Uniform;  // 1 - u keeps log() away from 0
  U2 := Uniform;
  Result := Sqrt(-2 * CLog(U1)) * CCos(2 * Double(Pi) * U2);
end;

const
  NStores = 25;
  TrueIntercept: Double = 2.0;
  TrueSlope: Double = 0.3;

var
  Penguins, SexDf, Df, Population, Store0: TDataFrame;
  SexModel, StoreModel: TGlmm;
  IsMale, BodyMass, Store, Promo, Visits: TArrayOfDouble;
  Island, Labels: TArrayOfString;
  StoreEffect: array[0..NStores - 1] of Double;
  I: Int64;
  N, S, Day, NDays, K, Idx: Integer;
  PromoIntensity, Lam, Threshold, P: Double;
begin
  // Flush Output after every WriteLn, so it interleaves with what the C++ library prints.
  TextRec(Output).FlushFunc := TextRec(Output).InOutFunc;
  Rng := TNetRandom.Create(4242);
  Df := TDataFrame.Create;
  Population := TDataFrame.Create;
  Store0 := TDataFrame.Create;
  StoreModel := nil;
  try
    WriteLn('=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================');
    // Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
    // confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
    // is not tied to location, so this is a well-behaved fit.
    Penguins := Data.Penguins;
    SexDf := TDataFrame.Create;
    try
      IsMale := nil;
      BodyMass := nil;
      Island := nil;
      for I := 0 to Penguins.Nrows - 1 do
      begin
        if Penguins.IsNull('sex', I) or Penguins.IsNull('body_mass_g', I) or Penguins.IsNull('island', I) then
          Continue;
        N := Length(IsMale);
        SetLength(IsMale, N + 1);
        SetLength(BodyMass, N + 1);
        SetLength(Island, N + 1);
        IsMale[N] := IfThen(Penguins.StringAt('sex', I) = 'male', 1.0, 0.0);
        BodyMass[N] := Penguins.NumericAt('body_mass_g', I);
        Island[N] := Penguins.StringAt('island', I);
      end;
      SexDf.AddNumericColumn('is_male', IsMale);
      SexDf.AddNumericColumn('body_mass_g', BodyMass);
      SexDf.AddStringColumn('island', Island);
      SexModel := TGlmm.Create(SexDf, 'is_male ~ body_mass_g + (1 | island)', 'binomial');
      SexModel.PrintSummary;
      SexModel.Free;
    finally
      SexDf.Free;
      Penguins.Free;
    end;

    WriteLn(#10'=================== Poisson mixed model on simulated multi-site count data ===================');
    // 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
    // the baseline traffic and the promo's effectiveness vary by store.
    for S := 0 to NStores - 1 do
      StoreEffect[S] := Gauss * Double(0.4);

    Store := nil;
    Promo := nil;
    Visits := nil;
    for S := 0 to NStores - 1 do
    begin
      NDays := 15 + Rng.Next(0, 11);
      for Day := 0 to NDays - 1 do
      begin
        PromoIntensity := Uniform * 3;
        Lam := CExp(TrueIntercept + StoreEffect[S] + TrueSlope * PromoIntensity);
        // Knuth's Poisson sampler.
        Threshold := CExp(-Lam);
        K := 0;
        P := 1.0;
        repeat
          Inc(K);
          P := P * Uniform;
        until not (P > Threshold);
        N := Length(Store);
        SetLength(Store, N + 1);
        SetLength(Promo, N + 1);
        SetLength(Visits, N + 1);
        Store[N] := S;
        Promo[N] := PromoIntensity;
        Visits[N] := K - 1;
      end;
    end;
    Df.AddNumericColumn('store', Store);
    Df.AddNumericColumn('promo', Promo);
    Df.AddNumericColumn('visits', Visits);
    StoreModel := TGlmm.Create(Df, 'visits ~ promo + (1 | store)', 'poisson');

    StoreModel.PrintSummary;

    WriteLn(#10'True generating values: intercept=', Fmt(TrueIntercept), ', slope=', Fmt(TrueSlope),
            ', random-intercept SD (log scale)=0.4');

    WriteLn(#10'--- BLUPs for a few stores ---');
    Labels := StoreModel.GroupLabels;
    for Idx := 0 to 2 do
      WriteLn('store ', Labels[Idx], ': intercept shift=', Fmt(StoreModel.RandomEffectsForGroup(Idx)[0]));

    WriteLn(#10'--- Prediction: population-level vs. store-adjusted ---');
    Population.AddNumericColumn('promo', [1.5]);
    Store0.AddNumericColumn('promo', [1.5]);
    Store0.AddNumericColumn('store', [0.0]);
    WriteLn('promo=1.5, unseen store:   ', Fmt(StoreModel.Predict(Population)[0]), ' expected visits (fixed effects only)');
    WriteLn('promo=1.5, store 0 (known): ', Fmt(StoreModel.Predict(Store0)[0]), ' expected visits (fixed effects + store 0''s BLUP)');
  finally
    StoreModel.Free;
    Store0.Free;
    Population.Free;
    Df.Free;
    Rng.Free;
  end;
end.
