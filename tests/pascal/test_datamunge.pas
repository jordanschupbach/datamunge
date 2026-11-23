// Smoke tests of the Rosetta-generated Free Pascal binding (just test-pascal).
program test_datamunge;

{$mode objfpc}{$H+}

uses
  SysUtils, datamunge;

type
  TDoubler = class(TCallback)
  public
    function Call(AX: Double): Double; override;
  end;

  EFromPascal = class(Exception);

  TFailing = class(TCallback)
  public
    function Call(AX: Double): Double; override;
  end;

var
  Checks: Integer = 0;

function TDoubler.Call(AX: Double): Double;
begin
  Result := 2 * AX;
end;

function TFailing.Call(AX: Double): Double;
begin
  Result := 0;
  raise EFromPascal.Create('from Pascal');
end;

procedure Check(Ok: Boolean; const Message: String);
begin
  Inc(Checks);
  if not Ok then
  begin
    WriteLn('check failed: ', Message);
    Halt(1);
  end;
end;

var
  V, Seq: TArrayOfDouble;
  P: TPairOfDoubleAndDouble;
  Native, Doubler: TCallback;
  Frame, Iris: TDataFrame;
  Raised: Boolean;
begin
  // Smoke test: call into the native library.
  Hello;

  // std::vector<double> and std::pair<double,double> round trips.
  V := MakeDvector(1.0, 2.0, 3.0);
  Check((Length(V) = 3) and (V[0] = 1.0) and (V[2] = 3.0), 'std::vector<double> is an array of Double');
  Check(SumDvector(V) = 6.0, 'SumDvector');
  Check(SumDvector([0.5, 1.5]) = 2.0, 'open arrays convert to std::vector');
  P := MakeDpair(1.25, 2.75);
  Check((P.First = 1.25) and (P.Second = 2.75), 'std::pair is a record');
  Check(SumDpair(P) = 4.0, 'SumDpair');

  // The C++ Callback class, and a Pascal subclass overriding its virtual Call (a director).
  Native := TCallback.Create;
  Check(CallWithCallback(3.0, Native) = 3.0, 'native Callback');
  Check(SumDvector(MapDvectorWithCallback([1.0, 2.0, 3.0], Native)) = 6.0, 'native map');
  Doubler := TDoubler.Create;
  Check(CallWithCallback(3.0, Doubler) = 6.0, 'the Pascal override runs');
  Check(SumDvector(MapDvectorWithCallback([1.0, 2.0, 3.0], Doubler)) = 12.0, 'override from C++ loops');
  Native.Free;
  Doubler.Free;

  // A Pascal exception inside an override is raised again after C++ returns.
  Raised := False;
  try
    CallWithCallback(1.0, TFailing.Create);
  except
    on E: EFromPascal do Raised := E.Message = 'from Pascal';
  end;
  Check(Raised, 'the Pascal exception propagates');

  // C++ exceptions raise ENativeError with their kind.
  Raised := False;
  try
    Runif(1.0, 0.0, 3);
  except
    on E: ENativeError do Raised := E.Kind = TNativeErrorKind.InvalidArgument;
  end;
  Check(Raised, 'C++ exceptions raise ENativeError');

  // Classes, methods and the policy namespace Data.
  Frame := TDataFrame.Create;
  Frame.AddNumericColumn('x', [1.0, 2.0, 3.0]);
  Check(Frame.Nrows = 3, 'TDataFrame');
  Frame.Free;
  Seq := Seql(0.0, 1.0, 5);
  Check((Length(Seq) = 5) and (Seq[1] = 0.25) and (Seq[4] = 1.0), 'Seql');
  Iris := Data.Iris;
  Check(Iris.Nrows = 150, 'the Data namespace');
  Iris.Free;

  WriteLn('test_datamunge.pas: all ', Checks, ' checks passed');
end.
