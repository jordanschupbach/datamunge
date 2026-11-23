--  Smoke tests of the Rosetta-generated Ada binding (just test-ada).
with Ada.Command_Line;
with Ada.Exceptions; use Ada.Exceptions;
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Datamunge_Test_Callbacks; use Datamunge_Test_Callbacks;

procedure Test_Datamunge is
   Checks : Natural := 0;

   procedure Check (Ok : Boolean; Message : String) is
   begin
      Checks := Checks + 1;
      if not Ok then
         Put_Line ("check failed: " & Message);
         Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
         raise Program_Error with Message;
      end if;
   end Check;

   V, Seq : Long_Float_Vector;
   P : Long_Float_Long_Float_Pair;
   Native : Callback_Access := New_Callback_Access;
   Doubler : Callback_Access := new Doubler_Record;
   Failing : Callback_Access := new Failing_Record;
   Frame, Iris : Data_Frame;
   Raised : Boolean;
begin
   --  Smoke test: call into the native library.
   Hello;

   --  std::vector<double> and std::pair<double,double> round trips.
   V := Make_Dvector (1.0, 2.0, 3.0);
   Check (Natural (V.Length) = 3 and then V (1) = 1.0 and then V (3) = 3.0, "std::vector<double> is a Long_Float_Vector");
   Check (Sum_Dvector (V) = 6.0, "Sum_Dvector");
   Check (Sum_Dvector ([0.5, 1.5]) = 2.0, "aggregates convert to std::vector");
   P := Make_Dpair (1.25, 2.75);
   Check (P.First = 1.25 and then P.Second = 2.75, "std::pair is a record");
   Check (Sum_Dpair (P) = 4.0, "Sum_Dpair");

   --  The C++ Callback class, and an Ada type extension overriding its virtual Call (a director).
   Check (Call_With_Callback (3.0, Native) = 3.0, "native Callback");
   Check (Sum_Dvector (Map_Dvector_With_Callback ([1.0, 2.0, 3.0], Native)) = 6.0, "native map");
   Doubler.Construct_Callback_Access;
   Check (Call_With_Callback (3.0, Doubler) = 6.0, "the Ada override runs");
   Check (Sum_Dvector (Map_Dvector_With_Callback ([1.0, 2.0, 3.0], Doubler)) = 12.0, "override from C++ loops");
   Free (Native);
   Free (Doubler);

   --  An Ada exception inside an override is raised again after C++ returns.
   Failing.Construct_Callback_Access;
   Raised := False;
   begin
      Check (Call_With_Callback (1.0, Failing) = 0.0, "unreachable");
   exception
      when E : From_Ada => Raised := Exception_Message (E) = "from Ada";
   end;
   Check (Raised, "the Ada exception propagates");
   Free (Failing);

   --  C++ exceptions raise the matching exception, with their kind.
   Raised := False;
   begin
      V := Runif (1.0, 0.0, 3);
   exception
      when E : Invalid_Argument_Error => Raised := Error_Kind (E) = Invalid_Argument;
   end;
   Check (Raised, "C++ exceptions raise Invalid_Argument_Error");

   --  Classes, methods and the policy namespace Data.
   Frame := New_Data_Frame;
   Frame.Add_Numeric_Column ("x", [1.0, 2.0, 3.0]);
   Check (Frame.Nrows = 3, "Data_Frame");
   Free (Frame);
   Seq := Seql (0.0, 1.0, 5);
   Check (Natural (Seq.Length) = 5 and then Seq (2) = 0.25 and then Seq (5) = 1.0, "Seql");
   Iris := Data.Iris;
   Check (Iris.Nrows = 150, "the Data namespace");
   Free (Iris);

   Put_Line ("test_datamunge.adb: all" & Checks'Image & " checks passed");
end Test_Datamunge;
