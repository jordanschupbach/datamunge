--  CMD: just TARGET=datamunge_ex run-ada
--  Smoke test of the basic bindings: vectors, pairs and a native Callback object.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Datamunge_Ex is
   Tab : constant Character := ASCII.HT;
   V, V2 : Long_Float_Vector;
   P : Long_Float_Long_Float_Pair;
   Cb : Callback_Access := New_Callback_Access;
begin
   Hello;

   V := Make_Dvector (1.0, 2.0, 3.0);
   Put_Line ("sum_dvector:" & Tab & Fmt (Sum_Dvector (V)));

   P := Make_Dpair (1.25, 2.75);
   Put_Line ("sum_dpair:" & Tab & Fmt (Sum_Dpair (P)));

   Put_Line ("call_with_callback(3.0):" & Tab & Fmt (Call_With_Callback (3.0, Cb)));
   V2 := Map_Dvector_With_Callback (Make_Dvector (1.0, 2.0, 3.0), Cb);
   Put_Line ("sum_dvector(map_dvector_with_callback(1,2,3)):" & Tab & Fmt (Sum_Dvector (V2)));
   Free (Cb);
end Datamunge_Ex;
