--  Ada type extensions of datamunge.Callback for test_datamunge.adb (a library-level package, so the
--  objects can be allocated for Callback_Access).
with Datamunge; use Datamunge;

package Datamunge_Test_Callbacks is
   From_Ada : exception;

   type Doubler_Record is new Callback_Record with null record;
   overriding function Call (Self : in out Doubler_Record; X : Long_Float) return Long_Float;

   type Failing_Record is new Callback_Record with null record;
   overriding function Call (Self : in out Failing_Record; X : Long_Float) return Long_Float;
end Datamunge_Test_Callbacks;
