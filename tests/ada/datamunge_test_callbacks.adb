package body Datamunge_Test_Callbacks is
   overriding function Call (Self : in out Doubler_Record; X : Long_Float) return Long_Float is (2.0 * X);

   overriding function Call (Self : in out Failing_Record; X : Long_Float) return Long_Float is
   begin
      raise From_Ada with "from Ada";
      return X;
   end Call;
end Datamunge_Test_Callbacks;
