--  CMD: just TARGET=naive_bayes_penguins_ex run-ada
--  Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Naive_Bayes_Penguins_Ex is
   --  Note: Naive_Bayes_Classifier drops incomplete rows for its own formula columns internally
   --  when fitting, matching every other formula-based model in this library.
   Penguins      : Data_Frame := Data.Penguins;
   Model         : Naive_Bayes_Classifier;
   Bill_Only     : Naive_Bayes_Classifier;
   Table         : Data_Frame;
   Predictions   : String_Vector;
   Misclassified : Long_Long_Integer := 0;
   Columns       : constant String_Vector := ["species", "bill_length_mm", "bill_depth_mm", "island", "sex"];
   Figure        : R_Plot;

   function Any_Null (I : Long_Long_Integer) return Boolean is
   begin
      for C of Columns loop
         if Penguins.Is_Null (C, I) then
            return True;
         end if;
      end loop;
      return False;
   end Any_Null;
begin
   Put_Line ("penguins: " & Img (Penguins.Nrows) & " rows x " & Img (Penguins.Ncols) & " cols" & ASCII.LF);

   --  A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
   --  predictors in one formula -- each modeled independently given the class, per the naive
   --  Bayes assumption.
   Model := New_Naive_Bayes_Classifier (Penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex");
   Model.Print_Summary;

   Table := Model.Confusion_Matrix;
   Put_Line (ASCII.LF & "Confusion matrix (rows = actual, cols = predicted):" & ASCII.LF & Table.Image);
   Free (Table);

   Put_Line (ASCII.LF & "Misclassified rows:");
   Predictions := Model.Predict (Penguins);
   for I in 0 .. Penguins.Nrows - 1 loop
      if not Any_Null (I) then
         declare
            Actual    : constant String := Penguins.String_At ("species", I);
            Predicted : constant String := Predictions (Positive (I + 1));
         begin
            if Predicted /= Actual then
               Misclassified := Misclassified + 1;
               Put_Line ("  row " & Img (I) & ": bill_length=" & Fmt (Penguins.Numeric_At ("bill_length_mm", I))
                         & " bill_depth=" & Fmt (Penguins.Numeric_At ("bill_depth_mm", I))
                         & " island=" & Penguins.String_At ("island", I)
                         & " sex=" & Penguins.String_At ("sex", I)
                         & "  actual=" & Actual & "  predicted=" & Predicted);
            end if;
         end;
      end if;
   end loop;
   Put_Line (Img (Misclassified) & " misclassified (of " & Img (Penguins.Nrows) & " rows, some incomplete)");

   --  Plot_Decision_Regions requires exactly two NUMERIC predictors, so build a separate
   --  two-predictor model (bill measurements alone) just for visualization.
   Bill_Only := New_Naive_Bayes_Classifier (Penguins, "species ~ bill_length_mm + bill_depth_mm");
   Put_Line (ASCII.LF & "bill-measurements-only model training accuracy: "
             & Fmt (100.0 * Bill_Only.Training_Accuracy) & "%");
   Figure := Bill_Only.Plot_Classification (Penguins, "bill_length_mm", "bill_depth_mm");
   Figure.Save ("naive_bayes_penguins_classification.svg");
   Free (Figure);
   Figure := Bill_Only.Plot_Decision_Regions ("bill_length_mm", "bill_depth_mm");
   Figure.Save ("naive_bayes_penguins_decision_regions.svg");
   Free (Figure);
   Put_Line ("Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg");

   Free (Bill_Only);
   Free (Model);
   Free (Penguins);
end Naive_Bayes_Penguins_Ex;
