--  CMD: just TARGET=pca_ex run-ada
--  PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Pca_Ex is
   Iris            : Data_Frame := Data.Iris;
   Feature_Columns : constant String_Vector := ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"];
   Species         : String_Vector;
   Scaled          : Pca;
   Unscaled        : Pca;
   Table           : Data_Frame;
   Figure          : R_Plot;
begin
   --  species labels, for the grouped (colored-by-species) scores plot
   for I in 0 .. Iris.Nrows - 1 loop
      Species.Append (Iris.String_At ("Species", I));
   end loop;

   Put_Line ("=================== PCA on iris (scaled) ===================");
   Scaled := New_Pca (Iris, Feature_Columns, True, True);
   Scaled.Print_Summary;

   Put_Line (ASCII.LF & "PC1 loadings (which original features drive it):");
   declare
      Names    : constant String_Vector := Scaled.Feature_Names;
      Loadings : constant Long_Float_Vector := Scaled.Component_Loadings (0);
   begin
      for I in 1 .. Natural'Min (Natural (Names.Length), Natural (Loadings.Length)) loop
         Put_Line ("  " & Names (I) & ": " & F (Loadings (I), 4));
      end loop;
   end;

   Table := Scaled.Scores_Frame;
   Put_Line (ASCII.LF & "Training scores as a DataFrame:" & ASCII.LF & Table.Image);
   Free (Table);

   Figure := Scaled.Plot_Scores_Grouped (Species);
   Figure.Save_Svg ("pca_iris_scores.svg");
   Free (Figure);
   Put_Line (ASCII.LF & "Scores scatter (colored by species) saved as pca_iris_scores.svg");

   Figure := Scaled.Plot_Scree;
   Figure.Save_Svg ("pca_iris_scree.svg");
   Free (Figure);
   Put_Line ("Scree plot saved as pca_iris_scree.svg");

   Put_Line (ASCII.LF & "=================== PCA on iris (unscaled) ===================");
   Unscaled := New_Pca (Iris, Feature_Columns, True, False);
   Put_Line ("PC1 explains " & F (100.0 * Unscaled.Explained_Variance_Ratio.First_Element, 2) & "% of variance (vs. "
             & F (100.0 * Scaled.Explained_Variance_Ratio.First_Element, 2)
             & "% scaled) -- Sepal.Length's larger raw variance dominates the unscaled covariance matrix.");

   Free (Unscaled);
   Free (Scaled);
   Free (Iris);
end Pca_Ex;
