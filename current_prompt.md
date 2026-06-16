
Other sparse methods:

Method: CG ✓ 
Requirement: SPD 
Memory/iter: O(n)
When to use: Laplacians, FEM stiffness matrices, anything symmetric+positive definite 
──────────────────────────────────────── 

Method: MINRES 
Requirement: Symmetric (can be indefinite) 
Memory/iter: O(n)
When to use: Saddle-point problems
──────────────────────────────────────── 

Method: BiCGSTAB 
Requirement: Non-symmetric
Memory/iter: O(n)
When to use: Convection-diffusion, non-symmetric FEM
────────────────────────────────────────

Method: GMRES(k)
Requirement: General
Memory/iter: O(k·n) 
When to use: Highly non-symmetric; needs restart parameter

