#pragma once

/// @file einsum.hpp
/// @brief Einstein summation (TF Einsum / numpy.einsum).
///
/// Equations use one letter per axis, operands separated by commas, and an
/// optional "->output" (without it the output is the letters that appear once,
/// sorted, after any "..."). "..." stands for broadcast batch axes. Repeated
/// letters within one operand take a diagonal ("ii->i", "ii->").
///
/// einsum lowers to transpose, reshape, diagonal, sum, mul and batched matmul,
/// so it is differentiable to any order and uses the GEMM backend:
///   "ij,jk->ik", "bij,bjk->bik", "bhqd,bhkd->bhqk", "...ij,...jk->...ik",
///   "i,i->", "i,j->ij", "sat,t->sa", "ij->ji", "ii->i", "ij->", ...
/// More than two operands are contracted left to right.

#include <string>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

/// einsum over any number of operands.
Tensor einsum(const std::string& equation, const std::vector<Tensor>& operands);

/// One-operand form: transposes, diagonals, traces and sums ("ij->ji", "ii->", "ij->j").
Tensor einsum(const std::string& equation, const Tensor& a);

/// Two-operand form ("ij,jk->ik", "bhqd,bhkd->bhqk", "...ij,...jk->...ik").
Tensor einsum(const std::string& equation, const Tensor& a, const Tensor& b);

}  // namespace datamunge::nn
