classdef LinalgMatrix < datamunge.NativeObject
  % LinalgMatrix  Binding counterpart of Mat/SpMat with shared, immutable expression operands. Transpose, multiplication and inverse actions defer native computation until eval(), values() or text(). Handles keep temporary operands alive across garbage collection. Sparse products use CSR; solves use the canonical dense LU/QR implementation. Concurrent const evaluation is safe; no evaluation cache is mutated.
  %
  %   obj = datamunge.LinalgMatrix (values, rows, cols)
  %   obj = datamunge.LinalgMatrix (values, rows, cols, sparse)
  %
  %   Example:
  %     dm::LinalgMatrix A({2, 0, 0, 4}, 2, 2), b({6, 8}, 2, 1);
  %     auto x = A.inverse_action().multiply(b);
  %     std::print("{:.0f}\n", x.values()[0]); // 3
  %
  %   Methods:
  %     rows  Read the deferred result's row count without evaluating.
  %     cols  Read the deferred result's column count without evaluating.
  %     is_sparse  Inspect the deferred result's storage policy without evaluating.
  %     transpose_  Form a lazy transpose, retaining shared ownership of this operand.
  %     multiply  Form a lazy matrix product, fusing preceding inverse actions into a solve. An expression solve(A) * B * C evaluates B * C before one native solve.
  %     inverse_action  Form a deferred square inverse action, without factorisation or inversion. Multiplication turns this into a solve; evaluating it alone forms an explicit inverse.
  %     solve  Evaluate a square system with LU or a tall least-squares system with QR.
  %     eval  Evaluate once into an owned snapshot with no deferred operands.
  %     values  Evaluate and copy all coefficients in row-major order.
  %     text  Evaluate once and produce the native capped Mat/SpMat preview.
  %
  %   C++: class datamunge::LinalgMatrix
  %
  %   See also: datamunge.Mat, datamunge.SpMat, datamunge.solve()
  methods
    function obj = LinalgMatrix (varargin)
      % LinalgMatrix  Construct a C++ datamunge::LinalgMatrix.
      %
      %   obj = datamunge.LinalgMatrix (values, rows, cols)
      %   obj = datamunge.LinalgMatrix (values, rows, cols, sparse)
      %
      %   Example:
      %     dm::LinalgMatrix A({1, 0}, 1, 2, true);
      %     std::print("{}\n", A.cols()); // 2
      %
      %   Construct an owned dense or CSR matrix from row-major values.
      %
      %   Inputs:
      %     values  double vector: Exactly rows * cols coefficients; sparse zero entries are omitted.
      %     rows  integer: Number of rows, including zero.
      %     cols  integer: Number of columns, including zero.
      %     sparse  logical, default false: Select CSR storage, default false.
      %
      %   Errors:
      %     datamunge:invalid_argument  If dimensions overflow or do not match values. (C++ std::invalid_argument)
      %
      %   C++: datamunge::LinalgMatrix::LinalgMatrix(const std::vector<double> & values, std::size_t rows, std::size_t cols, bool sparse = false)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1171, false, varargin{:});
    end
    function r = rows (obj)
      % rows  Read the deferred result's row count without evaluating.
      %
      %   r = obj.rows ()
      %
      %   Output:
      %     r  integer: Number of rows.
      %
      %   C++: std::size_t datamunge::LinalgMatrix::rows() const
      r = __datamunge__(1161, obj);
    end
    function r = cols (obj)
      % cols  Read the deferred result's column count without evaluating.
      %
      %   r = obj.cols ()
      %
      %   Output:
      %     r  integer: Number of columns.
      %
      %   C++: std::size_t datamunge::LinalgMatrix::cols() const
      r = __datamunge__(1162, obj);
    end
    function r = is_sparse (obj)
      % is_sparse  Inspect the deferred result's storage policy without evaluating.
      %
      %   r = obj.is_sparse ()
      %
      %   Output:
      %     r  logical: True for CSR results; inverse actions and solves return dense results.
      %
      %   C++: bool datamunge::LinalgMatrix::is_sparse() const
      r = __datamunge__(1163, obj);
    end
    function r = transpose_ (obj)
      % transpose_  Form a lazy transpose, retaining shared ownership of this operand.
      %
      %   r = obj.transpose_ ()
      %
      %   Example:
      %     dm::LinalgMatrix A({1, 2}, 1, 2);
      %     std::print("{}\n", A.transpose().rows()); // 2
      %
      %   Output:
      %     r  datamunge.LinalgMatrix: Deferred transpose handle.
      %
      %   C++: LinalgMatrix datamunge::LinalgMatrix::transpose() const
      r = __datamunge__(1164, obj);
    end
    function r = multiply (obj, other)
      % multiply  Form a lazy matrix product, fusing preceding inverse actions into a solve. An expression solve(A) * B * C evaluates B * C before one native solve.
      %
      %   r = obj.multiply (other)
      %
      %   Example:
      %     dm::LinalgMatrix A({1, 2}, 1, 2), b({3, 4}, 2, 1);
      %     std::print("{:.0f}\n", A.multiply(b).values()[0]); // 11
      %
      %   Inputs:
      %     other  datamunge.LinalgMatrix: Right operand, retained by shared ownership.
      %
      %   Output:
      %     r  datamunge.LinalgMatrix: Deferred product or solve handle.
      %
      %   Errors:
      %     datamunge:invalid_argument  If inner dimensions differ. (C++ std::invalid_argument)
      %
      %   C++: LinalgMatrix datamunge::LinalgMatrix::multiply(const LinalgMatrix & other) const
      r = __datamunge__(1165, obj, other);
    end
    function r = inverse_action (obj)
      % inverse_action  Form a deferred square inverse action, without factorisation or inversion. Multiplication turns this into a solve; evaluating it alone forms an explicit inverse.
      %
      %   r = obj.inverse_action ()
      %
      %   Example:
      %     dm::LinalgMatrix A({2}, 1, 1);
      %     std::print("{:.1f}\n", A.inverse_action().values()[0]); // 0.5
      %
      %   Output:
      %     r  datamunge.LinalgMatrix: Deferred inverse-action handle.
      %
      %   Errors:
      %     datamunge:invalid_argument  If this matrix is not square. (C++ std::invalid_argument)
      %
      %   C++: LinalgMatrix datamunge::LinalgMatrix::inverse_action() const
      r = __datamunge__(1166, obj);
    end
    function r = solve (obj, rhs)
      % solve  Evaluate a square system with LU or a tall least-squares system with QR.
      %
      %   r = obj.solve (rhs)
      %
      %   Example:
      %     dm::LinalgMatrix A({1, 1, 1}, 3, 1), b({1, 2, 3}, 3, 1);
      %     std::print("{:.0f}\n", A.solve(b).values()[0]); // 2
      %
      %   Inputs:
      %     rhs  datamunge.LinalgMatrix: Matrix of right-hand sides with rows() rows.
      %
      %   Output:
      %     r  datamunge.LinalgMatrix: Owned dense solution snapshot.
      %
      %   Errors:
      %     datamunge:invalid_argument  For incompatible/wide shapes or nonfinite solve inputs. (C++ std::invalid_argument)
      %     datamunge:native  For singular or rank-deficient systems. (C++ std::runtime_error)
      %
      %   C++: LinalgMatrix datamunge::LinalgMatrix::solve(const LinalgMatrix & rhs) const
      r = __datamunge__(1167, obj, rhs);
    end
    function r = eval (obj)
      % eval  Evaluate once into an owned snapshot with no deferred operands.
      %
      %   r = obj.eval ()
      %
      %   Example:
      %     dm::LinalgMatrix A({1, 2}, 1, 2);
      %     auto snapshot = A.transpose().eval();
      %     std::print("{}\n", snapshot.rows()); // 2
      %
      %   Output:
      %     r  datamunge.LinalgMatrix: Materialised matrix handle, sharing immutable storage when already concrete.
      %
      %   Errors:
      %     datamunge:invalid_argument  For nonfinite solve inputs. (C++ std::invalid_argument)
      %     datamunge:native  For singular or rank-deficient systems. (C++ std::runtime_error)
      %
      %   C++: LinalgMatrix datamunge::LinalgMatrix::eval() const
      r = __datamunge__(1168, obj);
    end
    function r = values (obj)
      % values  Evaluate and copy all coefficients in row-major order.
      %
      %   r = obj.values ()
      %
      %   Example:
      %     dm::LinalgMatrix A({1, 2}, 1, 2);
      %     std::print("{:.0f}\n", A.values()[1]); // 2
      %
      %   Output:
      %     r  double vector: Dense vector, including sparse zeros.
      %
      %   Errors:
      %     datamunge:invalid_argument  For nonfinite solve inputs. (C++ std::invalid_argument)
      %     datamunge:native  For singular or rank-deficient systems. (C++ std::runtime_error)
      %
      %   C++: std::vector<double> datamunge::LinalgMatrix::values() const
      r = __datamunge__(1169, obj);
    end
    function r = text (obj)
      % text  Evaluate once and produce the native capped Mat/SpMat preview.
      %
      %   r = obj.text ()
      %
      %   Example:
      %     dm::LinalgMatrix A({1}, 1, 1);
      %     std::print("{}\n", A.text());
      %     // Mat 1 x 1
      %     //      [,0]
      %     // [0,]    1
      %
      %   Output:
      %     r  char row: Shape, storage information and at most ten rows/eight columns.
      %
      %   Errors:
      %     datamunge:invalid_argument  For nonfinite solve inputs. (C++ std::invalid_argument)
      %     datamunge:native  For singular or rank-deficient systems. (C++ std::runtime_error)
      %
      %   C++: std::string datamunge::LinalgMatrix::text() const
      r = __datamunge__(1170, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::LinalgMatrix. Do not edit.
