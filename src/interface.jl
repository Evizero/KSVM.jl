# ==========================================================================
# Main interface for linear Support Vector Machines

"""
`svm(X, y⃗; nargs...) → SVM`

Description
============

Fits a Support Vector Machine (SVM) to the given training data `X` and `y⃗`.
Depending on the specified parameters in `nargs`, the SVM can be trained to
perform various forms of regression or classification tasks.
There are a number of parameters that can be configured to the individual needs,
such as the type of lossfunction, the type of regularization, or the algorithm 
used for training the SVM in the primal/dual.

Usage
======

    svm(X, y; solver = DualCD(), kernel = ScalarProductKernel(), loss = HingeLoss(), regtype = L2Penalty, bias = 1, C = 1, dual = false, maxiter = 1000, ftol = 1.0e-6, xtol = 1.0e-6, grtol = 1.0e-9, callback = nothing, show_trace = false, nargs...)

    svm(X, Y; nargs...) do t, alpha, v, g
      # this is the callback function
    end

Arguments
==========

- **`X`** : The input matrix of the training set, with the observations as columns.
It can be either a dense or sparse matrix of floatingpoint numbers.
Note that it is generally advised to center and scale your data before training a SVM.

- **`y⃗`** : The target vector of the training set. It has to have the same length as `X` has columns.
In the case of classification, it can be either a vector with elements yᵢ ∈ {-1.0, 1.0}
for two class prediction, or a vector with elements yᵢ ∈ N for multiclass prediction.

- **`solver`** : The concrete algorithm to train the SVM. Depending on the algorithm
this will either solve the linear SVM in the primal or dual formulation.
In contrast to other frameworks, the algorithm does not determine if the solution 
of the primal or dual problem is returned (see Details).

- **`kernel`** : The kernel that should be used. The ScalarProductKernel denotes the special
case of a linear SVM and allows for additional functionality and solvers that are specialized
for the linear case.

- **`loss`** : The utilized loss function. The typical loss functions for SVMs are
`L1HingeLoss()` (L1-SVM), `L2HingeLoss()` (L2-SVM), or `SmoothedL1HingeLoss(h)` for classification,
and `EpsilonInsLoss(e)` or `L2EpsilonInsLoss(e)` for support vector regression.
Note that in general each solvers is only able to deal with a subset of those
loss functions and some combinations might not be supported.
For example the implementation of `DualCD` does only support
`HingeLoss` and `L2HingeLoss`.

- **`regtype`** : The type of regularization that should be used. In general this can either
be `L1Penalty` or `L2Penalty`. Note that not all solver support `L1Penalty`.

- **`bias`** : The scaling factor of the bias. If set to 0, no intercept will be fitted.

- **`C`** : The penalty parameter. It plays a role similar to λ⁻¹ in Empirical Risk Minimization.
Conceptually, it denotes the trade off between model complexity and the training error.
Larger C will increase the penalization of training errors and thus lead to behavior
more similar to a hard margin classifier.

- **`dual`** : Boolean that specifies if the solution to the dual problem should be returned
instead of the solution to the primal (default = false).

- **`iterations`** : Specifies the maximum number of iterations before the solver exits with the
current (and probably suboptimal) solution.

- **`ftol`** : Specifies the tolerance for the change of the objective function value.

- **`xtol`** : Specifies the tolerance for the change of the solution.

- **`grtol`** : Specifies the tolerance for the change of the gradient norm.

- **`callback`** : The optional callback function with signature `f(i, w, v, G)`. If a callback is specified, then it will be called
every iteration with the following four parameters in order:
1) `i` … the current iteration number,
2) `w` … the current coefficients (i.e. α if dual=true, or θ if dual=false),
3) `v` … the current objective value
4) `G` … the current gradient.
Note: Some solver support early stopping if the callback function returns the symbol `:Exit`.

- **`show_trace`** : Instead of (or additional to) the callback function, the user can enable the output of the training information to STDOUT.

- **`nargs...`** : Additional named arguments that are passed unchecked to the specified solver.
This functionality can be used to pass around special arguments that the library does not natively implement.

Details
========

In contrast to other frameworks, the `dual` parameter does not influence whether the dual or
the primal problem is solved (this is instead specified implicitly by the `solver` parameter),
but the result the user is interested in.
It is generally recommended to leave `dual = false`, unless the user is either explicitly
interested in the support vectors, or the dataset is sparse and high-dimensional.
Note: Not all solver are able to natively provide both solutions. If the algorithm is unable
to provide the desired solution type then the solution will be converted accordingly, if feasible.

Returns
========

Either a subtype of `DualSVM` or `PrimalSVM`

Author(s)
==========

- Christof Stocker (email: c.stocker@hci-kdd.org)

Examples
=========

### Iris Dataset (adapted example from SVM.jl by John Myles White)

    using KSVM
    using RDatasets
    iris = dataset("datasets", "iris")
    X = convert(Array, iris[1:100, 1:2])'  # The observations have to be in the columns
    y = iris[1:100, :Species]
    train = bitrand(size(X,2))             # Split training and testset
    model = svm(X[:, train], y[train])     # Fit the linear SVM
    acc = accuracy(model, X[:, ~train], y[~train])

### Random Data (adapted example from Regression.jl by Dahua Lin)

    using KSVM
    d = 2             # Number of input dimensions
    n = 100           # Number of pbservations
    w = randn(d+1)    # generate the weight vector
    X = randn(d, n)   # generate input features
    y = sign(X'w[1:d] + w[d+1] + 0.2 * randn(n))  # generate (noisy) response
    model = svm(X, y) # Fit the linear SVM
    y_hat = classify(model, X)

References
===========

- Hsieh, Cho-Jui, et al. "A dual coordinate descent method for large-scale
linear SVM." Proceedings of the 25th international conference on Machine learning.
ACM, 2008. DOI=10.1145/1390156.1390208, http://doi.acm.org/10.1145/1390156.1390208
"""
@inline function svm{TReg<:Penalty}(
    X::AbstractMatrix, y⃗::AbstractVector;
    kernel::Kernel = ScalarProductKernel(),
    solver::SvmSolver = DualCD(),
    loss::Loss = L2HingeLoss(),
    regtype::Type{TReg} = L2Penalty,
    C::Real = 1.,
    nargs...)
  spec = CSVM(kernel = kernel,
              loss = loss,
              regtype = regtype,
              C = C)
  fit(spec, X, y⃗; solver = solver, nargs...)
end

# ==========================================================================
# Special method overload to support the "do" notation for the callback
# res = svm(X, Y) do t, alpha, v, g
#   println("Iteration $t: $v")
# end

@inline function svm(callback::Function,
             X::AbstractMatrix, y⃗::AbstractVector;
             nargs...)
  svm(X, y⃗; callback = callback, nargs...)
end
