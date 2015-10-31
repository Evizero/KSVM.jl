abstract SVM{TSpec <: SVMSpec} <: RegressionModel

# ==========================================================================

svmModel(spec::SVMSpec, solution::Solution, predmodel::PredictionModel, X::AbstractMatrix, Y::AbstractArray) =
  primalSVMModel(spec, solution, predmodel, X, Y)

svmModel(spec::SVMSpec, solution::DualSolution, predmodel::PredictionModel, X::AbstractMatrix, Y::AbstractArray) =
  dualSVMModel(spec, solution, predmodel, X, Y)

# ==========================================================================

"""
`PrimalSVM <: SVM`

Description
============

The result to the primal problem of a support vector machine.
It is the return value of the `svm` function if the parameter `dual = false`

Fields
=======

- **`params(this)`** : The structure and parameters of the utilized support vector machine

- **`nsv(this)`** : The number of support vectors

- **`svindex(this)`** : The indicies (i) of the support vectors within the training set

- **`features(this)`** : The full training set observations

- **`targets(this)`** : The full training set targets

- **`predmodel(this)`** : The underlying prediction model. It defines if the intercept
is present, as well as the input and output dimensions (univariate vs multivariate prediction)

- **`iterations(this)`** : Number of iterations used to fit the model.

- **`isconverged(this)`** : true, if the algorithm converged during training.

- **`fval(this)`** : The final objective value achieved during training.

- **`coef(this)`** : The fitted coefficients w⃗.

- **`details(this)`** : The training information returned by the solver itself.
It stores the final objective value, the number of used iterations,
the fitted coefficients w⃗, and a boolean that states if the algorithm was able to converge.

Methods
========

- **`predict(this, X)`** : predicts the response for the given observations in `X`.
Note that `X` should be of the same type as the data used for training the SVM.

- **`classify(this, X)`** : predicts the response for the given observations in `X`
and performs the decision function on the result.
Note that `X` should be of the same type as the data used for training the SVM.

- **`accuracy(this, X, y)`** : computes the accuracy by calculating the predictions
of `X` and comparing the results to `y`.
Note that `X` and `y` should have the same number of observations.

See also
=========

`svm`, `DualSVM`, `CSVM`

"""
abstract PrimalSVM{TSpec<:SVMSpec} <: SVM{TSpec}

"""
`DensePrimalSVM <: PrimalSVM`

See `PrimalSVM`
"""
type DensePrimalSVM{TSpec<:SVMSpec, TPred<:LinSvmPred, XT<:DenseMatrix, YT<:AbstractVector} <: PrimalSVM{TSpec}
  params::TSpec
  details::Solution
  predmodel::TPred
  nsv::Int
  svindex::Vector{Int}
  Xtrain::XT
  Ytrain::YT
end

"""
`SparsePrimalSVM <: PrimalSVM`

See `PrimalSVM`
"""
type SparsePrimalSVM{TSpec<:SVMSpec, TPred<:LinSvmPred, XT<:AbstractSparseMatrix, YT<:AbstractVector} <: PrimalSVM{TSpec}
  params::TSpec
  details::Solution
  predmodel::TPred
  nsv::Int
  svindex::Vector{Int}
  Xtrain::XT
  Ytrain::YT
end

# ==========================================================================

"""
`DualSVM <: SVM`

Description
============

The result to the dual problem of a support vector machine.
It is the return value of the `svm` function if the parameter `dual = true`

Fields
=======

- **`params(this)`** : The structure and parameters of the utilized support vector machine

- **`nsv(this)`** : The number of support vectors

- **`svindex(this)`** : The indicies (i) of the support vectors within the training set

- **`features(this)`** : The full training set observations to which the model was fit

- **`targets(this)`** : The full training set targets to which the model was fit

- **`predmodel(this)`** : The underlying prediction model. It defines if the intercept
is present, as well as the input and output dimensions (univariate vs multivariate prediction)

- **`iterations(this)`** : Number of iterations used to fit the model.

- **`isconverged(this)`** : true, if the algorithm converged during training.

- **`fval(this)`** : The final objective value achieved during training.

- **`coef(this)`** : The coefficients (α) of the support vectors

- **`xsv(this)`** : The support vectors (if dense), or the training set observations (if sparse)

- **`ysv(this)`** : The targets of the support vectors

- **`details(this)`** : The training information returned by the solver itself.
It includes the final objective value, the number of used iterations,
the full coefficient vector α, the bias if present, and a boolean that
states if the algorithm was able to converge.

Methods
========

- **`predict(this, X)`** : predicts the response for the given observations in `X`.
Note that `X` should be of the same form as the data used for training the SVM.

- **`classify(this, X)`** : predicts the response for the given observations in `X`
and performs the decision function on the result.
Note that `X` should be of the same type as the data used for training the SVM.

- **`accuracy(this, X, y)`** : computes the accuracy by calculating the predictions
of `X` and comparing the results to `y`.
Note that `X` and `y` should have the same number of observations.

See also
=========

`svm`, `PrimalSVM`, `CSVM`

"""
abstract DualSVM{TSpec<:SVMSpec} <: SVM{TSpec}

"""
`DenseDualSVM <: DualSVM`

See `DualSVM`
"""
type DenseDualSVM{TSpec<:SVMSpec, TPred<:LinSvmPred, XT<:DenseMatrix, YT<:AbstractVector} <: DualSVM{TSpec}
  params::TSpec
  details::DualSolution
  predmodel::TPred
  nsv::Int
  alpha::Vector{Float64}
  svindex::Vector{Int}
  Xsv::Vector{ContiguousView{Float64,1,Array{Float64,2}}}
  Ysv::Vector{Float64}
  Xtrain::XT
  Ytrain::YT
end

"""
`SparseDualSVM <: DualSVM`

See `DualSVM`
"""
type SparseDualSVM{TSpec<:SVMSpec, TPred<:LinSvmPred, XT<:AbstractSparseMatrix, YT<:AbstractVector} <: DualSVM{TSpec}
  params::TSpec
  details::DualSolution
  predmodel::TPred
  nsv::Int
  alpha::Vector{Float64}
  svindex::Vector{Int}
  Xsv::XT
  Ysv::Vector{Float64}
  Xtrain::XT
  Ytrain::YT
end

# ==========================================================================

function primalSVMModel{TSpec<:SVMSpec, TPred<:LinSvmPred, TReal<:Real}(
    params::TSpec,
    s::Solution,
    predmodel::TPred,
    Xtrain::StridedMatrix,
    Ytrain::AbstractVector{TReal})
  n = size(Xtrain, 2)
  p = EmpiricalRisks.predict(predmodel, s.sol, Xtrain)
  l = zeros(n)
  @inbounds for i = 1:n
    lc = EmpiricalRisks.value(params.loss, p[i] - sign(p[i]) * 0.0005, Float64(Ytrain[i]))
    lr = EmpiricalRisks.value(params.loss, p[i] + sign(p[i]) * 0.0005, Float64(Ytrain[i]))
    l[i] = lc != 0 || lr != 0
  end
  svindex = find(l)
  nsv = length(svindex)
  DensePrimalSVM{TSpec, TPred, typeof(Xtrain), typeof(Ytrain)}(params, s, predmodel, nsv, svindex, Xtrain, Ytrain)
end

function primalSVMModel{TSpec<:SVMSpec, TPred<:LinSvmPred, TReal<:Real}(
    params::TSpec,
    s::Solution,
    predmodel::TPred,
    Xtrain::AbstractSparseMatrix,
    Ytrain::AbstractVector{TReal})
  n = size(Xtrain, 2)
  w = s.sol
  p = if typeof(predmodel) <: AffinePred
    fill(w[end] * predmodel.bias, size(Ytrain))
  else
    zeros(size(Ytrain))
  end
  @inbounds for i = 1:n
    tstart = Xtrain.colptr[i]
    tstop  = Xtrain.colptr[i+1] - 1
    for j = tstart:tstop
      k = Xtrain.rowval[j]
      p[i] += Xtrain.nzval[j] * w[k]
    end
  end
  l = zeros(n)
  @inbounds for i = 1:n
    lc = EmpiricalRisks.value(params.loss, p[i] - sign(p[i]) * 0.0005, Float64(Ytrain[i]))
    lr = EmpiricalRisks.value(params.loss, p[i] + sign(p[i]) * 0.0005, Float64(Ytrain[i]))
    l[i] = lc != 0 || lr != 0
  end
  svindex = find(l)
  nsv = length(svindex)
  SparsePrimalSVM{TSpec, TPred, typeof(Xtrain), typeof(Ytrain)}(params, s, predmodel, nsv, svindex, Xtrain, Ytrain)
end

function dualSVMModel{TSpec<:SVMSpec, TPred<:LinSvmPred, TReal<:Real}(
    params::TSpec,
    s::DualSolution,
    p::TPred,
    Xtrain::StridedMatrix,
    Ytrain::AbstractVector{TReal})
  svindex = find(s.alpha)
  alpha = s.alpha[svindex]
  nsv = length(alpha)
  Xsv = Array(ContiguousView{Float64,1,Array{Float64,2}}, nsv)
  @inbounds for i in 1:nsv
    Xsv[i] = view(Xtrain, :, svindex[i])
  end
  Ysv = Ytrain[svindex]
  DenseDualSVM{TSpec, TPred, typeof(Xtrain), typeof(Ytrain)}(params, s, p, nsv, alpha, svindex, Xsv, Ysv, Xtrain, Ytrain)
end

function dualSVMModel{TSpec<:SVMSpec, TPred<:LinSvmPred, TReal<:Real}(
    params::TSpec,
    s::DualSolution,
    p::TPred,
    Xtrain::AbstractSparseMatrix,
    Ytrain::AbstractVector{TReal})
  svindex = find(s.alpha)
  alpha = s.alpha[svindex]
  nsv = length(alpha)
  Xsv = Xtrain
  Ysv = Ytrain[svindex]
  SparseDualSVM{TSpec, TPred, typeof(Xtrain), typeof(Ytrain)}(params, s, p, nsv, alpha, svindex, Xsv, Ysv, Xtrain, Ytrain)
end

# ==========================================================================

labels(svm::SVM) = labels(svc_or_svr(svm))
nobs(fit::SVM) = length(fit.Ytrain)
features(fit::SVM) = fit.Xtrain
targets(fit::SVM) = fit.Ytrain
model_response(fit::SVM) = fit.Ytrain

isclassifier(fit::SVM) = isclassifier(fit.params)
decision_function(fit::SVM) = decision_function(fit.params)

details(fit::SVM) = fit.details
isconverged(fit::SVM) = fit.details.converged
iterations(fit::SVM) = fit.details.niters
params(fit::SVM) = fit.params
fval(fit::SVM) = fit.details.fval
nsv(fit::SVM) = fit.nsv
svindex(fit::SVM) = fit.svindex
intercept(fit::SVM) = typeof(predmodel(fit)) <: Union{AffinePred, MvAffinePred}
predmodel(fit::SVM) = fit.predmodel
coef(fit::PrimalSVM) = fit.details.sol
coef(fit::DualSVM) = fit.alpha
bias(fit::DualSVM) = fit.details.bias
xsv(fit::DualSVM) = fit.Xsv
ysv(fit::DualSVM) = fit.Ysv

predict(fit::SVM) = predict(fit, features(fit))
accuracy(fit::SVM) = accuracy(fit, features(fit), targets(fit))

function predict(fit::PrimalSVM, X::DenseMatrix)
  EmpiricalRisks.predict(predmodel(fit), details(fit).sol, X)
end

function predict(fit::PrimalSVM, X::AbstractSparseMatrix)
  n = size(X,2)
  w = details(fit).sol
  p = if typeof(predmodel(fit)) <: AffinePred
    fill(w[end] * predmodel(fit).bias, size(targets(fit)))
  else
    zeros(size(targets(fit)))
  end
  @inbounds for i = 1:n
    tstart = X.colptr[i]
    tstop  = X.colptr[i+1] - 1
    for j = tstart:tstop
      k = X.rowval[j]
      p[i] += X.nzval[j] * w[k]
    end
  end
  p
end

function predict(fit::DenseDualSVM, X::AbstractMatrix)
  n = size(X,2)
  result = zeros(n)
  @inbounds for i in 1:n
    for j in 1:nsv(fit)
      result[i] += fit.alpha[j] * fit.Ysv[j] * dot(fit.Xsv[j], view(X,:,i))
    end
    result[i] += bias(fit)
  end
  result
end

function predict(fit::SparseDualSVM, X::AbstractMatrix)
  n = size(X,2)
  result = zeros(n)
  tmp = 0.
  @inbounds for i in 1:n
    for j in 1:fit.nsv
      tstart = fit.Xsv.colptr[fit.svindex[j]]
      tstop  = fit.Xsv.colptr[fit.svindex[j]+1] - 1
      tmp = 0.
      for k = tstart:tstop
        tmp += X[fit.Xsv.rowval[k],i] * fit.Xsv.nzval[k]
      end
      tmp *= fit.Ysv[j]
      tmp *= fit.alpha[j]
      result[i] += tmp
    end
    result[i] += bias(fit)
  end
  result
end

function classify(fit::SVM, args...)
  classify(svc_or_svr(fit), args...)
end

function accuracy(fit::SVM, X, y)
  n = size(X,2)
  n == length(y) || throw(DimensionMismatch("X and y have to have the same number of observations"))
  ȳ = classify(fit, X)
  countnz(ȳ .== y) / n
end

# ==========================================================================
# convert

function convert{TSpec <: SVMSpec{ScalarProductKernel{Float64}}}(
    ::Type{PrimalSVM},
    dual::DualSVM{TSpec})
  l = length(dual.Xsv)
  d = length(dual.Xsv[1])
  # w = ∑ yᵢαᵢxᵢ
  w = zeros(d+1)
  for j = 1:d
    for i = 1:l
      @inbounds w[j] += dual.Ysv[i] * dual.alpha[i] * dual.Xsv[i][j]
    end
  end
  sol = if typeof(dual.predmodel) <: AffinePred
    w[end] = dual.details.bias
    Solution(w, dual.details.fval, dual.details.niters, dual.details.converged)
  else
    Solution(w[1:d], dual.details.fval, dual.details.niters, dual.details.converged)    
  end
  svmModel(dual.params, sol, dual.predmodel, dual.Xtrain, dual.Ytrain)
end

function convert{TSpec <: SVMSpec{ScalarProductKernel{Float64}}}(
    ::Type{DualSVM},
    primal::PrimalSVM{TSpec})
  k = size(primal.Xtrain, 1)
  n = size(primal.Xtrain, 2)
  nsv = primal.nsv
  svindex = primal.svindex
  nsv <= k || throw(DimensionMismatch("Converting to dual solution is only possible if there are less (or equal) support vectors than there are features"))
  Q = zeros(k, nsv)
  for i = 1:nsv
    for j = 1:k
      @inbounds Q[j, i] = primal.Ytrain[svindex[i]] * primal.Xtrain[j, svindex[i]]
    end
  end
  w = primal.details.sol[1:k]
  α = Q \ w
  alpha = zeros(n)
  alpha[svindex] = α
  sol = DualSolution(alpha, primal.details.sol[end], primal.details.fval, primal.details.niters, primal.details.converged)
  svmModel(primal.params, sol, primal.predmodel, primal.Xtrain, primal.Ytrain)
end

# ==========================================================================
# Plotting

function scatterplot{TSpec <: SVMSpec{ScalarProductKernel{Float64}}}(
    fit::SVM{TSpec}, args...; nargs...)
  scatterplot(svc_or_svr(fit), args...; nargs...)
end

# ==========================================================================
# Base.show

function _showprimal(io::IO, fit)
  _printconverged(io, isconverged(fit), iterations(fit))
  _printvariable(io, 18, "details()", typeof(details(fit)))
  _printvariable(io, 18, "isconverged()", isconverged(fit))
  _printvariable(io, 18, "iterations()", iterations(fit))
  println(io, "\n  ◦  support vector machine:")
  _printvariable(io, 14, "params()", params(fit))
  println(io, "\n  ◦  objective value (f):")
  _printvariable(io, 14, "fval()", fval(fit))
  println(io, "\n  ◦  fitted coefficients (w⃗):")
  _printvariable(io, 17, "coef()", coef(fit))
  _printvariable(io, 17, "intercept()", intercept(fit))
  _printvariable(io, 17, "predmodel()", predmodel(fit))
  println(io, "\n  ◦  support vectors (estimated):")
  _printvariable(io, 17, "nsv()", nsv(fit))
  _printvariable(io, 17, "svindex()", svindex(fit))
  if isclassifier(fit) && size(features(fit),1) == 2 && size(features(fit),2) < 500 && typeof(predmodel(fit)) <: Union{AffinePred, LinearPred}
    println(io, "\n  ◦  classification plot (UnicodePlots.scatterplot(..)):")
    fig = scatterplot(fit, margin = 5, width = 30, height = 10, title = "")
    print(io, fig)
  end
end

function _showdual(io::IO, fit)
  _printconverged(io, isconverged(fit), iterations(fit))
  _printvariable(io, 18, "details()", typeof(details(fit)))
  _printvariable(io, 18, "isconverged()", isconverged(fit))
  _printvariable(io, 18, "iterations()", iterations(fit))
  println(io, "\n  ◦  support vector machine:")
  _printvariable(io, 14, "params()", params(fit))
  println(io, "\n  ◦  objective value (f):")
  _printvariable(io, 14, "fval()", fval(fit))
  println(io, "\n  ◦  fitted coefficients (α):")
  _printvariable(io, 17, "coef()", coef(fit))
  intercept(fit) && _printvariable(io, 17, "bias()", bias(fit))
  _printvariable(io, 17, "intercept()", intercept(fit))
  _printvariable(io, 17, "predmodel()", predmodel(fit))
  println(io, "\n  ◦  support vectors:")
  _printvariable(io, 17, "nsv()", nsv(fit))
  _printvariable(io, 17, "svindex()", svindex(fit))
  _printvariable(io, 17, "xsv()", typeof(xsv(fit)))
  _printvariable(io, 17, "ysv()", ysv(fit))
  if isclassifier(fit) && size(features(fit),1) == 2 && size(features(fit),2) < 500 && typeof(predmodel(fit)) <: Union{AffinePred, LinearPred}
    println(io, "\n  ◦  classification plot (UnicodePlots.scatterplot(..)):")
    fig = scatterplot(fit, margin = 5, width = 30, height = 10, title = "")
    print(io, fig)
  end
end

function show(io::IO, fit::PrimalSVM)
  println(io, typeof(fit), "\n")
  _showprimal(io, fit)
end

function show(io::IO, fit::DualSVM)
  println(io, typeof(fit), "\n")
  _showdual(io, fit)
end
