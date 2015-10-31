abstract SVM{TSpec <: SVMSpec} <: RegressionModel

# ==========================================================================

svmModel(spec::SVMSpec, solution::PrimalSolution, predmodel::Predictor, X::AbstractMatrix, Y::AbstractArray) =
  primalSVMModel(spec, solution, predmodel, X, Y)

svmModel(spec::SVMSpec, solution::DualSolution, predmodel::Predictor, X::AbstractMatrix, Y::AbstractArray) =
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
type DensePrimalSVM{TSpec<:SVMSpec, TDetails<:PrimalSolution, TPred<:Predictor, XT<:DenseMatrix, YT<:AbstractVector} <: PrimalSVM{TSpec}
  params::TSpec
  details::TDetails
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
type SparsePrimalSVM{TSpec<:SVMSpec, TDetails<:PrimalSolution, TPred<:Predictor, XT<:AbstractSparseMatrix, YT<:AbstractVector} <: PrimalSVM{TSpec}
  params::TSpec
  details::TDetails
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
type DenseDualSVM{TSpec<:SVMSpec, TPred<:Predictor, XT<:DenseMatrix, YT<:AbstractVector} <: DualSVM{TSpec}
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
type SparseDualSVM{TSpec<:SVMSpec, TPred<:Predictor, XT<:AbstractSparseMatrix, YT<:AbstractVector} <: DualSVM{TSpec}
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

function primalSVMModel{TSpec<:SVMSpec, TPred<:Predictor, TReal<:Real}(
    params::TSpec,
    s::PrimalSolution,
    predmodel::TPred,
    Xtrain::StridedMatrix,
    Ytrain::AbstractVector{TReal})
  n = size(Xtrain, 2)
  p = value(predmodel, Xtrain, minimizer(s))
  l = zeros(n)
  @inbounds for i = 1:n
    lc = value(params.loss, Float64(Ytrain[i]), p[i] - sign(p[i]) * 0.0005)
    lr = value(params.loss, Float64(Ytrain[i]), p[i] + sign(p[i]) * 0.0005)
    l[i] = lc != 0 || lr != 0
  end
  svindex = find(l)
  nsv = length(svindex)
  DensePrimalSVM{TSpec, typeof(s), TPred, typeof(Xtrain), typeof(Ytrain)}(params, s, predmodel, nsv, svindex, Xtrain, Ytrain)
end

function primalSVMModel{TSpec<:SVMSpec, TPred<:Predictor, TReal<:Real}(
    params::TSpec,
    s::PrimalSolution,
    predmodel::TPred,
    Xtrain::AbstractSparseMatrix,
    Ytrain::AbstractVector{TReal})
  n = size(Xtrain, 2)
  w = minimizer(s)
  p = if typeof(predmodel) <: LinearPredictor{true}
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
    lc = value(params.loss, Float64(Ytrain[i]), p[i] - sign(p[i]) * 0.0005,)
    lr = value(params.loss, Float64(Ytrain[i]), p[i] + sign(p[i]) * 0.0005)
    l[i] = lc != 0 || lr != 0
  end
  svindex = find(l)
  nsv = length(svindex)
  SparsePrimalSVM{TSpec, typeof(s), TPred, typeof(Xtrain), typeof(Ytrain)}(params, s, predmodel, nsv, svindex, Xtrain, Ytrain)
end

function dualSVMModel{TSpec<:SVMSpec, TPred<:Predictor, TReal<:Real}(
    params::TSpec,
    s::DualSolution,
    p::TPred,
    Xtrain::StridedMatrix,
    Ytrain::AbstractVector{TReal})
  svindex = find(minimizer(s))
  alpha = minimizer(s)[svindex]
  nsv = length(alpha)
  Xsv = Array(ContiguousView{Float64,1,Array{Float64,2}}, nsv)
  @inbounds for i in 1:nsv
    Xsv[i] = view(Xtrain, :, svindex[i])
  end
  Ysv = Ytrain[svindex]
  DenseDualSVM{TSpec, TPred, typeof(Xtrain), typeof(Ytrain)}(params, s, p, nsv, alpha, svindex, Xsv, Ysv, Xtrain, Ytrain)
end

function dualSVMModel{TSpec<:SVMSpec, TPred<:Predictor, TReal<:Real}(
    params::TSpec,
    s::DualSolution,
    p::TPred,
    Xtrain::AbstractSparseMatrix,
    Ytrain::AbstractVector{TReal})
  svindex = find(minimizer(s))
  alpha = minimizer(s)[svindex]
  nsv = length(alpha)
  Xsv = Xtrain
  Ysv = Ytrain[svindex]
  SparseDualSVM{TSpec, TPred, typeof(Xtrain), typeof(Ytrain)}(params, s, p, nsv, alpha, svindex, Xsv, Ysv, Xtrain, Ytrain)
end

# ==========================================================================

@inline labels{TSpec<:SVCSpec}(svm::SVM{TSpec}) = [-1., 1]
@inline nobs(fit::SVM) = length(fit.Ytrain)
@inline features(fit::SVM) = fit.Xtrain
@inline targets(fit::SVM) = fit.Ytrain
@inline model_response(fit::SVM) = fit.Ytrain

@inline details(fit::SVM) = fit.details
@inline isconverged(fit::SVM) = isconverged(details(fit))
@inline iterations(fit::SVM) = iterations(details(fit))
@inline params(fit::SVM) = fit.params
@inline minimum(fit::SVM) = minimum(details(fit))
@inline minimizer(fit::SVM) = minimizer(details(fit))
@inline nsv(fit::SVM) = fit.nsv
@inline svindex(fit::SVM) = fit.svindex
@inline intercept(fit::SVM) = typeof(predmodel(fit)) <: LinearPredictor{true}
@inline predmodel(fit::SVM) = fit.predmodel
@inline coef(fit::PrimalSVM) = coef(details(fit))
@inline coef(fit::DualSVM) = fit.alpha
@inline bias(fit::DualSVM) = bias(details(fit))
@inline xsv(fit::DualSVM) = fit.Xsv
@inline ysv(fit::DualSVM) = fit.Ysv

@inline predict(fit::SVM) = predict(fit, features(fit))
@inline classify{TSpec<:SVCSpec}(svm::SVM{TSpec}) = classify(svm, features(svm))
@inline accuracy{TSpec<:SVCSpec}(svm::SVM{TSpec}) = accuracy(svm, features(svm), targets(svm))

function classify{TSpec<:SVCSpec}(svm::SVM{TSpec}, X)
  ŷ = predict(svm, X)
  t = ndims(ŷ) == 1 ? sign(ŷ) : vec(mapslices(indmax, ŷ, 1))
  t
end

function accuracy{TSpec<:SVCSpec}(svm::SVM{TSpec}, X, y)
  n = size(X,2)
  n == length(y) || throw(DimensionMismatch("X and y have to have the same number of observations"))
  ȳ = classify(svm, X)
  countnz(ȳ .== y) / n
end

function predict(fit::PrimalSVM, X::DenseMatrix)
  p = value(predmodel(fit), X, minimizer(details(fit)))
  size(p, 1) == 1 ? vec(p) : p
end

function predict(fit::PrimalSVM, X::AbstractSparseMatrix)
  n = size(X,2)
  w = minimizer(details(fit))
  p = if typeof(predmodel(fit)) <: LinearPredictor{true}
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
      result[i] += coef(fit)[j] * ysv(fit)[j] * dot(xsv(fit)[j], view(X,:,i))
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
    for j in 1:nsv(fit)
      tstart = xsv(fit).colptr[svindex(fit)[j]]
      tstop  = xsv(fit).colptr[svindex(fit)[j]+1] - 1
      tmp = 0.
      for k = tstart:tstop
        tmp += X[xsv(fit).rowval[k],i] * xsv(fit).nzval[k]
      end
      tmp *= ysv(fit)[j]
      tmp *= coef(fit)[j]
      result[i] += tmp
    end
    result[i] += bias(fit)
  end
  result
end

# ==========================================================================
# convert

function convert{TSpec <: SVMSpec{ScalarProductKernel{Float64}}}(
    ::Type{PrimalSVM},
    dual::DualSVM{TSpec})
  l = length(xsv(dual))
  d = length(xsv(dual)[1])
  # w = ∑ yᵢαᵢxᵢ
  w = zeros(d+1)
  for j = 1:d
    for i = 1:l
      @inbounds w[j] += ysv(dual)[i] * coef(dual)[i] * xsv(dual)[i][j]
    end
  end
  sol = if typeof(predmodel(dual)) <: LinearPredictor{true}
    w[end] = bias(dual)
    PrimalSolution(w, minimum(dual), iterations(dual), isconverged(dual))
  else
    PrimalSolution(w[1:d], minimum(dual), iterations(dual), isconverged(dual))    
  end
  svmModel(params(dual), sol, predmodel(dual), features(dual), targets(dual))
end

function convert{TSpec <: SVMSpec{ScalarProductKernel{Float64}}}(
    ::Type{DualSVM},
    primal::PrimalSVM{TSpec})
  k = size(features(primal), 1)
  n = size(features(primal), 2)
  nsv = nsv(primal)
  svindex = svindex(primal)
  nsv <= k || throw(DimensionMismatch("Converting to dual solution is only possible if there are less (or equal) support vectors than there are features"))
  Q = zeros(k, nsv)
  for i = 1:nsv
    for j = 1:k
      @inbounds Q[j, i] = targets(primal)[svindex[i]] * features(primal)[j, svindex[i]]
    end
  end
  # DOESN'T WORK FOR NO BIAS
  w = minimizer(details(primal))[1:k]
  α = Q \ w
  alpha = zeros(n)
  alpha[svindex] = α
  sol = DualSolution(alpha, minimizer(details(primal))[end], minimum(primal), iterations(primal), isconverged(primal))
  svmModel(params(primal), sol, predmodel(primal), features(primal), targets(primal))
end

# ==========================================================================
# Plotting

function scatterplot{TSpec<:SVCSpec}(
    fit::PrimalSVM{TSpec};
    title::AbstractString = "Primal SVM Classification Plot",
    xlim = [0.,0.],
    ylim = [0.,0.],
    lbl = map(string, labels(fit)),
    nargs...)
  size(features(fit),1) == 2 || throw(DimensionMismatch("Can only plot the SVM classification for a two-dimensional featurespace (i.e. size(X,1) == 2)"))
  intercept_fit = typeof(predmodel(fit)) <: LinearPredictor{true}
  offset = intercept_fit ? -(minimizer(details(fit))[3] * predmodel(fit).bias) / minimizer(details(fit))[2] : 0.
  slope = -minimizer(details(fit))[1] / minimizer(details(fit))[2]
  x1 = vec(view(features(fit), 1, :))
  x2 = vec(view(features(fit), 2, :))
  x1sv = x1[svindex(fit)]
  x2sv = x2[svindex(fit)]
  xmin = minimum(x1); xmax = maximum(x1)
  ymin = minimum(x2); ymax = maximum(x2)
  xlim = xlim == [0.,0.] ? [xmin, xmax] : xlim
  ylim = ylim == [0.,0.] ? [ymin, ymax] : ylim
  notalphaindex = setdiff(1:size(features(fit),2), svindex(fit))
  x1 = x1[notalphaindex]
  x2 = x2[notalphaindex]
  y = targets(fit)[notalphaindex]
  fig = scatterplot(x1[y.<0], x2[y.<0]; title = title, xlim = xlim, ylim = ylim, name = lbl[1], nargs...)
  scatterplot!(fig, x1[y.>0], x2[y.>0], name = lbl[2])
  scatterplot!(fig, x1sv, x2sv, color = :yellow, name = "support vectors")
  lineplot!(fig, offset, slope, color = :white)
  xlabel!(fig, "X₁")
  ylabel!(fig, "X₂")
  fig
end

function scatterplot{TSpec<:SVCSpec}(
    fit::DualSVM{TSpec};
    title::AbstractString = "Dual SVM Classification Plot",
    xlim = [0.,0.],
    ylim = [0.,0.],
    lbl = map(string, labels(fit)),
    nargs...)
  size(features(fit),1) == 2 || throw(DimensionMismatch("Can only plot the SVM classification for a two-dimensional featurespace (i.e. size(X,1) == 2)"))
  x1 = vec(view(features(fit), 1, :))
  x2 = vec(view(features(fit), 2, :))
  x1sv = x1[svindex(fit)]
  x2sv = x2[svindex(fit)]
  xmin = minimum(x1); xmax = maximum(x1)
  ymin = minimum(x2); ymax = maximum(x2)
  xlim = xlim == [0.,0.] ? [xmin, xmax] : xlim
  ylim = ylim == [0.,0.] ? [ymin, ymax] : ylim
  notalphaindex = setdiff(1:size(features(fit),2), svindex(fit))
  x1 = x1[notalphaindex]
  x2 = x2[notalphaindex]
  y = targets(fit)[notalphaindex]
  fig = scatterplot(x1[y.<0], x2[y.<0]; title = title, xlim = xlim, ylim = ylim, name = lbl[1], nargs...)
  scatterplot!(fig, x1[y.>0], x2[y.>0], name = lbl[2])
  scatterplot!(fig, x1sv, x2sv, color = :yellow, name = "support vectors")
  xlabel!(fig, "X₁")
  ylabel!(fig, "X₂")
  fig
end

# ==========================================================================
# Base.show

function _showprimal(io::IO, fit)
  _printconverged(io, isconverged(fit), iterations(fit))
  _printvariable(io, 19, "details()", typeof(details(fit)))
  _printvariable(io, 19, "isconverged()", isconverged(fit))
  _printvariable(io, 19, "iterations()", iterations(fit))
  println(io, "\n  ◦  support vector machine:")
  _printvariable(io, 14, "params()", params(fit))
  println(io, "\n  ◦  objective value (f):")
  _printvariable(io, 17, "minimum()", minimum(fit))
  println(io, "\n  ◦  fitted coefficients (w⃗):")
  _printvariable(io, 17, "coef()", coef(fit))
  _printvariable(io, 17, "intercept()", intercept(fit))
  _printvariable(io, 17, "predmodel()", predmodel(fit))
  println(io, "\n  ◦  support vectors (estimated):")
  _printvariable(io, 17, "nsv()", nsv(fit))
  _printvariable(io, 17, "svindex()", svindex(fit))
  if size(features(fit),1) == 2 && size(features(fit),2) < 500
    println(io, "\n  ◦  classification plot (UnicodePlots.scatterplot(..)):")
    fig = scatterplot(fit, margin = 5, width = 30, height = 10, title = "")
    print(io, fig)
  end
end

function _showdual(io::IO, fit)
  _printconverged(io, isconverged(fit), iterations(fit))
  _printvariable(io, 19, "details()", typeof(details(fit)))
  _printvariable(io, 19, "isconverged()", isconverged(fit))
  _printvariable(io, 19, "iterations()", iterations(fit))
  println(io, "\n  ◦  support vector machine:")
  _printvariable(io, 14, "params()", params(fit))
  println(io, "\n  ◦  objective value (f):")
  _printvariable(io, 17, "minimum()", minimum(fit))
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
  if size(features(fit),1) == 2 && size(features(fit),2) < 500
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
