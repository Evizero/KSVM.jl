# ==========================================================================
# General low-level interface to fit Kernel-SVMs

function fit(spec::SVMSpec, X::AbstractMatrix, y⃗::AbstractVector;
             solver::Solver = DualCD(),
             loss::Loss = SqrHingeLoss(),
             nargs...)
  isclassifier(loss) || error()
  encoding = SignedClassEncoding(y⃗)
  t = labelencode(encoding, y⃗)
  model = fit(spec, X, t; solver = solver, loss = loss, nargs...)
  SVC(model, encoding)
end

# --------------------------------------------------------------------------

function fit{T<:Real}(spec::SVMSpec, X::AbstractMatrix, y⃗::AbstractVector{T};
             solver::Solver = DualCD(), # TODO: This must not be DualCD for Kernel SVMs
             nargs...)
  fit(spec, X, y⃗, solver; nargs...)
end

# ==========================================================================
# General low-level interface to fit Linear SVMs
# For them it is possible to request the primal or dual solution

function fit{T<:Real}(spec::LinearSVMSpec, X::AbstractMatrix, y⃗::AbstractVector{T};
                      solver::Solver = DualCD(),
                      dual::Bool = false,
                      nargs...)
  native_model = fit(spec, X, y⃗, solver; dual = dual, nargs...)
  # return the native_fit if it is the desired one
  dual == (typeof(native_model) <: DualSVM) && return native_model
  try # convert to desired format
    if dual == true
      info("Solver was unable to provide the dual solution. Attempting automatic conversion...")
      return convert(DualSVM, native_model)
    elseif dual == false
      info("Solver was unable to provide the primal solution. Attempting automatic conversion...")
      return convert(PrimalSVM, native_model)
    end
  catch
    warn("Automatic conversion failed! Returning solution natively provided by the solver.")
    return native_model
  end
end

# ==========================================================================
# This in-between function will compute the appropriate prediction function

function fit{T<:Real}(spec::SVMSpec, X::AbstractMatrix, y⃗::AbstractVector{T}, solver::Solver;
                      bias::Real = 1.,
                      nargs...)
  d, l = size(X)
  l == length(y⃗) || throw(DimensionMismatch("X and y⃗ have to have the same number of observations (columns)"))
  predmodel = if isclassifier(spec.loss) && !is_univariate(y⃗)
    k = maximum(y⃗)
    bias == zero(bias) ? MvLinearPred(d, k): MvAffinePred(d, k, Float64(bias))
  else
    bias == zero(bias) ? LinearPred(d): AffinePred(d, Float64(bias))
  end
  solution = fit(spec, X, y⃗, solver, predmodel; nargs...)
  svmModel(spec, solution, predmodel, X, y⃗)
end

# ==========================================================================
# Special method overload to support the "do" notation for the callback
#   model = fit(spec, X, Y) do t, alpha, v, g
#     println("Iteration $t: $v")
#   end

function fit(cb::Function, spec::SVMSpec, X::AbstractMatrix, y::AbstractVector;
             nargs...)
  fit(spec, X, y; callback = cb, nargs...)
end

# ==========================================================================
# Fallback for svm specific solver that are not implemented (for the given arguments)

function fit{PM<:Union{MvLinearPred, MvAffinePred},T<:Real}(
    spec::SVMSpec,
    X::AbstractMatrix, y⃗::AbstractVector{T},
    ::SvmDescentSolver,
    ::PM;
    nargs...)
  throw(MvNotNativelyHandled())
end

# --------------------------------------------------------------------------

function fit{T<:Real}(
    spec::SVMSpec,
    X::AbstractMatrix, y⃗::AbstractVector{T},
    solver::SvmDescentSolver,
    ::PredictionModel;
    nargs...)
  throw(ArgumentError("The types of the given arguments are not compatible with $(typeof(solver))"))
end

# ==========================================================================
# Fallback for Regression.jl solver to solve primal problem

function fit{TFun<:Union{Function,Void},T<:Real}(
    spec::SVMSpec,
    X::AbstractMatrix, y⃗::AbstractVector{T},
    solver::Solver,
    predmodel::PredictionModel;
    callback::TFun = nothing,
    maxiter::Real = 1000,
    ftol::Real = 1.0e-6,
    xtol::Real = 1.0e-6,
    grtol::Real = 1.0e-9,
    armijo::Real = .5,
    beta::Real = .5,
    verbosity::Symbol = :none,
    nargs...)
  options = Regression.Options(maxiter = maxiter, ftol = ftol, xtol = xtol,
                               grtol = grtol, armijo = armijo, beta = beta,
                               verbosity = verbosity)
  cb = TFun <: Function ? callback: Regression.no_op
  bias = (typeof(predmodel) <: Union{LinearPred, MvLinearPred}) ? zero(Float64) : Float64(predmodel.bias)
  model = Regression.UnivariateRegression(spec.loss, X, y⃗; bias = bias)
  Regression.solve(model,
                   solver = solver,
                   reg = spec.reg,
                   options = options,
                   callback = cb)
end
