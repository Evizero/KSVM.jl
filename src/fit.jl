# ==========================================================================
# General low-level interface to fit Kernel-SVMs

@inline function fit(spec::SVMSpec, X::AbstractMatrix, y⃗::AbstractVector;
             solver::SvmSolver = DualCD(),
             loss::MarginBasedLoss = L2HingeLoss(),
             nargs...)
  encoding = SignedClassEncoding(y⃗)
  t = labelencode(encoding, y⃗)
  model = fit(spec, X, t; solver = solver, loss = loss, nargs...)
  SVC(model, encoding)
end

# --------------------------------------------------------------------------

@inline function fit{T<:Real}(spec::SVMSpec, X::AbstractMatrix, y⃗::AbstractVector{T};
             solver::SvmSolver = DualCD(), # TODO: This must not be DualCD for Kernel SVMs
             nargs...)
  fit(spec, X, y⃗, solver; nargs...)
end

# ==========================================================================
# General low-level interface to fit Linear SVMs
# For them it is possible to request the primal or dual solution

function fit{T<:Real}(spec::LinearSVMSpec, X::AbstractMatrix, y⃗::AbstractVector{T};
                      solver::SvmSolver = DualCD(),
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

function fit{T<:Real}(spec::SVMSpec, X::AbstractMatrix, y⃗::AbstractVector{T}, solver::SvmSolver;
                      bias::Real = 1.,
                      nargs...)
  d, l = size(X)
  l == length(y⃗) || throw(DimensionMismatch("X and y⃗ have to have the same number of observations (columns)"))
  predtype = prediction_type(spec.loss, y⃗, bias)
  solution = fit(spec, X, y⃗, solver, predtype; nargs...)
  svmModel(spec, solution, LinearPredictor(bias), X, y⃗)
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

function fit{PM<:MvPredicton,T<:Real}(
    spec::SVMSpec,
    X::AbstractMatrix, y⃗::AbstractVector{T},
    ::SvmSolver,
    ::PM;
    nargs...)
  throw(MvNotNativelyHandled())
end

# --------------------------------------------------------------------------

function fit{T<:Real}(
    spec::SVMSpec,
    X::AbstractMatrix, y⃗::AbstractVector{T},
    solver::SvmSolver,
    ::PredictionType;
    nargs...)
  throw(ArgumentError("The types of the given arguments are not compatible with $(typeof(solver))"))
end
