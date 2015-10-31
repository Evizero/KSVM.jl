typealias L1orL2HingeLoss Union{L1HingeLoss, L2HingeLoss}

# ==========================================================================
# Predicton type to dispatch on univariate/multivariate prediction

abstract PredictionType{INTERCEPT}

immutable UvPredicton{INTERCEPT} <: PredictionType{INTERCEPT}
  bias::Float64
  function UvPredicton(bias::Real)
    ((typeof(INTERCEPT) <: Bool) && (bias != 0.) == INTERCEPT) || throw(MethodError())
    new(Float64(bias))
  end
end
UvPredicton(bias::Real) = UvPredicton{bias!=0.}(bias)

immutable MvPredicton{INTERCEPT} <: PredictionType{INTERCEPT}
  bias::Float64
  function MvPredicton(bias::Real)
    ((typeof(INTERCEPT) <: Bool) && (bias != 0.) == INTERCEPT) || throw(MethodError())
    new(Float64(bias))
  end
end
MvPredicton(bias::Real) = MvPredicton{bias!=0.}(bias)

function prediction_type{T<:Real}(loss::MarginBasedLoss, y::AbstractVector{T}, bias::Real)
  for i in y
    if i < 1
      return UvPredicton(bias)
    elseif i > 1
      return MvPredicton(bias)
    end
  end
  return UvPredicton(bias)
end

function prediction_type{T<:Real}(loss::DistanceBasedLoss, y::AbstractVector{T}, bias::Float64)
  UvPredicton(bias)
end

function prediction_type{T<:Real}(loss::DistanceBasedLoss, y::AbstractMatrix{T}, bias::Float64)
  MvPredicton(bias)
end

# ==========================================================================
# Special solver for SVM problems

abstract SvmSolver <: AbstractSolver

# ==========================================================================
# Typed problem definition for returning the primal or dual solution

abstract SvmProblem
immutable PrimalProblem <: SvmProblem end
immutable DualProblem <: SvmProblem end

# ==========================================================================
# Internal exception that is thrown if an algorithm doesn't handle the
# multivariate prediction case itself. This will result in a generic handleing
# such as a one-vs-all in the case of classification

type MvNotNativelyHandled <: Exception end

# ==========================================================================
# These functions make sure that if no callback is defined, then the empty
# callback function will be optimized away through inlining

# Primal problem only
function _docallback(::Any, ::Void, iter, w, w⃗, f, G) false end
function _docallback(::PredictionType{false}, cb::Function, iter, w, w⃗, f, G) cb(iter, w⃗, f, G) == :Exit end
function _docallback(::PredictionType{true}, cb::Function, iter, w, w⃗, f, G) cb(iter, w, f, G) == :Exit end

# Dual problem only
function _docallback(::Any, ::Void, iter, α, f, G) end
function _docallback(::PredictionType, cb::Function, iter, α, f, G) cb(iter, α, f, G) == :Exit end

# Primal or dual problem
function _docallback(::Any, ::Bool, ::Void, iter, w, w⃗, α, f, G) false end
function _docallback(::PredictionType{false}, dual::Bool, cb::Function, iter, w, w⃗, α, f, G) dual ? cb(iter, α, f, G) == :Exit : cb(iter, w⃗, f, G) == :Exit end
function _docallback(::PredictionType{true}, dual::Bool, cb::Function, iter, w, w⃗, α, f, G) dual ? cb(iter, α, f, G) == :Exit : cb(iter, w, f, G) == :Exit end

# ==========================================================================
# Custom print functions for the training output

function print_iter_head()
  @printf("%5s   %12s   %12s\n",
          "Iter", "f.value", "g.value")
  println("======================================================")
end

function print_iter(t::Int, J::Real, G::Real)
  @printf("%5d   %12.4e   %12.4e\n", t, J, G)
end
