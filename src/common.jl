typealias SvmL1orL2Loss Union{L1HingeLoss, L2HingeLoss}
typealias LinSvmPred Union{LinearPred, AffinePred, MvLinearPred, MvAffinePred}

# ==========================================================================
# Special solver for SVM problems

abstract SvmOptimizer <: DescentSolver

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
# Functions to dtermine the interpretation of the loss

isclassifier(::Loss) = false
isclassifier(::MarginBasedLoss) = true
decision_function(::MarginBasedLoss) = sign

# ==========================================================================
# These functions make sure that if no callback is defined, then the empty
# callback function will be optimized away through inlining

# Primal problem only
function _docallback(::Any, ::Void, iter, w, w⃗, f, G) false end
function _docallback(::LinearPred, cb::Function, iter, w, w⃗, f, G) cb(iter, w⃗, f, G) == :stop end
function _docallback(::AffinePred, cb::Function, iter, w, w⃗, f, G) cb(iter, w, f, G) == :stop end

# Dual problem only
function _docallback(::Any, ::Void, iter, α, f, G) end
function _docallback(::LinSvmPred, cb::Function, iter, α, f, G) cb(iter, α, f, G) == :stop end

# Primal or dual problem
function _docallback(::Any, ::Bool, ::Void, iter, w, w⃗, α, f, G) false end
function _docallback(::LinearPred, dual::Bool, cb::Function, iter, w, w⃗, α, f, G) dual ? cb(iter, α, f, G) == :stop : cb(iter, w⃗, f, G) == :stop end
function _docallback(::AffinePred, dual::Bool, cb::Function, iter, w, w⃗, α, f, G) dual ? cb(iter, α, f, G) == :stop : cb(iter, w, f, G) == :stop end

# ==========================================================================
# Checks if the given vector y is a univariate or multivariate target

function is_univariate{T<:Real}(y::AbstractVector{T})
  for i in y
    if i < 1
      return true 
    elseif i > 1
      return false
    end
  end
  return true
end

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
