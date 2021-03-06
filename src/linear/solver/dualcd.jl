# Copyright (c) 2015: Christof Stocker
#
# ==========================================================================
# Influenced by closely looking at the implementation of SVM.jl
#
# Copyright (c) 2013: John Myles White and other contributors.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ==========================================================================
# Influenced by looking at the implementation of LIBLINEAR
#
# Copyright (c) 2007-2015 The LIBLINEAR Project.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither name of copyright holders nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ==========================================================================

# ==========================================================================
# Specify the custom linear SVM solver

"""
`DualCD <: SvmSolver`

Description
============

Implementation of the dual coordinate descent algorithm as it
was proposed in (Hsieh et al., 2008).

Usage
======

    model = svm(X, y, solver = DualCD(shrinking=false))

References
===========

- Hsieh, Cho-Jui, et al. "A dual coordinate descent method for large-scale linear SVM."
Proceedings of the 25th international conference on Machine learning (ICML-08).
ACM, 2008. DOI=10.1145/1390156.1390208, http://doi.acm.org/10.1145/1390156.1390208
"""
immutable DualCD{T} <: SvmSolver
  shrinking::Bool
  DualCD() = new(T)
end

DualCD(;shrinking::Bool = false) = DualCD{shrinking}()
DualCDWithShrinking() = DualCD{true}()

show(io::IO, ::DualCD{false}) = print(io, "DualCD(shrinking = false)")
show(io::IO, ::DualCD{true}) = print(io, "DualCD(shrinking = true)")

# ==========================================================================
# Implementation of dual coordinate descent for linear L1-, and L2-SVMs
# This particular method is specialized for
# - dense arrays
# - univariate prediction

function fit{TKernel<:ScalarProductKernel, TLoss<:L1orL2HingeLoss, TReg<:L2Penalty, INTERCEPT, T}(
    spec::CSVM{TKernel, TLoss, TReg},
    X::StridedMatrix{T}, y⃗::StridedVector,
    ::DualCD{false},
    predmodel::UvPredicton{INTERCEPT},
    problem::SvmProblem;
    iterations::Int = 1000,
    ftol::Real = 1.0e-4,
    show_trace::Bool = false,
    callback::Union{Function,Void} = nothing,
    nargs...)

  # Do some housekeeping: use shorter variable names for the options
  C = spec.C::Float64
  has_callback = typeof(callback) <: Function
  bias = INTERCEPT ? Float64(predmodel.bias) : zero(Float64)
  dual = typeof(problem) <: DualProblem

  # Print log header if show_trace is set
  show_trace && print_iter_head()

  # Get the size of the design matrix
  #   k ... number of features
  #   l ... number of observations
  k, l = size(X)

  # Initialize α and w to all zero.
  # Note: w has length k+1 to have space for the potential intercept.
  #       If no intercept is fit, the bias term will be ignored
  α  = zeros(l)
  w  = zeros(k + 1)

  # We generate two view to the a.ctual coefficients in w⃗ and the
  # bias in w₀.
  w⃗  = view(w, 1:k)
  w₀ = view(w, k+1)

  # We store the whole gradient if the user is interested in it
  ▽  = has_callback ? zeros(Float64, l) : zeros(Float64, 0)

  # U and D depend on the type of lossfunction used (which is the type of `loss`)
  # Note: Although D is conceptually a matrix, the diagonal Dᵢᵢ is constant and
  #       we can thus save memory by using just a scalar to store it's value
  U, Dᵢᵢ = if typeof(spec.loss) <: HingeLoss
    # L1-SVM: U = C,  Dᵢᵢ = 0
    (C, zero(Float64))
  else # typeof(loss) <: L2HingeLoss
    # L2-SVM: U = ∞,  Dᵢᵢ = (2⋅C)⁻¹
    (typemax(Float64), Float64(.5 / C))
  end

  # The concrete implementation of Q̄  depends on the bias being present or not.
  # This is because the bias is fit seperately instead of being part of the
  # design matrix X.
  # Note: We skip defining Q and go straight for Q̄  = Q + D,
  #       because Q itself is not really required
  Q̄ = if INTERCEPT
    vec(sum(X .^ 2, 1)  + bias ^ 2) + Dᵢᵢ
  else
    vec(sum(X .^ 2, 1)) + Dᵢᵢ
  end

  # Indicies into the observations of X (columns)
  # This array defines the order in which the observations of X will be iterated over
  # Note: It will improve convergence if we permutate this array each iteration.
  #       In order to achieve good performance we preallocate this array and shuffle
  #       its elements inplace each itertation
  indicies = collect(1:l)

  # Preallocate the vector-views into the design matrix to iterate over the observations
  # Note: We have to process each observation of X in every iteration.
  #       This means that even though arrayviews are cheap,
  #       it is still cheaper to preallocate them beforehand.
  X̄ = Array(ContiguousView{T,1,Array{T,2}}, l)
  for i in indicies
    @inbounds X̄[i] = view(X, :, i)
  end

  iteration = 0; G = zero(Float64); PG = zero(Float64)
  PGmin = zero(Float64); PGmax = zero(Float64)
  converged = false; stopped = false
  while !converged && iteration < iterations && !stopped

    # The bounds and difference of the projected gradient of the current iteration
    PGmin = typemax(Float64)
    PGmax = typemin(Float64)

    # Shuffle the indicies to improve convergence
    shuffle!(indicies)

    # loop over all observations
    @inbounds for i in indicies

      # Buffer αᵢ since we need it a few times
      ᾱᵢ = α[i]

      # The concrete computations of G depends on the intercept being present
      # The compiler will remove the if block
      if !INTERCEPT
        G = y⃗[i] .* dot(w⃗, X̄[i]) - 1 + Dᵢᵢ .* ᾱᵢ
      else
        G = y⃗[i] .* (dot(w⃗, X̄[i]) + bias * w₀[1]) - 1. + Dᵢᵢ .* ᾱᵢ
      end

      if has_callback
        # Keep track of the gradient vector if a callback is specified
        ▽[i] = G
      end

      PG = G
      if ᾱᵢ == zero(Float64)
        PG = min(G, zero(Float64))
      elseif ᾱᵢ == U
        PG = max(G, zero(Float64))
      end

      # Keep track of the current gradient bounds
      PGmin = min(PGmin, PG)
      PGmax = max(PGmax, PG)

      # if |PG| != 0 then update αᵢ and w accordingly
      if abs(PG) > 1e-13
        # This will update aᵢ but leave āᵢ alone (we need both to update w)
        α[i] = min(max(ᾱᵢ - G / Q̄[i], zero(Float64)), U)
        # The concrete update of w depends on the intercept being present.
        # The compiler will remove the if block
        #   w⃗ = w⃗ + (αᵢ − ᾱᵢ)yᵢ ⋅ x⃗ᵢ
        #   w⃗ = w⃗ + (αᵢ − ᾱᵢ)yᵢ ⋅ x⃗ᵢ; w₀ += (αᵢ − ᾱᵢ)yᵢ * bias
        if !INTERCEPT
          Base.BLAS.axpy!((α[i] - ᾱᵢ) * y⃗[i], X̄[i], w⃗)
        else
          tmp = (α[i] - ᾱᵢ) * y⃗[i]
          Base.BLAS.axpy!(tmp, X̄[i], w⃗)
          w₀[1] += tmp * bias
        end
      end
    end

    # If the distance of the gradient bounds is close to zero we assume convergence
    converged = (PGmax - PGmin) < ftol
    iteration += 1

    # In case the user requested to print the learning process and/or provided
    # callback function this code will provide just that.
    # Note: If no callback function is provided then the compiler will be able
    #       to optimize the empty callback call away.
    if typeof(callback) <: Function || show_trace
      f = dot(w, w)
      @simd for i in 1:l
        @inbounds f += α[i] * (Dᵢᵢ * α[i] - 2.)
      end
      f = f / 2.
      show_trace && print_iter(iteration, f, PGmax - PGmin)
      stopped = _docallback(predmodel, dual, callback, iteration, w, w⃗, α, f, ▽)
    end
  end

  # Compute the final objective value to store in the result
  f = dot(w, w)
  for i in 1:l
    @inbounds f += α[i] * (Dᵢᵢ * α[i] - 2.)
  end
  f = f / 2.

  #show_trace && print_final(iteration, f, converged)
  if dual
    DualSolution(α, w₀[1], f, iteration, converged)
  else
    # The size of the fitted w depends on if an intercept was fit or not
    PrimalSolution(INTERCEPT ? w : w[1:k], f, iteration, converged)
  end
end


# ==========================================================================
# Implementation of dual coordinate descent for linear L1-, and L2-SVMs
# This particular method is specialized for
# - sparse arrays
# - univariate prediction

function fit{TKernel<:ScalarProductKernel, TLoss<:L1orL2HingeLoss, TReg<:L2Penalty, INTERCEPT}(
    spec::CSVM{TKernel, TLoss, TReg},
    X::SparseMatrixCSC, y⃗::StridedVector,
    ::DualCD{false},
    predmodel::UvPredicton{INTERCEPT},
    problem::SvmProblem;
    iterations::Int = 1000,
    ftol::Real = 1.0e-4,
    show_trace::Bool = false,
    callback::Union{Function,Void} = nothing,
    nargs...)

  # Do some housekeeping: use shorter variable names for the options
  C = spec.C::Float64
  has_callback = typeof(callback) <: Function
  bias = INTERCEPT ? Float64(predmodel.bias) : zero(Float64)
  dual = typeof(problem) <: DualProblem

  # Print log header if show_trace is set
  show_trace && print_iter_head()

  # Get the size of the design matrix
  #   k ... number of features
  #   l ... number of observations
  k, l = size(X)

  # Initialize α and w to all zero.
  # Note: w has length k+1 to have space for the potential intercept.
  #       If no intercept is fit, the bias term will be ignored
  α  = zeros(l)
  w  = zeros(k + 1)

  # We generate two view to the actual coefficients in w⃗ and the
  # bias in w₀.
  w⃗  = view(w, 1:k)
  w₀ = view(w, k+1)

  # We store the gradient if the user is interested in it
  ▽  = has_callback ? zeros(l) : zeros(0)

  # U and D depend on the type of lossfunction used (which is the type of `loss`)
  # Note: Although D is conceptually a matrix, the diagonal Dᵢᵢ is constant and
  #       we can thus save memory by using just a scalar to store it's value
  U, Dᵢᵢ = if typeof(spec.loss) <: HingeLoss
    # L1-SVM: U = C,  Dᵢᵢ = 0
    (C, zero(Float64))
  else # typeof(loss) <: L2HingeLoss
    # L2-SVM: U = ∞,  Dᵢᵢ = (2⋅C)⁻¹
    (typemax(Float64), convert(Float64, .5 / C))
  end

  # Indicies into the observations of X (columns)
  # This array defines the order in which the observations of X will be iterated over
  # Note: It will improve convergence if we permutate this array each iteration.
  #       In order to achieve good performance we preallocate this array and shuffle
  #       its elements inplace each iertation
  indicies = collect(1:l)

  # We compute Q̄  as efficiently as possible by using the inner structure
  # of the SparseMatrixCSC directly
  # Note: We skip defining Q and go straight for Q̄  = Q + D,
  #       because Q itself is not really required
  Q̄ = zeros(l)
  @inbounds for i in indicies
    tstart = X.colptr[i]
    tstop  = X.colptr[i+1] - 1
    for j = tstart:tstop
      Q̄[i] += (X.nzval[j])^2
    end
    if INTERCEPT
      Q̄[i] += bias ^ 2
    end
    Q̄[i] += Dᵢᵢ
  end

  iteration = 0; G = zero(Float64); PG = zero(Float64)
  PGmin = zero(Float64); PGmax = zero(Float64)
  converged = false; stopped = false
  while !converged && iteration < iterations && !stopped

    # The bounds and difference of the projected gradient of the current iteration
    PGmin = typemax(Float64)
    PGmax = typemin(Float64)

    # Shuffle the indicies to improve convergence
    shuffle!(indicies)

    # loop over all observations
    @inbounds for i in indicies

      # These two variables define where the column of interest and how many
      # elements that column contains. This way we can efficiently iterate
      # over exactly those values that correspond to the non-zero features
      # of the current column (i.e. observation)
      tstart = X.colptr[i]
      tstop  = X.colptr[i+1] - 1

      # Buffer αᵢ since we need it a few times
      ᾱᵢ = α[i]

      # Compute G efficiently by working directly with the inner structure
      # of the SparseMatrixCSC
      #   G = yᵢ * (w⃗ ⋅ Xᵢ + bias * w₀) - 1 + Dᵢᵢ * αᵢ
      G = zero(Float64)
      # This does: "w⃗ ⋅ Xᵢ"
      for j = tstart:tstop
        G += w⃗[X.rowval[j]] * X.nzval[j]
      end
      # This does: "(... + bias * w₀)"
      if INTERCEPT
        G += bias * w₀[1]
      end
      # This does: "yᵢ * (...)"
      G *= y⃗[i]
      # This does: "... - 1 + Dᵢᵢ * αᵢ"
      G += Dᵢᵢ .* ᾱᵢ - 1.

      # Keep track of the gradient vector if a callback is specified
      if has_callback
        ▽[i] = G
      end

      PG = G
      if ᾱᵢ == zero(Float64)
        PG = min(G, zero(Float64))
      elseif ᾱᵢ == U
        PG = max(G, zero(Float64))
      end

      # Keep track of the current gradient bounds
      PGmin = min(PGmin, PG)
      PGmax = max(PGmax, PG)

      # if PḠ != 0 then update αᵢ and w accordingly
      if abs(PG) > 1e-13
        # This will update aᵢ but leave āᵢ alone (we need both to update w)
        α[i] = min(max(ᾱᵢ - G / Q̄[i], zero(Float64)), U)
        # Again, we use the structure of SparseMatrixCSC to compute w efficiently
        #   w⃗ = w⃗ + (αᵢ − ᾱᵢ)yᵢ ⋅ x⃗ᵢ
        tmp = (α[i] - ᾱᵢ) * y⃗[i]
        for j = tstart:tstop
          w⃗[X.rowval[j]] += tmp * X.nzval[j]
        end
        #   w₀ += (αᵢ − ᾱᵢ)yᵢ * bias
        if INTERCEPT
          w₀[1] += tmp * bias
        end
      end
    end

    # If the distance of the gradient bounds is close to zero we assume convergence
    converged = (PGmax - PGmin) < ftol
    iteration += 1

    # In case the user requested to print the learning process and/or provided
    # callback function this code will provide just that.
    # Note: If no callback function is provided then the compiler will be able
    #       to optimize the empty callback call away.
    if typeof(callback) <: Function || show_trace
      f = dot(w, w)
      for i in 1:l
        @inbounds f += α[i] * (Dᵢᵢ * α[i] - 2.)
      end
      f = f / 2.
      show_trace && print_iter(iteration, f, PGmax - PGmin)
      stopped = _docallback(predmodel, dual, callback, iteration, w, w⃗, α, f, ▽)
    end
  end

  # Compute the final objective value to store in the result
  f = dot(w, w)
  for i in 1:l
    @inbounds f += α[i] * (Dᵢᵢ * α[i] - 2.)
  end
  f = f / 2.

  #show_trace && print_final(iteration, f, converged)
  if dual
    DualSolution(α, w₀[1], f, iteration, converged)
  else
    # The size of the fitted w depends on if an intercept was fit or not
    PrimalSolution(INTERCEPT ? w : w[1:k], f, iteration, converged)
  end
end
