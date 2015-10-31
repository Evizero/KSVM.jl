# This code extends the functionality of Regression.jl
#
# Copyright (c) 2013 Dahua Lin
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# ==========================================================================
# Regression.jl model definitions to use in Regression.solve

typealias SvmFunctional{XT<:StridedMatrix, YT<:StridedVector, PM<:PredictionModel{1,0}, L<:SvmLoss} Regression.RegRiskFun{Float64,XT,YT,Regression.SupervisedRiskModel{PM,L},SqrL2Reg{Float64}}

hingereg{T<:AbstractFloat}(X::StridedMatrix{T}, y::StridedVector{T}; bias::Real = 0.0) =
    Regression.UnivariateRegression(HingeLoss(), X, y; bias = bias)

sqrhingereg{T<:AbstractFloat}(X::StridedMatrix{T}, y::StridedVector{T}; bias::Real = 0.0) =
    Regression.UnivariateRegression(SqrHingeLoss(), X, y; bias = bias)

smoothedhingereg{T<:AbstractFloat}(X::StridedMatrix{T}, y::StridedVector{T}; h::Real = 0.5, bias::Real = 0.0) =
    Regression.UnivariateRegression(SmoothedHingeLoss(h), X, y; bias = bias)

# ==========================================================================
# Regression.jl interface for SVM solver that provide a primal solution

function solve!{XT<:StridedMatrix, YT<:StridedVector, L<:SvmLoss}(
    solver::SvmDescentSolver,
    f::Union{SvmFunctional{XT,YT,LinearPred,L},SvmFunctional{XT,YT,AffinePred,L}},
    w::Array{Float64},
    options::Regression.Options,
    callback::Function)

  X, y = f.X, f.Y
  _, l = size(X)
  reg  = f.reg
  loss = L()
  cb = callback == Regression.no_op ? nothing : callback
  spec = CSVM(ScalarProductKernel(), loss, reg)
  
  fit(spec, X, y, solver, f.rmodel.predmodel;
      dual = false,
      maxiter = options.maxiter,
      ftol = options.ftol,
      xtol = options.xtol,
      grtol = options.grtol,
      armijo = options.armijo,
      beta = options.beta,
      verbosity = options.verbosity,
      callback = cb)
end
