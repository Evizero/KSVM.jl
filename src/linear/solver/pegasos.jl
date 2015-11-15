
# ==========================================================================
# Specify the custom linear SVM solver

"""
`Pegasos <: SvmSolver`

Description
============

Implementation of the primal sub-gradient solver for SVM
as it was proposed in (Shalev-Shwartz et al., 2007).

Usage
======

    model = svm(X, y, solver = Pegasos())

References
===========

- Shalev-shwartz, Shai, et al. "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM."
Proceedings of the 24th International Conference on Machine Learning (ICML-07). 2007.
DOI=10.1145/1273496.1273598, http://dx.doi.org/10.1145/1273496.1273598,
"""
immutable Pegasos <: SvmSolver
end

# ==========================================================================
# Implementation of dual coordinate descent for linear SVMs
# This particular method is specialized for
# - dense arrays
# - univariate prediction

function fit{TKernel<:ScalarProductKernel, TLoss<:Union{MarginBasedLoss, DistanceBasedLoss}, TReg<:L2Penalty, INTERCEPT}(
        spec::CSVM{TKernel, TLoss, TReg},
        X::StridedMatrix, y⃗::StridedVector,
        ::Pegasos,
        predmodel::UvPredicton{INTERCEPT},
        problem::PrimalProblem;
        ftol::Real = 1.0e-4,
        iterations::Int = 1000,
        show_trace::Bool = false,
        callback::Union{Function,Void} = nothing,
        nargs...)
    islipschitzcont(spec.loss) || throw(ArgumentError("The loss has to be lipschitz continuous"))

    # Get the size of the design matrix
    #   k ... number of features
    #   m ... number of observations
    k, m = size(X)

    # Do some housekeeping: use shorter variable names for the options
    lambda = Float64(2 / (spec.C * m))
    has_callback = typeof(callback) <: Function
    bias = INTERCEPT ? Float64(predmodel.bias) : zero(Float64)
    loss = spec.loss
    reg = L2Penalty(lambda)

    # Print log header if show_trace is set
    show_trace && print_iter_head()

    # Initialize w to all zero.
    # Note: w has length k+1 to have space for the potential intercept.
    #       If no intercept is fit, the bias term will be ignored
    w  = zeros(Float64, INTERCEPT ? k+1 : k)

    # Buffer for the current gradient
    ▽  = zeros(Float64, INTERCEPT ? k+1 : k, 1)
    ▽⃗  = view(▽, 1:length(▽), 1)

    # Indicies into the observations of X (columns)
    # This array defines the order in which the observations of X will be iterated over
    # Note: It will improve convergence if we permutate this array each iteration.
    #       In order to achieve good performance we preallocate this array and shuffle
    #       its elements inplace each itertation
    S = collect(1:m)

    # Preallocate the vector-views into the design matrix to iterate over the observations
    # Note: We have to process each observation of X in every iteration.
    #       This means that even though arrayviews are cheap,
    #       it is still cheaper to preallocate them beforehand.
    X̄ = Array(ContiguousView{Float64,1,Array{Float64,2}}, m)
    for i in S
        @inbounds X̄[i] = view(X, :, i)
    end

    predictor = LinearPredictor(bias)
    risk = RiskModel(predictor, loss, reg)
    ŷ = zeros(1)
    ȳ = zeros(1)
    one_sqrtlambda = 1 / sqrt(lambda)
    mone_lambda = -1 / lambda

    t = 1; stopped = false
    while t < iterations && !stopped

        # Shuffle the indicies to improve convergence
        shuffle!(S)

        minus_eta = mone_lambda / t

        # loop over all observations
        @inbounds for i in S
            if t >= iterations
                break;
            end

            ȳ[1] = y⃗[i]
            ŷ[1] = value(predictor, X̄[i], w)
            grad!(▽, risk, X̄[i], w, ȳ, ŷ)
            axpy!(minus_eta, ▽⃗, w)

            nrm = one_sqrtlambda / vecnorm(w, 2)
            if nrm < 1
                broadcast!(*, w, w, nrm)
            end

            t += 1
        end
    end

    f = value(risk, X, w, y⃗)
    PrimalSolution(w, f, t, false)
end
