# Copyright (c) 2015: Christof Stocker

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
        iterations::Int = 10000,
        show_trace::Bool = false,
        callback::Union{Function,Void} = nothing,
        nargs...)
    islipschitzcont(spec.loss) || throw(ArgumentError("The loss has to be lipschitz continuous"))

    # Get the size of the design matrix
    #   k ... number of features
    #   m ... number of observations
    k, m = size(X)

    # Do some housekeeping: use shorter variable names for the options
    lambda = Float64(1 / (spec.C * m))
    has_callback = typeof(callback) <: Function
    bias = INTERCEPT ? Float64(predmodel.bias) : zero(Float64)
    loss = spec.loss
    reg = L2Penalty(lambda)
    predictor = LinearPredictor(bias)
    risk = EmpiricalRisk(predictor, loss, reg)

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
    S = rand(1:m, iterations)

    # Preallocate the vector-views into the design matrix to iterate over the observations
    # Note: We have to process each observation of X in every iteration.
    #       This means that even though arrayviews are cheap,
    #       it is still cheaper to preallocate them beforehand.
    X̄ = Array(ContiguousView{Float64,1,Array{Float64,2}}, m)
    for i in 1:m
        @inbounds X̄[i] = view(X, :, i)
    end

    one_sqrtlambda = 1 / sqrt(lambda)
    minus_one_lambda = -1 / lambda

    ybuffer = zeros(1, m)
    t = 0; stopped = false
    @inbounds for (t, i) in enumerate(S)
        if stopped; break; end

        minus_eta = minus_one_lambda / t

        ŷ = value(predictor, X̄[i], w)
        grad!(▽, risk, X̄[i], w, y⃗[i], ŷ)
        axpy!(minus_eta, ▽⃗, w)

        nrm = one_sqrtlambda / vecnorm(w, 2)
        if nrm < 1
            broadcast!(*, w, w, nrm)
        end

        # In case the user requested to print the learning process and/or provided
        # callback function this code will provide just that.
        # Note: If no callback function is provided then the compiler will be able
        #       to optimize the empty callback call away.
        if has_callback || show_trace
            f = value!(ybuffer, risk, X, w, y⃗)
            show_trace && print_iter(t, f, vecnorm(w, 2))
            stopped = _docallback(predmodel, callback, t, w, f, ▽)
        end
    end

    f = value!(ybuffer, risk, X, w, y⃗)
    PrimalSolution(w, f, t, false)
end
