# Copyright (c) 2015: Christof Stocker

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
    penalize_bias::Bool
end

Pegasos(; penalize_bias::Bool = false) = Pegasos(penalize_bias)

# ==========================================================================
# Implementation of primal sub-gradient solver for SVMs
# This particular method is specialized for
# - linear kernel
# - dense arrays
# - univariate prediction

function fit{TKernel<:ScalarProductKernel, TLoss<:Union{MarginBasedLoss, DistanceBasedLoss}, TReg<:L2Penalty, INTERCEPT}(
        spec::CSVM{TKernel, TLoss, TReg},
        X::StridedMatrix, y⃗::StridedVector,
        solver::Pegasos,
        predmodel::UvPredicton{INTERCEPT},
        problem::PrimalProblem;
        ftol::Real = 1.0e-4,
        iterations::Int = 10000,
        show_trace::Bool = false,
        callback::Union{Function,Void} = nothing,
        nargs...)
    islipschitzcont_deriv(spec.loss) || throw(ArgumentError("The loss has to have a lipschitz continuous derivative"))

    # The risk functional needs to be strongly convex,
    # but since we limit the regularizer to L2Penalty,
    # it suffices to test the loss for convexity.
    isconvex(spec.loss) || throw(ArgumentError("The loss has to be convex"))

    # Get the size of the design matrix
    #   k ... number of features
    #   m ... number of observations
    k, m = size(X)Y +

    # Do some housekeeping: use shorter variable names for the options
    lambda = Float64(1 / (spec.C * m))
    has_callback = typeof(callback) <: Function
    bias = INTERCEPT ? Float64(predmodel.bias) : zero(Float64)
    loss = spec.loss
    reg = L2Penalty(lambda)
    predictor = LinearPredictor(bias)
    risk = EmpiricalRisk(predictor, loss, reg, solver.penalize_bias)

    # Print log header if show_trace is set
    show_trace && print_iter_head()

    # Initialize w to all zero.
    # Note: w has length k+1 to have space for the potential intercept.
    #       If no intercept is fit, the bias term will be ignored
    w  = zeros(Float64, INTERCEPT ? k+1 : k)

    # Storage for the current gradient
    ▽  = zeros(Float64, INTERCEPT ? k+1 : k, 1)
    ▽⃗  = view(▽, 1:length(▽), 1)

    # Indicies into the observations of X (columns)
    # They are random (uniform) for optimal average convergence.
    S = rand(1:m, iterations)

    # Preallocate the vector-views into the design matrix to iterate over the observations
    # Note: We have to process each observation of X in every iteration.
    #       This means that even though arrayviews are cheap,
    #       it is still cheaper to preallocate them beforehand.
    X̄ = Array(ContiguousView{Float64,1,Array{Float64,2}}, m)
    for i in 1:m
        @inbounds X̄[i] = view(X, :, i)
    end

    # Precompute constants
    #   1/√(λ)
    one_sqrtlambda = 1 / sqrt(lambda)
    #   -1/λ
    minus_one_lambda = -1 / lambda

    iteration = 0; stopped = false
    @inbounds for (iteration, i) in enumerate(S)
        if stopped; break; end

        # Compute current learning rate η
        #   -ηₜ = -1/(λ⋅t)
        minus_eta = minus_one_lambda / iteration

        # Compute prediction for the current observation
        #   ŷ = h(x⃗ᵢ,w⃗ₜ)
        ŷ = value(predictor, X̄[i], w) # Real number !

        # Compute the subgradient for the current observation
        #   ▽ₜ = λ⋅w⃗ₜ + L'(yᵢ, h(x⃗ᵢ,w⃗ₜ))⋅x⃗ᵢ
        grad!(▽, risk, X̄[i], w, y⃗[i], ŷ)

        # Update the coefficients using the the current subgradient
        #   w⃗ₜ₊₁ = w⃗ₜ - ηₜ⋅▽⃗ₜ
        axpy!(minus_eta, ▽⃗, w) # axpy!(a,X,Y) <=> Y = Y + a⋅X

        # Project the coefficients w on a ball
        #   nrm = (1/√(λ)) / ||w⃗ₜ₊₁||
        nrm = one_sqrtlambda / vecnorm(w, 2)
        if nrm < 1
            # w⃗ₜ₊₁ = min(1, nrm) ⋅ w⃗ₜ₊₁
            broadcast!(*, w, w, nrm)
        end

        # In case the user requested to print the learning process and/or provided
        # callback function this code will provide just that.
        # Note: If no callback function is provided then the compiler will be able
        #       to optimize the empty callback call away.
        if has_callback || show_trace
            # compute the risk of the current observation.
            f = value(risk, X, w, y⃗[i], ŷ)
            show_trace && print_iter(iteration, f, vecnorm(w, 2))
            stopped = _docallback(predmodel, callback, iteration, w, f, ▽)
        end
    end

    # Compute final objective value (risk) for the whole trainingset
    f = value(risk, X, w, y⃗)
    PrimalSolution(w, f, iteration, false)
end
