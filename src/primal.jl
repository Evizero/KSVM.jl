# ==========================================================================
# Unprocessed Solution for the primal problem

immutable PrimalSolution{T<:AbstractArray}
  minimizer::T
  minimum::Float64
  iterations::Int
  isconverged::Bool
end

function PrimalSolution{T<:AbstractArray}(
    minimizer::T,
    minimum::Float64,
    iterations::Int,
    isconverged::Bool)
  PrimalSolution{T}(minimizer, minimum, iterations, isconverged)
end

function Base.show(io::IO, r::PrimalSolution)
  println(io, typeof(r))
  _printvariable(io, 14, "minimizer()", minimizer(r))
  _printvariable(io, 14, "minimum()", minimum(r))
  _printvariable(io, 14, "iterations()", iterations(r))
  _printvariable(io, 14, "isconverged()", isconverged(r), newline = false)
end

coef(s::PrimalSolution) = s.minimizer
minimizer(s::PrimalSolution) = s.minimizer
Base.minimum(s::PrimalSolution) = s.minimum
iterations(s::PrimalSolution) = s.iterations
isconverged(s::PrimalSolution) = s.isconverged
