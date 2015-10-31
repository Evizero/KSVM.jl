# ==========================================================================
# Unprocessed Solution for the dual problem

immutable DualSolution
  minimizer::Vector{Float64}
  bias::Float64
  minimum::Float64
  iterations::Int
  isconverged::Bool
end

function Base.show(io::IO, r::DualSolution)
  println(io, typeof(r))
  _printvariable(io, 14, "minimizer()", minimizer(r))
  _printvariable(io, 14, "bias()", bias(r))
  _printvariable(io, 14, "minimum()", minimum(r))
  _printvariable(io, 14, "iterations()", iterations(r))
  _printvariable(io, 14, "isconverged()", isconverged(r), newline = false)
end

coef(s::DualSolution) = s.minimizer
minimizer(s::DualSolution) = s.minimizer
Base.minimum(s::DualSolution) = s.minimum
bias(s::DualSolution) = s.bias
iterations(s::DualSolution) = s.iterations
isconverged(s::DualSolution) = s.isconverged
