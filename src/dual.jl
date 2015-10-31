# ==========================================================================
# Unprocessed Solution for the dual problem

immutable DualSolution
  alpha::Vector{Float64}
  bias::Float64
  fval::Float64
  niters::Int
  converged::Bool
end

function Base.show(io::IO, r::DualSolution)
  println(io, "DualSolution:")
  println(io, "- alpha:      $(size(r.alpha)) $(typeof(r.alpha))")
  println(io, "- bias:       $(r.bias)")
  println(io, "- fval:       $(r.fval)")
  println(io, "- niters:     $(r.niters)")
  println(io, "- converged:  $(r.converged)")
end
