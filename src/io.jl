function _printconverged(io::IO, converged, niters)
  symb = converged ? "✓ " : "✗ "
  col  = converged ? :green : :red
  str  = converged ? "converged" : "could not converge"
  print(io, "  ")
  print_with_color(col, io,  symb)
  print(io, " ", str, " after ")
  print_with_color(:white, io, string(niters))
  println(io, " iterations")
end

function _printvariable(io, i, symbol, coef; colored = true, newline = true)
  len = i - length(string(symbol))
  print(io, repeat(" ", len))
  if colored
    print_with_color(:cyan, io, string(symbol))
  else
    print(io,  string(symbol))
  end
  print(io, " = ")
  if typeof(coef) <: AbstractArray
    if size(coef,1) <= 10 && size(coef,2) == 1
      if typeof(coef[1]) <: Float64
        print_with_color(:white, io, string(map(x -> round(x, 4), coef)))
      else
        print_with_color(:white, io, string(coef))
      end
    else
      print_with_color(:white, io, "$(size(coef,1))x$(size(coef,2)), $(typeof(coef))")
    end
  else
    print_with_color(:white, io, string(coef))
  end
  if newline; print(io, "\n") end
end
