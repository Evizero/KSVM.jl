using KSVM
using SVMLightLoader
using Base.Test
using DataFrames

function msg(args...; newline = true)
  print("   --> ", args...)
  newline && println()
end

function msg2(args...; newline = false)
  print("       - ", args...)
  newline && println()
end

_dataDir() = joinpath(Pkg.dir("KSVM"), "test/data")
_referenceDir() = joinpath(Pkg.dir("KSVM"), "test/reference")
_pathOfData(filename::AbstractString) = joinpath(_dataDir(), filename)
_pathOfCsv(filename::AbstractString) = joinpath(_referenceDir(), "$(filename).csv")
_accuracy(y, t) = countnz(y .== t) / length(y)

function getFilenames(dir::AbstractString, ending::AbstractString)
  allfilenames = readdir(dir)
  idx = map(x->endswith(x, ".$(ending)"), allfilenames)
  allfilenames[idx]
end

function loadIfExists(f::Function, filename::AbstractString, text::AbstractString; sparse::Bool = false)
  path = _pathOfData(filename)
  csvpath = _pathOfCsv(filename)
  if isfile(path) && isfile(csvpath)
    msg("file: \"$filename\" - $text")
    X, y = load_svmlight_file(path)
    ref = readtable(csvpath)
    X = sparse ? X : full(X)
    f(X, y, ref)
  else
    msg("[skipped missing] file: \"$filename\" - $text")
  end
end

macro spc_time(expr)
  quote
    $expr # compile
    print("       - ")
    @time $expr
    println()
  end
end

# ==========================================================================
# Specify tests

tests = [
  "tst_linear_bias.jl"
  "tst_automated.jl"
]

perf = [
  #"bm_datasource.jl"
]

for t in tests
  println("[->] $t")
  include(t)
  println("[OK] $t")
  println("====================================================================")
end

for p in perf
  println("[->] $p")
  include(p)
  println("[OK] $p")
  println("====================================================================")
end
