using KSVM

using Regression: Options, SqrL2Reg, L1Reg

function _predict(sol, X)
  k = size(X,1)
  predmodel = EmpiricalRisks.LinearPred(k)
  EmpiricalRisks.predict(predmodel, sol.sol, X)
end

function _predictB(sol, X)
  k = size(X,1)
  predmodel = EmpiricalRisks.AffinePred(k, 1.)
  EmpiricalRisks.predict(predmodel, sol.sol, X)
end

reg = SqrL2Reg(1.0e-2)

#-----------------------------------------------------------

msg("Test interface stability")

a = hcat(rand(100)+0, rand(100)+2)
b = hcat(rand(100)+2, rand(100)+0)
X = vcat(a,b)'
Y = vcat(ones(100), ones(100)*-1)

@test_throws ArgumentError ret = svm(X, Y, regtype = L1Reg)

#-----------------------------------------------------------

msg("Simple separable without bias with L1-SVM")

a = hcat(rand(100)+0, rand(100)+2)
b = hcat(rand(100)+2, rand(100)+0)
X = vcat(a,b)'
Y = vcat(ones(100), ones(100)*-1)

ret = Regression.solve(hingereg(X, Y; bias = 0.0), reg = reg,
                       solver = DualCD())
@test _accuracy(sign(_predict(ret, X)), Y) == 1.0

ret = Regression.solve(hingereg(X, Y; bias = 1.0), reg = reg,
                       solver = DualCD())
@test _accuracy(sign(_predictB(ret, X)), Y) == 1.0

ret = Regression.solve(hingereg(X, Y; bias = 0.0), reg = reg)
@test _accuracy(sign(_predict(ret, X)), Y) == 1.0

#-----------------------------------------------------------

msg("Simple separable without bias with L2-SVM")

ret = Regression.solve(sqrhingereg(X, Y; bias = 0.0), reg = reg,
                       solver = DualCD())
@test _accuracy(sign(_predict(ret, X)), Y) == 1.0

ret = Regression.solve(sqrhingereg(X, Y; bias = 1.0), reg = reg,
                       solver = DualCD())
@test _accuracy(sign(_predictB(ret, X)), Y) == 1.0

ret = Regression.solve(sqrhingereg(X, Y; bias = 0.0), reg = reg)
@test _accuracy(sign(_predict(ret, X)), Y) == 1.0

#-----------------------------------------------------------

msg("Simple separable without bias with SmoothedHingeLoss")

ret = Regression.solve(smoothedhingereg(X, Y; bias = 0.0), reg = reg)
@test _accuracy(sign(_predict(ret, X)), Y) == 1.0

ret = Regression.solve(smoothedhingereg(X, Y; bias = 1.0), reg = reg)
@test _accuracy(sign(_predictB(ret, X)), Y) == 1.0

#-----------------------------------------------------------

msg("Simple not separable without bias with L1-SVM")

a = hcat(rand(100)+2, rand(100)+2)
b = hcat(rand(100)+0, rand(100)+0)
X = vcat(a,b)'
Y = vcat(ones(100), ones(100)*-1)
train = bitrand(size(X,2))

ret = Regression.solve(hingereg(X, Y; bias = 0.0), reg = reg,
                       solver = DualCD())
@test _accuracy(sign(_predict(ret, X)), Y) < 1.0

ret = Regression.solve(hingereg(X, Y; bias = 1.0), reg = reg, options=Options(maxiter=1000),
                       solver = DualCD())
@test _accuracy(sign(_predictB(ret, X)), Y) == 1.0

ret = Regression.solve(hingereg(X, Y; bias = 0.0), reg = reg)
@test _accuracy(sign(_predict(ret, X)), Y) < 1.0

ret = Regression.solve(hingereg(X, Y; bias = 1.0), reg = reg)
@test _accuracy(sign(_predictB(ret, X)), Y) == 1.0

X = X - 1.
ret = Regression.solve(hingereg(X, Y; bias = 0.0), reg = reg,
                       solver = DualCD())
@test _accuracy(sign(_predict(ret, X)), Y) == 1.0

#-----------------------------------------------------------

msg("Simple not separable without bias with L2-SVM")

a = hcat(rand(100)+2, rand(100)+2)
b = hcat(rand(100)+0, rand(100)+0)
X = vcat(a,b)'
Y = vcat(ones(100), ones(100)*-1)
train = bitrand(size(X,2))

ret = Regression.solve(sqrhingereg(X, Y; bias = 0.0), reg = reg,
                       solver = DualCD())
@test _accuracy(sign(_predict(ret, X)), Y) < 1.0

ret = Regression.solve(sqrhingereg(X, Y; bias = 1.0), reg = reg, options=Options(maxiter=1000),
                       solver = DualCD())
@test _accuracy(sign(_predictB(ret, X)), Y) == 1.0

ret = Regression.solve(sqrhingereg(X, Y; bias = 0.0), reg = reg)
@test _accuracy(sign(_predict(ret, X)), Y) < 1.0

ret = Regression.solve(sqrhingereg(X, Y; bias = 1.0), reg = reg)
@test _accuracy(sign(_predictB(ret, X)), Y) == 1.0

X = X - 1.
ret = Regression.solve(sqrhingereg(X, Y; bias = 0.0), reg = reg, options=Options(maxiter=1000),
                       solver = DualCD())
@test _accuracy(sign(_predict(ret, X)), Y) == 1.0

#-----------------------------------------------------------

msg("Simple not separable without bias with L2-SVM")

a = hcat(rand(100)+2, rand(100)+2)
b = hcat(rand(100)+0, rand(100)+0)
X = vcat(a,b)'
Y = vcat(ones(100), ones(100)*-1)
train = bitrand(size(X,2))

ret = Regression.solve(smoothedhingereg(X, Y; bias = 0.0), reg = reg)
@test _accuracy(sign(_predict(ret, X)), Y) < 1.0

ret = Regression.solve(smoothedhingereg(X, Y; bias = 1.0), reg = reg)
@test _accuracy(sign(_predictB(ret, X)), Y) == 1.0

#-----------------------------------------------------------

msg("Native interface using the svm function")

a = hcat(rand(100)+2, rand(100)+2)
b = hcat(rand(100)+0, rand(100)+0)
X = vcat(a,b)'
Y = vcat(ones(100), ones(100)*-1)

ret = svm(X, Y, bias = 0., loss = HingeLoss(), dual = false)
@test _accuracy(sign(predict(ret, X)), Y) < 1.0

ret = svm(X, Y, bias = 1., loss = HingeLoss(), dual = false)
@test _accuracy(sign(predict(ret, X)), Y) == 1.0

ret = svm(X, Y, bias = 0., loss = SqrHingeLoss(), dual = false)
@test _accuracy(sign(predict(ret, X)), Y) < 1.0

ret = svm(X, Y, bias = 1., loss = SqrHingeLoss(), dual = false)
@test _accuracy(sign(predict(ret, X)), Y) == 1.0

#-----------------------------------------------------------

msg("Converting dual to primal solution")

model1 = svm(X, Y, bias = 1., dual = true)
model2 = convert(KSVM.PrimalSVM, model1)
@test typeof(model1) <: DualSVM
@test typeof(model2) <: PrimalSVM
@test all(round(predict(model1, X), 4) .== round(predict(model2, X), 4))

#-----------------------------------------------------------

msg("Test callback function for primal solution")

ret = svm(X, Y, bias = 1., dual = false, verbosity = :final) do t, w, v, g
  @test length(w) == size(X,1) + 1
  @test length(g) == size(X,2) # can't provide grad of w ...
  t % 50 == 0 && msg2("$t : $v", newline = true)
end
@test _accuracy(sign(predict(ret, X)), Y) == 1.0

#-----------------------------------------------------------

msg("Test callback function for primal solution (sparse)")

ret = svm(sparse(X), Y, bias = 1., dual = false, verbosity = :final) do t, w, v, g
  @test length(w) == size(X,1) + 1
  @test length(g) == size(X,2) # can't provide grad of w ...
  t % 50 == 0 && msg2("$t : $v", newline = true)
end
@test _accuracy(sign(predict(ret, X)), Y) == 1.0

#-----------------------------------------------------------

msg("Test callback function for primal solution (Regression.jl)")

ret = svm(X, Y, bias = 1., dual = false, verbosity = :final, solver = Regression.BFGS()) do t, w, v, g
  @test length(w) == size(X,1) + 1
  @test length(g) == size(X,1) + 1
  t % 10 == 0 && msg2("$t : $v", newline = true)
end
@test _accuracy(sign(predict(ret, X)), Y) == 1.0

#-----------------------------------------------------------

msg("Test callback function for dual solution")

ret = svm(X, Y, bias = 1., dual = true, verbosity = :final) do t, alpha, v, g
  @test length(alpha) == size(X,2)
  @test length(g) == size(X,2)
  t % 50 == 0 && msg2("$t : $v", newline = true)
end
@test _accuracy(sign(predict(ret, X)), Y) == 1.0

#-----------------------------------------------------------

msg("Test callback function for dual solution (sparse)")

ret = svm(sparse(X), Y, bias = 1., dual = true, verbosity = :final) do t, alpha, v, g
  @test length(alpha) == size(X,2)
  @test length(g) == size(X,2)
  t % 50 == 0 && msg2("$t : $v", newline = true)
end
@test _accuracy(sign(predict(ret, X)), Y) == 1.0
