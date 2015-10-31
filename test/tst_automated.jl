
function test_with_reference(X, y, ref)
  for i in 1:nrow(ref)
    loss = eval(symbol(ref[i, :loss]))()
    regtype = eval(symbol(ref[i, :reg]))
    solver = eval(symbol(ref[i, :solver]))()
    tol = ref[i, :tol]
    C = ref[i, :C]
    iter = ref[i, :iter]
    fval = ref[i, :f_val]
    nsv = ref[i, :nsv]
    acc = ref[i, :acc]
    msg2(ref[i, :solver], ", ", ref[i, :loss], ", C = "); @printf("%0.0e : ", C)
    retdual = svm(X, y, C = C, ftol = tol, maxiter = 50_000, dual = true, solver = solver, regtype = regtype, loss = loss)
    retprimal = svm(X, y, C = C, ftol = tol, maxiter = 50_000, dual = false, solver = solver, regtype = regtype, loss = loss)
    println("niters = $(retdual.details.niters) (ref: $iter)")
    if C > 1.
      # Dual prediction can take a while if there are a lot of SV
      @test acc - tol <= _accuracy(classify(retdual, X), y) <= acc + tol
    end
    @test acc - tol <= _accuracy(classify(retprimal, full(X)), y) <= acc + tol
    @test nsv == retdual.nsv
    @test nsv - 2 <= retprimal.nsv <= nsv + 2
    @test fval - 0.00001 <= retdual.details.fval <= fval + 0.00001
  end
end

for csvname in getFilenames(_referenceDir(), "csv")
  dataname = splitext(csvname)[1]
  loadIfExists(test_with_reference, dataname, "Dense: Compare accuracy to Reference", sparse = false)
end

for dataname in ["a1a", "a9a", "data1.git", "data2.git"]
  loadIfExists(test_with_reference, dataname, "Sparse: Compare accuracy to Reference", sparse = true)
end
