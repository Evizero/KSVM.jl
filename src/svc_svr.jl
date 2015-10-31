# ==========================================================================

immutable SVC{TModel <: SVM, TEnc <: BinaryClassEncoding}
  model::TModel
  encoding::TEnc
end

function SVC(model::SVM, encoding::BinaryClassEncoding)
  SVC{typeof(model),typeof(encoding)}(model, encoding)
end

# ==========================================================================

immutable SVR{TModel <: SVM}
  model::TModel
end

SVR(model::SVM) = SVR{typeof(model)}(model)

# ==========================================================================

for kind = (:SVC, :SVR)
  for op = (:details, :isconverged, :iterations, :features, :params, :intercept,
            :nsv, :svindex, :xsv, :predmodel, :bias, :minimum, :minimizer,
            :decision_function, :nobs, :model_response, :coef, :predict, :accuracy)
    @eval begin
      ($op)(svm::$kind) = ($op)(svm.model)
    end
  end
  @eval (predict)(svm::$kind, X) = (predict)(svm.model, X)
end

targets(svc::SVC) = labeldecode(svc.encoding, targets(svc.model))
ysv(svc::SVC) = labeldecode(svc.encoding, ysv(svc.model))

targets(svc::SVR) = targets(svc.model)
ysv(svc::SVR) = ysv(svc.model)

# ==========================================================================

labels(svc::SVC) = labels(svc.encoding)

classify(svc::SVC) = classify(svc, features(svc.model))

function classify(svc::SVC, X)
  t = classify(svc.model, X)
  labeldecode(svc.encoding, t)
end

function accuracy(svc::SVC, X, y)
  accuracy(svc.model, X, labelencode(svc.encoding, y))
end

# ==========================================================================

function scatterplot(scv::SVC, args...; nargs...)
  scatterplot(scv.model, args...; lbl = map(string, labels(scv)), nargs...)
end

# ==========================================================================
# Base.show

function show(io::IO, fit::SVC)
  println(io, typeof(fit), "\n")
  if typeof(fit.model) <: PrimalSVM 
    _showprimal(io, fit)
  else
    _showdual(io, fit)
  end
end
