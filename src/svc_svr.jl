# ==========================================================================

immutable SVC{TModel <: SVM, TEnc <: BinaryClassEncoding}
  model::TModel
  encoding::TEnc
end

function SVC(model::SVM, encoding::BinaryClassEncoding = IdentityBinaryClassEncoding([-1,1]))
  SVC{typeof(model),typeof(encoding)}(model, encoding)
end

# ==========================================================================

immutable SVR{TModel <: SVM}
  model::TModel
end

SVR(model::SVM) = SVR{typeof(model)}(model)

function svc_or_svr(model::SVM)
  isclassifier(model) ? SVC(model) : SVR(model)
end

# ==========================================================================

for kind = (:SVC, :SVR)
  for op = (:details, :isconverged, :iterations, :features, :params, :intercept,
            :fval, :nsv, :svindex, :xsv, :predmodel, :bias, :isclassifier,
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
  fit = svc.model
  ŷ = predict(fit, X)
  t = ndims(ŷ) == 1 ? decision_function(fit.params)(ŷ) : vec(mapslices(indmax, ŷ, 1))
  labeldecode(svc.encoding, t)
end

function accuracy(svc::SVC, X, y)
  accuracy(svc.model, X, labelencode(svc.encoding, y))
end

# ==========================================================================

function scatterplot{TModel <: PrimalSVM}(
    svc::SVC{TModel};
    title::(@compat AbstractString) = "Primal SVM Classification Plot",
    xlim = [0.,0.],
    ylim = [0.,0.],
    nargs...)
  fit = svc.model
  size(fit.Xtrain,1) == 2 || throw(DimensionMismatch("Can only plot the SVM classification for a two-dimensional featurespace (i.e. size(X,1) == 2)"))
  intercept_fit = typeof(fit.predmodel) <: Union{AffinePred, MvAffinePred}
  offset = intercept_fit ? -(fit.details.sol[3] * fit.predmodel.bias) / fit.details.sol[2] : 0.
  slope = -fit.details.sol[1] / fit.details.sol[2]
  x1 = vec(view(fit.Xtrain, 1, :))
  x2 = vec(view(fit.Xtrain, 2, :))
  x1sv = x1[fit.svindex]
  x2sv = x2[fit.svindex]
  xmin = minimum(x1); xmax = maximum(x1)
  ymin = minimum(x2); ymax = maximum(x2)
  xlim = xlim == [0.,0.] ? [xmin, xmax] : xlim
  ylim = ylim == [0.,0.] ? [ymin, ymax] : ylim
  notalphaindex = setdiff(1:size(fit.Xtrain,2), fit.svindex)
  x1 = x1[notalphaindex]
  x2 = x2[notalphaindex]
  y = fit.Ytrain[notalphaindex]
  lbl = map(string, labels(svc))
  fig = scatterplot(x1[y.<0], x2[y.<0]; title = title, xlim = xlim, ylim = ylim, name = lbl[1], nargs...)
  scatterplot!(fig, x1[y.>0], x2[y.>0], name = lbl[2])
  scatterplot!(fig, x1sv, x2sv, color = :yellow, name = "support vectors")
  lineplot!(fig, offset, slope, color = :white)
  xlabel!(fig, "X₁")
  ylabel!(fig, "X₂")
  fig
end

function scatterplot{TModel <: DualSVM}(
    svc::SVC{TModel};
    title::(@compat AbstractString) = "Dual SVM Classification Plot",
    xlim = [0.,0.],
    ylim = [0.,0.],
    nargs...)
  fit = svc.model
  size(fit.Xtrain,1) == 2 || throw(DimensionMismatch("Can only plot the SVM classification for a two-dimensional featurespace (i.e. size(X,1) == 2)"))
  x1 = vec(view(fit.Xtrain, 1, :))
  x2 = vec(view(fit.Xtrain, 2, :))
  x1sv = x1[fit.svindex]
  x2sv = x2[fit.svindex]
  xmin = minimum(x1); xmax = maximum(x1)
  ymin = minimum(x2); ymax = maximum(x2)
  xlim = xlim == [0.,0.] ? [xmin, xmax] : xlim
  ylim = ylim == [0.,0.] ? [ymin, ymax] : ylim
  notalphaindex = setdiff(1:size(fit.Xtrain,2), fit.svindex)
  x1 = x1[notalphaindex]
  x2 = x2[notalphaindex]
  y = fit.Ytrain[notalphaindex]
  lbl = map(string, labels(svc))
  fig = scatterplot(x1[y.<0], x2[y.<0]; title = title, xlim = xlim, ylim = ylim, name = lbl[1], nargs...)
  scatterplot!(fig, x1[y.>0], x2[y.>0], name = lbl[2])
  scatterplot!(fig, x1sv, x2sv, color = :yellow, name = "support vectors")
  xlabel!(fig, "X₁")
  ylabel!(fig, "X₂")
  fig
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
