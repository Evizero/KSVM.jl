module KSVM

using Reexport
using LearnBase
@reexport using MLKernels

using Compat
using ArrayViews
using UnicodePlots
using Base.LinAlg: BlasReal

import StatsBase: predict, fit, fit!, coef, nobs, model_response
import MLBase: classify, labelencode, labeldecode, groupindices
import UnicodePlots: scatterplot
import Base: print, show, convert, minimum
import LearnBase: nclasses, labels, classdistribution

export

    HingeLoss,
    L1HingeLoss,
    L2HingeLoss,
    SmoothedL1HingeLoss,
    ModifiedHuberLoss,

    EpsilonInsLoss,
    L1EpsilonInsLoss,
    L2EpsilonInsLoss,

    L1Penalty,
    L2Penalty,

    DualCD,
    DualCDWithShrinking,

    LinearSVMSpec,
    SVMSpec,
      SVMRiskSpec,
        CSVM,

    PrimalSolution,
    DualSolution,

    SVM,
      PrimalSVM,
        DensePrimalSVM,
        SparsePrimalSVM,
      DualSVM,
        DenseDualSVM,
        SparseDualSVM,

    SVC,
    SVR,

    labels,

    details,
    intercept,
    isconverged,
    iterations,
    features,
    targets,
    params,
    fval,
    nsv,
    svindex,
    xsv,
    ysv,
    predmodel,
    bias,

    predict,
    classify,
    accuracy,

    svm


#include("deprecate/bridge.jl")
include("common.jl")
include("io.jl")
include("primal.jl")
include("dual.jl")
include("svm_spec.jl")
include("svm_model.jl")
include("svc_svr.jl")
include("fit.jl")
include("interface.jl")
include("linear/solver/dualcd.jl")
include("linear/solver/dualcd_shrinking.jl")

end
