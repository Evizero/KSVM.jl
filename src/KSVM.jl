module KSVM

using EmpiricalRisks: PredictionModel, LinearPred, AffinePred, MvLinearPred, MvAffinePred, Regularizer, Loss, HingeLoss, SqrHingeLoss, EpsilonInsLoss, SmoothedHingeLoss, SqrSmoothedHingeLoss, ModifiedHuberLoss, L1Reg, SqrL2Reg

using Reexport
@reexport using MLBase
@reexport using MLKernels

using Compat
using ArrayViews
using UnicodePlots
using Base.LinAlg: BlasReal
using Regression: DescentSolver, Solution, Options

import StatsBase: predict, fit, fit!, coef, nobs, model_response
import MLBase: classify, labelencode, labeldecode, groupindices
import UnicodePlots: scatterplot
import EmpiricalRisks
import Regression
import Regression: solve!, Solver
import Base: print, show, convert

export

    HingeLoss,
    L1HingeLoss,
    L2HingeLoss,
    EpsilonInsLoss,
    SmoothedL1HingeLoss,
    ModifiedHuberLoss,

    isclassifier,
    decision_function,

    L1Reg,
    L2Reg,

    DualCD,
    DualCDWithShrinking,

    LinearSVMSpec,
    SVMSpec,
      SVMRiskSpec,
        CSVM,

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
    accuracy,

    svm


include("deprecate/bridge.jl")
include("common.jl")
include("classencoding.jl")
include("io.jl")
include("dual.jl")
include("svm_spec.jl")
include("svm_model.jl")
include("svc_svr.jl")
include("fit.jl")
include("interface.jl")
include("linear/solver/dualcd.jl")
include("linear/solver/dualcd_shrinking.jl")

end
