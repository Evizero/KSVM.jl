typealias L1HingeLoss HingeLoss
typealias L2HingeLoss SqrHingeLoss
typealias SmoothedL1HingeLoss SmoothedHingeLoss

typealias MarginBasedLoss Union{L1HingeLoss, L2HingeLoss, SmoothedL1HingeLoss, ModifiedHuberLoss}

typealias L2Reg SqrL2Reg
typealias Optimizer Solver
