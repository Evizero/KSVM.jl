abstract SVMSpec{TKernel <: Kernel}

typealias LinearSVMSpec{TKernel<:ScalarProductKernel} SVMSpec{TKernel}

abstract SVMRiskSpec{TKernel<:Kernel, TLoss<:Loss} <: SVMSpec{TKernel}

typealias SVCSpec{TKernel<:Kernel, TLoss<:MarginBasedLoss} SVMRiskSpec{TKernel, TLoss}
typealias SVRSpec{TKernel<:Kernel, TLoss<:DistanceBasedLoss} SVMRiskSpec{TKernel, TLoss}

abstract SVMFuncSpec{TKernel<:Kernel, TLoss<:Loss, TReg<:Penalty} <: SVMRiskSpec{TKernel, TLoss}

"""
`CSVM <: SVMFuncSpec <: SVMRiskSpec <: SVMSpec`

Description
============

Specification for a soft-margin Support Vector Machine using the C formulation.

Fields
=======

- **`.kernel`** : The utilized kernel function.

- **`.loss`** : The utilized loss function.

- **`.reg`** : The utilized regularizer, including penality parameter.
Note that the stored parameter is usually not C, but λ (= C⁻¹)

- **`.C`** : The penalty parameter. It plays a role similar to λ⁻¹ in Empirical Risk Minimization.
Conceptually, it denotes the trade off between model complexity and the training error.
Larger C will increase the penalization of training errors and thus lead to behavior
more similar to a hard margin classifier.

See also
=========

`PrimalSVM`, `DualSVM`

"""
immutable CSVM{TKernel<:Kernel, TLoss<:Loss, TReg<:Penalty} <: SVMFuncSpec{TKernel, TLoss, TReg}
  kernel::TKernel
  loss::TLoss
  reg::TReg
  C::Float64
  
  function CSVM(kernel::TKernel, loss::TLoss, reg::TReg)
    new(kernel, loss, reg, Float64(1 / reg.λ))
  end
end

function CSVM(kernel::Kernel, loss::Loss, reg::Penalty)
  CSVM{typeof(kernel), typeof(loss), typeof(reg)}(kernel, loss, reg)
end

function CSVM{TReg<:Penalty}(;
              kernel::Kernel = ScalarProductKernel(),
              loss::Loss = L2HingeLoss(),
              regtype::Type{TReg} = L2Penalty,
              C::Real = 1)
  reg = TReg(Float64(1 / C))
  CSVM{typeof(kernel), typeof(loss), typeof(reg)}(kernel, loss, reg)
end

# ==========================================================================
# Base.show, Base.print

function print(io::IO, model::CSVM)
  print(io, "C-SVM ($(typeof(model.kernel).name.name), $(typeof(model.loss).name.name), $(typeof(model.reg).name.name), C = $(model.C))")
end

function show(io::IO, model::CSVM)
  println(io, typeof(model))
  _printvariable(io, 9, ".kernel", model.kernel)
  _printvariable(io, 9, ".loss", model.loss)
  _printvariable(io, 9, ".reg", model.reg)
  _printvariable(io, 9, ".C", model.C, newline = false)
end
