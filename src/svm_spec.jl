abstract SVMSpec{TKernel <: Kernel}

typealias LinearSVMSpec{TKernel <: ScalarProductKernel} SVMSpec{TKernel}

abstract SVMRiskSpec{TKernel<:Kernel, TLoss<:Loss, TReg<:Regularizer} <: SVMSpec{TKernel}

"""
`CSVM <: SVMRiskSpec <: SVMSpec`

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
immutable CSVM{TKernel<:Kernel, TLoss<:Loss, TReg<:Regularizer} <: SVMRiskSpec{TKernel, TLoss, TReg}
  kernel::TKernel
  loss::TLoss
  reg::TReg
  C::Float64
  
  function CSVM(kernel::TKernel, loss::TLoss, reg::TReg)
    new(kernel, loss, reg, Float64(1 / reg.c))
  end
end

function CSVM(kernel::Kernel, loss::Loss, reg::Regularizer)
  CSVM{typeof(kernel), typeof(loss), typeof(reg)}(kernel, loss, reg)
end

function CSVM{TReg<:Regularizer}(;
              kernel::Kernel = ScalarProductKernel(),
              loss::Loss = L2HingeLoss(),
              regtype::Type{TReg} = L2Reg,
              C::Real = 1)
  reg = TReg(Float64(1 / C))
  CSVM{typeof(kernel), typeof(loss), typeof(reg)}(kernel, loss, reg)
end

isclassifier(spec::CSVM) = isclassifier(spec.loss)
decision_function(spec::CSVM) = decision_function(spec.loss)

# ==========================================================================
# Base.show, Base.print

function print(io::IO, model::CSVM)
  print(io, "C-SVM ($(typeof(model.kernel).name.name), $(typeof(model.loss).name.name), $(typeof(model.reg).name.name), C = $(model.C))")
end

function show(io::IO, model::CSVM)
  println(io, "$(typeof(model))")
  _printvariable(io, 9, ".kernel", model.kernel)
  _printvariable(io, 9, ".loss", model.loss)
  _printvariable(io, 9, ".reg", model.reg)
  _printvariable(io, 9, ".C", model.C, newline = false)
end
