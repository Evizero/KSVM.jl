# KSVM

**UNDER CONSTRUCTION**

A Julia package for research and application of linear- and non-linear Support Vector Machines (SVM).

This package aims to make use of [Julia](http://julialang.org/)'s strength in order
to provide an easily extensible, highly modular, and at the same time computationally efficient framework for SVMs.

#### Why Julia?

Julia's high-level language design (which should feel familiar to the scientific community),
combined with it's high-performance, may have the potential to bridge the two-language problem,
that is arguably associated with research and application of machine learning algorithms.

It is our hope, that frameworks such as this enable a broader range of scientists to contribute
effective algorithms with fair comparisons, and without having to be fluent in low-level languages such as C

- [ ] TODO some objective benchmarks

#### Just for research?

While research is an important goal of this framework, it is of equal importance that all the included
algorithms are 1.) _well documented_, and thanks to Julia 2.) _efficient_.
Additionally, because of the framework's modular and transparent interface (e.g. callback functions),
it should be simple to use this library as basis for course exercises.

**Bottom Line:**
_The use of this framework for educational and/or commercial reasons is encouraged under the MIT "Expat" license._

## Installation

This is still a work in progress. Installation not yet advised.

## Usage

Here is a quick "Hello World" example of using a linear SVM on the example of the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/Iris). 

```Julia
# Adapted example from SVM.jl by John Myles White
using KSVM
using RDatasets
iris = dataset("datasets", "iris")
X = convert(Array, iris[1:100, 1:2])'  # The observations have to be in the columns
y = iris[1:100, :Species]
train = bitrand(size(X,2))             # Split training and testset
model = svm(X[:, train], y[train])     # Fit the linear SVM
acc = accuracy(model, X[:, ~train], y[~train])
```

![Iris Data SVM](https://cloud.githubusercontent.com/assets/10854026/10863507/1c5cf1e0-7fd0-11e5-8c72-cb30a4a67963.png)

_Note: The preview plot on the bottom is only provided if the dataset is small and lies in two dimensions_

## Linear Support Vector Machines

```Julia
svm(X, y⃗; nargs...) → SVM
```

The function `svm` fits a Support Vector Machine (SVM) to the given training data `X` and `y⃗`.
Depending on the specified parameters in `nargs`, the SVM can be trained to
perform various forms of regression or classification tasks.
There are a number of parameters that can be configured to the individual needs,
such as the type of lossfunction, the type of regularization, or the algorithm 
used for training the SVM in the primal/dual.

### Parameters

- **`X`** : The input matrix of the training set, with the observations as columns.
It can be either a dense or sparse matrix of floatingpoint numbers.
Note that it is generally advised to center and scale your data before training a SVM.

- **`y⃗`** : The target vector of the training set. It has to be the same length as `X` has columns.
In the case of classification, it can be either a vector with elements yᵢ ∈ {-1.0, 1.0}
for two class prediction, or a vector with elements yᵢ ∈ N for multi-class prediction.

- **`solver`** : The concrete algorithm to train the SVM. Depending on the algorithm
this will either solve the linear SVM in the primal or dual formulation.
In contrast to other frameworks, the algorithm does not determine if the solution 
of the primal or dual problem is returned (see parameter `dual`).

- **`loss`** : The utilized loss function. The typical loss functions for SVMs are
`HingeLoss()` (L1-SVM), `L2HingeLoss()` (L2-SVM), `SmoothedL1HingeLoss(h)`, or `ModifiedHuberLoss()` for classification,
and `EpsilonInsLoss(e)`, or `L2EpsilonInsLoss(e)` for support vector regression.
Note that in general each solvers is only able to deal with a subset of those
loss functions and some combinations might not be supported.
For example the implementation of `DualCD` does only support
`HingeLoss` and `L2HingeLoss`.

- **`regtype`** : The type of regularization that should be used. In general this can either
be `L1Reg` or `L2Reg`. Note that not all solver support `L1Reg`.

- **`bias`** : The scaling factor of the bias. If set to 0, no intercept will be fitted.

- **`C`** : The penalty parameter. It plays a role similar to λ⁻¹ in Empirical Risk Minimization.
Conceptually, it denotes the trade off between model complexity and the training error.
Larger C will increase the penalization of training errors and thus lead to behavior
similar to a hard margin classifier.

- **`dual`** : Boolean that specifies if the solution to the dual problem should be returned
instead of the solution to the primal (default = false).

    In contrast to other frameworks, the `dual` parameter does not influence whether the dual or
    the primal problem is solved (this is instead specified implicitly by the `solver` parameter),
    but the result the user is interested in.
    It is generally recommended to leave `dual = false`, unless the user is either explicitly
    interested in the support vectors, or the dataset is sparse and high-dimensional.
    
    _Note: Not all solver are able to natively provide both solutions. If the algorithm is unable
    to provide the desired solution type then the solution will be converted accordingly, if feasible._

- **`iterations`** : Specifies the maximum number of iterations before the solver exits with the
current (and probably suboptimal) solution.

- **`ftol`** : Specifies the tolerance for the change of the objective function value.

- **`xtol`** : Specifies the tolerance for the change of the solution.

- **`grtol`** : Specifies the tolerance for the change of the gradient norm.

- **`callback`** : The optional callback function with signature `f(i, w, v, G)`.
If a callback is specified, then it will be called
every iteration with the following four parameters in order:

    param | description 
    ---: | :----
    `i` | the current iteration number.
    `w` | the current coefficients (i.e. α if dual=true, or θ if dual=false).
    `v` | the current objective value.
    `G` | the current gradient.
    
    _Note: The callback function can be used for early stopping by returning the symbol `:Exit`._

- **`show_trace`** : Instead of (or additional to) the callback function, the user can enable the output of the training information to STDOUT.

- **`nargs...`** : Additional named arguments that are passed unchecked to the specified solver.
This functionality can be used to pass around special arguments that the library does not natively implement.

## Non-linear Support Vector Machines

coming soon

## License

This code is free to use under the terms of the MIT license.

## Acknowledgement

This package makes heavy use of the following packages in order to provide it's main functionality. To see at full list of utilized packages, please take a look at the REQUIRE file.

- [MLBase.jl](https://github.com/JuliaStats/MLBase.jl)
- [EmpiricalRisks.jl](https://github.com/lindahua/EmpiricalRisks.jl)
- [Regression.jl](https://github.com/lindahua/Regression.jl)
- [MLKernels.jl](https://github.com/trthatcher/MLKernels.jl)

Note that in the future the library will switch from EmpiricalRisks and Regression to LearnBase.jl and Optim.jl as backend

## References

- Chapelle, Olivier. "Training a support vector machine in the primal."
Neural Computation 19.5 (2007): 1155-1178. [link](http://www.mitpressjournals.org/doi/abs/10.1162/neco.2007.19.5.1155)

- Hsieh, Cho-Jui, et al. "A dual coordinate descent method for large-scale 
linear SVM." Proceedings of the 25th international conference on Machine learning. 
ACM, 2008. [link](http://doi.acm.org/10.1145/1390156.1390208)

- Platt, John. "Probabilistic outputs for support vector machines and comparisons 
to regularized likelihood methods." Advances in large margin classifiers 10.3 (1999): 61-74.
[link](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639)

- Platt, John. "Fast training of support vector machines using sequential minimal optimization." Advances in kernel methods—support vector learning 3 (1999). [link](http://dl.acm.org/citation.cfm?id=299105)
