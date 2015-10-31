
#-----------------------------------------------------------

msg("Test fit SVM as a Statsmodel on DataFrame")

using RDatasets
iris = dataset("datasets", "iris")
iris = iris[iris[:Species] .!= "setosa", :]

lm=labelmap(iris[:Species])
length(lm.vs) == 2
(labelencode(lm, iris[:Species]) - 1.5) * 2
