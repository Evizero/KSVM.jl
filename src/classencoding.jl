abstract ClassEncoding

abstract BinaryClassEncoding <: ClassEncoding

immutable IdentityBinaryClassEncoding <: BinaryClassEncoding
  labelmap::LabelMap{Int}
end

function IdentityBinaryClassEncoding(targets::AbstractVector)
  IdentityBinaryClassEncoding(labelmap(targets))
end

#-----------------------------------------------------------

immutable SignedClassEncoding{T} <: BinaryClassEncoding
  labelmap::LabelMap{T}

  function SignedClassEncoding(labelmap::LabelMap{T})
    numLabels = length(labelmap.vs)
    if numLabels != 2
      throw(ArgumentError("The given target vector must have exactly two classes"))
    end
    new(labelmap)
  end
end

function SignedClassEncoding{T}(lm::LabelMap{T})
  if T <: Float64
    if lm.vs == [-1., 1] || lm.vs == [1., -1] || lm.vs == [-1, 1] || lm.vs == [1, -1]
      return IdentityBinaryClassEncoding(labelmap([-1, 1]))
    end
  end
  SignedClassEncoding{T}(lm)
end

function SignedClassEncoding{T}(targets::AbstractVector{T})
  SignedClassEncoding(labelmap(targets))
end

nclasses(ce::BinaryClassEncoding) = 2

labels{C<:ClassEncoding}(ce::C) = ce.labelmap.vs

#-----------------------------------------------------------

function groupindices{T}(classEncoding::ClassEncoding, targets::AbstractVector{T})
  groupindices(classEncoding.labelmap, targets)
end

#-----------------------------------------------------------

function classdistribution{T}(labelmap::LabelMap{T}, targets::AbstractVector{T})
  labelmap.vs, map(length, groupindices(labelmap, targets))
end

function classdistribution{T}(classEncoding::ClassEncoding, targets::AbstractVector{T})
  classEncoding.labelmap.vs, map(length, groupindices(classEncoding, targets))
end

# ==========================================================================

function labelencode(classEncoding::IdentityBinaryClassEncoding, targets::AbstractVector)
  targets
end

function labeldecode(classEncoding::IdentityBinaryClassEncoding, values::AbstractVector)
  values
end

#-----------------------------------------------------------

function labelencode{T}(classEncoding::SignedClassEncoding{T}, targets::AbstractVector{T})
  indicies = labelencode(classEncoding.labelmap, targets)
  2(indicies - 1.5)
end

function labeldecode{T}(classEncoding::SignedClassEncoding{T}, values::AbstractVector{Float64})
  indicies = round(Integer, (values / 2.) + 1.5)
  labeldecode(classEncoding.labelmap, indicies)
end

# ==========================================================================

function getLabelString{T}(labelmap::LabelMap{T})
  labels = labelmap.vs
  c = length(labels)
  if c > 10
    labels = labels[1:10]
    labelString = string(join(labels, ", "), ", ... [TRUNC]")
  else
    labelString = join(labels, ", ")
  end
end

#-----------------------------------------------------------

function show(io::IO, classEncoding::IdentityBinaryClassEncoding)
  labelString = getLabelString(classEncoding.labelmap)
  print(io,
        """
        IdentityBinaryClassEncoding for {$labelString}""")
end

function show{T}(io::IO, classEncoding::SignedClassEncoding{T})
  labelString = getLabelString(classEncoding.labelmap)
  print(io,
        """
        SignedClassEncoding (Binary) to {-1, 1}
          .labelmap  ...  encoding for: {$labelString}""")
end
