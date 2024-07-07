using Flux: flatten, @layer

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define custom Reshape layer
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    Reshape(shape)

A custom layer for Flux that reshapes its input to a specified shape.

This layer is useful when you need to change the dimensions of your data within
a Flux model. Unlike the built-in reshape operation in Julia, this custom layer
can be saved and loaded using packages such as BSON or JLD2.

# Arguments
- `shape`: The target shape. This can be any tuple of integers and colons.
  Colons are used to indicate dimensions whose size should be inferred such that
  the total number of elements remains the same.

# Examples
```julia
julia> r = Reshape(10, :)
Reshape((10, :))

julia> r(rand(5, 2))
10×1 Matrix{Float64}:
```

# Note
When saving and loading the model, make sure to include Reshape in the list of
layers to be processed by BSON or JLD2.
"""
struct Reshape
    shape  # The target shape for the reshape operation
end

@doc raw"""
    Reshape(args...)

Constructor for the Reshape struct that takes variable arguments.

This function allows us to create a Reshape instance with any shape.

# Arguments
- `args...`: Variable arguments representing the dimensions of the target shape.

# Returns
- A `Reshape` instance with the target shape set to the provided dimensions.

# Examples
```julia
julia> r = Reshape(10, :)
Reshape((10, :))
```
"""
function Reshape(args...)
    return Reshape(args)
end

@doc raw"""
(r::Reshape)(x)

This function is called during the forward pass of the model. It reshapes the
input `x` to the target `shape` stored in the Reshape instance `r`.

# Arguments
- `r::Reshape`: An instance of the Reshape struct.
- `x`: The input to be reshaped.

# Returns
- The reshaped input.

# Examples
```julia
julia> r = Reshape(10, :)
Reshape((10, :))

julia> r(rand(5, 2))
10×1 Matrix{Float64}:
 ...
```
"""
function (r::Reshape)(x)
    return reshape(x, r.shape)
end

# Register Reshape as a Flux layer This allows Flux to recognize Reshape during
# the backward pass (gradient computation) and when saving/loading the model
@layer Reshape

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define custom Flatten layer
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    Flatten()

A custom layer for Flux that flattens its input into a 1D vector.

This layer is useful when you need to change the dimensions of your data within
a Flux model. Unlike the built-in flatten operation in Julia, this custom layer
can be saved and loaded by packages such as BSON and JLD2.

# Examples
```julia
julia> f = Flatten()

julia> f(rand(5, 2))
10-element Vector{Float64}:
```
# Note 
When saving and loading the model, make sure to include Flatten in the list of
layers to be processed by BSON or JLD2.
"""
struct Flatten end

@doc raw""" 
    (f::Flatten)(x)

This function is called during the forward pass of the model. It flattens the
input x into a 1D vector.

# Arguments
- `f::Flatten`: An instance of the Flatten struct.
- `x`: The input to be flattened.

# Returns
The flattened input.
"""
function (f::Flatten)(x)
    return flatten(x)
end

# Register Reshape as a Flux layer This allows Flux to recognize Flatten during
# the backward pass (gradient computation) and when saving/loading the model
@layer Flatten

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define custom activation layer over dimensions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    ActivationOverDims(σ::Function, dims::Int)

A custom layer for Flux that applies an activation function over specified
dimensions.

This layer is useful when you need to apply an activation function over specific
dimensions of your data within a Flux model. Unlike the built-in activation
functions in Julia, this custom layer can be saved and loaded using the BSON or
JLD2 package.

# Arguments
- `σ::Function`: The activation function to be applied.
- `dims`: The dimensions over which the activation function should be applied.

# Note
When saving and loading the model, make sure to include `ActivationOverDims` in
the list of layers to be processed by BSON or JLD2.
"""
struct ActivationOverDims
    σ::Function
    dims
end

@doc raw"""
    (σ::ActivationOverDims)(x)

This function is called during the forward pass of the model. It applies the
activation function `σ.σ` over the dimensions `σ.dims` of the input `x`.

# Arguments
- `σ::ActivationOverDims`: An instance of the ActivationOverDims struct.
- `x`: The input to which the activation function should be applied.

# Returns
- The input `x` with the activation function applied over the specified
  dimensions.

# Note
This custom layer can be saved and loaded using the BSON package. When saving
and loading the model, make sure to include `ActivationOverDims` in the list of
layers to be processed by BSON or JLD2.
"""
function (σ::ActivationOverDims)(x)
    return σ.σ(x, dims=σ.dims)
end

# Register Reshape as a Flux layer This allows Flux to recognize Flatten during
# the backward pass (gradient computation) and when saving/loading the model
@layer ActivationOverDims