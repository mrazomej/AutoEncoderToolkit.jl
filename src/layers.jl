using Flux: flatten, @functor

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define custom Reshape layer
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    Reshape(shape)

A custom layer for Flux that reshapes its input to a specified shape.

This layer is useful when you need to change the dimensions of your data within
a Flux model. Unlike the built-in reshape operation in Julia, this custom layer
can be saved and loaded using the BSON package.

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
layers to be processed by BSON. 
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
    apply(r::Reshape, x)

Defines the behavior of the Reshape layer when used in a model. This function is
called during the forward pass of the model. It reshapes the input `x` to the
target `shape` stored in the Reshape instance `r`.

# Arguments
- `r::Reshape`: An instance of the Reshape struct.
- `x`: The input to be reshaped.

# Returns
- The reshaped input.

# Examples
```julia
julia> r = Reshape(10, :)
Reshape((10, :))

julia> apply(r, rand(5, 2))
10×1 Matrix{Float64}:
 ...
```
"""
function (r::Reshape)(x)
    return reshape(x, r.shape)
end

# Register Reshape as a Flux layer This allows Flux to recognize Reshape during
# the backward pass (gradient computation) and when saving/loading the model
@functor Reshape

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define custom Flatten layer
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    Flatten()

A custom layer for Flux that flattens its input into a 1D vector.

This layer is useful when you need to change the dimensions of your data within
a Flux model. Unlike the built-in flatten operation in Julia, this custom layer
can be saved and loaded using the BSON package.

# Examples
```julia
julia> f = Flatten()

julia> f(rand(5, 2))
10-element Vector{Float64}:
```
# Note 
When saving and loading the model, make sure to include Flatten in the list of
layers to be processed by BSON. 
"""
struct Flatten end

@doc raw""" 
    Flatten()

Constructor for the Flatten struct.

This function allows us to create a Flatten instance.

# Returns
A Flatten instance.

# Examples
```julia
julia> f = Flatten()
```
"""
function Flatten()
    return Flatten()
end

@doc raw""" 
    (f::Flatten)(x)

Defines the behavior of the Flatten layer when used in a model. This function is
called during the forward pass of the model. It flattens the input x into a 1D
vector.

# Arguments
- `f::Flatten`: An instance of the Flatten struct.
- `x`: The input to be flattened.

# Returns
The flattened input.
"""
function (f::Flatten)(x)
    return flatten(x)
end

# Register Reshape as a Flux layer This allows Flux to recognize Reshape during
# the backward pass (gradient computation) and when saving/loading the model
@functor Flatten