# Custom Layers

`AutoEncoderToolkit.jl` provides a set of commonly-used custom layers for
building autoencoders. These layers need to be explicitly defined if you want to
save a train model and load it later. For example, if the input to the encoder
is an image in format `HWC` (height, width, channel), somewhere in the encoder
there must be a function that flattens its input to a vector for the mapping to
the latent space to be possible. If you were to define this with a simple
function, the libraries to save the the model such as `JLD2` or `BSON` would not
work with these anonymous function. This is why we provide this set of custom
layers that play along these libraries.

## [`Reshape`] (@id reshape)
```@docs
AutoEncoderToolkit.Reshape
AutoEncoderToolkit.Reshape(::AbstractArray)
```

## [`Flatten`] (@id flatten)
```@docs
AutoEncoderToolkit.Flatten
AutoEncoderToolkit.Flatten(::AbstractArray)
```

## `ActivationOverDims`
```@docs
AutoEncoderToolkit.ActivationOverDims
AutoEncoderToolkit.ActivationOverDims(::AbstractArray)
```