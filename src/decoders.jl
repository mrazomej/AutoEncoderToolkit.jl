# Import ML libraries
import Flux

## ============================================================================
# Abstract Decoder Types
## ============================================================================

@doc raw"""
    AbstractDecoder

This is an abstract type that serves as a parent for all decoder models in this
package.

A decoder is part of an autoencoder model. It takes a lower-dimensional
representation produced by the encoder and reconstructs the original input data
from it. The goal of the decoder is to produce a reconstruction that is as close
as possible to the original input.

Subtypes of this abstract type should define specific types of decoders, such as
deterministic decoders, variational decoders, or other specialized decoder
types.
"""
abstract type AbstractDecoder end

@doc raw"""
    AbstractDeterministicDecoder <: AbstractDecoder

This is an abstract type that serves as a parent for all deterministic decoder
models in this package.

A deterministic decoder is a type of decoder that provides a deterministic
mapping from the lower-dimensional representation to the reconstructed input
data. This contrasts with stochastic or variational decoders, where the decoding
process may involve a random sampling step.

Subtypes of this abstract type should define specific types of deterministic
decoders, such as linear decoders, non-linear decoders, or other specialized
deterministic decoder types.
"""
abstract type AbstractDeterministicDecoder <: AbstractDecoder end

@doc raw"""
    AbstractVariationalDecoder <: AbstractDecoder

This is an abstract type that serves as a parent for all variational decoder
models in this package.

A variational decoder is a type of decoder that maps the lower-dimensional
representation to the parameters of a probability distribution from which the
reconstructed input data is sampled. This introduces stochasticity into the
model.

Subtypes of this abstract type should define specific types of variational
decoders, such as Gaussian decoders, or other specialized variational decoder
types.
"""
abstract type AbstractVariationalDecoder <: AbstractDecoder end

@doc raw"""
    AbstractGaussianDecoder <: AbstractVariationalDecoder

This is an abstract type that serves as a parent for all Gaussian decoder models
in this package.

A Gaussian decoder is a type of variational decoder that maps the
lower-dimensional latent variables to the parameters of a Gaussian distribution
from which the reconstructed input data is sampled. This introduces
stochasticity into the model.

Subtypes of this abstract type should define specific types of Gaussian
decoders, or other specialized Gaussian decoder types.
"""
abstract type AbstractGaussianDecoder <: AbstractVariationalDecoder end

"""
    AbstractGaussianLogDecoder <: AbstractGaussianDecoder

An abstract type representing a variational autoencoder's decoder that returns
the log of the standard deviation.

This type is used to represent decoders that, given a latent variable `z`,
output not only the mean of the reconstructed data distribution but also the log
of the standard deviation. This is useful for variational autoencoders where the
decoder's output is modeled as a Gaussian distribution, and we want to learn
both its mean and standard deviation.

Subtypes of this abstract type should implement the `decode` method, which takes
a latent variable `z` and returns a tuple `(μ, logσ)`, where `μ` is the mean of
the reconstructed data distribution and `logσ` is the log of the standard
deviation.
"""
abstract type AbstractGaussianLogDecoder <: AbstractGaussianDecoder end

"""
    AbstractGaussianLinearDecoder <: AbstractGaussianDecoder

An abstract type representing a variational autoencoder's decoder that returns
the standard deviation directly.

This type is used to represent decoders that, given a latent variable `z`,
output not only the mean of the reconstructed data distribution but also the
standard deviation directly. This is useful for variational autoencoders where
the decoder's output is modeled as a Gaussian distribution, and we want to learn
both its mean and standard deviation.

The activation function for the last layer of the decoder should be strictly
positive to ensure the standard deviation is positive.

Subtypes of this abstract type should implement the `decode` method, which takes
a latent variable `z` and returns a tuple `(μ, σ)`, where `μ` is the mean of the
reconstructed data distribution and `σ` is the standard deviation.
"""
abstract type AbstractGaussianLinearDecoder <: AbstractGaussianDecoder end

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Deterministic Decoders
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ==============================================================================

@doc raw"""
`struct Decoder`

Default decoder function for deterministic autoencoders. The `decoder` network
is used to map the latent space representation directly back to the original
data space.

# Fields
- `decoder::Flux.Chain`: The primary neural network used to process the latent
  space representation and map it back to the data space.

# Example
```julia
dec = Decoder(Flux.Chain(Dense(20, 400, relu), Dense(400, 784)))
```
"""
struct Decoder <: AbstractDeterministicDecoder
    decoder::Flux.Chain
end # struct

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor Decoder

@doc raw"""
    Decoder(n_input, n_latent, decoder_neurons, decoder_activation, 
            output_activation; init=Flux.glorot_uniform)

Construct and initialize a `Decoder` struct that defines a decoder network for a
deterministic autoencoder.

# Arguments
- `n_input::Int`: The dimensionality of the output data (which typically matches
  the input data dimensionality of the autoencoder).
- `n_latent::Int`: The dimensionality of the latent space.
- `decoder_neurons::Vector{<:Int}`: A vector specifying the number of neurons in
  each layer of the decoder network.
- `decoder_activation::Vector{<:Function}`: Activation functions corresponding
  to each layer in the `decoder_neurons`.
- `output_activation::Function`: Activation function for the final output layer.

## Optional Keyword Arguments
- `init::Function=Flux.glorot_uniform`: The initialization function used for the
  neural network weights.

# Returns
- A `Decoder` struct initialized based on the provided arguments.

# Examples
```julia
decoder = Decoder(784, 20, sigmoid, [400], [relu])
```

# Notes
The length of decoder_neurons should match the length of decoder_activation,
ensuring that each layer in the decoder has a corresponding activation function.
"""
function Decoder(
    n_input::Int,
    n_latent::Int,
    decoder_neurons::Vector{<:Int},
    decoder_activation::Vector{<:Function},
    output_activation::Function;
    init::Function=Flux.glorot_uniform
)
    # Check there's enough activation functions for all layers
    if length(decoder_activation) != length(decoder_neurons)
        error("Each layer needs exactly one activation function in decoder")
    end # if

    # Initialize list with decoder layers
    layers = []

    # Add first layer from latent space to decoder
    push!(
        layers,
        Flux.Dense(
            n_latent => decoder_neurons[1],
            decoder_activation[1];
            init=init
        )
    )

    # Check if there are multiple middle layers
    if length(decoder_neurons) > 1
        # Loop through middle layers
        for i = 2:length(decoder_neurons)
            # Set middle layers of decoder
            push!(
                layers,
                Flux.Dense(
                    decoder_neurons[i-1] => decoder_neurons[i],
                    decoder_activation[i]; init=init
                )
            )
        end # for
    end # if

    # Add last layer from decoder to output
    push!(
        layers, Flux.Dense(decoder_neurons[end] => n_input, output_activation)
    )

    return Decoder(Flux.Chain(layers...))
end # function

@doc raw"""
    (decoder::Decoder)(z)

Forward propagate the encoded representation `z` through the `Decoder` to obtain
the reconstructed input data.

# Arguments
- `z::Array{Float32}`: Encoded representation in the latent space.

# Returns
- `x_reconstructed`: Reconstructed version of the original input data after
  decoding from the latent space.

# Description
This method allows for a direct call on an instance of `Decoder` with the
encoded data `z`. It runs the encoded representation through the decoder network
and outputs the reconstructed version of the original input data.

# Example
```julia
dec = Decoder(...)
x_reconstructed = dec(encoded_representation)
````

# Note
Ensure that the input z matches the expected dimensionality of the decoder's
input layer.
"""
function (decoder::Decoder)(z::AbstractVecOrMat{Float32})
    # Run encoded representation through the decoder network to obtain the
    # reconstructed data
    return decoder.decoder(z)
end # function

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Variational Decoders
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ==============================================================================
# struct SimpleDecoder <: AbstractGaussianDecoder
# ==============================================================================

@doc raw"""
    SimpleDecoder <: AbstractGaussianDecoder

A straightforward decoder structure for variational autoencoders (VAEs) that
contains only a single decoder network.

# Fields
- `decoder::Flux.Chain`: The primary neural network used to process the latent
  space and map it to the output (or reconstructed) space.

# Description
`SimpleDecoder` represents a basic VAE decoder without explicit components for
the latent space's mean (`µ`) or log standard deviation (`logσ`). It's commonly
used when the VAE's latent space distribution is implicitly defined, and there's
no need for separate paths or operations on the mean or log standard deviation.
"""
struct SimpleDecoder <: AbstractGaussianDecoder
    decoder::Flux.Chain
end # struct

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor SimpleDecoder

@doc raw"""
    SimpleDecoder(n_input, n_latent, decoder_neurons, decoder_activation, 
                output_activation; init=Flux.glorot_uniform)

Constructs and initializes a `SimpleDecoder` object designed for variational
autoencoders (VAEs). This function sets up a straightforward decoder network
that maps from a latent space to an output space.

# Arguments
- `n_input::Int`: Dimensionality of the output data (or the data to be
  reconstructed).
- `n_latent::Int`: Dimensionality of the latent space.
- `decoder_neurons::Vector{<:Int}`: Vector of layer sizes for the decoder
  network, not including the input latent layer and the final output layer.
- `decoder_activation::Vector{<:Function}`: Activation functions for each
  decoder layer, not including the final output layer.
- `output_activation::Function`: Activation function for the final output layer.

## Optional Keyword Arguments
- `init::Function=Flux.glorot_uniform`: Initialization function for the network
  parameters.

# Returns
A `SimpleDecoder` object with the specified architecture and initialized
weights.

# Description
This function constructs a `SimpleDecoder` object, setting up its decoder
network based on the provided specifications. The architecture begins with a
dense layer mapping from the latent space, goes through a sequence of middle
layers if specified, and finally maps to the output space.

The function ensures that there are appropriate activation functions provided
for each layer in the `decoder_neurons` and checks for potential mismatches in
length.

# Example
```julia
n_input = 28*28
n_latent = 64
decoder_neurons = [128, 256]
decoder_activation = [relu, relu]
output_activation = sigmoid
decoder = SimpleDecoder(
    n_input, n_latent, decoder_neurons, decoder_activation, output_activation
)
```

# Note
Ensure that the lengths of decoder_neurons and decoder_activation match,
excluding the output layer.
"""
function SimpleDecoder(
    n_input::Int,
    n_latent::Int,
    decoder_neurons::Vector{<:Int},
    decoder_activation::Vector{<:Function},
    output_activation::Function;
    init::Function=Flux.glorot_uniform
)
    # Check there's enough activation functions for all layers
    if (length(decoder_activation) != length(decoder_neurons))
        error("Each layer needs exactly one activation function")
    end # if

    # Initialize list with decoder layers
    decoder = Array{Flux.Dense}(undef, length(decoder_neurons) + 1)

    # Add first layer from latent space to decoder
    decoder[1] = Flux.Dense(
        n_latent => decoder_neurons[1], decoder_activation[1]; init=init
    )

    # Add last layer from decoder to output
    decoder[end] = Flux.Dense(
        decoder_neurons[end] => n_input, output_activation; init=init
    )

    # Check if there are multiple middle layers
    if length(decoder_neurons) > 1
        # Loop through middle layers
        for i = 2:length(decoder_neurons)
            # Set middle layers of decoder
            decoder[i] = Flux.Dense(
                decoder_neurons[i-1] => decoder_neurons[i],
                decoder_activation[i];
                init=init
            )
        end # for
    end # if

    # Initialize simple decoder
    return SimpleDecoder(Flux.Chain(decoder...))
end # function

@doc raw"""
    (decoder::SimpleDecoder)(z::AbstractVecOrMat{Float32})

Maps the given latent representation `z` through the `SimpleDecoder` network to
reconstruct the original input.

# Arguments
- `z::AbstractVecOrMat{Float32}`: The latent space representation to be decoded.
  This can be a vector or a matrix, where each column represents a separate
  sample from the latent space of a VAE.

# Returns
- A NamedTuple `(µ=µ,)` where `µ` is an array representing the output of the
  decoder, which should resemble the original input to the VAE (post encoding
  and sampling from the latent space).

# Description
This function processes the latent space representation `z` using the neural
network defined in the `SimpleDecoder` struct. The aim is to decode or
reconstruct the original input from this representation.

# Example
```julia
decoder = SimpleDecoder(...)
z = ... # some latent space representation
output = decoder(z)
```
# Note

Ensure that the latent space representation z matches the expected input
dimensionality for the SimpleDecoder.
"""
function (decoder::SimpleDecoder)(
    z::Float32Array
)
    # Run input to decoder network
    return (µ=decoder.decoder(z),)
end # function

# ==============================================================================
# struct JointLogDecoder <: AbstractVariationalDecoder
# ==============================================================================

@doc raw"""
    JointLogDecoder <: AbstractGaussianLogDecoder

An extended decoder structure for VAEs that incorporates separate layers for
mapping from the latent space to both its mean (`µ`) and log standard deviation
(`logσ`).

# Fields
- `decoder::Flux.Chain`: The primary neural network used to process the latent
  space before determining its mean and log standard deviation.
- `µ::Flux.Dense`: A dense layer that maps from the output of the `decoder` to
  the mean of the latent space.
- `logσ::Flux.Dense`: A dense layer that maps from the output of the `decoder`
  to the log standard deviation of the latent space.

# Description
`JointLogDecoder` is tailored for VAE architectures where the same decoder network
is used initially, and then splits into two separate paths for determining both
the mean and log standard deviation of the latent space.
"""
struct JointLogDecoder <: AbstractGaussianLogDecoder
    decoder::Flux.Chain
    µ::Flux.Dense
    logσ::Flux.Dense
end

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor JointLogDecoder

@doc raw"""
    JointLogDecoder(n_input, n_latent, decoder_neurons, decoder_activation, 
                latent_activation; init=Flux.glorot_uniform)

Constructs and initializes a `JointLogDecoder` object for variational
autoencoders (VAEs). This function sets up a decoder network that first
processes the latent space and then maps it separately to both its mean (`µ`)
and log standard deviation (`logσ`).

# Arguments
- `n_input::Int`: Dimensionality of the output data (or the data to be
  reconstructed).
- `n_latent::Int`: Dimensionality of the latent space.
- `decoder_neurons::Vector{<:Int}`: Vector of layer sizes for the primary
  decoder network, not including the input latent layer.
- `decoder_activation::Vector{<:Function}`: Activation functions for each
  primary decoder layer.
- `output_activation::Function`: Activation function for the mean (`µ`) and log
  standard deviation (`logσ`) layers.

# Optional Keyword Arguments
- `init::Function=Flux.glorot_uniform`: Initialization function for the network
  parameters.

# Returns
A `JointLogDecoder` object with the specified architecture and initialized
weights.

# Description
This function constructs a `JointLogDecoder` object, setting up its primary
decoder network based on the provided specifications. The architecture begins
with a dense layer mapping from the latent space and goes through a sequence of
middle layers if specified. After processing the latent space through the
primary decoder, it then maps separately to both its mean (`µ`) and log standard
deviation (`logσ`).

# Example
```julia
n_input = 28*28
n_latent = 64
decoder_neurons = [128, 256]
decoder_activation = [relu, relu]
output_activation = tanh
decoder = JointLogDecoder(
    n_input, n_latent, decoder_neurons, decoder_activation, output_activation
)
```

# Note
Ensure that the lengths of decoder_neurons and decoder_activation match.
"""
function JointLogDecoder(
    n_input::Int,
    n_latent::Int,
    decoder_neurons::Vector{<:Int},
    decoder_activation::Vector{<:Function},
    output_activation::Function;
    init::Function=Flux.glorot_uniform
)
    # Check there's enough activation functions for all layers
    if (length(decoder_activation) != length(decoder_neurons))
        error("Each layer needs exactly one activation function")
    end # if

    # Initialize list with decoder layers
    decoder_layers = Array{Flux.Dense}(undef, length(decoder_neurons))

    # Add first layer from latent space to decoder
    decoder_layers[1] = Flux.Dense(
        n_latent => decoder_neurons[1], decoder_activation[1]; init=init
    )

    # Check if there are multiple middle layers
    if length(decoder_neurons) > 1
        # Loop through middle layers if they exist
        for i = 2:length(decoder_neurons)
            decoder_layers[i] = Flux.Dense(
                decoder_neurons[i-1] => decoder_neurons[i],
                decoder_activation[i];
                init=init
            )
        end # for
    end # if

    # Construct the primary decoder
    decoder_chain = Flux.Chain(decoder_layers...)

    # Define layers that map from the last decoder layer to the mean and log
    # standard deviation
    µ_layer = Flux.Dense(
        decoder_neurons[end] => n_input, output_activation; init=init
    )
    logσ_layer = Flux.Dense(
        decoder_neurons[end] => n_input, output_activation; init=init
    )

    # Initialize joint decoder
    return JointLogDecoder(decoder_chain, µ_layer, logσ_layer)
end

@doc raw"""
    JointLogDecoder(n_input, n_latent, decoder_neurons, decoder_activation, 
                latent_activation; init=Flux.glorot_uniform)

Constructs and initializes a `JointLogDecoder` object for variational
autoencoders (VAEs). This function sets up a decoder network that first
processes the latent space and then maps it separately to both its mean (`µ`)
and log standard deviation (`logσ`).

# Arguments
- `n_input::Int`: Dimensionality of the output data (or the data to be
  reconstructed).
- `n_latent::Int`: Dimensionality of the latent space.
- `decoder_neurons::Vector{<:Int}`: Vector of layer sizes for the primary
  decoder network, not including the input latent layer.
- `decoder_activation::Vector{<:Function}`: Activation functions for each
  primary decoder layer.
- `output_activation::Vector{<:Function}`: Activation functions for the mean
  (`µ`) and log standard deviation (`logσ`) layers.

# Optional Keyword Arguments
- `init::Function=Flux.glorot_uniform`: Initialization function for the network
  parameters.

# Returns
A `JointLogDecoder` object with the specified architecture and initialized
weights.

# Description
This function constructs a `JointLogDecoder` object, setting up its primary
decoder network based on the provided specifications. The architecture begins
with a dense layer mapping from the latent space and goes through a sequence of
middle layers if specified. After processing the latent space through the
primary decoder, it then maps separately to both its mean (`µ`) and log standard
deviation (`logσ`).

# Example
```julia
n_input = 28*28
n_latent = 64
decoder_neurons = [128, 256]
decoder_activation = [relu, relu]
output_activation = [tanh, identity]
decoder = JointLogDecoder(
    n_input, n_latent, decoder_neurons, decoder_activation, latent_activation
)
```

# Note
Ensure that the lengths of decoder_neurons and decoder_activation match.
"""
function JointLogDecoder(
    n_input::Int,
    n_latent::Int,
    decoder_neurons::Vector{<:Int},
    decoder_activation::Vector{<:Function},
    output_activation::Vector{<:Function};
    init::Function=Flux.glorot_uniform
)
    # Check there's enough activation functions for all layers
    if (length(decoder_activation) != length(decoder_neurons))
        error("Each layer needs exactly one activation function")
    end # if

    # Initialize list with decoder layers
    decoder_layers = Array{Flux.Dense}(undef, length(decoder_neurons))

    # Add first layer from latent space to decoder
    decoder_layers[1] = Flux.Dense(
        n_latent => decoder_neurons[1], decoder_activation[1]; init=init
    )

    # Check if there are multiple middle layers
    if length(decoder_neurons) > 1
        # Loop through middle layers if they exist
        for i = 2:length(decoder_neurons)
            decoder_layers[i] = Flux.Dense(
                decoder_neurons[i-1] => decoder_neurons[i],
                decoder_activation[i];
                init=init
            )
        end # for
    end # if

    # Construct the primary decoder
    decoder_chain = Flux.Chain(decoder_layers...)

    # Define layers that map from the last decoder layer to the mean and log
    # standard deviation
    µ_layer = Flux.Dense(
        decoder_neurons[end] => n_input, output_activation[1]; init=init
    )
    logσ_layer = Flux.Dense(
        decoder_neurons[end] => n_input, output_activation[2]; init=init
    )

    # Initialize joint decoder
    return JointLogDecoder(decoder_chain, µ_layer, logσ_layer)
end

@doc raw"""
        (decoder::JointLogDecoder)(z::AbstractVecOrMat{Float32})

Maps the given latent representation `z` through the `JointLogDecoder` network
to produce both the mean (`µ`) and log standard deviation (`logσ`).

# Arguments
- `z::AbstractVecOrMat{Float32}`: The latent space representation to be decoded.
  This can be a vector or a matrix, where each column represents a separate
  sample from the latent space of a VAE.

# Returns
- A NamedTuple `(µ=µ, logσ=logσ,)` where:
    - `µ::Array{Float32}`: The mean representation obtained from the decoder.
    - `logσ::Array{Float32}`: The log standard deviation representation obtained
      from the decoder.

# Description
This function processes the latent space representation `z` using the primary
neural network of the `JointLogDecoder` struct. It then separately maps the
output of this network to the mean and log standard deviation using the `µ` and
`logσ` dense layers, respectively.

# Example
```julia
decoder = JointLogDecoder(...)
z = ... # some latent space representation
output = decoder(z)
```

# Note
Ensure that the latent space representation z matches the expected input
dimensionality for the JointLogDecoder.
"""
function (decoder::JointLogDecoder)(
    z::Float32Array
)
    # Run input through the primary decoder network
    h = decoder.decoder(z)
    # Map to mean
    µ = decoder.µ(h)
    # Map to log standard deviation
    logσ = decoder.logσ(h)
    return (µ=µ, logσ=logσ)
end # function

# ==============================================================================
# struct JointDecoder <: AbstractVariationalDecoder
# ==============================================================================

@doc raw"""
    JointDecoder <: AbstractGaussianLinearDecoder

An extended decoder structure for VAEs that incorporates separate layers for
mapping from the latent space to both its mean (`µ`) and standard deviation
(`σ`).

# Fields
- `decoder::Flux.Chain`: The primary neural network used to process the latent
  space before determining its mean and log standard deviation.
- `µ::Flux.Dense`: A dense layer that maps from the output of the `decoder` to
  the mean of the latent space.
- `σ::Flux.Dense`: A dense layer that maps from the output of the `decoder` to
  the standard deviation of the latent space.

# Description
`JointDecoder` is tailored for VAE architectures where the same decoder network
is used initially, and then splits into two separate paths for determining both
the mean and standard deviation of the latent space.
"""
struct JointDecoder <: AbstractGaussianLinearDecoder
    decoder::Flux.Chain
    µ::Flux.Dense
    σ::Flux.Dense
end

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor JointDecoder

@doc raw"""
    JointDecoder(n_input, n_latent, decoder_neurons, decoder_activation, 
                latent_activation; init=Flux.glorot_uniform)

Constructs and initializes a `JointLogDecoder` object for variational
autoencoders (VAEs). This function sets up a decoder network that first
processes the latent space and then maps it separately to both its mean (`µ`)
and log standard deviation (`logσ`).

# Arguments
- `n_input::Int`: Dimensionality of the output data (or the data to be
  reconstructed).
- `n_latent::Int`: Dimensionality of the latent space.
- `decoder_neurons::Vector{<:Int}`: Vector of layer sizes for the primary
  decoder network, not including the input latent layer.
- `decoder_activation::Vector{<:Function}`: Activation functions for each
  primary decoder layer.
- `output_activation::Function`: Activation function for the mean (`µ`) and log
  standard deviation (`logσ`) layers.

# Optional Keyword Arguments
- `init::Function=Flux.glorot_uniform`: Initialization function for the network
  parameters.

# Returns
A `JointDecoder` object with the specified architecture and initialized weights.

# Description
This function constructs a `JointDecoder` object, setting up its primary decoder
network based on the provided specifications. The architecture begins with a
dense layer mapping from the latent space and goes through a sequence of middle
layers if specified. After processing the latent space through the primary
decoder, it then maps separately to both its mean (`µ`) and standard deviation
(`σ`).

# Example
```julia
n_input = 28*28
n_latent = 64
decoder_neurons = [128, 256]
decoder_activation = [relu, relu]
output_activation = tanh
decoder = JointDecoder(
    n_input, n_latent, decoder_neurons, decoder_activation, output_activation
)
```

# Note
Ensure that the lengths of decoder_neurons and decoder_activation match.
"""
function JointDecoder(
    n_input::Int,
    n_latent::Int,
    decoder_neurons::Vector{<:Int},
    decoder_activation::Vector{<:Function},
    output_activation::Function;
    init::Function=Flux.glorot_uniform
)
    # Check there's enough activation functions for all layers
    if (length(decoder_activation) != length(decoder_neurons))
        error("Each layer needs exactly one activation function")
    end # if

    # Initialize list with decoder layers
    decoder_layers = Array{Flux.Dense}(undef, length(decoder_neurons))

    # Add first layer from latent space to decoder
    decoder_layers[1] = Flux.Dense(
        n_latent => decoder_neurons[1], decoder_activation[1]; init=init
    )

    # Check if there are multiple middle layers
    if length(decoder_neurons) > 1
        # Loop through middle layers if they exist
        for i = 2:length(decoder_neurons)
            decoder_layers[i] = Flux.Dense(
                decoder_neurons[i-1] => decoder_neurons[i],
                decoder_activation[i];
                init=init
            )
        end # for
    end # if

    # Construct the primary decoder
    decoder_chain = Flux.Chain(decoder_layers...)

    # Define layers that map from the last decoder layer to the mean and log
    # standard deviation
    µ_layer = Flux.Dense(
        decoder_neurons[end] => n_input, output_activation; init=init
    )
    logσ_layer = Flux.Dense(
        decoder_neurons[end] => n_input, output_activation; init=init
    )

    # Initialize joint decoder
    return JointDecoder(decoder_chain, µ_layer, logσ_layer)
end

@doc raw"""
    JointDecoder(n_input, n_latent, decoder_neurons, decoder_activation, 
                latent_activation; init=Flux.glorot_uniform)

Constructs and initializes a `JointDecoder` object for variational autoencoders
(VAEs). This function sets up a decoder network that first processes the latent
space and then maps it separately to both its mean (`µ`) and standard deviation
(`σ`).

# Arguments
- `n_input::Int`: Dimensionality of the output data (or the data to be
  reconstructed).
- `n_latent::Int`: Dimensionality of the latent space.
- `decoder_neurons::Vector{<:Int}`: Vector of layer sizes for the primary
  decoder network, not including the input latent layer.
- `decoder_activation::Vector{<:Function}`: Activation functions for each
  primary decoder layer.
- `output_activation::Function`: Activation function for the mean (`µ`) and
  standard deviation (`σ`) layers.

# Optional Keyword Arguments
- `init::Function=Flux.glorot_uniform`: Initialization function for the network
  parameters.

# Returns
A `JointDecoder` object with the specified architecture and initialized weights.

# Description
This function constructs a `JointDecoder` object, setting up its primary decoder
network based on the provided specifications. The architecture begins with a
dense layer mapping from the latent space and goes through a sequence of middle
layers if specified. After processing the latent space through the primary
decoder, it then maps separately to both its mean (`µ`) and standard deviation
(`σ`).

# Example
```julia
n_input = 28*28
n_latent = 64
decoder_neurons = [128, 256]
decoder_activation = [relu, relu]
latent_activation = [tanh, softplus]
decoder = JointDecoder(
    n_input, n_latent, decoder_neurons, decoder_activation, latent_activation
)
```

# Note
Ensure that the lengths of decoder_neurons and decoder_activation match.
"""
function JointDecoder(
    n_input::Int,
    n_latent::Int,
    decoder_neurons::Vector{<:Int},
    decoder_activation::Vector{<:Function},
    output_activation::Vector{<:Function};
    init::Function=Flux.glorot_uniform
)
    # Check there's enough activation functions for all layers
    if (length(decoder_activation) != length(decoder_neurons))
        error("Each layer needs exactly one activation function")
    end # if

    # Initialize list with decoder layers
    decoder_layers = Array{Flux.Dense}(undef, length(decoder_neurons))

    # Add first layer from latent space to decoder
    decoder_layers[1] = Flux.Dense(
        n_latent => decoder_neurons[1], decoder_activation[1]; init=init
    )

    # Check if there are multiple middle layers
    if length(decoder_neurons) > 1
        # Loop through middle layers if they exist
        for i = 2:length(decoder_neurons)
            decoder_layers[i] = Flux.Dense(
                decoder_neurons[i-1] => decoder_neurons[i],
                decoder_activation[i];
                init=init
            )
        end # for
    end # if

    # Construct the primary decoder
    decoder_chain = Flux.Chain(decoder_layers...)

    # Define layers that map from the last decoder layer to the mean and log
    # standard deviation
    µ_layer = Flux.Dense(
        decoder_neurons[end] => n_input, output_activation[1]; init=init
    )
    σ_layer = Flux.Dense(
        decoder_neurons[end] => n_input, output_activation[2]; init=init
    )

    # Initialize joint decoder
    return JointDecoder(decoder_chain, µ_layer, σ_layer)
end

@doc raw"""
        (decoder::JointDecoder)(z::Float32Array)

Maps the given latent representation `z` through the `JointDecoder` network to
produce both the mean (`µ`) and standard deviation (`σ`).

# Arguments
- `z::Float32Array`: The latent space representation to be decoded. This can be
  a vector, a matrix, or a 3D tensor, where each column (or slice, in the case
  of 3D tensor) represents a separate sample from the latent space of a VAE.

# Returns
- A NamedTuple `(µ=µ, σ=σ,)` where:
    - `µ::Array{Float32}`: The mean representation obtained from the decoder.
    - `σ::Array{Float32}`: The standard deviation representation obtained from
      the decoder.

# Description
This function processes the latent space representation `z` using the primary
neural network of the `JointDecoder` struct. It then separately maps the output
of this network to the mean and standard deviation using the `µ` and `σ` dense
layers, respectively.

# Example
```julia
decoder = JointDecoder(...)
z = ... # some latent space representation
output = decoder(z)
```

# Note
Ensure that the latent space representation z matches the expected input
dimensionality for the JointDecoder.
"""
function (decoder::JointDecoder)(
    z::Float32Array
)
    # Run input through the primary decoder network
    h = decoder.decoder(z)
    # Map to mean
    µ = decoder.µ(h)
    # Map to standard deviation
    σ = decoder.σ(h)
    return (µ=µ, σ=σ)
end # function

# ==============================================================================
# struct SplitLogDecoder <: AbstractGaussianLogDecoder
# ==============================================================================

@doc raw"""
    SplitLogDecoder <: AbstractGaussianLogDecoder

A specialized decoder structure for VAEs that uses distinct neural networks for
determining the mean (`µ`) and log standard deviation (`logσ`) of the latent
space.

# Fields
- `decoder_µ::Flux.Chain`: A neural network dedicated to processing the latent
  space and mapping it to its mean.
- `decoder_logσ::Flux.Chain`: A neural network dedicated to processing the
  latent space and mapping it to its log standard deviation.

# Description
`SplitLogDecoder` is designed for VAE architectures where separate decoder
networks are preferred for computing the mean and log standard deviation,
ensuring that each has its own distinct set of parameters and transformation
logic.
"""
struct SplitLogDecoder <: AbstractGaussianLogDecoder
    µ::Flux.Chain
    logσ::Flux.Chain
end

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor SplitLogDecoder

@doc raw"""
    SplitLogDecoder(n_input, n_latent, µ_neurons, µ_activation, logσ_neurons, 
                logσ_activation; init=Flux.glorot_uniform)

Constructs and initializes a `SplitLogDecoder` object for variational
autoencoders (VAEs). This function sets up two distinct decoder networks, one
dedicated for determining the mean (`µ`) and the other for the log standard
deviation (`logσ`) of the latent space.

# Arguments
- `n_input::Int`: Dimensionality of the output data (or the data to be
  reconstructed).
- `n_latent::Int`: Dimensionality of the latent space.
- `µ_neurons::Vector{<:Int}`: Vector of layer sizes for the `µ` decoder network,
  not including the input latent layer.
- `µ_activation::Vector{<:Function}`: Activation functions for each `µ` decoder
  layer.
- `logσ_neurons::Vector{<:Int}`: Vector of layer sizes for the `logσ` decoder
  network, not including the input latent layer.
- `logσ_activation::Vector{<:Function}`: Activation functions for each `logσ`
  decoder layer.

# Optional Keyword Arguments
- `init::Function=Flux.glorot_uniform`: Initialization function for the network
  parameters.

# Returns
A `SplitLogDecoder` object with two distinct networks initialized with the
specified architectures and weights.

# Description
This function constructs a `SplitLogDecoder` object, setting up two separate
decoder networks based on the provided specifications. The first network,
dedicated to determining the mean (`µ`), and the second for the log standard
deviation (`logσ`), both begin with a dense layer mapping from the latent space
and go through a sequence of middle layers if specified.

# Example
```julia
n_latent = 64
µ_neurons = [128, 256]
µ_activation = [relu, relu]
logσ_neurons = [128, 256]
logσ_activation = [relu, relu]
decoder = SplitLogDecoder(
    n_latent, µ_neurons, µ_activation, logσ_neurons, logσ_activation
)
```

# Notes
- Ensure that the lengths of µ_neurons with µ_activation and logσ_neurons with
  logσ_activation match respectively.
- If µ_neurons[end] or logσ_neurons[end] do not match n_input, the function
  automatically changes this number to match the right dimensionality
"""
function SplitLogDecoder(
    n_input::Int,
    n_latent::Int,
    µ_neurons::Vector{<:Int},
    µ_activation::Vector{<:Function},
    logσ_neurons::Vector{<:Int},
    logσ_activation::Vector{<:Function};
    init::Function=Flux.glorot_uniform
)
    # Check for matching length between neurons and activations for µ
    if (length(µ_activation) != length(µ_neurons))
        error("Each layer of µ decoder needs exactly one activation function")
    end # if

    # Check for matching length between neurons and activations for logσ
    if (length(logσ_activation) != length(logσ_neurons))
        error("Each layer of logσ decoder needs exactly one activation function")
    end # if

    # Check that final number of neurons matches input dimension
    if µ_neurons[end] ≠ n_input
        println("We changed the last layer number of µ_neurons to match the " *
                "input dimension")
        µ_neurons[end] = n_input
    end # if

    # Check that final number of neurons matches input dimension
    if logσ_neurons[end] ≠ n_input
        println("We changed the last layer number of logσ_neurons to match " *
                "the input dimension")
        logσ_neurons[end] = n_input
    end # if


    # Initialize µ decoder layers
    µ_layers = Array{Flux.Dense}(undef, length(µ_neurons))

    # Add first layer from latent space to decoder
    µ_layers[1] = Flux.Dense(
        n_latent => µ_neurons[1], µ_activation[1]; init=init
    )

    # Loop through rest of the layers
    for i = 2:length(µ_neurons)
        # Add next layer to list
        µ_layers[i] = Flux.Dense(
            µ_neurons[i-1] => µ_neurons[i], µ_activation[i]; init=init
        )
    end # for

    # Initialize µ decoder layers
    logσ_layers = Array{Flux.Dense}(undef, length(logσ_neurons))

    # Add first layer from latent space to decoder
    logσ_layers[1] = Flux.Dense(
        n_latent => logσ_neurons[1], logσ_activation[1]; init=init
    )

    # Loop through rest of the layers
    for i = 2:length(logσ_neurons)
        # Add next layer to list
        logσ_layers[i] = Flux.Dense(
            logσ_neurons[i-1] => logσ_neurons[i], logσ_activation[i]; init=init
        )
    end # for

    # Initialize split decoder
    return SplitLogDecoder(Flux.Chain(µ_layers...), Flux.Chain(logσ_layers...))
end # function

@doc raw"""
        (decoder::SplitLogDecoder)(z::Float32Array)

Maps the given latent representation `z` through the separate networks of the
`SplitLogDecoder` to produce both the mean (`µ`) and log standard deviation
(`logσ`).

# Arguments
- `z::Float32Array`: The latent space representation to be decoded. This can be
  a vector, a matrix, or a 3D tensor, where each column (or slice, in the case
  of 3D tensor) represents a separate sample from the latent space of a VAE.

# Returns
- A NamedTuple `(µ=µ, logσ=logσ,)` where:
    - `µ::Array{Float32}`: The mean representation obtained using the dedicated
      `decoder_µ` network.
    - `logσ::Array{Float32}`: The log standard deviation representation obtained
      using the dedicated `decoder_logσ` network.

# Description
This function processes the latent space representation `z` through two distinct
neural networks within the `SplitLogDecoder` struct. The `decoder_µ` network is
used to produce the mean representation, while the `decoder_logσ` network is
utilized for the log standard deviation.

# Example
```julia
decoder = SplitLogDecoder(...)
z = ... # some latent space representation
output = decoder(z))
```

# Note
Ensure that the latent space representation z matches the expected input
dimensionality for both networks in the SplitLogDecoder.
"""
function (decoder::SplitLogDecoder)(
    z::Float32Array
)
    # Map through the decoder dedicated to the mean
    µ = decoder.µ(z)
    # Map through the decoder dedicated to the log standard deviation
    logσ = decoder.logσ(z)
    return (µ=µ, logσ=logσ)
end # function

# ==============================================================================
# struct SplitDecoder <: AbstractGaussianLinearDecoder
# ==============================================================================

@doc raw"""
    SplitDecoder <: AbstractGaussianLinearDecoder

A specialized decoder structure for VAEs that uses distinct neural networks for
determining the mean (`µ`) and standard deviation (`logσ`) of the latent space.

# Fields
- `decoder_µ::Flux.Chain`: A neural network dedicated to processing the latent
  space and mapping it to its mean.
- `decoder_σ::Flux.Chain`: A neural network dedicated to processing the latent
  space and mapping it to its standard deviation.

# Description
`SplitDecoder` is designed for VAE architectures where separate decoder
networks are preferred for computing the mean and log standard deviation,
ensuring that each has its own distinct set of parameters and transformation
logic.
"""
struct SplitDecoder <: AbstractGaussianLinearDecoder
    µ::Flux.Chain
    σ::Flux.Chain
end

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor SplitDecoder

@doc raw"""
    SplitDecoder(n_input, n_latent, µ_neurons, µ_activation, logσ_neurons, 
                logσ_activation; init=Flux.glorot_uniform)

Constructs and initializes a `SplitDecoder` object for variational autoencoders
(VAEs). This function sets up two distinct decoder networks, one dedicated for
determining the mean (`µ`) and the other for the standard deviation (`σ`) of the
latent space.

# Arguments
- `n_input::Int`: Dimensionality of the output data (or the data to be
  reconstructed).
- `n_latent::Int`: Dimensionality of the latent space.
- `µ_neurons::Vector{<:Int}`: Vector of layer sizes for the `µ` decoder network,
  not including the input latent layer.
- `µ_activation::Vector{<:Function}`: Activation functions for each `µ` decoder
  layer.
- `σ_neurons::Vector{<:Int}`: Vector of layer sizes for the `σ` decoder network,
  not including the input latent layer.
- `σ_activation::Vector{<:Function}`: Activation functions for each `σ` decoder
  layer.

# Optional Keyword Arguments
- `init::Function=Flux.glorot_uniform`: Initialization function for the network
  parameters.

# Returns
A `SplitDecoder` object with two distinct networks initialized with the
specified architectures and weights.

# Description
This function constructs a `SplitDecoder` object, setting up two separate
decoder networks based on the provided specifications. The first network,
dedicated to determining the mean (`µ`), and the second for the standard
deviation (`σ`), both begin with a dense layer mapping from the latent space and
go through a sequence of middle layers if specified.

# Example
```julia
n_latent = 64
µ_neurons = [128, 256]
µ_activation = [relu, relu]
σ_neurons = [128, 256]
σ_activation = [relu, relu]
decoder = SplitDecoder(
    n_latent, µ_neurons, µ_activation, σ_neurons, σ_activation
)
```

# Notes
- Ensure that the lengths of µ_neurons with µ_activation and σ_neurons with
  σ_activation match respectively.
- If µ_neurons[end] or σ_neurons[end] do not match n_input, the function
  automatically changes this number to match the right dimensionality
- Ensure that σ_neurons[end] maps to a **positive** value. Activation functions
  such as `softplus` are needed to guarantee the positivity of the standard
  deviation.
"""
function SplitDecoder(
    n_input::Int,
    n_latent::Int,
    µ_neurons::Vector{<:Int},
    µ_activation::Vector{<:Function},
    σ_neurons::Vector{<:Int},
    σ_activation::Vector{<:Function};
    init::Function=Flux.glorot_uniform
)
    # Check for matching length between neurons and activations for µ
    if (length(µ_activation) != length(µ_neurons))
        error("Each layer of µ decoder needs exactly one activation function")
    end # if

    # Check for matching length between neurons and activations for logσ
    if (length(σ_activation) != length(σ_neurons))
        error(
            "Each layer of logσ decoder needs exactly one activation function"
        )
    end # if

    # Check that final number of neurons matches input dimension
    if µ_neurons[end] ≠ n_input
        println("We changed the last layer number of µ_neurons to match the " *
                "input dimension")
        µ_neurons[end] = n_input
    end # if

    # Check that final number of neurons matches input dimension
    if σ_neurons[end] ≠ n_input
        println("We changed the last layer number of σ_neurons to match " *
                "the input dimension")
        σ_neurons[end] = n_input
    end # if


    # Initialize µ decoder layers
    µ_layers = Array{Flux.Dense}(undef, length(µ_neurons))

    # Add first layer from latent space to decoder
    µ_layers[1] = Flux.Dense(
        n_latent => µ_neurons[1], µ_activation[1]; init=init
    )

    # Loop through rest of the layers
    for i = 2:length(µ_neurons)
        # Add next layer to list
        µ_layers[i] = Flux.Dense(
            µ_neurons[i-1] => µ_neurons[i], µ_activation[i]; init=init
        )
    end # for

    # Initialize σ decoder layers
    σ_layers = Array{Flux.Dense}(undef, length(σ_neurons))

    # Add first layer from latent space to decoder
    σ_layers[1] = Flux.Dense(
        n_latent => σ_neurons[1], σ_activation[1]; init=init
    )

    # Loop through rest of the layers
    for i = 2:length(σ_neurons)
        # Add next layer to list
        σ_layers[i] = Flux.Dense(
            σ_neurons[i-1] => σ_neurons[i], σ_activation[i]; init=init
        )
    end # for

    # Initialize split decoder
    return SplitDecoder(Flux.Chain(µ_layers...), Flux.Chain(σ_layers...))
end # function

@doc raw"""
        (decoder::SplitDecoder)(z::Float32Array)

Maps the given latent representation `z` through the separate networks of the
`SplitDecoder` to produce both the mean (`µ`) and standard deviation (`σ`).

# Arguments
- `z::Float32Array`: The latent space representation to be decoded. This can be
  a vector, a matrix, or a 3D tensor, where each column (or slice, in the case
  of 3D tensor) represents a separate sample from the latent space of a VAE.

# Returns
- A NamedTuple `(µ=µ, σ=σ,)` where:
    - `µ::Array{Float32}`: The mean representation obtained using the dedicated
      `decoder_µ` network.
    - `σ::Array{Float32}`: The standard deviation representation obtained using
      the dedicated `decoder_σ` network.

# Description
This function processes the latent space representation `z` through two distinct
neural networks within the `SplitDecoder` struct. The `decoder_µ` network is
used to produce the mean representation, while the `decoder_σ` network is
utilized for the standard deviation.

# Example
```julia
decoder = SplitDecoder(...)
z = ... # some latent space representation
output = decoder(z)
```

# Note
Ensure that the latent space representation z matches the expected input
dimensionality for both networks in the SplitDecoder.
"""
function (decoder::SplitDecoder)(
    z::Float32Array
)
    # Map through the decoder dedicated to the mean
    µ = decoder.µ(z)
    # Map through the decoder dedicated to the standard deviation
    σ = decoder.σ(z)
    return (µ=µ, σ=σ)
end # function

# ==============================================================================
# struct BernoulliDecoder <: AbstractVariationalDecoder
# ==============================================================================

@doc raw"""
        BernoulliDecoder <: AbstractVariationalDecoder

A decoder structure for variational autoencoders (VAEs) that models the output
data as a Bernoulli distribution. This is typically used when the outputs of the
decoder are probabilities.

# Fields
- `decoder::Flux.Chain`: The primary neural network used to process the latent
    space and map it to the output (or reconstructed) space.

# Description
`BernoulliDecoder` represents a VAE decoder that models the output data as a
Bernoulli distribution. It's commonly used when the outputs of the decoder are
probabilities, such as in a binary classification task or when modeling binary
data. Unlike a Gaussian decoder, there's no need for separate paths or
operations on the mean or log standard deviation.

# Note
Ensure the last layer of the decoder outputs a value between 0 and 1, as this is
required for a Bernoulli distribution.
"""
struct BernoulliDecoder <: AbstractVariationalDecoder
    decoder::Flux.Chain
end # struct

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor BernoulliDecoder

# ------------------------------------------------------------------------------

@doc raw"""
        BernoulliDecoder(n_input, n_latent, decoder_neurons, decoder_activation, 
                                output_activation; init=Flux.glorot_uniform)

Constructs and initializes a `BernoulliDecoder` object designed for variational
autoencoders (VAEs). This function sets up a decoder network that maps from a
latent space to an output space.

# Arguments
- `n_input::Int`: Dimensionality of the output data (or the data to be
    reconstructed).
- `n_latent::Int`: Dimensionality of the latent space.
- `decoder_neurons::Vector{<:Int}`: Vector of layer sizes for the decoder
    network, not including the input latent layer and the final output layer.
- `decoder_activation::Vector{<:Function}`: Activation functions for each
    decoder layer, not including the final output layer.
- `output_activation::Function`: Activation function for the final output layer.

## Optional Keyword Arguments
- `init::Function=Flux.glorot_uniform`: Initialization function for the network
    parameters.

# Returns
A `BernoulliDecoder` object with the specified architecture and initialized
weights.

# Description
This function constructs a `BernoulliDecoder` object, setting up its decoder
network based on the provided specifications. The architecture begins with a
dense layer mapping from the latent space, goes through a sequence of middle
layers if specified, and finally maps to the output space.

The function ensures that there are appropriate activation functions provided
for each layer in the `decoder_neurons` and checks for potential mismatches in
length.

# Example
```julia
n_input = 28*28
n_latent = 64
decoder_neurons = [128, 256]
decoder_activation = [relu, relu]
output_activation = sigmoid
decoder = BernoulliDecoder(
    n_input, 
    n_latent, 
    decoder_neurons, 
    decoder_activation, 
    output_activation
)
```

# Note 
Ensure that the lengths of decoder_neurons and decoder_activation match,
excluding the output layer. Also, the output activation function should return
values between 0 and 1, as the decoder models the output data as a Bernoulli
distribution. 
"""
function BernoulliDecoder(
    n_input::Int,
    n_latent::Int,
    decoder_neurons::Vector{<:Int},
    decoder_activation::Vector{<:Function},
    output_activation::Function;
    init::Function=Flux.glorot_uniform
)
    # Check there's enough activation functions for all layers
    if (length(decoder_activation) != length(decoder_neurons))
        error("Each layer needs exactly one activation function")
    end # if

    # Initialize list with decoder layers
    decoder = Array{Flux.Dense}(undef, length(decoder_neurons) + 1)

    # Add first layer from latent space to decoder
    decoder[1] = Flux.Dense(
        n_latent => decoder_neurons[1], decoder_activation[1]; init=init
    )

    # Add last layer from decoder to output
    decoder[end] = Flux.Dense(
        decoder_neurons[end] => n_input, output_activation; init=init
    )

    # Check if there are multiple middle layers
    if length(decoder_neurons) > 1
        # Loop through middle layers
        for i = 2:length(decoder_neurons)
            # Set middle layers of decoder
            decoder[i] = Flux.Dense(
                decoder_neurons[i-1] => decoder_neurons[i],
                decoder_activation[i];
                init=init
            )
        end # for
    end # if

    # Check if output activation function returns values between 0 and 1 for
    # inputs between -1 and 1
    for x in -5:0.1:5
        if output_activation(x) < 0 || output_activation(x) > 1
            @warn "The output activation function should return values between 0 and 1 for a BernoulliDecoder"
            break
        end
    end

    # Initialize Bernoulli decoder
    return BernoulliDecoder(Flux.Chain(decoder...))
end # function

# ------------------------------------------------------------------------------

@doc raw"""
        (decoder::BernoulliDecoder)(z::AbstractVecOrMat{Float32})

Maps the given latent representation `z` through the `BernoulliDecoder` network
to reconstruct the original input.

# Arguments
- `z::AbstractVecOrMat{Float32}`: The latent space representation to be decoded.
    This can be a vector or a matrix, where each column represents a separate
    sample from the latent space of a VAE.

# Returns
- A NamedTuple `(p=p,)` where `p` is an array representing the output of the
    decoder, which should resemble the original input to the VAE (post encoding
    and sampling from the latent space).

# Description
This function processes the latent space representation `z` using the neural
network defined in the `BernoulliDecoder` struct. The aim is to decode or
reconstruct the original input from this representation.

# Note
Ensure that the latent space representation z matches the expected input
dimensionality for the BernoulliDecoder.
"""
function (decoder::BernoulliDecoder)(
    z::AbstractVecOrMat{Float32}
)
    # Run input to decoder network
    return (p=decoder.decoder(z),)
end # function