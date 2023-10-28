# Import ML libraries
import Flux
import Zygote

# Import basic math
import Random
import StatsBase
import Distributions

# Import GPU library
import CUDA

##

# Import Abstract Types

using ..AutoEncode: AbstractAutoEncoder, AbstractVariationalAutoEncoder,
    AbstractEncoder, AbstractDecoder, AbstractVariationalEncoder,
    AbstractVariationalDecoder

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Kingma, D. P. & Welling, M. Auto-Encoding Variational Bayes. Preprint at
#    http://arxiv.org/abs/1312.6114 (2014).
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ==============================================================================

@doc raw"""
`struct JointEncoder`

Default encoder function for variational autoencoders where the same `encoder`
network is used to map to the latent space mean `µ` and log standard deviation
`logσ`.

# Fields
- `encoder::Flux.Chain`: The primary neural network used to process input data
  and map it into a latent space representation.
- `µ::Flux.Dense`: A dense layer mapping from the output of the `encoder` to the
  mean of the latent space.
- `logσ::Flux.Dense`: A dense layer mapping from the output of the `encoder` to
  the log standard deviation of the latent space.

# Example
```julia
enc = JointEncoder(
    Flux.Chain(Dense(784, 400, relu)), Flux.Dense(400, 20), Flux.Dense(400, 20)
)
"""
mutable struct JointEncoder <: AbstractVariationalEncoder
    encoder::Flux.Chain
    µ::Flux.Dense
    logσ::Flux.Dense
end # struct

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor JointEncoder

@doc raw"""
    JointEncoder(n_input, n_latent, encoder_neurons, encoder_activation, 
                 latent_activation; init=Flux.glorot_uniform)

Construct and initialize a `JointEncoder` struct that defines an encoder network
for a variational autoencoder.

# Arguments
- `n_input::Int`: The dimensionality of the input data.
- `n_latent::Int`: The dimensionality of the latent space.
- `encoder_neurons::Vector{<:Int}`: A vector specifying the number of neurons in
  each layer of the encoder network.
- `encoder_activation::Vector{<:Function}`: Activation functions corresponding
  to each layer in the `encoder_neurons`.
- `latent_activation::Function`: Activation function for the latent space layers
  (both µ and logσ).

## Optional Keyword Arguments
- `init::Function=Flux.glorot_uniform`: The initialization function used for the
  neural network weights.

# Returns
- A `JointEncoder` struct initialized based on the provided arguments.

# Examples
```julia
encoder = JointEncoder(784, 20, [400], [relu], tanh)
```

# Notes
The length of encoder_neurons should match the length of encoder_activation,
ensuring that each layer in the encoder has a corresponding activation function.
"""
function JointEncoder(
    n_input::Int,
    n_latent::Int,
    encoder_neurons::Vector{<:Int},
    encoder_activation::Vector{<:Function},
    latent_activation::Function;
    init::Function=Flux.glorot_uniform
)
    # Check there's enough activation functions for all layers
    if (length(encoder_activation) != length(encoder_neurons))
        error("Each layer needs exactly one activation function")
    end # if

    # Initialize list with encoder layers
    encoder_layers = Array{Flux.Dense}(undef, length(encoder_neurons))

    # Loop through layers   
    for i in eachindex(encoder_neurons)
        # Check if it is the first layer
        if i == 1
            # Set first layer from input to encoder with activation
            encoder_layers[i] = Flux.Dense(
                n_input => encoder_neurons[i], encoder_activation[i]; init=init
            )
        else
            # Set middle layers from input to encoder with activation
            encoder_layers[i] = Flux.Dense(
                encoder_neurons[i-1] => encoder_neurons[i],
                encoder_activation[i];
                init=init
            )
        end # if
    end # for

    # Define layer that maps from encoder to latent space with activation
    µ_layer = Flux.Dense(
        encoder_neurons[end] => n_latent, latent_activation; init=init
    )
    logσ_layer = Flux.Dense(
        encoder_neurons[end] => n_latent, latent_activation; init=init
    )

    # Initialize decoder
    return JointEncoder(Flux.Chain(encoder_layers...), µ_layer, logσ_layer)
end # function

@doc raw"""
    (encoder::JointEncoder)(x)

Forward propagate the input `x` through the `JointEncoder` to obtain the mean
(`mu`) and log standard deviation (`logσ`) of the latent space.

# Arguments
- `x::Array{Float32}`: Input data to be encoded.

# Returns
- `mu`: Mean of the latent space after passing the input through the encoder and
  subsequently through the `µ` layer.
- `logσ`: Log standard deviation of the latent space after passing the input
  through the encoder and subsequently through the `logσ` layer.

# Description
This method allows for a direct call on an instance of `JointEncoder` with the
input data `x`. It first runs the input through the encoder network, then maps
the output of the last encoder layer to both the mean and log standard deviation
of the latent space.

# Example
```julia
je = JointEncoder(...)
mu, logσ = je(some_input)
```

# Note
Ensure that the input x matches the expected dimensionality of the encoder's
input layer.
"""
function (encoder::JointEncoder)(x::AbstractVecOrMat{Float32})
    # Run input to encoder network
    h = encoder.encoder(x)
    # Map from last encoder layer to latent space mean
    µ = encoder.µ(h)
    # Map from last encoder layer to latent space log standard deviation
    logσ = encoder.logσ(h)

    # Return description of latent variables
    return µ, logσ
end # function

# ==============================================================================

@doc raw"""
    reparameterize(µ, logσ; prior=Distributions.Normal{Float32}(0.0f0, 1.0f0), n_samples=1)

Reparameterize the latent space using the given mean (`µ`) and log standard
deviation (`logσ`), employing the reparameterization trick. This function helps
in sampling from the latent space in variational autoencoders (or similar
models) while keeping the gradient flow intact.

# Arguments
- `µ::AbstractVecOrMat{Float32}`: The mean of the latent space. If it is a
  vector, it represents the mean for a single data point. If it is a matrix,
  each column corresponds to the mean for a specific data point, and each row
  corresponds to a dimension of the latent space.
- `logσ::AbstractVecOrMat{Float32}`: The log standard deviation of the latent
  space. Like `µ`, if it's a vector, it represents the log standard deviation
  for a single data point. If a matrix, each column corresponds to the log
  standard deviation for a specific data point.


# Optional Keyword Arguments
- `prior::Distributions.Sampleable`: The prior distribution for the latent
  space. By default, this is a standard normal distribution.
- `n_samples::Int=1`: The number of samples to draw using the reparametrization
  trick.

# Returns
An array containing `n_samples` samples from the reparameterized latent space,
obtained by applying the reparameterization trick on the provided mean and log
standard deviation, using the specified prior distribution.

# Description
This function employs the reparameterization trick to sample from the latent
space without breaking the gradient flow. The trick involves expressing the
random variable as a deterministic variable transformed by a standard random
variable, allowing for efficient backpropagation through stochastic nodes.

If the provided `prior` is a univariate distribution, the function samples using
the dimensions of `µ`. For multivariate distributions, it assumes a single
sample should be generated and broadcasted accordingly.

# Example
```julia
µ = Float32[0.5, 0.2]
logσ = Float32[-0.1, -0.2]
sampled_point = reparameterize(µ, logσ)
```
# Notes
Ensure that the dimensions of µ and logσ match, and that the chosen prior
distribution is consistent with the expectations of the latent space.

# Citation
Kingma, D. P. & Welling, M. Auto-Encoding Variational Bayes. Preprint at
http://arxiv.org/abs/1312.6114 (2014).
"""
function reparameterize(
    µ::T,
    logσ::T;
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
    n_samples::Int=1
) where {T<:AbstractVecOrMat{Float32}}

    # Sample result depending on type of prior distribution
    result = if typeof(prior) <: Distributions.UnivariateDistribution
        # Sample n_samples random latent variable point estimates given the mean
        # and standard deviation
        µ .+ Random.rand(prior, size(µ)..., n_samples) .* exp.(logσ)
    elseif typeof(prior) <: Distributions.MultivariateDistribution
        # Sample n_samples random latent variable point estimates given the mean
        # and standard deviation
        µ .+ Random.rand(prior, n_samples) .* exp.(logσ)
    end # if

    # Remove dimensions of size 1 based on type of T and n_samples = 1
    if n_samples == 1
        if T <: AbstractVector
            # Drop second dimension when input is vector and n_samples = 1
            return dropdims(result, dims=2)
        elseif T <: AbstractMatrix
            # Drop third dimension when input is matrix and n_samples = 1
            return dropdims(result, dims=3)
        end # if
    end # if

    return result
end # function

# ==============================================================================

@doc raw"""
    SimpleDecoder <: AbstractVariationalDecoder

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
mutable struct SimpleDecoder <: AbstractVariationalDecoder
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
    (decoder::SimpleDecoder)(
        z::Union{AbstractVector{Float32},AbstractMatrix{Float32},Array{Float32,3}}
    )
Maps the given latent representation `z` through the `SimpleDecoder` network.

# Arguments
- `z::Union{AbstractVector{Float32}, AbstractMatrix{Float32}, Array{Float32,
  3}}`: The latent space representation to be decoded. This can be a vector (1D
  tensor), a matrix (2D tensor), or a 3D tensor, where each column (or slice, in
  the case of 3D tensor) represents a separate sample from the latent space of a
  VAE.

# Returns
An array representing the output of the decoder, which should resemble the
original input to the VAE (post encoding and sampling from the latent space).

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
    z::Union{AbstractVector{Float32},AbstractMatrix{Float32},Array{Float32,3}}
)
    # Run input to decoder network
    return decoder.decoder(z)
end # function

# ==============================================================================

@doc raw"""
    JointDecoder <: AbstractVariationalDecoder

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
`JointDecoder` is tailored for VAE architectures where the same decoder network
is used initially, and then splits into two separate paths for determining both
the mean and log standard deviation of the latent space.
"""
mutable struct JointDecoder <: AbstractVariationalDecoder
    decoder::Flux.Chain
    µ::Flux.Dense
    logσ::Flux.Dense
end

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor JointDecoder

@doc raw"""
    JointDecoder(n_input, n_latent, decoder_neurons, decoder_activation, 
                latent_activation; init=Flux.glorot_uniform)

Constructs and initializes a `JointDecoder` object for variational autoencoders
(VAEs). This function sets up a decoder network that first processes the latent
space and then maps it separately to both its mean (`µ`) and log standard
deviation (`logσ`).

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
decoder, it then maps separately to both its mean (`µ`) and log standard
deviation (`logσ`).

# Example
```julia
n_input = 28*28
n_latent = 64
decoder_neurons = [128, 256]
decoder_activation = [relu, relu]
latent_activation = tanh
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
    (decoder::JointDecoder)(
        z::Union{AbstractVector{Float32},AbstractMatrix{Float32},Array{Float32,3}}
    )

Maps the given latent representation `z` through the `JointDecoder` network to
produce both the mean (`µ`) and log standard deviation (`logσ`).

# Arguments
- `z::Union{AbstractVector{Float32}, AbstractMatrix{Float32}, Array{Float32,
  3}}`: The latent space representation to be decoded. This can be a vector (1D
  tensor), a matrix (2D tensor), or a 3D tensor, where each column (or slice, in
  the case of 3D tensor) represents a separate sample from the latent space of a
  VAE.

# Returns
- `µ::Array{Float32}`: The mean representation obtained from the decoder.
- `logσ::Array{Float32}`: The log standard deviation representation obtained
  from the decoder.

# Description
This function processes the latent space representation `z` using the primary
neural network of the `JointDecoder` struct. It then separately maps the output
of this network to the mean and log standard deviation using the `µ` and `logσ`
dense layers, respectively.

# Example
```julia
decoder = JointDecoder(...)
z = ... # some latent space representation
µ, logσ = decoder(z)
```

# Note
Ensure that the latent space representation z matches the expected input
dimensionality for the JointDecoder.
"""
function (decoder::JointDecoder)(
    z::Union{AbstractVector{Float32},AbstractMatrix{Float32},Array{Float32,3}}
)
    # Run input through the primary decoder network
    h = decoder.decoder(z)
    # Map to mean
    µ = decoder.µ(h)
    # Map to log standard deviation
    logσ = decoder.logσ(h)
    return µ, logσ
end # function

# ==============================================================================

@doc raw"""
    SplitDecoder <: AbstractVariationalDecoder

A specialized decoder structure for VAEs that uses distinct neural networks for
determining the mean (`µ`) and log standard deviation (`logσ`) of the latent
space.

# Fields
- `decoder_µ::Flux.Chain`: A neural network dedicated to processing the latent
  space and mapping it to its mean.
- `decoder_logσ::Flux.Chain`: A neural network dedicated to processing the
  latent space and mapping it to its log standard deviation.

# Description
`SplitDecoder` is designed for VAE architectures where separate decoder networks
are preferred for computing the mean and log standard deviation, ensuring that
each has its own distinct set of parameters and transformation logic.
"""
mutable struct SplitDecoder <: AbstractVariationalDecoder
    µ::Flux.Chain
    logσ::Flux.Chain
end

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor SplitDecoder

@doc raw"""
    SplitDecoder(n_input, n_latent, µ_neurons, µ_activation, logσ_neurons, 
                logσ_activation; init=Flux.glorot_uniform)

Constructs and initializes a `SplitDecoder` object for variational autoencoders
(VAEs). This function sets up two distinct decoder networks, one dedicated for
determining the mean (`µ`) and the other for the log standard deviation (`logσ`)
of the latent space.

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
A `SplitDecoder` object with two distinct networks initialized with the
specified architectures and weights.

# Description
This function constructs a `SplitDecoder` object, setting up two separate
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
decoder = SplitDecoder(
    n_latent, µ_neurons, µ_activation, logσ_neurons, logσ_activation
)
```

# Notes
- Ensure that the lengths of µ_neurons with µ_activation and logσ_neurons with
  logσ_activation match respectively.
- If µ_neurons[end] or logσ_neurons[end] do not match n_input, the function
  automatically changes this number to match the right dimensionality
"""
function SplitDecoder(
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
        println("We changed the last layer number of µ_neurons to match the input dimension")
        µ_neurons[end] = n_input
    end # if

    # Check that final number of neurons matches input dimension
    if logσ_neurons[end] ≠ n_input
        println("We changed the last layer number of logσ_neurons to match the input dimension")
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
    return SplitDecoder(Flux.Chain(µ_layers...), Flux.Chain(logσ_layers...))
end # function

@doc raw"""
    (decoder::SplitDecoder)(
        z::Union{AbstractVector{Float32},AbstractMatrix{Float32},Array{Float32,3}}
    )

Maps the given latent representation `z` through the separate networks of the
`SplitDecoder` to produce both the mean (`µ`) and log standard deviation
(`logσ`).

# Arguments
- `z::Union{AbstractVector{Float32}, AbstractMatrix{Float32}, Array{Float32,
  3}}`: The latent space representation to be decoded. This can be a vector (1D
  tensor), a matrix (2D tensor), or a 3D tensor, where each column (or slice, in
  the case of 3D tensor) represents a separate sample from the latent space of a
  VAE.

# Returns
- `µ::Array{Float32}`: The mean representation obtained using the dedicated
  `decoder_µ` network.
- `logσ::Array{Float32}`: The log standard deviation representation obtained
  using the dedicated `decoder_logσ` network.

# Description
This function processes the latent space representation `z` through two distinct
neural networks within the `SplitDecoder` struct. The `decoder_µ` network is
used to produce the mean representation, while the `decoder_logσ` network is
utilized for the log standard deviation.

# Example
```julia
decoder = SplitDecoder(...)
z = ... # some latent space representation
µ, logσ = decoder(z)
```

# Note
Ensure that the latent space representation z matches the expected input
dimensionality for both networks in the SplitDecoder.
"""
function (decoder::SplitDecoder)(
    z::Union{AbstractVector{Float32},AbstractMatrix{Float32},Array{Float32,3}}
)
    # Map through the decoder dedicated to the mean
    µ = decoder.µ(z)
    # Map through the decoder dedicated to the log standard deviation
    logσ = decoder.logσ(z)
    return µ, logσ
end # function

# ==============================================================================

@doc raw"""
`struct VAE{E<:AbstractVariationalEncoder, D<:AbstractVariationalDecoder}`

Variational autoencoder (VAE) model defined for `Flux.jl`

# Fields
- `encoder::E`: Neural network that encodes the input into the latent space. `E`
  is a subtype of `AbstractVariationalEncoder`.
- `decoder::D`: Neural network that decodes the latent representation back to
  the original input space. `D` is a subtype of `AbstractVariationalDecoder`.

A VAE consists of an encoder and decoder network with a bottleneck latent space
in between. The encoder compresses the input into a low-dimensional
probabilistic representation q(z|x). The decoder tries to reconstruct the
original input from a sampled point in the latent space p(x|z). 
"""
mutable struct VAE{E<:AbstractVariationalEncoder,D<:AbstractVariationalDecoder} <: AbstractVariationalAutoEncoder
    encoder::E
    decoder::D
end # struct

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor VAE

@doc raw"""
    (vae::VAE{JointEncoder,SimpleDecoder})(x::AbstractVecOrMat{Float32}; 
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0), 
    latent::Bool=false)

Processes the given input data `x` through a VAE that consists of a
`JointEncoder` and a `SimpleDecoder`.

# Arguments
- `x::AbstractVecOrMat{Float32}`: The data to be decoded. This can be a vector
  or a matrix where each column represents a separate sample from the data
  space.

# Optional Keyword Arguments
- `prior::Distributions.Sampleable`: Specifies the prior distribution to be used
  during the reparametrization trick. Defaults to a standard normal
  distribution.
- `latent::Bool`: If set to `true`, returns the latent variables (mean, log
  standard deviation, and the sampled latent representation) alongside the
  reconstructed data. Defaults to `false`.
- `n_samples::Int=1`: The number of samples to draw using the reparametrization
  trick.

# Returns
- If `latent=false`: `Array{Float32}`, the reconstructed data after processing
  through the encoder, performing the reparametrization trick, and passing
  through the decoder. The last dimension of the array will be of size
  `n_samples`, containing multiple reconstructions based on the number of
  samples from the latent space.
- If `latent=true`: A tuple containing the mean, log standard deviation, an
  array of `n_samples` sampled latent representations, and the reconstructed
  data. The last dimension of the reconstructed data in the tuple will be of
  size `n_samples`.

# Description
The function first encodes the input data `x` using the `JointEncoder` to obtain
the mean and log standard deviation of the latent space representation. It then
uses the reparametrization trick to sample from this latent distribution, which
is then decoded using the `SimpleDecoder` to produce the final reconstructed
data.

# Example
```julia
vae_model = VAE{JointEncoder,SimpleDecoder}(...)
input_data = ... 
reconstructed_data = vae_model(input_data; latent=true)
```

# Note
Ensure that the input data x matches the expected input dimensionality for the
encoder in the VAE.
"""
function (vae::VAE{<:AbstractVariationalEncoder,<:AbstractVariationalDecoder})(
    x::AbstractVecOrMat{Float32},
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0);
    latent::Bool=false,
    n_samples::Int=1
)
    # Run input through encoder to obtain mean and log std
    encoder_µ, encoder_logσ = vae.encoder(x)

    # Run reparametrization trick
    z_sample = reparameterize(
        encoder_µ, encoder_logσ; prior=prior, n_samples=n_samples
    )

    # Check if latent variables should be returned
    if latent
        # Run latent sample through decoder
        return encoder_µ, encoder_logσ, z_sample, vae.decoder(z_sample)
    else
        # Run latent sample through decoder
        return vae.decoder(z_sample)
    end # if
end # function

# ==============================================================================

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# VAE loss functions
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@doc raw"""
    `loss(vae, x; σ=1.0f0, β=1.0f0, n_samples=1)`

Computes the loss for the variational autoencoder (VAE) by averaging over
`n_samples` latent space samples.

The loss function combines the reconstruction loss with the Kullback-Leibler
(KL) divergence, defined as:

loss = -⟨π(x|z)⟩ + β × Dₖₗ[qᵩ(z|x) || π(z)]

Where:
- π(x|z) is a probabilistic decoder: π(x|z) = N(f(z), σ² I̲̲)) - f(z) is the
function defining the mean of the decoder π(x|z) - qᵩ(z|x) is the approximated
encoder: qᵩ(z|x) = N(g(x), h(x))
- g(x) and h(x) define the mean and covariance of the encoder respectively.

# Arguments
- `vae::VAE{<:AbstractVariationalEncoder,SimpleDecoder}`: A VAE model with
  encoder and decoder networks.
- `x::AbstractVector{Float32}`: Input vector. For batch processing or evaluating
  the entire dataset, use: `sum(loss.(Ref(vae), eachcol(x)))`.

# Optional Keyword Arguments
- `σ::Float32=1.0f0`: Standard deviation for the probabilistic decoder π(x|z).
- `β::Float32=1.0f0`: Weighting factor for the KL-divergence term, used for
  annealing.
- `n_samples::Int=1`: The number of samples to draw from the latent space when
  computing the loss.

# Returns
- `Float32`: The computed average loss value for the input `x` and its
  reconstructed counterparts over `n_samples` samples.

# Note
Ensure that the input data `x` matches the expected input dimensionality for the
encoder in the VAE.
"""
function loss(
    vae::VAE{<:AbstractVariationalEncoder,SimpleDecoder},
    x::AbstractVector{Float32};
    σ::Float32=1.0f0,
    β::Float32=1.0f0,
    n_samples::Int=1
)
    # Forward Pass (run input through reconstruct function with n_samples)
    µ, logσ, z_samples, x̂ = vae(x; latent=true, n_samples=n_samples)

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder averaged over all samples
    logπ_x_z = -1 / (2 * σ^2 * n_samples) * sum((x .- x̂) .^ 2)

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. exp(2.0f0 * logσ) + μ^2 - 1.0f0 - 2.0f0 * logσ
    )

    # Compute average loss function
    return -logπ_x_z + β * kl_qᵩ_π
end # function


@doc raw"""
    `loss(vae, x_in, x_out; σ=1.0f0, β=1.0f0, n_samples=1)`

Computes the loss for the variational autoencoder (VAE).

The loss function combines the reconstruction loss with the Kullback-Leibler
(KL) divergence, defined as:

loss = -⟨π(x_out|z)⟩ + β × Dₖₗ[qᵩ(z|x_in) || π(z)]

Where:
- π(x_out|z) is a probabilistic decoder: π(x_out|z) = N(f(z), σ² I̲̲)) - f(z) is
the function defining the mean of the decoder π(x_out|z) - qᵩ(z|x_in) is the
approximated encoder: qᵩ(z|x_in) = N(g(x_in), h(x_in))
- g(x_in) and h(x_in) define the mean and covariance of the encoder
  respectively.

# Arguments
- `vae::VAE{<:AbstractVariationalEncoder,SimpleDecoder}`: A VAE model with
  encoder and decoder networks.
- `x_in::AbstractVector{Float32}`: Input vector to the VAE encoder.
- `x_out::AbstractVector{Float32}`: Target vector to compute the reconstruction
  error.

# Optional Keyword Arguments
- `σ::Float32=1.0f0`: Standard deviation for the probabilistic decoder
  π(x_out|z).
- `β::Float32=1.0f0`: Weighting factor for the KL-divergence term, used for
  annealing.
- `n_samples::Int=1`: The number of samples to draw from the latent space when
  computing the loss.

# Returns
- `Float32`: The computed loss value between the input `x_out` and its
  reconstructed counterpart from `x_in`.

# Note
Ensure that the input data `x_in` matches the expected input dimensionality for
the encoder in the VAE.
"""
function loss(
    vae::VAE{<:AbstractVariationalEncoder,SimpleDecoder},
    x_in::AbstractVector{Float32},
    x_out::AbstractVector{Float32};
    σ::Float32=1.0f0,
    β::Float32=1.0f0,
    n_samples::Int=1
)
    # 1. Forward Pass (run input x_in through reconstruct function)
    µ, logσ, _, x̂ = vae(x_in; latent=true, n_samples=n_samples)

    # Compute ⟨log π(x_out|z)⟩ for a Gaussian decoder
    logπ_x_z = -1 / (2 * σ^2 * n_samples) * sum((x_out .- x̂) .^ 2)

    # Compute Kullback-Leibler divergence between approximated decoder
    # qᵩ(z|x_in) and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. exp(2.0f0 * logσ) + μ^2 - 1.0f0 - 2.0f0 * logσ
    )

    # Compute loss function
    return -logπ_x_z + β * kl_qᵩ_π
end # function

# ==============================================================================

@doc raw"""
    loss(vae::VAE{<:AbstractVariationalEncoder,T}, 
        x::AbstractVector{Float32}; β::Float32=1.0f0, n_samples=1)

Calculate the loss for a variational autoencoder (VAE) by combining the
reconstruction loss and the Kullback-Leibler (KL) divergence, averaged over
`n_samples` latent space samples.

The VAE loss is given by: loss = -⟨logπ(x|z)⟩ + β × Dₖₗ[qᵨ(z|x) ‖ π(z)]

Where:
- π(x|z) is the probabilistic decoder represented by a Gaussian distribution:
  π(x|z) = N(µ(z), exp(logσ(z))²I).
  - µ(z) is the mean derived from the decoder.
  - logσ(z) is the log standard deviation, also derived from the decoder.
- qᵨ(z|x) is the approximated encoder with Gaussian distribution: qᵨ(z|x) =
  N(g(x), h(x)).
  - g(x) and h(x) respectively define the mean and covariance of the encoder.

# Arguments
- `vae::VAE{<:AbstractVariationalEncoder,JointDecoder}`: A VAE model.
- `x::AbstractVector{Float32}`: Input vector.

# Optional Keyword Arguments
- `β::Float32=1.0f0`: Weighting factor for the KL-divergence term, adjusting the
  balance between reconstruction and regularization.
- `n_samples::Int=1`: The number of samples to draw from the latent space when
  computing the loss.

# Returns
- `loss::Float32`: The computed average VAE loss value for the given input `x`
  over `n_samples` samples.

# Notes
- Ensure that the dimensionality of the input data `x` aligns with the encoder's
expected input in the VAE.
- For batch processing or evaluating an entire dataset, use:
`sum(loss.(Ref(vae), eachcol(x)))`.
"""
function loss(
    vae::VAE{<:AbstractVariationalEncoder,T},
    x::AbstractVector{Float32};
    β::Float32=1.0f0,
    n_samples::Int=1
) where {T<:Union{JointDecoder,SplitDecoder}}
    # Run input through reconstruct function with n_samples
    encoder_µ, encoder_logσ, z_samples, (decoder_µ, decoder_logσ) = vae(
        x; latent=true, n_samples=n_samples
    )

    # Compute average reconstruction loss for a Gaussian decoder over all
    # samples
    logπ_x_z = -1 / (2.0f0 * n_samples) * length(decoder_µ) * log(2 * π) -
               1 / n_samples * sum(decoder_logσ) -
               1 / (2.0f0 * n_samples) * sum((x .- decoder_µ) .^ 2 ./
                                             exp.(2 * decoder_logσ))

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. (exp(2.0f0 * encoder_logσ) + encoder_μ^2 - 1.0f0) -
           2.0f0 * encoder_logσ
    )

    # Compute average total loss
    return -logπ_x_z + β * kl_qᵩ_π
end # function

@doc raw"""
    loss(vae::VAE{<:AbstractVariationalEncoder,T}, 
         x_in::AbstractVector{Float32}, x_out::AbstractVector{Float32};
         β::Float32=1.0f0, n_samples=1)

Calculate the loss for a variational autoencoder (VAE) by combining the
reconstruction loss and the Kullback-Leibler (KL) divergence.

The VAE loss is given by: loss = -⟨logπ(x_out|z)⟩ + β × Dₖₗ[qᵨ(z|x_in) ‖ π(z)]

Where:
- π(x_out|z) is the probabilistic decoder represented by a Gaussian
  distribution: π(x_out|z) = N(µ(z), exp(logσ(z))²I).
  - µ(z) is the mean derived from the decoder.
  - logσ(z) is the log standard deviation, also derived from the decoder.
- qᵨ(z|x_in) is the approximated encoder with Gaussian distribution: qᵨ(z|x_in)
  =
  N(g(x_in), h(x_in)).
  - g(x_in) and h(x_in) respectively define the mean and covariance of the
    encoder.

# Arguments
- `vae::VAE{<:AbstractVariationalEncoder,JointDecoder}`: A VAE model.
- `x_in::AbstractVector{Float32}`: Input vector to the VAE encoder.
- `x_out::AbstractVector{Float32}`: Target vector to compute the reconstruction
  error.

# Optional Keyword Arguments
- `β::Float32=1.0f0`: Weighting factor for the KL-divergence term, adjusting the
  balance between reconstruction and regularization.

# Returns
- `loss::Float32`: The computed VAE loss value between `x_out` and its
  reconstructed counterpart from `x_in`.
- `n_samples::Int=1`: The number of samples to draw from the latent space when
  computing the loss.


# Notes
- Ensure that the dimensionality of the input data `x_in` aligns with the
encoder's expected input in the VAE.
- For batch processing or evaluating an entire dataset, use:
`sum(loss.(Ref(vae), eachcol(x_in), eachcol(x_out)))`.
"""
function loss(
    vae::VAE{<:AbstractVariationalEncoder,T},
    x_in::AbstractVector{Float32},
    x_out::AbstractVector{Float32};
    β::Float32=1.0f0,
    n_samples::Int=1
) where {T<:Union{JointDecoder,SplitDecoder}}
    # Run input x_in through reconstruct function
    encoder_μ, encoder_logσ, z, (decoder_μ, decoder_logσ) = vae(
        x_in; latent=true
    )

    # Compute reconstruction loss for a Gaussian decoder
    logπ_x_z = -1 / (2.0f0 * n_samples) * length(decoder_μ) * log(2 * π) -
               1 / n_samples * sum(decoder_logσ) -
               1 / (2.0f0 * n_samples) * sum((x_out .- decoder_μ) .^ 2 ./
                                             exp.(2 * decoder_logσ))

    # Compute Kullback-Leibler divergence between approximated decoder
    # qᵩ(z|x_in) and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. (exp(2.0f0 * encoder_logσ) + encoder_μ^2 - 1.0f0) -
           2.0f0 * encoder_logσ
    )

    # Compute total loss
    return -logπ_x_z + β * kl_qᵩ_π
end # function

# ==============================================================================

@doc raw"""
    `loss_terms(vae, x; σ=1.0f0, β=1.0f0, logpost=true, kl=true, n_samples=1)`

Computes the individual loss terms for the variational autoencoder (VAE).

The function can compute and return the reconstruction loss `logπ(x|z)` and the
Kullback-Leibler (KL) divergence `Dₖₗ[qᵩ(z|x) || π(z)]` separately.

# Arguments
- `vae::VAE{<:AbstractVariationalEncoder,SimpleDecoder}`: A VAE model with
  encoder and decoder networks.
- `x::AbstractVector{Float32}`: Input vector.

# Optional Keyword Arguments
- `σ::Float32=1.0f0`: Standard deviation for the probabilistic decoder π(x|z).
- `logpost::Bool=true`: If true, computes the `logπ(x|z)` term. 
- `kl::Bool=true`: If true, computes the `Dₖₗ[qᵩ(z|x) || π(z)]` term.
- `n_samples::Int=1`: The number of samples to draw from the latent space when
  computing the loss.

# Returns
- `(logπ_x_z, kl_qᵩ_π)`: Tuple containing the computed loss terms. Returns
  `nothing` for the terms not computed based on the flags.

# Note
Ensure that the input data `x` matches the expected input dimensionality for the
encoder in the VAE.
"""
function loss_terms(
    vae::VAE{<:AbstractVariationalEncoder,SimpleDecoder},
    x::AbstractVector{Float32};
    σ::Float32=1.0f0,
    logpost::Bool=true,
    kl::Bool=true,
    n_samples::Int=1
)
    # Forward Pass (run input through reconstruct function)
    μ, logσ, _, x̂ = vae(x; latent=true, n_samples=n_samples)

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder if logpost is true, otherwise
    # return nothing
    logπ_x_z = logpost ? (
        -1 / (2 * σ^2 * n_samples) * sum((x .- x̂) .^ 2)
    ) : nothing

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z) if kl is true, otherwise return nothing
    kl_qᵩ_π = kl ? (
        sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)
    ) : nothing

    return logπ_x_z, kl_qᵩ_π
end # function

@doc raw"""
    `loss_terms(vae, x_in, x_out; σ=1.0f0, β=1.0f0, logpost=true, kl=true, 
                n_samples=1)`

Computes the individual loss terms for the variational autoencoder (VAE) using
`x_in` for encoding and `x_out` for decoding.

The function can compute and return the reconstruction loss `logπ(x_out|z)` and
the Kullback-Leibler (KL) divergence `Dₖₗ[qᵩ(z|x_in) || π(z)]` separately.

# Arguments
- `vae::VAE{<:AbstractVariationalEncoder,SimpleDecoder}`: A VAE model with
  encoder and decoder networks.
- `x_in::AbstractVector{Float32}`: Input vector to the VAE encoder.
- `x_out::AbstractVector{Float32}`: Target vector to compute the reconstruction
  error.

# Optional Keyword Arguments
- `σ::Float32=1.0f0`: Standard deviation for the probabilistic decoder
  π(x_out|z).
- `logpost::Bool=true`: If true, computes the `logπ(x_out|z)` term. 
- `kl::Bool=true`: If true, computes the `Dₖₗ[qᵩ(z|x_in) || π(z)]` term.
- `n_samples::Int=1`: The number of samples to draw from the latent space when
  computing the loss.

# Returns
- `(logπ_x_z, kl_qᵩ_π)`: Tuple containing the computed loss terms. Returns
  `nothing` for the terms not computed based on the flags.

# Note
Ensure that the input data `x_in` matches the expected input dimensionality for
the encoder in the VAE.
"""
function loss_terms(
    vae::VAE{<:AbstractVariationalEncoder,SimpleDecoder},
    x_in::AbstractVector{Float32},
    x_out::AbstractVector{Float32};
    σ::Float32=1.0f0,
    logpost::Bool=true,
    kl::Bool=true,
    n_samples::Int=1
)
    # Forward Pass (run input x_in through reconstruct function)
    μ, logσ, _, x̂ = vae(x_in; latent=true, n_samples=n_samples)

    # Compute ⟨log π(x_out|z)⟩ for a Gaussian decoder if logpost is true,
    # otherwise return nothing
    logπ_x_z = logpost ? (
        -1 / (2 * σ^2 * n_samples) * sum((x_out .- x̂) .^ 2)
    ) : nothing

    # Compute Kullback-Leibler divergence between approximated decoder
    # qᵩ(z|x_in) and latent prior distribution π(z) if kl is true, otherwise
    # return nothing
    kl_qᵩ_π = kl ? (
        sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)
    ) : nothing

    return logπ_x_z, kl_qᵩ_π
end # function

# ==============================================================================

@doc raw"""
    `loss_terms(vae, x; β=1.0f0, logpost=true, kl=true, n_samples=1)`

Computes the individual loss terms for the variational autoencoder (VAE) using
`x` as both encoding and decoding input.

The function can compute and return the reconstruction loss `logπ(x|z)` and the
Kullback-Leibler (KL) divergence `Dₖₗ[qᵩ(z|x) || π(z)]` separately.

# Arguments
- `vae::VAE{<:AbstractVariationalEncoder,T}`: A VAE model.
  - `T<:Union{JointDecoder,SplitDecoder}` ensures the correct VAE type.
- `x::AbstractVector{Float32}`: Input vector.

# Optional Keyword Arguments
- `logpost::Bool=true`: If true, computes the `logπ(x|z)` term. 
- `kl::Bool=true`: If true, computes the `Dₖₗ[qᵩ(z|x) || π(z)]` term.
- `n_samples::Int=1`: The number of samples to draw from the latent space when
  computing the loss.

# Returns
- `(logπ_x_z, kl_qᵩ_π)`: Tuple containing the computed loss terms. Returns
  `nothing` for the terms not computed based on the flags.

# Notes
Ensure that the input data `x` matches the expected input dimensionality for the
encoder in the VAE.
"""
function loss_terms(
    vae::VAE{<:AbstractVariationalEncoder,T},
    x::AbstractVector{Float32};
    logpost::Bool=true,
    kl::Bool=true,
    n_samples::Int=1
) where {T<:Union{JointDecoder,SplitDecoder}}
    # Run input through reconstruct function
    encoder_μ, encoder_logσ, z, (decoder_µ, decoder_logσ) = vae(
        x; latent=true, n_samples=n_samples
    )

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder if logpost is true, otherwise
    # return nothing
    logπ_x_z = logpost ? (
        -1 / (2.0f0 * n_samples) * length(decoder_µ) * log(2 * π) -
        1 / n_samples * sum(decoder_logσ) -
        1 / (2.0f0 * n_samples) * sum((x .- decoder_µ) .^ 2 ./
                                      exp.(2 * decoder_logσ))
    ) : nothing

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z) if kl is true, otherwise return nothing
    kl_qᵩ_π = kl ? (
        1 / 2.0f0 * sum(
            @. (exp(2.0f0 * encoder_logσ) + encoder_μ^2 - 1.0f0) -
               2.0f0 * encoder_logσ
        )
    ) : nothing

    return logπ_x_z, kl_qᵩ_π
end # function

@doc raw"""
    `loss_terms(vae, x_in, x_out; β=1.0f0, logpost=true, kl=true, n_samples=1)`

Computes the individual loss terms for the variational autoencoder (VAE) using
`x_in` as encoding input and `x_out` as the reconstruction target.

The function can compute and return the reconstruction loss `logπ(x_out|z)` and
the Kullback-Leibler (KL) divergence `Dₖₗ[qᵩ(z|x_in) || π(z)]` separately.

# Arguments
- `vae::VAE{<:AbstractVariationalEncoder,T}`: A VAE model.
  - `T<:Union{JointDecoder,SplitDecoder}` ensures the correct VAE type.
- `x_in::AbstractVector{Float32}`: Input vector to the VAE encoder.
- `x_out::AbstractVector{Float32}`: Target vector for the reconstruction.

# Optional Keyword Arguments
- `logpost::Bool=true`: If true, computes the `logπ(x_out|z)` term. 
- `kl::Bool=true`: If true, computes the `Dₖₗ[qᵩ(z|x_in) || π(z)]` term.
- `n_samples::Int=1`: The number of samples to draw from the latent space when
  computing the loss.

# Returns
- `(logπ_x_z, kl_qᵩ_π)`: Tuple containing the computed loss terms. Returns
  `nothing` for the terms not computed based on the flags.

# Notes
Ensure that the input data `x_in` matches the expected input dimensionality for
the encoder in the VAE.
"""
function loss_terms(
    vae::VAE{<:AbstractVariationalEncoder,T},
    x_in::AbstractVector{Float32},
    x_out::AbstractVector{Float32};
    logpost::Bool=true,
    kl::Bool=true,
    n_samples::Int=1
) where {T<:Union{JointDecoder,SplitDecoder}}
    # Run input x_in through reconstruct function
    encoder_μ, encoder_logσ, z, (decoder_μ, decoder_logσ) = vae(
        x_in; latent=true
    )

    # Compute ⟨log π(x_out|z)⟩ for a Gaussian decoder if logpost is true,
    # otherwise return nothing
    logπ_x_z = logpost ? (
        -1 / (2.0f0 * n_samples) * length(decoder_μ) * log(2 * π) -
        1 / n_samples * sum(decoder_logσ) -
        1 / (2.0f0 * n_samples) * sum((x_out .- decoder_μ) .^ 2 ./
                                      exp.(2 * decoder_logσ))
    ) : nothing

    # Compute Kullback-Leibler divergence between approximated decoder
    # qᵩ(z|x_in) and latent prior distribution π(z) if kl is true, otherwise
    # return nothing
    kl_qᵩ_π = kl ? (
        1 / 2.0f0 * sum(
            @. (exp(2.0f0 * encoder_logσ) + encoder_μ^2 - 1.0f0) -
               2.0f0 * encoder_logσ
        )
    ) : nothing

    return logπ_x_z, kl_qᵩ_π
end # function

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# VAE training functions
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@doc raw"""
    `train!(vae, x, opt; loss_kwargs...)`

Customized training function to update parameters of a variational autoencoder
given a loss function.

# Arguments
- `vae::VAE{<:AbstractEncoder,<:AbstractDecoder}`: A struct containing the
  elements of a variational autoencoder.
- `x::AbstractVector{Float32}`: Data on which to evaluate the loss function.
  Columns represent individual samples.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Arguments for the loss
  function. These might include parameters like `σ`, `β`, or `n_samples`,
  depending on the specific loss function in use.

# Description
Trains the VAE by:
1. Computing the gradient of the loss w.r.t the VAE parameters.
2. Updating the VAE parameters using the optimizer.

# Examples
```julia
opt = Flux.setup(Optax.adam(1e-3), vae)
for x in dataloader
    train!(vae, x, opt; β=1.0f0, n_samples=5) 
end
```
"""
function train!(
    vae::VAE{<:AbstractEncoder,<:AbstractDecoder},
    x::AbstractVector{Float32},
    opt::NamedTuple;
    loss_kwargs::Union{NamedTuple,Dict}=Dict()
)
    # Compute VAE gradient
    ∇loss_ = Flux.gradient(vae) do vae_model
        loss(vae_model, x; loss_kwargs...)
    end # do block
    # Update parameters
    Flux.Optimisers.update!(opt, vae, ∇loss_[1])
end # function

@doc raw"""
    `train!(vae, x, opt; loss_kwargs...)`

Customized training function to update parameters of a variational autoencoder
when provided with matrix data.

# Arguments
- `vae::VAE{<:AbstractEncoder,<:AbstractDecoder}`: A struct containing the
  elements of a variational autoencoder.
- `x::AbstractMatrix{Float32}`: Matrix of data samples on which to evaluate the
  loss function. Each column represents an individual sample.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Arguments for the loss
  function. These might include parameters like `σ`, `β`, or `n_samples`,
  depending on the specific loss function in use.
- `average::Bool = true`: If `true`, computes and averages the gradient for all
  samples in `x` before updating parameters. If `false`, updates parameters
  after computing the gradient for each sample.

# Description
Trains the VAE on matrix data by:
1. Computing the gradient of the loss w.r.t the VAE parameters, either for each
   sample individually or averaged across all samples.
2. Updating the VAE parameters using the optimizer.

# Examples
```julia
opt = Flux.setup(Optax.adam(1e-3), vae)
for x in dataloader # assuming dataloader yields matrices
    train!(vae, x, opt; β=1.0f0, n_samples=5) 
end
```
"""
function train!(
    vae::VAE{<:AbstractEncoder,<:AbstractDecoder},
    x::AbstractMatrix{Float32},
    opt::NamedTuple;
    average=true,
    loss_kwargs::Union{NamedTuple,Dict}=Dict()
)
    # Decide on training approach based on 'average'
    if average
        # Compute the averaged gradient across all samples
        ∇loss_ = Flux.gradient(vae) do vae_model
            StatsBase.mean(loss.(Ref(vae_model), eachcol(x); loss_kwargs...))
        end # do block
        # Update parameters using the optimizer
        Flux.Optimisers.update!(opt, vae, ∇loss_[1])
    else
        foreach(col -> train!(vae, col, opt; loss_kwargs...), eachcol(x))
    end # if
end # function

@doc raw"""
    `train!(vae, x, opt; loss_kwargs...)`

Customized training function to update parameters of a variational autoencoder
when provided with 3D tensor data.

# Arguments
- `vae::VAE{<:AbstractEncoder,<:AbstractDecoder}`: A struct containing the
  elements of a variational autoencoder.
- `x::Array{Float32, 3}`: 3D tensor of data samples on which to evaluate the
  loss function. Each slice represents a matrix, and within each matrix, each
  column represents an individual sample.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Arguments for the loss
  function. These might include parameters like `σ`, `β`, or `n_samples`,
  depending on the specific loss function in use.
- `average::Bool = true`: If `true`, computes and averages the gradient for all
  samples in `x` before updating parameters. If `false`, updates parameters
  after computing the gradient for each sample.

# Description
Trains the VAE on 3D tensor data by:
1. Computing the gradient of the loss w.r.t the VAE parameters, either for each
   sample individually or averaged across all samples within a slice.
2. Updating the VAE parameters using the optimizer.

# Examples
```julia
opt = Flux.setup(Optax.adam(1e-3), vae)
for x in dataloader # assuming dataloader yields 3D tensors
    train!(vae, x, opt; β=1.0f0, n_samples=5) 
end
```
"""
function train!(
    vae::VAE{<:AbstractEncoder,<:AbstractDecoder},
    x::Array{Float32,3},
    opt::NamedTuple;
    average=true,
    loss_kwargs::Union{NamedTuple,Dict}=Dict()
)
    # Decide on training approach based on 'average'
    if average
        # Compute the averaged gradient across all slices of the tensor
        ∇loss_ = Flux.gradient(vae) do vae_model
            StatsBase.mean([
                StatsBase.mean(
                    loss.(Ref(vae_model), eachcol(slice)); loss_kwargs...
                )
                for slice in eachslice(x, dims=3)
            ])
        end # do block
        # Update parameters using the optimizer
        Flux.Optimisers.update!(opt, vae, ∇loss_[1])
    else
        foreach(
            slice -> foreach(
                col -> train!(vae, col, opt; loss_kwargs...), eachcol(slice)
            ),
            eachslice(x, dims=3)
        )
    end # if
end # function

# ==============================================================================

@doc raw"""
    `train!(vae, x_in, x_out, opt; loss_kwargs...)`

Customized training function to update parameters of a variational autoencoder
given a loss function.

# Arguments
- `vae::VAE{<:AbstractEncoder,<:AbstractDecoder}`: A struct containing the
  elements of a variational autoencoder.
- `x_in::AbstractVector{Float32}`: Input data for the loss function. Represents
  an individual sample.
- `x_out::AbstractVector{Float32}`: Target output data for the loss function.
  Represents the corresponding output for the `x_in` sample.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Arguments for the loss
  function. These might include parameters like `σ`, `β`, or `n_samples`,
  depending on the specific loss function in use.

# Description
Trains the VAE by:
1. Computing the gradient of the loss w.r.t the VAE parameters.
2. Updating the VAE parameters using the optimizer.

# Examples
```julia
opt = Flux.setup(Optax.adam(1e-3), vae)
for (x_in, x_out) in dataloader
    train!(vae, x_in, x_out, opt) 
end
```
"""
function train!(
    vae::VAE{<:AbstractEncoder,<:AbstractDecoder},
    x_in::AbstractVector{Float32},
    x_out::AbstractVector{Float32},
    opt::NamedTuple;
    loss_kwargs::Union{NamedTuple,Dict}=Dict()
)
    # Compute VAE gradient
    ∇loss_ = Flux.gradient(vae) do vae_model
        loss(vae_model, x_in, x_out; loss_kwargs...)
    end # do block
    # Update parameters
    Flux.Optimisers.update!(opt, vae, ∇loss_[1])
end # function

@doc raw"""
    `train!(vae, x_in, x_out, opt; loss_kwargs...)`

Customized training function to update parameters of a variational autoencoder
when provided with matrix data.

# Arguments
- `vae::VAE{<:AbstractEncoder,<:AbstractDecoder}`: A struct containing the
  elements of a variational autoencoder.
- `x_in::AbstractMatrix{Float32}`: Matrix of input data samples on which to
  evaluate the loss function. Each column represents an individual sample.
- `x_out::AbstractMatrix{Float32}`: Matrix of target output data samples. Each
  column represents the corresponding output for an individual sample in `x_in`.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Arguments for the loss
  function. These might include parameters like `σ`, `β`, or `n_samples`,
  depending on the specific loss function in use.
- `average::Bool = true`: If `true`, computes and averages the gradient for all
  samples in `x_in` before updating parameters. If `false`, updates parameters
  after computing the gradient for each sample.

# Description
Trains the VAE on matrix data by:
1. Computing the gradient of the loss w.r.t the VAE parameters, either for each
   sample individually or averaged across all samples.
2. Updating the VAE parameters using the optimizer.

# Examples
```julia
opt = Flux.setup(Optax.adam(1e-3), vae)
for (x_in_batch, x_out_batch) in dataloader # assuming dataloader yields matrices
    train!(vae, x_in_batch, x_out_batch, opt) 
end
```
"""
function train!(
    vae::VAE{<:AbstractEncoder,<:AbstractDecoder},
    x_in::AbstractMatrix{Float32},
    x_out::AbstractMatrix{Float32},
    opt::NamedTuple;
    average=true,
    loss_kwargs::Union{NamedTuple,Dict}=Dict()
)
    # Decide on training approach based on 'average'
    if average
        # Compute the averaged gradient across all samples
        ∇loss_ = Flux.gradient(vae) do vae_model
            StatsBase.mean(
                loss.(
                    Ref(vae_model),
                    eachcol(x_in),
                    eachcol(x_out);
                    loss_kwargs...
                )
            )
        end
        # Update parameters using the optimizer
        Flux.Optimisers.update!(opt, vae, ∇loss_[1])
    else
        foreach(
            (col_in, col_out) -> train!(
                vae, col_in, col_out, opt; loss_kwargs...
            ), zip(eachcol(x_in), eachcol(x_out)
            )
        )
    end # for
end # function

@doc raw"""
    `train!(vae, x_in, x_out, opt; loss_kwargs...)`

Customized training function to update parameters of a variational autoencoder
when provided with 3D tensor data.

# Arguments
- `vae::VAE{<:AbstractEncoder,<:AbstractDecoder}`: A struct containing the
  elements of a variational autoencoder.
- `x_in::Array{Float32, 3}`: 3D tensor of input data samples on which to
  evaluate the loss function. Each slice represents a matrix, and within each
  matrix, each column represents an individual sample.
- `x_out::Array{Float32, 3}`: 3D tensor of target output data samples. Each
  slice represents a matrix, and within each matrix, each column represents the
  corresponding output for an individual sample in `x_in`.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Arguments for the loss
  function. These might include parameters like `σ`, `β`, or `n_samples`,
  depending on the specific loss function in use.
- `average::Bool = true`: If `true`, computes and averages the gradient for all
  samples in `x_in` before updating parameters. If `false`, updates parameters
  after computing the gradient for each sample.

# Description
Trains the VAE on 3D tensor data by:
1. Computing the gradient of the loss w.r.t the VAE parameters, either for each
   sample individually or averaged across all samples within a slice.
2. Updating the VAE parameters using the optimizer.

# Examples
```julia
opt = Flux.setup(Optax.adam(1e-3), vae)
for (x_in_batch, x_out_batch) in dataloader # assuming dataloader yields 3D tensors
    train!(vae, x_in_batch, x_out_batch, opt; β=1.0f0, n_samples=5) 
end
```
"""
function train!(
    vae::VAE{<:AbstractEncoder,<:AbstractDecoder},
    x_in::Array{Float32,3},
    x_out::Array{Float32,3},
    opt::NamedTuple;
    average=true,
    loss_kwargs::Union{NamedTuple,Dict}=Dict()
)
    # Decide on training approach based on 'average'
    if average
        # Compute the averaged gradient across all slices of the tensor
        ∇loss_ = Flux.gradient(vae) do vae_model
            StatsBase.mean([
                StatsBase.mean(
                    loss.(
                        Ref(vae_model),
                        eachcol(slice_in),
                        eachcol(slice_out);
                        loss_kwargs...
                    )
                )
                for (slice_in, slice_out) in zip(
                    eachslice(x_in, dims=3), eachslice(x_out, dims=3)
                )
            ])
        end
        # Update parameters using the optimizer
        Flux.Optimisers.update!(opt, vae, ∇loss_[1])
    else
        foreach(
            (slice_in, slice_out) -> foreach(
                (col_in, col_out) -> train!(
                    vae, col_in, col_out, opt; loss_kwargs...
                ),
                zip(eachcol(slice_in), eachcol(slice_out))
            ),
            zip(eachslice(x_in, dims=3), eachslice(x_out, dims=3))
        )
    end # if
end # function