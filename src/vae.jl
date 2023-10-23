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
    reparameterize(µ, logσ; prior=Distributions.Normal{Float32}(0.0f0, 1.0f0))

Reparameterize the latent space using the given mean (`µ`) and log standard
deviation (`logσ`), employing the reparameterization trick. This function helps
in sampling from the latent space in variational autoencoders (or similar
models) while keeping the gradient flow intact.

# Arguments
- `µ::Array{Float32}`: The mean of the latent space.
- `logσ::Array{Float32}`: The log standard deviation of the latent space.

# Optional Keyword Arguments
- `prior::Distributions.Sampleable`: The prior distribution for the latent
  space. By default, this is a standard normal distribution
  (`Distributions.Normal{Float32}(0.0f0, 1.0f0)`). The function supports both
  univariate and multivariate distributions from the `Distributions` package.

# Returns
A sampled point from the reparameterized latent space, obtained by applying the
reparameterization trick on the provided mean and log standard deviation, using
the specified prior distribution.

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
    µ::Array{Float32},
    logσ::Array{Float32};
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0)
)
    # Check type of prior distribution
    if typeof(prior) <: Distributions.UnivariateDistribution
        # Sample random latent variable point estimate given the mean and
        # standard deviation
        return µ .+ Random.rand(prior, size(µ)...) .* exp.(logσ)
    elseif typeof(prior) <: Distributions.MultivariateDistribution
        # Sample random latent variable point estimate given the mean and
        # standard deviation
        return µ .+ Random.rand(prior, 1) .* exp.(logσ)
    end # if
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
    (decoder::SimpleDecoder)(z::Array{Float32})

Maps the given latent representation `z` through the `SimpleDecoder` network.

# Arguments
- `z::Array{Float32}`: The latent space representation to be decoded. Typically,
  this is a point or sample from the latent space of a VAE.

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
dimensionality for the SimpleDecoder
"""
function (decoder::SimpleDecoder)(z::Array{Float32})
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
    if length(decoder) > 1
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
    (decoder::JointDecoder)(z::Array{Float32})

Maps the given latent representation `z` through the `JointDecoder` network to
produce both the mean (`µ`) and log standard deviation (`logσ`).

# Arguments
- `z::Array{Float32}`: The latent space representation to be decoded. Typically,
  this is a point or sample from the latent space of a VAE.

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
function (decoder::JointDecoder)(z::AbstractVecOrMat{Float32})
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
    decoder_µ::Flux.Chain
    decoder_logσ::Flux.Chain
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

# Keyword Arguments
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
    µ_layers = [
        Flux.Dense(n_latent => µ_neurons[1], µ_activation[1]; init=init)
    ]

    # Loop through rest of the layers
    for i = 2:length(µ_neurons)
        # Add next layer to list
        push!(
            µ_layers,
            Flux.Dense(
                µ_neurons[i-1] => µ_neurons[i], µ_activation[i]; init=init
            )
        )
    end

    # Initialize logσ decoder layers
    logσ_layers = [
        Flux.Dense(n_latent => logσ_neurons[1], logσ_activation[1]; init=init)
    ]

    # Loop through rest of the layers
    for i = 2:length(logσ_neurons)
        # Add next layer to list
        push!(
            logσ_layers,
            Flux.Dense(
                logσ_neurons[i-1] => logσ_neurons[i],
                logσ_activation[i];
                init=init
            )
        )
    end

    # Initialize split decoder
    return SplitDecoder(Flux.Chain(µ_layers...), Flux.Chain(logσ_layers...))
end # function

@doc raw"""
    (decoder::SplitDecoder)(z::Array{Float32})

Maps the given latent representation `z` through the separate networks of the
`SplitDecoder` to produce both the mean (`µ`) and log standard deviation
(`logσ`).

# Arguments
- `z::Array{Float32}`: The latent space representation to be decoded. Typically,
  this is a point or sample from the latent space of a VAE.

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
function (decoder::SplitDecoder)(z::AbstractVecOrMat{Float32})
    # Map through the decoder dedicated to the mean
    µ = decoder.decoder_µ(z)
    # Map through the decoder dedicated to the log standard deviation
    logσ = decoder.decoder_logσ(z)
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
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0))

Processes the given input data `x` through a VAE that consists of a
`JointEncoder` and a `SimpleDecoder`.

# Arguments
- `x::AbstractVecOrMat{Float32}`: Input data to be processed by the VAE. 

# Optional Keyword Arguments
- `prior::Distributions.Sampleable`: Specifies the prior distribution to be used
  during the reparametrization trick. Defaults to a standard normal
  distribution.

# Returns
- `Array{Float32}`: The reconstructed data after processing through the encoder,
  performing the reparametrization trick, and passing through the decoder.

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
reconstructed_data = vae_model(input_data)
```
# Note
Ensure that the input data x matches the expected input dimensionality for the
encoder in the VAE.
"""
function (vae::VAE{JointEncoder,SimpleDecoder})(
    x::AbstractVecOrMat{Float32};
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0)
)
    # Run input through encoder to obtain mean and log std
    encoder_µ, encoder_logσ = vae.encoder(x)

    # Run reparametrization trick
    z_sample = reparameterize(encoder_µ, encoder_logσ; prior)

    # Run latent sample through decoder
    return vae.decoder(z_sample)
end # function

# ==============================================================================

@doc raw"""
    vae_init(n_input, n_latent, latent_activation, output_activation, 
             encoder, encoder_activation, decoder, decoder_activation;
             init=Flux.glorot_uniform)

Initialize a `VAE` model architecture `struct` using Flux.jl.

# Arguments
- `n_input::Int`: Dimensionality of the input data. 
- `n_latent::Int`: Dimensionality of the latent space.
- `latent_activation::Function`: Activation function for the latent space
  layers.
- `output_activation::Function`: Activation function for the output layer.
- `encoder::Vector{<:Int}`: Vector of layer sizes for the encoder network.
- `encoder_activation::Vector{<:Function}`: Activation functions for each
  encoder layer.
- `decoder::Vector{<:Int}`: Vector of layer sizes for the decoder network. 
- `decoder_activation::Vector{<:Function}`: Activation functions for each
  decoder layer.

# Keyword Arguments
- `init=Flux.glorot_uniform`: Initialization function for network parameters.

# Returns
- `vae`: A VAE model with encoder, latent variables and decoder.

# Examples
```julia
vae = vae_init(
    28^2, 10, tanh, sigmoid, [128, 64], [relu, relu], [64, 128], [relu, relu]
)
```
"""
function vae_init(
    n_input::Int,
    n_latent::Int,
    latent_activation::Function,
    output_activation::Function,
    encoder::Vector{<:Int},
    encoder_activation::Vector{<:Function},
    decoder::Vector{<:Int},
    decoder_activation::Vector{<:Function};
    init::Function=Flux.glorot_uniform
)
    # Check there's enough activation functions for all layers
    if (length(encoder_activation) != length(encoder)) |
       (length(decoder_activation) != length(decoder))
        error("Each layer needs exactly one activation function")
    end # if

    # Initialize list with encoder layers
    Encoder = Array{Flux.Dense}(undef, length(encoder))

    # Loop through layers   
    for i = 1:length(encoder)
        # Check if it is the first layer
        if i == 1
            # Set first layer from input to encoder with activation
            Encoder[i] = Flux.Dense(
                n_input => encoder[i], encoder_activation[i]; init=init
            )
        else
            # Set middle layers from input to encoder with activation
            Encoder[i] = Flux.Dense(
                encoder[i-1] => encoder[i], encoder_activation[i]; init=init
            )
        end # if
    end # for

    # Define layer that maps from encoder to latent space with activation
    Latent_µ = Flux.Dense(
        encoder[end] => n_latent, latent_activation; init=init
    )
    Latent_logσ = Flux.Dense(
        encoder[end] => n_latent, latent_activation; init=init
    )

    # Initialize list with decoder layers
    Decoder = Array{Flux.Dense}(undef, length(decoder) + 1)

    # Add first layer from latent space to decoder
    Decoder[1] = Flux.Dense(
        n_latent => decoder[1], decoder_activation[1]; init=init
    )

    # Add last layer from decoder to output
    Decoder[end] = Flux.Dense(
        decoder[end] => n_input, output_activation; init=init
    )

    # Check if there are multiple middle layers
    if length(decoder) > 1
        # Loop through middle layers
        for i = 2:length(decoder)
            # Set middle layers of decoder
            Decoder[i] = Flux.Dense(
                decoder[i-1] => decoder[i], decoder_activation[i]; init=init
            )
        end # for
    end # if

    # Compile encoder and decoder into single chain
    return VAE(
        Flux.Chain(Encoder...), Latent_µ, Latent_logσ, Flux.Chain(Decoder...)
    )
end # function

@doc raw"""
`VAE(input; latent)`

Pass input through the VAE and return reconstructed output. 

This follows 3 steps:

1. Encode input via the encoder network.
2. Sample the latent code with the reparameterization trick. 
3. Decode the sampled latent code with the decoder network.


# Arguments
- `input::AbstractVecOrMat{Float32}`: Input to the neural network.

## Optional Arguments
- `latent::Bool=true`: Boolean indicating if the parameters of the latent
  representation (mean `µ`, log standard deviation `logσ`) should be returned.

# Returns
- `µ::Vector{Float32}`: Array containing the mean value of the input when mapped
to the latent space.
- `logσ::Vector{Float32}`: Array containing the log of the standard deviation of
the input when mapped to the latent space.
- `z::Vector{Float32}`: Random sample of latent space code for the input. NOTE:
  This will change every time the function is called on the same input because
  of the random number sampling.
- `x̂::Vector{Float32}`: The reconstructed input `x` after passing through the
  autoencoder. NOTE: This will change every time the function is called on the
  same input because of the random number sampling.

# Examples
```julia
x = rand(28^2)
x̂ = VAE(x) # reconstruct x 
μ, logσ, z, x̂ = VAE(x, latent=true) # return latent params
```
"""
function (vae::VAE)(
    input::AbstractVecOrMat{Float32};
    latent::Bool=false
)

    # 1. Map input through encoder layer
    encode_input = vae.encoder(input)

    # 2. map encoder layer output to mean and log standard deviation of latent
    #    variables
    µ = vae.µ(encode_input)
    logσ = vae.logσ(encode_input)

    # 3. Sample random latent variable point estimate given the mean and
    #    standard deviation
    z = µ .+ Random.rand(
        Distributions.Normal{Float32}(0.0f0, 1.0f0), size(µ)...
    ) .* exp.(logσ)

    # 4. Run sampled latent variables through decoder and return values
    if latent
        return µ, logσ, z, vae.decoder(z)
    else
        return vae.decoder(z)
    end # if
end # function

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# VAE loss functions
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@doc raw"""
    `loss(vae, x; σ, β)`

Loss function for the variational autoencoder. The loss function is defined as

loss = argmin -⟨log π(x|z)⟩ + β Dₖₗ(qᵩ(z | x) || π(z)),

where the minimization is taken over the functions f̲, g̲, and h̲̲. f̲(z)
encodes the function that defines the mean ⟨x|z⟩ of the decoder π(x|z), i.e.,

    π(x|z) = Normal(f̲(x), σI).

g̲ and h̲̲ define the mean and covariance of the approximate decoder qᵩ(z|x),
respectively, i.e.,

    π(z|x) ≈ qᵩ(z|x) = Normal(g̲(x), h̲̲(x)).

# Arguments
- `vae::VAE`: Struct containint the elements of the variational autoencoder.
- `x::AbstractVector{Float32}`: Input to the neural network. NOTE: This only
  takes a vector as input. If a batch or the entire data is to be evaluated, use
  something like `sum(loss.(Ref(vae), eachcol(x)))`.

## Optional arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder π(x|z).
- `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.

# Returns
- `loss::Float32`: Single value defining the loss function for entry `x` when
compared with reconstructed output `x̂`.
"""
function loss(
    vae::VAE,
    x::AbstractVector{Float32};
    σ::Float32=1.0f0,
    β::Float32=1.0f0
)
    # Run input through reconstruct function
    µ, logσ, _, x̂ = vae(x; latent=true)

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder
    logπ_x_z = -1 / (2 * σ^2) * sum((x .- x̂) .^ 2)

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_qᵩ_π = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

    # Compute loss function
    return -logπ_x_z + β * kl_qᵩ_π

end #function

@doc raw"""
    `loss(vae, x, x_true; σ, β)`

Loss function for the variational autoencoder. The loss function is defined as

loss = argmin -⟨log π(x|z)⟩ + β Dₖₗ(qᵩ(z | x) || π(z)),

where the minimization is taken over the functions f̲, g̲, and h̲̲. f̲(z)
encodes the function that defines the mean ⟨x|z⟩ of the decoder π(x|z), i.e.,

    π(x|z) = Normal(f̲(x), σI).

g̲ and h̲̲ define the mean and covariance of the approximate decoder qᵩ(z|x),
respectively, i.e.,

    π(z|x) ≈ qᵩ(z|x) = Normal(g̲(x), h̲̲(x)).

NOTE: This method accepts an extra argument `x_true` as the ground truth against
which to compare the input values that is not necessarily the same as the input
value.

# Arguments
- `x::AbstractVector{Float32}`: Input to the neural network.
- `x_true::AbstractVector{Float32}`: True input against which to compare
  autoencoder reconstruction. NOTE: This only takes a vector as input. If a
  batch or the entire data is to be evaluated, use something like
  `sum(loss.(Ref(vae), eachcol(x)), eachcol(x_true))`.
- `vae::VAE`: Struct containint the elements of the variational autoencoder.

## Optional arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder π(x|z).
- `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.

# Returns
- `loss::Float32`: Single value defining the loss function for entry `x` when
compared with reconstructed output `x̂`.
"""
function loss(
    vae::VAE,
    x::AbstractVector{Float32},
    x_true::AbstractVector{Float32};
    σ::Float32=1.0f0,
    β::Float32=1.0f0
)
    # Run input through reconstruct function
    µ, logσ, _, x̂ = vae(x; latent=true)

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder
    logπ_x_z = -1 / (2 * σ^2) * sum((x_true .- x̂) .^ 2)

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_qᵩ_π = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

    # Compute loss function
    return -logπ_x_z + β * kl_qᵩ_π

end #function

@doc raw"""
    `kl_div(vae, x)`

Function to compute the KL divergence between the approximate encoder qᵩ(z) and
the latent variable prior distribution π(z). Since we assume
        π(z) = Normal(0̲, 1̲),
and
        qᵩ(z) = Normal(f̲(x̲), σI̲̲),
the KL divergence has a closed form

# Arguments
- `vae::VAE`: Struct containint the elements of the variational autoencoder.        
- `x::AbstractVector{Float32}`: Input to the neural network.

# Returns
Dₖₗ(qᵩ(z)||π(z))
"""
function kl_div(vae::VAE, x::AbstractVector{Float32})
    # Map input to mean and log standard deviation of latent variables
    µ = Flux.Chain(vae.encoder..., vae.µ)(x)
    logσ = Flux.Chain(vae.encoder..., vae.logσ)(x)

    return sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)
end # function

@doc raw"""
    `loss_terms(vae, x; σ, β)`

Loss function for the variational autoencoder. NOTE: This function performs the
same computations as the `loss` function, but simply returns each term
individually.

# Arguments
- `vae::VAE`: Struct containint the elements of the variational autoencoder.
- `x::AbstractVector{Float32}`: Input to the neural network.

## Optional arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder π(x|z).
- `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.

# Returns
- `loss::Float32`: Single value defining the loss function for entry `x` when
compared with reconstructed output `x̂`.
"""
function loss_terms(
    vae::VAE,
    x::AbstractVector{Float32};
    σ::Float32=1.0f0,
    β::Float32=1.0f0
)
    # Run input through reconstruct function
    µ, logσ, _, x̂ = vae(x; latent=true)

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder
    logπ_x_z = -1 / (2 * σ^2) * sum((x .- x̂) .^ 2)

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_qᵩ_π = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

    # Compute loss function
    return [logπ_x_z, β * kl_qᵩ_π]
end #function

@doc raw"""
    `loss_terms(vae, x, x_true; σ, β)`

Loss function for the variational autoencoder. NOTE: This function performs the
same computations as the `loss` function, but simply returns each term
individually.

# Arguments
- `x::AbstractVector{Float32}`: Input to the neural network.
- `x_true::AbstractVector{Float32}`: True input against which to compare
  autoencoder reconstruction.
- `vae::VAE`: Struct containint the elements of the variational autoencoder.

## Optional arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder π(x|z).
- `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.

# Returns
- `loss::Float32`: Single value defining the loss function for entry `x` when
compared with reconstructed output `x̂`.
"""
function loss_terms(
    vae::VAE,
    x::AbstractVector{Float32},
    x_true::AbstractVector{Float32};
    σ::Float32=1.0f0,
    β::Float32=1.0f0
)
    # Run input through reconstruct function
    µ, logσ, _, x̂ = vae(x; latent=true)

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder
    logπ_x_z = -1 / (2 * σ^2) * sum((x_true .- x̂) .^ 2)

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_qᵩ_π = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

    # Compute loss function
    return [logπ_x_z, β * kl_qᵩ_π]
end #function

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# VAE training functions
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@doc raw"""
    `train!(vae, x, opt; kwargs...)`

Customized training function to update parameters of variational autoencoder
given a loss function.

# Arguments
- `vae::VAE`: Struct containint the elements of a variational autoencoder.
- `x::AbstractVecOrMat{Float32}`: Matrix containing the data on which to
  evaluate the loss function. NOTE: Every column should represent a single
  input.
- `opt::NamedTuple`: State of optimizer that will be used to update parameters.
  NOTE: This is in agreement with `Flux.jl ≥ 0.13` where implicit `Zygote`
  gradients are not allowed. This `opt` object can be initialized using
  `Flux.Train.setup`. For example, one can run
  ```
  opt_state = Flux.Train.setup(Flux.Optimisers.Adam(1E-1), vae)
  ```

## Optional arguments
- `loss_kwargs::Union{NamedTuple,Dict}`: Tuple containing arguments for the loss
    function. For `loss`, for example, we have
    - `σ::Float32=1`: Standard deviation of the probabilistic decoder π(x|z).
    - `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
- `average::Bool`: Boolean variable indicating if the gradient should be
  computed for all elements in `x`, averaged and then update the parameters once
  (`average=true`) or compute the gradient for each element in `x` and update
  the parameter every time. Default is true.

# Description
1. Compute the gradient of the loss w.r.t the VAE parameters
2. Update the VAE parameters using the optimizer

The loss function depends on the data `x` and hyperparameters `σ` and `β`. This
allows full customization during training.

# Examples
```julia 
opt = Flux.setup(Optax.adam(1e-3), vae)
for x in dataloader
    train!(vae, x, opt) 
end
```
"""
function train!(
    vae::VAE,
    x::AbstractVecOrMat{Float32},
    opt::NamedTuple;
    loss_kwargs::Union{NamedTuple,Dict}=Dict(:σ => 1.0f0, :β => 1.0f0,),
    average::Bool=true
)
    if typeof(x) <: Vector{Float32}
        # Compute gradient
        ∇loss_ = Flux.gradient(vae) do vae_model
            loss(vae_model, x; loss_kwargs...)
        end # do

        # Update the network parameters averaging gradient from all datasets
        Flux.Optimisers.update!(
            opt,
            vae,
            ∇loss_[1]
        )
        # Check if average should be computed
    elseif (average) & (typeof(x) <: Matrix{Float32})
        # Compute gradient
        ∇loss_ = Flux.gradient(vae) do vae_model
            sum(loss.(Ref(vae_model), eachcol(x); loss_kwargs...)) ./ size(x, 2)
        end # do

        # Update the network parameters averaging gradient from all datasets
        Flux.Optimisers.update!(
            opt,
            vae,
            ∇loss_[1]
        )
    else
        # Loop through the rest of elements
        for d in eachcol(x)
            # Update gradient
            ∇loss_ = Flux.gradient(vae) do vae_model
                loss(vae_model, d; loss_kwargs...)
            end # do
            # Update the network parameters averaging gradient from all datasets
            Flux.Optimisers.update!(
                opt,
                vae,
                ∇loss_[1]
            )
        end # for
    end # if
end # function

@doc raw"""
    `train!(vae, x, x_true, opt; kwargs...)`

Customized training function to update parameters of variational autoencoder
given a loss function.

# Arguments
- `vae::VAE`: Struct containint the elements of a variational autoencoder.
- `x::AbstractVecOrMat{Float32}`: Array containing the data to be ran through
  the VAE. NOTE: Every column should represent a single input.
- `x_true;:AbstractVecOrMat{Float32}`: Array containing the data used to compare
  the reconstruction for the loss function. This can be used to train denoising
  VAE, for exmaple.
- `opt::NamedTuple`: State of optimizer that will be used to update parameters.
  NOTE: This is in agreement with `Flux.jl ≥ 0.13` where implicit `Zygote`
  gradients are not allowed. This `opt` object can be initialized using
  `Flux.Train.setup`. For example, one can run
  ```
  opt_state = Flux.Train.setup(Flux.Optimisers.Adam(1E-1), vae)
  ```

## Optional arguments
- `loss_kwargs::Union{NamedTuple,Dict}`: Tuple containing arguments for the loss
    function. For `loss`, for example, we have
    - `σ::Float32=1`: Standard deviation of the probabilistic decoder π(x|z).
    - `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
- `average::Bool`: Boolean variable indicating if the gradient should be
    computed for all elements in `x`, averaged and then update the parameters
    once (`average=true`) or compute the gradient for each element in `x` and
    update the parameter every time. Default is `true`.

# Description
1. Compute the gradient of the loss w.r.t the VAE parameters
2. Update the VAE parameters using the optimizer

The loss function depends on the data `x` and `x_true` and hyperparameters `σ`
and `β`. This allows full customization during training. The main difference
with the method that only takes `x` as input is that the comparison at the
output layer does not need to necessarily match that of the input. Useful for
data augmentation training schemes.

# Examples
```julia 
opt = Flux.setup(Optax.adam(1e-3), vae)
for x in dataloader
    train!(vae, x, opt) 
end
```
"""
function train!(
    vae::VAE,
    x::AbstractVecOrMat{Float32},
    x_true::AbstractVecOrMat{Float32},
    opt::NamedTuple;
    loss_kwargs::Union{NamedTuple,Dict}=Dict(:σ => 1.0f0, :β => 1.0f0,),
    average::Bool=true
)
    if typeof(x) <: Vector{Float32}
        # Compute gradient
        ∇loss_ = Flux.gradient(vae) do vae_model
            loss(vae_model, x, x_true; loss_kwargs...)
        end # do

        # Update the network parameters
        Flux.Optimisers.update!(
            opt,
            vae,
            ∇loss_[1]
        )
        # Check if average should be computed
    elseif (average) & (typeof(x) <: Matrix{Float32})
        # Compute gradient
        ∇loss_ = Flux.gradient(vae) do vae_model
            sum(
                loss.(Ref(vae_model), eachcol(x), eachcol(x_true); loss_kwargs...)
            ) ./ size(x, 2)
        end # do

        # Update the network parameters averaging gradient from all datasets
        Flux.Optimisers.update!(
            opt,
            vae,
            ∇loss_[1]
        )
    else
        # Loop through the elements
        for i in axes(x, 2)
            # Compute gradient
            ∇loss_ = Flux.gradient(vae) do vae_model
                loss(vae_model, x[:, i], x_true[:, i]; loss_kwargs...)
            end # do
            # Update the network parameters
            Flux.Optimisers.update!(
                opt,
                vae,
                ∇loss_[1]
            )
        end # for
    end # if
end # function