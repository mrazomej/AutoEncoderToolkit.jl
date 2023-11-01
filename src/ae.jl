# Import ML libraries
import Flux
import NNlib
import SimpleChains

# Import Stats libraries
import StatsBase

##

# Import Abstract Types

using ..AutoEncode: AbstractAutoEncoder, AbstractDeterministicAutoEncoder,
    AbstractDeterministicEncoder, AbstractDeterministicDecoder

## 

# ==============================================================================

@doc raw"""
`struct Encoder`

Default encoder function for deterministic autoencoders. The `encoder` network
is used to map the input data directly into the latent space representation.

# Fields
- `encoder::Flux.Chain`: The primary neural network used to process input data
  and map it into a latent space representation.

# Example
```julia
enc = Encoder(Flux.Chain(Dense(784, 400, relu), Dense(400, 20)))
```
"""
mutable struct Encoder <: AbstractDeterministicEncoder
    encoder::Flux.Chain
end # struct

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor Encoder

@doc raw"""
    Encoder(n_input, n_latent, latent_activation, encoder_neurons, 
            encoder_activation; init=Flux.glorot_uniform)

Construct and initialize an `Encoder` struct that defines an encoder network for
a deterministic autoencoder.

# Arguments
- `n_input::Int`: The dimensionality of the input data.
- `n_latent::Int`: The dimensionality of the latent space.
- `encoder_neurons::Vector{<:Int}`: A vector specifying the number of neurons in
  each layer of the encoder network.
- `encoder_activation::Vector{<:Function}`: Activation functions corresponding
  to each layer in the `encoder_neurons`.
- `latent_activation::Function`: Activation function for the latent space layer.

## Optional Keyword Arguments
- `init::Function=Flux.glorot_uniform`: The initialization function used for the
  neural network weights.

# Returns
- An `Encoder` struct initialized based on the provided arguments.

# Examples
```julia
encoder = Encoder(784, 20, tanh, [400], [relu])
````
# Notes
The length of encoder_neurons should match the length of encoder_activation,
ensuring that each layer in the encoder has a corresponding activation function.
"""
function Encoder(
    n_input::Int,
    n_latent::Int,
    encoder_neurons::Vector{<:Int},
    encoder_activation::Vector{<:Function},
    latent_activation::Function;
    init::Function=Flux.glorot_uniform
)
    # Ensure there's a matching activation function for every layer in the encoder
    if length(encoder_activation) != length(encoder_neurons)
        error("Each layer needs exactly one activation function in encoder")
    end # if
    # Initialize list with encoder layers
    layers = []

    # Iterate through encoder layers and add them to the list
    for i = 1:length(encoder_neurons)
        # For the first layer
        if i == 1
            push!(
                layers,
                Flux.Dense(
                    n_input => encoder_neurons[i],
                    encoder_activation[i];
                    init=init
                )
            )
        else
            # For subsequent layers
            push!(
                layers,
                Flux.Dense(
                    encoder_neurons[i-1] => encoder_neurons[i],
                    encoder_activation[i];
                    init=init
                )
            )
        end # if
    end # for

    # Add the layer mapping to the latent space, with its specified activation
    # function
    push!(
        layers,
        Flux.Dense(
            encoder_neurons[end] => n_latent,
            latent_activation;
            init=init
        )
    )

    return Encoder(Flux.Chain(layers...))
end # function

@doc raw"""
    (encoder::Encoder)(x)

Forward propagate the input `x` through the `Encoder` to obtain the encoded
representation in the latent space.

# Arguments
- `x::Array{Float32}`: Input data to be encoded.

# Returns
- `z`: Encoded representation of the input data in the latent space.

# Description
This method allows for a direct call on an instance of `Encoder` with the input
data `x`. It runs the input through the encoder network and outputs the encoded
representation in the latent space.

# Example
```julia
enc = Encoder(...)
z = enc(some_input)
```
# Note

Ensure that the input x matches the expected dimensionality of the encoder's
input layer.
"""
function (encoder::Encoder)(x::AbstractVecOrMat{Float32})
    # Run input through the encoder network to obtain the encoded representation
    return encoder.encoder(x)
end # function
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
mutable struct Decoder <: AbstractDeterministicDecoder
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

# ==============================================================================

@doc raw"""
`struct AE{E<:AbstractDeterministicEncoder, D<:AbstractDeterministicDecoder}`

Autoencoder (AE) model defined for `Flux.jl`

# Fields
- `encoder::E`: Neural network that encodes the input into the latent space. `E`
  is a subtype of `AbstractDeterministicEncoder`.
- `decoder::D`: Neural network that decodes the latent representation back to
  the original input space. `D` is a subtype of `AbstractDeterministicDecoder`.

An AE consists of an encoder and decoder network with a bottleneck latent space
in between. The encoder compresses the input into a low-dimensional
representation. The decoder tries to reconstruct the original input from the
point in the latent space. 
"""
mutable struct AE{
    E<:AbstractDeterministicEncoder,
    D<:AbstractDeterministicDecoder
} <: AbstractDeterministicAutoEncoder
    encoder::E
    decoder::D
end # struct

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor AE

@doc raw"""
    (ae::AE{Encoder, Decoder})(x::AbstractVecOrMat{Float32}; latent::Bool=false)

Processes the input data `x` through the autoencoder (AE) that consists of an
encoder and a decoder.

# Arguments
- `x::AbstractVecOrMat{Float32}`: The data to be decoded. This can be a vector
  or a matrix where each column represents a separate sample.

# Optional Keyword Arguments
- `latent::Bool`: If set to `true`, returns a dictionary containing the latent
  representation alongside the reconstructed data. Defaults to `false`.

# Returns
- If `latent=false`: `Array{Float32}`, the reconstructed data after processing
  through the encoder and decoder.
- If `latent=true`: A dictionary with keys `:z`, and `:reconstructed`,
  containing the corresponding values.

# Description
The function first encodes the input `x` using the encoder to get the encoded
representation in the latent space. This latent representation is then decoded
using the decoder to produce the reconstructed data. If `latent` is set to true,
it also returns the latent representation.

# Note
Ensure the input data `x` matches the expected input dimensionality for the
encoder in the AE.
"""

function (ae::AE{Encoder,Decoder})(
    x::AbstractVecOrMat{Float32}; latent::Bool=false
)
    # Run input through encoder to get encoded representation
    z = ae.encoder(x)

    # Run encoded representation through decoder
    x_reconstructed = ae.decoder(z)

    if latent
        return Dict(:z => z, :reconstructed => x_reconstructed)
    else
        return x_reconstructed
    end # if
end # function

# ==============================================================================

@doc raw"""
    mse_loss(ae::AE, 
             x::AbstractVector{Float32}; 
             regularization::Union{Function, Nothing}=nothing, 
             reg_strength::Float32=1.0f0)

Calculate the loss for an autoencoder (AE) by computing the mean squared error
(MSE) reconstruction loss and a possible regularization term.

The AE loss is given by: loss = MSE(x, x̂) + reg_strength × reg_term

Where:
- x is the input vector.
- x̂ is the reconstructed output from the AE.
- reg_strength × reg_term is an optional regularization term.

# Arguments
- `ae::AE`: An AE model.
- `x::AbstractVector{Float32}`: Input vector.

# Optional Keyword Arguments
- `regularization::Union{Function, Nothing}=nothing`: A function that computes
  the regularization term based on the AE outputs. If provided, it should return
  a Float32. If not, no regularization will be applied.
- `reg_strength::Float32=1.0f0`: The strength of the regularization term.

# Returns
- `loss::Float32`: The computed average AE loss value for the given input `x`,
  including possible regularization terms.

# Notes
- Ensure that the dimensionality of the input data `x` aligns with the encoder's
  expected input in the AE.
- For batch processing or evaluating an entire dataset, use:
  `sum(mse_loss.(Ref(ae), eachcol(x)))`.
"""
function mse_loss(
    ae::AE,
    x::AbstractVector{Float32};
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
)
    # Run input through the AE to obtain the reconstruction
    x̂ = ae(x)

    # Compute the MSE loss
    mse = Flux.mse(x̂, x)

    # Initialize total loss with MSE
    total_loss = mse

    # Add regularization term if provided
    if regularization !== nothing
        # Compute regularization term
        reg_term = regularization(ae, x)
        # Add regularization to total loss
        total_loss += reg_strength * reg_term
    end # if

    return total_loss
end # function

@doc raw"""
    mse_loss(ae::AE, 
             x_in::AbstractVector{Float32}, 
             x_out::AbstractVector{Float32};
             regularization::Union{Function, Nothing}=nothing, 
             reg_strength::Float32=1.0f0)

Calculate the mean squared error (MSE) loss for an autoencoder (AE) using
separate input and target output vectors.

The AE loss is computed as: loss = MSE(x_out, x̂) + reg_strength × reg_term

Where:
- x_out is the target output vector.
- x̂ is the reconstructed output from the AE given x_in as input.
- reg_strength × reg_term is an optional regularization term.

# Arguments
- `ae::AE`: An AE model.
- `x_in::AbstractVector{Float32}`: Input vector to the AE encoder.
- `x_out::AbstractVector{Float32}`: Target output vector to compute the
  reconstruction error.

# Optional Keyword Arguments
- `regularization::Union{Function, Nothing}=nothing`: A function that computes
  the regularization term based on the AE outputs. Should return a Float32.
- `reg_strength::Float32=1.0f0`: The strength of the regularization term.

# Returns
- `Float32`: The computed loss value between the target `x_out` and its
  reconstructed counterpart from `x_in`, including possible regularization
  terms.

# Note
Ensure that the input data `x_in` matches the expected input dimensionality for
the encoder in the AE.
"""
function mse_loss(
    ae::AE,
    x_in::AbstractVector{Float32},
    x_out::AbstractVector{Float32};
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
)
    # Run input through the AE to obtain the reconstruction
    x̂ = ae(x_in)

    # Compute the MSE loss between the target output and the reconstructed output
    mse = Flux.mse(x̂, x_out)

    # Initialize total loss with MSE
    total_loss = mse

    # Add regularization term if provided
    if regularization !== nothing
        # Compute regularization term
        reg_term = regularization(ae, x_in, x_out)
        # Add regularization to total loss
        total_loss += reg_strength * reg_term
    end

    return total_loss
end # function

# ==============================================================================

@doc raw"""
    `train!(ae, x, opt; loss_function, loss_kwargs...)`

Customized training function to update parameters of an autoencoder given a
specified loss function.

# Arguments
- `ae::AE{<:AbstractDeterministicEncoder,<:AbstractDeterministicDecoder}`: A
  struct containing the elements of an autoencoder.
- `x::AbstractVector{Float32}`: Input data on which the autoencoder will be
  trained.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments
- `loss_function::Function`: The loss function used for training. It should
  accept the autoencoder model and input data `x`, and return a loss value.
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Additional arguments for the
  loss function.

# Description
Trains the autoencoder by:
1. Computing the gradient of the loss with respect to the autoencoder
   parameters.
2. Updating the autoencoder parameters using the optimizer.

# Examples
```julia
opt = Flux.setup(Optax.adam(1e-3), ae)
for x in dataloader
    train!(ae, x, opt; loss_function)
end
```
"""
function train!(
    ae::AE{<:AbstractDeterministicEncoder,<:AbstractDeterministicDecoder},
    x::AbstractVector{Float32},
    opt::NamedTuple;
    loss_function::Function=mse_loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict()
)
    # Compute AE gradient
    ∇loss_ = Flux.gradient(params(ae)) do
        loss_function(ae(x), x; loss_kwargs...)
    end # do block
    # Update parameters
    Flux.Optimisers.update!(opt, ae, ∇loss_[1])
end # function

@doc raw"""
train!(ae, x, opt; loss_function, loss_kwargs, average)

Customized training function to update parameters of an autoencoder when
provided with matrix data, where each column in x_in and x_out represents an
individual sample.

# Arguments

- `ae::AE{<:AbstractDeterministicEncoder,<:AbstractDeterministicDecoder}`: A
  struct containing the elements of an autoencoder.
- `x::AbstractMatrix{Float32}`: Matrix of input data samples on which to
  evaluate the loss function. Each column represents an individual sample.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments

- `loss_function::Function=loss`: The loss function used for training. It should
  accept the autoencoder model, input data `x_in`, output data `x_out`, and
  keyword arguments in that order.
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Arguments for the loss
  function. These might include parameters like `reg_strength` if a
  regularization function is included.
- `average::Bool=true`: If true, computes and averages the gradient for all
  samples in x_in and x_out before updating parameters. If false, updates
  parameters after computing the gradient for each sample.

# Description

Trains the autoencoder on matrix data by:

1. Computing the gradient of the loss w.r.t the autoencoder parameters, either
for each sample individually or averaged across all samples.
2. Updating the autoencoder parameters using the optimizer.

# Examples
```
opt = Flux.setup(Optax.adam(1e-3), ae)
for x in dataloader
    train!(ae, x, opt; loss_function, average=true)
end
```
"""
function train!(
    ae::AE{<:AbstractDeterministicEncoder,<:AbstractDeterministicDecoder},
    x::AbstractMatrix{Float32},
    opt::NamedTuple;
    loss_function::Function=mse_loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    average=true
)
    # Decide on training approach based on 'average'
    if average
        # Compute the averaged gradient across all samples
        ∇loss_ = Flux.gradient(ae) do ae_model
            StatsBase.mean(
                loss_function.(Ref(ae_model), eachcol(x); loss_kwargs...)
            )
        end # do block
        # Update parameters using the optimizer
        Flux.Optimisers.update!(opt, ae, ∇loss_[1])
    else
        foreach(
            col -> train!(ae, col, opt; loss_function, loss_kwargs...),
            eachcol(x)
        )
    end # if
end # function


# ==============================================================================

@doc raw"""
    `train!(ae, x_in, x_out, opt; loss_function, loss_kwargs...)`

Customized training function to update parameters of an autoencoder given a
specified loss function, where the loss is calculated between `x_in` and
`x_out`.

# Arguments
- `ae::AE{<:AbstractDeterministicEncoder,<:AbstractDeterministicDecoder}`: A struct containing the
  elements of an autoencoder.
- `x_in::AbstractVector{Float32}`: Input data on which to evaluate the loss
  function. Columns represent individual samples.
- `x_out::AbstractVector{Float32}`: Output data to which the model's output will
  be compared.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments
- `loss_function::Function=loss`: The loss function used for training. It should
  accept the autoencoder model, input data `x_in`, output data `x_out`, and
  keyword arguments in that order.
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Arguments for the loss
  function. These might include parameters like `reg_strength` if a
  regularization function is included.

# Description
Trains the autoencoder by:
1. Computing the gradient of the loss w.r.t the autoencoder parameters.
2. Updating the autoencoder parameters using the optimizer.

# Examples
```julia
opt = Flux.setup(Optax.adam(1e-3), ae)
for (x_in, x_out) in dataloader
    train!(ae, x_in, x_out, opt; loss_function)
end
```
"""
function train!(
    ae::AE{<:AbstractDeterministicEncoder,<:AbstractDeterministicDecoder},
    x_in::AbstractVector{Float32},
    x_out::AbstractVector{Float32},
    opt::NamedTuple;
    loss_function::Function=mse_loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict()
)
    # Compute AE gradient
    ∇loss_ = Flux.gradient(params(ae)) do
        loss_function(ae, x_in, x_out; loss_kwargs...)
    end
    # Update parameters
    Flux.Optimisers.update!(opt, vae, ∇loss_[1])
end

@doc raw"""
train!(ae, x_in, x_out, opt; loss_function, loss_kwargs, average)

Customized training function to update parameters of an autoencoder when
provided with matrix data, where each column in x_in and x_out represents an
individual sample.

# Arguments

- `ae::AE{<:AbstractDeterministicEncoder,<:AbstractDeterministicDecoder}`: A
  struct containing the elements of an autoencoder.
- `x_in::AbstractMatrix{Float32}`: Matrix of input data samples on which to
  evaluate the loss function. Each column represents an individual sample.
- `x_out::AbstractMatrix{Float32}`: Matrix of target output data samples. Each
  column represents the corresponding output for an individual sample in `x_in`.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments

- `loss_function::Function=loss`: The loss function used for training. It should
  accept the autoencoder model, input data `x_in`, output data `x_out`, and
  keyword arguments in that order.
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Arguments for the loss
  function. These might include parameters like `reg_strength` if a
  regularization function is included.
- `average::Bool=true`: If true, computes and averages the gradient for all
  samples in x_in and x_out before updating parameters. If false, updates
  parameters after computing the gradient for each sample.

# Description

Trains the autoencoder on matrix data by:

1. Computing the gradient of the loss w.r.t the autoencoder parameters, either
for each sample individually or averaged across all samples.
2. Updating the autoencoder parameters using the optimizer.

# Examples
```
opt = Flux.setup(Optax.adam(1e-3), ae)
for (x_in, x_out) in dataloader
    train!(ae, x_in, x_out, opt; loss_function)
end
```
"""
function train!(
    ae::AE{<:AbstractDeterministicEncoder,<:AbstractDeterministicDecoder},
    x_in::AbstractMatrix{Float32},
    x_out::AbstractMatrix{Float32},
    opt::NamedTuple;
    loss_function::Function=mse_loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    average=true
)
    # Decide on training approach based on 'average'
    if average
        # Compute the averaged gradient across all samples
        ∇loss_ = Flux.gradient(ae) do ae_model
            StatsBase.mean(
                loss_function.(
                    Ref(ae_model),
                    eachcol(x_in),
                    eachcol(x_out);
                    loss_kwargs...
                )
            )
        end
        # Update parameters using the optimizer
        Flux.Optimisers.update!(opt, ae, ∇loss_[1])
    else
        foreach(
            (col_in, col_out) -> train!(
                ae, col_in, col_out, opt; loss_function, loss_kwargs...
            ), zip(eachcol(x_in), eachcol(x_out)
            )
        )
    end # for
end # function

# ==============================================================================

@doc raw"""
    SimpleAE

`mutable struct` representing a basic autoencoder (AE) specifically designed for
the `SimpleChains.jl` package.

# Fields
- `AE::SimpleChains.SimpleChain`: The autoencoder model defined as a sequence of
  layers and operations using the `SimpleChain` construct from the
  `SimpleChains.jl` package. This chain encompasses both the encoder and decoder
  parts of the AE.
- `param::DenseArray{Float32}`: An array of model parameters, presumably to aid
  in training or other operations. The exact nature and organization of these
  parameters depend on the AE's design and requirements.

# Description
This struct encapsulates the components of a basic autoencoder model suitable
for use with the `SimpleChains.jl` framework. While the autoencoder's
architecture is defined in the `AE` field, associated parameters, possibly used
for optimization routines or other tasks, are stored in the `param` field.

# Note
Users should ensure compatibility between the `SimpleAE` structure and
functions/methods provided by `SimpleChains.jl` when integrating or extending
this structure.
"""
mutable struct SimpleAE <: AbstractDeterministicAutoEncoder
    AE::SimpleChains.SimpleChain
    param::DenseArray{Float32}
end # struct


##

@doc raw"""
    SimpleAE(
        n_input::Int, 
        n_latent::Int, 
        latent_activation::Function,
        output_activation::Function,
        encoder::Vector{Int}, 
        encoder_activation::Vector{Function},
        decoder::Vector{Int}, 
        decoder_activation::Vector{Function}
    ) -> SimpleAE

Constructs and initializes a `SimpleAE` autoencoder using components from the
`SimpleChains.jl` package.

# Arguments
- `n_input::Int`: Dimensionality of the input space.
- `n_latent::Int`: Dimensionality of the latent space.
- `latent_activation::Function`: Activation function applied at the latent space
  layer.
- `output_activation::Function`: Activation function applied at the output
  layer.
- `encoder::Vector{Int}`: Specifies the dimensions of each hidden layer in the
  encoder network.
- `encoder_activation::Vector{Function}`: Activation functions for each hidden
  layer in the encoder. The number of functions should match the length of
  `encoder`.
- `decoder::Vector{Int}`: Specifies the dimensions of each hidden layer in the
  decoder network.
- `decoder_activation::Vector{Function}`: Activation functions for each hidden
  layer in the decoder. The number of functions should match the length of
  `decoder`.

# Returns
- A `SimpleAE` instance representing the constructed autoencoder.

# Description
The function creates a `SimpleAE` autoencoder by sequentially combining the
provided encoder and decoder layers, while ensuring the correct matching of
activation functions for each layer. The resultant autoencoder is structured to
operate seamlessly within the `SimpleChains.jl` ecosystem.

# Example
```julia
ae_model = SimpleAE(
    784, 32, relu, sigmoid, [512, 256], [relu, relu], [256, 512], [relu, relu]
)
```

# Note
Ensure that the number of dimensions in encoder matches the number of activation
functions in encoder_activation, and similarly for decoder and
decoder_activation.
"""
function SimpleAE(
    n_input::Int,
    n_latent::Int,
    latent_activation::Function,
    output_activation::Function,
    encoder::Vector{<:Int},
    encoder_activation::Vector{<:Function},
    decoder::Vector{<:Int},
    decoder_activation::Vector{<:Function}
)
    # Check there's enough activation functions for all layers
    if (length(encoder_activation) != length(encoder)) |
       (length(decoder_activation) != length(decoder))
        error("Each layer needs exactly one activation function")
    end # if

    # Initialize list with encoder layers
    Encoder = Array{SimpleChains.TurboDense}(undef, length(encoder) + 1)

    # Loop through layers
    for i = 1:length(encoder)
        # Add layer
        Encoder[i] = SimpleChains.TurboDense(encoder[i], encoder_activation[i])
    end # for
    # Add last layer from encoder to latent space with activation
    Encoder[end] = SimpleChains.TurboDense(n_latent, latent_activation)

    # Initialize list with decoder layers
    Decoder = Array{SimpleChains.TurboDense}(undef, length(decoder))

    # Loop through layers
    for i = 1:(length(decoder))
        # Add layer
        Decoder[i] = SimpleChains.TurboDense(decoder[i], decoder_activation[i])
    end # for

    # Define autoencoder
    ae = SimpleChains.SimpleChain(
        SimpleChains.static(n_input),
        Encoder...,
        Decoder...,
        SimpleChains.TurboDense(output_activation, n_input)
    )
    # Initialize parameters
    param = SimpleChains.init_params(ae)

    return SimpleAE(ae, param)
end # function

##

@doc raw"""
    `simple_to_flux(simple_ae::SimpleAE, ae::AE)`

Function to transfer the parameters from a `SimpleChains.jl` trained network to
a `Flux.jl` network with the same architecture for downstream manipulation.

NOTE: This function is agnostic to the activation functions in the
`SimpleChains.jl` network from where `param` was extracted. Therefore, for this
transfer to make sense, you must ensure that both networks have the same
architecture!

# Arguments
- `simple_ae::SimpleAE`: A `SimpleAE` instance representing an autoencoder built
  using `SimpleChains.jl`.
- `ae::AE`: An `AE` instance representing an autoencoder built using `Flux.jl`.

# Returns
- `AE`: The `Flux.jl` autoencoder (`AE`) with the modified parameters dictated
  by the `SimpleAE`.
"""
function simple_to_flux(simple_ae::SimpleAE, ae::AE)
    # Extract parameters from the SimpleChains.jl autoencoder
    param = simple_ae.param

    # Concatenate the encoder and decoder chains from the Flux.jl autoencoder
    # into a single chain for easier parameter extraction
    fluxchain = Flux.Chain(ae.encoder..., ae.decoder...)

    # Extract individual layer parameters from the Flux.jl chain. This creates a
    # nested list where each entry is a list of parameters for a specific layer
    # (usually [weights, biases])
    param_flux = [collect(Flux.params(layer)) for layer in fluxchain]

    # Deepcopy the extracted parameters to create a container for transferred 
    # parameters. This avoids altering the original parameters.
    param_transfer = deepcopy(param_flux)

    # Initialize a counter to keep track of the index position in the
    # SimpleChains parameter vector
    idx = 1

    # Loop through each layer of parameters in the Flux.jl model
    for (i, layer) in enumerate(param_flux)
        # Within each layer, loop through the set of parameters (usually weights 
        # and biases)
        for (j, p) in enumerate(layer)
            # Create a container with the same shape as the current parameter
            # set
            par = similar(p)
            # Transfer the parameters from the SimpleChains model into this
            # container
            par .= param[idx:(idx+length(par)-1)]
            # Store the reshaped transferred parameters in the appropriate slot
            # in the parameter transfer container
            param_transfer[i][j] = reshape(par, size(p))
            # Update the index counter to move to the next set of parameters in
            # the SimpleChains parameter vector
            idx += length(par)
        end # for
    end # for

    # Convert the list of transferred parameters into a Flux.Params object for 
    # compatibility with Flux.jl
    param_transfer = Flux.Params(param_transfer)

    # Create a container to hold the Flux.Dense layers that will be constructed
    # with the transferred parameters
    layers_transfer = Array{Flux.Dense}(undef, length(param_transfer))

    # Loop through each layer in the original Flux.jl Chain to construct new
    # layers using the transferred parameters
    for (i, layer) in enumerate(fluxchain)
        if layer.bias == false
            # Construct a new Flux.Dense layer with no bias using the
            # transferred parameters and the same activation function as the
            # original layer
            layers_transfer[i] = Flux.Dense(
                param_transfer[i]..., layer.bias, layer.σ
            )
        else
            # Construct a new Flux.Dense layer with bias using the transferred
            # parameters and the same activation function as the original layer
            layers_transfer[i] = Flux.Dense(param_transfer[i]..., layer.σ)
        end # if
    end # for

    # Return a new AE autoencoder constructed using the Encoder and Decoder
    # structs and the layers with transferred parameters
    return AE(
        Encoder(layers_transfer[1:length(ae.encoder)]...),
        Decoder(layers_transfer[length(ae.encoder)+1:end]...)
    )
end # function
