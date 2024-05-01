# Import ML libraries
import Flux

##

# Import Abstract Types
using ..AutoEncoderToolkit: AbstractAutoEncoder, AbstractDeterministicAutoEncoder,
    AbstractDeterministicEncoder, AbstractDeterministicDecoder

# Import Concrete Types
using ..AutoEncoderToolkit: Encoder, Decoder

##

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
    (ae::AE{Encoder, Decoder})(x::AbstractArray; latent::Bool=false)

Processes the input data `x` through the autoencoder (AE) that consists of an
encoder and a decoder.

# Arguments
- `x::AbstractVecOrMat{Float32}`: The data to be decoded. This can be a vector
  or a matrix where each column represents a separate sample.

# Optional Keyword Arguments
- `latent::Bool`: If set to `true`, returns a dictionary containing the latent
  representation alongside the reconstructed data. Defaults to `false`.

# Returns
- If `latent=false`: A `Namedtuple` with key `:decoder` that contains the
  reconstructed data after processing through the encoder and decoder.
- If `latent=true`: A `Namedtuple `with keys `:encoder`, and `:decoder`,
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
function (ae::AE)(
    x::AbstractArray; latent::Bool=false
)
    # Run input through encoder to get encoded representation
    z = ae.encoder(x)

    # Run encoded representation through decoder
    x_reconstructed = ae.decoder(z)

    if latent
        return (encoder=z, decoder=x_reconstructed,)
    else
        return (decoder=x_reconstructed,)
    end # if
end # function

# ==============================================================================

@doc raw"""
    mse_loss(ae::AE, 
             x::AbstractArray; 
             regularization::Union{Function, Nothing}=nothing, 
             reg_strength::Float32=1.0f0
    )

Calculate the loss for an autoencoder (AE) by computing the mean squared error
(MSE) reconstruction loss and a possible regularization term.

The AE loss is given by: loss = MSE(x, x̂) + reg_strength × reg_term

Where:
- x is the input Array.
- x̂ is the reconstructed output from the AE.
- reg_strength × reg_term is an optional regularization term.

# Arguments
- `ae::AE`: An AE model.
- `x::AbstractArray`: Input data.

# Optional Keyword Arguments
- `reg_function::Union{Function, Nothing}=nothing`: A function that computes the
  regularization term based on the ae outputs. Should return a Float32. This
  function must take as input the ae outputs and the keyword arguments provided
  in `reg_kwargs`.
- `reg_kwargs::Union{NamedTuple,Dict}=Dict()`: Keyword arguments to pass to the
  regularization function.
- `reg_strength::Number=1.0f0`: The strength of the regularization term.

# Returns
- The computed average AE loss value for the given input `x`, including possible
  regularization terms.

# Notes
Ensure that the dimensionality of the input data `x` aligns with the encoder's
expected input in the AE.
"""
function mse_loss(
    ae::AE,
    x::AbstractArray;
    reg_function::Union{Function,Nothing}=nothing,
    reg_kwargs::Union{NamedTuple,Dict}=Dict(),
    reg_strength::Number=1.0f0
)
    # Run input through the AE to obtain the reconstruction
    x̂ = ae(x).decoder

    # Compute the MSE loss
    mse = Flux.mse(x̂, x)

    # Compute regularization term if regularization function is provided
    reg_term = (reg_function !== nothing) ?
               reg_function(ae_outputs; reg_kwargs...) : 0.0f0

    # Return loss
    return mse + reg_strength * reg_term
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    mse_loss(ae::AE, 
             x_in::AbstractArray, 
             x_out::AbstractArray;
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
- `x_in::AbstractArray`: Input vector to the AE encoder.
- `x_out::AbstractArray`: Target output vector to compute the reconstruction
  error.

# Optional Keyword Arguments
- `reg_function::Union{Function, Nothing}=nothing`: A function that computes the
  regularization term based on the ae outputs. Should return a Float32. This
  function must take as input the ae outputs and the keyword arguments provided
  in `reg_kwargs`.
- `reg_kwargs::Union{NamedTuple,Dict}=Dict()`: Keyword arguments to pass to the
  regularization function.
- `reg_strength::Number=1.0f0`: The strength of the regularization term.

# Returns
- The computed loss value between the target `x_out` and its reconstructed
  counterpart from `x_in`, including possible regularization terms.

# Note
Ensure that the input data `x_in` matches the expected input dimensionality for
the encoder in the AE.
"""
function mse_loss(
    ae::AE,
    x_in::AbstractArray,
    x_out::AbstractArray;
    reg_function::Union{Function,Nothing}=nothing,
    reg_kwargs::Union{NamedTuple,Dict}=Dict(),
    reg_strength::Number=1.0f0
)
    # Run input through the AE to obtain the reconstruction
    x̂ = ae(x_in).decoder

    # Compute the MSE loss between the target output and the reconstructed output
    mse = Flux.mse(x̂, x_out)

    # Compute regularization term if regularization function is provided
    reg_term = (reg_function !== nothing) ?
               reg_function(ae_outputs; reg_kwargs...) : 0.0f0

    # Return loss
    return mse + reg_strength * reg_term
end # function

# ==============================================================================

@doc raw"""
    `train!(ae, x, opt; loss_function, loss_kwargs...)`

Customized training function to update parameters of an autoencoder given a
specified loss function.

# Arguments
- `ae::AE`: A struct containing the elements of an autoencoder.
- `x::AbstractArray`: Input data on which the autoencoder will be trained.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments
- `loss_function::Function`: The loss function used for training. It should
  accept the autoencoder model and input data `x`, and return a loss value.
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Additional arguments for the
  loss function.
- `verbose::Bool=false`: If true, the loss value will be printed during
  training.
- `loss_return::Bool=false`: If true, the loss value will be returned after
  training.

# Description
Trains the autoencoder by:
1. Computing the gradient of the loss with respect to the autoencoder
   parameters.
2. Updating the autoencoder parameters using the optimizer.
"""
function train!(
    ae::AE,
    x::AbstractArray,
    opt::NamedTuple;
    loss_function::Function=mse_loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    verbose::Bool=false,
    loss_return::Bool=false
)
    # Compute ae gradient
    L, ∇L = Flux.withgradient(ae) do ae_model
        loss_function(ae_model, x; loss_kwargs...)
    end # do block

    # Update parameters
    Flux.Optimisers.update!(opt, ae, ∇L[1])

    # Check if loss should be returned
    if loss_return
        return L
    end # if

    # Check if loss should be printed
    if verbose
        println("Loss: ", L)
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    train!(ae, x_in, x_out, opt; loss_function, loss_kwargs...)

Customized training function to update parameters of an autoencoder given a
specified loss function.

# Arguments
- `ae::AE`: A struct containing the elements of an autoencoder.
- `x_in::AbstractArray`: Input data on which the autoencoder will be trained.
- `x_out::AbstractArray`: Target output data for the autoencoder.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments
- `loss_function::Function`: The loss function used for training. It should
  accept the autoencoder model and input data `x`, and return a loss value.
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Additional arguments for the
  loss function.
- `verbose::Bool=false`: If true, the loss value will be printed during
  training.
- `loss_return::Bool=false`: If true, the loss value will be returned after
  training.

# Description
Trains the autoencoder by:
1. Computing the gradient of the loss with respect to the autoencoder
   parameters.
2. Updating the autoencoder parameters using the optimizer.
"""
function train!(
    ae::AE,
    x_in::AbstractArray,
    x_out::AbstractArray,
    opt::NamedTuple;
    loss_function::Function=mse_loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    verbose::Bool=false,
    loss_return::Bool=false
)
    # Compute ae gradient
    L, ∇L = Flux.withgradient(ae) do ae_model
        loss_function(ae_model, x_in, x_out; loss_kwargs...)
    end # do block

    # Update parameters
    Flux.Optimisers.update!(opt, ae, ∇L[1])

    # Check if loss should be returned
    if loss_return
        return L
    end # if

    # Check if loss should be printed
    if verbose
        println("Loss: ", L)
    end # if
end # function