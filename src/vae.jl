# Import ML libraries
import Flux
import Zygote
import CUDA

# Import basic math
import StatsBase
import Distributions

# Import Abstract Types
using ..AutoEncode: AbstractVariationalAutoEncoder,
    AbstractVariationalEncoder, AbstractGaussianEncoder,
    AbstractGaussianLogEncoder,
    AbstractVariationalDecoder, AbstractGaussianDecoder,
    AbstractGaussianLogDecoder, AbstractGaussianLinearDecoder,

    # Import Concrete Encoder Types
    using..AutoEncode:JointLogEncoder

# Import Concrete Decoder Types
using ..AutoEncode: BernoulliDecoder, SimpleDecoder,
    JointLogDecoder, SplitLogDecoder,
    JointDecoder, SplitDecoder

# Import functions
using ..AutoEncode: encoder_kl, encoder_logposterior, decoder_loglikelihood

# Export functions to use elsewhere
export reparameterize

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Kingma, D. P. & Welling, M. Auto-Encoding Variational Bayes. Preprint at
#    http://arxiv.org/abs/1312.6114 (2014).
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ==============================================================================
# Reparametrization trick
# ==============================================================================

@doc raw"""
        reparameterize(µ, σ; log::Bool=true)

Reparameterize the latent space using the given mean (`µ`) and (log)standard
deviation (`σ` or `logσ`), employing the reparameterization trick. This function
helps in sampling from the latent space in variational autoencoders (or similar
models) while keeping the gradient flow intact.

# Arguments
- `µ::AbstractVecOrMat{Float32}`: The mean of the latent space. If it is a
  vector, it represents the mean for a single data point. If it is a matrix,
  each column corresponds to the mean for a specific data point, and each row
  corresponds to a dimension of the latent space.
- `σ::AbstractVecOrMat{Float32}`: The (log )standard deviation of the latent
  space. Like `µ`, if it's a vector, it represents the (log) standard deviation
  for a single data point. If a matrix, each column corresponds to the (log)
  standard deviation for a specific data point.

# Optional Keyword Arguments
- `log::Bool=true`: Boolean indicating whether the provided standard deviation
  is in log scale or not. If `true` (default), then `σ = exp(logσ)` is computed.

# Returns
An array containing samples from the reparameterized latent space, obtained by
applying the reparameterization trick on the provided mean and log standard
deviation.

# Description
This function employs the reparameterization trick to sample from the latent
space without breaking the gradient flow. The trick involves expressing the
random variable as a deterministic variable transformed by a standard random
variable, allowing for efficient backpropagation through stochastic nodes.

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
    µ::AbstractVecOrMat{T},
    σ::AbstractVecOrMat{T};
    log::Bool=true
) where {T<:Number}
    # Sample random Gaussian number
    r = Zygote.ignore() do
        randn(T, size(µ)...)
    end
    # Check if logσ is provided
    if log
        # Sample random latent variable point estimates given the mean and log
        # standard deviation
        return µ .+ r .* exp.(σ)
    else
        # Sample random latent variable point estimates given the mean and
        # standard deviation
        return µ .+ r .* σ
    end # if
end # function

# -----------------------------------------------------------------------------

@doc raw"""
        reparameterize(µ, σ; log::Bool=true)

Reparameterize the latent space using the given mean (`µ`) and (log)standard
deviation (`σ` or `logσ`), employing the reparameterization trick. This function
helps in sampling from the latent space in variational autoencoders (or similar
models) while keeping the gradient flow intact.

# Arguments
- `µ::CuVecOrMat{Float32}`: The mean of the latent space. If it is a
  vector, it represents the mean for a single data point. If it is a matrix,
  each column corresponds to the mean for a specific data point, and each row
  corresponds to a dimension of the latent space.
- `σ::CuVecOrMat{Float32}`: The (log )standard deviation of the latent
  space. Like `µ`, if it's a vector, it represents the (log) standard deviation
  for a single data point. If a matrix, each column corresponds to the (log)
  standard deviation for a specific data point.

# Optional Keyword Arguments
- `log::Bool=true`: Boolean indicating whether the provided standard deviation
  is in log scale or not. If `true` (default), then `σ = exp(logσ)` is computed.

# Returns
An array containing samples from the reparameterized latent space, obtained by
applying the reparameterization trick on the provided mean and log standard
deviation.

# Description
This function employs the reparameterization trick to sample from the latent
space without breaking the gradient flow. The trick involves expressing the
random variable as a deterministic variable transformed by a standard random
variable, allowing for efficient backpropagation through stochastic nodes.

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
    µ::CUDA.CuVecOrMat{T},
    σ::CUDA.CuVecOrMat{T};
    log::Bool=true
) where {T<:Number}
    # Sample random Gaussian number
    r = Zygote.ignore() do
        CUDA.randn(T, size(µ)...)
    end
    # Check if logσ is provided
    if log
        # Sample random latent variable point estimates given the mean and log
        # standard deviation
        return µ .+ r .* exp.(σ)
    else
        # Sample random latent variable point estimates given the mean and
        # standard deviation
        return µ .+ r .* σ
    end # if
end # function

# -----------------------------------------------------------------------------

@doc raw"""
        reparameterize(
                encoder::AbstractGaussianLogEncoder,
                encoder_outputs::NamedTuple
        )

Reparameterize the latent space using the outputs of a Gaussian log encoder.
This function helps in sampling from the latent space in variational
autoencoders (or similar models) while keeping the gradient flow intact.

# Arguments
- `encoder::AbstractGaussianLogEncoder`: The Gaussian log encoder. This argument
  is not used in the function itself, but is used to infer which method to call.
- `encoder_outputs::NamedTuple`: The outputs of the encoder. This should be a
  NamedTuple with keys `µ` and `logσ`, representing the mean and log standard
  deviation of the latent space.

# Returns
An array containing samples from the reparameterized latent space, obtained by
applying the reparameterization trick on the provided mean and log standard
deviation.

# Description
This function employs the reparameterization trick to sample from the latent
space without breaking the gradient flow. The trick involves expressing the
random variable as a deterministic variable transformed by a standard random
variable, allowing for efficient backpropagation through stochastic nodes.

# Notes
Ensure that the dimensions of µ and logσ match.

# Citation
Kingma, D. P. & Welling, M. Auto-Encoding Variational Bayes. Preprint at
http://arxiv.org/abs/1312.6114 (2014).
"""
function reparameterize(
    encoder::AbstractGaussianLogEncoder,
    encoder_outputs::NamedTuple
)
    # Call reparameterize function with encoder outputs
    reparameterize(encoder_outputs.µ, encoder_outputs.logσ; log=true)
end # function

# ==============================================================================
# `struct VAE{E<:AbstractVariationalEncoder, D<:AbstractVariationalDecoder}`
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
struct VAE{
    E<:AbstractVariationalEncoder,
    D<:AbstractVariationalDecoder
} <: AbstractVariationalAutoEncoder
    encoder::E
    decoder::D
end # struct

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor VAE

@doc raw"""
        (vae::VAE)(x::AbstractArray{<:Number}; latent::Bool=false)

Perform the forward pass of a Variational Autoencoder (VAE).

This function takes as input a VAE and a vector or matrix of input data `x`. It
first runs the input through the encoder to obtain the mean and log standard
deviation of the latent variables. It then uses the reparameterization trick to
sample from the latent distribution. Finally, it runs the latent sample through
the decoder to obtain the output.

# Arguments
- `vae::VAE`: The VAE used to encode the input data and decode the latent space.
- `x::AbstractArray{<:Number}`: The input data. If array, the last dimension
  contains each of the samples in a batch.

# Optional Keyword Arguments
- `latent::Bool`: Whether to return the latent variables along with the decoder
  output. If `true`, the function returns a tuple containing the encoder
  outputs, the latent sample, and the decoder outputs. If `false`, the function
  only returns the decoder outputs. Defaults to `false`.  

# Returns
- If `latent` is `true`, returns a tuple containing:
    - `encoder`: The outputs of the encoder.
    - `z`: The latent sample.
    - `decoder`: The outputs of the decoder.
- If `latent` is `false`, returns the outputs of the decoder.

# Example
```julia
# Define a VAE
vae = VAE(
    encoder=Flux.Chain(Flux.Dense(784, 400, relu), Flux.Dense(400, 20)),
    decoder=Flux.Chain(Flux.Dense(20, 400, relu), Flux.Dense(400, 784))
)

# Define input data
x = rand(Float32, 784)

# Perform the forward pass
outputs = vae(x, latent=true)
```
"""
function (vae::VAE)(x::AbstractArray{T}; latent::Bool=false) where {T<:Number}
    # Run input through encoder to obtain mean and log std
    encoder_outputs = vae.encoder(x)

    # Run reparametrization trick
    z_sample = reparameterize(vae.encoder, encoder_outputs)

    # Check if latent variables should be returned
    if latent
        # Run latent sample through decoder and collect outpus in dictionary
        return (
            encoder=encoder_outputs,
            z=z_sample,
            decoder=vae.decoder(z_sample)
        )
    else
        # Run latent sample through decoder
        return vae.decoder(z_sample)
    end # if
end # function

# ==============================================================================
# VAE loss functions
# ==============================================================================

@doc raw"""
    loss(
        vae::VAE,
        x::AbstractArray{T};
        β::T=1.0f0,
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        kl_divergence::Function=encoder_kl,
        reg_function::Union{Function,Nothing}=nothing,
        reg_kwargs::Union{NamedTuple,Dict}=Dict(),
        reg_strength::T=1.0f0
    ) where {T<:Number}

Computes the loss for the variational autoencoder (VAE).

The loss function combines the reconstruction loss with the Kullback-Leibler
(KL) divergence, and possibly a regularization term, defined as:

loss = -⟨logπ(x|z)⟩ + β × Dₖₗ[qᵩ(z|x) || π(z)] + reg_strength × reg_term

Where:
- π(x|z) is a probabilistic decoder: π(x|z) = N(f(z), σ² I̲̲)) - f(z) is the
  function defining the mean of the decoder π(x|z) - qᵩ(z|x) is the approximated
  encoder: qᵩ(z|x) = N(g(x), h(x))
- g(x) and h(x) define the mean and covariance of the encoder respectively.

# Arguments
- `vae::VAE`: A VAE model with encoder and decoder networks.
- `x::AbstractArray{T}`: Input data. The last dimension is taken as having each
  of the samples in a batch.

# Optional Keyword Arguments
- `β::T=1.0f0`: Weighting factor for the KL-divergence term, used for annealing.
- `reconstruction_loglikelihood::Function=decoder_loglikelihood`: A function
  that computes the reconstruction log likelihood.
- `kl_divergence::Function=encoder_kl`: A function that computes the
  Kullback-Leibler divergence between the encoder output and a standard normal.
- `reg_function::Union{Function, Nothing}=nothing`: A function that computes the
  regularization term based on the VAE outputs. Should return a Float32. This
  function must take as input the VAE outputs and the keyword arguments provided
  in `reg_kwargs`.
- `reg_kwargs::Union{NamedTuple,Dict}=Dict()`: Keyword arguments to pass to the
    regularization function.
- `reg_strength::T=1.0f0`: The strength of the regularization term.

# Returns
- `T`: The computed average loss value for the input `x` and its reconstructed
  counterparts, including possible regularization terms.

# Note
- Ensure that the input data `x` matches the expected input dimensionality for
  the encoder in the VAE.
"""
function loss(
    vae::VAE,
    x::AbstractArray{T};
    β::T=1.0f0,
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    kl_divergence::Function=encoder_kl,
    reg_function::Union{Function,Nothing}=nothing,
    reg_kwargs::Union{NamedTuple,Dict}=Dict(),
    reg_strength::T=1.0f0
) where {T<:Number}
    # Forward Pass (run input through reconstruct function)
    vae_output = vae(x; latent=true)

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder averaged over all samples
    log_likelihood = StatsBase.mean(
        reconstruction_loglikelihood(vae.decoder, x, vae_output.decoder)
    )

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_div = StatsBase.mean(
        kl_divergence(vae.encoder, vae_output.encoder)
    )

    # Compute regularization term if regularization function is provided
    reg_term = (reg_function !== nothing) ?
               reg_function(vae_outputs; reg_kwargs...) : 0.0f0

    # Compute average loss function
    return -log_likelihood + β * kl_div + reg_strength * reg_term
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    loss(
        vae::VAE,
        x_in::AbstractArray{T},
        x_out::AbstractArray{T};
        β::T=1.0f0,
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        kl_divergence::Function=encoder_kl,
        reg_function::Union{Function,Nothing}=nothing,
        reg_kwargs::Union{NamedTuple,Dict}=Dict(),
        reg_strength::T=1.0f0
    ) where {T<:Number}

Computes the loss for the variational autoencoder (VAE).

The loss function combines the reconstruction loss with the Kullback-Leibler
(KL) divergence and possibly a regularization term, defined as:

loss = -⟨logπ(x_out|z)⟩ + β × Dₖₗ[qᵩ(z|x_in) || π(z)] + reg_strength × reg_term

Where:
- π(x_out|z) is a probabilistic decoder: π(x_out|z) = N(f(z), σ² I̲̲)) - f(z) is
the function defining the mean of the decoder π(x_out|z) - qᵩ(z|x_in) is the
approximated encoder: qᵩ(z|x_in) = N(g(x_in), h(x_in))
- g(x_in) and h(x_in) define the mean and covariance of the encoder
  respectively.

# Arguments
- `vae::VAE`: A VAE model with encoder and decoder networks.
- `x_in::AbstractArray{T}`: Input data to the VAE encoder. The last dimension is
  taken as having each of the samples in a batch.
- `x_out::AbstractArray{T}`: Target data to compute the reconstruction error.
  The last dimension is taken as having each of the samples in a batch.

# Optional Keyword Arguments
- `β::T=1.0f0`: Weighting factor for the KL-divergence term, used for annealing.
- `reconstruction_loglikelihood::Function=decoder_loglikelihood`: A function
  that computes the reconstruction log likelihood.
- `kl_divergence::Function=encoder_kl`: A function that computes the
  Kullback-Leibler divergence.
- `reg_function::Union{Function, Nothing}=nothing`: A function that computes the
  regularization term based on the VAE outputs. Should return a Float32. This
  function must take as input the VAE outputs and the keyword arguments provided
  in `reg_kwargs`.
- `reg_kwargs::Union{NamedTuple,Dict}=Dict()`: Keyword arguments to pass to the
  regularization function.
- `reg_strength::T=1.0f0`: The strength of the regularization term.

# Returns
- `T`: The computed average loss value for the input `x_in` and its
  reconstructed counterparts `x_out`, including possible regularization terms.

# Note
- Ensure that the input data `x_in` and `x_out` match the expected input
  dimensionality for the encoder in the VAE.
"""
function loss(
    vae::VAE,
    x_in::AbstractArray{T};
    x_out::AbstractArray{T};
    β::T=1.0f0,
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    kl_divergence::Function=encoder_kl,
    reg_function::Union{Function,Nothing}=nothing,
    reg_kwargs::Union{NamedTuple,Dict}=Dict(),
    reg_strength::T=1.0f0
) where {T<:Number}
    # Forward Pass (run input through reconstruct function)
    vae_output = vae(x_in; latent=true)

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder averaged over all samples
    log_likelihood = StatsBase.mean(
        reconstruction_loglikelihood(vae.decoder, x_out, vae_output.decoder)
    )

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_div = StatsBase.mean(
        kl_divergence(vae.encoder, vae_output.encoder)
    )

    # Compute regularization term if regularization function is provided
    reg_term = (reg_function !== nothing) ?
               reg_function(vae_outputs; reg_kwargs...) : 0.0f0

    # Compute average loss function
    return -log_likelihood + β * kl_div + reg_strength * reg_term
end # function

# ==============================================================================
# VAE training functions
# ==============================================================================

@doc raw"""
    train!(vae, x, opt; loss_function, loss_kwargs, verbose)

Customized training function to update parameters of a variational autoencoder
given a specified loss function.

# Arguments
- `vae::VAE`: A struct containing the elements of a variational autoencoder.
- `x::AbstractArray{<:Number}`: Data on which to evaluate the loss function. The
  last dimension is taken as having each of the samples in a batch.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments
- `loss_function::Function=loss`: The loss function used for training. It should
  accept the VAE model, data `x`, and keyword arguments in that order.
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Arguments for the loss
  function. These might include parameters like `σ`, or `β`, depending on the
  specific loss function in use.
- `verbose::Bool=false`: If true, the loss value will be printed during
  training.

# Description
Trains the VAE by:
1. Computing the gradient of the loss w.r.t the VAE parameters.
2. Updating the VAE parameters using the optimizer.

# Examples
```julia
opt = Flux.setup(Optax.adam(1e-3), vae)
for x in dataloader
        train!(vae, x, opt; loss_fn, loss_kwargs=Dict(:β => 1.0f0,), verbose=true)
end
```
"""
function train!(
    vae::VAE,
    x::AbstractArray{<:Number},
    opt::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    verbose::Bool=false,
)
    # Compute VAE gradient
    L, ∇L = Flux.withgradient(vae) do vae_model
        loss_function(vae_model, x; loss_kwargs...)
    end # do block

    # Update parameters
    Flux.Optimisers.update!(opt, vae, ∇L[1])

    # Check if loss should be printed
    if verbose
        println("Loss: ", L)
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
        `train!(vae, x_in, x_out, opt; loss_function, loss_kwargs...)`

Customized training function to update parameters of a variational autoencoder
given a loss function.

# Arguments
- `vae::VAE`: A struct containing the elements of a variational autoencoder.
- `x_in::AbstractArray{T}`: Input data for the loss function. Represents an
  individual sample. The last dimension is taken as having each of the samples
  in a batch.
- `x_out::AbstractArray{T}`: Target output data for the loss function.
  Represents the corresponding output for the `x_in` sample. The last dimension
  is taken as having each of the samples in a batch.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
    initialized using `Flux.Optimisers.update!`.

# Optional Keyword Arguments
- `loss_function::Function=loss`: The loss function used for training. It should
  accept the VAE model, data `x_in`, `x_out`, and keyword arguments in that
  order.  
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Arguments for the loss
  function. These might include parameters like `σ`, or `β`, depending on the
  specific loss function in use.
- `verbose::Bool=false`: Whether to print the loss value after each training
  step.

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
    vae::VAE,
    x_in::AbstractArray{T},
    x_out::AbstractArray{T},
    opt::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    verbose::Bool=false
) where {T<:Number}
    # Compute VAE gradient
    L, ∇L = Flux.withgradient(vae) do vae_model
        loss_function(vae_model, x_in, x_out; loss_kwargs...)
    end # do block

    # Update parameters
    Flux.Optimisers.update!(opt, vae, ∇L[1])

    # Check if loss should be printed
    if verbose
        println("Loss: ", L)
    end # if
end # function