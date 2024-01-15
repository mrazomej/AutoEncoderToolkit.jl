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

using ..AutoEncode: AbstractVariationalAutoEncoder,
    AbstractVariationalEncoder,
    AbstractVariationalDecoder, AbstractGaussianDecoder,
    AbstractGaussianLogDecoder, AbstractGaussianLinearDecoder,
    Float32Array

# Import Concrete Encoder Types
using ..AutoEncode: JointLogEncoder

# Import Concrete Decoder Types
using ..AutoEncode: SimpleDecoder,
    JointLogDecoder, SplitLogDecoder,
    JointDecoder, SplitDecoder

# Export functions to use elsewhere
export reparameterize, reconstruction_gaussian_decoder, kl_gaussian_encoder

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Kingma, D. P. & Welling, M. Auto-Encoding Variational Bayes. Preprint at
#    http://arxiv.org/abs/1312.6114 (2014).
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ==============================================================================
# Reparametrization trick
# ==============================================================================

@doc raw"""
    reparameterize(µ, σ; prior=Distributions.Normal{Float32}(0.0f0, 1.0f0), 
    n_samples=1, log::Bool=true)

Reparameterize the latent space using the given mean (`µ`) and (log) standard
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
- `prior::Distributions.Sampleable`: The prior distribution for the latent
  space. By default, this is a standard normal distribution.
- `n_samples::Int=1`: The number of samples to draw using the reparametrization
  trick.
- `log::Bool=true`: Boolean indicating whether the provided standard deviation
  is in log scale or not. If `true` (default), then `σ = exp(logσ)` is computed.

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
    σ::T;
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
    n_samples::Int=1,
    log::Bool=true
) where {T<:AbstractVecOrMat{Float32}}
    # Check if logσ is provided
    if log
        # Sample result depending on type of prior distribution
        result = if isa(prior, Distributions.UnivariateDistribution)
            # Sample n_samples random latent variable point estimates given the
            # mean and standard deviation
            µ .+ Random.rand(prior, size(µ)..., n_samples) .* exp.(σ)
        elseif isa(prior, Distributions.MultivariateDistribution) &
               isa(µ, AbstractVector)
            # Sample n_samples random latent variable point estimates given the
            # mean and standard deviation
            µ .+ Random.rand(prior, n_samples) .* exp.(σ)
        elseif isa(prior, Distributions.MultivariateDistribution) &
               isa(µ, AbstractMatrix)
            # Sample n_samples random latent variable point estimates given the
            # mean and standard deviation
            µ .+ reshape(
                reduce(hcat, Random.rand(prior, size(µ, 2), n_samples)),
                :, size(µ, 2), n_samples
            ) .* exp.(σ)
        end # if
    else
        # Sample result depending on type of prior distribution
        result = if isa(prior, Distributions.UnivariateDistribution)
            # Sample n_samples random latent variable point estimates given the
            # mean and standard deviation
            µ .+ Random.rand(prior, size(µ)..., n_samples) .* σ
        elseif isa(prior, Distributions.MultivariateDistribution) &
               isa(µ, AbstractVector)
            # Sample n_samples random latent variable point estimates given the
            # mean and standard deviation
            µ .+ Random.rand(prior, n_samples) .* σ
        elseif isa(prior, Distributions.MultivariateDistribution) &
               isa(µ, AbstractMatrix)
            # Sample n_samples random latent variable point estimates given the
            # mean and standard deviation
            µ .+ reshape(
                reduce(hcat, Random.rand(prior, size(µ, 2), n_samples)),
                :, size(µ, 2), n_samples
            ) .* σ

        end # if
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
mutable struct VAE{
    E<:AbstractVariationalEncoder,
    D<:AbstractVariationalDecoder
} <: AbstractVariationalAutoEncoder
    encoder::E
    decoder::D
end # struct

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor VAE

@doc raw"""
        (vae::VAE{<:AbstractVariationalEncoder,SimpleDecoder})(
                x::AbstractVecOrMat{Float32}; 
                prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0), 
                latent::Bool=false, n_samples::Int=1)

Processes the input data `x` through a VAE that consists of an encoder and a
`SimpleDecoder`.

# Arguments
- `x::AbstractVecOrMat{Float32}`: The data to be processed. This can be a vector
    or a matrix where each column represents a separate sample.

# Optional Keyword Arguments
- `prior::Distributions.Sampleable`: Specifies the prior distribution to be used
    during the reparametrization trick. Defaults to a standard normal
    distribution.
- `latent::Bool`: If set to `true`, returns a dictionary containing the latent
    variables (mean, log standard deviation, and the sampled latent
    representation) alongside the mean of the reconstructed data. Defaults to
    `false`.
- `n_samples::Int=1`: Number of samples to draw using the reparametrization
    trick.

# Returns
- If `latent=false`: A NamedTuple with the decoder outputs. See the description
  of the decoder type for more details
- If `latent=true`: A NamedTuple with keys `encoder`, `z`, `decoder`, containing
    the corresponding values. See the description of the encoder and decoder
    types for more details.

# Description
The function first encodes the input `x` using the encoder to get the mean and
log standard deviation of the latent space representation. Using the
reparametrization trick, it samples from this latent distribution, which is then
decoded using the `SimpleDecoder`.

# Note
Ensure the input data `x` matches the expected input dimensionality for the
encoder in the VAE.
"""
function (vae::VAE{JointLogEncoder,SimpleDecoder})(
    x::AbstractVecOrMat{Float32};
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
    latent::Bool=false,
    n_samples::Int=1
)
    # Run input through encoder to obtain mean and log std
    encoder_outputs = vae.encoder(x)

    # Run reparametrization trick
    z_sample = reparameterize(
        encoder_outputs.µ, encoder_outputs.logσ;
        prior=prior, n_samples=n_samples
    )

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

@doc raw"""
        (vae::VAE{JointLogEncoder,T})(
                        x::AbstractVecOrMat{Float32}; 
                        prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0), 
                        latent::Bool=false, n_samples::Int=1) where {T<:AbstractGaussianLogDecoder}

Processes the input data `x` through a VAE, consisting of an
`AbstractGaussianLogDecoder`.

# Arguments
- `x::AbstractVecOrMat{Float32}`: The data to be processed. This can be a vector
        or a matrix where each column represents a separate sample.

# Optional Keyword Arguments
- `prior::Distributions.Sampleable`: Specifies the prior distribution to be used
    during the reparametrization trick. Defaults to a standard normal
    distribution.
- `latent::Bool`: If set to `true`, returns a NamedTuple containing the latent
    variables (mean, log standard deviation, and the sampled latent
    representation) as well as the mean and log standard deviation of the
    reconstructed data. Defaults to `false`.
- `n_samples::Int=1`: Number of samples to draw using the reparametrization
    trick.

# Returns
- If `latent=false`: A NamedTuple with the decoder outputs. See the description
  of the decoder type for more details
- If `latent=true`: A NamedTuple with keys `encoder`, `z`, `decoder`, containing
  the corresponding values. See the description of the encoder and decoder types
  for more details.

# Description
The function first encodes the input `x` to obtain the mean and log standard
deviation of the latent space. Using the reparametrization trick, it samples
from this distribution, which is then decoded.

# Note
Ensure the input data `x` matches the expected input dimensionality for the
encoder in the VAE.
"""
function (vae::VAE{JointLogEncoder,T})(
    x::AbstractVecOrMat{Float32};
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
    latent::Bool=false,
    n_samples::Int=1
) where {T<:AbstractGaussianLogDecoder}
    # Run input through encoder to obtain mean and log std
    encoder_outputs = vae.encoder(x)

    # Run reparametrization trick
    z_sample = reparameterize(
        encoder_outputs.µ, encoder_outputs.logσ;
        prior=prior, n_samples=n_samples, log=true
    )

    # Check if latent variables should be returned
    if latent
        # Run latent sample through decoder
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

@doc raw"""
        (vae::VAE{JointLogEncoder,T})(
                x::AbstractVecOrMat{Float32}; 
                prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0), 
                latent::Bool=false, n_samples::Int=1) 
                where {T<:AbstractGaussianLinearDecoder}

Processes the input data `x` through a VAE, consisting of an
`AbstractGaussianLinearDecoder`.

# Arguments
- `x::AbstractVecOrMat{Float32}`: The data to be processed. This can be a vector
    or a matrix where each column represents a separate sample.

# Optional Keyword Arguments
- `prior::Distributions.Sampleable`: Specifies the prior distribution to be used
    during the reparametrization trick. Defaults to a standard normal
    distribution.
- `latent::Bool`: If set to `true`, returns a NamedTuple containing the latent
    variables (mean, log standard deviation, and the sampled latent
    representation) as well as the mean and log standard deviation of the
    reconstructed data. Defaults to `false`.
- `n_samples::Int=1`: Number of samples to draw using the reparametrization
    trick.

# Returns
- If `latent=false`: A NamedTuple with the decoder outputs. See the description
  of the decoder type for more details.
- If `latent=true`: A NamedTuple with keys `encoder`, `z`, `decoder`, containing
  the corresponding values. See the description of the encoder and decoder types
  for more details.

# Description
The function first encodes the input `x` to obtain the mean and log standard
deviation of the latent space. Using the reparametrization trick, it samples
from this distribution, which is then decoded.

# Note
Ensure the input data `x` matches the expected input dimensionality for the
encoder in the VAE.
"""
function (vae::VAE{JointLogEncoder,T})(
    x::AbstractVecOrMat{Float32};
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
    latent::Bool=false,
    n_samples::Int=1
) where {T<:AbstractGaussianLinearDecoder}
    # Run input through encoder to obtain mean and log std
    encoder_outputs = vae.encoder(x)

    # Run reparametrization trick
    z_sample = reparameterize(
        encoder_outputs.µ, encoder_outputs.logσ;
        prior=prior, n_samples=n_samples, log=true
    )

    # Check if latent variables should be returned
    if latent
        # Colect outputs in dictionary
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
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# VAE loss functions
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ==============================================================================

# ==============================================================================
# Gaussian decoder reconstruction loss
# ==============================================================================

@doc raw"""
    reconstruction_gaussian_decoder(decoder, x, vae_outputs; n_samples=1)

Calculate the reconstruction loss for a Gaussian decoder in a variational
autoencoder.

This function computes the negative log-likelihood of the input data `x` given
the Gaussian distribution parameters `decoder_µ` (mean) and `decoder_σ`
(standard deviation) provided by the decoder. It assumes that the actual mapping
from the latent space to the parameters (`decoder_µ`, `decoder_σ`) is done by
the specified `decoder`.

# Arguments
- `decoder::T`: Decoder network of type `SimpleDecoder`, which is assumed to
  have already mapped the latent variables to the parameters of the Gaussian
  distribution.
- `x::AbstractVecOrMat{Float32}`: The original input data to the encoder, to be
  compared with the reconstruction produced by the decoder. Each column
  represents a separate data sample.
- `vae_outputs::NamedTuple`: `NamedTuple` containing the all the VAE outputs.

## Optional Keyword Arguments
- `n_samples::Int=1`: The number of latent samples to average over when
  computing the reconstruction loss. More samples provide a better approximation
  of the expected reconstruction log likelihood.

# Returns
- `Float32`: The average reconstruction loss computed across all provided
  samples and data points.

# Note
- It is assumed that the mapping from latent space to decoder parameters
  (`decoder_µ`) has been performed prior to calling this function. The `decoder`
  argument is provided to indicate the type of decoder network used, but it is
  not used within the function itself.
- The reconstruction assumes a constant variance for the decoder of σ=1.
"""
function reconstruction_gaussian_decoder(
    decoder::SimpleDecoder,
    x::AbstractVecOrMat{Float32},
    vae_outputs::NamedTuple;
    n_samples::Int=1
)
    # Compute batch size
    batch_size = size(x, 2)

    # Unpack needed outputs
    decoder_µ = vae_outputs.decoder.µ

    # Validate input dimensions
    if size(x, 2) ≠ size(decoder_µ, 2)
        throw(
            DimensionMismatch(
                "Input data and decoder outputs must have the same dimensions"
            )
        )
    end

    # Validate n_samples
    if n_samples < 1
        throw(ArgumentError("Number of samples must be at least 1"))
    end # if

    # Compute average reconstruction loss as the log-likelihood of a Gaussian
    # decoder.
    log_likelihood = -0.5f0 * (
        log(2.0f0π) * length(decoder_µ) +
        sum(abs2, x - decoder_µ)
    )

    # Average over the number of samples and batch size
    return log_likelihood / (n_samples * batch_size)
end # function

@doc raw"""
    reconstruction_gaussian_decoder(decoder, x, vae_outputs; n_samples=1)

Calculate the reconstruction loss for a Gaussian decoder in a variational
autoencoder.

This function computes the negative log-likelihood of the input data `x` given
the Gaussian distribution parameters `decoder_µ` (mean) and `decoder_σ`
(standard deviation) provided by the decoder. It assumes that the actual mapping
from the latent space to the parameters (`decoder_µ`, `decoder_σ`) is done by
the specified `decoder`.

# Arguments
- `decoder::T`: Decoder network of type `AbstractGaussianLinearDecoder`,
  which is assumed to have already mapped the latent variables to the parameters
  of the Gaussian distribution.
- `x::AbstractVecOrMat{Float32}`: The original input data to the encoder, to be
  compared with the reconstruction produced by the decoder. Each column
  represents a separate data sample.
- `vae_outputs::NamedTuple`: `NamedTuple` containing the all the VAE outputs.

## Optional Keyword Arguments
- `n_samples::Int=1`: The number of latent samples to average over when
  computing the reconstruction loss. More samples provide a better approximation
  of the expected reconstruction log likelihood.

# Returns
- `Float32`: The average reconstruction loss computed across all provided
  samples and data points.

# Note
- It is assumed that the mapping from latent space to decoder parameters
  (`decoder_µ` and `decoder_σ`) has been performed prior to calling this
  function. The `decoder` argument is provided to indicate the type of decoder
  network used, but it is not used within the function itself.
"""
function reconstruction_gaussian_decoder(
    decoder::T,
    x::AbstractVecOrMat{Float32},
    vae_outputs::NamedTuple;
    n_samples::Int=1
) where {T<:AbstractGaussianLinearDecoder}
    # Compute batch size
    batch_size = size(x, 2)

    # Unpack needed ouput
    decoder_µ = vae_outputs.decoder.µ
    decoder_σ = vae_outputs.decoder.σ

    # Validate input dimensions
    if size(x, 2) ≠ size(decoder_µ, 2) || size(x, 2) ≠ size(decoder_σ, 2)
        throw(
            DimensionMismatch(
                "Input data and decoder outputs must have the same dimensions"
            )
        )
    end

    # Validate n_samples
    if n_samples < 1
        throw(ArgumentError("Number of samples must be at least 1"))
    end

    # Compute average reconstruction loss as the log-likelihood of a Gaussian
    # decoder.
    log_likelihood = -0.5f0 * (
        log(2.0f0π) * length(decoder_µ) +
        2.0f0 * sum(log, decoder_σ) +
        sum(abs2, (x - decoder_µ) ./ decoder_σ)
    )

    # Average over the number of samples and batch size
    return log_likelihood / (n_samples * batch_size)
end # function

"""
    reconstruction_gaussian_decoder(decoder, x, vae_outputs; n_samples=1)

Calculate the reconstruction loss for a Gaussian decoder in a variational
autoencoder, where the decoder outputs log standard deviations instead of
standard deviations.

# Arguments
- `decoder::T`: Decoder network of type `AbstractGaussianLogDecoder`, which
  outputs the log of the standard deviation of the Gaussian distribution.
- `x::AbstractVecOrMat{Float32}`: The original input data to the encoder, to be
  compared with the reconstruction produced by the decoder. Each column
  represents a separate data sample.
- `vae_outputs::NamedTuple`: `NamedTuple` containing the all the VAE outputs.

## Optional Keyword Arguments
- `n_samples::Int=1`: The number of latent samples to average over when
  computing the reconstruction loss.

# Returns
- `Float32`: The average reconstruction loss computed across all provided
  samples and data points.

# Note
- It is assumed that the mapping from latent space to decoder parameters
  (`decoder_µ` and `decoder_logσ`) has been performed prior to calling this
  function. The `decoder` argument is provided to indicate the type of decoder
  network used, but it is not used within the function itself.
"""
function reconstruction_gaussian_decoder(
    decoder::T,
    x::AbstractVecOrMat{Float32},
    vae_outputs::NamedTuple;
    n_samples::Int=1
) where {T<:AbstractGaussianLogDecoder}
    # Compute batch size
    batch_size = size(x, 2)

    # Unpack needed ouput
    decoder_µ = vae_outputs.decoder.µ
    decoder_logσ = vae_outputs.decoder.logσ

    # Convert log standard deviation to standard deviation
    decoder_σ = exp.(decoder_logσ)

    # Validate input dimensions
    if size(x, 2) ≠ size(decoder_µ, 2) || size(x, 2) ≠ size(decoder_logσ, 2)
        throw(
            DimensionMismatch(
                "Input data and decoder outputs must have the same dimensions"
            )
        )
    end

    # Validate n_samples
    if n_samples < 1
        throw(ArgumentError("Number of samples must be at least 1"))
    end

    # Compute average reconstruction loss as the log-likelihood of a Gaussian
    # decoder.
    log_likelihood = -0.5f0 * (
        log(2.0f0π) * length(decoder_µ) +
        2.0f0 * sum(decoder_logσ) +
        sum(abs2, (x - decoder_µ) ./ decoder_σ)
    )

    # Average over the number of samples and batch size
    return log_likelihood / (n_samples * batch_size)
end # function

# ==============================================================================
# Gaussian encoder KL loss
# ==============================================================================

@doc raw"""
    kl_gaussian_encoder(encoder, vae_outputs)

Calculate the Kullback-Leibler (KL) divergence between the approximate posterior
distribution and the prior distribution in a variational autoencoder with a
Gaussian encoder.

The KL divergence for a Gaussian encoder with mean `encoder_µ` and log standard
deviation `encoder_logσ` is computed against a standard Gaussian prior.

# Arguments
- `encoder::JointLogEncoder`: Encoder network.
- `x::AbstractVecOrMat{Float32}`: The original input data to the encoder, to be
  compared with the reconstruction produced by the decoder. Each column
  represents a separate data sample.
- `vae_outputs::NamedTuple`: `NamedTuple` containing the all the VAE outputs.

## Optional Keyword Arguments
- `n_samples::Int=1`: The number of latent samples to average over when
  computing the KL divergence loss.

# Returns
- `Float32`: The KL divergence for the entire batch of data points.

# Note
- It is assumed that the mapping from data space to laten parameters
  (`encoder_µ` and `encoder_logσ`) has been performed prior to calling this
  function. The `encoder` argument is provided to indicate the type of decoder
  network used, but it is not used within the function itself.
"""
function kl_gaussian_encoder(
    encoder::JointLogEncoder,
    x::AbstractVecOrMat{Float32},
    vae_outputs::NamedTuple;
    n_samples::Int=1
)
    # Compute batch size
    batch_size = size(x, 2)

    # Unpack needed ouput
    encoder_μ = vae_outputs.encoder.μ
    encoder_logσ = vae_outputs.encoder.logσ

    # Compute KL divergence
    return 0.5f0 * sum(
        @. (exp(2.0f0 * encoder_logσ) + encoder_μ^2 - 1.0f0) -
           2.0f0 * encoder_logσ
    ) / (n_samples * batch_size)
end # function


# ==============================================================================
# Loss VAE
# ==============================================================================

@doc raw"""
        `loss(vae, x; β=1.0f0, n_samples=1, reg_function=nothing, reg_kwargs=Dict(), 
                reg_strength=1.0f0)`

Computes the loss for the variational autoencoder (VAE) by averaging over
`n_samples` latent space samples.

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
- `x::AbstractVecOrMat{Float32}`: Input data. Each column represents a single
    data point.

# Optional Keyword Arguments
- `β::Float32=1.0f0`: Weighting factor for the KL-divergence term, used for
  annealing.
- `n_samples::Int=1`: The number of samples to draw from the latent space when
  computing the loss.
- `reg_function::Union{Function, Nothing}=nothing`: A function that computes the
  regularization term based on the VAE outputs. Should return a Float32. This
  function must take as input the VAE outputs and the keyword arguments provided
  in `reg_kwargs`.
- `reg_kwargs::Union{NamedTuple,Dict}=Dict()`: Keyword arguments to pass to the
  regularization function.
- `reg_strength::Float32=1.0f0`: The strength of the regularization term.

# Returns
- `Float32`: The computed average loss value for the input `x` and its
  reconstructed counterparts over `n_samples` samples, including possible
  regularization terms.

# Note
- Ensure that the input data `x` matches the expected input dimensionality for
    the encoder in the VAE.
"""
function loss(
    vae::VAE,
    x::AbstractVecOrMat{Float32};
    β::Float32=1.0f0,
    n_samples::Int=1,
    reg_function::Union{Function,Nothing}=nothing,
    reg_kwargs::Union{NamedTuple,Dict}=Dict(),
    reg_strength::Float32=1.0f0
)
    # Forward Pass (run input through reconstruct function with n_samples)
    vae_outputs = vae(x; latent=true, n_samples=n_samples)

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder averaged over all samples
    log_likelihood = reconstruction_gaussian_decoder(
        vae.decoder, x, vae_outputs; n_samples=n_samples
    )

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_div = kl_gaussian_encoder(
        vae.encoder, x, vae_outputs; n_samples=n_samples
    )

    # Compute regularization term if regularization function is provided
    reg_term = (reg_function !== nothing) ?
               reg_function(vae_outputs; reg_kwargs...) : 0.0f0

    # Compute average loss function
    return -log_likelihood + β * kl_div + reg_strength * reg_term
end # function

@doc raw"""
    `loss(vae, x_in, x_out; σ=1.0f0, β=1.0f0, n_samples=1, 
            regularization=nothing, reg_strength=1.0f0)`

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
- `x_in::AbstractVecOrMat{Float32}`: Input data to the VAE encoder. Each column
  represents a single data point.
- `x_out::AbstractVecOrMat{Float32}`: Target data to compute the reconstruction
  error.

# Optional Keyword Arguments
- `β::Float32=1.0f0`: Weighting factor for the KL-divergence term, used for
  annealing.
- `n_samples::Int=1`: The number of samples to draw from the latent space when
  computing the loss.
- `reg_function::Union{Function, Nothing}=nothing`: A function that computes the
  regularization term based on the VAE outputs. Should return a Float32. This
  function must take as input the VAE outputs and the keyword arguments provided
  in `reg_kwargs`.
- `reg_kwargs::Union{NamedTuple,Dict}=Dict()`: Keyword arguments to pass to the
  regularization function.
- `reg_strength::Float32=1.0f0`: The strength of the regularization term.

# Returns
- `Float32`: The computed loss value between the input `x_out` and its
  reconstructed counterpart from `x_in`, including possible regularization
  terms.

# Note
- Ensure that the input data `x_in` and `x_out` match the expected input
  dimensionality for the encoder in the VAE.
"""
function loss(
    vae::VAE,
    x_in::AbstractVecOrMat{Float32},
    x_out::AbstractVecOrMat{Float32};
    β::Float32=1.0f0,
    n_samples::Int=1,
    reg_function::Union{Function,Nothing}=nothing,
    reg_kwargs::Union{NamedTuple,Dict}=Dict(),
    reg_strength::Float32=1.0f0
)
    # Forward Pass (run input through reconstruct function with n_samples)
    vae_outputs = vae(x_in; latent=true, n_samples=n_samples)

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder averaged over all samples
    log_likelihood = reconstruction_gaussian_decoder(
        vae.decoder, x_out, vae_outputs; n_samples=n_samples
    )

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_div = kl_gaussian_encoder(
        vae.encoder, x_in, vae_outputs; n_samples=n_samples
    )

    # Compute regularization term if regularization function is provided
    reg_term = (reg_function !== nothing) ?
               reg_function(vae_outputs; reg_kwargs...) : 0.0f0

    # Compute loss function
    return -log_likelihood + β * kl_div + reg_strength * reg_term
end # function

# ==============================================================================
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# VAE training functions
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ==============================================================================

@doc raw"""
    `train!(vae, x, opt; loss_function, loss_kwargs)`

Customized training function to update parameters of a variational autoencoder
given a specified loss function.

# Arguments
- `vae::VAE`: A struct containing the elements of a variational autoencoder.
- `x::AbstractVecOrMat{Float32}`: Data on which to evaluate the loss function.
  Columns represent individual samples.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments
- `loss_function::Function=VAEs.loss`: The loss function used for training. It
  should accept the VAE model, data `x`, and keyword arguments in that order.
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
    train!(vae, x, opt; loss_fn, loss_kwargs=Dict(:β => 1.0f0, :n_samples => 5))
end
```
"""
function train!(
    vae::VAE,
    x::AbstractVecOrMat{Float32},
    opt::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict()
)
    # Compute VAE gradient
    ∇loss_ = Flux.gradient(vae) do vae_model
        loss_function(vae_model, x; loss_kwargs...)
    end # do block
    # Update parameters
    Flux.Optimisers.update!(opt, vae, ∇loss_[1])
end # function

# ==============================================================================

@doc raw"""
    `train!(vae, x_in, x_out, opt; loss_function, loss_kwargs...)`

Customized training function to update parameters of a variational autoencoder
given a loss function.

# Arguments
- `vae::VAE{<:AbstractVariationalEncoder,<:AbstractVariationalDecoder}`: A
  struct containing the elements of a variational autoencoder.
- `x_in::AbstractVecOrMat{Float32}`: Input data for the loss function.
  Represents an individual sample.
- `x_out::AbstractVecOrMat{Float32}`: Target output data for the loss function.
  Represents the corresponding output for the `x_in` sample.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments
- `loss_function::Function=VAEs.loss`: The loss function used for training. It
  should accept the VAE model, data `x`, and keyword arguments in that order.
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
    vae::VAE{<:AbstractVariationalEncoder,<:AbstractVariationalDecoder},
    x_in::AbstractVecOrMat{Float32},
    x_out::AbstractVecOrMat{Float32},
    opt::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict()
)
    # Compute VAE gradient
    ∇loss_ = Flux.gradient(vae) do vae_model
        loss_function(vae_model, x_in, x_out; loss_kwargs...)
    end # do block
    # Update parameters
    Flux.Optimisers.update!(opt, vae, ∇loss_[1])
end # function