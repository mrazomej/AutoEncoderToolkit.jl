# Import ML libraries
import Flux

# Import AutoDiff backends
import ChainRulesCore
import TaylorDiff
import Zygote
import ForwardDiff

# Import GPU libraries
using CUDA

# Import basic math
import LinearAlgebra
import Random
import StatsBase
import Distributions

# Import library to use Ellipsis Notation
using EllipsisNotation

# Import Abstract Types
using ..AutoEncode: AbstractVariationalAutoEncoder,
    AbstractVariationalEncoder, AbstractGaussianEncoder,
    AbstractGaussianLogEncoder,
    AbstractVariationalDecoder, AbstractGaussianDecoder,
    AbstractGaussianLogDecoder, AbstractGaussianLinearDecoder

# Import Concrete Encoder Types
using ..AutoEncode: JointLogEncoder

# Import Concrete Decoder Types
using ..AutoEncode: BernoulliDecoder, SimpleDecoder,
    JointLogDecoder, SplitLogDecoder,
    JointDecoder, SplitDecoder

# Import Concrete VAE type
using ..AutoEncode: VAE

# Import log-probability functions
using ..AutoEncode: decoder_loglikelihood, spherical_logprior,
    encoder_logposterior

# Import functions from other modules
using ..VAEs: reparameterize

# Import functions
using ..utils: finite_difference_gradient, taylordiff_gradient,
    storage_type, randn_sample

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Caterini, A. L., Doucet, A. & Sejdinovic, D. Hamiltonian Variational
# Auto-Encoder. 11 (2018).
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ==============================================================================
# HVAE struct and forward pass methods
# ==============================================================================

@doc raw"""
    struct HVAE{
        V<:VAE{<:AbstractVariationalEncoder,<:AbstractVariationalDecoder}
    } <: AbstractVariationalAutoEncoder

Hamiltonian Variational Autoencoder (HVAE) model defined for `Flux.jl`.

# Fields
- `vae::V`: A Variational Autoencoder (VAE) model that forms the basis of the
    HVAE. `V` is a subtype of `VAE` with a specific `AbstractVariationalEncoder`
    and `AbstractVariationalDecoder`.

An HVAE is a type of Variational Autoencoder (VAE) that uses Hamiltonian Monte
Carlo (HMC) to sample from the posterior distribution in the latent space. The
VAE's encoder compresses the input into a low-dimensional probabilistic
representation q(z|x). The VAE's decoder tries to reconstruct the original input
from a sampled point in the latent space p(x|z). 

The HMC sampling in the latent space allows the HVAE to better capture complex
posterior distributions compared to a standard VAE, which assumes a simple
Gaussian posterior. This can lead to more accurate reconstructions and better
disentanglement of latent variables.
"""
struct HVAE{
    V<:VAE{<:AbstractVariationalEncoder,<:AbstractVariationalDecoder}
} <: AbstractVariationalAutoEncoder
    vae::V
end # struct

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor HVAE

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ==============================================================================
# Hamiltonian Dynamics
# ==============================================================================
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ==============================================================================
# Potential Energy computation
# ==============================================================================

@doc raw"""
    potential_energy(
        x::AbstractVector,
        z::AbstractVector,
        decoder::AbstractVariationalDecoder,
        decoder_output::NamedTuple;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        latent_logprior::Function=spherical_logprior
    ) 

Compute the potential energy of a Hamiltonian Variational Autoencoder (HVAE). In
the context of Hamiltonian Monte Carlo (HMC), the potential energy is defined as
the negative log-posterior. This function computes the potential energy for
given data `x` and latent variable `z`. It does this by computing the
log-likelihood of `x` under the distribution defined by
`reconstruction_loglikelihood(x, z, decoder, decoder_output)`, and the log-prior
of `z` under the `latent_logprior` distribution. The potential energy is then
computed as:
    
        U(x, z) = -log p(x | z) - log p(z)

# Arguments
- `x::AbstractArray`: An array representing the input data. The last dimension
  corresponds to different data points.
- `z::AbstractVecOrMat`: A latent variable encoding of the input data. If a
  matrix, each column corresponds to a different data point.
- `decoder::AbstractVariationalDecoder`: A decoder that maps the latent
  variables to the data space.
- `decoder_output::NamedTuple`: The output of the decoder.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function=decoder_loglikelihood`: A function
  representing the log-likelihood function used by the decoder. The function
  must take as first input a vector `x` representing the data, as second input a
  vector `z` representing the latent variable, as third input a decoder, and as
  fourth input a NamedTuple representing the decoder output. Default is
  `decoder_loglikelihood`.
- `latent_logprior::Function=spherical_logprior`: A function representing the
  log-prior distribution used in the autoencoder. The function must take as
  single input a vector `z` representing the latent variable. Default is
  `spherical_logprior`.  

# Returns
- `energy`: The computed potential energy for the given input `x` and latent
  variable `z`.
"""
function potential_energy(
    x::AbstractArray,
    z::AbstractVecOrMat,
    decoder::AbstractVariationalDecoder,
    decoder_output::NamedTuple;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    latent_logprior::Function=spherical_logprior,
)
    # Compute log-likelihood
    loglikelihood = reconstruction_loglikelihood(
        x, z, decoder, decoder_output
    )

    # Compute log-prior
    logprior = latent_logprior(z)

    # Compute potential energy
    return -loglikelihood - logprior
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    potential_energy(
        x::AbstractArray,
        z::AbstractVecOrMat,
        hvae::HVAE;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        latent_logprior::Function=spherical_logprior
    ) 

Compute the potential energy of a Hamiltonian Variational Autoencoder (HVAE). In
the context of Hamiltonian Monte Carlo (HMC), the potential energy is defined as
the negative log-posterior. This function computes the potential energy for
given data `x` and latent variable `z`. It does this by computing the
log-likelihood of `x` under the distribution defined by
`reconstruction_loglikelihood(x, z, hvae.vae.decoder, decoder_output)`, and the
log-prior of `z` under the `latent_logprior` distribution. The potential energy
is then computed as:
        
                U(x, z) = -log p(x | z) - log p(z)

# Arguments
- `x::AbstractArray`: An array representing the input data. The last dimension
  corresponds to different data points.
- `z::AbstractVecOrMat`: A latent variable encoding of the input data. If a
  matrix, each column corresponds to a different data point.
- `hvae::HVAE`: A Hamiltonian Variational Autoencoder that contains the decoder.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function=decoder_loglikelihood`: A function
  representing the log-likelihood function used by the decoder. The function
  must take as first input an array `x` representing the data, as second input a
  vector or matrix `z` representing the latent variable, as third input a
  decoder, and as fourth input a NamedTuple representing the decoder output.
  Default is `decoder_loglikelihood`.
- `latent_logprior::Function=spherical_logprior`: A function representing the
  log-prior distribution used in the autoencoder. The function must take as
  single input a vector or matrix `z` representing the latent variable.  Default
  is `spherical_logprior`.  

# Returns
- `energy`: The computed potential energy for the given input `x` and latent
    variable `z`.
"""
function potential_energy(
    x::AbstractArray,
    z::AbstractVecOrMat,
    hvae::HVAE;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    latent_logprior::Function=spherical_logprior,
)
    # Compute decoder output
    decoder_output = hvae.vae.decoder(z)

    # Compute log-likelihood
    loglikelihood = reconstruction_loglikelihood(
        x, z, hvae.vae.decoder, decoder_output
    )

    # Compute log-prior
    logprior = latent_logprior(z)

    # Compute potential energy
    return -loglikelihood - logprior
end # function

# ==============================================================================
# Gradient of the Potential Energy computation
# ==============================================================================

# ------------------------------------------------------------------------------
# Finite Difference Method
# ------------------------------------------------------------------------------

@doc raw"""
    ∇potential_energy_finite(
        x::AbstractArray,
        z::AbstractVecOrMat,
        decoder::AbstractVariationalDecoder,
        decoder_output::NamedTuple;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        latent_logprior::Function=spherical_logprior,
        fdtype::Symbol=:central
    )

Compute the gradient of the potential energy of a Hamiltonian Variational
Autoencoder (HVAE) with respect to the latent variables `z` using finite
difference method. This function returns the gradient of the potential energy
computed for given data `x` and latent variable `z`.

# Arguments
- `x::AbstractArray`: An array representing the input data. The last dimension
  corresponds to different data points.
- `z::AbstractVecOrMat`: A latent variable encoding of the input data. If a
  matrix, each column corresponds to a different data point.
- `decoder::AbstractVariationalDecoder`: A decoder that maps the latent
  variables to the data space.
- `decoder_output::NamedTuple`: The output of the decoder.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function=decoder_loglikelihood`: A function
  representing the log-likelihood function used by the decoder. The function
  must take as first input an `AbstractVariationalDecoder` struct, as second
  input an array `x` representing the data, and as third input a vector or
  matrix `z` representing the latent variable. Default is
  `decoder_loglikelihood`.
- `latent_logprior::Function=spherical_logprior`: A function representing the
  log-prior distribution used in the autoencoder. The function must take as
  single input a vector or matrix `z` representing the latent variable. Default
  is `spherical_logprior`.  
- `fdtype::Symbol=:central`: A symbol representing the type of finite difference
  method to use. Default is `:central`, but it can also be `:forward`.

# Returns
- `gradient`: The computed gradient of the potential energy for the given input
  `x` and latent variable `z`.
"""
function ∇potential_energy_finite(
    x::AbstractArray,
    z::AbstractVecOrMat,
    decoder::AbstractVariationalDecoder,
    decoder_output::NamedTuple;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    latent_logprior::Function=spherical_logprior,
    fdtype::Symbol=:central,
)
    # Compute the gradient with respect to z
    return finite_difference_gradient(
        z -> potential_energy(
            x, z, decoder, decoder_output;
            reconstruction_loglikelihood=reconstruction_loglikelihood,
            latent_logprior=latent_logprior,
        ),
        z; fdtype=fdtype
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    ∇potential_energy_finite(
        x::AbstractArray,
        z::AbstractVecOrMat,
        hvae::HVAE;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        latent_logprior::Function=spherical_logprior,
        fdtype::Symbol=:central
    )

Compute the gradient of the potential energy of a Hamiltonian Variational
Autoencoder (HVAE) with respect to the latent variables `z` using finite
difference method. This function returns the gradient of the potential energy
computed for given data `x` and latent variable `z`.

# Arguments
- `x::AbstractArray`: An array representing the input data. The last dimension
  corresponds to different data points.
- `z::AbstractVecOrMat`: A latent variable encoding of the input data. If a
  matrix, each column corresponds to a different data point.
- `hvae::HVAE`: An HVAE model that contains a decoder which maps the latent
  variables to the data space.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function=decoder_loglikelihood`: A function
  representing the log-likelihood function used by the decoder. The function
  must take as first input an array `x` representing the data, as second input a
  vector or matrix `z` representing the latent variable, and as third input a
  decoder. Default is `decoder_loglikelihood`.
- `latent_logprior::Function=spherical_logprior`: A function representing the
  log-prior distribution used in the autoencoder. The function must take as
  single input a vector or matrix `z` representing the latent variable. Default
  is `spherical_logprior`.  
- `fdtype::Symbol=:central`: A symbol representing the type of finite difference
  method to use. Default is `:central`, but it can also be `:forward`.

# Returns
- `gradient`: The computed gradient of the potential energy for the given input
  `x` and latent variable `z`.
"""
function ∇potential_energy_finite(
    x::AbstractArray,
    z::AbstractVecOrMat,
    hvae::HVAE;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    latent_logprior::Function=spherical_logprior,
    fdtype::Symbol=:central,
)
    # Compute decoder output
    decoder_output = hvae.vae.decoder(z)

    # Compute the gradient with respect to z
    return finite_difference_gradient(
        z -> potential_energy(
            x, z, hvae.vae.decoder, decoder_output;
            reconstruction_loglikelihood=reconstruction_loglikelihood,
            latent_logprior=latent_logprior,
        ),
        z; fdtype=fdtype
    )
end # function

# ------------------------------------------------------------------------------
# TaylorDiff Method
# ------------------------------------------------------------------------------

@doc raw"""
    ∇potential_energy_TaylorDiff(
        x::AbstractArray,
        z::AbstractVecOrMat,
        hvae::HVAE;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        latent_logprior::Function=spherical_logprior,
    )

Compute the gradient of the potential energy of a Hamiltonian Variational
Autoencoder (HVAE) with respect to the latent variables `z` using Taylor series
differentiation. This function returns the gradient of the potential energy
computed for given data `x` and latent variable `z`.

# Arguments
- `x::AbstractArray`: An array representing the input data. The last dimension
  corresponds to different data points.
- `z::AbstractVecOrMat`: A latent variable encoding of the input data. If a
  matrix, each column corresponds to a different data point.
- `hvae::HVAE`: An HVAE model that contains a decoder which maps the latent
  variables to the data space.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function=decoder_loglikelihood`: A function
  representing the log-likelihood function used by the decoder. The function
  must take as first input an array `x` representing the data, as second input a
  vector or matrix `z` representing the latent variable, and as third input a
  decoder. Default is `decoder_loglikelihood`.
- `latent_logprior::Function=spherical_logprior`: A function representing the
  log-prior distribution used in the autoencoder. The function must take as
  single input a vector or matrix `z` representing the latent variable.  Default
  is `spherical_logprior`.  

# Returns
- `gradient`: The computed gradient of the potential energy for the given input
    `x` and latent variable `z`.
"""
function ∇potential_energy_TaylorDiff(
    x::AbstractArray,
    z::AbstractVecOrMat,
    decoder::AbstractVariationalDecoder,
    decoder_output::NamedTuple;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    latent_logprior::Function=spherical_logprior,
)
    # Compute the gradient with respect to z
    return taylordiff_gradient(
        z -> potential_energy(
            x, z, decoder, decoder_output;
            reconstruction_loglikelihood=reconstruction_loglikelihood,
            latent_logprior=latent_logprior,
        ),
        z;
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    ∇potential_energy_TaylorDiff(
        x::AbstractArray,
        z::AbstractVecOrMat,
        hvae::HVAE;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        latent_logprior::Function=spherical_logprior,
    )

Compute the gradient of the potential energy of a Hamiltonian Variational
Autoencoder (HVAE) with respect to the latent variables `z` using Taylor series
differentiation. This function returns the gradient of the potential energy
computed for given data `x` and latent variable `z`.

# Arguments
- `x::AbstractArray`: An array representing the input data. The last dimension
  corresponds to different data points.
- `z::AbstractVecOrMat`: A latent variable encoding of the input data. If a
  matrix, each column corresponds to a different data point.
- `hvae::HVAE`: An HVAE model that contains a decoder which maps the latent
  variables to the data space.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function=decoder_loglikelihood`: A function
  representing the log-likelihood function used by the decoder. The function
  must take as first input an array `x` representing the data, as second input a
  vector or matrix `z` representing the latent variable, and as third input a
  decoder. Default is `decoder_loglikelihood`.
- `latent_logprior::Function=spherical_logprior`: A function representing the
  log-prior distribution used in the autoencoder. The function must take as
  single input a vector or matrix `z` representing the latent variable.  Default
  is `spherical_logprior`.  

# Returns
- `gradient`: The computed gradient of the potential energy for the given input
    `x` and latent variable `z`.
"""
function ∇potential_energy_TaylorDiff(
    x::AbstractArray,
    z::AbstractVecOrMat,
    hvae::HVAE;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    latent_logprior::Function=spherical_logprior,
)
    # Compute decoder output
    decoder_output = hvae.vae.decoder(z)

    # Compute the gradient with respect to z
    return taylordiff_gradient(
        z -> potential_energy(
            x, z, hvae.vae.decoder, decoder_output;
            reconstruction_loglikelihood=reconstruction_loglikelihood,
            latent_logprior=latent_logprior,
        ),
        z;
    )
end # function

# ------------------------------------------------------------------------------

"""
    ∇energyhvae

A `NamedTuple` mapping automatic differentiation types to their corresponding
gradient computation functions for the potential energy
"""
const ∇energyhvae = (
    finite=∇potential_energy_finite,
    TaylorDiff=∇potential_energy_TaylorDiff,
)

# ------------------------------------------------------------------------------

@doc raw"""
    ∇potential_energy(
        x::AbstractArray,
        z::AbstractVecOrMat,
        decoder::AbstractVariationalDecoder,
        decoder_output::NamedTuple;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        latent_logprior::Function=spherical_logprior,
        adtype::Union{Symbol,Nothing}=nothing,
        adkwargs::Union{NamedTuple,Dict}=Dict(),
    )

Compute the gradient of the potential energy of a Hamiltonian Variational
Autoencoder (HVAE) with respect to the latent variables `z` using the specified
automatic differentiation method. This function returns the gradient of the
potential energy computed for given data `x` and latent variable `z`.

# Arguments
- `x::AbstractArray`: An array representing the input data. The last dimension
  corresponds to different data points.
- `z::AbstractVecOrMat`: A latent variable encoding of the input data. If a
  matrix, each column corresponds to a different data point.
- `decoder::AbstractVariationalDecoder`: A decoder that maps the latent
  variables to the data space.
- `decoder_output::NamedTuple`: The output of the decoder.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function=decoder_loglikelihood`: A function
  representing the log-likelihood function used by the decoder. The function
  must take as first input an `AbstractVariationalDecoder` struct, as second
  input an array `x` representing the data, and as third input a vector or
  matrix `z` representing the latent variable. Default is
  `decoder_loglikelihood`.
- `latent_logprior::Function=spherical_logprior`: A function representing the
  log-prior distribution used in the autoencoder. The function must take as
  single input a vector or matrix `z` representing the latent variable. Default
  is `spherical_logprior`.  
- `adtype::Union{Symbol,Nothing}=nothing`: A symbol representing the type of
  automatic differentiation method to use. Default is `nothing`, which means the
  method will be chosen based on the type of `x`.
- `adkwargs::Union{NamedTuple,Dict}=Dict()`: Additional keyword arguments to
  pass to the automatic differentiation method.

# Returns
- `gradient`: The computed gradient of the potential energy for the given input
    `x` and latent variable `z`.
"""
function ∇potential_energy(
    x::AbstractArray,
    z::AbstractVecOrMat,
    decoder::AbstractVariationalDecoder,
    decoder_output::NamedTuple;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    latent_logprior::Function=spherical_logprior,
    adtype::Union{Symbol,Nothing}=nothing,
    adkwargs::Union{NamedTuple,Dict}=Dict(),
)
    # Obtain x storage type
    stx = storage_type(x)

    # Check if no automatic differentiation method is specified
    if (stx <: CUDA.CuArray) && (adtype === nothing)
        # If no automatic differentiation method is specified, use finite
        # differences for CUDA arrays
        adtype = :finite
    elseif adtype === nothing
        # If no automatic differentiation method is specified, use TaylorDiff
        adtype = :finite
    elseif (adtype <: Symbol) && (adtype ∉ keys(∇Hhvae))
        # If automatic differentiation method is specified, check if it is valid
        error("adtype must be one of $(keys(∇energyhvae))")
    end

    # Compute gradient with respect to var
    return ∇energyhvae[adtype](
        x, z, decoder, decoder_output;
        reconstruction_loglikelihood=reconstruction_loglikelihood,
        latent_logprior=latent_logprior,
        adkwargs...
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    ∇potential_energy(
        x::AbstractArray,
        z::AbstractVecOrMat,
        hvae::HVAE;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        latent_logprior::Function=spherical_logprior,
        adtype::Union{Symbol,Nothing}=nothing,
        adkwargs::Union{NamedTuple,Dict}=Dict(),
    )

Compute the gradient of the potential energy of a Hamiltonian Variational
Autoencoder (HVAE) with respect to the latent variables `z` using the specified
automatic differentiation method. This function returns the gradient of the
potential energy computed for given data `x` and latent variable `z`.

# Arguments
- `x::AbstractArray`: An array representing the input data. The last dimension
  corresponds to different data points.
- `z::AbstractVecOrMat`: A latent variable encoding of the input data. If a
  matrix, each column corresponds to a different data point.
- `hvae::HVAE`: An HVAE model that contains a decoder which maps the latent
  variables to the data space.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function=decoder_loglikelihood`: A function
  representing the log-likelihood function used by the decoder. The function
  must take as first input an array `x` representing the data, as second input a
  vector or matrix `z` representing the latent variable, and as third input a
  decoder. Default is `decoder_loglikelihood`.
- `latent_logprior::Function=spherical_logprior`: A function representing the
  log-prior distribution used in the autoencoder. The function must take as
  single input a vector or matrix `z` representing the latent variable.  Default
  is `spherical_logprior`.  
- `adtype::Union{Symbol,Nothing}=nothing`: A symbol representing the type of
  automatic differentiation method to use. Default is `nothing`, which means the
  method will be chosen based on the type of `x`.
- `adkwargs::Union{NamedTuple,Dict}=Dict()`: Additional keyword arguments to
  pass to the automatic differentiation method.

# Returns
- `gradient`: The computed gradient of the potential energy for the given input
  `x` and latent variable `z`.
"""
function ∇potential_energy(
    x::AbstractArray,
    z::AbstractVecOrMat,
    hvae::HVAE;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    latent_logprior::Function=spherical_logprior,
    adtype::Union{Symbol,Nothing}=nothing,
    adkwargs::Union{NamedTuple,Dict}=Dict(),
)
    # Obtain x storage type
    stx = storage_type(x)

    # Check if no automatic differentiation method is specified
    if (stx <: CUDA.CuArray) && (adtype === nothing)
        # If no automatic differentiation method is specified, use finite
        # differences for CUDA arrays
        adtype = :finite
    elseif adtype === nothing
        # If no automatic differentiation method is specified, use TaylorDiff
        adtype = :finite
    elseif (adtype <: Symbol) && (adtype ∉ keys(∇Hhvae))
        # If automatic differentiation method is specified, check if it is valid
        error("adtype must be one of $(keys(∇energyhvae))")
    end

    # Compute gradient with respect to var
    return ∇energyhvae[adtype](
        x, z, hvae;
        reconstruction_loglikelihood=reconstruction_loglikelihood,
        latent_logprior=latent_logprior,
        adkwargs...
    )
end # function

# ==============================================================================
# Hamiltonian Dynamics
# ==============================================================================

@doc raw"""
    leapfrog_step(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        decoder::AbstractVariationalDecoder,
        decoder_output::NamedTuple;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        ∇U_kwargs::Union{Dict,NamedTuple}=(
            reconstruction_loglikelihood=reconstruction_loglikelihood,
            latent_logprior=spherical_logprior,
        )
    )

Perform a full step of the leapfrog integrator for Hamiltonian dynamics.

The leapfrog integrator is a numerical integration scheme used to simulate
Hamiltonian dynamics. It consists of three steps:

1. Half update of the momentum variable: 

        ρ(t + ϵ/2) = ρ(t) - 0.5 * ϵ * ∇z_U(z(t), ρ(t + ϵ/2)).

2. Full update of the position variable: 

        z(t + ϵ) = z(t) + ϵ * ρ(t + ϵ/2).

3. Half update of the momentum variable: 

        ρ(t + ϵ) = ρ(t + ϵ/2) - 0.5 * ϵ * ∇z_U(z(t + ϵ), ρ(t + ϵ/2)).

This function performs these three steps in sequence.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not necessarily
  need to be a vector. Array inputs are supported. The last dimension is assumed
  to have each of the data points.
- `z::AbstractVecOrMat`: The point in the latent space. If matrix, each column
  represents a point in the latent space.
- `ρ::AbstractVecOrMat`: The momentum. If matrix, each column represents a
  momentum vector.
- `decoder::AbstractVariationalDecoder`: The decoder instance.
- `decoder_output::NamedTuple`: The output of the decoder.

# Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4)`: The step size. Default is
  0.0001.
- `∇U_kwargs::Union{Dict,NamedTuple}`: The keyword arguments for
  `∇potential_energy`.  Default is a tuple with `reconstruction_loglikelihood`
  and `latent_logprior`.

# Returns
A tuple `(z̄, ρ̄, decoder_output_z̄)` representing the updated position and
momentum after performing the full leapfrog step as well as the decoder output
of the updated position.
"""
function leapfrog_step(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    decoder::AbstractVariationalDecoder,
    decoder_output::NamedTuple;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    ∇U_kwargs::Union{Dict,NamedTuple}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        latent_logprior=spherical_logprior,
    )
)
    # Update momentum variable with half-step
    ρ̃ = ρ - (0.5f0 * ϵ) .* ∇potential_energy(
        x, z, decoder, decoder_output;
        ∇U_kwargs...
    )

    # Update position variable with full-step
    z̄ = z + ϵ .* ρ̃

    # Update decoder output
    decoder_output_z̄ = decoder(z̄)

    # Update momentum variable with half-step
    ρ̄ = ρ̃ - (0.5f0 * ϵ) .* ∇potential_energy(
        x, z̄, decoder, decoder_output_z̄;
        ∇U_kwargs...
    )

    return z̄, ρ̄, decoder_output_z̄
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    leapfrog_step(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        hvae::HVAE;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        ∇U_kwargs::Union{Dict,NamedTuple}=(
            reconstruction_loglikelihood=reconstruction_loglikelihood,
            latent_logprior=spherical_logprior,
        )
    )

Perform a full step of the leapfrog integrator for Hamiltonian dynamics.

The leapfrog integrator is a numerical integration scheme used to simulate
Hamiltonian dynamics. It consists of three steps:

1. Half update of the momentum variable: 

        ρ(t + ϵ/2) = ρ(t) - 0.5 * ϵ * ∇z_U(z(t), ρ(t + ϵ/2)).

2. Full update of the position variable: 

        z(t + ϵ) = z(t) + ϵ * ρ(t + ϵ/2).

3. Half update of the momentum variable: 

        ρ(t + ϵ) = ρ(t + ϵ/2) - 0.5 * ϵ * ∇z_U(z(t + ϵ), ρ(t + ϵ/2)).

This function performs these three steps in sequence.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not necessarily
  need to be a vector. Array inputs are supported. The last dimension is assumed
  to have each of the data points.
- `z::AbstractVecOrMat`: The point in the latent space. If matrix, each column
  represents a point in the latent space.
- `ρ::AbstractVecOrMat`: The momentum. If matrix, each column represents a
  momentum vector.
- `hvae::HVAE`: An HVAE model that contains the decoder.

# Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4)`: The step size. Default is
  0.0001.
- `∇U_kwargs::Union{Dict,NamedTuple}`: The keyword arguments for
  `∇potential_energy`.  Default is a tuple with `reconstruction_loglikelihood`
  and `latent_logprior`.

# Returns
A tuple `(z̄, ρ̄, decoder_output_z̄)` representing the updated position and
momentum after performing the full leapfrog step as well as the decoder output
of the updated position.
"""
function leapfrog_step(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    hvae::HVAE;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    ∇U_kwargs::Union{Dict,NamedTuple}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        latent_logprior=spherical_logprior,
    )
)
    # Compute the output of the decoder
    decoder_output = hvae.vae.decoder(z)

    # Update momentum variable with half-step
    ρ̃ = ρ - (0.5f0 * ϵ) .* ∇potential_energy(
        x, z, hvae.vae.decoder, decoder_output;
        ∇U_kwargs...
    )

    # Update position variable with full-step
    z̄ = z + ϵ .* ρ̃

    # Update decoder output
    decoder_output_z̄ = hvae.vae.decoder(z̄)

    # Update momentum variable with half-step
    ρ̄ = ρ̃ - (0.5f0 * ϵ) .* ∇potential_energy(
        x, z̄, hvae.vae.decoder, decoder_output_z̄;
        ∇U_kwargs...
    )

    return z̄, ρ̄, decoder_output_z̄
end # function

# ==============================================================================
# Tempering Functions
# ==============================================================================

@doc raw"""
    quadratic_tempering(βₒ::AbstractFloat, k::Int, K::Int)

Compute the inverse temperature `βₖ` at a given stage `k` of a tempering
schedule with `K` total stages, using a quadratic tempering scheme. 

Tempering is a technique used in sampling algorithms to improve mixing and
convergence. It involves running parallel chains of the algorithm at different
"temperatures", and swapping states between the chains. The "temperature" of a
chain is controlled by an inverse temperature parameter `β`, which is varied
according to a tempering schedule. 

In a quadratic tempering schedule, the inverse temperature `βₖ` at stage `k` is
computed as the square of the quantity `((1 - 1 / √(βₒ)) * (k / K)^2 + 1 /
√(βₒ))`, where `βₒ` is the initial inverse temperature. This schedule starts at
`βₒ` when `k = 0`, and increases quadratically as `k` increases, reaching 1 when
`k = K`.

# Arguments
- `βₒ::AbstractFloat`: The initial inverse temperature.
- `k::Int`: The current stage of the tempering schedule.
- `K::Int`: The total number of stages in the tempering schedule.

# Returns
- `βₖ::AbstractFloat`: The inverse temperature at stage `k`.
"""
function quadratic_tempering(
    βₒ::T,
    k::Int,
    K::Int,
) where {T<:AbstractFloat}
    # Compute βₖ
    βₖ = ((1 - 1 / √(βₒ)) * (k / K)^2 + 1 / √(βₒ))^(-2)

    return T(βₖ)
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
        null_tempering(βₒ::T, k::Int, K::Int) where {T<:AbstractFloat}

Return the initial inverse temperature `βₒ`. This function is used in the
context of tempered Hamiltonian Monte Carlo (HMC) methods, where tempering
involves running HMC at different "temperatures" to improve mixing and
convergence. 

In this case, `null_tempering` is a simple tempering schedule that does not
actually change the temperature—it always returns the initial inverse
temperature `βₒ`. This can be useful as a default or placeholder tempering
schedule.

# Arguments
- `βₒ::AbstractFloat`: The initial inverse temperature. 
- `k::Int`: The current step in the tempering schedule. Not used in this
    function, but included for compatibility with other tempering schedules.
- `K::Int`: The total number of steps in the tempering schedule. Not used in
    this function, but included for compatibility with other tempering
    schedules.

# Returns
- `β::T`: The inverse temperature for the current step, which is always `βₒ` in
  this case.

# Example
```julia
βₒ = 0.5
k = 1
K = 10
β = null_tempering(βₒ, k, K)  # β will be 0.5
```
"""
function null_tempering(
    βₒ::AbstractFloat,
    k::Int,
    K::Int,
)
    return βₒ
end # function

# ==============================================================================
# Combining Leapfrog and Tempering Steps
# ==============================================================================

@doc raw"""
    leapfrog_tempering_step(
        x::AbstractArray,
        zₒ::AbstractVecOrMat,
        decoder::AbstractVariationalDecoder,
        decoder_output::NamedTuple;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        K::Int=3,
        βₒ::Number=0.3f0,
        ∇U_kwargs::Union{Dict,NamedTuple}=(
            reconstruction_loglikelihood=reconstruction_loglikelihood,
            latent_logprior=spherical_logprior,
        ),
        tempering_schedule::Function=quadratic_tempering,
    )

Combines the leapfrog and tempering steps into a single function for the
Hamiltonian Variational Autoencoder (HVAE).

# Arguments
- `x::AbstractArray`: The data to be processed. If `Array`, the last dimension
  must be of size 1.
- `zₒ::AbstractVecOrMat`: The initial latent variable. 
- `decoder::AbstractVariationalDecoder`: The decoder of the HVAE model.
- `decoder_output::NamedTuple`: The output of the decoder.

# Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}`: The step size for the leapfrog steps in
  the HMC algorithm. This can be a scalar or an array. Default is 0.0001.  
- `K::Int`: The number of leapfrog steps to perform in the Hamiltonian Monte
  Carlo (HMC) algorithm. Default is 3.
- `βₒ::Number`: The initial inverse temperature for the tempering schedule.
  Default is 0.3f0.
- `∇U_kwargs::Union{Dict,NamedTuple}`: Additional keyword arguments to be passed
  to the `∇potential_energy` function. Default is a NamedTuple with
  `reconstruction_loglikelihood` and `latent_logprior`.
- `tempering_schedule::Function`: The function to compute the inverse
  temperature at each step in the HMC algorithm. Defaults to
  `quadratic_tempering`. This function must take three arguments: First, `βₒ`,
  an initial inverse temperature, second, `k`, the current step in the tempering
  schedule, and third, `K`, the total number of steps in the tempering schedule.

# Returns
- A `NamedTuple` with the following keys: 
    - `z_init`: The initial latent variable. 
    - `ρ_init`: The initial momentum variable. 
    - `z_final`: The final latent variable after `K` leapfrog steps. 
    - `ρ_final`: The final momentum variable after `K` leapfrog steps. 
- The decoder output at the final latent variable is also returned. Note: This
  is not in the same named tuple as the other outputs, but as a separate output.

# Description
The function first samples a random momentum variable `γₒ` from a standard
normal distribution and scales it by the inverse square root of the initial
inverse temperature `βₒ` to obtain the initial momentum variable `ρₒ`. Then, it
performs `K` leapfrog steps, each followed by a tempering step, to generate a
new sample from the latent space.

# Note
Ensure the input data `x` and the initial latent variable `zₒ` match the
expected input dimensionality for the HVAE model.
"""
function leapfrog_tempering_step(
    x::AbstractArray,
    zₒ::AbstractVecOrMat,
    decoder::AbstractVariationalDecoder,
    decoder_output::NamedTuple;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    K::Int=3,
    βₒ::Number=0.3f0,
    ∇U_kwargs::Union{Dict,NamedTuple}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        latent_logprior=spherical_logprior,
    ),
    tempering_schedule::Function=quadratic_tempering,
)
    # Extract latent-space dimensionality
    ldim = size(zₒ, 1)

    # Sample γₒ ~ N(0, I)
    γₒ = ChainRulesCore.@ignore_derivatives randn_sample(zₒ)

    # Define ρₒ = γₒ / √βₒ
    ρₒ = γₒ ./ √(βₒ)

    # Define initial value of z and ρ before loop
    zₖ₋₁ = deepcopy(zₒ)
    ρₖ₋₁ = deepcopy(ρₒ)
    decoderₖ₋₁ = deepcopy(decoder_output)

    # Loop over K steps
    for k = 1:K
        # 1) Leapfrog step
        zₖ, ρₖ, decoderₖ = leapfrog_step(
            x, zₖ₋₁, ρₖ₋₁, decoder, decoder_output;
            ϵ=ϵ, ∇U_kwargs=∇U_kwargs
        )

        # 2) Tempering step
        # Compute previous step's inverse temperature
        βₖ₋₁ = tempering_schedule(βₒ, k - 1, K)
        # Compute current step's inverse temperature
        βₖ = tempering_schedule(βₒ, k, K)

        # Update momentum variable with tempering Update zₖ₋₁, ρₖ₋₁ for next
        # iteration. The momentum variable is updated with tempering. Also, note
        # this is the last step as well, thus we return zₖ₋₁, ρₖ₋₁ as the final
        # points.
        zₖ₋₁ = zₖ
        ρₖ₋₁ = ρₖ .* √(βₖ₋₁ / βₖ)
        # Update decoderₖ₋₁ for next iteration
        decoderₖ₋₁ = decoderₖ
    end # for

    return (
        z_init=zₒ,
        ρ_init=ρₒ,
        z_final=zₖ₋₁,
        ρ_final=ρₖ₋₁,
    ), decoderₖ₋₁
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    leapfrog_tempering_step(
        x::AbstractArray,
        zₒ::AbstractVecOrMat,
        hvae::HVAE;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        K::Int=3,
        βₒ::Number=0.3f0,
        ∇U_kwargs::Union{Dict,NamedTuple}=(
            reconstruction_loglikelihood=reconstruction_loglikelihood,
            latent_logprior=spherical_logprior,
        ),
        tempering_schedule::Function=quadratic_tempering,
    )

Combines the leapfrog and tempering steps into a single function for the
Hamiltonian Variational Autoencoder (HVAE).

# Arguments
- `x::AbstractArray`: The data to be processed. If `Array`, the last dimension
  must be of size 1.
- `zₒ::AbstractVecOrMat`: The initial latent variable. 
- `hvae::HVAE`: An HVAE model that contains the decoder.

# Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}`: The step size for the leapfrog steps in
  the HMC algorithm. This can be a scalar or an array. Default is 0.0001.  
- `K::Int`: The number of leapfrog steps to perform in the Hamiltonian Monte
  Carlo (HMC) algorithm. Default is 3.
- `βₒ::Number`: The initial inverse temperature for the tempering schedule.
  Default is 0.3f0.
- `∇U_kwargs::Union{Dict,NamedTuple}`: Additional keyword arguments to be passed
  to the `∇potential_energy` function. Default is a NamedTuple with
  `reconstruction_loglikelihood` and `latent_logprior`.
- `tempering_schedule::Function`: The function to compute the inverse
  temperature at each step in the HMC algorithm. Defaults to
  `quadratic_tempering`. This function must take three arguments: First, `βₒ`,
  an initial inverse temperature, second, `k`, the current step in the tempering
  schedule, and third, `K`, the total number of steps in the tempering schedule.

# Returns
- A `NamedTuple` with the following keys: 
    - `z_init`: The initial latent variable. 
    - `ρ_init`: The initial momentum variable. 
    - `z_final`: The final latent variable after `K` leapfrog steps. 
    - `ρ_final`: The final momentum variable after `K` leapfrog steps. 
- The decoder output at the final latent variable is also returned. Note: This
  is not in the same named tuple as the other outputs, but as a separate output.

# Description
The function first samples a random momentum variable `γₒ` from a standard
normal distribution and scales it by the inverse square root of the initial
inverse temperature `βₒ` to obtain the initial momentum variable `ρₒ`. Then, it
performs `K` leapfrog steps, each followed by a tempering step, to generate a
new sample from the latent space.

# Note
Ensure the input data `x` and the initial latent variable `zₒ` match the
expected input dimensionality for the HVAE model.
"""
function leapfrog_tempering_step(
    x::AbstractArray,
    zₒ::AbstractVecOrMat,
    hvae::HVAE;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    K::Int=3,
    βₒ::Number=0.3f0,
    ∇U_kwargs::Union{Dict,NamedTuple}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        latent_logprior=spherical_logprior,
    ),
    tempering_schedule::Function=quadratic_tempering,
)
    # Compute the output of the decoder
    decoder_output = hvae.vae.decoder(zₒ)

    # Extract latent-space dimensionality
    ldim = size(zₒ, 1)

    # Sample γₒ ~ N(0, I)
    γₒ = ChainRulesCore.@ignore_derivatives randn_sample(zₒ)

    # Define ρₒ = γₒ / √βₒ
    ρₒ = γₒ ./ √(βₒ)

    # Define initial value of z and ρ before loop
    zₖ₋₁ = deepcopy(zₒ)
    ρₖ₋₁ = deepcopy(ρₒ)
    decoderₖ₋₁ = deepcopy(decoder_output)

    # Loop over K steps
    for k = 1:K
        # 1) Leapfrog step
        zₖ, ρₖ, decoderₖ = leapfrog_step(
            x, zₖ₋₁, ρₖ₋₁, hvae.vae.decoder, decoder_output;
            ϵ=ϵ, ∇U_kwargs=∇U_kwargs
        )

        # 2) Tempering step
        # Compute previous step's inverse temperature
        βₖ₋₁ = tempering_schedule(βₒ, k - 1, K)
        # Compute current step's inverse temperature
        βₖ = tempering_schedule(βₒ, k, K)

        # Update momentum variable with tempering Update zₖ₋₁, ρₖ₋₁ for next
        # iteration. The momentum variable is updated with tempering. Also, note
        # this is the last step as well, thus we return zₖ₋₁, ρₖ₋₁ as the final
        # points.
        zₖ₋₁ = zₖ
        ρₖ₋₁ = ρₖ .* √(βₖ₋₁ / βₖ)
        # Update decoderₖ₋₁ for next iteration
        decoderₖ₋₁ = decoderₖ
    end # for

    return (
        z_init=zₒ,
        ρ_init=ρₒ,
        z_final=zₖ₋₁,
        ρ_final=ρₖ₋₁,
    ), decoderₖ₋₁
end # function

# ==============================================================================
# Forward pass methods for HVAE with Hamiltonian steps
# ==============================================================================

@doc raw"""
    (hvae::HVAE{VAE{E,D}})(
        x::AbstractArray;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        K::Int=3,
        βₒ::Number=0.3f0,
        ∇U_kwargs::Union{Dict,NamedTuple}=(
                reconstruction_loglikelihood=reconstruction_loglikelihood,
                latent_logprior=spherical_logprior,
        ),
        tempering_schedule::Function=quadratic_tempering,
        latent::Bool=false,
    ) where {E<:AbstractGaussianLogEncoder,D<:AbstractVariationalDecoder}

Run the Hamiltonian Variational Autoencoder (HVAE) on the given input.

# Arguments
- `x::AbstractArray`: The input to the HVAE. If `Vector`, it represents a single
  data point. If `Array`, the last dimension must contain each of the data
  points.

# Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}=0.0001`: The step size for the leapfrog
  steps in the HMC part of the HVAE. If it is a scalar, the same step size is
  used for all dimensions. If it is an array, each element corresponds to the
  step size for a specific dimension.
- `K::Int=3`: The number of leapfrog steps to perform in the Hamiltonian Monte
  Carlo (HMC) part of the HVAE.
- `βₒ::Number=0.3f0`: The initial inverse temperature for the tempering
  schedule.
- `∇U_kwargs::Union{Dict,NamedTuple}`: Additional keyword arguments to be passed
  to the `∇potential_energy` function. Default is a NamedTuple with
  `reconstruction_loglikelihood` and `latent_logprior`.
- `tempering_schedule::Function=quadratic_tempering`: The function to compute
    the tempering schedule in the HVAE.
- `latent::Bool=false`: If `true`, the function returns a NamedTuple containing
  the outputs of the encoder and decoder, and the final state of the phase space
  after the leapfrog and tempering steps. If `false`, the function only returns
  the output of the decoder.

# Returns
If `latent=true`, the function returns a NamedTuple with the following fields:
- `encoder`: The outputs of the encoder.
- `decoder`: The output of the decoder.
- `phase_space`: The final state of the phase space after the leapfrog and
    tempering steps.

If `latent=false`, the function only returns the output of the decoder.

# Description
This function runs the HVAE on the given input. It first passes the input
through the encoder to obtain the mean and log standard deviation of the latent
space. It then uses the reparameterization trick to sample from the latent
space. After that, it performs the leapfrog and tempering steps to refine the
sample from the latent space. Finally, it passes the refined sample through the
decoder to obtain the output.

# Notes
Ensure that the dimensions of `x` match the input dimensions of the HVAE, and
that the dimensions of `ϵ` match the dimensions of the latent space.
"""
function (hvae::HVAE{VAE{E,D}})(
    x::AbstractArray;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    K::Int=3,
    βₒ::Number=0.3f0,
    ∇U_kwargs::Union{Dict,NamedTuple}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        latent_logprior=spherical_logprior,
    ),
    tempering_schedule::Function=quadratic_tempering,
    latent::Bool=false,
) where {E<:AbstractGaussianLogEncoder,D<:AbstractVariationalDecoder}
    # Run input through encoder
    encoder_output = hvae.vae.encoder(x)

    # Run reparametrize trick to generate latent variable zₒ
    zₒ = reparameterize(hvae.vae.encoder, encoder_output)

    # Initial decoder output
    decoder_output = hvae.vae.decoder(zₒ)

    # Run leapfrog and tempering steps
    phase_space, decoder_update = leapfrog_tempering_step(
        x, zₒ, hvae.vae.decoder, decoder_output;
        K=K, ϵ=ϵ, βₒ=βₒ, ∇U_kwargs=∇U_kwargs,
        tempering_schedule=tempering_schedule
    )

    # Check if latent variables should be returned
    if latent
        return (
            encoder=encoder_output,
            decoder=decoder_update,
            phase_space=phase_space,
        )
    else
        return decoder_update
    end # if
end # function

# ==============================================================================
# Hamiltonian ELBO
# ==============================================================================

@doc raw"""
    _log_p̄(
        x::AbstractArray,
        hvae::HVAE{VAE{E,D}},
        hvae_outputs::NamedTuple;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        logprior::Function=spherical_logprior,
        prefactor::AbstractArray=ones(Float32, 3),
    )

This is an internal function used in `hamiltonian_elbo` to compute the numerator
of the unbiased estimator of the marginal likelihood. The function computes the
sum of the log likelihood of the data given the latent variables, the log prior
of the latent variables, and the log prior of the momentum variables.

        log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)

# Arguments
- `x::AbstractArray`: The input data. If `Array`, the last dimension must
  contain each of the data points.
- `hvae::HVAE{<:VAE{<:AbstractGaussianEncoder,<:AbstractGaussianLogDecoder}}`:
  The Hamiltonian Variational Autoencoder (HVAE) model.
- `hvae_outputs::NamedTuple`: The outputs of the HVAE, including the final
  latent variables `zₖ` and the final momentum variables `ρₖ`.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function`: The function to compute the log
  likelihood of the data given the latent variables. Default is
  `decoder_loglikelihood`.
- `logprior::Function`: The function to compute the log prior of the latent
  variables. Default is `spherical_logprior`.
- `prefactor::AbstractArray`: A 3-element array to scale the log likelihood, log
  prior of the latent variables, and log prior of the momentum variables.
  Default is an array of ones.

# Returns
- `log_p̄::AbstractVector`: The first term of the log of the unbiased estimator
  of the marginal likelihood for each data point.

# Note
This is an internal function and should not be called directly. It is used as
part of the `hamiltonian_elbo` function.
"""
function _log_p̄(
    x::AbstractArray,
    hvae::HVAE,
    hvae_outputs::NamedTuple;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    logprior::Function=spherical_logprior,
    prefactor::AbstractArray=ones(Float32, 3),
)
    # Check that prefactor is the correct size
    if length(prefactor) != 3
        throw(ArgumentError("prefactor must be a 3-element array"))
    end # if

    # log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)   

    # Compute log p(x | zₖ)
    log_p_x_given_zₖ = reconstruction_loglikelihood(
        x,
        hvae_outputs.phase_space.z_final,
        hvae.vae.decoder,
        hvae_outputs.decoder
    ) .* prefactor[1]

    # Compute log p(ρₖ | zₖ)
    log_p_ρₖ_given_zₖ = logprior(
        hvae_outputs.phase_space.ρ_final
    ) .* prefactor[2]

    # Compute log p(zₖ)
    log_p_zₖ = logprior(
        hvae_outputs.phase_space.z_final
    ) .* prefactor[3]

    return log_p_x_given_zₖ + log_p_zₖ + log_p_ρₖ_given_zₖ
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    _log_q̄(
        hvae::HVAE,
        hvae_outputs::NamedTuple,
        βₒ::Number;
        logprior::Function=spherical_logprior,
        prefactor::AbstractArray=ones(Float32, 3),
    )

This is an internal function used in `hamiltonian_elbo` to compute the second
term of the unbiased estimator of the marginal likelihood. The function computes
the sum of the log posterior of the initial latent variables and the log prior
of the initial momentum variables, minus a term that depends on the
dimensionality of the latent space and the initial temperature.

    log q̄ = log q(zₒ | x) + log p(ρₒ | zₒ) - d/2 log(βₒ)

# Arguments
- `hvae::HVAE`: The Hamiltonian Variational Autoencoder (HVAE) model.
- `hvae_outputs::NamedTuple`: The outputs of the HVAE, including the initial
  latent variables `zₒ` and the initial momentum variables `ρₒ`.
- `βₒ::Number`: The initial temperature for the tempering steps.

# Optional Keyword Arguments
- `logprior::Function`: The function to compute the log prior of the momentum
  variables. Default is `spherical_logprior`.
- `prefactor::AbstractArray`: A 3-element array to scale the log posterior of
  the initial latent variables, log prior of the initial momentum variables, and
  the tempering Jacobian term. Default is an array of ones.

# Returns
- `log_q̄::Vector`: The second term of the log of the unbiased estimator of the
  marginal likelihood for each data point.

# Note
This is an internal function and should not be called directly. It is used as
part of the `hamiltonian_elbo` function.
"""
function _log_q̄(
    hvae::HVAE,
    hvae_outputs::NamedTuple,
    βₒ::Number;
    logprior::Function=spherical_logprior,
    prefactor::AbstractArray=ones(Float32, 3),
)
    # Check that prefactor is the correct size
    if length(prefactor) != 3
        throw(ArgumentError("prefactor must be a 2-element array"))
    end # if
    # log q̄ = log q(zₒ | x) + log p(ρₒ | zₒ) - d/2 log(βₒ)

    # Compute log q(zₒ | x).
    log_q_zₒ_given_x = encoder_logposterior(
        hvae_outputs.phase_space.z_init,
        hvae.vae.encoder,
        hvae_outputs.encoder
    ) .* prefactor[1]

    # Compute log p(ρₒ|zₒ)
    log_p_ρₒ_given_zₒ = logprior(
        hvae_outputs.phase_space.ρ_init,
    ) .* prefactor[2]

    # Compute tempering Jacobian term
    tempering_jacobian = prefactor[3] * 0.5f0 *
                         size(hvae_outputs.phase_space.z_init, 1) * log(βₒ)

    return log_q_zₒ_given_x + log_p_ρₒ_given_zₒ .- tempering_jacobian
end # function

# ------------------------------------------------------------------------------
# Hamiltonian ELBO computation
# ------------------------------------------------------------------------------

@doc raw"""
    hamiltonian_elbo(
        hvae::HVAE,
        x::AbstractArray;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        K::Int=3,
        βₒ::Number=0.3f0,
        ∇U_kwargs::Union{Dict,NamedTuple}=(
            reconstruction_loglikelihood=decoder_loglikelihood,
            latent_logprior=spherical_logprior,
        ),
        tempering_schedule::Function=quadratic_tempering,
        return_outputs::Bool=false,
        logp_prefactor::AbstractArray=ones(Float32, 3),
        logq_prefactor::AbstractArray=ones(Float32, 3),
    )

Compute the Hamiltonian Monte Carlo (HMC) estimate of the evidence lower bound
(ELBO) for a Hamiltonian Variational Autoencoder (HVAE).

This function takes as input an HVAE and a vector of input data `x`. It performs
`K` HMC steps with a leapfrog integrator and a tempering schedule to estimate
the ELBO. The ELBO is computed as the difference between the `log p̄` and `log
q̄` as

elbo = mean(log p̄ - log q̄),

# Arguments
- `hvae::HVAE`: The HVAE used to encode the input data and decode the latent
  space.
- `x::AbstractArray`: The input data. If `Array`, the last dimension must
  contain each of the data points.

## Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}`: The step size for the leapfrog
  integrator (default is 0.01).
- `K::Int`: The number of HMC steps (default is 3).
- `βₒ::Number`: The initial inverse temperature (default is 0.3).
- `∇U_kwargs::Union{Dict,NamedTuple}`: Additional keyword arguments to be passed
  to the `∇potential_energy` function. Defaults to a NamedTuple with
  `:reconstruction_loglikelihood` set to `decoder_loglikelihood` and
  `:latent_logprior` set to `spherical_logprior`.
- `tempering_schedule::Function`: The tempering schedule function used in the
  HMC (default is `quadratic_tempering`).
- `return_outputs::Bool`: Whether to return the outputs of the HVAE. Defaults to
  `false`. NOTE: This is necessary to avoid computing the forward pass twice
  when computing the loss function with regularization.
- `logp_prefactor::AbstractArray`: A 3-element array to scale the log
  likelihood, log prior of the latent variables, and log prior of the momentum
  variables. Default is an array of ones.
- `logq_prefactor::AbstractArray`: A 3-element array to scale the log posterior
  of the initial latent variables, log prior of the initial momentum variables,
  and the tempering Jacobian term. Default is an array of ones.

# Returns
- `elbo::Number`: The HMC estimate of the ELBO. If `return_outputs` is `true`,
  also returns the outputs of the HVAE.
"""
function hamiltonian_elbo(
    hvae::HVAE,
    x::AbstractArray;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    K::Int=3,
    βₒ::Number=0.3f0,
    ∇U_kwargs::Union{Dict,NamedTuple}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        latent_logprior=spherical_logprior,
    ),
    tempering_schedule::Function=quadratic_tempering,
    return_outputs::Bool=false,
    logp_prefactor::AbstractArray=ones(Float32, 3),
    logq_prefactor::AbstractArray=ones(Float32, 3),
)
    # Forward Pass (run input through reconstruct function)
    hvae_outputs = hvae(
        x;
        K=K, ϵ=ϵ, βₒ=βₒ,
        ∇U_kwargs=∇U_kwargs,
        tempering_schedule=tempering_schedule,
        latent=true
    )

    # Compute log evidence estimate log π̂(x) = log p̄ - log q̄

    # log p̄ = log p(x, zₖ) + log p(ρₖ)
    # log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)
    log_p = _log_p̄(
        x, hvae, hvae_outputs;
        prefactor=logp_prefactor,
        reconstruction_loglikelihood=∇U_kwargs.reconstruction_loglikelihood,
        logprior=∇U_kwargs.latent_logprior,
    )

    # log q̄ = log q(zₒ) + log p(ρₒ) - d/2 log(βₒ)
    log_q = _log_q̄(
        hvae, hvae_outputs, βₒ;
        prefactor=logq_prefactor,
        logprior=∇U_kwargs.latent_logprior
    )

    if return_outputs
        return StatsBase.mean(log_p - log_q), hvae_outputs
    else
        return StatsBase.mean(log_p - log_q)
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    hamiltonian_elbo(
        hvae::HVAE,
        x_in::AbstractArray,
        x_out::AbstractArray;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        K::Int=3,
        βₒ::Number=0.3f0,
        ∇U_kwargs::Union{Dict,NamedTuple}=(
            reconstruction_loglikelihood=decoder_loglikelihood,
            latent_logprior=spherical_logprior,
        ),
        tempering_schedule::Function=quadratic_tempering,
        return_outputs::Bool=false,
        logp_prefactor::AbstractArray=ones(Float32, 3),
        logq_prefactor::AbstractArray=ones(Float32, 3),
    )

Compute the Hamiltonian Monte Carlo (HMC) estimate of the evidence lower bound
(ELBO) for a Hamiltonian Variational Autoencoder (HVAE).

This function takes as input an HVAE and a vector of input data `x`. It performs
`K` HMC steps with a leapfrog integrator and a tempering schedule to estimate
the ELBO. The ELBO is computed as the difference between the `log p̄` and `log
q̄` as

elbo = mean(log p̄ - log q̄),

# Arguments
- `hvae::HVAE`: The HVAE used to encode the input data and decode the latent
  space.
- `x_in::AbstractArray`: The input data. If `Array`, the last dimension must
  contain each of the data points.
- `x_out::AbstractArray`: The data against which the reconstruction is compared.
  If `Array`, the last dimension must contain each of the data points.

## Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}`: The step size for the leapfrog
  integrator (default is 0.01).
- `K::Int`: The number of HMC steps (default is 3).
- `βₒ::Number`: The initial inverse temperature (default is 0.3).
- `∇U_kwargs::Union{Dict,NamedTuple}`: Additional keyword arguments to be passed
  to the `∇potential_energy` function. Defaults to a NamedTuple with
  `:reconstruction_loglikelihood` set to `decoder_loglikelihood` and
  `:latent_logprior` set to `spherical_logprior`.
- `tempering_schedule::Function`: The tempering schedule function used in the
  HMC (default is `quadratic_tempering`).
- `return_outputs::Bool`: Whether to return the outputs of the HVAE. Defaults to
  `false`. NOTE: This is necessary to avoid computing the forward pass twice
  when computing the loss function with regularization.
- `logp_prefactor::AbstractArray`: A 3-element array to scale the log
  likelihood, log prior of the latent variables, and log prior of the momentum
  variables. Default is an array of ones.
- `logq_prefactor::AbstractArray`: A 3-element array to scale the log posterior
  of the initial latent variables, log prior of the initial momentum variables,
  and the tempering Jacobian term. Default is an array of ones.

# Returns
- `elbo::Number`: The HMC estimate of the ELBO. If `return_outputs` is `true`,
  also returns the outputs of the HVAE.
"""
function hamiltonian_elbo(
    hvae::HVAE,
    x_in::AbstractArray,
    x_out::AbstractArray;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    K::Int=3,
    βₒ::Number=0.3f0,
    ∇U_kwargs::Union{Dict,NamedTuple}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        latent_logprior=spherical_logprior,
    ),
    tempering_schedule::Function=quadratic_tempering,
    return_outputs::Bool=false,
    logp_prefactor::AbstractArray=ones(Float32, 3),
    logq_prefactor::AbstractArray=ones(Float32, 3),
)
    # Forward Pass (run input through reconstruct function)
    hvae_outputs = hvae(
        x_in;
        K=K, ϵ=ϵ, βₒ=βₒ,
        ∇U_kwargs=∇U_kwargs,
        tempering_schedule=tempering_schedule,
        latent=true
    )

    # Compute log evidence estimate log π̂(x) = log p̄ - log q̄

    # log p̄ = log p(x, zₖ) + log p(ρₖ)
    # log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)
    log_p = _log_p̄(
        x_out, hvae, hvae_outputs;
        prefactor=logp_prefactor,
        reconstruction_loglikelihood=∇U_kwargs.reconstruction_loglikelihood,
        logprior=∇U_kwargs.latent_logprior,
    )

    # log q̄ = log q(zₒ) + log p(ρₒ) - d/2 log(βₒ)
    log_q = _log_q̄(
        hvae, hvae_outputs, βₒ;
        prefactor=logq_prefactor,
        logprior=∇U_kwargs.latent_logprior
    )

    if return_outputs
        return StatsBase.mean(log_p - log_q), hvae_outputs
    else
        return StatsBase.mean(log_p - log_q)
    end # if
end # function

# ==============================================================================
# HVAE loss function
# ==============================================================================

@doc raw"""
    loss(
        hvae::HVAE,
        x::AbstractArray;
        K::Int=3,
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        βₒ::Number=0.3f0,
        ∇U_kwargs::Union{Dict,NamedTuple}=(
            reconstruction_loglikelihood=reconstruction_loglikelihood,
            latent_logprior=spherical_logprior,
        ),
        tempering_schedule::Function=quadratic_tempering,
        reg_function::Union{Function,Nothing}=nothing,
        reg_kwargs::Union{NamedTuple,Dict}=Dict(),
        reg_strength::Float32=1.0f0,
        logp_prefactor::AbstractArray=ones(Float32, 3),
        logq_prefactor::AbstractArray=ones(Float32, 3),
    )

Compute the loss for a Hamiltonian Variational Autoencoder (HVAE).

# Arguments
- `hvae::HVAE`: The HVAE used to encode the input data and decode the latent
  space.
- `x::AbstractArray`: Input data to the HVAE encoder. The last dimension is
  taken as having each of the samples in a batch.

## Optional Keyword Arguments
- `K::Int`: The number of HMC steps (default is 3).
- `ϵ::Union{<:Number,<:AbstractVector}`: The step size for the leapfrog
  integrator (default is 0.001).
- `βₒ::Number`: The initial inverse temperature (default is 0.3).
- `∇U_kwargs::Union{Dict,NamedTuple}`: Additional keyword arguments to be passed
  to the `∇potential_energy` function.
- `tempering_schedule::Function`: The tempering schedule function used in the
  HMC (default is `quadratic_tempering`).
- `reg_function::Union{Function, Nothing}=nothing`: A function that computes the
  regularization term based on the VAE outputs. This function must take as input
  the VAE outputs and the keyword arguments provided in `reg_kwargs`.
- `reg_kwargs::Union{NamedTuple,Dict}=Dict()`: Keyword arguments to pass to the
  regularization function.
- `reg_strength::Float32=1.0f0`: The strength of the regularization term.
- `logp_prefactor::AbstractArray`: A 3-element array to scale the log
  likelihood, log prior of the latent variables, and log prior of the momentum
  variables. Default is an array of ones.
- `logq_prefactor::AbstractArray`: A 3-element array to scale the log posterior
  of the initial latent variables, log prior of the initial momentum variables,
  and the tempering Jacobian term. Default is an array of ones.

# Returns
- The computed loss.
"""
function loss(
    hvae::HVAE,
    x::AbstractArray;
    K::Int=3,
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    βₒ::Number=0.3f0,
    ∇U_kwargs::Union{Dict,NamedTuple}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        latent_logprior=spherical_logprior,
    ),
    tempering_schedule::Function=quadratic_tempering,
    reg_function::Union{Function,Nothing}=nothing,
    reg_kwargs::Union{NamedTuple,Dict}=Dict(),
    reg_strength::Float32=1.0f0,
    logp_prefactor::AbstractArray=ones(Float32, 3),
    logq_prefactor::AbstractArray=ones(Float32, 3),
)
    # Check if there is regularization 
    if reg_function !== nothing
        # Compute ELBO
        elbo, hvae_outputs = hamiltonian_elbo(
            hvae, x;
            K=K, ϵ=ϵ, βₒ=βₒ,
            ∇U_kwargs=∇U_kwargs,
            tempering_schedule=tempering_schedule,
            return_outputs=true,
            logp_prefactor=logp_prefactor,
            logq_prefactor=logq_prefactor
        )

        # Compute regularization
        reg_term = reg_function(hvae_outputs; reg_kwargs...)

        return -elbo + reg_strength * reg_term
    else
        # Compute ELBO
        return -hamiltonian_elbo(
            hvae, x;
            K=K, ϵ=ϵ, βₒ=βₒ,
            ∇U_kwargs=∇U_kwargs,
            tempering_schedule=tempering_schedule,
            logp_prefactor=logp_prefactor,
            logq_prefactor=logq_prefactor
        )
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    loss(
        hvae::HVAE,
        x_in::AbstractArray,
        x_out::AbstractArray;
        K::Int=3,
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        βₒ::Number=0.3f0,
        ∇U_kwargs::Union{Dict,NamedTuple}=(
            reconstruction_loglikelihood=reconstruction_loglikelihood,
            latent_logprior=spherical_logprior,
        ),
        tempering_schedule::Function=quadratic_tempering,
        reg_function::Union{Function,Nothing}=nothing,
        reg_kwargs::Union{NamedTuple,Dict}=Dict(),
        reg_strength::Float32=1.0f0,
        logp_prefactor::AbstractArray=ones(Float32, 3),
        logq_prefactor::AbstractArray=ones(Float32, 3),
    )

Compute the loss for a Hamiltonian Variational Autoencoder (HVAE).

# Arguments
- `hvae::HVAE`: The HVAE used to encode the input data and decode the latent
  space.
- `x_in::AbstractArray`: Input data to the HVAE encoder. The last dimension is
  taken as having each of the samples in a batch.
- `x_out::AbstractArray`: The data against which the reconstruction is compared.
  If `Array`, the last dimension must contain each of the data points.

## Optional Keyword Arguments
- `K::Int`: The number of HMC steps (default is 3).
- `ϵ::Union{<:Number,<:AbstractVector}`: The step size for the leapfrog
  integrator (default is 0.001).
- `βₒ::Number`: The initial inverse temperature (default is 0.3).
- `∇U_kwargs::Union{Dict,NamedTuple}`: Additional keyword arguments to be passed
  to the `∇potential_energy` function.
- `tempering_schedule::Function`: The tempering schedule function used in the
  HMC (default is `quadratic_tempering`).
- `reg_function::Union{Function, Nothing}=nothing`: A function that computes the
  regularization term based on the VAE outputs. This function must take as input
  the VAE outputs and the keyword arguments provided in `reg_kwargs`.
- `reg_kwargs::Union{NamedTuple,Dict}=Dict()`: Keyword arguments to pass to the
  regularization function.
- `reg_strength::Float32=1.0f0`: The strength of the regularization term.
- `logp_prefactor::AbstractArray`: A 3-element array to scale the log
  likelihood, log prior of the latent variables, and log prior of the momentum
  variables. Default is an array of ones.
- `logq_prefactor::AbstractArray`: A 3-element array to scale the log posterior
  of the initial latent variables, log prior of the initial momentum variables,
  and the tempering Jacobian term. Default is an array of ones.

# Returns
- The computed loss.
"""
function loss(
    hvae::HVAE,
    x_in::AbstractArray,
    x_out::AbstractArray;
    K::Int=3,
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    βₒ::Number=0.3f0,
    ∇U_kwargs::Union{Dict,NamedTuple}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        latent_logprior=spherical_logprior,
    ),
    tempering_schedule::Function=quadratic_tempering,
    reg_function::Union{Function,Nothing}=nothing,
    reg_kwargs::Union{NamedTuple,Dict}=Dict(),
    reg_strength::Float32=1.0f0,
    logp_prefactor::AbstractArray=ones(Float32, 3),
    logq_prefactor::AbstractArray=ones(Float32, 3),
)
    # Check if there is regularization 
    if reg_function !== nothing
        # Compute ELBO
        elbo, hvae_outputs = hamiltonian_elbo(
            hvae, x_in, x_out;
            K=K, ϵ=ϵ, βₒ=βₒ,
            ∇U_kwargs=∇U_kwargs,
            tempering_schedule=tempering_schedule,
            return_outputs=true,
            logp_prefactor=logp_prefactor,
            logq_prefactor=logq_prefactor
        )

        # Compute regularization
        reg_term = reg_function(hvae_outputs; reg_kwargs...)

        return -elbo + reg_strength * reg_term
    else
        # Compute ELBO
        return -hamiltonian_elbo(
            hvae, x_in, x_out;
            K=K, ϵ=ϵ, βₒ=βₒ,
            ∇U_kwargs=∇U_kwargs,
            tempering_schedule=tempering_schedule,
            logp_prefactor=logp_prefactor,
            logq_prefactor=logq_prefactor
        )
    end # if
end # function

# ==============================================================================
# HVAE training
# ==============================================================================

@doc raw"""
    train!(
        hvae::HVAE, 
        x::AbstractArray, 
        opt::NamedTuple; 
        loss_function::Function=loss, 
        loss_kwargs::Union{NamedTuple,Dict}=Dict(),
        verbose::Bool=false,
        loss_return::Bool=false,
    )

Customized training function to update parameters of a Hamiltonian Variational
Autoencoder given a specified loss function.

# Arguments
- `hvae::HVAE`: A struct containing the elements of a Hamiltonian Variational
  Autoencoder.
- `x::AbstractArray`: Input data to the HVAE encoder. The last dimension is
  taken as having each of the samples in a batch.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Optimisers.update!`.

# Optional Keyword Arguments
- `loss_function::Function=loss`: The loss function used for training. It should
  accept the HVAE model, data `x`, and keyword arguments in that order.
- `loss_kwargs::Dict=Dict()`: Arguments for the loss function. These might
  include parameters like `K`, `ϵ`, `βₒ`, `steps`, `∇H`, `∇H_kwargs`,
  `tempering_schedule`, `reg_function`, `reg_kwargs`, `reg_strength`, depending
  on the specific loss function in use.
- `verbose::Bool=false`: Whether to print the loss at each iteration.
- `loss_return::Bool=false`: Whether to return the loss at each iteration.

# Description
Trains the HVAE by:
1. Computing the gradient of the loss w.r.t the HVAE parameters.
2. Updating the HVAE parameters using the optimizer.
3. Updating the metric parameters.
"""
function train!(
    hvae::HVAE,
    x::AbstractArray,
    opt::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    verbose::Bool=false,
    loss_return::Bool=false,
)
    # Compute VAE gradient
    L, ∇L = CUDA.allowscalar() do
        Flux.withgradient(hvae) do hvae_model
            loss_function(hvae_model, x; loss_kwargs...)
        end # do block
    end

    # Update parameters
    Flux.Optimisers.update!(opt, hvae, ∇L[1])

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
# ------------------------------------------------------------------------------

@doc raw"""
    train!(
        hvae::HVAE, 
        x_in::AbstractArray,
        x_out::AbstractArray,
        opt::NamedTuple; 
        loss_function::Function=loss, 
        loss_kwargs::Union{NamedTuple,Dict}=Dict(),
        verbose::Bool=false,
        loss_return::Bool=false,
    )

Customized training function to update parameters of a Hamiltonian Variational
Autoencoder given a specified loss function.

# Arguments
- `hvae::HVAE`: A struct containing the elements of a Hamiltonian Variational
  Autoencoder.
- `x_in::AbstractArray`: Input data to the HVAE encoder. The last dimension is
  taken as having each of the samples in a batch.
- `x_out::AbstractArray`: Target data to compute the reconstruction error. The
  last dimension is taken as having each of the samples in a batch.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Optimisers.update!`.

# Optional Keyword Arguments
- `loss_function::Function=loss`: The loss function used for training. It should
  accept the HVAE model, data `x`, and keyword arguments in that order.
- `loss_kwargs::Dict=Dict()`: Arguments for the loss function. These might
  include parameters like `K`, `ϵ`, `βₒ`, `steps`, `∇H`, `∇H_kwargs`,
  `tempering_schedule`, `reg_function`, `reg_kwargs`, `reg_strength`, depending
  on the specific loss function in use.
- `verbose::Bool=false`: Whether to print the loss at each iteration.
- `loss_return::Bool=false`: Whether to return the loss at each iteration.

# Description
Trains the HVAE by:
1. Computing the gradient of the loss w.r.t the HVAE parameters.
2. Updating the HVAE parameters using the optimizer.
3. Updating the metric parameters.
"""
function train!(
    hvae::HVAE,
    x_in::AbstractArray,
    x_out::AbstractArray,
    opt::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    verbose::Bool=false,
    loss_return::Bool=false,
)
    # Compute VAE gradient
    L, ∇L = CUDA.allowscalar() do
        Flux.withgradient(hvae) do hvae_model
            loss_function(hvae_model, x_in, x_out; loss_kwargs...)
        end # do block
    end

    # Update parameters
    Flux.Optimisers.update!(opt, hvae, ∇L[1])

    # Update metric
    update_metric!(hvae)

    # Check if loss should be returned
    if loss_return
        return L
    end # if

    # Check if loss should be printed
    if verbose
        println("Loss: ", L)
    end # if
end # function