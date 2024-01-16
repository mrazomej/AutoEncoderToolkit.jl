# Import ML libraries
import Flux
import Zygote

# Import basic math
import Random
import StatsBase
import Distributions

# Import Abstract Types
using ..AutoEncode: AbstractVariationalAutoEncoder,
    AbstractVariationalEncoder, AbstractGaussianEncoder,
    AbstractGaussianLogEncoder,
    AbstractVariationalDecoder, AbstractGaussianDecoder,
    AbstractGaussianLogDecoder, AbstractGaussianLinearDecoder,
    Float32Array

# Import Concrete Encoder Types
using ..AutoEncode: JointLogEncoder

# Import Concrete Decoder Types
using ..AutoEncode: SimpleDecoder,
    JointLogDecoder, SplitLogDecoder,
    JointDecoder, SplitDecoder

# Import Concrete VAE type
using ..AutoEncode: VAE

using ..VAEs: reparameterize

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
# Functions to compute decoder log likelihood
# ==============================================================================

@doc raw"""
    decoder_loglikelihood(
        decoder::SimpleDecoder,
        x::AbstractVector{T},
        z::AbstractVector{T};
        σ::T=1.0f0,
    ) where {T<:Float32}

Computes the log-likelihood of the observed data `x` given the latent variable
`z` under a Gaussian distribution with mean given by the decoder and a specified
standard deviation.

# Arguments
- `decoder::SimpleDecoder`: The decoder of the VAE, which is used to compute the
  mean of the Gaussian distribution.
- `x::AbstractVector{T}`: The observed data for which the log-likelihood is to
  be computed.
- `z::AbstractVector{T}`: The latent variable associated with the observed data
  `x`.

# Optional Keyword Arguments
- `σ::T=1.0f0`: The standard deviation of the Gaussian distribution. Defaults to
  `1.0f0`.

# Returns
- `loglikelihood::Float32`: The computed log-likelihood of the observed data
  `x` given the latent variable `z`.

# Description
The function computes the log-likelihood of the observed data `x` given the
latent variable `z` under a Gaussian distribution. The mean of the Gaussian
distribution is computed by passing `z` through the `decoder`. The standard
deviation of the Gaussian distribution is specified by the `σ` argument. The
log-likelihood is computed using the formula for the log-likelihood of a
Gaussian distribution.

# Note
Ensure the dimensions of `x` and `z` match the expected input and output
dimensionality of the `decoder`.
"""
function decoder_loglikelihood(
    decoder::SimpleDecoder,
    x::AbstractVector{T},
    z::AbstractVector{T};
    σ::T=1.0f0,
) where {T<:Float32}
    # Compute the mean of the decoder output given the latent variable z
    μ = decoder(z).µ

    # Compute log-likelihood
    loglikelihood = -0.5f0 * sum(abs2, (x - μ) / σ) -
                    0.5f0 * length(x) * (2.0f0 * log(σ) + log(2.0f0π))
    return loglikelihood
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    decoder_loglikelihood(
        decoder::AbstractGaussianLogDecoder,
        x::AbstractVector{T},
        z::AbstractVector{T}
    ) where {T<:Float32}

Computes the log-likelihood of the observed data `x` given the latent variable
`z` under a Gaussian distribution with mean and standard deviation given by the
decoder.

# Arguments
- `decoder::AbstractGaussianLogDecoder`: The decoder of the VAE, which is used
  to compute the mean and log standard deviation of the Gaussian distribution.
- `x::AbstractVector{T}`: The observed data for which the log-likelihood is to
  be computed.
- `z::AbstractVector{T}`: The latent variable associated with the observed data
  `x`.

# Returns
- `loglikelihood::Float32`: The computed log-likelihood of the observed data
  `x` given the latent variable `z`.

# Description
The function computes the log-likelihood of the observed data `x` given the
latent variable `z` under a Gaussian distribution. The mean and log standard
deviation of the Gaussian distribution are computed by passing `z` through the
`decoder`. The standard deviation is then computed by exponentiating the log
standard deviation. The log-likelihood is computed using the formula for the
log-likelihood of a Gaussian distribution.

# Note
Ensure the dimensions of `x` and `z` match the expected input and output
dimensionality of the `decoder`.
"""
function decoder_loglikelihood(
    decoder::AbstractGaussianLogDecoder,
    x::AbstractVector{T},
    z::AbstractVector{T};
) where {T<:Float32}
    # Compute the mean and log standard deviation of the decoder output given the
    # latent variable z
    μ, logσ = decoder(z)

    # Compute standard deviation
    σ = exp.(logσ)

    # Compute log-likelihood
    loglikelihood = -0.5f0 * sum(abs2, (x - μ) ./ σ) -
                    sum(logσ) -
                    0.5f0 * length(x) * log(2.0f0π)

    return loglikelihood
end # function

@doc raw"""
    decoder_loglikelihood(
        decoder::AbstractGaussianLinearDecoder,
        x::AbstractVector{T},
        z::AbstractVector{T}
    ) where {T<:Float32}

Computes the log-likelihood of the observed data `x` given the latent variable
`z` under a Gaussian distribution with mean and standard deviation given by the
decoder.

# Arguments
- `decoder::AbstractGaussianLinearDecoder`: The decoder of the VAE, which is
  used to compute the mean and standard deviation of the Gaussian distribution.
- `x::AbstractVector{T}`: The observed data for which the log-likelihood is to
  be computed.
- `z::AbstractVector{T}`: The latent variable associated with the observed data
  `x`.

# Returns
- `loglikelihood::Float32`: The computed log-likelihood of the observed data
  `x` given the latent variable `z`.

# Description
The function computes the log-likelihood of the observed data `x` given the
latent variable `z` under a Gaussian distribution. The mean and standard
deviation of the Gaussian distribution are computed by passing `z` through the
`decoder`. The log-likelihood is computed using the formula for the
log-likelihood of a Gaussian distribution.

# Note
Ensure the dimensions of `x` and `z` match the expected input and output
dimensionality of the `decoder`.
"""
function decoder_loglikelihood(
    decoder::AbstractGaussianLinearDecoder,
    x::AbstractVector{T},
    z::AbstractVector{T};
) where {T<:Float32}
    # Compute the mean and standard deviation of the decoder output given the
    # latent variable z
    μ, σ = decoder(z)

    # Compute log-likelihood
    loglikelihood = -0.5f0 * sum(abs2, (x - μ) ./ σ) -
                    sum(log, σ) -
                    0.5f0 * length(x) * log(2.0f0π)

    return loglikelihood
end # function

# ==============================================================================
# Function to compute log prior
# ==============================================================================

@doc raw"""
    spherical_logprior(
        z::AbstractVector{T},
        σ::T=1.0f0,
    ) where {T<:Float32}

Computes the log-prior of the latent variable `z` under a spherical Gaussian
distribution with zero mean and standard deviation `σ`.

# Arguments
- `z::AbstractVector{T}`: The latent variable for which the log-prior is to be
  computed.
- `σ::T=1.0f0`: The standard deviation of the spherical Gaussian distribution.
  Defaults to `1.0f0`.

# Returns
- `log_prior::Float32`: The computed log-prior of the latent variable `z`.

# Description
The function computes the log-prior of the latent variable `z` under a spherical
Gaussian distribution with zero mean and standard deviation `σ`. The log-prior
is computed using the formula for the log-prior of a Gaussian distribution.

# Note
Ensure the dimension of `z` matches the expected dimensionality of the latent
space.
"""
function spherical_logprior(
    z::AbstractVector{T},
    σ::T=1.0f0,
) where {T<:Float32}
    # Compute log-prior
    log_prior = -0.5f0 * sum(abs2, z / σ) -
                0.5f0 * length(z) * (2.0f0 * log(σ) + log(2.0f0π))

    return log_prior
end # function

# ==============================================================================
# Function to compute potential energy
# ==============================================================================

@doc raw"""
        potential_energy(
                hvae::HVAE,
                x::AbstractVector{T},
                z::AbstractVector{T};
                decoder_loglikelihood::Function=decoder_loglikelihood,
                spherical_logprior::Function=spherical_logprior
        ) where {T<:AbstractFloat}

Compute the potential energy of a Hamiltonian Variational Autoencoder (HVAE). In
the context of Hamiltonian Monte Carlo (HMC), the potential energy is defined as
the negative log-posterior. This function computes the potential energy for
given data `x` and latent variable `z`. It does this by computing the
log-likelihood of `x` under the distribution defined by
`decoder_loglikelihood(hvae.vae.decoder, x, z)`, and the log-prior of `z` under
the `spherical_logprior` distribution. The potential energy is then computed as
the negative sum of the log-likelihood and the log-prior.

# Arguments
- `hvae::HVAE`: An HVAE model that contains a decoder which maps the latent
  variables to the data space.
- `x::AbstractVector{T}`: A vector representing the data.
- `z::AbstractVector{T}`: A vector representing the latent variable.

# Optional Keyword Arguments
- `decoder_loglikelihood::Function=decoder_loglikelihood`: A function
  representing the log-likelihood function used by the decoder. The function
  must take as first input an `AbstractVariationalDecoder` struct, as second
  input a vector `x` representing the data, and as third input a vector `z`
  representing the latent variable. Default is `decoder_loglikelihood`.
- `spherical_logprior::Function=spherical_logprior`: A function representing the
  log-prior distribution used in the autoencoder. The function must take as
  single input a vector `z` representing the latent variable. Default is
  `spherical_logprior`.  

# Returns
- `energy::AbstractFloat`: The computed potential energy for the given input `x`
  and latent variable `z`.

# Example
```julia
# Define HVAE
hvae = ...build HVAE here...

# Compute the potential energy
x = rand(2)
z = rand(2)
energy = potential_energy(hvae, x, z)
```
"""
function potential_energy(
    hvae::HVAE,
    x::AbstractVector{T},
    z::AbstractVector{T};
    decoder_loglikelihood::Function=decoder_loglikelihood,
    log_prior::Function=spherical_logprior,
) where {T<:Float32}
    # Compute log-likelihood
    loglikelihood = decoder_loglikelihood(hvae.vae.decoder, x, z)

    # Compute log-prior
    logprior = log_prior(z)

    # Compute potential energy
    return -loglikelihood - logprior
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
        ∇potential_energy(
                hvae::HVAE,
                x::AbstractVector{T},
                z::AbstractVector{T};
                decoder_loglikelihood::Function=decoder_loglikelihood,
                log_prior::Function=spherical_logprior
        ) where {T<:AbstractFloat}

Compute the gradient of the potential energy of a Hamiltonian Variational
Autoencoder (HVAE) with respect to the latent variables `z` using `Zygote.jl`
AutoDiff. This function returns the gradient of the potential energy computed
for given data `x` and latent variable `z`.

# Arguments
- `hvae::HVAE`: An HVAE model that contains a decoder which maps the latent
    variables to the data space.
- `x::AbstractVector{T}`: A vector representing the data.
- `z::AbstractVector{T}`: A vector representing the latent variable.

# Optional Keyword Arguments
- `decoder_loglikelihood::Function=decoder_loglikelihood`: A function
  representing the log-likelihood function used by the decoder. The function
  must take as first input an `AbstractVariationalDecoder` struct, as second
  input a vector `x` representing the data, and as third input a vector `z`
  representing the latent variable. Default is `decoder_loglikelihood`.
- `log_prior::Function=spherical_logprior`: A function representing the
  log-prior distribution used in the autoencoder. The function must take as
  single input a vector `z` representing the latent variable. Default is
  `spherical_logprior`.  

# Returns
- `gradient::AbstractVector{T}`: The computed gradient of the potential energy
  for the given input `x` and latent variable `z`.

# Example
```julia
# Define HVAE
hvae = HVAE(JointLogDecoder(Flux.Chain(Dense(10, 5, relu), Dense(5, 2))))

# Define data x and latent variable z
x = rand(2)
z = rand(2)

# Compute the gradient of the potential energy
gradient = ∇potential_energy(hvae, x, z)
```
"""
function ∇potential_energy(
    hvae::HVAE,
    x::AbstractVector{T},
    z::AbstractVector{T};
    decoder_loglikelihood::Function=decoder_loglikelihood,
    log_prior::Function=spherical_logprior,
) where {T<:Float32}
    # Define potential energy
    function U(z::AbstractVector{T})
        # Compute log-likelihood
        loglikelihood = decoder_loglikelihood(hvae.vae.decoder, x, z)

        # Compute log-prior
        logprior = log_prior(z)

        # Compute potential energy
        return -loglikelihood - logprior
    end # function
    # Define gradient of potential energy function
    return Zygote.gradient(U, z)[1]
end # function

# ==============================================================================
# Hamiltonian Dynamics
# ==============================================================================

@doc raw"""
        leapfrog_step(
                hvae::HVAE,
                x::AbstractVector{T}, 
                z::AbstractVector{T}, 
                ρ::AbstractVector{T}, 
                ϵ::Union{T,AbstractVector{T}},
                ∇U::Function=∇potential_energy,
                ∇U_kwargs::Union{Dict,NamedTuple}=(
                        decoder_loglikelihood=decoder_loglikelihood,
                        log_prior=spherical_logprior,
                )
        ) where {T<:AbstractFloat}

Perform a single leapfrog step in Hamiltonian Monte Carlo (HMC) algorithm. The
leapfrog step is a numerical integration scheme used in HMC to simulate the
dynamics of a physical system (the position `z` and momentum `ρ` variables)
under a potential energy function `U`. The leapfrog step consists of three
parts: a half-step update of the momentum, a full-step update of the position,
and another half-step update of the momentum.

# Arguments
- `hvae::HVAE`: An HVAE model that contains a decoder which maps the latent
  variables to the data space.
- `x::AbstractVector{T}`: The data that defines the potential energy function
  used in the HMC algorithm.
- `z::AbstractVector{T}`: The position variable in the HMC algorithm,
  representing the latent variable.
- `ρ::AbstractVector{T}`: The momentum variable in the HMC algorithm.
- `ϵ::Union{T,AbstractArray{T}}`: The leapfrog step size. This can be a scalar
  used for all dimensions, or an array of the same size as `z` used for each
  dimension.
- `∇U::Function=∇potential_energy`: The gradient of the potential energy
  function used in the HMC algorithm. This function must take three inputs:
  First, an `HVAE` model, then, `x`, the data, and finally, `z`, the latent
  variables.
- `∇U_kwargs::Union{Dict,NamedTuple}`: Additional keyword arguments to be passed
  to the `∇U` function. Default is a NamedTuple with `decoder_loglikelihood` and
  `log_prior` functions.

# Returns
- `NamedTuple`: A named tuple with the updated position variable `z` and the
  updated momentum variable `ρ`.

# Example
```julia
# Define HVAE
hvae = HVAE(JointLogDecoder(Flux.Chain(Dense(10, 5, relu), Dense(5, 2))))

# Define data x, position, momentum, and step size
x = rand(2)
z = rand(2)
ρ = rand(2)
ϵ = 0.01

# Perform a leapfrog step
result = leapfrog_step(hvae, x, z, ρ, ϵ)
```
"""
function leapfrog_step(
    hvae::HVAE,
    x::AbstractVector{T},
    z::AbstractVector{T},
    ρ::AbstractVector{T},
    ϵ::Union{T,<:AbstractArray{T}};
    ∇U::Function=∇potential_energy,
    ∇U_kwargs::Union{Dict,NamedTuple}=(
        decoder_loglikelihood=decoder_loglikelihood,
        log_prior=spherical_logprior,
    )
) where {T<:Float32}
    # Update momentum variable with half-step
    ρ̃ = ρ - (0.5f0 * ϵ) .* ∇U(hvae, x, z; ∇U_kwargs...)

    # Update position variable with full-step
    z̄ = z + ϵ .* ρ̃

    # Update momentum variable with half-step
    ρ̄ = ρ̃ - (0.5f0 * ϵ) .* ∇U(hvae, x, z̄; ∇U_kwargs...)

    return z̄, ρ̄
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
        leapfrog_step(
                hvae::HVAE,
                x::AbstractMatrix{T}, 
                z::AbstractMatrix{T}, 
                ρ::AbstractMatrix{T}, 
                ϵ::Union{T,AbstractVector{T}},
                ∇U::Function=∇potential_energy,
                ∇U_kwargs::Union{Dict,NamedTuple}=(
                        decoder_loglikelihood=decoder_loglikelihood,
                        log_prior=spherical_logprior,
                )
        ) where {T<:AbstractFloat}

Perform a single leapfrog step in Hamiltonian Monte Carlo (HMC) algorithm for
each column of the input matrices. The leapfrog step is a numerical integration
scheme used in HMC to simulate the dynamics of a physical system (the position
`z` and momentum `ρ` variables) under a potential energy function `U`. The
leapfrog step consists of three parts: a half-step update of the momentum, a
full-step update of the position, and another half-step update of the momentum.

# Arguments
- `hvae::HVAE`: An HVAE model that contains a decoder which maps the latent
  variables to the data space.
- `x::AbstractMatrix{T}`: The input data. Each column is treated as a separate
  vector of data.
- `z::AbstractMatrix{T}`: The position variables in the HMC algorithm,
  representing the latent variables. Each column corresponds to the position
  variable for the corresponding column in `x`.
- `ρ::AbstractMatrix{T}`: The momentum variables in the HMC algorithm. Each
  column corresponds to the momentum variable for the corresponding column in
  `x`.
- `ϵ::Union{T,AbstractArray{T}}`: The step size for the HMC algorithm. This can
  be a scalar or an array.
- `∇U::Function=∇potential_energy`: The gradient of the potential energy
  function used in the HMC algorithm. This function must take three inputs:
  First, an `HVAE` model, then, `x`, the data, and finally, `z`, the latent
  variables.
- `∇U_kwargs::Union{Dict,NamedTuple}`: Additional keyword arguments to be passed
  to the `∇U` function. Default is a NamedTuple with `decoder_loglikelihood` and
  `log_prior` functions.

# Returns
- `NamedTuple`: A named tuple with the updated position variable `z` and the
  updated momentum variable `ρ`. Each field is an `AbstractMatrix{T}` where each
  column corresponds to the updated variable for the corresponding column in
  `x`.

# Example
```julia
# Define HVAE
hvae = HVAE(JointLogDecoder(Flux.Chain(Dense(10, 5, relu), Dense(5, 2))))

# Define input data, position, momentum, and step size
x = rand(2, 100)
z = rand(2, 100)
ρ = rand(2, 100)
ϵ = 0.01

# Perform a leapfrog step
result = leapfrog_step(hvae, x, z, ρ, ϵ)
```
"""
function leapfrog_step(
    hvae::HVAE,
    x::AbstractMatrix{T},
    z::AbstractMatrix{T},
    ρ::AbstractMatrix{T},
    ϵ::Union{T,<:AbstractArray{T}};
    ∇U::Function=∇potential_energy,
    ∇U_kwargs::Union{Dict,NamedTuple}=(
        decoder_loglikelihood=decoder_loglikelihood,
        log_prior=spherical_logprior,
    )
) where {T<:Float32}
    # Apply leapfrog_step to each column and collect the results
    results = [
        leapfrog_step(
            hvae, x[:, i], z[:, i], ρ[:, i], ϵ;
            ∇U=∇U, ∇U_kwargs=∇U_kwargs
        )
        for i in axes(z, 2)
    ]

    # Split the results into separate matrices for z̄ and ρ̄
    z̄ = reduce(hcat, [result[1] for result in results])
    ρ̄ = reduce(hcat, [result[2] for result in results])

    return z̄, ρ̄
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

# Example
```julia
# Define the initial inverse temperature, current stage, and total stages
βₒ = 0.5
k = 5
K = 10

# Compute the inverse temperature at stage k
βₖ = quadratic_tempering(βₒ, k, K)
```
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
                hvae::HVAE,
                x::AbstractVecOrMat{T},
                zₒ::AbstractVecOrMat{T},
                K::Int=3,
                ϵ::Union{T,<:AbstractVector{T}}=0.001f0,
                βₒ::T=0.3f0,
                ∇U::Function=∇potential_energy,
                ∇U_kwargs::Union{Dict,NamedTuple}=(
                        decoder_loglikelihood=decoder_loglikelihood,
                        log_prior=spherical_logprior,
                ),
                tempering_schedule::Function=quadratic_tempering,
        ) where {T<:Float32}

Combines the leapfrog and tempering steps into a single function for the
Hamiltonian Variational Autoencoder (HVAE).

# Arguments
- `hvae::HVAE`: The Hamiltonian Variational Autoencoder model.
- `x::AbstractVecOrMat{T}`: The data to be processed. Can be a vector or a
  matrix.
- `zₒ::AbstractVecOrMat{T}`: The initial latent variable. Can be a vector or a
  matrix.  

# Optional Keyword Arguments
- `K::Int`: The number of leapfrog steps to perform in the Hamiltonian Monte
  Carlo (HMC) algorithm. Default is 3.
- `ϵ::Union{T,<:AbstractVector{T}}`: The step size for the leapfrog steps in the
  HMC algorithm. This can be a scalar or a vector. Default is 0.001f0.  
- `βₒ::T`: The initial inverse temperature for the tempering schedule. Default
  is 0.3f0.
- `∇U::Function`: The gradient function of the potential energy. This function
  must takes both `x` and `z` as arguments, but only computes the gradient with
  respect to `z`. Default is `∇potential_energy`.
- `∇U_kwargs::Union{Dict,NamedTuple}`: Additional keyword arguments to be passed
  to the `∇U` function. Default is a NamedTuple with `decoder_loglikelihood` and
  `log_prior`.
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

# Description
The function first samples a random momentum variable `γₒ` from a standard
normal distribution and scales it by the inverse square root of the initial
inverse temperature `βₒ` to obtain the initial momentum variable `ρₒ`. Then, it
performs `K` leapfrog steps, each followed by a tempering step, to generate a
new sample from the latent space.

# Note
Ensure the input data `x` and the initial latent variable `zₒ` match the
expected input dimensionality for the HVAE model. Both `x` and `zₒ` can be
either vectors or matrices.
"""
function leapfrog_tempering_step(
    hvae::HVAE,
    x::AbstractVecOrMat{T},
    zₒ::AbstractVecOrMat{T};
    K::Int=3,
    ϵ::Union{T,<:AbstractVector{T}}=0.001f0,
    βₒ::T=0.3f0,
    ∇U::Function=∇potential_energy,
    ∇U_kwargs::Union{Dict,NamedTuple}=(
        decoder_loglikelihood=decoder_loglikelihood,
        log_prior=spherical_logprior,
    ),
    tempering_schedule::Function=quadratic_tempering,
) where {T<:Float32}
    # Extract latent-space dimensionality
    ldim = size(zₒ, 1)

    # Sample γₒ ~ N(0, I)
    if isa(zₒ, AbstractVector)
        γₒ = Random.rand(Distributions.MvNormal(zeros(T, ldim), ones(T, ldim)))
    else
        γₒ = Random.rand(
            Distributions.MvNormal(zeros(T, ldim), ones(T, ldim)), size(zₒ, 2)
        )
    end # if

    # Define ρₒ = γₒ / √βₒ
    ρₒ = γₒ ./ √(βₒ)

    # Define initial value of z and ρ before loop
    zₖ₋₁ = deepcopy(zₒ)
    ρₖ₋₁ = deepcopy(ρₒ)

    # Loop over K steps
    for k = 1:K
        # 1) Leapfrog step
        zₖ, ρₖ = leapfrog_step(
            hvae, x, zₖ₋₁, ρₖ₋₁, ϵ;
            ∇U=∇U, ∇U_kwargs=∇U_kwargs
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
    end # for

    return (
        z_init=zₒ,
        ρ_init=ρₒ,
        z_final=zₖ₋₁,
        ρ_final=ρₖ₋₁,
    )
end # function

# ==============================================================================
# Forward pass methods for HVAE with Hamiltonian steps
# ==============================================================================

@doc raw"""
    (hvae::HVAE{VAE{JointLogEncoder,D}})(
        x::AbstractVecOrMat{T},
        K::Int,
        ϵ::Union{T,<:AbstractVector{T}},
        βₒ::T;
        ∇U::Function=∇potential_energy,
        ∇U_kwargs::Union{Dict,NamedTuple}=(
            decoder_loglikelihood=decoder_loglikelihood,
            log_prior=spherical_logprior,
        ),
        tempering_schedule::Function=quadratic_tempering,
        latent::Bool=false,
    ) where {D<:AbstractGaussianDecoder,T<:Float32}

Run the Hamiltonian Variational Autoencoder (HVAE) on the given input.

# Arguments
- `x::AbstractVecOrMat{T}`: The input to the HVAE. If it is a vector, it
  represents a single data point. If it is a matrix, each column corresponds to
  a specific data point, and each row corresponds to a dimension of the input
  space.

# Optional Keyword Arguments
- `K::Int=3`: The number of leapfrog steps to perform in the Hamiltonian Monte
  Carlo (HMC) part of the HVAE.
- `ϵ::Union{T,<:AbstractVector{T}}=0.001f0`: The step size for the leapfrog
  steps in the HMC part of the HVAE. If it is a scalar, the same step size is
  used for all dimensions. If it is a vector, each element corresponds to the
  step size for a specific dimension.
- `βₒ::T=0.3f0`: The initial inverse temperature for the tempering schedule.
- `∇U::Function=∇potential_energy`: The function to compute the gradient of the
  potential energy in the HMC part of the HVAE.
- `∇U_kwargs::Union{Dict,NamedTuple}`: The keyword arguments to pass to the `∇U`
  function.
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
function (hvae::HVAE{VAE{JointLogEncoder,D}})(
    x::AbstractVecOrMat{T};
    K::Int=3,
    ϵ::Union{T,<:AbstractVector{T}}=0.001f0,
    βₒ::T=0.3f0,
    ∇U::Function=∇potential_energy,
    ∇U_kwargs::Union{Dict,NamedTuple}=(
        decoder_loglikelihood=decoder_loglikelihood,
        log_prior=spherical_logprior,
    ),
    tempering_schedule::Function=quadratic_tempering,
    latent::Bool=false,
) where {D<:AbstractGaussianDecoder,T<:Float32}
    # Run input through encoder
    encoder_outputs = hvae.vae.encoder(x)

    # Run reparametrize trick to generate latent variable zₒ
    zₒ = reparameterize(hvae.vae.encoder, encoder_outputs, n_samples=1)

    # Run leapfrog and tempering steps
    phase_space = leapfrog_tempering_step(
        hvae, x, zₒ;
        K=K, ϵ=ϵ, βₒ=βₒ, ∇U=∇U, ∇U_kwargs=∇U_kwargs,
        tempering_schedule=tempering_schedule
    )

    # Run final zₖ through decoder
    decoder_outputs = hvae.vae.decoder(phase_space.z_final)

    # Check if latent variables should be returned
    if latent
        return (
            encoder=encoder_outputs,
            decoder=decoder_outputs,
            phase_space=phase_space,
        )
    else
        return decoder_outputs
    end # if
end # function

# ==============================================================================
# Hamiltonian ELBO
# ==============================================================================

@doc raw"""
        log_p̄(
                decoder::SimpleDecoder,
                x::AbstractVector{T},
                hvae_outputs::NamedTuple,
        ) where {T<:Float32}

This is an internal function used in `hamiltonian_elbo` to compute the numerator
of the unbiased estimator of the marginal likelihood. The function computes the
sum of the log likelihood of the data given the latent variables, the log prior
of the latent variables, and the log prior of the momentum variables.

        log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)

# Arguments
- `decoder::SimpleDecoder`: The decoder of the HVAE. This argument is only used
    to determine which method to call and is not used in the computation itself.
- `x::AbstractVector{T}`: The input data, where `T` is a subtype of
    `AbstractFloat`.
- `hvae_outputs::NamedTuple`: The outputs of the HVAE, including the final
  latent variables `zₖ` and the final momentum variables `ρₖ`.

# Returns
- `log_p̄::T`: The first term of the log of the unbiased estimator of the
    marginal likelihood.

# Note
This is an internal function and should not be called directly. It is used as
part of the `hamiltonian_elbo` function.
"""
function log_p̄(
    decoder::SimpleDecoder,
    x::AbstractVector{T},
    hvae_outputs::NamedTuple,
) where {T<:Float32}
    # Unpack necessary variables
    µ = hvae_outputs.decoder.µ
    σ = 1.0f0
    zₖ = hvae_outputs.phase_space.z_final
    ρₖ = hvae_outputs.phase_space.ρ_final

    # log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)   

    # Compute log p(x | zₖ)
    log_p_x_given_zₖ = -0.5f0 * sum(abs2, (x - μ) / σ) -
                       0.5f0 * length(x) * (2.0f0 * log(σ) + log(2.0f0π))

    # Compute log p(zₖ)
    log_p_zₖ = spherical_logprior(zₖ)

    # Compute log p(ρₖ)
    log_p_ρₖ = spherical_logprior(ρₖ)

    return log_p_x_given_zₖ + log_p_zₖ + log_p_ρₖ
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    log_p̄(
        decoder::AbstractGaussianLogDecoder,
        x::AbstractVector{T},
        hvae_outputs::NamedTuple,
    ) where {T<:Float32}

This is an internal function used in `hamiltonian_elbo` to compute the numerator
of the unbiased estimator of the marginal likelihood. The function computes the
sum of the log likelihood of the data given the latent variables, the log prior
of the latent variables, and the log prior of the momentum variables.

        log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)

# Arguments
- `decoder::AbstractGaussianLogDecoder`: The decoder of the HVAE. This argument
  is only used to determine which method to call and is not used in the
  computation itself.
- `x::AbstractVector{T}`: The input data, where `T` is a subtype of
  `AbstractFloat`.
- `hvae_outputs::NamedTuple`: The outputs of the HVAE, including the final
  latent variables `zₖ` and the final momentum variables `ρₖ`.

# Returns
- `log_p̄::T`: The first term of the log of the unbiased estimator of the
  marginal likelihood.

# Note
This is an internal function and should not be called directly. It is used as
part of the `hamiltonian_elbo` function.
"""
function log_p̄(
    decoder::AbstractGaussianLogDecoder,
    x::AbstractVector{T},
    hvae_outputs::NamedTuple,
) where {T<:Float32}
    # Unpack necessary variables
    µ = hvae_outputs.decoder.µ
    logσ = hvae_outputs.decoder.logσ
    σ = exp.(logσ)
    zₖ = hvae_outputs.phase_space.z_final
    ρₖ = hvae_outputs.phase_space.ρ_final

    # log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)   

    # Compute log p(x | zₖ)
    log_p_x_given_zₖ = -0.5f0 * sum(abs2, (x - μ) ./ σ) -
                       sum(logσ) -
                       0.5f0 * length(x) * log(2.0f0π)

    # Compute log p(zₖ)
    log_p_zₖ = spherical_logprior(zₖ)

    # Compute log p(ρₖ)
    log_p_ρₖ = spherical_logprior(ρₖ)

    return log_p_x_given_zₖ + log_p_zₖ + log_p_ρₖ
end # function

# ------------------------------------------------------------------------------

@doc raw"""
        log_p̄(
                decoder::AbstractGaussianLinearDecoder,
                x::AbstractVector{T},
                hvae_outputs::NamedTuple,
        ) where {T<:Float32}

This is an internal function used in `hamiltonian_elbo` to compute the numerator
of the unbiased estimator of the marginal likelihood. The function computes the
sum of the log likelihood of the data given the latent variables, the log prior
of the latent variables, and the log prior of the momentum variables.

        log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)

# Arguments
- `decoder::AbstractGaussianLinearDecoder`: The decoder of the HVAE. This
  argument is only used to determine which method to call and is not used in the
  computation itself.
- `x::AbstractVector{T}`: The input data, where `T` is a subtype of
  `AbstractFloat`.
- `hvae_outputs::NamedTuple`: The outputs of the HVAE, including the final
  latent variables `zₖ` and the final momentum variables `ρₖ`.

# Returns
- `log_p̄::T`: The first term of the log of the unbiased estimator of the
    marginal likelihood.

# Note
This is an internal function and should not be called directly. It is used as
part of the `hamiltonian_elbo` function.
"""
function log_p̄(
    decoder::AbstractGaussianLinearDecoder,
    x::AbstractVector{T},
    hvae_outputs::NamedTuple,
) where {T<:Float32}
    # Unpack necessary variables
    µ = hvae_outputs.decoder.µ
    σ = hvae_outputs.decoder.σ
    zₖ = hvae_outputs.phase_space.z_final
    ρₖ = hvae_outputs.phase_space.ρ_final

    # log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)   

    # Compute log p(x | zₖ)
    log_p_x_given_zₖ = -0.5f0 * sum(abs2, (x - μ) ./ σ) -
                       sum(log, σ) -
                       0.5f0 * length(x) * log(2.0f0π)

    # Compute log p(zₖ)
    log_p_zₖ = spherical_logprior(zₖ)

    # Compute log p(ρₖ)
    log_p_ρₖ = spherical_logprior(ρₖ)

    return log_p_x_given_zₖ + log_p_zₖ + log_p_ρₖ
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    log_q̄(
        encoder::AbstractGaussianLogEncoder,
        hvae_outputs::NamedTuple,
        βₒ::T,
    ) where {T<:Float32}

This is an internal function used in `hamiltonian_elbo` to compute the second
term of the unbiased estimator of the marginal likelihood. The function computes
the sum of the log posterior of the initial latent variables and the log prior
of the initial momentum variables, minus a term that depends on the
dimensionality of the latent space and the initial temperature.

        log q̄ = log q(zₒ) + log p(ρₒ) - d/2 log(βₒ)

# Arguments
- `encoder::AbstractGaussianLogEncoder`: The encoder of the HVAE. This argument
    is only used to determine which method to call and is not used in the
    computation itself.
- `hvae_outputs::NamedTuple`: The outputs of the HVAE, including the initial
  latent variables `zₒ` and the initial momentum variables `ρₒ`.
- `βₒ::T`: The initial temperature, where `T` is a subtype of `AbstractFloat`.

# Returns
- `log_q̄::T`: The second term of the log of the unbiased estimator of the
    marginal likelihood.

# Note
This is an internal function and should not be called directly. It is used as
part of the `hamiltonian_elbo` function.
"""
function log_q̄(
    encoder::AbstractGaussianLogEncoder,
    hvae_outputs::NamedTuple,
    βₒ::T
) where {T<:Float32}
    # Unpack necessary variables
    µ = hvae_outputs.encoder.µ
    logσ = hvae_outputs.encoder.logσ
    zₒ = hvae_outputs.phase_space.z_init
    ρₒ = hvae_outputs.phase_space.ρ_init

    # log q̄ = log q(zₒ) + log p(ρₒ) - d/2 log(βₒ)

    # Compute log q(zₒ)
    log_q = -0.5f0 * sum(abs2, (zₒ - μ) ./ exp.(logσ)) -
            sum(logσ) - 0.5f0 * length(zₒ) * log(2.0f0π)

    # Compute log p(ρₒ)
    log_ρ = spherical_logprior(ρₒ, βₒ^-1)

    return log_q + log_ρ - 0.5f0 * length(zₒ) * log(βₒ)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    hamiltonian_elbo(
        hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractGaussianDecoder}},
        x::AbstractVector{T};
        K::Int=3,
        ϵ::Union{T,<:AbstractVector{T}}=0.001f0,
        βₒ::T=0.3f0,
        U::Function=potential_energy,
        ∇U::Function=∇potential_energy,
        U_kwargs::Union{Dict,NamedTuple}=(
                decoder_loglikelihood=decoder_loglikelihood,
                log_prior=spherical_logprior,
        ),
        tempering_schedule::Function=quadratic_tempering,
        return_outputs::Bool=false,
    ) where {T<:Float32}

Compute the Hamiltonian Monte Carlo (HMC) estimate of the evidence lower bound
(ELBO) for a Hamiltonian Variational Autoencoder (HVAE).

This function takes as input an HVAE and a vector of input data `x`. It performs
`K` HMC steps with a leapfrog integrator and a tempering schedule to estimate
the ELBO. The ELBO is computed as the difference between the log evidence
estimate `log p̄` and the log variational estimate `log q̄`.

# Arguments
- `hvae::HVAE`: The HVAE used to encode the input data and decode the latent
  space.
- `x::AbstractVector{T}`: The input data, where `T` is a subtype of
  `AbstractFloat`.

# Optional Keyword Arguments
- `∇U::Function`: The gradient function of the potential energy. This function
  must take both `x` and `z` as arguments, but only computes the gradient with
  respect to `z`.
- `∇U_kwargs::Union{Dict,NamedTuple}`: Additional keyword arguments to be passed
  to the `∇U` function. Defaults to a NamedTuple with `:decoder_loglikelihood`
  set to `decoder_loglikelihood` and `:log_prior` set to `spherical_logprior`.
- `K::Int`: The number of HMC steps (default is 3).
- `ϵ::Union{T,<:AbstractVector{T}}`: The step size for the leapfrog integrator
  (default is 0.001).
- `βₒ::T`: The initial inverse temperature (default is 0.3).
- `tempering_schedule::Function`: The tempering schedule function used in the
  HMC (default is `quadratic_tempering`).
- `return_outputs::Bool`: Whether to return the outputs of the HVAE. Defaults to
  `false`. NOTE: This is necessary to avoid computing the forward pass twice
  when computing the loss function with regularization.

# Returns
- `elbo::T`: The HMC estimate of the ELBO. If `return_outputs` is `true`, also
  returns a NamedTuple containing the outputs of the HVAE.

# Example
```julia
# Define a VAE
vae = VAE(
    JointLogEncoder(
        Flux.Chain(Flux.Dense(784, 400, relu)), 
        Flux.Dense(400, 20), 
        Flux.Dense(400, 20)
    ),
    AbstractGaussianDecoder()
)

# Define an HVAE
hvae = HVAE(vae)

# Define input data
x = rand(Float32, 784)

# Compute the Hamiltonian ELBO
elbo = hamiltonian_elbo(hvae, x, K=3, ϵ=0.001f0, βₒ=0.3f0)
```
"""
function hamiltonian_elbo(
    hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractGaussianDecoder}},
    x::AbstractVector{T};
    K::Int=3,
    ϵ::Union{T,<:AbstractVector{T}}=0.001f0,
    βₒ::T=0.3f0,
    ∇U::Function=∇potential_energy,
    ∇U_kwargs::Union{Dict,NamedTuple}=(
        decoder_loglikelihood=decoder_loglikelihood,
        log_prior=spherical_logprior,
    ),
    tempering_schedule::Function=quadratic_tempering,
    return_outputs::Bool=false,
) where {T<:Float32}
    # Forward Pass (run input through reconstruct function)
    hvae_outputs = hvae(
        x;
        K=K, ϵ=ϵ, βₒ=βₒ,
        ∇U=∇U, ∇U_kwargs=∇U_kwargs,
        tempering_schedule=tempering_schedule,
        latent=true
    )

    # Compute log evidence estimate log π̂(x) = log p̄ - log q̄

    # log p̄ = log p(x, zₖ) + log p(ρₖ)
    # log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)
    log_p = log_p̄(hvae.vae.decoder, x, hvae_outputs)

    # log q̄ = log q(zₒ) + log p(ρₒ) - d/2 log(βₒ)
    log_q = log_q̄(hvae.vae.encoder, hvae_outputs, βₒ)

    # Check if HVAE outputs should be returned
    if return_outputs
        return log_p - log_q, hvae_outputs
    else
        return log_p - log_q
    end # if
end # function

# ------------------------------------------------------------------------------ 

# @doc raw"""
#         hamiltonian_elbo(
#                 hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractVariationalDecoder}},
#                 x::AbstractMatrix{T};
#                 K::Int=3,
#                 ϵ::Union{T,<:AbstractVector{T}}=0.001f0,
#                 βₒ::T=0.3f0,
#                 U::Function=potential_energy,
#                 ∇U::Function=∇potential_energy,
#                 U_kwargs::Union{Dict,NamedTuple}=Dict(
#                         :decoder_dist => MvDiagGaussianDecoder,
#                         :prior => SphericalPrior,
#                 ),
#                 tempering_schedule::Function=quadratic_tempering,
#                 return_outputs::Bool=false,
#         ) where {T<:Float32}

# Compute the Hamiltonian Monte Carlo (HMC) estimate of the evidence lower bound
# (ELBO) for a Hamiltonian Variational Autoencoder (HVAE).

# This function takes as input an HVAE and a matrix of input data `x`. It performs
# `K` HMC steps with a leapfrog integrator and a tempering schedule to estimate
# the ELBO. The ELBO is computed as the difference between the log evidence
# estimate `log p̄` and the log variational estimate `log q̄`.

# # Arguments
# - `hvae::HVAE`: The HVAE used to encode the input data and decode the latent
#     space.
# - `x::AbstractMatrix{T}`: The input data, where `T` is a subtype of
#     `AbstractFloat`. Each column represents a single data point.

# ## Optional Keyword Arguments
# # Optional Keyword Arguments
# - `U::Function`: The potential energy. This function must take both `x` and `z`
#   as arguments.
# - `∇U::Function`: The gradient function of the potential energy. This function
#   must take both `x` and `z` as arguments, but only computes the gradient with
#   respect to `z`.
# - `U_kwargs::Union{Dict,NamedTuple}`: Additional keyword arguments to be passed
#   to the `U` and `∇U` function. Defaults to a NamedTuple with
#   `:decoder_loglikelihood` set to `decoder_loglikelihood` and `:log_prior` set
#   to `spherical_logprior`.
# - `K::Int`: The number of HMC steps (default is 3).
# - `ϵ::Union{T,<:AbstractVector{T}}`: The step size for the leapfrog integrator
#   (default is 0.001).
# - `βₒ::T`: The initial inverse temperature (default is 0.3).
# - `tempering_schedule::Function`: The tempering schedule function used in the
#   HMC (default is `quadratic_tempering`).
# - `return_outputs::Bool`: Whether to return the outputs of the HVAE. Defaults to
#   `false`. NOTE: This is necessary to avoid computing the forward pass twice
#   when computing the loss function with regularization.

# # Returns
# - `elbo::T`: The HMC estimate of the ELBO. If `return_outputs` is `true`, also
#     returns the outputs of the HVAE.

# # Example
# ```julia
# # Define a VAE
# vae = VAE(
#         JointLogEncoder(
#                 Flux.Chain(Flux.Dense(784, 400, relu)), 
#                 Flux.Dense(400, 20), 
#                 Flux.Dense(400, 20)
#         ),
#         AbstractVariationalDecoder()
# )

# # Define an HVAE
# hvae = HVAE(vae)

# # Define input data
# x = rand(Float32, 784, 100)  # 100 data points

# # Compute the Hamiltonian ELBO
# elbo = hamiltonian_elbo(hvae, x, K=3, ϵ=0.001f0, βₒ=0.3f0)
# ```
# """
# function hamiltonian_elbo(
#     hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractVariationalDecoder}},
#     x::AbstractMatrix{T};
#     K::Int=3,
#     ϵ::Union{T,<:AbstractVector{T}}=0.001f0,
#     βₒ::T=0.3f0,
#     U::Function=potential_energy,
#     ∇U::Function=∇potential_energy,
#     U_kwargs::Union{Dict,NamedTuple}=(
#         decoder_loglikelihood=decoder_loglikelihood,
#         log_prior=spherical_logprior,
#     ),
#     tempering_schedule::Function=quadratic_tempering,
#     return_outputs::Bool=false,
# ) where {T<:Float32}
#     # Forward Pass (run input through reconstruct function)
#     hvae_outputs = hvae(
#         x;
#         K=K, ϵ=ϵ, βₒ=βₒ,
#         ∇U=∇U, ∇U_kwargs=U_kwargs,
#         tempering_schedule=tempering_schedule,
#         latent=true
#     )

#     # Unpack position and momentum variables
#     zₒ = hvae_outputs.phase_space.z_init
#     zₖ = hvae_outputs.phase_space.z_final
#     ρₒ = hvae_outputs.phase_space.ρ_init
#     ρₖ = hvae_outputs.phase_space.ρ_final

#     # Initialize value to save ELBO
#     elbo = zero(Float32)

#     # Compute log evidence estimate log π̂(x) = log p̄ - log q̄

#     # Loop through each column of input data
#     for i in axes(x, 2)
#         # log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)
#         # log p̄ = - U(x, zₖ) + log p(ρₖ)
#         log_p̄ = -U(hvae, x[:, i], zₖ[:, i]; U_kwargs...) +
#                  spherical_logprior(ρₖ[:, i])

#         # log q̄ = log q(zₒ) + log p(ρₒ)

#         # Extract mean and log standard deviation from the encoder outputs
#         μ = hvae_outputs.encoder.µ
#         logσ = hvae_outputs.encoder.logσ

#         # Compute standard deviation
#         σ = exp.(logσ)

#         log_q̄ = -0.5f0 * sum(abs2, (zₒ[:, 1] - μ[:, 1]) ./ σ[:, 1]) -
#                  sum(logσ[:, 1]) - 0.5f0 * size(zₒ, 1) * log(2.0f0π) +
#                  spherical_logprior(ρₒ[:, 1], βₒ^-1) -
#                  0.5f0 * size(zₒ, 1) * log(βₒ)

#         # Update ELBO
#         elbo += log_p̄ - log_q̄
#     end # for

#     if return_outputs
#         return elbo / size(x, 2), hvae_outputs
#     else
#         # Return ELBO normalized by number of samples
#         return elbo / size(x, 2)
#     end # if
# end # function

# ==============================================================================
# Define HVAE loss function
# ==============================================================================

"""
        loss(
                hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractVariationalDecoder}},
                x::AbstractVecOrMat{Float32};
                energy_functions::NamedTuple=NamedTuple(),
                K::Int=3,
                ϵ::Union{Float32,<:AbstractVector{Float32}}=0.001f0,
                βₒ::Float32=0.3f0,
                tempering_schedule::Function=quadratic_tempering,
                prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
                reg_function::Union{Function,Nothing}=nothing,
                reg_kwargs::Union{NamedTuple,Dict}=Dict(),
                reg_strength::Float32=1.0f0
        )

Compute the loss for a Hamiltonian Variational Autoencoder (HVAE).

# Arguments
- `hvae::HVAE`: The HVAE used to encode the input data and decode the latent
    space.
- `x::AbstractVecOrMat{Float32}`: The input data. If vector, the function
    assumes a single data point. If matrix, the function assumes a batch of data
    points.

## Optional Keyword Arguments
# Optional Keyword Arguments
- `U::Function`: The potential energy. This function must take both `x` and `z`
  as arguments.
- `∇U::Function`: The gradient function of the potential energy. This function
  must take both `x` and `z` as arguments, but only computes the gradient with
  respect to `z`.
- `U_kwargs::Union{Dict,NamedTuple}`: Additional keyword arguments to be passed
  to the `U` and `∇U` function. Defaults to a NamedTuple with
  `:decoder_loglikelihood` set to `decoder_loglikelihood` and `:log_prior` set
  to `spherical_logprior`.
- `K::Int`: The number of HMC steps (default is 3).
- `ϵ::Union{T,<:AbstractVector{T}}`: The step size for the leapfrog integrator
  (default is 0.001).
- `βₒ::T`: The initial inverse temperature (default is 0.3).
- `tempering_schedule::Function`: The tempering schedule function used in the
  HMC (default is `quadratic_tempering`).
- `reg_function::Union{Function, Nothing}=nothing`: A function that computes the
  regularization term based on the VAE outputs. Should return a Float32. This
  function must take as input the VAE outputs and the keyword arguments provided
  in `reg_kwargs`.
- `reg_kwargs::Union{NamedTuple,Dict}=Dict()`: Keyword arguments to pass to the
  regularization function.
- `reg_strength::Float32=1.0f0`: The strength of the regularization term.

# Returns
- The computed loss.
"""
function loss(
    hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractVariationalDecoder}},
    x::AbstractVecOrMat{Float32};
    K::Int=3,
    ϵ::Union{Float32,<:AbstractVector{Float32}}=0.001f0,
    βₒ::Float32=0.3f0,
    U::Function=potential_energy,
    ∇U::Function=∇potential_energy,
    U_kwargs::Union{Dict,NamedTuple}=(
        decoder_loglikelihood=decoder_loglikelihood,
        log_prior=spherical_logprior,
    ),
    tempering_schedule::Function=quadratic_tempering,
    reg_function::Union{Function,Nothing}=nothing,
    reg_kwargs::Union{NamedTuple,Dict}=Dict(),
    reg_strength::Float32=1.0f0
)
    # Check if there is regularization
    if reg_function !== nothing
        # Compute ELBO and regularization term
        elbo, hvae_outputs = hamiltonian_elbo(
            hvae, x;
            K=K, ϵ=ϵ, βₒ=βₒ,
            U=U, ∇U=∇U, U_kwargs=U_kwargs,
            tempering_schedule=tempering_schedule,
            return_outputs=true
        )

        # Compute regularization
        reg_term = reg_function(hvae_outputs; reg_kwargs...)

        return -elbo + reg_strength * reg_term
    else
        # Compute ELBO
        return -hamiltonian_elbo(
            hvae, x;
            K=K, ϵ=ϵ, βₒ=βₒ,
            U=U, ∇U=∇U, U_kwargs=U_kwargs,
            tempering_schedule=tempering_schedule
        )
    end # if
end # function

# ==============================================================================
# HVAE training
# ==============================================================================

@doc raw"""
        `train!(hvae, x, opt; loss_function, loss_kwargs)`

Customized training function to update parameters of a Hamiltonian Variational
Autoencoder given a specified loss function.

# Arguments
- `hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractVariationalDecoder}}`: A struct
  containing the elements of a Hamiltonian Variational Autoencoder.
- `x::AbstractVecOrMat{Float32}`: Data on which to evaluate the loss function.
  Columns represent individual samples.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments
- `loss_function::Function=loss`: The loss function used for training. It should
  accept the HVAE model, data `x`, and keyword arguments in that order.
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Arguments for the loss
  function. These might include parameters like `K`, `ϵ`, `βₒ`,
  `potential_energy`, `potential_energy_kwargs`, `tempering_schedule`, `prior`,
  `reg_function`, `reg_kwargs`, `reg_strength`, depending on the specific loss
  function in use.

# Description
Trains the HVAE by:
1. Computing the gradient of the loss w.r.t the HVAE parameters.
2. Updating the HVAE parameters using the optimizer.

# Examples
```julia
opt = Flux.setup(Optax.adam(1e-3), hvae)
for x in dataloader
        train!(hvae, x, opt; loss_fn, loss_kwargs=Dict(:K => 3, :ϵ => 0.001f0, :βₒ => 0.3f0))
end
```
"""
function train!(
    hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractVariationalDecoder}},
    x::AbstractVecOrMat{Float32},
    opt::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Dict=Dict()
)
    # Compute VAE gradient
    ∇loss_ = Flux.gradient(hvae) do hvae_model
        loss_function(hvae_model, x; loss_kwargs...)
    end # do block
    # Update parameters
    Flux.Optimisers.update!(opt, hvae, ∇loss_[1])
end # function

## =============================================================================
# Flux.gradient(hvae) do hvae_model
#     HVAEs.potential_energy(hvae, x_vec, z_vec)
# end


## =============================================================================
# function ∇∇potential_energy(
#     hvae::HVAE,
#     x::AbstractVector{T},
#     z::AbstractVector{T};
#     decoder_dist::Function=MvDiagGaussianDecoder,
#     prior::Function=SphericalPrior,
# ) where {T<:AbstractFloat}
#     # Function to compute the first gradient
#     function first_grad_function(hvae)
#         # Define potential energy using the modified decoder
#         function U(z::AbstractVector{T})
#             loglikelihood = Distributions.logpdf(
#                 decoder_dist(hvae.vae.decoder, z), x
#             )
#             log_prior = Distributions.logpdf(prior(hvae.vae.encoder), z)
#             return -loglikelihood - log_prior
#         end

#         # Compute and return the first gradient
#         Zygote.gradient(U, z)[1]
#     end

#     # Extract parameters of the decoder
#     # decoder_params = ... # Get the parameters of the decoder from `hvae`

#     # Compute the gradient of the first gradient function with respect to the decoder parameters
#     second_grad = Zygote.gradient(first_grad_function, hvae)[1]

#     return second_grad
# end