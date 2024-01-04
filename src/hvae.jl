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

using ..AutoEncode: AbstractVariationalAutoEncoder, AbstractVariationalEncoder,
    AbstractVariationalDecoder, JointLogEncoder, SimpleDecoder, JointLogDecoder,
    SplitLogDecoder, JointDecoder, SplitDecoder, VAE


## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Caterini, A. L., Doucet, A. & Sejdinovic, D. Hamiltonian Variational
# Auto-Encoder. 11 (2018).
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ==============================================================================
# Hamiltonian Dynamics
# ==============================================================================
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ==============================================================================
# Functions to construct decoder distribution
# ==============================================================================

"""
        MvDiagGaussianDecoder(decoder, z; σ=1.0f0)

Constructs a multivariate diagonal Gaussian distribution for decoding.

# Arguments
- `decoder::SimpleDecoder`: A `SimpleDecoder` struct representing the decoder
  model.
- `z::AbstractVector{T}`: An abstract vector of type `AbstractVector{T}`
  representing the latent variable, where `T` is a subtype of `AbstractFloat`.

# Optional Keyword Arguments
- `σ::T`: A float of type `T` representing the standard deviation of the
  Gaussian distribution. Default is 1.0f0.

# Returns
- `decoder_dist`: A `Distributions.MvNormal{T}` object representing the
  multivariate diagonal Gaussian distribution, where `T` is the type of elements
  in `z`.

# Example
```julia
# Define a decoder
decoder = SimpleDecoder(Flux.Chain(Dense(10, 5, relu), Dense(5, 2)))

# Define a latent variable
z = rand(Float32, 2)

# Construct the decoder distribution
decoder_dist = MvDiagGaussianDecoder(decoder, z, σ=1.0f0)
```
"""
function MvDiagGaussianDecoder(
    decoder::SimpleDecoder,
    z::AbstractVector{T};
    σ::T=1.0f0,
) where {T<:AbstractFloat}
    μ = decoder(z)
    decoder_dist = Distributions.MvNormal(μ, σ)
    return decoder_dist
end # function

"""
        MvDiagGaussianDecoder(decoder, z)

Constructs a multivariate diagonal Gaussian distribution for decoding.

# Arguments
- `decoder::JointLogDecoder`: A `JointLogDecoder` struct representing the
    decoder model.
- `z::AbstractVector{T}`: A vector representing the latent variable, where `T`
  is a subtype of `AbstractFloat`.

# Returns
- `decoder_dist`: A `Distributions.MvNormal{T}` object representing the
  multivariate diagonal Gaussian distribution, where `T` is the type of elements
  in `z`.

# Example
```julia
# Define a decoder
decoder = JointLogDecoder(Flux.Chain(Dense(10, 5, relu), Dense(5, 2)), Flux.Dense(2, 2), Flux.Dense(2, 2))

# Define a latent variable
z = rand(Float32, 2)

# Construct the decoder distribution
decoder_dist = MvDiagGaussianDecoder(decoder, z)
```
"""
function MvDiagGaussianDecoder(
    decoder::JointLogDecoder,
    z::AbstractVector{T}
) where {T<:AbstractFloat}
    # Compute intermediate representation
    intermediate = decoder.decoder(z)

    # Compute mean
    μ = decoder.µ(intermediate)

    # Compute log standard deviation
    logσ = decoder.logσ(intermediate)

    # Compute standard deviation
    σ = exp.(logσ)

    # Construct multivariate diagonal Gaussian distribution
    decoder_dist = Distributions.MvNormal(μ, σ)

    # Return decoder distribution
    return decoder_dist
end # function

"""
        MvDiagGaussianDecoder(decoder, z)

Constructs a multivariate diagonal Gaussian distribution for decoding.

# Arguments
- `decoder::JointDecoder`: A `JointDecoder` struct representing the decoder
  model.
- `z::AbstractVector{T}`: A vector representing the latent variable, where `T`
  is a subtype of `AbstractFloat`.

# Returns
- `decoder_dist`: A `Distributions.MvNormal{T}` object representing the
  multivariate diagonal Gaussian distribution, where `T` is the type of elements
  in `z`.

# Example
```julia
# Define a decoder
decoder = JointDecoder(Flux.Chain(Dense(10, 5, relu), Dense(5, 2)), Flux.Dense(2, 2), Flux.Dense(2, 2))

# Define a latent variable
z = rand(Float32, 2)

# Construct the decoder distribution
decoder_dist = MvDiagGaussianDecoder(decoder, z)
```
"""
function MvDiagGaussianDecoder(
    decoder::JointDecoder,
    z::AbstractVector{T}
) where {T<:AbstractFloat}
    # Compute intermediate representation
    intermediate = decoder.decoder(z)

    # Compute mean
    μ = decoder.µ(intermediate)

    # Compute standard deviation
    σ = decoder.σ(intermediate)

    # Construct multivariate diagonal Gaussian distribution
    decoder_dist = Distributions.MvNormal(μ, σ)

    # Return decoder distribution
    return decoder_dist
end # function

# ==============================================================================
# Function to construct prior distribution
# ==============================================================================

"""
        SphericalPrior(z::AbstractVector{T}, σ::T=1.0f0) where {T<:AbstractFloat}

Generates a prior distribution as a spherical Gaussian. A spherical Gaussian is
a multivariate Gaussian distribution with a diagonal covariance matrix where all
the diagonal elements (variances) are the same. This results in a distribution
that is symmetric (or "spherical") in all dimensions.

# Arguments
- `z::AbstractVector{T}`: A vector representing the latent variable, where `T`
  is a subtype of `AbstractFloat`.
- `σ::T=1.0f0`: The standard deviation of the spherical Gaussian distribution.
  Defaults to 1.0.

# Returns
- `prior`: A `Distributions.MvNormal{T}` object representing the multivariate
  spherical Gaussian distribution, where `T` is the type of elements in `z`.

# Example
```julia
# Define a latent variable
z = rand(Float32, 2)
# Define the standard deviation
σ = 0.5f0
# Generate the spherical Gaussian prior distribution
prior = SphericalPrior(z, σ)
```
"""
function SphericalPrior(
    z::AbstractVector{T}, σ::T=1.0f0
) where {T<:AbstractFloat}
    # Generate prior distribution as spherical Gaussian
    prior = Distributions.MvNormal(zeros(T, length(z)), σ .* ones(T, length(z)))
    # Return prior
    return prior
end # function

# ==============================================================================
# Function to compute potential energy
# ==============================================================================

"""
        potential_energy(decoder::JointDecoder;
                         decoder_dist::Function = MvDiagGaussianDecoder,
                         prior::Function = SphericalPrior)

Compute the potential energy of a variational autoencoder (VAE). In the context
of Hamiltonian Monte Carlo (HMC), the potential energy is defined as the
negative log-posterior. This function returns a function `U` that computes the
potential energy for given data `x` and latent variable `z`. It does this by
computing the log-likelihood of `x` under the distribution defined by
`decoder_dist(decoder, z)`, and the log-prior of `z` under the `prior`
distribution. The potential energy is then computed as the negative sum of the
log-likelihood and the log-prior.

# Arguments
- `decoder::JointDecoder`: A `JointDecoder` struct representing the decoder
  model used in the autoencoder.

# Optional Keyword Arguments
- `decoder_dist::Function=MvDiagGaussianDecoder`: A function representing the
  distribution function used by the decoder. Default is `MvDiagGaussianDecoder`.
- `prior::Function=SphericalPrior`: A function representing the prior
  distribution used in the autoencoder. Default is `SphericalPrior`.

# Returns
- `U::Function`: A function that computes the potential energy given an input
  `x` and latent variable `z`.

# Example
```julia
# Define a decoder
decoder = JointDecoder(
    Flux.Chain(Dense(10, 5, relu), Dense(5, 2)), 
    Flux.Dense(2, 2), Flux.Dense(2, 2)
)

# Compute the potential energy
U = potential_energy(decoder, MvDiagGaussianDecoder, SphericalPrior)

# Use the function U to compute the potential energy for a given x and z
x = rand(2)
z = rand(2)
energy = U(x, z)
```
"""
function potential_energy(
    decoder::AbstractVariationalDecoder;
    decoder_dist::Function=MvDiagGaussianDecoder,
    prior::Function=SphericalPrior,
)
    # Define function to compute potential energy
    function U(
        x::AbstractVector{T}, z::AbstractVector{T}
    ) where {T<:AbstractFloat}
        # Compute log-likelihood
        log_likelihood = Distributions.logpdf(decoder_dist(decoder, z), x)

        # Compute log-prior
        log_prior = Distributions.logpdf(prior(z), z)

        # Compute potential energy
        energy = -log_likelihood - log_prior

        return energy
    end # function

    return U
end # function

"""
    leapfrog_step(decoder, x, z, ρ, ϵ, potential_energy, potential_energy_kwargs)

Perform a single leapfrog step in Hamiltonian Monte Carlo (HMC) algorithm. The
leapfrog step is a numerical integration scheme used in HMC to simulate the
dynamics of a physical system (the position `z` and momentum `ρ` variables)
under a potential energy function `U`. The leapfrog step consists of three
parts: a half-step update of the momentum, a full-step update of the position,
and another half-step update of the momentum.

# Arguments
- `decoder::AbstractVariationalDecoder`: The decoder model used in the
  autoencoder.
- `x::AbstractVector{T}`: The input data.
- `z::AbstractVector{T}`: The position variable in the HMC algorithm,
  representing the latent variable.
- `ρ::AbstractVector{T}`: The momentum variable in the HMC algorithm.
- `ϵ::AbstractVector{T}`: The step size for each dimension in the HMC algorithm.

## Optional Keyword Arguments
- `potential_energy::Function=potential_energy`: The potential energy function
  used in the HMC algorithm.
- `potential_energy_kwargs::Dict=Dict()`: The keyword arguments for the
  potential energy function. See `potential_energy` for details.

# Returns
- `z̄::AbstractVector{T}`: The updated position variable.
- `ρ̄::AbstractVector{T}`: The updated momentum variable.

# Example
```julia
# Define a decoder
decoder = JointDecoder(Flux.Chain(Dense(10, 5, relu), Dense(5, 2)), Flux.Dense(2, 2), Flux.Dense(2, 2))

# Define input data, position, momentum, and step size
x = rand(2)
z = rand(2)
ρ = rand(2)
ϵ = rand(2)

# Perform a leapfrog step
z̄, ρ̄ = leapfrog_step(decoder, x, z, ρ, ϵ, potential_energy, Dict())
```
"""
function leapfrog_step(
    decoder::AbstractVariationalDecoder,
    x::AbstractVector{T},
    z::AbstractVector{T},
    ρ::AbstractVector{T},
    ϵ::AbstractVector{T};
    potential_energy::Function=potential_energy,
    potential_energy_kwargs::Dict=Dict(),
) where {T<:AbstractFloat}
    # Build potential energy function
    U = potential_energy(decoder; potential_energy_kwargs...)

    # Define gradient of potential energy function
    ∇U(Ƶ::AbstractVector{T}) = first(Zygote.gradient(Ƶ -> U(x, Ƶ), Ƶ))

    # Update momentum variable with half-step
    ρ̃ = ρ - ϵ .* ∇U(z) / 2

    # Update position variable with full-step
    z̄ = z + ϵ .* ρ̃

    # Update momentum variable with half-step
    ρ̄ = ρ̃ - ϵ .* ∇U(z̄) / 2

    return z̄, ρ̄
end # function

"""
    leapfrog_step(decoder, x, z, ρ, ϵ, potential_energy, potential_energy_kwargs)

Perform a single leapfrog step in Hamiltonian Monte Carlo (HMC) algorithm. The
leapfrog step is a numerical integration scheme used in HMC to simulate the
dynamics of a physical system (the position `z` and momentum `ρ` variables)
under a potential energy function `U`. The leapfrog step consists of three
parts: a half-step update of the momentum, a full-step update of the position,
and another half-step update of the momentum.

# Arguments
- `decoder::AbstractVariationalDecoder`: The decoder model used in the
  autoencoder.
- `x::AbstractVector{T}`: The input data.
- `z::AbstractVector{T}`: The position variable in the HMC algorithm,
  representing the latent variable.
- `ρ::AbstractVector{T}`: The momentum variable in the HMC algorithm.
- `ϵ::AbstractFloat`: The step size for the HMC algorithm.

## Optional Keyword Arguments
- `potential_energy::Function=potential_energy`: The potential energy function
  used in the HMC algorithm.
- `potential_energy_kwargs::Dict=Dict()`: The keyword arguments for the
  potential energy function. See `potential_energy` for details.

# Returns
- `z̄::AbstractVector{T}`: The updated position variable.
- `ρ̄::AbstractVector{T}`: The updated momentum variable.

# Example
```julia
# Define a decoder
decoder = JointDecoder(
    Flux.Chain(Dense(10, 5, relu), Dense(5, 2)),
    Flux.Dense(2, 2), Flux.Dense(2, 2)
)

# Define input data, position, momentum, and step size
x = rand(2)
z = rand(2)
ρ = rand(2)
ϵ = 0.01

# Perform a leapfrog step
z̄, ρ̄ = leapfrog_step(decoder, x, z, ρ, ϵ, potential_energy, Dict())
```
"""
function leapfrog_step(
    decoder::AbstractVariationalDecoder,
    x::AbstractVector{T},
    z::AbstractVector{T},
    ρ::AbstractVector{T},
    ϵ::T;
    potential_energy::Function=potential_energy,
    potential_energy_kwargs::Dict=Dict(),
) where {T<:AbstractFloat}
    # Build potential energy function
    U = potential_energy(decoder; potential_energy_kwargs...)

    # Define gradient of potential energy function
    ∇U(Ƶ::AbstractVector{T}) = first(Zygote.gradient(Ƶ -> U(x, Ƶ), Ƶ))

    # Update momentum variable with half-step
    ρ̃ = ρ - ϵ * ∇U(z) / 2

    # Update position variable with full-step
    z̄ = z + ϵ * ρ̃

    # Update momentum variable with half-step
    ρ̄ = ρ̃ - ϵ * ∇U(z̄) / 2

    return z̄, ρ̄
end # function

# ==============================================================================
# Tempering Functions
# ==============================================================================
"""
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

# ==============================================================================
# Hamiltonian ELBO, Fixed Tempering
# ==============================================================================

function hamiltonian_elbo_fixed_tempering(
    vae::VAE{JointLogEncoder,<:AbstractVariationalDecoder},
    x::AbstractVector{T},
    vae_outputs::Dict{Symbol,<:AbstractVector{T}};
    K::Int=3,
    βₒ::T=0.3f0,
    ϵ::Union{T,<:AbstractVector{T}}=0.001f0,
    potential_energy::Function=potential_energy,
    potential_energy_kwargs::Dict=Dict(
        :decoder_dist => MvDiagGaussianDecoder, :prior => SphericalPrior
    ),
    tempering_schedule::Function=quadratic_tempering,
) where {T<:AbstractFloat}
    # Unpack zₒ from vae_outputs 
    # (equivalent to sampling zₒ ~ variational prior) 
    zₒ = vae_outputs[:z]

    # Sample γₒ ~ N(0, I)
    γₒ = Random.rand(SphericalPrior(zₒ))
    # Define ρₒ = γₒ / √βₒ
    ρₒ = γₒ ./ √(βₒ)

    # Define initial value of z and ρ before loop
    zₖ₋₁, ρₖ₋₁ = zₒ, ρₒ

    # Initialize variables to have them available outside the for loop
    zₖ = Vector{T}(undef, length(zₒ))
    ρₖ = Vector{T}(undef, length(ρₒ))

    # Loop over K steps
    for k = 1:K
        # 1) Leapfrog step
        zₖ, ρₖ = leapfrog_step(
            vae.decoder,
            x,
            zₖ₋₁,
            ρₖ₋₁,
            ϵ;
            potential_energy=potential_energy,
            potential_energy_kwargs=potential_energy_kwargs,
        )

        # 2) Tempering step
        # Compute previous step's inverse temperature
        βₖ₋₁ = tempering_schedule(βₒ, k - 1, K)
        # Compute current step's inverse temperature
        βₖ = tempering_schedule(βₒ, k, K)

        # Update momentum variable with tempering
        ρₖ = ρₖ * √(βₖ₋₁ / βₖ)
        # Update zₖ₋₁, ρₖ₋₁ for next iteration
        zₖ₋₁, ρₖ₋₁ = zₖ, ρₖ
    end # for

    # Compute log evidence estimate log π̂(x) = log p̄ - log q̄

    # log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)
    # log p̄ = - U(x, zₖ) + log p(ρₖ)
    log_p̄ = -potential_energy(vae.decoder; potential_energy_kwargs...)(x, zₖ) +
             Distributions.logpdf(SphericalPrior(ρₖ), ρₖ)

    # log q̄ = log q(zₒ) + log p(ρₒ)
    log_q̄ = Distributions.logpdf(
        Distributions.MvNormal(
            vae_outputs[:encoder_µ], vae_outputs[:encoder_logσ]
        ),
        zₒ
    ) + Distributions.logpdf(SphericalPrior(ρₒ, βₒ^-1), ρₒ)

    return log_p̄ - log_q̄
end # function